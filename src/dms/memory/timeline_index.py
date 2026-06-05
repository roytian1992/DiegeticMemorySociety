from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dms.memory.temporal_scope import infer_memory_temporal_scope, is_story_time_bound_scope


@dataclass(frozen=True)
class MemoryTimelineIndexConfig:
    memory_path: Path
    timeline_graph_path: Path
    output_dir: Path
    min_match_score: float = 0.25
    overwrite: bool = False


def build_memory_timeline_index(config: MemoryTimelineIndexConfig) -> dict[str, Any]:
    """Attach diegetic timeline indexes to episodic memories without changing retrieval."""

    memory_file = _resolve_memory_file(config.memory_path)
    timeline_graph_file = _resolve_timeline_graph_file(config.timeline_graph_path)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output dir exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memories = _read_jsonl(memory_file)
    graph = _read_json(timeline_graph_file)
    timeline_index = _timeline_event_index(graph)
    events_by_scene: dict[str, list[dict[str, Any]]] = {}
    for event in timeline_index["events"]:
        events_by_scene.setdefault(str(event.get("scene_id") or ""), []).append(event)

    enriched_records: list[dict[str, Any]] = []
    link_records: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []
    matched_count = 0
    scope_counts: dict[str, int] = {}
    not_story_time_bound_count = 0

    for memory in memories:
        memory = {**memory, **infer_memory_temporal_scope(memory)}
        scope = str(memory.get("memory_temporal_scope") or "uncertain")
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        scene_id = str(memory.get("scene_id") or memory.get("parent_unit_id") or "")
        candidates = events_by_scene.get(scene_id, [])
        if not is_story_time_bound_scope(scope):
            not_story_time_bound_count += 1
            match = None
            enriched = _enrich_not_story_time_bound(memory, candidate_count=len(candidates))
        else:
            match = _best_event_match(memory, candidates, min_score=config.min_match_score)
        if is_story_time_bound_scope(scope) and match is None:
            enriched = _enrich_unmatched(memory, candidate_count=len(candidates))
            unmatched_records.append(enriched)
        elif match is not None:
            matched_count += 1
            enriched = _enrich_matched(memory, match["event"], match)
        enriched_records.append(enriched)
        link_records.append(_link_record(enriched, match, candidate_count=len(candidates)))

    story_sorted = sorted(enriched_records, key=_story_sort_key)
    artifacts = {
        "enriched_episodic_memories": str(output_dir / "enriched_episodic_memories.jsonl"),
        "story_time_memory_index": str(output_dir / "story_time_memory_index.jsonl"),
        "memory_timeline_links": str(output_dir / "memory_timeline_links.jsonl"),
        "unmatched_memories": str(output_dir / "unmatched_memories.jsonl"),
        "summary": str(output_dir / "summary.json"),
    }
    _write_jsonl(output_dir / "enriched_episodic_memories.jsonl", enriched_records)
    _write_jsonl(output_dir / "story_time_memory_index.jsonl", story_sorted)
    _write_jsonl(output_dir / "memory_timeline_links.jsonl", link_records)
    _write_jsonl(output_dir / "unmatched_memories.jsonl", unmatched_records)

    summary = {
        "run_type": "memory_timeline_index",
        "status": "complete",
        "inputs": {
            "memory_file": str(memory_file),
            "timeline_graph": str(timeline_graph_file),
            "min_match_score": config.min_match_score,
        },
        "counts": {
            "memory_count": len(memories),
            "timeline_event_count": len(timeline_index["events"]),
            "matched_memory_count": matched_count,
            "unmatched_memory_count": len(unmatched_records),
            "not_story_time_bound_memory_count": not_story_time_bound_count,
            "timeline_bucket_count": len(timeline_index["bucket_by_event"]),
        },
        "memory_temporal_scope_counts": scope_counts,
        "policy": {
            "existing_timeline_index_preserved": True,
            "existing_timeline_index_semantics": "discourse_scene_sequence",
            "story_time_index_added": True,
            "not_story_time_bound_scope_status": "scope_not_story_time_bound",
            "retrieval_filtering_changed": False,
            "default_visibility_filter": "revealed_at_order before target discourse order",
        },
        "artifacts": artifacts,
        "config": {
            **asdict(config),
            "memory_path": str(config.memory_path),
            "timeline_graph_path": str(config.timeline_graph_path),
            "output_dir": str(config.output_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _resolve_memory_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "episodic_memories.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(f"Episodic memory JSONL not found: {candidate}")
    return candidate


def _resolve_timeline_graph_file(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "timeline_graph.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Timeline graph JSON not found: {candidate}")
    return candidate


def _timeline_event_index(graph: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in graph.get("events") or [] if isinstance(event, dict)]
    order_by_event = {
        str(item.get("event_id")): item
        for item in graph.get("timeline_order") or []
        if isinstance(item, dict) and item.get("event_id")
    }
    bucket_by_event: dict[str, dict[str, Any]] = {}
    for bucket in graph.get("timeline_buckets") or []:
        if not isinstance(bucket, dict):
            continue
        for event_id in bucket.get("event_ids") or []:
            bucket_by_event[str(event_id)] = bucket
    relations_by_event: dict[str, list[str]] = {}
    for relation in graph.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        relation_id = str(relation.get("relation_id") or "")
        if not relation_id:
            continue
        for event_id in (relation.get("source_event_id"), relation.get("target_event_id")):
            if event_id:
                relations_by_event.setdefault(str(event_id), []).append(relation_id)
    indexed_events = []
    for event in events:
        event_id = str(event.get("event_id") or "")
        order = order_by_event.get(event_id, {})
        bucket = bucket_by_event.get(event_id, {})
        indexed_events.append(
            {
                **event,
                "story_time_rank": order.get("timeline_rank"),
                "story_time_bucket": bucket.get("timeline_bucket"),
                "story_time_bucket_event_ids": bucket.get("event_ids") or [],
                "temporal_relation_basis": sorted(set(relations_by_event.get(event_id, []))),
            }
        )
    return {
        "events": indexed_events,
        "order_by_event": order_by_event,
        "bucket_by_event": bucket_by_event,
    }


def _best_event_match(
    memory: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    min_score: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for event in candidates:
        score, components = _match_score(memory, event)
        if best is None or score > float(best.get("match_score") or 0):
            best = {"event": event, "match_score": score, "match_components": components}
    if best is None or float(best.get("match_score") or 0) < min_score:
        return None
    return best


def _match_score(memory: dict[str, Any], event: dict[str, Any]) -> tuple[float, dict[str, float]]:
    evidence_score = _evidence_score(str(memory.get("evidence_text") or ""), str(event.get("evidence") or ""))
    summary_score = _similarity(str(memory.get("summary") or ""), str(event.get("summary") or ""))
    sequence_score = _sequence_score(memory, event)
    score = round(0.55 * evidence_score + 0.35 * summary_score + 0.10 * sequence_score, 4)
    return score, {
        "evidence": round(evidence_score, 4),
        "summary": round(summary_score, 4),
        "sequence": round(sequence_score, 4),
    }


def _evidence_score(memory_evidence: str, event_evidence: str) -> float:
    left = memory_evidence.strip()
    right = event_evidence.strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.9
    return _similarity(left, right)


def _similarity(left: str, right: str) -> float:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _sequence_score(memory: dict[str, Any], event: dict[str, Any]) -> float:
    memory_sequence = _optional_int(memory.get("sequence_index"))
    event_id = str(event.get("event_id") or "")
    event_sequence = _optional_int(event_id.rsplit("_", 1)[-1]) if "_" in event_id else None
    if memory_sequence is not None and event_sequence is not None and memory_sequence == event_sequence:
        return 1.0
    return 0.0


def _enrich_matched(memory: dict[str, Any], event: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    discourse_timeline_index = memory.get("timeline_index")
    return {
        **memory,
        "discourse_timeline_index": discourse_timeline_index,
        "timeline_index_semantics": "discourse_scene_sequence",
        "memory_timeline_index_status": "matched",
        "story_event_id": event.get("event_id"),
        "story_time_index": _story_time_index(event),
        "story_time_bucket": event.get("story_time_bucket"),
        "story_time_rank": event.get("story_time_rank"),
        "story_time_mode": event.get("event_time_mode"),
        "story_time_track": event.get("event_track"),
        "story_time_hint": event.get("story_time_hint"),
        "story_time_granularity": event.get("granularity"),
        "story_time_confidence": event.get("confidence"),
        "revealed_at_scene_id": event.get("revealed_at_scene_id") or event.get("scene_id"),
        "revealed_at_order": event.get("revealed_at_source_record_id") or event.get("source_record_id"),
        "temporal_relation_basis": event.get("temporal_relation_basis") or [],
        "story_time_match_score": match.get("match_score"),
        "story_time_match_components": match.get("match_components") or {},
    }


def _enrich_unmatched(memory: dict[str, Any], *, candidate_count: int) -> dict[str, Any]:
    return {
        **memory,
        "discourse_timeline_index": memory.get("timeline_index"),
        "timeline_index_semantics": "discourse_scene_sequence",
        "memory_timeline_index_status": "unmatched" if candidate_count else "no_scene_temporal_event",
        "story_event_id": None,
        "story_time_index": None,
        "story_time_bucket": None,
        "story_time_rank": None,
        "story_time_mode": None,
        "story_time_track": None,
        "story_time_hint": None,
        "story_time_granularity": None,
        "story_time_confidence": None,
        "revealed_at_scene_id": memory.get("scene_id"),
        "revealed_at_order": _scene_order_from_id(str(memory.get("scene_id") or "")),
        "temporal_relation_basis": [],
        "story_time_match_score": 0.0,
        "story_time_match_components": {},
    }


def _enrich_not_story_time_bound(memory: dict[str, Any], *, candidate_count: int) -> dict[str, Any]:
    return {
        **_enrich_unmatched(memory, candidate_count=candidate_count),
        "memory_timeline_index_status": "scope_not_story_time_bound",
        "story_time_match_score": None,
        "story_time_match_components": {},
    }


def _link_record(enriched: dict[str, Any], match: dict[str, Any] | None, *, candidate_count: int) -> dict[str, Any]:
    return {
        "memory_record_id": enriched.get("record_id"),
        "memory_scene_id": enriched.get("scene_id"),
        "status": enriched.get("memory_timeline_index_status"),
        "memory_temporal_scope": enriched.get("memory_temporal_scope"),
        "candidate_event_count": candidate_count,
        "story_event_id": enriched.get("story_event_id"),
        "story_time_index": enriched.get("story_time_index"),
        "story_time_bucket": enriched.get("story_time_bucket"),
        "story_time_rank": enriched.get("story_time_rank"),
        "revealed_at_scene_id": enriched.get("revealed_at_scene_id"),
        "revealed_at_order": enriched.get("revealed_at_order"),
        "match_score": enriched.get("story_time_match_score"),
        "match_components": enriched.get("story_time_match_components"),
        "event_summary": (match or {}).get("event", {}).get("summary") if match else None,
    }


def _story_time_index(event: dict[str, Any]) -> str | None:
    bucket = event.get("story_time_bucket")
    event_id = event.get("event_id")
    if not bucket or not event_id:
        return None
    return f"{bucket}:{event_id}"


def _story_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    rank = _optional_int(record.get("story_time_rank"))
    revealed_order = _optional_int(record.get("revealed_at_order"))
    return (
        rank if rank is not None else 10**9,
        revealed_order if revealed_order is not None else 10**9,
        str(record.get("record_id") or ""),
    )


def _scene_order_from_id(scene_id: str) -> int | None:
    suffix = scene_id.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                records.append(data)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
