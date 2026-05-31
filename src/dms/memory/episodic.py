from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.entity_alignment import align_entity_to_candidates, build_entity_candidate_index
from dms.entity_types import entity_trackability_issue, is_deictic_surface, normalize_entity_type
from dms.memory.unit_metadata import parent_evidence_span, unit_metadata
from dms.source_evidence import build_source_text_index, locate_evidence


def build_episodic_memory(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    require_entity_candidates: bool = False,
) -> dict[str, Any]:
    """Build ordered episodic memory artifacts from parsed extractor outputs."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    memories_path = out_path / "episodic_memories.jsonl"
    links_path = out_path / "entity_memory_links.jsonl"
    rejections_path = out_path / "evidence_rejections.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "episodic_memory_count": 0,
        "entity_memory_link_count": 0,
        "skipped_episodic_memory_count": 0,
        "skipped_entity_memory_link_count": 0,
        "skipped_deictic_entity_memory_link_count": 0,
        "skipped_non_trackable_entity_memory_link_count": 0,
        "skipped_unresolved_entity_memory_link_count": 0,
        "rejected_episodic_memory_count": 0,
        "rejected_entity_memory_link_count": 0,
    }
    evidence_counts: dict[str, int] = {"exact": 0, "fuzzy_aligned": 0, "rejected": 0}

    with (
        memories_path.open("w", encoding="utf-8") as memories_handle,
        links_path.open("w", encoding="utf-8") as links_handle,
        rejections_path.open("w", encoding="utf-8") as rejections_handle,
    ):
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_scene_count"] += 1
                continue

            data = payload["data"]
            scene_id = _record_scene_id(data, payload, parsed_file)
            unit_payload = _unit_payload_for_run(run_path, scene_id)
            metadata = unit_metadata(unit_payload, scene_id)
            source_text_index = build_source_text_index(unit_payload) if unit_payload else {}
            candidate_index = build_entity_candidate_index(_extracted_candidates_for_run(run_path, scene_id))
            counts["accepted_scene_count"] += 1

            for index, item in enumerate(_as_list(data.get("episodic_memories")), start=1):
                memory_id = f"{scene_id}_memory_{index:03d}"
                if isinstance(item, dict):
                    sequence_index = item.get("sequence_index") if isinstance(item.get("sequence_index"), int) else index
                    evidence_record = _verified_evidence(item.get("evidence", ""), unit_payload)
                    if _evidence_rejected(evidence_record):
                        _count_evidence_status(evidence_counts, evidence_record)
                        counts["skipped_episodic_memory_count"] += 1
                        counts["rejected_episodic_memory_count"] += 1
                        _write_rejection(
                            rejections_handle,
                            scene_id=scene_id,
                            record_type="episodic_memory",
                            record_id=memory_id,
                            item=item,
                            evidence_record=evidence_record,
                        )
                        continue
                    memory_record = {
                        "memory_layer": "episodic_memory",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": memory_id,
                        "memory_id_hint": item.get("memory_id_hint", ""),
                        "sequence_index": sequence_index,
                        "timeline_index": _timeline_index(scene_id, sequence_index),
                        "timeline_label": item.get("timeline_label", ""),
                        "memory_type": item.get("memory_type", ""),
                        "summary": item.get("summary", ""),
                        **evidence_record,
                        "source_text_index": source_text_index,
                    }
                    links = _as_list(item.get("entity_links"))
                else:
                    evidence_record = _verified_evidence("", unit_payload)
                    if _evidence_rejected(evidence_record):
                        _count_evidence_status(evidence_counts, evidence_record)
                        counts["skipped_episodic_memory_count"] += 1
                        counts["rejected_episodic_memory_count"] += 1
                        _write_rejection(
                            rejections_handle,
                            scene_id=scene_id,
                            record_type="episodic_memory",
                            record_id=memory_id,
                            item=item,
                            evidence_record=evidence_record,
                        )
                        continue
                    memory_record = {
                        "memory_layer": "episodic_memory",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": memory_id,
                        "memory_id_hint": "",
                        "sequence_index": index,
                        "timeline_index": _timeline_index(scene_id, index),
                        "timeline_label": "",
                        "memory_type": "other",
                        "summary": str(item),
                        **evidence_record,
                        "source_text_index": source_text_index,
                    }
                    links = []
                _count_evidence_status(evidence_counts, evidence_record)
                _write_jsonl(memories_handle, memory_record)
                counts["episodic_memory_count"] += 1

                for link_index, link in enumerate(links, start=1):
                    if isinstance(link, dict):
                        issue = entity_trackability_issue(
                            surface=link.get("entity", ""),
                            entity_type=link.get("entity_type", ""),
                            evidence=link.get("evidence", ""),
                        )
                        if issue:
                            counts["skipped_entity_memory_link_count"] += 1
                            if is_deictic_surface(link.get("entity", "")):
                                counts["skipped_deictic_entity_memory_link_count"] += 1
                            else:
                                counts["skipped_non_trackable_entity_memory_link_count"] += 1
                            continue
                        candidate = align_entity_to_candidates(
                            entity=link.get("entity", ""),
                            entity_type=link.get("entity_type", ""),
                            candidate_index=candidate_index,
                        )
                        if require_entity_candidates and candidate is None:
                            counts["skipped_entity_memory_link_count"] += 1
                            counts["skipped_unresolved_entity_memory_link_count"] += 1
                            continue
                    else:
                        candidate = None
                    link_record = _entity_link_record(
                        scene_id,
                        memory_id,
                        link_index,
                        link,
                        unit_payload=unit_payload,
                        candidate=candidate,
                    )
                    _count_evidence_status(evidence_counts, link_record)
                    if _evidence_rejected(link_record):
                        counts["skipped_entity_memory_link_count"] += 1
                        counts["rejected_entity_memory_link_count"] += 1
                        _write_rejection(
                            rejections_handle,
                            scene_id=scene_id,
                            record_type="entity_memory_link",
                            record_id=link_record["record_id"],
                            parent_record_id=memory_id,
                            item=link,
                            evidence_record=link_record,
                        )
                        continue
                    _write_jsonl(links_handle, link_record)
                    counts["entity_memory_link_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "episodic_memories": str(memories_path),
            "entity_memory_links": str(links_path),
            "evidence_rejections": str(rejections_path),
            "summary": str(summary_path),
        },
        "evidence_verification_counts": evidence_counts,
        "rejected_evidence_count": evidence_counts.get("rejected", 0),
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _entity_link_record(
    scene_id: str,
    memory_id: str,
    link_index: int,
    link: Any,
    *,
    unit_payload: dict[str, Any] | None,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(link, dict):
        evidence_record = _verified_evidence(link.get("evidence", ""), unit_payload)
        entity = str(link.get("entity", "")).strip()
        entity_type = normalize_entity_type(link.get("entity_type"))
        canonical_entity = entity
        if candidate:
            entity = str(candidate.get("canonical_name") or candidate.get("surface") or entity)
            entity_type = normalize_entity_type(candidate.get("entity_type") or entity_type)
            canonical_entity = str(candidate.get("canonical_name") or entity)
        return {
            "memory_layer": "entity_memory_link",
            "scene_id": scene_id,
            "unit_id": scene_id,
            **unit_metadata(unit_payload, scene_id),
            "memory_record_id": memory_id,
            "record_id": f"{memory_id}_entity_{link_index:03d}",
            "entity": entity,
            "entity_type": entity_type,
            "canonical_entity": canonical_entity,
            "model_entity": link.get("entity", ""),
            "link_role": link.get("link_role", ""),
            **evidence_record,
        }
    evidence_record = _verified_evidence("", unit_payload)
    return {
        "memory_layer": "entity_memory_link",
        "scene_id": scene_id,
        "unit_id": scene_id,
        **unit_metadata(unit_payload, scene_id),
        "memory_record_id": memory_id,
        "record_id": f"{memory_id}_entity_{link_index:03d}",
        "entity": str(link),
        "entity_type": "",
        "canonical_entity": str(link),
        "model_entity": str(link),
        "link_role": "",
        **evidence_record,
    }


def _timeline_index(scene_id: str, sequence_index: int) -> str:
    return f"{scene_id}:{sequence_index:03d}"


def _record_scene_id(data: dict[str, Any], payload: dict[str, Any], parsed_file: Path) -> str:
    return str(data.get("unit_id") or data.get("scene_id") or payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)


def _unit_payload_for_run(run_path: Path, scene_id: str) -> dict[str, Any] | None:
    input_path = run_path / "inputs" / f"{scene_id}.json"
    if not input_path.is_file():
        return None
    payload = _read_json(input_path)
    unit = payload.get("unit") if isinstance(payload.get("unit"), dict) else payload
    return unit if isinstance(unit, dict) else None


def _extracted_candidates_for_run(run_path: Path, scene_id: str) -> dict[str, Any]:
    input_path = run_path / "inputs" / f"{scene_id}.json"
    if not input_path.is_file():
        return {}
    payload = _read_json(input_path)
    candidates = payload.get("extracted_candidates") if isinstance(payload.get("extracted_candidates"), dict) else {}
    return candidates if isinstance(candidates, dict) else {}


def _evidence_location(evidence: Any, unit_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not unit_payload:
        return locate_evidence(evidence, {})
    return locate_evidence(evidence, unit_payload)


def _verified_evidence(evidence: Any, unit_payload: dict[str, Any] | None) -> dict[str, Any]:
    location = _evidence_location(evidence, unit_payload)
    aligned = str(location.get("evidence_aligned_text") or "")
    original = str(location.get("evidence_text") or "")
    status = str(location.get("evidence_verification_status") or "rejected")
    return {
        "evidence": aligned if status in {"exact", "fuzzy_aligned"} and aligned else original,
        "model_evidence": original,
        **location,
        **parent_evidence_span(location, unit_payload),
    }


def _count_evidence_status(counts: dict[str, int], record: dict[str, Any]) -> None:
    status = str(record.get("evidence_verification_status") or "rejected")
    counts[status] = counts.get(status, 0) + 1


def _evidence_rejected(record: dict[str, Any]) -> bool:
    return str(record.get("evidence_verification_status") or "rejected") == "rejected"


def _write_rejection(
    handle: Any,
    *,
    scene_id: str,
    record_type: str,
    record_id: str,
    item: Any,
    evidence_record: dict[str, Any],
    parent_record_id: str = "",
) -> None:
    raw_item = item if isinstance(item, dict) else {"value": str(item)}
    handle.write(
        json.dumps(
            {
                "scene_id": scene_id,
                "record_type": record_type,
                "record_id": record_id,
                "parent_record_id": parent_record_id,
                "reason": "evidence did not align to a contiguous source span",
                "item": raw_item,
                "evidence": evidence_record.get("model_evidence", evidence_record.get("evidence", "")),
                "evidence_verification_status": evidence_record.get("evidence_verification_status"),
                "evidence_alignment_score": evidence_record.get("evidence_alignment_score"),
                "evidence_source_field": evidence_record.get("evidence_source_field"),
                "evidence_start": evidence_record.get("evidence_start"),
                "evidence_end": evidence_record.get("evidence_end"),
                "parent_evidence_start": evidence_record.get("parent_evidence_start"),
                "parent_evidence_end": evidence_record.get("parent_evidence_end"),
                "parent_source_sha256": evidence_record.get("parent_source_sha256"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
