from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.memory.unit_metadata import unit_metadata


def build_scene_summary_memory(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build compact scene summary artifacts for recap and semantic retrieval."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    unit_summaries_path = out_path / "unit_summaries.jsonl"
    scene_summaries_path = out_path / "scene_summaries.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_unit_count": 0,
        "skipped_unit_count": 0,
        "unit_summary_count": 0,
        "scene_summary_count": 0,
        "salient_point_count": 0,
        "continuity_hook_count": 0,
    }
    unit_records: list[dict[str, Any]] = []
    scene_groups: dict[str, list[dict[str, Any]]] = {}

    with unit_summaries_path.open("w", encoding="utf-8") as unit_summaries_handle:
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_unit_count"] += 1
                continue

            data = payload["data"]
            unit_id = _record_scene_id(data, payload, parsed_file)
            unit_payload = _unit_payload_for_run(run_path, unit_id)
            metadata = unit_metadata(unit_payload, unit_id)
            salient_points = _string_list(data.get("salient_points"))
            continuity_hooks = _string_list(data.get("continuity_hooks"))
            summary_text = _clean_string(data.get("summary"))
            retrieval_text = _clean_string(data.get("retrieval_text")) or _default_retrieval_text(
                summary=summary_text,
                salient_points=salient_points,
                continuity_hooks=continuity_hooks,
            )

            record = {
                "memory_layer": "unit_summary",
                "source_run_dir": str(run_path),
                "scene_id": unit_id,
                "unit_id": unit_id,
                **metadata,
                "record_id": f"{unit_id}_summary",
                "summary": summary_text,
                "salient_points": salient_points,
                "continuity_hooks": continuity_hooks,
                "retrieval_text": retrieval_text,
            }
            _write_jsonl(unit_summaries_handle, record)
            unit_records.append(record)
            scene_groups.setdefault(str(record.get("parent_unit_id") or unit_id), []).append(record)
            counts["accepted_unit_count"] += 1
            counts["unit_summary_count"] += 1
            counts["salient_point_count"] += len(salient_points)
            counts["continuity_hook_count"] += len(continuity_hooks)

    with scene_summaries_path.open("w", encoding="utf-8") as scene_summaries_handle:
        for parent_unit_id in sorted(scene_groups, key=_scene_group_sort_key):
            scene_records = sorted(scene_groups[parent_unit_id], key=_unit_record_sort_key)
            scene_record = _parent_scene_record(parent_unit_id, scene_records, run_path)
            _write_jsonl(scene_summaries_handle, scene_record)
            counts["scene_summary_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "unit_summaries": str(unit_summaries_path),
            "scene_summaries": str(scene_summaries_path),
            "summary": str(summary_path),
        },
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _record_scene_id(data: dict[str, Any], payload: dict[str, Any], parsed_file: Path) -> str:
    return str(data.get("unit_id") or data.get("scene_id") or payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)


def _unit_payload_for_run(run_path: Path, unit_id: str) -> dict[str, Any] | None:
    input_path = run_path / "inputs" / f"{unit_id}.json"
    if not input_path.is_file():
        return None
    payload = _read_json(input_path)
    unit = payload.get("unit") if isinstance(payload.get("unit"), dict) else payload
    return unit if isinstance(unit, dict) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean_string(item) for item in value if _clean_string(item)]


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _default_retrieval_text(*, summary: str, salient_points: list[str], continuity_hooks: list[str]) -> str:
    parts = [summary, *salient_points, *continuity_hooks]
    return "\n".join(part for part in parts if part)


def _parent_scene_record(parent_unit_id: str, unit_records: list[dict[str, Any]], run_path: Path) -> dict[str, Any]:
    first = unit_records[0] if unit_records else {}
    summaries = [_clean_string(record.get("summary")) for record in unit_records]
    salient_points = _dedupe(
        point for record in unit_records for point in _string_list(record.get("salient_points"))
    )
    continuity_hooks = _dedupe(
        hook for record in unit_records for hook in _string_list(record.get("continuity_hooks"))
    )
    retrieval_parts = [_clean_string(record.get("retrieval_text")) for record in unit_records]
    chunk_ids = [str(record.get("unit_id")) for record in unit_records if record.get("unit_id")]
    return {
        "memory_layer": "scene_summary",
        "source_run_dir": str(run_path),
        "scene_id": parent_unit_id,
        "unit_id": parent_unit_id,
        "parent_unit_id": parent_unit_id,
        "chunk_id": parent_unit_id,
        "chunk_index": 1,
        "chunk_count": len(unit_records),
        "unit_source_start": first.get("unit_source_start"),
        "unit_source_end": unit_records[-1].get("unit_source_end") if unit_records else first.get("unit_source_end"),
        "unit_source_sha256": first.get("unit_source_sha256"),
        "chunk_unit_count": sum(int(record.get("chunk_unit_count") or 0) for record in unit_records) or None,
        "max_chunk_units": first.get("max_chunk_units"),
        "record_id": f"{parent_unit_id}_summary",
        "summary": "\n".join(summary for summary in summaries if summary),
        "salient_points": salient_points,
        "continuity_hooks": continuity_hooks,
        "retrieval_text": "\n".join(part for part in retrieval_parts if part)
        or _default_retrieval_text(
            summary="\n".join(summary for summary in summaries if summary),
            salient_points=salient_points,
            continuity_hooks=continuity_hooks,
        ),
        "summary_source_unit_ids": chunk_ids,
    }


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_string(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _scene_group_sort_key(scene_id: str) -> tuple[int, str]:
    suffix = scene_id.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 10**9, scene_id)


def _unit_record_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    try:
        chunk_index = int(record.get("chunk_index") or 1)
    except (TypeError, ValueError):
        chunk_index = 1
    return (chunk_index, str(record.get("unit_id") or ""))


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
