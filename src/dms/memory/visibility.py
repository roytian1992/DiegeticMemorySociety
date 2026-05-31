from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_visibility_memory(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build staged JSONL visibility artifacts from parsed visibility-note outputs."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    visibility_path = out_path / "visibility_records.jsonl"
    hidden_path = out_path / "hidden_or_future_sensitive_items.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "visibility_record_count": 0,
        "hidden_or_future_sensitive_count": 0,
    }

    with (
        visibility_path.open("w", encoding="utf-8") as visibility_handle,
        hidden_path.open("w", encoding="utf-8") as hidden_handle,
    ):
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_scene_count"] += 1
                continue

            data = payload["data"]
            scene_id = _record_scene_id(data, payload, parsed_file)
            counts["accepted_scene_count"] += 1

            for index, item in enumerate(_as_list(data.get("visibility_records")), start=1):
                record = _base_record(scene_id, f"{scene_id}_vis_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "fact_or_event": item.get("fact_or_event", ""),
                            "character": item.get("character", ""),
                            "visibility": item.get("visibility", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"fact_or_event": str(item), "character": "", "visibility": "", "evidence": ""})
                _write_jsonl(visibility_handle, record)
                counts["visibility_record_count"] += 1

            for index, item in enumerate(_as_list(data.get("hidden_or_future_sensitive_items")), start=1):
                record = _base_record(scene_id, f"{scene_id}_hidden_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "item": _first_string(item, ("item", "fact_or_event", "summary", "content")),
                            "hidden_from": item.get("hidden_from") if isinstance(item.get("hidden_from"), list) else [],
                            "reason": item.get("reason", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"item": str(item), "hidden_from": [], "reason": "", "evidence": ""})
                _write_jsonl(hidden_handle, record)
                counts["hidden_or_future_sensitive_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "visibility_records": str(visibility_path),
            "hidden_or_future_sensitive_items": str(hidden_path),
            "summary": str(summary_path),
        },
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _base_record(scene_id: str, record_id: str) -> dict[str, Any]:
    return {
        "memory_layer": "staged_visibility_extraction",
        "scene_id": scene_id,
        "record_id": record_id,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _record_scene_id(data: dict[str, Any], payload: dict[str, Any], parsed_file: Path) -> str:
    return str(data.get("unit_id") or data.get("scene_id") or payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)


def _first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
