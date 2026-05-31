from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_scene_event_memory(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build staged JSONL event memory artifacts from parsed event-candidate outputs."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    events_path = out_path / "events.jsonl"
    transfers_path = out_path / "knowledge_transfers.jsonl"
    state_changes_path = out_path / "state_changes.jsonl"
    threads_path = out_path / "thread_candidates.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "event_count": 0,
        "knowledge_transfer_count": 0,
        "state_change_count": 0,
        "thread_candidate_count": 0,
    }

    with (
        events_path.open("w", encoding="utf-8") as events_handle,
        transfers_path.open("w", encoding="utf-8") as transfers_handle,
        state_changes_path.open("w", encoding="utf-8") as state_changes_handle,
        threads_path.open("w", encoding="utf-8") as threads_handle,
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

            for index, item in enumerate(_as_list(data.get("events")), start=1):
                record = _base_record(scene_id, f"{scene_id}_event_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "event_id_hint": item.get("event_id_hint", ""),
                            "summary": item.get("summary", ""),
                            "participants": item.get("participants") if isinstance(item.get("participants"), list) else [],
                            "location": item.get("location", ""),
                            "event_type": item.get("event_type", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"summary": str(item), "participants": [], "location": "", "event_type": "", "evidence": ""})
                _write_jsonl(events_handle, record)
                counts["event_count"] += 1

            for index, item in enumerate(_as_list(data.get("knowledge_transfers")), start=1):
                record = _base_record(scene_id, f"{scene_id}_kt_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "source": item.get("source", ""),
                            "receiver": item.get("receiver", ""),
                            "content": item.get("content", ""),
                            "epistemic_status": item.get("epistemic_status", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"source": "", "receiver": "", "content": str(item), "epistemic_status": "", "evidence": ""})
                _write_jsonl(transfers_handle, record)
                counts["knowledge_transfer_count"] += 1

            for index, item in enumerate(_as_list(data.get("state_changes")), start=1):
                record = _base_record(scene_id, f"{scene_id}_state_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "entity": item.get("entity", ""),
                            "before": item.get("before", ""),
                            "after": item.get("after", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"entity": str(item), "before": "", "after": "", "evidence": ""})
                _write_jsonl(state_changes_handle, record)
                counts["state_change_count"] += 1

            for index, item in enumerate(_as_list(data.get("thread_candidates")), start=1):
                record = _base_record(scene_id, f"{scene_id}_thread_{index:03d}")
                if isinstance(item, dict):
                    record.update(
                        {
                            "thread_type": item.get("thread_type", ""),
                            "summary": item.get("summary", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
                else:
                    record.update({"thread_type": "", "summary": str(item), "evidence": ""})
                _write_jsonl(threads_handle, record)
                counts["thread_candidate_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "events": str(events_path),
            "knowledge_transfers": str(transfers_path),
            "state_changes": str(state_changes_path),
            "thread_candidates": str(threads_path),
            "summary": str(summary_path),
        },
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _base_record(scene_id: str, record_id: str) -> dict[str, Any]:
    return {
        "memory_layer": "staged_event_extraction",
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


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
