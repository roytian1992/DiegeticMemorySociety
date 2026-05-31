from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.memory.unit_metadata import unit_metadata


def build_scene_inventory_memory(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build staged JSONL memory artifacts from parsed scene inventory outputs.

    These artifacts are intentionally staged. They are derived from extractor
    outputs and should be reviewed or reconciled before becoming canonical DMS
    world-model memory.
    """

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    scenes_path = out_path / "scenes.jsonl"
    characters_path = out_path / "characters.jsonl"
    objects_path = out_path / "objects.jsonl"
    facts_path = out_path / "stated_facts.jsonl"
    questions_path = out_path / "open_questions.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "character_count": 0,
        "object_count": 0,
        "stated_fact_count": 0,
        "open_question_count": 0,
    }

    with (
        scenes_path.open("w", encoding="utf-8") as scenes_handle,
        characters_path.open("w", encoding="utf-8") as characters_handle,
        objects_path.open("w", encoding="utf-8") as objects_handle,
        facts_path.open("w", encoding="utf-8") as facts_handle,
        questions_path.open("w", encoding="utf-8") as questions_handle,
    ):
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_scene_count"] += 1
                continue

            data = payload["data"]
            scene_id = _record_scene_id(data, payload, parsed_file)
            metadata = unit_metadata(_unit_payload_for_run(run_path, scene_id), scene_id)
            counts["accepted_scene_count"] += 1
            _write_jsonl(
                scenes_handle,
                {
                    "memory_layer": "staged_extraction",
                    "source_run_dir": str(run_path),
                    "scene_id": scene_id,
                    "unit_id": scene_id,
                    **metadata,
                    "setting": _normalize_setting(data.get("setting")),
                },
            )

            for index, item in enumerate(_as_list(data.get("characters")), start=1):
                _write_jsonl(
                    characters_handle,
                    {
                        "memory_layer": "staged_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_char_{index:03d}",
                        "name": item.get("name") if isinstance(item, dict) else str(item),
                        "evidence": item.get("evidence", "") if isinstance(item, dict) else "",
                    },
                )
                counts["character_count"] += 1

            for index, item in enumerate(_as_list(data.get("objects")), start=1):
                _write_jsonl(
                    objects_handle,
                    {
                        "memory_layer": "staged_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_obj_{index:03d}",
                        "name": item.get("name") if isinstance(item, dict) else str(item),
                        "state_or_role": item.get("state_or_role", "") if isinstance(item, dict) else "",
                        "evidence": item.get("evidence", "") if isinstance(item, dict) else "",
                    },
                )
                counts["object_count"] += 1

            for index, item in enumerate(_as_list(data.get("stated_facts")), start=1):
                _write_jsonl(
                    facts_handle,
                    {
                        "memory_layer": "staged_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_fact_{index:03d}",
                        "proposition": item.get("proposition") if isinstance(item, dict) else str(item),
                        "speaker_or_source": item.get("speaker_or_source", "") if isinstance(item, dict) else "",
                        "evidence": item.get("evidence", "") if isinstance(item, dict) else "",
                    },
                )
                counts["stated_fact_count"] += 1

            for index, item in enumerate(_as_list(data.get("open_questions")), start=1):
                _write_jsonl(
                    questions_handle,
                    {
                        "memory_layer": "staged_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_question_{index:03d}",
                        "question": item.get("question") if isinstance(item, dict) else str(item),
                        "evidence": item.get("evidence", "") if isinstance(item, dict) else "",
                    },
                )
                counts["open_question_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "scenes": str(scenes_path),
            "characters": str(characters_path),
            "objects": str(objects_path),
            "stated_facts": str(facts_path),
            "open_questions": str(questions_path),
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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _record_scene_id(data: dict[str, Any], payload: dict[str, Any], parsed_file: Path) -> str:
    return str(data.get("unit_id") or data.get("scene_id") or payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)


def _unit_payload_for_run(run_path: Path, scene_id: str) -> dict[str, Any] | None:
    input_path = run_path / "inputs" / f"{scene_id}.json"
    if not input_path.is_file():
        return None
    payload = _read_json(input_path)
    unit = payload.get("unit") if isinstance(payload.get("unit"), dict) else payload
    return unit if isinstance(unit, dict) else None


def _normalize_setting(value: Any) -> dict[str, str]:
    setting = value if isinstance(value, dict) else {}
    location = str(setting.get("location") or "")
    time_hint = str(setting.get("time_hint") or setting.get("time_of_day") or "")
    spatial_context = str(setting.get("spatial_context") or setting.get("interior_exterior") or "")
    return {
        "location": location,
        "time_hint": time_hint,
        "spatial_context": spatial_context,
        "time_of_day": time_hint,
        "interior_exterior": spatial_context,
    }


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
