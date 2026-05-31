from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.entity_alignment import sanitize_kg_entity_output
from dms.entity_types import normalize_entity_type, normalize_scene_tag_type
from dms.memory.unit_metadata import unit_metadata


def build_kg_entity_memory(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build staged KG entity mention artifacts from parsed extractor outputs."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    mentions_path = out_path / "entity_mentions.jsonl"
    scene_tags_path = out_path / "scene_tags.jsonl"
    unresolved_path = out_path / "unresolved_mentions.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "entity_mention_count": 0,
        "scene_tag_count": 0,
        "unresolved_mention_count": 0,
        "entity_type_counts": {},
        "scene_tag_type_counts": {},
    }

    with (
        mentions_path.open("w", encoding="utf-8") as mentions_handle,
        scene_tags_path.open("w", encoding="utf-8") as scene_tags_handle,
        unresolved_path.open("w", encoding="utf-8") as unresolved_handle,
    ):
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_scene_count"] += 1
                continue

            data = sanitize_kg_entity_output(payload["data"])
            scene_id = _record_scene_id(data, payload, parsed_file)
            metadata = unit_metadata(_unit_payload_for_run(run_path, scene_id), scene_id)
            counts["accepted_scene_count"] += 1

            for index, item in enumerate(_as_list(data.get("entity_mentions")), start=1):
                if isinstance(item, dict):
                    entity_type = normalize_entity_type(item.get("entity_type"))
                    surface = str(item.get("surface", ""))
                    canonical_hint = str(item.get("canonical_hint", ""))
                    evidence = str(item.get("evidence", ""))
                    record = {
                        "memory_layer": "staged_kg_entity_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_kg_ent_{index:03d}",
                        "surface": surface,
                        "entity_type": entity_type,
                        "canonical_hint": canonical_hint,
                        "description": item.get("description", ""),
                        "role_in_unit": item.get("role_in_unit") or item.get("role_in_scene", ""),
                        "attributes_or_state": item.get("attributes_or_state", ""),
                        "evidence": evidence,
                    }
                else:
                    entity_type = "other"
                    record = {
                        "memory_layer": "staged_kg_entity_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_kg_ent_{index:03d}",
                        "surface": str(item),
                        "entity_type": entity_type,
                        "canonical_hint": "",
                        "description": "",
                        "role_in_unit": "",
                        "attributes_or_state": "",
                        "evidence": "",
                    }
                _write_jsonl(mentions_handle, record)
                counts["entity_mention_count"] += 1
                type_counts = counts["entity_type_counts"]
                type_counts[entity_type] = int(type_counts.get(entity_type, 0)) + 1

            for index, item in enumerate(_as_list(data.get("scene_tags")), start=1):
                if isinstance(item, dict):
                    tag_type = normalize_scene_tag_type(item.get("tag_type"))
                    record = {
                        "memory_layer": "scene_tag",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_scene_tag_{index:03d}",
                        "surface": item.get("surface", ""),
                        "tag_type": tag_type,
                        "reason": item.get("reason", ""),
                        "evidence": item.get("evidence", ""),
                    }
                else:
                    tag_type = "other"
                    record = {
                        "memory_layer": "scene_tag",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_scene_tag_{index:03d}",
                        "surface": str(item),
                        "tag_type": tag_type,
                        "reason": "",
                        "evidence": "",
                    }
                _write_jsonl(scene_tags_handle, record)
                counts["scene_tag_count"] += 1
                tag_counts = counts["scene_tag_type_counts"]
                tag_counts[tag_type] = int(tag_counts.get(tag_type, 0)) + 1

            for index, item in enumerate(_as_list(data.get("unresolved_mentions")), start=1):
                if isinstance(item, dict):
                    record = {
                        "memory_layer": "staged_kg_entity_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_kg_unresolved_{index:03d}",
                        "surface": item.get("surface", ""),
                        "reason": item.get("reason", ""),
                        "evidence": item.get("evidence", ""),
                    }
                else:
                    record = {
                        "memory_layer": "staged_kg_entity_extraction",
                        "scene_id": scene_id,
                        "unit_id": scene_id,
                        **metadata,
                        "record_id": f"{scene_id}_kg_unresolved_{index:03d}",
                        "surface": str(item),
                        "reason": "",
                        "evidence": "",
                    }
                _write_jsonl(unresolved_handle, record)
                counts["unresolved_mention_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "entity_mentions": str(mentions_path),
            "scene_tags": str(scene_tags_path),
            "unresolved_mentions": str(unresolved_path),
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


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
