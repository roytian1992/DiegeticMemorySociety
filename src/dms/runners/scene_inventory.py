from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dms.llm import LLMClient
from dms.parsing import extract_json_value, validate_scene_inventory
from dms.prompts import YAMLPromptLoader
from dms.runners.prompt_payloads import narrative_unit_payload
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes


@dataclass(frozen=True)
class SceneInventoryRunConfig:
    script_path: Path
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    task_settings_path: Path = Path("task_specs/task_settings/scene_inventory_task.json")
    start: int = 1
    limit: int = 5
    dry_run: bool = True
    overwrite: bool = False


def run_scene_inventory(config: SceneInventoryRunConfig, *, llm_client: LLMClient | None = None) -> dict[str, Any]:
    """Render scene-inventory prompts for an ordered scene prefix.

    This runner intentionally starts with a dry-run mode. It creates the
    reproducibility surface needed before adding an LLM client: exact selected
    scenes, rendered prompts, static task settings, trace records, and summary.
    """

    if not config.dry_run and llm_client is None:
        raise ValueError("llm_client is required when dry_run is false")
    if config.limit < 1:
        raise ValueError("limit must be >= 1")
    if config.start < 1:
        raise ValueError("start must be >= 1")

    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {output_dir}")

    prompts_dir = output_dir / "prompts"
    inputs_dir = output_dir / "inputs"
    raw_outputs_dir = output_dir / "raw_outputs"
    parsed_dir = output_dir / "parsed"
    for directory in (prompts_dir, inputs_dir, raw_outputs_dir, parsed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    task_settings = _read_json(config.task_settings_path)
    prompt_id = str(task_settings["prompt_id"])
    extraction_policy = _format_policy(task_settings.get("extraction_policy", []))

    scenes = load_script_scenes(config.script_path)
    selected = _select_scenes(scenes, start=config.start, limit=config.limit)
    loader = YAMLPromptLoader(config.prompt_dir)
    prompt_spec = loader.load(prompt_id)
    created_at = datetime.now(timezone.utc).isoformat()

    trace_path = output_dir / "trace.jsonl"
    trace_records: list[dict[str, Any]] = []
    completed_count = 0
    parsed_count = 0
    failed_count = 0

    for ordinal, scene in enumerate(selected, start=1):
        unit_payload = narrative_unit_payload(scene)
        unit_json = json.dumps(unit_payload, ensure_ascii=False, indent=2)
        prompt_text = loader.render(
            prompt_spec,
            task_values={"unit_json": unit_json},
            static_values={"extraction_policy": extraction_policy},
        )

        input_path = inputs_dir / f"{scene.scene_id}.json"
        prompt_path = prompts_dir / f"{scene.scene_id}.txt"
        raw_output_path = raw_outputs_dir / f"{scene.scene_id}.json"
        parsed_path = parsed_dir / f"{scene.scene_id}.json"

        _write_text(input_path, json.dumps(unit_payload, ensure_ascii=False, indent=2) + "\n")
        _write_text(prompt_path, prompt_text.rstrip() + "\n")

        if config.dry_run:
            raw_payload = {
                "scene_id": scene.scene_id,
                "unit_id": scene.scene_id,
                "status": "not_run",
                "reason": "dry_run",
                "raw_text": "",
            }
            parsed_payload = {
                "scene_id": scene.scene_id,
                "unit_id": scene.scene_id,
                "status": "not_parsed",
                "reason": "dry_run",
                "data": None,
                "validation_errors": [],
            }
            status = "dry_run_rendered"
            error = None
        else:
            try:
                assert llm_client is not None
                result = llm_client.complete(prompt_text)
                completed_count += 1
                parse_result = extract_json_value(result.text)
                validation_errors = (
                    validate_scene_inventory(parse_result.data, expected_scene_id=scene.scene_id)
                    if parse_result.ok
                    else []
                )
                parsed_ok = parse_result.ok and not validation_errors
                if parsed_ok:
                    parsed_count += 1
                    status = "completed"
                else:
                    failed_count += 1
                    status = "parse_failed" if not parse_result.ok else "validation_failed"
                error = parse_result.error if not parse_result.ok else "; ".join(validation_errors) or None
                raw_payload = {
                    "scene_id": scene.scene_id,
                    "unit_id": scene.scene_id,
                    "status": "completed",
                    "provider": result.provider,
                    "model": result.model,
                    "raw_text": result.text,
                    "usage": result.usage,
                    "raw_response": result.raw_response,
                }
                parsed_payload = {
                    "scene_id": scene.scene_id,
                    "unit_id": scene.scene_id,
                    "status": "parsed" if parsed_ok else status,
                    "data": parse_result.data if parse_result.ok else None,
                    "parse_error": parse_result.error,
                    "validation_errors": validation_errors,
                }
            except Exception as exc:  # noqa: BLE001 - runner must preserve per-scene failures.
                failed_count += 1
                status = "llm_failed"
                error = str(exc)
                raw_payload = {
                    "scene_id": scene.scene_id,
                    "unit_id": scene.scene_id,
                    "status": "llm_failed",
                    "error": str(exc),
                    "raw_text": "",
                }
                parsed_payload = {
                    "scene_id": scene.scene_id,
                    "unit_id": scene.scene_id,
                    "status": "not_parsed",
                    "reason": "llm_failed",
                    "data": None,
                    "validation_errors": [],
                }

        _write_text(raw_output_path, json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n")
        _write_text(parsed_path, json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n")

        trace_records.append(
            {
                "ordinal": ordinal,
                "scene_id": scene.scene_id,
                "unit_id": scene.scene_id,
                "source_record_id": scene.source_record_id,
                "discourse_index": scene.discourse_index,
                "title": scene.title,
                "input_path": str(input_path),
                "prompt_path": str(prompt_path),
                "raw_output_path": str(raw_output_path),
                "parsed_path": str(parsed_path),
                "status": status,
                "error": error,
                "prompt_char_count": len(prompt_text),
                "input_char_count": scene.character_count,
            }
        )

    with trace_path.open("w", encoding="utf-8") as handle:
        for record in trace_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "run_type": "scene_inventory",
        "created_at": created_at,
        "dry_run": config.dry_run,
        "llm": {
            "provider": getattr(llm_client, "provider", None) if llm_client else None,
            "model": getattr(llm_client, "model", None) if llm_client else None,
        },
        "script_path": str(config.script_path.resolve()),
        "prompt_dir": str(config.prompt_dir.resolve()),
        "prompt_id": prompt_id,
        "prompt_path": str(prompt_spec.path.resolve()),
        "task_settings_path": str(config.task_settings_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selection": {
            "start": config.start,
            "limit": config.limit,
            "selected_count": len(selected),
            "first_scene_id": selected[0].scene_id if selected else None,
            "last_scene_id": selected[-1].scene_id if selected else None,
        },
        "artifact_paths": {
            "inputs_dir": str(inputs_dir),
            "prompts_dir": str(prompts_dir),
            "raw_outputs_dir": str(raw_outputs_dir),
            "parsed_dir": str(parsed_dir),
            "trace_path": str(trace_path),
            "summary_path": str(output_dir / "summary.json"),
        },
        "config": {
            **asdict(config),
            "script_path": str(config.script_path),
            "output_dir": str(config.output_dir),
            "prompt_dir": str(config.prompt_dir),
            "task_settings_path": str(config.task_settings_path),
        },
    }
    summary = {
        "run_type": "scene_inventory",
        "status": "dry_run_complete" if config.dry_run else "complete",
        "selected_count": len(selected),
        "rendered_prompt_count": len(trace_records),
        "llm_completed_count": completed_count,
        "parsed_output_count": parsed_count,
        "failed_count": failed_count,
        "raw_output_count": 0 if config.dry_run else completed_count,
        "input_char_count": sum(scene.character_count for scene in selected),
        "prompt_char_count": sum(record["prompt_char_count"] for record in trace_records),
        "trace_path": str(trace_path),
    }

    _write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _select_scenes(scenes: list[ScriptScene], *, start: int, limit: int) -> list[ScriptScene]:
    offset = start - 1
    return scenes[offset : offset + limit]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _format_policy(policy: Any) -> str:
    if isinstance(policy, list):
        return "\n".join(f"- {item}" for item in policy)
    return str(policy or "")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
