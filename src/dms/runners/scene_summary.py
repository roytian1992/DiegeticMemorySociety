from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dms.chunking import DEFAULT_MAX_CHUNK_UNITS, NarrativeChunk, chunk_scene
from dms.llm import LLMClient
from dms.parsing import extract_json_value, validate_scene_summary
from dms.prompts import YAMLPromptLoader
from dms.runners.prompt_payloads import narrative_unit_payload
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes


@dataclass(frozen=True)
class SceneSummaryRunConfig:
    script_path: Path
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    task_settings_path: Path = Path("task_specs/task_settings/scene_summary_task.json")
    start: int = 1
    limit: int = 5
    dry_run: bool = True
    overwrite: bool = False
    max_chunk_units: int = DEFAULT_MAX_CHUNK_UNITS


def run_scene_summary(config: SceneSummaryRunConfig, *, llm_client: LLMClient | None = None) -> dict[str, Any]:
    """Render and optionally execute scene-summary prompts for ordered narrative units."""

    if not config.dry_run and llm_client is None:
        raise ValueError("llm_client is required when dry_run is false")
    if config.limit < 1:
        raise ValueError("limit must be >= 1")
    if config.start < 1:
        raise ValueError("start must be >= 1")
    if config.max_chunk_units < 1:
        raise ValueError("max_chunk_units must be >= 1")

    output_dir = config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output dir exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)

    prompts_dir = output_dir / "prompts"
    inputs_dir = output_dir / "inputs"
    raw_outputs_dir = output_dir / "raw_outputs"
    parsed_dir = output_dir / "parsed"
    for directory in (prompts_dir, inputs_dir, raw_outputs_dir, parsed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    task_settings = _read_json(config.task_settings_path)
    prompt_id = str(task_settings["prompt_id"])
    summary_policy = _format_policy(task_settings.get("summary_policy", []))

    scenes = load_script_scenes(config.script_path)
    selected = _select_scenes(scenes, start=config.start, limit=config.limit)
    chunks_by_scene = [(scene, chunk_scene(scene, max_chunk_units=config.max_chunk_units)) for scene in selected]
    selected_chunks = [chunk for _, chunks in chunks_by_scene for chunk in chunks]
    loader = YAMLPromptLoader(config.prompt_dir)
    prompt_spec = loader.load(prompt_id)
    created_at = datetime.now(timezone.utc).isoformat()

    trace_path = output_dir / "trace.jsonl"
    trace_records: list[dict[str, Any]] = []
    completed_count = 0
    parsed_count = 0
    failed_count = 0

    for ordinal, unit in enumerate(selected_chunks, start=1):
        unit_payload = narrative_unit_payload(unit)
        unit_id = str(unit_payload["unit_id"])
        unit_json = json.dumps(unit_payload, ensure_ascii=False, indent=2)
        prompt_text = loader.render(
            prompt_spec,
            task_values={"unit_json": unit_json},
            static_values={"summary_policy": summary_policy},
        )

        input_path = inputs_dir / f"{unit_id}.json"
        prompt_path = prompts_dir / f"{unit_id}.txt"
        raw_output_path = raw_outputs_dir / f"{unit_id}.json"
        parsed_path = parsed_dir / f"{unit_id}.json"

        _write_text(input_path, json.dumps(unit_payload, ensure_ascii=False, indent=2) + "\n")
        _write_text(prompt_path, prompt_text.rstrip() + "\n")

        if config.dry_run:
            raw_payload = _wrapper(unit, status="not_run", reason="dry_run", raw_text="")
            parsed_payload = _wrapper(unit, status="not_parsed", reason="dry_run", data=None, validation_errors=[])
            status = "dry_run_rendered"
            error = None
        else:
            try:
                assert llm_client is not None
                result = llm_client.complete(prompt_text)
                completed_count += 1
                parse_result = extract_json_value(result.text)
                validation_errors = (
                    validate_scene_summary(parse_result.data, expected_scene_id=unit_id) if parse_result.ok else []
                )
                parsed_ok = parse_result.ok and not validation_errors
                if parsed_ok:
                    parsed_count += 1
                    status = "completed"
                else:
                    failed_count += 1
                    status = "parse_failed" if not parse_result.ok else "validation_failed"
                error = parse_result.error if not parse_result.ok else "; ".join(validation_errors) or None
                raw_payload = _wrapper(
                    unit,
                    status="completed",
                    provider=result.provider,
                    model=result.model,
                    raw_text=result.text,
                    usage=result.usage,
                    raw_response=result.raw_response,
                )
                parsed_payload = _wrapper(
                    unit,
                    status="parsed" if parsed_ok else status,
                    data=parse_result.data if parse_result.ok else None,
                    parse_error=parse_result.error,
                    validation_errors=validation_errors,
                )
            except Exception as exc:  # noqa: BLE001 - runner must preserve per-unit failures.
                failed_count += 1
                status = "llm_failed"
                error = str(exc)
                raw_payload = _wrapper(unit, status="llm_failed", error=str(exc), raw_text="")
                parsed_payload = _wrapper(unit, status="not_parsed", reason="llm_failed", data=None, validation_errors=[])

        _write_text(raw_output_path, json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n")
        _write_text(parsed_path, json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n")

        trace_records.append(
            {
                "ordinal": ordinal,
                "scene_id": unit.scene_id,
                "unit_id": unit_id,
                "parent_unit_id": getattr(unit, "parent_unit_id", unit.scene_id),
                "chunk_id": unit_id,
                "chunk_index": getattr(unit, "chunk_index", 1),
                "chunk_count": getattr(unit, "chunk_count", 1),
                "source_record_id": unit.source_record_id,
                "discourse_index": unit.discourse_index,
                "title": unit.title,
                "input_path": str(input_path),
                "prompt_path": str(prompt_path),
                "raw_output_path": str(raw_output_path),
                "parsed_path": str(parsed_path),
                "status": status,
                "error": error,
                "prompt_char_count": len(prompt_text),
                "input_char_count": len(json.dumps(unit_payload, ensure_ascii=False, indent=2)),
            }
        )

    _write_jsonl(trace_path, trace_records)

    manifest = {
        "run_type": "scene_summary",
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
            "chunk_count": len(selected_chunks),
            "max_chunk_units": config.max_chunk_units,
            "first_scene_id": selected[0].scene_id if selected else None,
            "last_scene_id": selected[-1].scene_id if selected else None,
            "first_unit_id": selected_chunks[0].chunk_id if selected_chunks else None,
            "last_unit_id": selected_chunks[-1].chunk_id if selected_chunks else None,
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
        "run_type": "scene_summary",
        "status": "dry_run_complete" if config.dry_run else "complete",
        "selected_count": len(selected),
        "chunk_count": len(selected_chunks),
        "max_chunk_units": config.max_chunk_units,
        "rendered_prompt_count": len(trace_records),
        "llm_completed_count": completed_count,
        "parsed_output_count": parsed_count,
        "failed_count": failed_count,
        "raw_output_count": 0 if config.dry_run else completed_count,
        "input_char_count": sum(record["input_char_count"] for record in trace_records),
        "prompt_char_count": sum(record["prompt_char_count"] for record in trace_records),
        "trace_path": str(trace_path),
    }

    _write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _select_scenes(scenes: list[ScriptScene], *, start: int, limit: int) -> list[ScriptScene]:
    offset = start - 1
    return scenes[offset : offset + limit]


def _wrapper(unit: ScriptScene | NarrativeChunk, **values: Any) -> dict[str, Any]:
    unit_id = getattr(unit, "chunk_id", unit.scene_id)
    return {
        "scene_id": unit.scene_id,
        "unit_id": unit_id,
        "parent_unit_id": getattr(unit, "parent_unit_id", unit.scene_id),
        "chunk_id": unit_id,
        "chunk_index": getattr(unit, "chunk_index", 1),
        "chunk_count": getattr(unit, "chunk_count", 1),
        **values,
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _format_policy(policy: Any) -> str:
    if isinstance(policy, list):
        return "\n".join(f"- {item}" for item in policy)
    return str(policy or "")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
