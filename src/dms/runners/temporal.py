from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dms.llm import LLMClient
from dms.parsing import extract_json_value, validate_temporal_extraction
from dms.progress import print_progress
from dms.prompts import YAMLPromptLoader
from dms.runners.prompt_payloads import narrative_unit_payload
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes
from dms.timeline import (
    TimelineBuildConfig,
    apply_temporal_audit_annotations,
    build_timeline_graph,
    format_temporal_audit_report,
    format_timeline_report,
    normalize_temporal_scene_output,
    verify_temporal_outputs,
)


@dataclass(frozen=True)
class TemporalExtractionRunConfig:
    script_path: Path
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    task_settings_path: Path = Path("task_specs/task_settings/temporal_extraction_task.json")
    prior_timeline_path: Path | None = None
    start: int = 1
    limit: int = 5
    dry_run: bool = True
    overwrite: bool = False


def run_temporal_extraction(
    config: TemporalExtractionRunConfig,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Render and optionally run diegetic temporal extraction over ordered scenes."""

    if not config.dry_run and llm_client is None:
        raise ValueError("llm_client is required when dry_run is false")
    if config.limit < 1:
        raise ValueError("limit must be >= 1")
    if config.start < 1:
        raise ValueError("start must be >= 1")

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
    extraction_policy = _format_policy(task_settings.get("extraction_policy", []))
    prior_timeline = _read_prior_timeline(config.prior_timeline_path)
    scenes = load_script_scenes(config.script_path)
    selected = _select_scenes(scenes, start=config.start, limit=config.limit)
    loader = YAMLPromptLoader(config.prompt_dir)
    prompt_spec = loader.load(prompt_id)
    created_at = datetime.now(timezone.utc).isoformat()
    trace_path = output_dir / "trace.jsonl"

    trace_records: list[dict[str, Any]] = []
    normalized_outputs: list[dict[str, Any]] = []
    source_units_by_scene: dict[str, dict[str, Any]] = {}
    completed_count = 0
    parsed_count = 0
    failed_count = 0

    for ordinal, scene in enumerate(selected, start=1):
        print_progress(
            "temporal_extraction:scene",
            ordinal - 1,
            len(selected),
            detail=f"scene={scene.scene_id} status=start",
        )
        unit_payload = narrative_unit_payload(scene)
        source_units_by_scene[scene.scene_id] = unit_payload
        unit_json = json.dumps(unit_payload, ensure_ascii=False, indent=2)
        previous_scene_id = _previous_scene_id(scene, scenes)
        next_scene_id = _next_scene_id(scene, scenes)
        prompt_text = loader.render(
            prompt_spec,
            task_values={
                "unit_json": unit_json,
                "prior_timeline": prior_timeline,
                "previous_scene_id": previous_scene_id or "",
                "next_scene_id": next_scene_id or "",
            },
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
                finish_reason = _completion_finish_reason(result.raw_response)
                truncated_output = finish_reason == "length"
                validation_errors = (
                    validate_temporal_extraction(parse_result.data, expected_scene_id=scene.scene_id)
                    if parse_result.ok
                    else []
                )
                parsed_ok = parse_result.ok and not validation_errors and not truncated_output
                if parsed_ok:
                    parsed_count += 1
                    status = "completed"
                    normalized_outputs.append(
                        normalize_temporal_scene_output(
                            parse_result.data,
                            scene_id=scene.scene_id,
                            source_record_id=scene.source_record_id,
                            discourse_index=scene.discourse_index,
                        )
                    )
                else:
                    failed_count += 1
                    status = (
                        "truncated_output"
                        if truncated_output
                        else "parse_failed"
                        if not parse_result.ok
                        else "validation_failed"
                    )
                error = (
                    f"completion finish_reason={finish_reason}"
                    if truncated_output
                    else parse_result.error
                    if not parse_result.ok
                    else "; ".join(validation_errors) or None
                )
                raw_payload = {
                    "scene_id": scene.scene_id,
                    "unit_id": scene.scene_id,
                    "status": "truncated_output" if truncated_output else "completed",
                    "provider": result.provider,
                    "model": result.model,
                    "raw_text": result.text,
                    "usage": result.usage,
                    "raw_response": result.raw_response,
                    "finish_reason": finish_reason,
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
        print_progress(
            "temporal_extraction:scene",
            ordinal,
            len(selected),
            detail=f"scene={scene.scene_id} status={status}",
        )
        trace_records.append(
            {
                "ordinal": ordinal,
                "scene_id": scene.scene_id,
                "unit_id": scene.scene_id,
                "source_record_id": scene.source_record_id,
                "discourse_index": scene.discourse_index,
                "title": scene.title,
                "previous_scene_id": previous_scene_id,
                "next_scene_id": next_scene_id,
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

    temporal_audit = verify_temporal_outputs(normalized_outputs, source_units_by_scene)
    apply_temporal_audit_annotations(normalized_outputs, temporal_audit)
    graph = build_timeline_graph(normalized_outputs, config=TimelineBuildConfig())
    _write_jsonl(output_dir / "events.jsonl", [event for output in normalized_outputs for event in output["temporal_events"]])
    _write_jsonl(
        output_dir / "relations.jsonl",
        [relation for output in normalized_outputs for relation in output["temporal_relations"]],
    )
    _write_jsonl(
        output_dir / "scene_temporal_index.jsonl",
        [output["scene_temporal_index"] for output in normalized_outputs],
    )
    _write_jsonl(output_dir / "timeline_order.jsonl", graph.get("timeline_order") or [])
    _write_text(output_dir / "timeline_graph.json", json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "conflicts.json", json.dumps(graph.get("conflicts") or [], ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "timeline_report.md", format_timeline_report(graph))
    _write_text(output_dir / "temporal_audit.json", json.dumps(temporal_audit, ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "temporal_audit_report.md", format_temporal_audit_report(temporal_audit))
    _write_jsonl(trace_path, trace_records)

    manifest = {
        "run_type": "temporal_extraction",
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
        "prior_timeline_path": str(config.prior_timeline_path.resolve()) if config.prior_timeline_path else None,
        "output_dir": str(output_dir.resolve()),
        "selection": {
            "start": config.start,
            "limit": config.limit,
            "selected_count": len(selected),
            "first_scene_id": selected[0].scene_id if selected else None,
            "last_scene_id": selected[-1].scene_id if selected else None,
        },
        "artifact_paths": _artifact_paths(output_dir),
        "config": {
            **asdict(config),
            "script_path": str(config.script_path),
            "output_dir": str(config.output_dir),
            "prompt_dir": str(config.prompt_dir),
            "task_settings_path": str(config.task_settings_path),
            "prior_timeline_path": str(config.prior_timeline_path) if config.prior_timeline_path else None,
        },
    }
    summary = {
        "run_type": "temporal_extraction",
        "status": "dry_run_complete" if config.dry_run else "complete",
        "selected_count": len(selected),
        "rendered_prompt_count": len(trace_records),
        "llm_completed_count": completed_count,
        "parsed_output_count": parsed_count,
        "failed_count": failed_count,
        "raw_output_count": 0 if config.dry_run else completed_count,
        "input_char_count": sum(scene.character_count for scene in selected),
        "prompt_char_count": sum(record["prompt_char_count"] for record in trace_records),
        "timeline_counts": graph.get("counts") or {},
        "temporal_audit_counts": temporal_audit.get("counts") or {},
        "trace_path": str(trace_path),
        "artifact_paths": _artifact_paths(output_dir),
    }
    _write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_text(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _select_scenes(scenes: list[ScriptScene], *, start: int, limit: int) -> list[ScriptScene]:
    offset = start - 1
    return scenes[offset : offset + limit]


def _previous_scene_id(scene: ScriptScene, scenes: list[ScriptScene]) -> str | None:
    ordered = sorted(scenes, key=lambda item: item.source_record_id)
    for index, item in enumerate(ordered):
        if item.scene_id == scene.scene_id and index > 0:
            return ordered[index - 1].scene_id
    return None


def _next_scene_id(scene: ScriptScene, scenes: list[ScriptScene]) -> str | None:
    ordered = sorted(scenes, key=lambda item: item.source_record_id)
    for index, item in enumerate(ordered):
        if item.scene_id == scene.scene_id and index + 1 < len(ordered):
            return ordered[index + 1].scene_id
    return None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_prior_timeline(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Prior timeline path not found: {path}")
    if path.is_dir():
        report = path / "timeline_report.md"
        if report.is_file():
            return report.read_text(encoding="utf-8")
        graph = path / "timeline_graph.json"
        if graph.is_file():
            return graph.read_text(encoding="utf-8")
        return "\n".join(sorted(item.name for item in path.iterdir()))
    return path.read_text(encoding="utf-8")


def _format_policy(policy: Any) -> str:
    if isinstance(policy, list):
        return "\n".join(f"- {item}" for item in policy)
    return str(policy or "")


def _artifact_paths(output_dir: Path) -> dict[str, str]:
    return {
        "inputs_dir": str(output_dir / "inputs"),
        "prompts_dir": str(output_dir / "prompts"),
        "raw_outputs_dir": str(output_dir / "raw_outputs"),
        "parsed_dir": str(output_dir / "parsed"),
        "trace_path": str(output_dir / "trace.jsonl"),
        "events": str(output_dir / "events.jsonl"),
        "relations": str(output_dir / "relations.jsonl"),
        "scene_temporal_index": str(output_dir / "scene_temporal_index.jsonl"),
        "timeline_graph": str(output_dir / "timeline_graph.json"),
        "timeline_order": str(output_dir / "timeline_order.jsonl"),
        "conflicts": str(output_dir / "conflicts.json"),
        "timeline_report": str(output_dir / "timeline_report.md"),
        "temporal_audit": str(output_dir / "temporal_audit.json"),
        "temporal_audit_report": str(output_dir / "temporal_audit_report.md"),
        "summary": str(output_dir / "summary.json"),
    }


def _completion_finish_reason(raw_response: dict[str, Any]) -> str | None:
    choices = raw_response.get("choices") if isinstance(raw_response, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    reason = first.get("finish_reason") or first.get("stop_reason")
    return str(reason) if reason is not None else None


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
