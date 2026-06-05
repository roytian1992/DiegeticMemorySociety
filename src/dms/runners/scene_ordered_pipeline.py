from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dms.author_context import (
    author_entities_from_context,
    format_author_entity_context_for_prompt,
    load_author_entity_context,
)
from dms.chunking import DEFAULT_MAX_CHUNK_UNITS, NarrativeChunk, chunk_scene
from dms.entity_alignment import sanitize_kg_entity_output
from dms.llm import LLMClient
from dms.memory import (
    build_canonical_memory,
    build_entity_resolution_artifacts,
    build_episodic_memory,
    build_kg_entity_memory,
    build_prefix_commits,
    build_prefix_world_model,
    build_relationship_memory,
    build_scene_inventory_memory,
    build_scene_summary_memory,
)
from dms.narrative_units import DEFAULT_UNIT_LABEL, DEFAULT_UNIT_TYPE
from dms.parsing import (
    JSONParseResult,
    extract_json_value,
    validate_durable_relationships,
    validate_episodic_memories,
    validate_kg_entity_mentions,
    validate_scene_inventory,
    validate_scene_summary,
)
from dms.progress import print_progress
from dms.prompts import YAMLPromptLoader
from dms.runners.prompt_payloads import narrative_unit_payload
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes


TASK_SCENE_SUMMARY = "scene_summary"
TASK_SCENE_INVENTORY = "scene_inventory"
TASK_KG_ENTITIES = "kg_entity_mentions"
TASK_KG_REFINEMENT = "kg_entity_refinement"
TASK_EPISODIC_MEMORIES = "episodic_memories"
TASK_DURABLE_RELATIONSHIPS = "durable_relationships"

FIRST_WAVE_TASKS = (TASK_SCENE_SUMMARY, TASK_SCENE_INVENTORY, TASK_KG_ENTITIES)
DEPENDENT_TASKS = (TASK_EPISODIC_MEMORIES, TASK_DURABLE_RELATIONSHIPS)
ALL_TASKS = (*FIRST_WAVE_TASKS, TASK_KG_REFINEMENT, *DEPENDENT_TASKS)
DEBUG_DIR_NAME = "_debug"


@dataclass(frozen=True)
class SceneOrderedPipelineConfig:
    script_path: Path
    output_root: Path
    base_output_root: Path | None = None
    prompt_dir: Path = Path("task_specs/prompts")
    scene_summary_task_settings_path: Path = Path("task_specs/task_settings/scene_summary_task.json")
    scene_inventory_task_settings_path: Path = Path("task_specs/task_settings/scene_inventory_task.json")
    kg_entity_task_settings_path: Path = Path("task_specs/task_settings/kg_entity_mentions_task.json")
    kg_entity_refinement_task_settings_path: Path = Path("task_specs/task_settings/kg_entity_refinement_task.json")
    episodic_memory_task_settings_path: Path = Path("task_specs/task_settings/episodic_memories_task.json")
    durable_relationship_task_settings_path: Path = Path("task_specs/task_settings/durable_relationships_task.json")
    prior_entity_context_path: Path | None = None
    start: int = 1
    limit: int = 5
    dry_run: bool = False
    overwrite: bool = False
    scene_task_concurrency: int = 3
    max_chunk_units: int = DEFAULT_MAX_CHUNK_UNITS
    unit_type: str = DEFAULT_UNIT_TYPE
    unit_label: str = DEFAULT_UNIT_LABEL


@dataclass(frozen=True)
class _TaskContext:
    task_name: str
    run_type: str
    run_dir: Path
    prompt_id: str
    prompt_path: Path
    prompt_spec: Any
    static_values: dict[str, Any]
    base_task_values: dict[str, Any]
    validate: Callable[..., list[str]]


@dataclass(frozen=True)
class _TaskUnitResult:
    task_name: str
    scene_id: str
    unit_id: str
    unit_type: str
    unit_label: str
    unit_order: int | None
    status: str
    error: str | None
    completed_count: int
    parsed_count: int
    failed_count: int
    prompt_char_count: int
    input_char_count: int
    paths: dict[str, str]
    parsed_payload: dict[str, Any]
    parent_unit_id: str
    chunk_index: int
    chunk_count: int

    @property
    def parsed_ok(self) -> bool:
        return self.parsed_payload.get("status") == "parsed"


def run_scene_ordered_pipeline(
    config: SceneOrderedPipelineConfig,
    *,
    llm_clients: dict[str, LLMClient] | None = None,
) -> dict[str, Any]:
    """Run units in order while executing independent extraction tasks per unit concurrently."""

    if config.limit < 1:
        raise ValueError("limit must be >= 1")
    if config.start < 1:
        raise ValueError("start must be >= 1")
    if config.scene_task_concurrency < 1:
        raise ValueError("scene_task_concurrency must be >= 1")
    if config.max_chunk_units < 1:
        raise ValueError("max_chunk_units must be >= 1")
    if not config.dry_run:
        missing = [task for task in ALL_TASKS if not llm_clients or task not in llm_clients]
        if missing:
            raise ValueError(f"Missing llm_clients for tasks: {missing}")

    output_root = config.output_root
    if output_root.exists() and any(output_root.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output root exists and is not empty: {output_root}")
        shutil.rmtree(output_root)
    if config.base_output_root:
        if config.dry_run:
            raise ValueError("base_output_root is not supported with dry_run")
        _copy_base_run(config.base_output_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = load_script_scenes(config.script_path, unit_type=config.unit_type, unit_label=config.unit_label)
    selected = _select_scenes(scenes, start=config.start, limit=config.limit)
    chunks_by_scene = [(scene, chunk_scene(scene, max_chunk_units=config.max_chunk_units)) for scene in selected]
    selected_chunks = [chunk for _, chunks in chunks_by_scene for chunk in chunks]
    loader = YAMLPromptLoader(config.prompt_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    author_entity_context = load_author_entity_context(config.prior_entity_context_path)
    task_contexts = _build_task_contexts(config, loader, author_entity_context=author_entity_context)

    task_records: dict[str, list[dict[str, Any]]] = {task: [] for task in ALL_TASKS}
    base_unit_records = _load_jsonl(output_root / DEBUG_DIR_NAME / "unit_trace.jsonl") if config.base_output_root else []
    task_records.update(_load_existing_task_records(output_root) if config.base_output_root else {})
    unit_records: list[dict[str, Any]] = list(base_unit_records)
    chunk_manifest_path = output_root / DEBUG_DIR_NAME / "chunk_manifest.jsonl"
    base_chunk_manifest = _load_jsonl(chunk_manifest_path) if config.base_output_root else []
    _write_jsonl(
        chunk_manifest_path,
        [
            *base_chunk_manifest,
            *(
                _chunk_manifest_record(index, chunk)
                for index, chunk in enumerate(selected_chunks, start=len(base_chunk_manifest) + 1)
            ),
        ],
    )
    prior_entity_mentions = _load_prior_entity_mentions_from_run(output_root) if config.base_output_root else []
    author_entities = author_entities_from_context(author_entity_context)

    max_workers = min(config.scene_task_concurrency, len(FIRST_WAVE_TASKS))
    chunk_ordinal = len(base_chunk_manifest)
    processed_chunks = 0
    print_progress(
        "scene_ordered:start",
        0,
        len(selected_chunks),
        detail=f"scenes={len(selected)} chunks={len(selected_chunks)} output_root={output_root}",
    )
    for _scene, chunks in chunks_by_scene:
        for chunk in chunks:
            chunk_ordinal += 1
            processed_chunks += 1
            first_wave_results: dict[str, _TaskUnitResult] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    task: executor.submit(
                        _run_task_for_scene,
                        task_contexts[task],
                        chunk,
                        llm_clients[task] if llm_clients else None,
                        dry_run=config.dry_run,
                        task_values=(
                            {"prior_entity_context": _format_prior_entity_context(author_entities, prior_entity_mentions)}
                            if task == TASK_KG_ENTITIES
                            else None
                        ),
                    )
                    for task in FIRST_WAVE_TASKS
                }
                for task, future in futures.items():
                    first_wave_results[task] = future.result()

            kg_refinement_result = _run_task_for_scene(
                task_contexts[TASK_KG_REFINEMENT],
                chunk,
                llm_clients[TASK_KG_REFINEMENT] if llm_clients else None,
                dry_run=config.dry_run,
                task_values={
                    "unit_context_json": json.dumps(_unit_refinement_context(chunk), ensure_ascii=False, indent=2),
                    "scene_inventory_json": json.dumps(
                        _candidate_payload(first_wave_results[TASK_SCENE_INVENTORY]),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "initial_entity_json": json.dumps(
                        _candidate_payload(first_wave_results[TASK_KG_ENTITIES]),
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
                input_payload_extra={
                    "unit_context": _unit_refinement_context(chunk),
                    "scene_inventory": _candidate_payload(first_wave_results[TASK_SCENE_INVENTORY]),
                    "initial_entities": _candidate_payload(first_wave_results[TASK_KG_ENTITIES]),
                },
            )
            kg_candidate_result = kg_refinement_result if kg_refinement_result.parsed_ok else first_wave_results[TASK_KG_ENTITIES]
            _append_prior_entity_mentions(prior_entity_mentions, kg_candidate_result)
            dependent_candidates = {
                TASK_SCENE_INVENTORY: _candidate_payload(first_wave_results[TASK_SCENE_INVENTORY]),
                TASK_KG_ENTITIES: _candidate_payload(kg_candidate_result),
            }
            dependent_results: dict[str, _TaskUnitResult] = {}
            with ThreadPoolExecutor(max_workers=min(config.scene_task_concurrency, len(DEPENDENT_TASKS))) as executor:
                futures = {
                    task: executor.submit(
                        _run_task_for_scene,
                        task_contexts[task],
                        chunk,
                        llm_clients[task] if llm_clients else None,
                        dry_run=config.dry_run,
                        task_values={"extracted_candidates": dependent_candidates},
                        input_payload_extra={"extracted_candidates": dependent_candidates},
                    )
                    for task in DEPENDENT_TASKS
                }
                for task, future in futures.items():
                    dependent_results[task] = future.result()

            scene_results = {
                **first_wave_results,
                TASK_KG_REFINEMENT: kg_refinement_result,
                **dependent_results,
            }
            for task in ALL_TASKS:
                task_records[task].append(_trace_record(chunk_ordinal, chunk, scene_results[task]))

            unit_records.append(
                {
                    "ordinal": chunk_ordinal,
                    "scene_id": chunk.scene_id,
                    "unit_id": chunk.chunk_id,
                    "unit_type": chunk.unit_type,
                    "unit_label": chunk.unit_label,
                    "unit_order": chunk.discourse_index,
                    "parent_unit_id": chunk.parent_unit_id,
                    "parent_unit_type": chunk.unit_type,
                    "parent_unit_label": chunk.unit_label,
                    "parent_unit_order": chunk.discourse_index,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": chunk.chunk_count,
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                    "source_sha256": chunk.source_sha256,
                    "source_record_id": chunk.source_record_id,
                    "discourse_index": chunk.discourse_index,
                    "title": chunk.title,
                    "task_statuses": {task: scene_results[task].status for task in ALL_TASKS},
                    "task_errors": {task: scene_results[task].error for task in ALL_TASKS if scene_results[task].error},
                }
            )
            failed_tasks = [task for task in ALL_TASKS if scene_results[task].error]
            print_progress(
                "scene_ordered:chunk",
                processed_chunks,
                len(selected_chunks),
                detail=f"unit={chunk.chunk_id} scene={chunk.scene_id} failed_tasks={len(failed_tasks)}",
            )

    task_summaries = {
        task: _write_task_run_summary(
            context=task_contexts[task],
            records=task_records[task],
            created_at=created_at,
            config=config,
            selected=selected_chunks,
            llm_client=(llm_clients or {}).get(task),
        )
        for task in ALL_TASKS
    }

    unit_trace_path = output_root / DEBUG_DIR_NAME / "unit_trace.jsonl"
    _write_jsonl(unit_trace_path, unit_records)

    memory_summaries: dict[str, Any] = {}
    if not config.dry_run:
        print_progress("scene_ordered:memory", 0, 1, detail=f"output_root={output_root}")
        memory_summaries = _build_pipeline_memory(output_root, task_contexts, author_entity_context=author_entity_context)
        print_progress("scene_ordered:memory", 1, 1, detail=f"output_root={output_root}")

    summary = {
        "run_type": "scene_ordered_pipeline",
        "status": "dry_run_complete" if config.dry_run else "complete",
        "created_at": created_at,
        "output_root": str(output_root),
        "selected_count": len(selected),
        "chunk_count": len(selected_chunks),
        "max_chunk_units": config.max_chunk_units,
        "scene_task_concurrency": config.scene_task_concurrency,
        "narrative_unit": {
            "unit_type": config.unit_type,
            "unit_label": config.unit_label,
            "legacy_scene_id_compatibility": True,
        },
        "task_order": {
            "per_unit_parallel": list(FIRST_WAVE_TASKS),
            "per_unit_refinement": TASK_KG_REFINEMENT,
            "per_unit_after_entities_parallel": list(DEPENDENT_TASKS),
            "cross_unit_order": "sequential",
        },
        "task_summaries": task_summaries,
        "memory_summaries": memory_summaries,
        "artifact_paths": _public_artifact_paths(output_root),
        "unit_trace_path": str(unit_trace_path),
        "chunk_manifest_path": str(chunk_manifest_path),
        "config": _serializable_config(config),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_run_readme(output_root, summary)
    return summary


def _run_task_for_scene(
    context: _TaskContext,
    scene: ScriptScene | NarrativeChunk,
    llm_client: LLMClient | None,
    *,
    dry_run: bool,
    task_values: dict[str, Any] | None = None,
    input_payload_extra: dict[str, Any] | None = None,
) -> _TaskUnitResult:
    unit_payload = narrative_unit_payload(scene)
    merged_task_values = {"unit_json": json.dumps(unit_payload, ensure_ascii=False, indent=2)}
    merged_task_values.update(context.base_task_values)
    merged_task_values.update(task_values or {})
    prompt_text = YAMLPromptLoader(context.prompt_path.parent.parent).render(
        context.prompt_spec,
        task_values=merged_task_values,
        static_values=context.static_values,
    )

    input_payload: dict[str, Any] = {"unit": unit_payload} if input_payload_extra else unit_payload
    if input_payload_extra:
        input_payload.update(input_payload_extra)
    input_text = json.dumps(input_payload, ensure_ascii=False, indent=2) + "\n"

    unit_id = str(unit_payload["unit_id"])
    input_path = context.run_dir / "inputs" / f"{unit_id}.json"
    prompt_path = context.run_dir / "prompts" / f"{unit_id}.txt"
    raw_output_path = context.run_dir / "raw_outputs" / f"{unit_id}.json"
    repair_prompt_path = context.run_dir / "prompts" / f"{unit_id}.repair.txt"
    repair_raw_output_path = context.run_dir / "raw_outputs" / f"{unit_id}.repair.json"
    parsed_path = context.run_dir / "parsed" / f"{unit_id}.json"

    _write_text(input_path, input_text)
    _write_text(prompt_path, prompt_text.rstrip() + "\n")

    if dry_run:
        raw_payload = _wrapper(scene, status="not_run", reason="dry_run", raw_text="")
        parsed_payload = _wrapper(scene, status="not_parsed", reason="dry_run", data=None, validation_errors=[])
        status = "dry_run_rendered"
        error = None
        completed_count = 0
        parsed_count = 0
        failed_count = 0
    else:
        try:
            if llm_client is None:
                raise ValueError(f"llm_client is required for task {context.task_name}")
            result = llm_client.complete(prompt_text)
            parse_result = extract_json_value(result.text)
            parse_result = _normalize_parse_result_for_task(context.task_name, parse_result)
            validation_errors = (
                context.validate(
                    parse_result.data,
                    expected_scene_id=unit_id,
                    source_unit=unit_payload,
                )
                if parse_result.ok and context.task_name in {TASK_EPISODIC_MEMORIES, TASK_DURABLE_RELATIONSHIPS}
                else (context.validate(parse_result.data, expected_scene_id=unit_id) if parse_result.ok else [])
            )
            repair_payload: dict[str, Any] | None = None
            if (
                parse_result.ok
                and context.task_name in {TASK_EPISODIC_MEMORIES, TASK_DURABLE_RELATIONSHIPS}
                and _has_evidence_alignment_error(validation_errors)
            ):
                repair_prompt = _repair_prompt(
                    original_prompt=prompt_text,
                    original_output=parse_result.data,
                    validation_errors=validation_errors,
                    unit_payload=unit_payload,
                )
                _write_text(repair_prompt_path, repair_prompt.rstrip() + "\n")
                repair_result = llm_client.complete(repair_prompt)
                repair_parse_result = extract_json_value(repair_result.text)
                repair_parse_result = _normalize_parse_result_for_task(context.task_name, repair_parse_result)
                repair_validation_errors = (
                    context.validate(
                        repair_parse_result.data,
                        expected_scene_id=unit_id,
                        source_unit=unit_payload,
                    )
                    if repair_parse_result.ok
                    else []
                )
                repair_hard_errors, repair_warnings = _split_validation_issues(repair_validation_errors, context.task_name)
                repair_accepts_with_warnings = repair_parse_result.ok and not repair_hard_errors
                repair_payload = _wrapper(
                    scene,
                    status=(
                        "parsed"
                        if repair_accepts_with_warnings
                        else ("parse_failed" if not repair_parse_result.ok else "validation_failed")
                    ),
                    provider=repair_result.provider,
                    model=repair_result.model,
                    raw_text=repair_result.text,
                    usage=repair_result.usage,
                    raw_response=repair_result.raw_response,
                    parse_error=repair_parse_result.error,
                    validation_errors=repair_hard_errors,
                    validation_warnings=repair_warnings,
                    repair_ok=repair_accepts_with_warnings,
                )
                _write_text(repair_raw_output_path, json.dumps(repair_payload, ensure_ascii=False, indent=2) + "\n")
                if repair_accepts_with_warnings:
                    parse_result = repair_parse_result
                    validation_errors = repair_validation_errors
            hard_validation_errors, validation_warnings = _split_validation_issues(validation_errors, context.task_name)
            parsed_ok = parse_result.ok and not hard_validation_errors
            status = "completed" if parsed_ok else ("parse_failed" if not parse_result.ok else "validation_failed")
            error = parse_result.error if not parse_result.ok else "; ".join(hard_validation_errors) or None
            raw_payload = _wrapper(
                scene,
                status="completed",
                provider=result.provider,
                model=result.model,
                raw_text=result.text,
                usage=result.usage,
                raw_response=result.raw_response,
            )
            parsed_payload = _wrapper(
                scene,
                status="parsed" if parsed_ok else status,
                data=parse_result.data if parse_result.ok else None,
                parse_error=parse_result.error,
                validation_errors=hard_validation_errors,
                validation_warnings=validation_warnings,
                repair_attempted=repair_payload is not None,
                repair_status=repair_payload.get("status") if repair_payload else None,
            )
            completed_count = 1
            parsed_count = 1 if parsed_ok else 0
            failed_count = 0 if parsed_ok else 1
        except Exception as exc:  # noqa: BLE001 - per-task failures must be persisted.
            status = "llm_failed"
            error = str(exc)
            raw_payload = _wrapper(scene, status="llm_failed", error=str(exc), raw_text="")
            parsed_payload = _wrapper(scene, status="not_parsed", reason="llm_failed", data=None, validation_errors=[])
            completed_count = 0
            parsed_count = 0
            failed_count = 1

    _write_text(raw_output_path, json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n")
    _write_text(parsed_path, json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n")

    return _TaskUnitResult(
        task_name=context.task_name,
        scene_id=scene.scene_id,
        unit_id=unit_id,
        unit_type=getattr(scene, "unit_type", DEFAULT_UNIT_TYPE),
        unit_label=getattr(scene, "unit_label", DEFAULT_UNIT_LABEL),
        unit_order=scene.discourse_index,
        status=status,
        error=error,
        completed_count=completed_count,
        parsed_count=parsed_count,
        failed_count=failed_count,
        prompt_char_count=len(prompt_text),
        input_char_count=len(input_text),
        paths={
            "input_path": str(input_path),
            "prompt_path": str(prompt_path),
            "raw_output_path": str(raw_output_path),
            "parsed_path": str(parsed_path),
            "repair_prompt_path": str(repair_prompt_path) if repair_prompt_path.exists() else "",
            "repair_raw_output_path": str(repair_raw_output_path) if repair_raw_output_path.exists() else "",
        },
        parsed_payload=parsed_payload,
        parent_unit_id=getattr(scene, "parent_unit_id", scene.scene_id),
        chunk_index=getattr(scene, "chunk_index", 1),
        chunk_count=getattr(scene, "chunk_count", 1),
    )


def _has_evidence_alignment_error(validation_errors: list[str]) -> bool:
    return any(".evidence must align to a contiguous span" in error for error in validation_errors)


def _normalize_parse_result_for_task(task_name: str, parse_result: JSONParseResult) -> JSONParseResult:
    if not parse_result.ok:
        return parse_result
    if task_name in {TASK_KG_ENTITIES, TASK_KG_REFINEMENT}:
        return JSONParseResult(ok=True, data=sanitize_kg_entity_output(parse_result.data), error=None)
    return parse_result


def _split_validation_issues(validation_errors: list[str], task_name: str) -> tuple[list[str], list[str]]:
    if task_name not in {TASK_EPISODIC_MEMORIES, TASK_DURABLE_RELATIONSHIPS}:
        return validation_errors, []
    warnings = [error for error in validation_errors if _is_soft_evidence_error(error, task_name)]
    hard_errors = [error for error in validation_errors if error not in warnings]
    return hard_errors, warnings


def _is_soft_evidence_error(error: str, task_name: str) -> bool:
    if task_name not in {TASK_EPISODIC_MEMORIES, TASK_DURABLE_RELATIONSHIPS}:
        return False
    if ".evidence must align to a contiguous span" not in error:
        return False
    return True


def _repair_prompt(
    *,
    original_prompt: str,
    original_output: Any,
    validation_errors: list[str],
    unit_payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "You previously returned JSON for a narrative extraction task, but some evidence fields could not be aligned to the source text.",
            "Revise only evidence fields that caused validation errors. Keep the same JSON schema, memory ordering, summaries, entities, relation labels, and IDs unless a field is impossible to support from the source.",
            "Every evidence value must be one exact contiguous substring copied from the current narrative unit title, subtitle, or content. Do not paraphrase, translate, join separated spans, add ellipses, or include text from extracted candidates.",
            "",
            "# Validation Errors",
            json.dumps(validation_errors, ensure_ascii=False, indent=2),
            "",
            "# Current Narrative Unit",
            json.dumps(unit_payload, ensure_ascii=False, indent=2),
            "",
            "# Previous Output To Repair",
            json.dumps(original_output, ensure_ascii=False, indent=2),
            "",
            "# Original Task Prompt",
            original_prompt,
        ]
    )


def _build_task_contexts(
    config: SceneOrderedPipelineConfig,
    loader: YAMLPromptLoader,
    *,
    author_entity_context: dict[str, Any],
) -> dict[str, _TaskContext]:
    task_specs = {
        TASK_SCENE_SUMMARY: (
            "scene_summary",
            config.scene_summary_task_settings_path,
            validate_scene_summary,
            {},
            {},
        ),
        TASK_SCENE_INVENTORY: (
            "scene_inventory",
            config.scene_inventory_task_settings_path,
            validate_scene_inventory,
            {},
            {},
        ),
        TASK_KG_ENTITIES: (
            "kg_entity_mentions",
            config.kg_entity_task_settings_path,
            validate_kg_entity_mentions,
            {"prior_entity_context": format_author_entity_context_for_prompt(author_entity_context)},
            {},
        ),
        TASK_KG_REFINEMENT: (
            "kg_entity_refinement",
            config.kg_entity_refinement_task_settings_path,
            validate_kg_entity_mentions,
            {},
            {"unit_context_json": "{}", "scene_inventory_json": "{}", "initial_entity_json": "{}"},
        ),
        TASK_EPISODIC_MEMORIES: (
            "episodic_memories",
            config.episodic_memory_task_settings_path,
            validate_episodic_memories,
            {},
            {"extracted_candidates": {}},
        ),
        TASK_DURABLE_RELATIONSHIPS: (
            "durable_relationships",
            config.durable_relationship_task_settings_path,
            validate_durable_relationships,
            {},
            {"extracted_candidates": {}},
        ),
    }

    contexts: dict[str, _TaskContext] = {}
    for task_name, (run_type, settings_path, validator, base_task_values, default_task_values) in task_specs.items():
        settings = _read_json(settings_path)
        prompt_id = str(settings["prompt_id"])
        prompt_spec = loader.load(prompt_id)
        run_dir = config.output_root / DEBUG_DIR_NAME / "extractions" / task_name
        for directory in ("inputs", "prompts", "raw_outputs", "parsed"):
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
        static_values = {"extraction_policy": _format_policy(settings.get("extraction_policy", []))}
        if task_name == TASK_SCENE_SUMMARY:
            static_values["summary_policy"] = _format_policy(settings.get("summary_policy", []))
        if task_name == TASK_KG_REFINEMENT:
            static_values["refinement_policy"] = _format_policy(settings.get("refinement_policy", []))
        if task_name in {TASK_KG_ENTITIES, TASK_KG_REFINEMENT}:
            static_values["entity_type_policy"] = _format_policy(settings.get("entity_type_policy", []))
        values = {**default_task_values, **base_task_values}
        contexts[task_name] = _TaskContext(
            task_name=task_name,
            run_type=run_type,
            run_dir=run_dir,
            prompt_id=prompt_id,
            prompt_path=prompt_spec.path,
            prompt_spec=prompt_spec,
            static_values=static_values,
            base_task_values=values,
            validate=validator,
        )
    return contexts


def _write_task_run_summary(
    *,
    context: _TaskContext,
    records: list[dict[str, Any]],
    created_at: str,
    config: SceneOrderedPipelineConfig,
    selected: list[NarrativeChunk],
    llm_client: LLMClient | None,
) -> dict[str, Any]:
    trace_path = context.run_dir / "trace.jsonl"
    _write_jsonl(trace_path, records)

    completed_count = sum(1 for record in records if record["status"] in {"completed", "validation_failed", "parse_failed"})
    parsed_count = sum(1 for record in records if record["status"] == "completed")
    failed_count = sum(1 for record in records if record.get("error"))
    summary = {
        "run_type": context.run_type,
        "status": "dry_run_complete" if config.dry_run else "complete",
        "selected_count": len(records),
        "chunk_count": len(records),
        "new_selected_count": len(selected),
        "new_chunk_count": len(selected),
        "rendered_prompt_count": len(records),
        "llm_completed_count": 0 if config.dry_run else completed_count,
        "parsed_output_count": parsed_count,
        "failed_count": failed_count,
        "raw_output_count": 0 if config.dry_run else completed_count,
        "input_char_count": sum(int(record["input_char_count"]) for record in records),
        "prompt_char_count": sum(int(record["prompt_char_count"]) for record in records),
        "trace_path": str(trace_path),
    }
    manifest = {
        "run_type": context.run_type,
        "created_at": created_at,
        "dry_run": config.dry_run,
        "llm": {
            "provider": getattr(llm_client, "provider", None) if llm_client else None,
            "model": getattr(llm_client, "model", None) if llm_client else None,
        },
        "script_path": str(config.script_path.resolve()),
        "prompt_dir": str(config.prompt_dir.resolve()),
        "prompt_id": context.prompt_id,
        "prompt_path": str(context.prompt_path.resolve()),
        "output_dir": str(context.run_dir.resolve()),
        "selection": {
            "start": config.start,
            "limit": config.limit,
            "selected_count": len(selected),
            "chunk_count": len(selected),
            "unit_type": config.unit_type,
            "unit_label": config.unit_label,
            "first_scene_id": selected[0].scene_id if selected else None,
            "last_scene_id": selected[-1].scene_id if selected else None,
            "first_unit_id": selected[0].chunk_id if selected else None,
            "last_unit_id": selected[-1].chunk_id if selected else None,
        },
        "artifact_paths": {
            "inputs_dir": str(context.run_dir / "inputs"),
            "prompts_dir": str(context.run_dir / "prompts"),
            "raw_outputs_dir": str(context.run_dir / "raw_outputs"),
            "parsed_dir": str(context.run_dir / "parsed"),
            "trace_path": str(trace_path),
            "summary_path": str(context.run_dir / "summary.json"),
        },
    }
    _write_text(context.run_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_text(context.run_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _copy_base_run(base_output_root: Path, output_root: Path) -> None:
    base_path = Path(base_output_root)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base output root not found: {base_path}")
    shutil.copytree(
        base_path,
        output_root,
        ignore=shutil.ignore_patterns(
            "scene_context",
            "knowledge_graph",
            "memories",
            "prefix_commits",
            "summaries",
            "summary.json",
            "README.md",
            "canonical_memory",
            "world_model",
            "entity_resolution",
            "staged_memory",
            "kg_entity_memory",
            "episodic_memory",
            "relationship_memory",
        ),
    )


def _load_existing_task_records(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {task: [] for task in ALL_TASKS}
    for task in ALL_TASKS:
        trace_path = output_root / DEBUG_DIR_NAME / "extractions" / task / "trace.jsonl"
        records[task] = _load_jsonl(trace_path)
    return records


def _load_prior_entity_mentions_from_run(output_root: Path) -> list[dict[str, Any]]:
    parsed_dir = _prior_entity_mentions_parsed_dir(output_root)
    mentions: list[dict[str, Any]] = []
    if not parsed_dir.is_dir():
        return mentions
    for parsed_file in sorted(parsed_dir.glob("*.json")):
        payload = _read_json(parsed_file)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if not isinstance(data, dict):
            continue
        sanitized = sanitize_kg_entity_output(data)
        unit_id = str(payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)
        scene_id = str(payload.get("scene_id") or unit_id)
        parent_unit_id = str(payload.get("parent_unit_id") or scene_id)
        for item in sanitized.get("entity_mentions", []) if isinstance(sanitized.get("entity_mentions"), list) else []:
            if not isinstance(item, dict):
                continue
            mentions.append(
                {
                    "unit_id": unit_id,
                    "scene_id": scene_id,
                    "parent_unit_id": parent_unit_id,
                    "surface": item.get("surface", ""),
                    "entity_type": item.get("entity_type", ""),
                    "canonical_hint": item.get("canonical_hint", ""),
                    "description": item.get("description", ""),
                    "evidence": item.get("evidence", ""),
                }
            )
    return mentions


def _build_pipeline_memory(
    output_root: Path,
    task_contexts: dict[str, _TaskContext],
    *,
    author_entity_context: dict[str, Any],
) -> dict[str, Any]:
    debug_intermediate_dir = output_root / DEBUG_DIR_NAME / "intermediate"
    staged_dir = output_root / "scene_context"
    summaries_dir = output_root / "summaries"
    kg_memory_dir = debug_intermediate_dir / "kg_entity_memory"
    episodic_memory_dir = output_root / "memories"
    relationship_memory_dir = debug_intermediate_dir / "relationship_memory"
    canonical_dir = debug_intermediate_dir / "canonical_memory"
    world_model_dir = debug_intermediate_dir / "world_model"
    entity_resolution_dir = output_root / "knowledge_graph"
    prefix_commits_dir = output_root / "prefix_commits"

    inventory = build_scene_inventory_memory(task_contexts[TASK_SCENE_INVENTORY].run_dir, staged_dir)
    scene_summary = build_scene_summary_memory(task_contexts[TASK_SCENE_SUMMARY].run_dir, summaries_dir)
    kg_source_dir = _prepare_kg_memory_source_run_dir(output_root, task_contexts)
    kg = build_kg_entity_memory(kg_source_dir, kg_memory_dir)
    scene_tags_path = kg_memory_dir / "scene_tags.jsonl"
    if scene_tags_path.is_file():
        shutil.copyfile(scene_tags_path, staged_dir / "scene_tags.jsonl")
    episodic = build_episodic_memory(
        task_contexts[TASK_EPISODIC_MEMORIES].run_dir,
        episodic_memory_dir,
        require_entity_candidates=True,
    )
    relationships = build_relationship_memory(
        task_contexts[TASK_DURABLE_RELATIONSHIPS].run_dir,
        relationship_memory_dir,
        require_entity_candidates=True,
    )
    canonical = build_canonical_memory(staged_dir, canonical_dir)
    world_model = build_prefix_world_model(
        canonical_dir=canonical_dir,
        event_memory_dir=None,
        kg_entity_memory_dir=kg_memory_dir,
        episodic_memory_dir=episodic_memory_dir,
        relationship_memory_dir=relationship_memory_dir,
        scene_summary_dir=summaries_dir,
        output_dir=world_model_dir,
    )
    entity_resolution = build_entity_resolution_artifacts(
        world_model_dir,
        entity_resolution_dir,
        author_entities=author_entities_from_context(author_entity_context),
    )
    prefix_commits = build_prefix_commits(world_model_dir, entity_resolution_dir, prefix_commits_dir)
    return {
        "scene_context": inventory,
        "scene_summary": scene_summary,
        "kg_entity_memory": kg,
        "episodic_memory": episodic,
        "relationship_memory": relationships,
        "canonical_memory": canonical,
        "world_model": world_model,
        "entity_resolution": entity_resolution,
        "prefix_commits": prefix_commits,
    }


def _prepare_kg_memory_source_run_dir(output_root: Path, task_contexts: dict[str, _TaskContext]) -> Path:
    """Build a merged KG source: initial inputs for metadata, refined parsed JSON when available."""

    initial_dir = task_contexts[TASK_KG_ENTITIES].run_dir
    refinement_dir = task_contexts[TASK_KG_REFINEMENT].run_dir
    merged_dir = output_root / DEBUG_DIR_NAME / "intermediate" / "kg_entity_source"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    for directory in ("inputs", "parsed"):
        (merged_dir / directory).mkdir(parents=True, exist_ok=True)

    _copy_files(initial_dir / "inputs", merged_dir / "inputs")
    _copy_kg_parsed_files(initial_dir / "parsed", merged_dir / "parsed")
    refinement_parsed_dir = refinement_dir / "parsed"
    if refinement_parsed_dir.is_dir():
        for parsed_file in sorted(refinement_parsed_dir.glob("*.json")):
            payload = _read_json(parsed_file)
            if payload.get("status") == "parsed" and isinstance(payload.get("data"), dict):
                _write_sanitized_kg_parsed_file(parsed_file, merged_dir / "parsed" / parsed_file.name)
    return merged_dir


def _prior_entity_mentions_parsed_dir(output_root: Path) -> Path:
    for parsed_dir in (
        output_root / DEBUG_DIR_NAME / "intermediate" / "kg_entity_source" / "parsed",
        output_root / DEBUG_DIR_NAME / "extractions" / TASK_KG_REFINEMENT / "parsed",
        output_root / DEBUG_DIR_NAME / "extractions" / TASK_KG_ENTITIES / "parsed",
    ):
        if parsed_dir.is_dir() and any(parsed_dir.glob("*.json")):
            return parsed_dir
    return output_root / DEBUG_DIR_NAME / "extractions" / TASK_KG_ENTITIES / "parsed"


def _copy_files(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source_dir.glob("*.json")):
        shutil.copyfile(source_file, target_dir / source_file.name)


def _copy_kg_parsed_files(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source_dir.glob("*.json")):
        _write_sanitized_kg_parsed_file(source_file, target_dir / source_file.name)


def _write_sanitized_kg_parsed_file(source_file: Path, target_file: Path) -> None:
    payload = _read_json(source_file)
    if payload.get("status") == "parsed" and isinstance(payload.get("data"), dict):
        payload = dict(payload)
        payload["data"] = sanitize_kg_entity_output(payload["data"])
    _write_text(target_file, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _public_artifact_paths(output_root: Path) -> dict[str, str]:
    return {
        "readme": str(output_root / "README.md"),
        "summary": str(output_root / "summary.json"),
        "scene_context_dir": str(output_root / "scene_context"),
        "summaries_dir": str(output_root / "summaries"),
        "memories_dir": str(output_root / "memories"),
        "knowledge_graph_dir": str(output_root / "knowledge_graph"),
        "prefix_commits_dir": str(output_root / "prefix_commits"),
        "debug_dir": str(output_root / DEBUG_DIR_NAME),
    }


def _trace_record(ordinal: int, scene: ScriptScene | NarrativeChunk, result: _TaskUnitResult) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "scene_id": scene.scene_id,
        "unit_id": result.unit_id,
        "unit_type": result.unit_type,
        "unit_label": result.unit_label,
        "unit_order": scene.discourse_index,
        "parent_unit_id": result.parent_unit_id,
        "parent_unit_type": result.unit_type,
        "parent_unit_label": result.unit_label,
        "parent_unit_order": scene.discourse_index,
        "chunk_id": result.unit_id,
        "chunk_index": result.chunk_index,
        "chunk_count": result.chunk_count,
        "source_record_id": scene.source_record_id,
        "discourse_index": scene.discourse_index,
        "title": scene.title,
        **result.paths,
        "status": result.status,
        "error": result.error,
        "prompt_char_count": result.prompt_char_count,
        "input_char_count": result.input_char_count,
    }


def _candidate_payload(result: _TaskUnitResult) -> dict[str, Any]:
    data = result.parsed_payload.get("data")
    if result.task_name in {TASK_KG_ENTITIES, TASK_KG_REFINEMENT} and isinstance(data, dict):
        data = sanitize_kg_entity_output(data)
    return {
        "status": result.parsed_payload.get("status"),
        "unit_id": result.unit_id,
        "scene_id": result.scene_id,
        "unit_type": result.unit_type,
        "unit_label": result.unit_label,
        "unit_order": result.unit_order,
        "parent_unit_id": result.parent_unit_id,
        "parent_unit_type": result.unit_type,
        "parent_unit_label": result.unit_label,
        "parent_unit_order": result.unit_order,
        "chunk_index": result.chunk_index,
        "chunk_count": result.chunk_count,
        "data": data,
        "parse_error": result.parsed_payload.get("parse_error"),
        "validation_errors": result.parsed_payload.get("validation_errors", []),
    }


def _chunk_manifest_record(ordinal: int, chunk: NarrativeChunk) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "scene_id": chunk.scene_id,
        "unit_id": chunk.chunk_id,
        "unit_type": chunk.unit_type,
        "unit_label": chunk.unit_label,
        "unit_order": chunk.discourse_index,
        "chunk_id": chunk.chunk_id,
        "parent_unit_id": chunk.parent_unit_id,
        "parent_unit_type": chunk.unit_type,
        "parent_unit_label": chunk.unit_label,
        "parent_unit_order": chunk.discourse_index,
        "chunk_index": chunk.chunk_index,
        "chunk_count": chunk.chunk_count,
        "source_record_id": chunk.source_record_id,
        "discourse_index": chunk.discourse_index,
        "title": chunk.title,
        "source_start": chunk.source_start,
        "source_end": chunk.source_end,
        "source_sha256": chunk.source_sha256,
        "chunk_unit_count": chunk.chunk_unit_count,
        "max_chunk_units": chunk.max_chunk_units,
        "character_count": chunk.character_count,
    }


def _unit_refinement_context(scene: ScriptScene | NarrativeChunk) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "unit_id": getattr(scene, "chunk_id", scene.scene_id),
        "unit_type": getattr(scene, "unit_type", DEFAULT_UNIT_TYPE),
        "unit_label": getattr(scene, "unit_label", DEFAULT_UNIT_LABEL),
        "unit_order": scene.discourse_index,
        "parent_unit_id": getattr(scene, "parent_unit_id", scene.scene_id),
        "parent_unit_type": getattr(scene, "unit_type", DEFAULT_UNIT_TYPE),
        "parent_unit_label": getattr(scene, "unit_label", DEFAULT_UNIT_LABEL),
        "parent_unit_order": scene.discourse_index,
        "chunk_index": getattr(scene, "chunk_index", 1),
        "chunk_count": getattr(scene, "chunk_count", 1),
        "source_record_id": scene.source_record_id,
        "discourse_index": scene.discourse_index,
        "title": scene.title,
        "subtitle": scene.subtitle,
    }


def _append_prior_entity_mentions(prior_entity_mentions: list[dict[str, Any]], result: _TaskUnitResult) -> None:
    data = result.parsed_payload.get("data") if isinstance(result.parsed_payload, dict) else {}
    if not isinstance(data, dict):
        return
    data = sanitize_kg_entity_output(data)
    for item in data.get("entity_mentions", []) if isinstance(data.get("entity_mentions"), list) else []:
        if isinstance(item, dict):
            prior_entity_mentions.append(
                {
                    "unit_id": result.unit_id,
                    "scene_id": result.scene_id,
                    "parent_unit_id": result.parent_unit_id,
                    "surface": item.get("surface", ""),
                    "entity_type": item.get("entity_type", ""),
                    "canonical_hint": item.get("canonical_hint", ""),
                    "description": item.get("description", ""),
                    "evidence": item.get("evidence", ""),
                }
            )


def _format_prior_entity_context(
    author_entities: list[dict[str, Any]],
    prior_entity_mentions: list[dict[str, Any]],
    *,
    limit: int = 80,
) -> str:
    if not author_entities and not prior_entity_mentions:
        return ""
    return json.dumps(
        {
            "author_defined_entities": author_entities,
            "prior_entity_mentions": prior_entity_mentions[-limit:],
            "policy": "author_defined_descriptions_are_baselines; current-unit extraction may only add evidence-supported supplements",
        },
        ensure_ascii=False,
        indent=2,
    )


def _write_run_readme(output_root: Path, summary: dict[str, Any]) -> None:
    memory_summaries = summary.get("memory_summaries") if isinstance(summary.get("memory_summaries"), dict) else {}
    scene_context = memory_summaries.get("scene_context", {})
    scene_summary = memory_summaries.get("scene_summary", {})
    memories = memory_summaries.get("episodic_memory", {})
    relationships = memory_summaries.get("relationship_memory", {})
    graph = memory_summaries.get("entity_resolution", {})
    world = memory_summaries.get("world_model", {})
    task_summaries = summary.get("task_summaries") if isinstance(summary.get("task_summaries"), dict) else {}

    lines = [
        "# DMS Run",
        "",
        "This compact run directory keeps only the agreed working artifacts at the top level.",
        "",
        "## Read These First",
        "",
        "- `knowledge_graph/entities.jsonl`: resolved KG entities and aliases",
        "- `knowledge_graph/relationships.jsonl`: durable relationship states",
        "- `prefix_commits/current_snapshot.json`: materialized state after the latest committed scene",
        "- `prefix_commits/commits.jsonl`: one commit record per parent scene",
        "- `prefix_commits/operations.jsonl`: append-only state-change log",
        "- `summaries/scene_summaries.jsonl`: compact recap and semantic retrieval text per processing unit",
        "- `memories/episodic_memories.jsonl`: ordered source-grounded memories",
        "- `memories/entity_memory_links.jsonl`: entity-to-memory links",
        "- `memories/evidence_rejections.jsonl`: rejected evidence audit",
        "- `scene_context/stated_facts.jsonl`: stated facts from the unit",
        "- `scene_context/open_questions.jsonl`: open questions from the unit",
        "- `scene_context/scene_tags.jsonl`: local low-value scene tags excluded from KG/pre-context",
        "",
        "## Debug Data",
        "",
        "Prompts, raw LLM outputs, parsed extraction JSON, canonical memory, KG mention memory, and world-model merge artifacts are under `_debug/`.",
        "",
        "## Counts",
        "",
        f"- selected scenes: {summary.get('selected_count', 0)}",
        f"- processing chunks: {summary.get('chunk_count', 0)}",
        f"- max chunk units: {summary.get('max_chunk_units', 0)}",
        f"- scene task concurrency: {summary.get('scene_task_concurrency', 0)}",
        f"- scene summaries: {_count(scene_summary, 'scene_summary_count')}",
        f"- stated facts: {_count(scene_context, 'stated_fact_count')}",
        f"- open questions: {_count(scene_context, 'open_question_count')}",
        f"- scene tags: {_count(world, 'scene_tag_count')}",
        f"- resolved entities: {_count(graph, 'entity_count')}",
        f"- durable relationships: {_count(graph, 'relationship_count')}",
        f"- episodic memories: {_count(memories, 'episodic_memory_count')}",
        f"- entity-memory links: {_count(memories, 'entity_memory_link_count')}",
        f"- relationship observations: {_count(relationships, 'relationship_observation_count')}",
        f"- KG mentions: {_count(world, 'kg_entity_mention_count')}",
        "",
        "## Evidence",
        "",
        f"- exact: {_evidence_count(memories, 'exact')}",
        f"- fuzzy-aligned: {_evidence_count(memories, 'fuzzy_aligned')}",
        f"- rejected: {_evidence_count(memories, 'rejected')}",
        f"- skipped deictic links: {_count(memories, 'skipped_deictic_entity_memory_link_count')}",
        "",
        "## Task Status",
        "",
    ]
    for task in ALL_TASKS:
        task_summary = task_summaries.get(task, {})
        lines.append(
            f"- {task}: parsed {_count(task_summary, 'parsed_output_count')}, failed {_count(task_summary, 'failed_count')}"
        )
    lines.append("")
    _write_text(output_root / "README.md", "\n".join(lines))


def _count(payload: Any, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _evidence_count(payload: Any, key: str) -> int:
    counts = payload.get("evidence_verification_counts", {}) if isinstance(payload, dict) else {}
    return _count(counts, key)


def _wrapper(scene: ScriptScene | NarrativeChunk, **values: Any) -> dict[str, Any]:
    unit_id = getattr(scene, "chunk_id", scene.scene_id)
    return {
        "scene_id": scene.scene_id,
        "unit_id": unit_id,
        "unit_type": getattr(scene, "unit_type", DEFAULT_UNIT_TYPE),
        "unit_label": getattr(scene, "unit_label", DEFAULT_UNIT_LABEL),
        "unit_order": scene.discourse_index,
        "parent_unit_id": getattr(scene, "parent_unit_id", scene.scene_id),
        "parent_unit_type": getattr(scene, "unit_type", DEFAULT_UNIT_TYPE),
        "parent_unit_label": getattr(scene, "unit_label", DEFAULT_UNIT_LABEL),
        "parent_unit_order": scene.discourse_index,
        "chunk_id": unit_id,
        "chunk_index": getattr(scene, "chunk_index", 1),
        "chunk_count": getattr(scene, "chunk_count", 1),
        **values,
    }


def _select_scenes(scenes: list[ScriptScene], *, start: int, limit: int) -> list[ScriptScene]:
    offset = start - 1
    return scenes[offset : offset + limit]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_prior_context(path: Path | None, *, label: str) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"{label} path not found: {path}")
    if path.is_dir():
        for name in ("entities.jsonl", "canonical_memory.json", "prefix_world_model.json", "summary.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        return "\n".join(sorted(item.name for item in path.iterdir()))
    return path.read_text(encoding="utf-8")


def _format_policy(policy: Any) -> str:
    if isinstance(policy, list):
        return "\n".join(f"- {item}" for item in policy)
    return str(policy or "")


def _serializable_config(config: SceneOrderedPipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
