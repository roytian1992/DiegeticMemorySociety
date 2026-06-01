from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.config import build_openai_client_from_config, embedding_kwargs_from_config, load_local_config, redact_model_config
from dms.evaluation import WritingEvaluationConfig, build_scene_eligibility_splits, evaluate_writing
from dms.intent_levels import normalize_intent_level
from dms.llm import LLMClient, LLMResult
from dms.parsing import extract_json_value
from dms.progress import print_progress
from dms.prompts import YAMLPromptLoader
from dms.retrieval import MemoryPacketConfig, build_memory_packet, format_memory_packet_markdown
from dms.runners import SceneOrderedPipelineConfig, run_scene_ordered_pipeline
from dms.runners.scene_ordered_pipeline import ALL_TASKS
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes
from dms.simulation import AttributeCardConfig, SocialSimulationConfig, build_entity_attribute_cards, run_social_simulation
from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
from dms.writing import (
    SocialWritingGenerationConfig,
    format_previous_scene_context,
    generate_writing_with_social_simulation_client,
)


@dataclass(frozen=True)
class WritingBenchmarkPrepareConfig:
    script_path: Path
    output_dir: Path
    model_config_path: Path = Path("configs/local_config.yaml")
    llm_section: str = "llm"
    embedding_section: str = "embedding"
    prompt_dir: Path = Path("task_specs/prompts")
    ordered_run_dir: Path | None = None
    extraction_output_root: Path | None = None
    run_extraction: bool = False
    start: int = 1
    limit: int | None = None
    scene_task_concurrency: int = 3
    max_chunk_units: int = 800
    db_path: Path | None = None
    chroma_dir: Path | None = None
    collection_name: str = "dms_retrieval_documents"
    chroma_upsert_batch_size: int = 1000
    dry_run: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class WritingBenchmarkRunConfig:
    script_path: Path
    db_path: Path
    chroma_dir: Path
    output_dir: Path
    model_config_path: Path = Path("configs/local_config.yaml")
    llm_section: str = "llm"
    writing_llm_section: str = "writing_llm"
    embedding_section: str = "embedding"
    prompt_dir: Path = Path("task_specs/prompts")
    eligibility_dir: Path | None = None
    eligibility_targets_file: Path | None = None
    target_scene_ids: tuple[str, ...] = ()
    start_scene_order: int | None = None
    limit: int | None = 3
    dry_run: bool = False
    overwrite: bool = False
    stop_on_error: bool = False
    intent_only: bool = False
    collection_name: str = "dms_retrieval_documents"
    memory_intent_level: str = "writing_intent"
    social_simulation_intent_level: str = "social_simulation_intent"
    generation_intent_level: str = "writing_intent"
    evaluation_intent_level: str = "writing_spec"
    scene_top_k: int = 5
    entity_memory_top_k: int = 12
    max_entity_memories_before_vector: int = 50
    entity_match_limit: int = 1
    attribute_entity_types: tuple[str, ...] = ("character",)
    attribute_entity_names: tuple[str, ...] = ()
    max_memories_per_entity: int = 16
    previous_scene_context_mode: str = "previous_scene"
    previous_scene_context_max_chars: int = 800
    style_reference_mode: str = "none"
    length_margin: float = 0.2
    length_requirement: str = ""
    output_requirements: str = (
        "- Chinese output\n"
        "- Output final narrative prose only\n"
        "- Do not include analysis, headings, bullet points, or reference IDs\n"
        "- Do not leak M/R memory or reference IDs\n"
        "- Do not turn social-simulation assumptions into facts when memory does not support them"
    )


def prepare_writing_benchmark_assets(config: WritingBenchmarkPrepareConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    if output_dir.exists() and config.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_script_scenes(config.script_path)
    extraction_root = config.extraction_output_root or output_dir / "ordered_run"
    ordered_run_dir = config.ordered_run_dir or extraction_root
    eligibility_dir = output_dir / "eligibility"
    eligibility_summary = build_scene_eligibility_splits(config.script_path, eligibility_dir)
    limit = config.limit if config.limit is not None else max(len(scenes) - config.start + 1, 0)

    model_config = load_local_config(config.model_config_path) if config.model_config_path.is_file() else {}
    extraction_summary = None
    asset_summary = None
    chroma_summary = None

    if config.run_extraction and not config.dry_run:
        llm_client = build_openai_client_from_config(model_config, config.llm_section)
        llm_clients = {task: llm_client for task in ALL_TASKS}
        extraction_summary = run_scene_ordered_pipeline(
            SceneOrderedPipelineConfig(
                script_path=config.script_path,
                output_root=extraction_root,
                prompt_dir=config.prompt_dir,
                start=config.start,
                limit=limit,
                dry_run=False,
                overwrite=config.overwrite,
                scene_task_concurrency=config.scene_task_concurrency,
                max_chunk_units=config.max_chunk_units,
            ),
            llm_clients=llm_clients,
        )
        ordered_run_dir = extraction_root

    if config.db_path and not config.dry_run:
        asset_summary = import_run_assets(
            AssetStoreImportConfig(
                db_path=config.db_path,
                ordered_run_dir=ordered_run_dir,
                reset=config.overwrite,
            )
        )

    if config.db_path and config.chroma_dir and not config.dry_run:
        embedding_kwargs = embedding_kwargs_from_config(model_config, config.embedding_section)
        chroma_summary = build_chroma_memory_index(
            ChromaMemoryIndexConfig(
                db_path=config.db_path,
                persist_dir=config.chroma_dir,
                collection_name=config.collection_name,
                reset=config.overwrite,
                upsert_batch_size=config.chroma_upsert_batch_size,
                **embedding_kwargs,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "dry_run_complete" if config.dry_run else "complete",
        "script_path": str(config.script_path),
        "scene_count": len(scenes),
        "selection": {
            "start": config.start,
            "limit": limit,
            "run_extraction": config.run_extraction,
        },
        "paths": {
            "output_dir": str(output_dir),
            "eligibility_dir": str(eligibility_dir),
            "ordered_run_dir": str(ordered_run_dir),
            "db_path": str(config.db_path) if config.db_path else None,
            "chroma_dir": str(config.chroma_dir) if config.chroma_dir else None,
            "summary": str(output_dir / "summary.json"),
        },
        "model_config_path": str(config.model_config_path),
        "model_config": {
            "llm": redact_model_config(model_config.get(config.llm_section, {})) if model_config else {},
            "embedding": redact_model_config(model_config.get(config.embedding_section, {})) if model_config else {},
        },
        "eligibility": eligibility_summary,
        "extraction": extraction_summary,
        "asset_store": asset_summary,
        "chroma": chroma_summary,
        "next_command_hint": (
            "python -m dms.cli run-writing-benchmark "
            f"{config.script_path} --db-path {config.db_path or '<db_path>'} "
            f"--chroma-dir {config.chroma_dir or '<chroma_dir>'} --eligibility-dir {eligibility_dir}"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_writing_benchmark(
    config: WritingBenchmarkRunConfig,
    *,
    llm_client: LLMClient | None = None,
    writing_llm_client: LLMClient | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    if output_dir.exists() and config.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_script_scenes(config.script_path)
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    targets = _select_benchmark_targets(config, scenes, output_dir)
    _write_jsonl(output_dir / "target_manifest.jsonl", targets)

    if config.dry_run:
        summary = _benchmark_summary(
            config,
            status="dry_run_complete",
            scenes=scenes,
            targets=targets,
            target_results=[],
            failures=[],
            model_config={},
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    model_config = model_config if model_config is not None else load_local_config(config.model_config_path)
    llm_client = llm_client if llm_client is not None else build_openai_client_from_config(model_config, config.llm_section)
    writing_llm_client = (
        writing_llm_client
        if writing_llm_client is not None
        else build_openai_client_from_config(model_config, config.writing_llm_section)
    )
    embedding_kwargs = embedding_kwargs_from_config(model_config, config.embedding_section)

    target_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print_progress(
        "writing_benchmark:start",
        0,
        len(targets),
        detail=f"output_dir={output_dir} writing_llm_section={config.writing_llm_section}",
    )
    for index, target in enumerate(targets, start=1):
        scene = scene_by_id[str(target["scene_id"])]
        target_dir = output_dir / "targets" / scene.scene_id
        try:
            result = _run_one_writing_target(
                config,
                scene,
                scenes=scenes,
                output_dir=target_dir,
                llm_client=llm_client,
                writing_llm_client=writing_llm_client,
                model_config=model_config,
                embedding_kwargs=embedding_kwargs,
            )
            target_results.append(result)
            _write_jsonl(output_dir / "metrics.jsonl", target_results)
            print_progress(
                "writing_benchmark:target",
                index,
                len(targets),
                detail=(
                    f"status=complete scene={scene.scene_id} completed={len(target_results)} "
                    f"failures={len(failures)}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should record per-target failures.
            failure = {
                "scene_id": scene.scene_id,
                "title": scene.title,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "target_dir": str(target_dir),
            }
            failures.append(failure)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "error.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print_progress(
                "writing_benchmark:target",
                index,
                len(targets),
                detail=(
                    f"status=failed scene={scene.scene_id} completed={len(target_results)} "
                    f"failures={len(failures)} error_type={type(exc).__name__}"
                ),
            )
            if config.stop_on_error:
                raise

    summary = _benchmark_summary(
        config,
        status="complete" if not failures else "completed_with_failures",
        scenes=scenes,
        targets=targets,
        target_results=target_results,
        failures=failures,
        model_config=model_config,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_one_writing_target(
    config: WritingBenchmarkRunConfig,
    scene: ScriptScene,
    *,
    scenes: list[ScriptScene],
    output_dir: Path,
    llm_client: LLMClient,
    writing_llm_client: LLMClient,
    model_config: dict[str, Any],
    embedding_kwargs: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    intents_dir = output_dir / "intent"
    print_progress("writing_target:stage", 0, 8, detail=f"scene={scene.scene_id} stage=start")
    social_simulation_intent = _extract_intent(
        scene,
        level="social_simulation_intent",
        prompt_id="dms/social_simulation_intent",
        task_settings_path=Path("task_specs/task_settings/social_simulation_intent_task.json"),
        output_dir=intents_dir / "social_simulation_intent",
        prompt_dir=config.prompt_dir,
        llm_client=llm_client,
        output_key="social_simulation_intent",
    )
    print_progress("writing_target:stage", 1, 8, detail=f"scene={scene.scene_id} stage=social_simulation_intent")
    writing_intent = _extract_intent(
        scene,
        level="writing_intent",
        prompt_id="dms/writing_intent",
        task_settings_path=Path("task_specs/task_settings/writing_intent_task.json"),
        output_dir=intents_dir / "writing_intent",
        prompt_dir=config.prompt_dir,
        llm_client=llm_client,
        output_key="writing_intent",
    )
    print_progress("writing_target:stage", 2, 8, detail=f"scene={scene.scene_id} stage=writing_intent")
    writing_spec = _extract_intent(
        scene,
        level="writing_spec",
        prompt_id="dms/writing_spec",
        task_settings_path=Path("task_specs/task_settings/writing_spec_task.json"),
        output_dir=intents_dir / "writing_spec",
        prompt_dir=config.prompt_dir,
        llm_client=llm_client,
        output_key="writing_spec",
        output_aliases=("reference_scene_spec",),
    )
    print_progress("writing_target:stage", 3, 8, detail=f"scene={scene.scene_id} stage=writing_spec")
    intents = {
        "social_simulation_intent": social_simulation_intent,
        "writing_intent": writing_intent,
        "writing_spec": writing_spec,
    }

    target_summary: dict[str, Any] = {
        "scene_id": scene.scene_id,
        "source_record_id": scene.source_record_id,
        "title": scene.title,
        "status": "intent_only" if config.intent_only else "complete",
        "intents": {
            "social_simulation_intent": social_simulation_intent.get("social_simulation_intent"),
            "writing_intent": writing_intent.get("writing_intent"),
            "writing_spec": writing_spec.get("writing_spec"),
            "writing_spec_text": writing_spec.get("intent_text"),
        },
        "legacy_intent_aliases": {
            "reference_scene_spec": "writing_spec",
            "sparse": "social_simulation_intent",
            "detailed": "writing_spec",
        },
        "paths": {
            "target_dir": str(output_dir),
            "social_simulation_intent": str(intents_dir / "social_simulation_intent" / "summary.json"),
            "writing_intent": str(intents_dir / "writing_intent" / "summary.json"),
            "writing_spec": str(intents_dir / "writing_spec" / "summary.json"),
        },
    }
    if config.intent_only:
        (output_dir / "summary.json").write_text(json.dumps(target_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target_summary

    memory_intent = _intent_text(intents, config.memory_intent_level)
    social_intent = _intent_text(intents, config.social_simulation_intent_level)
    generation_intent = _intent_text(intents, config.generation_intent_level)
    evaluation_intent = _intent_text(intents, config.evaluation_intent_level)

    memory_packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=config.db_path,
            chroma_dir=config.chroma_dir,
            writing_intent=memory_intent,
            before_scene_id=scene.scene_id,
            scene_top_k=config.scene_top_k,
            entity_memory_top_k=config.entity_memory_top_k,
            max_entity_memories_before_vector=config.max_entity_memories_before_vector,
            entity_match_limit=config.entity_match_limit,
            collection_name=config.collection_name,
            **embedding_kwargs,
        )
    )
    memory_json = output_dir / "memory_packet.json"
    memory_md = output_dir / "memory_packet.md"
    memory_json.write_text(json.dumps(memory_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    memory_md.write_text(format_memory_packet_markdown(memory_packet), encoding="utf-8")
    print_progress(
        "writing_target:stage",
        4,
        8,
        detail=(
            f"scene={scene.scene_id} stage=memory_packet entities={len(memory_packet.get('entities') or [])} "
            f"memories={len(memory_packet.get('episodic_memories') or [])}"
        ),
    )

    attribute_cards_dir = output_dir / "attribute_cards"
    attribute_summary = build_entity_attribute_cards(
        AttributeCardConfig(
            memory_packet_path=memory_json,
            output_dir=attribute_cards_dir,
            prompt_dir=config.prompt_dir,
            entity_types=config.attribute_entity_types,
            entity_names=config.attribute_entity_names,
            max_memories_per_entity=config.max_memories_per_entity,
            overwrite=True,
        ),
        llm_client=llm_client,
    )
    print_progress(
        "writing_target:stage",
        5,
        8,
        detail=f"scene={scene.scene_id} stage=attribute_cards cards={attribute_summary.get('card_count')}",
    )

    social_dir = output_dir / "social_simulation"
    social_summary = run_social_simulation(
        SocialSimulationConfig(
            attribute_cards_path=attribute_cards_dir / "attribute_cards.json",
            social_simulation_intent=social_intent,
            output_dir=social_dir,
            prompt_dir=config.prompt_dir,
            overwrite=True,
        ),
        llm_client=llm_client,
    )
    print_progress(
        "writing_target:stage",
        6,
        8,
        detail=f"scene={scene.scene_id} stage=social_simulation characters={social_summary.get('character_simulation_count')}",
    )

    previous_scene = _previous_scene(scene, scenes) if config.previous_scene_context_mode == "previous_scene" else None
    previous_scene_context = _benchmark_previous_scene_context(
        previous_scene,
        memory_packet=memory_packet,
        max_chars=config.previous_scene_context_max_chars,
    )
    writing_dir = output_dir / "writing"
    writing_summary = generate_writing_with_social_simulation_client(
        SocialWritingGenerationConfig(
            writing_request=f"写一段新的叙事内容：{generation_intent}",
            memory_packet_path=memory_md,
            attribute_cards_path=attribute_cards_dir / "attribute_cards.md",
            social_simulation_path=social_dir / "writer_packet.md",
            output_dir=writing_dir,
            model_config_path=config.model_config_path,
            model_section=config.writing_llm_section,
            prompt_dir=config.prompt_dir,
            previous_scene_context=previous_scene_context,
            previous_scene_context_max_chars=config.previous_scene_context_max_chars,
            style_reference_script=config.script_path if config.style_reference_mode == "previous_scene" else None,
            style_reference_scene_id=_previous_scene_id(scene, scenes) if config.style_reference_mode == "previous_scene" else None,
            length_requirement=config.length_requirement or _length_requirement(scene, margin=config.length_margin),
            output_requirements=config.output_requirements,
            overwrite=True,
        ),
        llm_client=writing_llm_client,
        model_config=model_config,
    )
    print_progress(
        "writing_target:stage",
        7,
        8,
        detail=f"scene={scene.scene_id} stage=writing chars={writing_summary.get('output', {}).get('body_chars')}",
    )

    evaluation_dir = output_dir / "evaluation"
    evaluation_summary = evaluate_writing(
        WritingEvaluationConfig(
            writing_intent=evaluation_intent,
            generated_text=(writing_dir / "draft.md").read_text(encoding="utf-8").strip(),
            memory_packet=memory_md.read_text(encoding="utf-8"),
            reference_text=scene.content.strip(),
            output_dir=evaluation_dir,
            prompt_dir=config.prompt_dir,
            overwrite=True,
        ),
        llm_client=llm_client,
    )
    print_progress("writing_target:stage", 8, 8, detail=f"scene={scene.scene_id} stage=evaluation")

    target_summary.update(
        {
            "status": "complete",
            "intent_levels": {
                "memory": normalize_intent_level(config.memory_intent_level),
                "social_simulation": normalize_intent_level(config.social_simulation_intent_level),
                "generation": normalize_intent_level(config.generation_intent_level),
                "evaluation": normalize_intent_level(config.evaluation_intent_level),
            },
            "counts": {
                "retrieved_entities": len(memory_packet.get("entities") or []),
                "retrieved_memories": len(memory_packet.get("episodic_memories") or []),
                "retrieved_relations": len(memory_packet.get("relations") or []),
                "related_scene_summaries": len(memory_packet.get("related_scene_summaries") or []),
                "attribute_cards": attribute_summary.get("card_count"),
                "character_simulations": social_summary.get("character_simulation_count"),
            },
            "metrics": _extract_metrics(evaluation_summary),
            "paths": {
                **target_summary["paths"],
                "memory_packet_json": str(memory_json),
                "memory_packet_markdown": str(memory_md),
                "attribute_cards": str(attribute_cards_dir / "attribute_cards.md"),
                "social_simulation": str(social_dir / "writer_packet.md"),
                "draft": str(writing_dir / "draft.md"),
                "evaluation": str(evaluation_dir / "summary.json"),
                "summary": str(output_dir / "summary.json"),
                "previous_scene_context": str(writing_dir / "previous_scene_context.md")
                if previous_scene_context
                else None,
            },
            "writing": {
                "draft_chars": writing_summary.get("output", {}).get("body_chars"),
                "draft_non_ws_chars": writing_summary.get("output", {}).get("body_non_ws_chars"),
                "ref_ids_present": writing_summary.get("output", {}).get("ref_ids_present"),
                "request_anchors": writing_summary.get("output", {}).get("request_anchors"),
                "missing_request_anchors": writing_summary.get("output", {}).get("missing_request_anchors"),
                "writer_packet_artifact_terms_present": writing_summary.get("output", {}).get(
                    "writer_packet_artifact_terms_present"
                ),
                "dialogue_risk_phrases_present": writing_summary.get("output", {}).get(
                    "dialogue_risk_phrases_present"
                ),
                "previous_scene_context_chars": len(previous_scene_context),
                "previous_scene_context_source_scene_id": previous_scene.scene_id if previous_scene else None,
            },
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(target_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target_summary


def _extract_intent(
    scene: ScriptScene,
    *,
    level: str,
    prompt_id: str,
    task_settings_path: Path,
    output_dir: Path,
    prompt_dir: Path,
    llm_client: LLMClient,
    output_key: str = "writing_intent",
    output_aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads(task_settings_path.read_text(encoding="utf-8"))
    intent_policy = "\n".join(f"- {item}" for item in settings.get("intent_policy", []))
    unit_json = scene.to_dict()
    prompt = YAMLPromptLoader(prompt_dir).render(
        prompt_id,
        task_values={
            "unit_json": json.dumps(unit_json, ensure_ascii=False, indent=2),
            "intent_policy": intent_policy,
        },
    )
    (output_dir / "input.json").write_text(json.dumps(unit_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    result = llm_client.complete(prompt)
    (output_dir / "raw_response.json").write_text(
        json.dumps(_llm_result_to_dict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    parsed = extract_json_value(result.text)
    parsed_payload = {
        "level": level,
        "status": "parsed" if parsed.ok else "parse_failed",
        "data": parsed.data,
        "parse_error": parsed.error,
    }
    (output_dir / "parsed.json").write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not parsed.ok:
        raise ValueError(f"Failed to parse {level} writing intent for {scene.scene_id}: {parsed.error}")
    data = parsed.data if isinstance(parsed.data, dict) else {}
    artifact = _first_present(data, (output_key, *output_aliases))
    intent_text = _intent_artifact_text(artifact)
    natural_text = _intent_artifact_natural_language_text(artifact)
    source_non_ws_chars = len(re.sub(r"\s+", "", scene.content or ""))
    artifact_non_ws_chars = len(re.sub(r"\s+", "", natural_text))
    formatted_text_non_ws_chars = len(re.sub(r"\s+", "", intent_text))
    if source_non_ws_chars and artifact_non_ws_chars > source_non_ws_chars:
        raise ValueError(
            f"{level} for {scene.scene_id} is longer than source text: "
            f"{artifact_non_ws_chars} > {source_non_ws_chars}"
        )
    summary = {
        "level": level,
        "scene_id": scene.scene_id,
        "title": scene.title,
        output_key: artifact,
        "intent_text": intent_text,
        "length_check": {
            "source_non_ws_chars": source_non_ws_chars,
            "artifact_non_ws_chars": artifact_non_ws_chars,
            "formatted_text_non_ws_chars": formatted_text_non_ws_chars,
            "artifact_to_source_ratio": round(artifact_non_ws_chars / source_non_ws_chars, 4)
            if source_non_ws_chars
            else None,
            "must_not_exceed_source": True,
        },
        "llm": {"provider": llm_client.provider, "model": llm_client.model},
        "usage": result.usage,
        "paths": {
            "input": str(output_dir / "input.json"),
            "prompt": str(output_dir / "prompt.txt"),
            "raw_response": str(output_dir / "raw_response.json"),
            "parsed": str(output_dir / "parsed.json"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"{output_key}.txt").write_text(intent_text + "\n", encoding="utf-8")
    return summary


def _select_benchmark_targets(
    config: WritingBenchmarkRunConfig,
    scenes: list[ScriptScene],
    output_dir: Path,
) -> list[dict[str, Any]]:
    if config.eligibility_targets_file:
        records = _read_jsonl(config.eligibility_targets_file)
    else:
        eligibility_dir = config.eligibility_dir or output_dir / "eligibility"
        targets_path = eligibility_dir / "writing_eval_targets.jsonl"
        if not targets_path.is_file():
            build_scene_eligibility_splits(config.script_path, eligibility_dir)
        records = _read_jsonl(targets_path)

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    allowed_ids = set(config.target_scene_ids)
    selected: list[dict[str, Any]] = []
    for record in records:
        scene_id = str(record.get("scene_id") or "")
        scene = scene_by_id.get(scene_id)
        if scene is None:
            continue
        if allowed_ids and scene_id not in allowed_ids:
            continue
        if config.start_scene_order is not None and scene.source_record_id < config.start_scene_order:
            continue
        selected.append({**record, "content_char_count": len(scene.content or "")})
    selected.sort(key=lambda item: int(item.get("source_record_id") or 0))
    if config.limit is not None:
        selected = selected[: max(config.limit, 0)]
    return selected


def _benchmark_summary(
    config: WritingBenchmarkRunConfig,
    *,
    status: str,
    scenes: list[ScriptScene],
    targets: list[dict[str, Any]],
    target_results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    aggregate = _aggregate_metrics(target_results)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "script_path": str(config.script_path),
        "scene_count": len(scenes),
        "target_count": len(targets),
        "completed_count": len(target_results),
        "failure_count": len(failures),
        "config": _serializable_config(config),
        "model_config": {
            "llm": redact_model_config(model_config.get(config.llm_section, {})) if model_config else {},
            "embedding": redact_model_config(model_config.get(config.embedding_section, {})) if model_config else {},
            "writing_llm": redact_model_config(model_config.get(config.writing_llm_section, {})) if model_config else {},
        },
        "aggregate_metrics": aggregate,
        "failures": failures,
        "artifacts": {
            "summary": str(Path(config.output_dir) / "summary.json"),
            "target_manifest": str(Path(config.output_dir) / "target_manifest.jsonl"),
            "metrics": str(Path(config.output_dir) / "metrics.jsonl"),
            "targets_dir": str(Path(config.output_dir) / "targets"),
        },
    }


def _extract_metrics(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    generated = evaluation_summary.get("candidates", {}).get("generated", {})
    reference = evaluation_summary.get("candidates", {}).get("reference", {})
    return {
        "generated": _candidate_scores(generated),
        "reference": _candidate_scores(reference),
        "deltas": evaluation_summary.get("deltas") or {},
    }


def _candidate_scores(candidate: dict[str, Any]) -> dict[str, float | None]:
    if not isinstance(candidate, dict):
        return {}
    return {
        "writing_intent_consistency": _nested_score(candidate, "writing_intent_consistency"),
        "writing_quality": _nested_score(candidate, "writing_quality"),
        "memory_faithfulness": _nested_score(candidate, "memory_faithfulness"),
        "overall": candidate.get("overall") if isinstance(candidate.get("overall"), (int, float)) else None,
    }


def _nested_score(candidate: dict[str, Any], key: str) -> float | None:
    value = candidate.get(key)
    score = value.get("score") if isinstance(value, dict) else None
    return float(score) if isinstance(score, (int, float)) else None


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = [
        ("generated", "writing_intent_consistency"),
        ("generated", "writing_quality"),
        ("generated", "memory_faithfulness"),
        ("generated", "overall"),
        ("reference", "writing_intent_consistency"),
        ("reference", "writing_quality"),
        ("reference", "memory_faithfulness"),
        ("reference", "overall"),
    ]
    aggregate: dict[str, Any] = {}
    for label, metric in metric_paths:
        values = [
            result.get("metrics", {}).get(label, {}).get(metric)
            for result in results
            if isinstance(result.get("metrics", {}).get(label, {}).get(metric), (int, float))
        ]
        if values:
            aggregate[f"{label}_{metric}_mean"] = round(sum(float(value) for value in values) / len(values), 4)
    for metric in ("writing_intent_consistency", "writing_quality", "memory_faithfulness", "overall"):
        values = [
            result.get("metrics", {}).get("deltas", {}).get(metric)
            for result in results
            if isinstance(result.get("metrics", {}).get("deltas", {}).get(metric), (int, float))
        ]
        if values:
            aggregate[f"delta_{metric}_mean"] = round(sum(float(value) for value in values) / len(values), 4)
    return aggregate


def _intent_text(intents: dict[str, dict[str, Any]], level: str) -> str:
    key = normalize_intent_level(level)
    if key not in intents:
        raise ValueError(f"Unknown intent level: {level}")
    return str(
        intents[key].get("intent_text")
        or intents[key].get("writing_intent")
        or intents[key].get("writing_spec")
        or ""
    ).strip()


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _intent_artifact_text(artifact: Any) -> str:
    if artifact is None:
        return ""
    if isinstance(artifact, str):
        return artifact.strip()
    if isinstance(artifact, dict):
        lines: list[str] = []
        purpose = str(artifact.get("scene_purpose") or "").strip()
        if purpose:
            lines.append(f"Scene purpose: {purpose}")
        for key, label in (
            ("required_entities", "Required entities"),
            ("required_narrative_units", "Required narrative units"),
            ("required_state_or_relationship", "Required state or relationship"),
            ("style_or_form_constraints", "Style or form constraints"),
        ):
            values = artifact.get(key)
            if isinstance(values, list) and values:
                cleaned = [str(value).strip() for value in values if str(value).strip()]
                if cleaned:
                    lines.append(f"{label}: " + "; ".join(cleaned))
        if lines:
            return "\n".join(lines).strip()
    return json.dumps(artifact, ensure_ascii=False, indent=2).strip()


def _intent_artifact_natural_language_text(artifact: Any) -> str:
    if artifact is None:
        return ""
    if isinstance(artifact, str):
        return artifact.strip()
    if isinstance(artifact, dict):
        parts: list[str] = []
        purpose = str(artifact.get("scene_purpose") or "").strip()
        if purpose:
            parts.append(purpose)
        for key in (
            "required_entities",
            "required_narrative_units",
            "required_state_or_relationship",
            "style_or_form_constraints",
        ):
            values = artifact.get(key)
            if isinstance(values, list):
                parts.extend(str(value).strip() for value in values if str(value).strip())
        if parts:
            return "\n".join(parts).strip()
    return json.dumps(artifact, ensure_ascii=False, indent=2).strip()


def _previous_scene_id(scene: ScriptScene, scenes: list[ScriptScene]) -> str | None:
    previous = _previous_scene(scene, scenes)
    return previous.scene_id if previous else None


def _previous_scene(scene: ScriptScene, scenes: list[ScriptScene]) -> ScriptScene | None:
    previous = None
    for candidate in scenes:
        if candidate.scene_id == scene.scene_id:
            return previous
        previous = candidate
    return None


def _benchmark_previous_scene_context(
    scene: ScriptScene | None,
    *,
    memory_packet: dict[str, Any],
    max_chars: int,
) -> str:
    if scene is None:
        return ""
    return format_previous_scene_context(
        scene,
        max_chars=max_chars,
        summary=_related_scene_summary_text(memory_packet, scene.scene_id),
        entities=_memory_packet_entity_names(memory_packet),
    )


def _related_scene_summary_text(memory_packet: dict[str, Any], scene_id: str) -> str:
    for summary in memory_packet.get("related_scene_summaries") or []:
        if str(summary.get("scene_id") or "") == scene_id:
            return str(summary.get("summary") or "").strip()
    return ""


def _memory_packet_entity_names(memory_packet: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entity in memory_packet.get("entities") or []:
        name = str(entity.get("canonical_name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _length_requirement(scene: ScriptScene, *, margin: float) -> str:
    non_ws = len(re.sub(r"\s+", "", scene.content or ""))
    lower = max(20, int(non_ws * (1.0 - max(margin, 0.0))))
    upper = max(lower + 1, int(non_ws * (1.0 + max(margin, 0.0))))
    return f"正文必须为{lower}-{upper}个中文字符（不含空白）。"


def _llm_result_to_dict(result: LLMResult) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "raw_response": result.raw_response,
        "usage": result.usage,
    }


def _serializable_config(config: WritingBenchmarkRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
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
