from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.config import build_openai_client_from_config, embedding_kwargs_from_config, load_local_config, redact_model_config
from dms.evaluation import WritingEvaluationConfig, evaluate_writing
from dms.retrieval import MemoryPacketConfig, build_memory_packet, format_memory_packet_markdown
from dms.scripts.wandering_earth import load_script_scenes
from dms.simulation import AttributeCardConfig, SocialSimulationConfig, build_entity_attribute_cards, run_social_simulation
from dms.writing import SocialWritingGenerationConfig, generate_writing_with_social_simulation_client


@dataclass(frozen=True)
class WritingE2EConfig:
    db_path: Path
    chroma_dir: Path
    writing_intent: str
    output_dir: Path
    model_config_path: Path = Path("configs/local_config.yaml")
    llm_section: str = "llm"
    writing_llm_section: str = "writing_llm"
    embedding_section: str = "embedding"
    prompt_dir: Path = Path("task_specs/prompts")
    before_scene_id: str | None = None
    before_scene_order: int | None = None
    scene_top_k: int = 5
    entity_memory_top_k: int = 12
    max_entity_memories_before_vector: int = 50
    entity_match_limit: int = 1
    collection_name: str = "dms_retrieval_documents"
    attribute_entity_types: tuple[str, ...] = ("character",)
    attribute_entity_names: tuple[str, ...] = ()
    max_memories_per_entity: int = 16
    style_reference_path: Path | None = None
    style_reference_script: Path | None = None
    style_reference_scene_id: str | None = None
    length_requirement: str = ""
    output_requirements: str = ""
    reference_text: str | None = None
    reference_text_file: Path | None = None
    reference_script: Path | None = None
    reference_scene_id: str | None = None
    overwrite: bool = False
    skip_evaluation: bool = False


def run_writing_e2e(config: WritingE2EConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_local_config(config.model_config_path)
    llm_client = build_openai_client_from_config(model_config, config.llm_section)
    writing_llm_client = build_openai_client_from_config(model_config, config.writing_llm_section)
    embedding_kwargs = embedding_kwargs_from_config(model_config, config.embedding_section)

    memory_packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=config.db_path,
            chroma_dir=config.chroma_dir,
            writing_intent=config.writing_intent,
            before_scene_id=config.before_scene_id,
            before_scene_order=config.before_scene_order,
            scene_top_k=config.scene_top_k,
            entity_memory_top_k=config.entity_memory_top_k,
            max_entity_memories_before_vector=config.max_entity_memories_before_vector,
            entity_match_limit=config.entity_match_limit,
            collection_name=config.collection_name,
            **embedding_kwargs,
        )
    )
    memory_packet_json_path = output_dir / "memory_packet.json"
    memory_packet_md_path = output_dir / "memory_packet.md"
    memory_packet_json_path.write_text(json.dumps(memory_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    memory_packet_md_path.write_text(format_memory_packet_markdown(memory_packet), encoding="utf-8")

    attribute_cards_dir = output_dir / "attribute_cards"
    attribute_summary = build_entity_attribute_cards(
        AttributeCardConfig(
            memory_packet_path=memory_packet_json_path,
            output_dir=attribute_cards_dir,
            prompt_dir=config.prompt_dir,
            entity_types=config.attribute_entity_types,
            entity_names=config.attribute_entity_names,
            max_memories_per_entity=config.max_memories_per_entity,
            overwrite=config.overwrite,
        ),
        llm_client=llm_client,
    )

    social_simulation_dir = output_dir / "social_simulation"
    social_summary = run_social_simulation(
        SocialSimulationConfig(
            attribute_cards_path=attribute_cards_dir / "attribute_cards.json",
            writing_intent=config.writing_intent,
            output_dir=social_simulation_dir,
            prompt_dir=config.prompt_dir,
            overwrite=config.overwrite,
        ),
        llm_client=llm_client,
    )

    writing_dir = output_dir / "writing"
    writing_summary = generate_writing_with_social_simulation_client(
        SocialWritingGenerationConfig(
            writing_request=config.writing_intent,
            memory_packet_path=memory_packet_md_path,
            attribute_cards_path=attribute_cards_dir / "attribute_cards.md",
            social_simulation_path=social_simulation_dir / "social_simulation.md",
            output_dir=writing_dir,
            model_config_path=config.model_config_path,
            model_section=config.writing_llm_section,
            prompt_dir=config.prompt_dir,
            style_reference_path=config.style_reference_path,
            style_reference_script=config.style_reference_script,
            style_reference_scene_id=config.style_reference_scene_id,
            length_requirement=config.length_requirement,
            output_requirements=config.output_requirements,
            overwrite=config.overwrite,
        ),
        llm_client=writing_llm_client,
        model_config=model_config,
    )

    evaluation_summary = None
    if not config.skip_evaluation:
        reference_text = _resolve_reference_text(config)
        evaluation_dir = output_dir / "evaluation"
        evaluation_summary = evaluate_writing(
            WritingEvaluationConfig(
                writing_intent=config.writing_intent,
                generated_text=(writing_dir / "draft.md").read_text(encoding="utf-8").strip(),
                memory_packet=memory_packet_md_path.read_text(encoding="utf-8"),
                reference_text=reference_text,
                output_dir=evaluation_dir,
                prompt_dir=config.prompt_dir,
                overwrite=config.overwrite,
            ),
            llm_client=llm_client,
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "policy": {
            "post_generation_repair": "disabled",
            "evaluation_text": "raw writing/draft.md",
        },
        "model_config_path": str(config.model_config_path),
        "model_sections": {
            "llm": config.llm_section,
            "embedding": config.embedding_section,
            "writing_llm": config.writing_llm_section,
        },
        "model_config": {
            "llm": redact_model_config(model_config.get(config.llm_section, {})),
            "embedding": redact_model_config(model_config.get(config.embedding_section, {})),
            "writing_llm": redact_model_config(model_config.get(config.writing_llm_section, {})),
        },
        "inputs": {
            "db_path": str(config.db_path),
            "chroma_dir": str(config.chroma_dir),
            "writing_intent": config.writing_intent,
            "before_scene_id": config.before_scene_id,
            "before_scene_order": config.before_scene_order,
            "style_reference_path": str(config.style_reference_path) if config.style_reference_path else None,
            "style_reference_script": str(config.style_reference_script) if config.style_reference_script else None,
            "style_reference_scene_id": config.style_reference_scene_id,
            "has_reference_text": _has_reference_text(config),
        },
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "memory_packet_json": str(memory_packet_json_path),
            "memory_packet_markdown": str(memory_packet_md_path),
            "attribute_cards_dir": str(attribute_cards_dir),
            "social_simulation_dir": str(social_simulation_dir),
            "writing_dir": str(writing_dir),
            "evaluation_dir": str(output_dir / "evaluation") if evaluation_summary is not None else None,
        },
        "counts": {
            "retrieved_entities": len(memory_packet.get("entities") or []),
            "retrieved_memories": len(memory_packet.get("episodic_memories") or []),
            "retrieved_relations": len(memory_packet.get("relations") or []),
            "related_scene_summaries": len(memory_packet.get("related_scene_summaries") or []),
            "attribute_cards": attribute_summary.get("card_count"),
            "character_simulations": social_summary.get("character_simulation_count"),
        },
        "writing": writing_summary,
        "evaluation": evaluation_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _resolve_reference_text(config: WritingE2EConfig) -> str | None:
    if config.reference_text is not None and config.reference_text_file is not None:
        raise ValueError("Use either reference_text or reference_text_file, not both")
    if config.reference_text is not None:
        explicit_reference = config.reference_text
    elif config.reference_text_file is not None:
        explicit_reference = config.reference_text_file.read_text(encoding="utf-8").strip()
    else:
        explicit_reference = None

    scene_reference = _read_scene_text(config.reference_script, config.reference_scene_id)
    if explicit_reference is not None and scene_reference is not None:
        raise ValueError("Use either explicit reference text or reference scene, not both")
    return explicit_reference if explicit_reference is not None else scene_reference


def _has_reference_text(config: WritingE2EConfig) -> bool:
    return any(
        [
            config.reference_text is not None,
            config.reference_text_file is not None,
            config.reference_script is not None and config.reference_scene_id is not None,
        ]
    )


def _read_scene_text(script_path: Path | None, scene_id: str | None) -> str | None:
    if script_path is None or scene_id is None:
        return None
    scenes = load_script_scenes(script_path)
    for scene in scenes:
        if scene.scene_id == scene_id:
            return scene.content.strip()
    raise ValueError(f"Scene not found in {script_path}: {scene_id}")
