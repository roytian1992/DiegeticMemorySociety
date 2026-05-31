from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.config import build_openai_client_from_config, load_local_config, redact_model_config
from dms.llm import LLMClient, LLMResult
from dms.prompts import YAMLPromptLoader
from dms.scripts.wandering_earth import load_script_scenes


@dataclass(frozen=True)
class SocialWritingGenerationConfig:
    writing_request: str
    memory_packet_path: Path
    attribute_cards_path: Path
    social_simulation_path: Path
    output_dir: Path
    model_config_path: Path = Path("configs/local_config.yaml")
    model_section: str = "writing_llm"
    prompt_dir: Path = Path("task_specs/prompts")
    style_reference_path: Path | None = None
    style_reference_script: Path | None = None
    style_reference_scene_id: str | None = None
    length_requirement: str = ""
    output_requirements: str = ""
    overwrite: bool = False


def generate_writing_with_social_simulation(config: SocialWritingGenerationConfig) -> dict[str, Any]:
    return generate_writing_with_social_simulation_client(config)


def generate_writing_with_social_simulation_client(
    config: SocialWritingGenerationConfig,
    llm_client: LLMClient | None = None,
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = model_config if model_config is not None else load_local_config(config.model_config_path)
    llm_client = llm_client if llm_client is not None else build_openai_client_from_config(model_config, config.model_section)
    style_reference = _read_style_reference(config)
    if style_reference:
        (output_dir / "style_reference.md").write_text(style_reference.rstrip() + "\n", encoding="utf-8")

    prompt = YAMLPromptLoader(config.prompt_dir).render(
        "dms/writing_generation_social",
        task_values={
            "writing_request": config.writing_request,
            "memory_packet": Path(config.memory_packet_path).read_text(encoding="utf-8"),
            "attribute_cards": Path(config.attribute_cards_path).read_text(encoding="utf-8"),
            "social_simulation": Path(config.social_simulation_path).read_text(encoding="utf-8"),
            "style_reference": style_reference,
            "length_requirement": config.length_requirement,
            "output_requirements": config.output_requirements,
        },
    )
    prompt_path = output_dir / "prompt.md"
    draft_path = output_dir / "draft.md"
    raw_path = output_dir / "raw_response.json"
    quick_eval_path = output_dir / "quick_eval.json"
    metadata_path = output_dir / "metadata.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = llm_client.complete(prompt)
    draft = result.text.strip()
    draft_path.write_text(draft + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps(_llm_result_to_dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quick_eval = _quick_eval(draft, prompt_path=prompt_path, draft_path=draft_path)
    quick_eval_path.write_text(json.dumps(quick_eval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    section_config = model_config.get(config.model_section) if isinstance(model_config.get(config.model_section), dict) else {}
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "template": "task_specs/prompts/dms/writing_generation_social.yaml",
        "model_config_path": str(config.model_config_path),
        "model_section": config.model_section,
        "model_config": redact_model_config(section_config),
        "inputs": {
            "memory_packet_path": str(config.memory_packet_path),
            "attribute_cards_path": str(config.attribute_cards_path),
            "social_simulation_path": str(config.social_simulation_path),
            "style_reference_path": str(config.style_reference_path) if config.style_reference_path else None,
            "style_reference_script": str(config.style_reference_script) if config.style_reference_script else None,
            "style_reference_scene_id": config.style_reference_scene_id,
            "writing_request": config.writing_request,
        },
        "length_requirement": config.length_requirement,
        "output_requirements": config.output_requirements,
        "output": quick_eval,
        "usage": result.usage,
        "artifacts": {
            "prompt": str(prompt_path),
            "draft": str(draft_path),
            "raw_response": str(raw_path),
            "quick_eval": str(quick_eval_path),
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _read_style_reference(config: SocialWritingGenerationConfig) -> str:
    if config.style_reference_path:
        return Path(config.style_reference_path).read_text(encoding="utf-8").strip()
    if config.style_reference_script and config.style_reference_scene_id:
        scenes = load_script_scenes(config.style_reference_script)
        for scene in scenes:
            if scene.scene_id == config.style_reference_scene_id:
                return scene.content.strip()
        raise ValueError(f"Scene not found in {config.style_reference_script}: {config.style_reference_scene_id}")
    return ""


def _quick_eval(draft: str, *, prompt_path: Path, draft_path: Path) -> dict[str, Any]:
    return {
        "draft_path": str(draft_path),
        "prompt_path": str(prompt_path),
        "body_chars": len(draft),
        "body_non_ws_chars": len(re.sub(r"\s+", "", draft)),
        "entities_present": {name: name in draft for name in ["张鹏", "刘培强", "J20C", "UEG"]},
        "ref_ids_present": sorted(set(re.findall(r"\b[MR]\d+\b|R\*|M\*|reference id|memory index", draft))),
    }


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
