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


_WRITER_PACKET_ARTIFACT_TERMS = (
    "posture only",
    "not_final_dialogue",
    "beat_id",
    "action_type",
    "selected sequence score",
    "hard violations",
    "soft warnings",
    "source isolation",
    "social simulation intent",
)

_DIALOGUE_RISK_PHRASES = (
    "别跟地球赌气",
    "和自己和解",
    "你是在逃避",
    "放下",
)

_DEFAULT_REQUEST_ANCHORS = (
    "刘培强",
    "张鹏",
    "J20C",
    "歼二零C",
    "歼20C",
    "UEG",
    "利伯维尔",
    "月亮",
    "地球",
)

_VEHICLE_OR_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9-]{1,20})(?![A-Za-z0-9])"
    r"|歼[0-9A-Za-z一二三四五六七八九零〇-]{1,12}"
)


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
    previous_scene_context: str = ""
    previous_scene_context_path: Path | None = None
    previous_scene_context_script: Path | None = None
    previous_scene_context_scene_id: str | None = None
    previous_scene_context_max_chars: int = 800
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
    previous_scene_context = _read_previous_scene_context(config)
    if previous_scene_context:
        (output_dir / "previous_scene_context.md").write_text(
            previous_scene_context.rstrip() + "\n",
            encoding="utf-8",
        )
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
            "previous_scene_context": previous_scene_context,
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

    quick_eval = _quick_eval(
        draft,
        prompt_path=prompt_path,
        draft_path=draft_path,
        writing_request=config.writing_request,
    )
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
            "previous_scene_context_path": str(config.previous_scene_context_path)
            if config.previous_scene_context_path
            else None,
            "previous_scene_context_script": str(config.previous_scene_context_script)
            if config.previous_scene_context_script
            else None,
            "previous_scene_context_scene_id": config.previous_scene_context_scene_id,
            "previous_scene_context_chars": len(previous_scene_context),
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


def format_previous_scene_context(
    scene: Any | None,
    *,
    max_chars: int = 800,
    summary: str = "",
    entities: list[str] | tuple[str, ...] = (),
) -> str:
    if scene is None:
        return ""

    limit = max(int(max_chars or 0), 80)
    scene_id = str(getattr(scene, "scene_id", "") or "").strip()
    title = str(getattr(scene, "title", "") or "").strip()
    subtitle = str(getattr(scene, "subtitle", "") or "").strip()
    location = str(getattr(scene, "location_hint", "") or "").strip()
    content = _collapse_blank_lines(str(getattr(scene, "content", "") or "").strip())

    header_parts = [part for part in [scene_id, title] if part]
    lines = ["Previous scene: " + (" | ".join(header_parts) if header_parts else "unknown")]
    if subtitle:
        lines.append(f"Subtitle: {subtitle}")
    if location:
        lines.append(f"Location: {location}")

    if content:
        full_context = "\n".join([*lines, "Full text:", content]).strip()
        if len(re.sub(r"\s+", "", full_context)) <= limit:
            return full_context

    entity_names = _unique_nonempty([*entities, *_extract_scene_entity_names(content)])
    summary_text = str(summary or "").strip() or _fallback_scene_summary(content, max_chars=max(limit - 120, 80))
    summary_budget = max(limit - sum(len(name) for name in entity_names[:12]) - 120, 80)
    summary_text = _truncate_non_ws(summary_text, summary_budget)
    lines.append("Summary:")
    lines.append(summary_text or "No compact summary available.")
    lines.append("Entities:")
    lines.append("、".join(entity_names[:12]) if entity_names else "No entities extracted.")
    return "\n".join(lines).strip()


def _read_previous_scene_context(config: SocialWritingGenerationConfig) -> str:
    explicit_sources = [
        bool(str(config.previous_scene_context or "").strip()),
        config.previous_scene_context_path is not None,
        config.previous_scene_context_script is not None or config.previous_scene_context_scene_id is not None,
    ]
    if sum(1 for present in explicit_sources if present) > 1:
        raise ValueError(
            "Use only one previous-scene context source: direct text, context file, or script+scene_id."
        )
    if str(config.previous_scene_context or "").strip():
        return str(config.previous_scene_context).strip()
    if config.previous_scene_context_path:
        return Path(config.previous_scene_context_path).read_text(encoding="utf-8").strip()
    if config.previous_scene_context_script or config.previous_scene_context_scene_id:
        if not config.previous_scene_context_script or not config.previous_scene_context_scene_id:
            raise ValueError("previous_scene_context_script and previous_scene_context_scene_id must be provided together")
        scenes = load_script_scenes(config.previous_scene_context_script)
        for scene in scenes:
            if scene.scene_id == config.previous_scene_context_scene_id:
                return format_previous_scene_context(scene, max_chars=config.previous_scene_context_max_chars)
        raise ValueError(
            f"Scene not found in {config.previous_scene_context_script}: {config.previous_scene_context_scene_id}"
        )
    return ""


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


def _quick_eval(
    draft: str,
    *,
    prompt_path: Path,
    draft_path: Path,
    writing_request: str = "",
) -> dict[str, Any]:
    request_anchors = _extract_request_anchors(writing_request)
    return {
        "draft_path": str(draft_path),
        "prompt_path": str(prompt_path),
        "body_chars": len(draft),
        "body_non_ws_chars": len(re.sub(r"\s+", "", draft)),
        "entities_present": {
            name: name in draft for name in ["张鹏", "刘培强", "J20C", "歼二零C", "歼20C", "UEG"]
        },
        "request_anchors": request_anchors,
        "request_anchors_present": {anchor: _anchor_present(draft, anchor) for anchor in request_anchors},
        "missing_request_anchors": [
            anchor for anchor in request_anchors if not _anchor_present(draft, anchor)
        ],
        "ref_ids_present": sorted(set(re.findall(r"\b[MR]\d+\b|R\*|M\*|reference id|memory index", draft))),
        "writer_packet_artifact_terms_present": _terms_present(draft, _WRITER_PACKET_ARTIFACT_TERMS),
        "dialogue_risk_phrases_present": _terms_present(draft, _DIALOGUE_RISK_PHRASES),
    }


def _extract_request_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    anchors.extend(match.group(0) for match in _VEHICLE_OR_MODEL_RE.finditer(text))
    anchors.extend(name for name in _DEFAULT_REQUEST_ANCHORS if name in text)
    return _unique_nonempty(anchors)


def _anchor_present(text: str, anchor: str) -> bool:
    aliases = {
        "J20C": ("J20C", "歼二零C", "歼20C"),
        "歼二零C": ("J20C", "歼二零C", "歼20C"),
        "歼20C": ("J20C", "歼二零C", "歼20C"),
    }.get(anchor, (anchor,))
    return any(alias in text for alias in aliases)


def _terms_present(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    present = []
    for term in terms:
        haystack = lowered if term.isascii() else text
        needle = term.lower() if term.isascii() else term
        if needle in haystack:
            present.append(term)
    return sorted(set(present))


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_scene_entity_names(text: str) -> list[str]:
    names: list[str] = []
    for match in re.findall(r"^([^\n：:]{1,20})[：:]", text, flags=re.MULTILINE):
        label = match.strip()
        if any(token in label for token in ("注释", "备注", "（", "(", "，", "。", "、")):
            continue
        names.append(label)
    names.extend(match.strip() for match in re.findall(r"\b[A-Z][A-Z0-9-]{1,20}\b", text))
    names.extend(match.strip() for match in re.findall(r"歼[\u4e00-\u9fff0-9A-Za-z-]{1,12}", text))
    return _unique_nonempty(names)


def _fallback_scene_summary(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    half = max((max_chars - 5) // 2, 20)
    return f"{compact[:half]} ... {compact[-half:]}"


def _truncate_non_ws(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(max_chars - 1, 1)].rstrip() + "…"


def _unique_nonempty(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


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
