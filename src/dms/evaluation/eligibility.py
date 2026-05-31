from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dms.scripts.wandering_earth import ScriptScene, load_script_scenes


_DIALOGUE_RE = re.compile(r"[：:][“\"'‘（(]|\w+\s*[：:]|[\u4e00-\u9fffA-Za-z0-9_（）()·]+[：:]")
_VO_RE = re.compile(r"(解说|旁白|画外音|VO|V\.O\.|OS|O\.S\.)", re.IGNORECASE)
_VISUAL_TERMS = (
    "镜头",
    "画面",
    "特效",
    "CG",
    "全景",
    "远景",
    "近景",
    "航拍",
    "俯瞰",
    "闪回",
    "蒙太奇",
    "切至",
    "转场",
    "字幕",
    "屏幕上",
    "新闻画面",
    "影像",
)
_CG_TERMS = (
    "爆炸",
    "坍塌",
    "碎片",
    "火光",
    "火球",
    "冲击波",
    "海啸",
    "地震",
    "山火",
    "旱灾",
    "灾难",
    "太空",
    "地球",
    "月球",
    "发动机",
    "飞船",
)
_DECISION_CONFLICT_TERMS = (
    "决定",
    "命令",
    "拒绝",
    "同意",
    "争执",
    "冲突",
    "质问",
    "威胁",
    "谈判",
    "追问",
    "为什么",
    "必须",
    "不能",
    "逃避",
    "禁止",
    "救",
    "杀",
    "牺牲",
)
_ACTION_TERMS = (
    "走",
    "跑",
    "冲",
    "拿",
    "打开",
    "关闭",
    "启动",
    "按下",
    "检查",
    "扫描",
    "连接",
    "抓",
    "推",
    "打",
    "进入",
    "离开",
    "举起",
    "递给",
)
_EXPOSITION_TERMS = (
    "理论上",
    "本质上",
    "意味着",
    "说明",
    "介绍",
    "解释",
    "技术",
    "规则",
    "计划",
    "系统",
    "文明",
)
_CHARACTER_HINT_TERMS = (
    "科学家",
    "受试者",
    "科研人员",
    "工作人员",
    "士兵",
    "警卫",
    "孩子",
    "男人",
    "女人",
    "老人",
    "父亲",
    "母亲",
    "老师",
    "学生",
)


def classify_scene(scene: ScriptScene) -> dict[str, Any]:
    """Classify one scene for memory inclusion and writing-eval eligibility.

    This is intentionally deterministic. It is a first-pass filter for building
    evaluation splits, not a replacement for later human or LLM adjudication.
    """

    text = scene.content or ""
    title = scene.title or ""
    combined = f"{title}\n{text}"
    dialogue_count = _dialogue_count(text)
    content_len = len(text)
    visual_hits = _count_terms(combined, _VISUAL_TERMS)
    cg_hits = _count_terms(combined, _CG_TERMS)
    decision_hits = _count_terms(combined, _DECISION_CONFLICT_TERMS)
    action_hits = _count_terms(combined, _ACTION_TERMS)
    exposition_hits = _count_terms(combined, _EXPOSITION_TERMS)
    character_hint_hits = _count_terms(combined, _CHARACTER_HINT_TERMS)
    vo_hits = len(_VO_RE.findall(combined))

    has_dialogue = dialogue_count > 0
    has_named_character = character_hint_hits > 0 or has_dialogue
    has_state_change = action_hits > 0 or cg_hits > 0
    has_decision_or_conflict = decision_hits > 0
    is_visual_or_cg_heavy = visual_hits + cg_hits >= 3 and dialogue_count <= 1
    is_transition_or_montage = any(term in combined for term in ("蒙太奇", "转场", "切至", "闪回", "字幕"))
    is_vo_or_exposition = vo_hits > 0 or exposition_hits >= 2
    low_character_action_density = dialogue_count == 0 and decision_hits == 0 and action_hits <= 1

    unit_type = _unit_type(
        has_dialogue=has_dialogue,
        has_decision_or_conflict=has_decision_or_conflict,
        is_visual_or_cg_heavy=is_visual_or_cg_heavy,
        is_transition_or_montage=is_transition_or_montage,
        is_vo_or_exposition=is_vo_or_exposition,
        has_state_change=has_state_change,
        character_hint_hits=character_hint_hits,
        content_len=content_len,
    )
    writing_eval_policy = _writing_eval_policy(
        unit_type=unit_type,
        has_decision_or_conflict=has_decision_or_conflict,
        is_visual_or_cg_heavy=is_visual_or_cg_heavy,
        is_transition_or_montage=is_transition_or_montage,
        low_character_action_density=low_character_action_density,
    )
    audit_eval_policy = "include" if has_state_change or has_dialogue or is_vo_or_exposition else "exclude"
    memory_policy = "include" if content_len > 0 else "exclude"
    reasons = _reasons(
        unit_type=unit_type,
        has_dialogue=has_dialogue,
        has_decision_or_conflict=has_decision_or_conflict,
        is_visual_or_cg_heavy=is_visual_or_cg_heavy,
        is_transition_or_montage=is_transition_or_montage,
        is_vo_or_exposition=is_vo_or_exposition,
        low_character_action_density=low_character_action_density,
        has_state_change=has_state_change,
    )

    return {
        "scene_id": scene.scene_id,
        "source_record_id": scene.source_record_id,
        "discourse_index": scene.discourse_index,
        "title": scene.title,
        "unit_type": unit_type,
        "memory_policy": memory_policy,
        "writing_eval_policy": writing_eval_policy,
        "audit_eval_policy": audit_eval_policy,
        "reasons": reasons,
        "signals": {
            "content_char_count": content_len,
            "dialogue_count": dialogue_count,
            "visual_hit_count": visual_hits,
            "cg_hit_count": cg_hits,
            "decision_conflict_hit_count": decision_hits,
            "action_hit_count": action_hits,
            "exposition_hit_count": exposition_hits,
            "vo_hit_count": vo_hits,
            "character_hint_hit_count": character_hint_hits,
            "has_dialogue": has_dialogue,
            "has_named_character": has_named_character,
            "has_state_change": has_state_change,
            "has_decision_or_conflict": has_decision_or_conflict,
            "is_visual_or_cg_heavy": is_visual_or_cg_heavy,
            "is_transition_or_montage": is_transition_or_montage,
            "is_vo_or_exposition": is_vo_or_exposition,
            "low_character_action_density": low_character_action_density,
        },
    }


def build_scene_eligibility_splits(script_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    scenes = load_script_scenes(script_path)
    records = [classify_scene(scene) for scene in scenes]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "all": out_path / "scene_eligibility_all.jsonl",
        "memory_prefix": out_path / "memory_prefix.jsonl",
        "writing_eval_targets": out_path / "writing_eval_targets.jsonl",
        "audit_eval_targets": out_path / "audit_eval_targets.jsonl",
        "excluded_from_generation_eval": out_path / "excluded_from_generation_eval.jsonl",
        "summary": out_path / "summary.json",
    }

    _write_jsonl(paths["all"], records)
    _write_jsonl(paths["memory_prefix"], [record for record in records if record["memory_policy"] == "include"])
    _write_jsonl(paths["writing_eval_targets"], [record for record in records if record["writing_eval_policy"] == "include"])
    _write_jsonl(paths["audit_eval_targets"], [record for record in records if record["audit_eval_policy"] == "include"])
    _write_jsonl(
        paths["excluded_from_generation_eval"],
        [record for record in records if record["writing_eval_policy"] != "include"],
    )

    unit_counts = Counter(str(record["unit_type"]) for record in records)
    writing_counts = Counter(str(record["writing_eval_policy"]) for record in records)
    audit_counts = Counter(str(record["audit_eval_policy"]) for record in records)
    summary = {
        "script_path": str(Path(script_path).resolve()),
        "output_dir": str(out_path),
        "scene_count": len(records),
        "memory_prefix_count": sum(1 for record in records if record["memory_policy"] == "include"),
        "writing_eval_target_count": sum(1 for record in records if record["writing_eval_policy"] == "include"),
        "audit_eval_target_count": sum(1 for record in records if record["audit_eval_policy"] == "include"),
        "excluded_from_generation_eval_count": sum(1 for record in records if record["writing_eval_policy"] != "include"),
        "unit_type_counts": dict(sorted(unit_counts.items())),
        "writing_eval_policy_counts": dict(sorted(writing_counts.items())),
        "audit_eval_policy_counts": dict(sorted(audit_counts.items())),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _unit_type(
    *,
    has_dialogue: bool,
    has_decision_or_conflict: bool,
    is_visual_or_cg_heavy: bool,
    is_transition_or_montage: bool,
    is_vo_or_exposition: bool,
    has_state_change: bool,
    character_hint_hits: int,
    content_len: int,
) -> str:
    if content_len == 0:
        return "empty"
    if is_transition_or_montage:
        return "montage_or_transition"
    if is_visual_or_cg_heavy:
        return "pure_visual_vfx"
    if has_dialogue and has_decision_or_conflict:
        return "conflict_scene"
    if has_dialogue:
        return "dialogue_scene"
    if has_decision_or_conflict or (has_state_change and character_hint_hits > 0):
        return "character_action_scene"
    if is_vo_or_exposition:
        return "exposition_or_vo"
    return "establishing_or_visual_description"


def _writing_eval_policy(
    *,
    unit_type: str,
    has_decision_or_conflict: bool,
    is_visual_or_cg_heavy: bool,
    is_transition_or_montage: bool,
    low_character_action_density: bool,
) -> str:
    if unit_type in {"empty", "pure_visual_vfx", "montage_or_transition", "establishing_or_visual_description"}:
        return "exclude"
    if is_visual_or_cg_heavy or is_transition_or_montage:
        return "exclude"
    if low_character_action_density and not has_decision_or_conflict:
        return "exclude"
    if unit_type in {"conflict_scene", "dialogue_scene", "character_action_scene"}:
        return "include"
    return "review"


def _reasons(
    *,
    unit_type: str,
    has_dialogue: bool,
    has_decision_or_conflict: bool,
    is_visual_or_cg_heavy: bool,
    is_transition_or_montage: bool,
    is_vo_or_exposition: bool,
    low_character_action_density: bool,
    has_state_change: bool,
) -> list[str]:
    reasons = [f"classified as {unit_type}"]
    if has_dialogue:
        reasons.append("contains dialogue or speaker-style lines")
    if has_decision_or_conflict:
        reasons.append("contains decision/conflict signals")
    if has_state_change:
        reasons.append("contains action/state-change or world-state signals")
    if is_visual_or_cg_heavy:
        reasons.append("visual/CG-heavy with low dialogue density")
    if is_transition_or_montage:
        reasons.append("contains transition, montage, subtitle, or flashback signals")
    if is_vo_or_exposition:
        reasons.append("contains VO/exposition/world-rule signals")
    if low_character_action_density:
        reasons.append("low character decision/action density")
    return reasons


def _dialogue_count(text: str) -> int:
    return len(_DIALOGUE_RE.findall(text or ""))


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum((text or "").count(term) for term in terms)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
