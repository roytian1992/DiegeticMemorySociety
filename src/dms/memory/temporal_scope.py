from __future__ import annotations

from typing import Any


MEMORY_TEMPORAL_SCOPES = (
    "temporal_episode",
    "atemporal_fact",
    "durable_state",
    "uncertain",
)

TIME_BOUND_MEMORY_TYPES = {
    "action",
    "dialogue",
    "decision",
    "knowledge_transfer",
    "relationship_change",
    "state_change",
}

ATEMPORAL_MEMORY_TYPES = {
    "setting",
    "observation",
}

DURABLE_MEMORY_TYPES = {
    "state",
    "durable_state",
    "relationship",
}

ATEMPORAL_KEYWORDS = (
    "本质",
    "原理",
    "规则",
    "技术",
    "定义",
    "意味着",
    "说明",
    "存在",
    "是一个",
    "是一种",
    "可以",
    "能够",
    "用于",
    "禁止",
    "不是",
    "身份",
    "属于",
    "位于",
    "由",
    "由此",
    "系统",
    "计划",
    "工程",
    "危机",
    "concept",
    "principle",
    "technology",
    "rule",
    "exists",
)

DURABLE_STATE_KEYWORDS = (
    "通过",
    "负责",
    "担任",
    "处于",
    "保持",
    "拥有",
    "掌握",
    "知道",
    "认为",
    "相信",
    "承诺",
    "照顾",
    "关系",
    "身份",
    "状态",
    "禁止",
    "准备",
    "即将",
    "计划",
    "has",
    "knows",
    "believes",
    "responsible",
    "status",
)

TEMPORAL_EPISODE_KEYWORDS = (
    "进入",
    "离开",
    "走",
    "飞",
    "驾驶",
    "打开",
    "关闭",
    "启动",
    "操作",
    "询问",
    "回答",
    "说",
    "告诉",
    "提醒",
    "回忆",
    "看到",
    "望向",
    "开始",
    "继续",
    "随后",
    "接着",
    "先",
    "后",
    "最终",
    "起飞",
    "降落",
    "爆发",
    "融化",
    "坠毁",
    "arrives",
    "enters",
    "leaves",
    "says",
    "asks",
    "starts",
)


def infer_memory_temporal_scope(memory: dict[str, Any]) -> dict[str, Any]:
    """Classify whether a memory needs story-time ordering.

    The classifier is deliberately conservative. It only marks a memory as
    atemporal when the text looks like a rule, setting, concept, or stable fact;
    otherwise concrete actions and utterances remain time-bound.
    """

    explicit = str(memory.get("memory_temporal_scope") or "").strip()
    if explicit in MEMORY_TEMPORAL_SCOPES:
        return {
            "memory_temporal_scope": explicit,
            "memory_temporal_scope_confidence": _confidence(memory.get("memory_temporal_scope_confidence"), 0.95),
            "memory_temporal_scope_reason": str(memory.get("memory_temporal_scope_reason") or "model provided temporal scope"),
        }

    memory_type = str(memory.get("memory_type") or "").strip().lower()
    text = " ".join(
        str(memory.get(field) or "")
        for field in ("summary", "evidence", "evidence_text", "timeline_label")
    )
    lowered = text.lower()

    if memory_type in DURABLE_MEMORY_TYPES or _contains_any(lowered, DURABLE_STATE_KEYWORDS):
        if not _contains_any(lowered, TEMPORAL_EPISODE_KEYWORDS[:12]):
            return _scope("durable_state", 0.72, "stable state, belief, role, plan, or responsibility cue")

    if memory_type in ATEMPORAL_MEMORY_TYPES and _contains_any(lowered, ATEMPORAL_KEYWORDS):
        return _scope("atemporal_fact", 0.78, "setting/observation with rule, concept, or stable fact cue")

    if memory_type in TIME_BOUND_MEMORY_TYPES or _contains_any(lowered, TEMPORAL_EPISODE_KEYWORDS):
        return _scope("temporal_episode", 0.8, "concrete action, speech act, reveal, or ordered episode cue")

    if _contains_any(lowered, ATEMPORAL_KEYWORDS):
        return _scope("atemporal_fact", 0.62, "stable fact or concept cue")

    return _scope("uncertain", 0.45, "no strong temporal-scope cue")


def is_story_time_bound_scope(scope: str) -> bool:
    return str(scope or "") in {"temporal_episode", "uncertain"}


def _scope(scope: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "memory_temporal_scope": scope,
        "memory_temporal_scope_confidence": confidence,
        "memory_temporal_scope_reason": reason,
    }


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in terms)


def _confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(parsed, 1.0))
