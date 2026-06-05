from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TEMPORAL_RELATION_TYPES = (
    "before",
    "after",
    "overlaps",
    "same_time",
    "contains",
    "causes",
    "anticipates",
    "claims",
    "reveals_past",
    "uncertain",
)

EVENT_TIME_MODES = (
    "present_scene",
    "past_recalled",
    "future_anticipated",
    "habitual",
    "hypothetical",
    "uncertain",
)

TEMPORAL_GRANULARITIES = (
    "absolute_date",
    "year",
    "season",
    "day",
    "time_of_day",
    "scene_relative",
    "event_relative",
    "unknown",
)

EVENT_TRACKS = (
    "plot",
    "context",
    "forecast",
    "memory",
    "hypothetical",
    "unknown",
)

ENVIRONMENT_PARTICIPANT_TERMS = (
    "太阳",
    "恒星",
    "冰川",
    "海啸",
    "山火",
    "旱灾",
    "灾难",
    "城市",
    "物种",
    "环境",
    "地球",
    "海洋",
    "风暴",
    "太阳风暴",
)

SCENE_TEMPORAL_ROLES = (
    "plot_scene",
    "visual_context",
    "montage",
    "exposition",
    "forecast",
    "memory_or_flashback",
    "mixed",
    "unknown",
)


@dataclass(frozen=True)
class TimelineBuildConfig:
    use_discourse_order_prior: bool = True
    discourse_prior_confidence: float = 0.35
    scene_block_prior_confidence: float = 0.55
    hard_relation_confidence: float = 0.75


def normalize_temporal_scene_output(
    data: Any,
    *,
    scene_id: str,
    source_record_id: int | None = None,
    discourse_index: int | None = None,
) -> dict[str, Any]:
    payload = data if isinstance(data, dict) else {}
    unit_id = str(payload.get("unit_id") or scene_id)
    scene_index = _normalize_scene_temporal_index(
        payload.get("scene_temporal_index"),
        scene_id=scene_id,
        source_record_id=source_record_id,
        discourse_index=discourse_index,
    )
    events = [
        _normalize_event(
            item,
            scene_id=scene_id,
            unit_id=unit_id,
            source_record_id=source_record_id,
            discourse_index=discourse_index,
            index=index,
            scene_temporal_role=scene_index.get("scene_temporal_role"),
        )
        for index, item in enumerate(_list(payload.get("temporal_events")), start=1)
        if isinstance(item, dict)
    ]
    event_ids = {event["event_id"] for event in events}
    relations = [
        _normalize_relation(
            item,
            scene_id=scene_id,
            unit_id=unit_id,
            source_record_id=source_record_id,
            discourse_index=discourse_index,
            index=index,
            event_ids=event_ids,
        )
        for index, item in enumerate(_list(payload.get("temporal_relations")), start=1)
        if isinstance(item, dict)
    ]
    warnings = [
        _normalize_warning(item, scene_id=scene_id, index=index)
        for index, item in enumerate(_list(payload.get("temporal_warnings")), start=1)
        if isinstance(item, dict) or str(item).strip()
    ]
    return {
        "unit_id": unit_id,
        "scene_id": scene_id,
        "source_record_id": source_record_id,
        "discourse_index": discourse_index,
        "temporal_events": events,
        "temporal_relations": relations,
        "scene_temporal_index": scene_index,
        "temporal_warnings": warnings,
    }


def _normalize_event(
    item: dict[str, Any],
    *,
    scene_id: str,
    unit_id: str,
    source_record_id: int | None,
    discourse_index: int | None,
    index: int,
    scene_temporal_role: Any = None,
) -> dict[str, Any]:
    event_id = str(item.get("event_id") or item.get("event_id_hint") or "").strip()
    if not event_id:
        event_id = f"{scene_id}:event_{index:03d}"
    elif ":" not in event_id:
        event_id = f"{scene_id}:{_safe_id(event_id)}"
    mode = _enum(item.get("event_time_mode"), EVENT_TIME_MODES, "present_scene")
    confidence = _confidence(item.get("confidence"), default=0.6)
    revealed_at_scene_id = str(item.get("revealed_at_scene_id") or scene_id)
    participants = _strings(item.get("participants"))
    return {
        "event_id": event_id,
        "scene_id": scene_id,
        "unit_id": unit_id,
        "source_record_id": source_record_id,
        "discourse_index": discourse_index,
        "summary": str(item.get("summary") or "").strip(),
        "participants": participants,
        "location": str(item.get("location") or "").strip(),
        "event_track": _event_track(item, mode=mode, participants=participants, scene_temporal_role=scene_temporal_role),
        "event_time_mode": mode,
        "story_time_hint": str(item.get("story_time_hint") or "").strip(),
        "granularity": _enum(item.get("granularity"), TEMPORAL_GRANULARITIES, "unknown"),
        "evidence": str(item.get("evidence") or "").strip(),
        "confidence": confidence,
        "revealed_at_scene_id": revealed_at_scene_id,
        "revealed_at_source_record_id": source_record_id if revealed_at_scene_id == scene_id else None,
        "is_reveal_of_past": mode == "past_recalled" or bool(item.get("is_reveal_of_past")),
        "discourse_visible": True,
    }


def _normalize_relation(
    item: dict[str, Any],
    *,
    scene_id: str,
    unit_id: str,
    source_record_id: int | None,
    discourse_index: int | None,
    index: int,
    event_ids: set[str],
) -> dict[str, Any]:
    relation_id = str(item.get("relation_id") or "").strip() or f"{scene_id}:rel_{index:03d}"
    source = _normalize_ref(str(item.get("source_event_id") or item.get("source") or "").strip(), scene_id, event_ids)
    target = _normalize_ref(str(item.get("target_event_id") or item.get("target") or "").strip(), scene_id, event_ids)
    relation_type = _enum(item.get("relation_type"), TEMPORAL_RELATION_TYPES, "uncertain")
    return {
        "relation_id": relation_id if ":" in relation_id else f"{scene_id}:{_safe_id(relation_id)}",
        "scene_id": scene_id,
        "unit_id": unit_id,
        "source_record_id": source_record_id,
        "discourse_index": discourse_index,
        "source_event_id": source,
        "target_event_id": target,
        "relation_type": relation_type,
        "evidence": str(item.get("evidence") or "").strip(),
        "confidence": _confidence(item.get("confidence"), default=0.6),
        "is_inferred": bool(item.get("is_inferred")),
        "source": "llm",
    }


def _normalize_scene_temporal_index(
    value: Any,
    *,
    scene_id: str,
    source_record_id: int | None,
    discourse_index: int | None,
) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {
        "scene_id": scene_id,
        "source_record_id": source_record_id,
        "discourse_index": discourse_index,
        "dominant_time_mode": _enum(payload.get("dominant_time_mode"), EVENT_TIME_MODES, "present_scene"),
        "scene_temporal_role": _enum(payload.get("scene_temporal_role"), SCENE_TEMPORAL_ROLES, "unknown"),
        "relative_to_previous_scene": str(payload.get("relative_to_previous_scene") or "unknown").strip(),
        "absolute_time_hints": _strings(payload.get("absolute_time_hints")),
        "relative_time_hints": _strings(payload.get("relative_time_hints")),
        "contains_flashback_or_recalled_past": bool(payload.get("contains_flashback_or_recalled_past")),
        "contains_parallel_or_overlap": bool(payload.get("contains_parallel_or_overlap")),
        "confidence": _confidence(payload.get("confidence"), default=0.5),
        "evidence": str(payload.get("evidence") or "").strip(),
    }


def _normalize_warning(item: Any, *, scene_id: str, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "warning_id": str(item.get("warning_id") or f"{scene_id}:warning_{index:03d}"),
            "scene_id": scene_id,
            "warning_type": str(item.get("warning_type") or item.get("type") or "temporal_uncertainty"),
            "detail": str(item.get("detail") or item.get("warning") or "").strip(),
            "evidence": str(item.get("evidence") or "").strip(),
        }
    return {
        "warning_id": f"{scene_id}:warning_{index:03d}",
        "scene_id": scene_id,
        "warning_type": "temporal_uncertainty",
        "detail": str(item).strip(),
        "evidence": "",
    }


def _normalize_ref(ref: str, scene_id: str, event_ids: set[str]) -> str:
    if not ref:
        return ""
    if ref in event_ids or ":" in ref:
        return ref
    local = f"{scene_id}:{_safe_id(ref)}"
    return local if local in event_ids else ref


def _enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _event_track(
    item: dict[str, Any],
    *,
    mode: str,
    participants: list[str],
    scene_temporal_role: Any = None,
) -> str:
    explicit = _enum(item.get("event_track"), EVENT_TRACKS, "")
    if explicit:
        return explicit
    if mode == "future_anticipated":
        return "forecast"
    if mode == "past_recalled":
        return "memory"
    if mode == "hypothetical":
        return "hypothetical"
    scene_role = str(scene_temporal_role or "").strip()
    environment_only = _has_only_environment_participants(participants)
    if scene_role in {"visual_context", "montage"} and environment_only:
        return "context"
    if mode == "habitual" and environment_only:
        return "context"
    text = " ".join(
        str(item.get(key) or "")
        for key in ("summary", "evidence", "location", "story_time_hint")
    )
    if environment_only and _looks_like_visual_or_environment_context(text):
        return "context"
    return "plot"


def _has_only_environment_participants(participants: list[str]) -> bool:
    if not participants:
        return True
    for participant in participants:
        text = str(participant or "").strip()
        if not text:
            continue
        if not any(term in text for term in ENVIRONMENT_PARTICIPANT_TERMS):
            return False
    return True


def _looks_like_visual_or_environment_context(text: str) -> bool:
    terms = (
        "蒙太奇",
        "画面",
        "字卡",
        "冰川",
        "海啸",
        "太阳",
        "恒星",
        "日珥",
        "黑子",
        "膨胀",
        "灾难",
        "山火",
        "旱灾",
        "物种",
        "城市",
        "爆炸",
        "枪战",
        "硝烟",
        "废墟",
        "环境",
        "世界状态",
    )
    return any(term in text for term in terms)


def _confidence(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return round(max(0.0, min(1.0, float(value))), 4)
    try:
        return round(max(0.0, min(1.0, float(str(value).strip()))), 4)
    except ValueError:
        return default


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _list(value) if str(item).strip()]


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value.strip())
    return safe.strip("_") or "unknown"
