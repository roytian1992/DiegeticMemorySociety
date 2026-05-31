from __future__ import annotations

import re


ALLOWED_ENTITY_TYPES = {
    "character",
    "group",
    "organization",
    "location",
    "object",
    "concept",
    "occasion",
}

ALLOWED_SCENE_TAG_TYPES = {
    "background_element",
    "atmosphere",
    "illustrative_example",
    "generic_activity",
    "minor_visual_detail",
    "other",
}

SCENE_TAG_ENTITY_TYPE_ALIASES = {
    "background_element",
    "atmosphere",
    "illustrative_example",
    "generic_activity",
    "minor_visual_detail",
}

LOW_VALUE_OCCASION_SURFACES = {
    "山火",
    "旱灾",
    "一个物种的灭绝",
    "物种的灭绝",
    "一座城市的消失",
    "城市的消失",
    "视频录制",
    "录制",
}

LOW_VALUE_BACKGROUND_SURFACES = {
    "乌鸦",
    "水管",
    "积水",
    "巴士",
    "太阳",
    "杂草",
    "弹痕",
    "残垣断壁",
    "残破民用建筑",
    "超市栅栏门",
    "六层建筑",
    "老旧歪斜的摄像头",
}

NON_TRACKABLE_ENTITY_SURFACES = {
    "人",
    "脑",
    "大脑",
    "脑子",
    "身体",
    "人体",
    "这",
    "这个",
    "这个世界",
    "这儿",
    "这里",
    "那",
    "那个",
    "那里",
    "那儿",
    "这根线",
    "这条线",
}

ANATOMY_SURFACES = {
    "脑",
    "大脑",
    "脑子",
    "身体",
    "人体",
}

_ENTITY_TYPE_ALIASES = {
    "world_rule_or_concept": "concept",
    "world_rule": "concept",
    "worldbuilding_concept": "concept",
    "scientific_concept": "concept",
    "event_or_disaster": "occasion",
    "event": "occasion",
    "disaster": "occasion",
    "operation": "occasion",
    "activity": "occasion",
    "technology": "concept",
    "technology_or_facility": "object",
    "facility": "location",
    "device": "object",
    "vehicle": "object",
    "machine": "object",
    "document": "object",
    "media": "object",
    "media_or_document": "object",
    "time": "concept",
    "deadline": "concept",
    "time_or_deadline": "concept",
}


def normalize_entity_type(entity_type: object) -> str:
    value = str(entity_type or "").strip().lower()
    if not value:
        return "concept"
    value = value.replace(" ", "_").replace("-", "_")
    value = _ENTITY_TYPE_ALIASES.get(value, value)
    return value if value in ALLOWED_ENTITY_TYPES else "concept"


def is_supported_entity_type_label(entity_type: object) -> bool:
    value = str(entity_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    return value in ALLOWED_ENTITY_TYPES or value in _ENTITY_TYPE_ALIASES


def normalize_entity_name_key(name: object) -> str:
    return re.sub(r"[\s,\-.，。:：/（）()]+", "", str(name or "").strip().lower())


def extract_countdown_core_entity(surface: object) -> str:
    text = str(surface or "").strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if "还剩" not in compact and "剩余" not in compact and "还有" not in compact:
        return ""
    match = re.search(r"(?:距|距离)?(.+?)(?:还剩|剩余|还有)", compact)
    if not match:
        return ""
    candidate = match.group(1).strip("：:，,。；;")
    candidate = re.sub(r"^(?:字卡|字幕|倒计时)", "", candidate).strip("：:，,。；;")
    if 2 <= len(candidate) <= 12 and any(
        term in candidate for term in ("氦闪", "危机", "灾难", "事件", "事故", "计划", "行动", "工程", "任务")
    ):
        return candidate
    return ""


def is_deictic_surface(surface: object) -> bool:
    clean = str(surface or "").strip().lower()
    if not clean:
        return False
    compact = clean.replace(" ", "")
    if compact in {"这", "这个", "这儿", "这里", "那", "那个", "那儿", "那里", "他", "她", "它", "他们", "她们", "它们"}:
        return True
    if clean in {"this", "that", "here", "there", "he", "she", "it", "they"}:
        return True
    return clean.startswith(("这", "那", "this ", "that "))


def embedded_alnum_code_key(name: object) -> str:
    text = str(name or "")
    matches = re.findall(r"[A-Za-z]*\d+[A-Za-z]*", text)
    return matches[-1].lower() if matches else ""


def normalize_entity_canonical_hint(
    *,
    surface: object,
    entity_type: object,
    canonical_hint: object,
    evidence: object,
) -> str:
    hint = str(canonical_hint or "").strip()
    if not hint:
        return ""

    surface_text = str(surface or "").strip()
    evidence_text = str(evidence or "")
    normalized_type = normalize_entity_type(entity_type)
    if hint == surface_text or hint in evidence_text:
        return hint

    surface_code = embedded_alnum_code_key(surface_text)
    hint_code = embedded_alnum_code_key(hint)
    if surface_code and hint_code and surface_code == hint_code:
        return hint

    if normalized_type in {"character", "group"} and _looks_like_supported_name_alias(surface_text, hint):
        return hint

    return ""


def entity_trackability_issue(
    *,
    surface: object,
    entity_type: object,
    canonical_hint: object = "",
    evidence: object = "",
) -> str:
    surface_text = str(surface or "").strip()
    if not surface_text:
        return "empty entity surface"

    compact = normalize_entity_name_key(surface_text)
    if compact in {normalize_entity_name_key(value) for value in NON_TRACKABLE_ENTITY_SURFACES}:
        return "generic or deictic phrase is not a trackable story entity"

    if is_deictic_surface(surface_text):
        hint = normalize_entity_canonical_hint(
            surface=surface_text,
            entity_type=entity_type,
            canonical_hint=canonical_hint,
            evidence=evidence,
        )
        if not hint:
            return "deictic mention without supported named referent"

    normalized_type = normalize_entity_type(entity_type)
    if normalized_type in {"concept", "object"} and compact in {
        normalize_entity_name_key(value) for value in ANATOMY_SURFACES
    }:
        return "ordinary body part is not a trackable story entity here"

    if normalized_type == "concept" and len(surface_text) <= 1:
        return "concept surface is too broad to track"

    return ""


def scene_tag_reason(
    *,
    surface: object,
    entity_type: object,
    role_in_unit: object = "",
    attributes_or_state: object = "",
    evidence: object = "",
) -> str:
    """Return a scene-tag type when an extracted entity is useful but not KG-worthy."""

    surface_text = str(surface or "").strip()
    if not surface_text:
        return ""
    compact = normalize_entity_name_key(surface_text)
    normalized_type = normalize_entity_type(entity_type)
    role_text = str(role_in_unit or "").lower()
    attributes_text = str(attributes_or_state or "").lower()
    evidence_text = str(evidence or "")
    role_and_attributes = f"{role_text} {attributes_text}"

    if compact in {normalize_entity_name_key(value) for value in LOW_VALUE_BACKGROUND_SURFACES}:
        if surface_text == "太阳" and any(term in evidence_text for term in ("太阳危机", "太阳氦闪")):
            return ""
        if any(term in surface_text for term in ("弹痕", "残垣", "杂草", "积水", "水管", "巴士", "建筑", "栅栏门")):
            return "background_element"
        if surface_text == "太阳":
            return "atmosphere"
        if normalized_type in {"character", "object"}:
            return "minor_visual_detail"
        return "background_element"

    if normalized_type == "occasion":
        if compact in {normalize_entity_name_key(value) for value in LOW_VALUE_OCCASION_SURFACES}:
            if any(term in surface_text for term in ("山火", "旱灾", "灭绝", "消失")):
                return "illustrative_example"
            return "generic_activity"
        if _looks_like_rhetorical_list_example(surface_text, evidence_text):
            return "illustrative_example"
        if not _occasion_has_tracking_signal(surface_text, role_text, attributes_text, evidence_text):
            return "generic_activity"

    if normalized_type in {"character", "object", "location", "concept"}:
        if "background" in role_text or "background" in attributes_text or "背景" in surface_text:
            return "background_element"
        if any(term in role_text for term in ("atmosphere", "mood")) or any(term in attributes_text for term in ("atmosphere", "mood")):
            return "atmosphere"

    if normalized_type == "character" and any(term in role_and_attributes for term in ("animal", "creature", "wildlife")):
        return "minor_visual_detail"

    if normalized_type in {"object", "location"}:
        if any(term in role_and_attributes for term in ("ruin", "wreckage", "debris", "environment", "scenery", "visual detail")):
            return "background_element"
        if any(term in surface_text for term in ("残垣", "弹痕", "废墟", "杂草", "积水")):
            return "background_element"

    return ""


def is_trackable_entity_candidate(
    *,
    surface: object,
    entity_type: object,
    canonical_hint: object = "",
    evidence: object = "",
) -> bool:
    return not entity_trackability_issue(
        surface=surface,
        entity_type=entity_type,
        canonical_hint=canonical_hint,
        evidence=evidence,
    )


def normalize_scene_tag_type(tag_type: object) -> str:
    value = str(tag_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    return value if value in ALLOWED_SCENE_TAG_TYPES else "other"


def _looks_like_rhetorical_list_example(surface: str, evidence: str) -> bool:
    if any(term in surface for term in ("山火", "旱灾", "物种", "城市")) and any(mark in evidence for mark in ("一场", "一次", "一个", "一座", "，", "、")):
        return True
    return False


def _occasion_has_tracking_signal(surface: str, role_text: str, attributes_text: str, evidence: str) -> bool:
    if len(surface) >= 6 and any(term in surface for term in ("计划", "行动", "工程", "会议", "实验", "事故", "灾难", "危机", "项目", "任务")):
        return True
    if len(surface) >= 4 and any(term in surface for term in ("氦闪", "事件", "事故", "灾难", "危机")):
        return True
    if any(term in role_text for term in ("named", "trackable", "recurring", "central", "planned", "operation")):
        return True
    if any(term in attributes_text for term in ("named", "trackable", "recurring", "central", "planned", "formal")):
        return True
    if any(term in evidence for term in ("计划", "行动", "工程", "会议", "实验", "事故", "灾难", "危机", "项目", "任务", "氦闪", "事件")) and len(surface) >= 4:
        return True
    return False


def _looks_like_supported_name_alias(surface: str, canonical_hint: str) -> bool:
    surface_clean = str(surface or "").strip()
    hint_clean = str(canonical_hint or "").strip()
    if not surface_clean or not hint_clean:
        return False
    if _contains_cjk(surface_clean) and _contains_cjk(hint_clean):
        return len(surface_clean) >= 2 and hint_clean.endswith(surface_clean)

    surface_parts = _latin_tokens(surface_clean)
    hint_parts = _latin_tokens(hint_clean)
    if not surface_parts or len(hint_parts) < 2:
        return False
    surface_key = " ".join(part.lower() for part in surface_parts)
    return any(surface_key == part.lower() for part in hint_parts)


def _contains_cjk(name: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", name or ""))


def _latin_tokens(name: str) -> list[str]:
    return [part for part in re.split(r"[\s,\-.]+", str(name or "").strip()) if part]
