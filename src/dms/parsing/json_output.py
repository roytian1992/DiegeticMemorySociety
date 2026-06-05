from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from dms.entity_types import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_SCENE_TAG_TYPES,
    is_supported_entity_type_label,
    normalize_entity_type,
    normalize_scene_tag_type,
)
from dms.source_evidence import require_aligned_evidence


@dataclass(frozen=True)
class JSONParseResult:
    ok: bool
    data: Any
    error: str | None = None


def extract_json_value(text: str) -> JSONParseResult:
    raw = (text or "").strip()
    if not raw:
        return JSONParseResult(ok=False, data=None, error="empty output")

    candidates = [raw]
    fenced = _extract_fenced_json(raw)
    if fenced:
        candidates.insert(0, fenced)

    sliced = _slice_outer_json(raw)
    if sliced and sliced not in candidates:
        candidates.append(sliced)

    errors: list[str] = []
    for candidate in candidates:
        try:
            return JSONParseResult(ok=True, data=json.loads(candidate), error=None)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
        repaired = _repair_redundant_closing_braces(candidate)
        if repaired != candidate:
            try:
                return JSONParseResult(ok=True, data=json.loads(repaired), error=None)
            except json.JSONDecodeError as exc:
                errors.append(f"repair failed: {exc}")
        repair_result = _try_json_repair(candidate)
        if repair_result.ok:
            return repair_result
        if repair_result.error:
            errors.append(repair_result.error)

    return JSONParseResult(ok=False, data=None, error="; ".join(errors[:3]))


def _try_json_repair(text: str) -> JSONParseResult:
    try:
        from json_repair import repair_json
    except ImportError:
        return JSONParseResult(ok=False, data=None, error=None)
    try:
        return JSONParseResult(ok=True, data=repair_json(text, return_objects=True), error=None)
    except Exception as exc:  # noqa: BLE001 - parser fallback must preserve the original parse path.
        return JSONParseResult(ok=False, data=None, error=f"json_repair failed: {exc}")


def _repair_redundant_closing_braces(text: str) -> str:
    current = text
    while True:
        next_text = _remove_one_redundant_closing_brace(current)
        if next_text == current:
            return current
        try:
            json.loads(next_text)
            return next_text
        except json.JSONDecodeError:
            current = next_text


def _remove_one_redundant_closing_brace(text: str) -> str:
    in_string = False
    escaped = False
    stack: list[tuple[str, int]] = []
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append((char, index))
            continue
        if char not in "}]":
            continue
        expected = "{" if char == "}" else "["
        if stack and stack[-1][0] == expected:
            stack.pop()
            continue
        if _looks_like_redundant_closer_context(text, index):
            return text[:index] + text[index + 1 :]
    return text


def _looks_like_redundant_closer_context(text: str, index: int) -> bool:
    left = text[:index].rstrip()
    right = text[index + 1 :].lstrip()
    return bool(left and left[-1] in "}]") and bool(right and right[0] in ",]}")


def validate_scene_inventory(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["scene inventory output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    setting = data.get("setting")
    if not isinstance(setting, dict):
        errors.append("setting must be an object")
    else:
        _validate_setting(errors, setting)

    for key in ("stated_facts", "open_questions"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    for key in ("characters", "objects", "entity_mentions"):
        if key in data:
            errors.append(f"{key} must not be emitted by scene_inventory; use kg_entity_mentions")

    return errors


def validate_scene_summary(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["scene summary output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)
    _require_top_level_string_fields(errors, data, ("summary", "retrieval_text"))

    for key in ("salient_points", "continuity_hooks"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")
        else:
            for index, item in enumerate(data[key], start=1):
                if not isinstance(item, str):
                    errors.append(f"{key}[{index}] must be a string")

    for key in (
        "seed_entities",
        "retrieval_needs",
        "query_plan",
        "entity_mentions",
        "episodic_memories",
        "relationship_observations",
    ):
        if key in data:
            errors.append(f"{key} must not be emitted by scene_summary")

    return errors


def validate_kg_entity_mentions(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["kg entity mentions output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    for key in ("entity_mentions", "unresolved_mentions"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    if "scene_tags" in data and not isinstance(data.get("scene_tags"), list):
        errors.append("scene_tags must be a list")

    for index, item in enumerate(
        data.get("entity_mentions") if isinstance(data.get("entity_mentions"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"entity_mentions[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "entity_mentions",
            index,
            ("surface", "entity_type", "canonical_hint", "attributes_or_state", "evidence"),
        )
        if "description" in item and not isinstance(item.get("description"), str):
            errors.append(f"entity_mentions[{index}].description must be a string")
        _validate_entity_type(errors, item, "entity_mentions", index, "entity_type")
        if not _has_string_alias(item, ("role_in_unit", "role_in_scene")):
            errors.append(f"entity_mentions[{index}].role_in_unit must be a string")

    for index, item in enumerate(
        data.get("unresolved_mentions") if isinstance(data.get("unresolved_mentions"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"unresolved_mentions[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "unresolved_mentions", index, ("surface", "reason", "evidence"))

    for index, item in enumerate(
        data.get("scene_tags") if isinstance(data.get("scene_tags"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"scene_tags[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "scene_tags", index, ("surface", "tag_type", "reason", "evidence"))
        if normalize_scene_tag_type(item.get("tag_type")) not in ALLOWED_SCENE_TAG_TYPES:
            errors.append(f"scene_tags[{index}].tag_type must be one of {sorted(ALLOWED_SCENE_TAG_TYPES)}")

    return errors


def validate_scene_event_candidates(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["scene event candidates output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    for key in ("events", "knowledge_transfers", "state_changes", "thread_candidates"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    for index, item in enumerate(data.get("events") if isinstance(data.get("events"), list) else [], start=1):
        if not isinstance(item, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "events", index, ("event_id_hint", "summary", "location", "event_type", "evidence"))
        if not isinstance(item.get("participants"), list):
            errors.append(f"events[{index}].participants must be a list")

    for index, item in enumerate(
        data.get("knowledge_transfers") if isinstance(data.get("knowledge_transfers"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"knowledge_transfers[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "knowledge_transfers",
            index,
            ("source", "receiver", "content", "epistemic_status", "evidence"),
        )

    for index, item in enumerate(
        data.get("state_changes") if isinstance(data.get("state_changes"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"state_changes[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "state_changes", index, ("entity", "before", "after", "evidence"))

    for index, item in enumerate(
        data.get("thread_candidates") if isinstance(data.get("thread_candidates"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"thread_candidates[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "thread_candidates", index, ("thread_type", "summary", "evidence"))

    return errors


def validate_visibility_notes(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["visibility notes output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    for key in ("visibility_records", "hidden_or_future_sensitive_items"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    for index, item in enumerate(
        data.get("visibility_records") if isinstance(data.get("visibility_records"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"visibility_records[{index}] must be an object")
            continue
        _require_string_fields(errors, item, "visibility_records", index, ("fact_or_event", "character", "visibility", "evidence"))

    for index, item in enumerate(
        data.get("hidden_or_future_sensitive_items")
        if isinstance(data.get("hidden_or_future_sensitive_items"), list)
        else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"hidden_or_future_sensitive_items[{index}] must be an object")
            continue
        if not _has_any_string_field(item, ("item", "fact_or_event", "summary", "content")):
            errors.append(
                f"hidden_or_future_sensitive_items[{index}] must include a string item, fact_or_event, summary, or content"
            )
        _require_string_fields(errors, item, "hidden_or_future_sensitive_items", index, ("reason", "evidence"))
        if not isinstance(item.get("hidden_from"), list):
            errors.append(f"hidden_or_future_sensitive_items[{index}].hidden_from must be a list")

    return errors


def validate_temporal_extraction(data: Any, *, expected_scene_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["temporal extraction output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    for key in ("temporal_events", "temporal_relations"):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")
    if "temporal_warnings" in data and not isinstance(data.get("temporal_warnings"), list):
        errors.append("temporal_warnings must be a list")

    scene_index = data.get("scene_temporal_index")
    if scene_index is not None and not isinstance(scene_index, dict):
        errors.append("scene_temporal_index must be an object when present")

    for index, item in enumerate(
        data.get("temporal_events") if isinstance(data.get("temporal_events"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"temporal_events[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "temporal_events",
            index,
            ("event_id", "summary", "event_time_mode", "evidence"),
        )
        _validate_optional_enum(
            errors,
            item,
            "temporal_events",
            index,
            "event_track",
            {"plot", "context", "forecast", "memory", "hypothetical", "unknown"},
        )
        _validate_optional_enum(
            errors,
            item,
            "temporal_events",
            index,
            "event_time_mode",
            {"present_scene", "past_recalled", "future_anticipated", "habitual", "hypothetical", "uncertain"},
        )
        if not isinstance(item.get("participants"), list):
            errors.append(f"temporal_events[{index}].participants must be a list")
        if "confidence" in item and not isinstance(item.get("confidence"), (int, float, str)):
            errors.append(f"temporal_events[{index}].confidence must be numeric or string")

    for index, item in enumerate(
        data.get("temporal_relations") if isinstance(data.get("temporal_relations"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"temporal_relations[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "temporal_relations",
            index,
            ("source_event_id", "target_event_id", "relation_type", "evidence"),
        )
        _validate_optional_enum(
            errors,
            item,
            "temporal_relations",
            index,
            "relation_type",
            {
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
            },
        )
        if "confidence" in item and not isinstance(item.get("confidence"), (int, float, str)):
            errors.append(f"temporal_relations[{index}].confidence must be numeric or string")

    return errors


def validate_episodic_memories(
    data: Any,
    *,
    expected_scene_id: str | None = None,
    source_unit: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["episodic memories output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    for key in ("episodic_memories",):
        if key not in data:
            errors.append(f"{key} is required")
        elif not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")

    for index, item in enumerate(
        data.get("episodic_memories") if isinstance(data.get("episodic_memories"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"episodic_memories[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "episodic_memories",
            index,
            ("memory_id_hint", "timeline_label", "memory_type", "summary", "evidence"),
        )
        if source_unit is not None:
            require_aligned_evidence(errors, item.get("evidence"), source_unit, label=f"episodic_memories[{index}]")
        if not isinstance(item.get("sequence_index"), int):
            errors.append(f"episodic_memories[{index}].sequence_index must be an integer")
        if not isinstance(item.get("entity_links"), list):
            errors.append(f"episodic_memories[{index}].entity_links must be a list")
        for link_index, link in enumerate(item.get("entity_links") if isinstance(item.get("entity_links"), list) else [], start=1):
            if not isinstance(link, dict):
                errors.append(f"episodic_memories[{index}].entity_links[{link_index}] must be an object")
                continue
            _require_string_fields(
                errors,
                link,
                f"episodic_memories[{index}].entity_links",
                link_index,
                ("entity", "entity_type", "link_role", "evidence"),
            )
            _validate_entity_type(
                errors,
                link,
                f"episodic_memories[{index}].entity_links",
                link_index,
                "entity_type",
            )
            if source_unit is not None:
                require_aligned_evidence(
                    errors,
                    link.get("evidence"),
                    source_unit,
                    label=f"episodic_memories[{index}].entity_links[{link_index}]",
                )

    if "relationship_observations" in data:
        errors.append("relationship_observations must not be emitted by episodic_memories; use durable_relationships")

    return errors


def validate_durable_relationships(
    data: Any,
    *,
    expected_scene_id: str | None = None,
    source_unit: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["durable relationships output must be a JSON object"]

    _validate_unit_id(errors, data, expected_scene_id)

    if "relationship_observations" not in data:
        errors.append("relationship_observations is required")
    elif not isinstance(data.get("relationship_observations"), list):
        errors.append("relationship_observations must be a list")

    for index, item in enumerate(
        data.get("relationship_observations") if isinstance(data.get("relationship_observations"), list) else [],
        start=1,
    ):
        if not isinstance(item, dict):
            errors.append(f"relationship_observations[{index}] must be an object")
            continue
        _require_string_fields(
            errors,
            item,
            "relationship_observations",
            index,
            ("source_entity", "target_entity", "relation_type", "status_or_change", "evidence"),
        )
        if source_unit is not None:
            require_aligned_evidence(errors, item.get("evidence"), source_unit, label=f"relationship_observations[{index}]")

    return errors


def _require_string_fields(
    errors: list[str],
    item: dict[str, Any],
    label: str,
    index: int,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        if not isinstance(item.get(field), str):
            errors.append(f"{label}[{index}].{field} must be a string")


def _require_top_level_string_fields(errors: list[str], data: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if not isinstance(data.get(field), str):
            errors.append(f"{field} must be a string")


def _validate_entity_type(
    errors: list[str],
    item: dict[str, Any],
    label: str,
    index: int,
    field: str,
) -> None:
    value = item.get(field)
    if not isinstance(value, str):
        return
    normalized = normalize_entity_type(value)
    if not is_supported_entity_type_label(value):
        allowed = ", ".join(sorted(ALLOWED_ENTITY_TYPES))
        errors.append(f"{label}[{index}].{field} must be one of: {allowed}")


def _validate_optional_enum(
    errors: list[str],
    item: dict[str, Any],
    label: str,
    index: int,
    field: str,
    allowed: set[str],
) -> None:
    value = item.get(field)
    if value is None:
        return
    if not isinstance(value, str) or value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        errors.append(f"{label}[{index}].{field} must be one of: {allowed_text}")


def _validate_unit_id(errors: list[str], data: dict[str, Any], expected_unit_id: str | None) -> None:
    unit_id = data.get("unit_id") or data.get("scene_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        errors.append("unit_id must be a non-empty string")
    elif expected_unit_id and unit_id != expected_unit_id:
        errors.append(f"unit_id mismatch: expected {expected_unit_id}, got {unit_id}")


def _validate_setting(errors: list[str], setting: dict[str, Any]) -> None:
    if not isinstance(setting.get("location"), str):
        errors.append("setting.location must be a string")
    if not _has_string_alias(setting, ("time_hint", "time_of_day")):
        errors.append("setting.time_hint must be a string")
    if not _has_string_alias(setting, ("spatial_context", "interior_exterior")):
        errors.append("setting.spatial_context must be a string")


def _has_string_alias(item: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return any(field in item and isinstance(item.get(field), str) for field in fields)


def _has_any_string_field(item: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return any(isinstance(item.get(field), str) for field in fields)


def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _slice_outer_json(text: str) -> str | None:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        return None
    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    end = text.rfind(closing)
    if end <= start:
        return None
    return text[start : end + 1].strip()
