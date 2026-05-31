from __future__ import annotations

from typing import Any

from dms.entity_types import (
    embedded_alnum_code_key,
    entity_trackability_issue,
    extract_countdown_core_entity,
    normalize_entity_canonical_hint,
    normalize_entity_name_key,
    normalize_entity_type,
    normalize_scene_tag_type,
    scene_tag_reason,
    SCENE_TAG_ENTITY_TYPE_ALIASES,
)


def sanitize_kg_entity_output(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"unit_id": "", "entity_mentions": [], "unresolved_mentions": []}

    sanitized = dict(data)
    accepted: list[dict[str, Any]] = []
    scene_tags = [item for item in _as_list(data.get("scene_tags")) if isinstance(item, dict)]
    unresolved = [item for item in _as_list(data.get("unresolved_mentions")) if isinstance(item, dict)]

    for item in _as_list(data.get("entity_mentions")):
        if not isinstance(item, dict):
            continue
        surface = str(item.get("surface", "")).strip()
        raw_entity_type = str(item.get("entity_type") or "").strip().lower().replace(" ", "_").replace("-", "_")
        if raw_entity_type in SCENE_TAG_ENTITY_TYPE_ALIASES:
            scene_tags.append(
                {
                    "surface": surface,
                    "tag_type": normalize_scene_tag_type(raw_entity_type),
                    "reason": "model emitted a scene-tag type in entity_mentions; demoted from persistent entity",
                    "evidence": str(item.get("evidence", "")),
                }
            )
            continue
        if raw_entity_type in {"unresolved", "unknown", "ambiguous", "unclear"}:
            unresolved.append(
                {
                    "surface": surface,
                    "reason": "model marked this mention as unresolved",
                    "evidence": str(item.get("evidence", "")),
                }
            )
            continue
        entity_type = normalize_entity_type(item.get("entity_type"))
        evidence = str(item.get("evidence", ""))
        countdown_core = extract_countdown_core_entity(surface)
        if countdown_core:
            surface = countdown_core
            entity_type = "occasion"
        canonical_hint = normalize_entity_canonical_hint(
            surface=surface,
            entity_type=entity_type,
            canonical_hint=item.get("canonical_hint", ""),
            evidence=evidence,
        )
        tag_type = scene_tag_reason(
            surface=surface,
            entity_type=entity_type,
            role_in_unit=item.get("role_in_unit") or item.get("role_in_scene", ""),
            attributes_or_state=item.get("attributes_or_state", ""),
            evidence=evidence,
        )
        if tag_type:
            scene_tags.append(
                {
                    "surface": surface,
                    "tag_type": tag_type,
                    "reason": "useful local context but not important enough for the persistent knowledge graph",
                    "evidence": evidence,
                }
            )
            continue
        issue = entity_trackability_issue(
            surface=surface,
            entity_type=entity_type,
            canonical_hint=canonical_hint,
            evidence=evidence,
        )
        if issue:
            unresolved.append({"surface": surface, "reason": issue, "evidence": evidence})
            continue
        record = dict(item)
        record["surface"] = surface
        record["entity_type"] = entity_type
        record["canonical_hint"] = canonical_hint
        record["description"] = str(item.get("description") or "").strip()
        accepted.append(record)

    sanitized["entity_mentions"] = accepted
    sanitized["scene_tags"] = scene_tags
    sanitized["unresolved_mentions"] = unresolved
    return sanitized


def build_entity_candidate_index(extracted_candidates_or_data: Any) -> dict[str, dict[str, Any]]:
    data = _kg_entity_data(extracted_candidates_or_data)
    sanitized = sanitize_kg_entity_output(data)
    index: dict[str, dict[str, Any]] = {}
    for item in _as_list(sanitized.get("entity_mentions")):
        if not isinstance(item, dict):
            continue
        surface = str(item.get("surface") or "").strip()
        entity_type = normalize_entity_type(item.get("entity_type"))
        canonical_hint = str(item.get("canonical_hint") or "").strip()
        canonical_name = canonical_hint or surface
        if not surface:
            continue
        candidate = {
            "surface": surface,
            "canonical_name": canonical_name,
            "entity_type": entity_type,
            "evidence": item.get("evidence", ""),
            "description": item.get("description", ""),
            "role_in_unit": item.get("role_in_unit") or item.get("role_in_scene", ""),
        }
        for name in _candidate_aliases(surface, canonical_name, entity_type):
            key = normalize_entity_name_key(name)
            if key:
                index.setdefault(key, candidate)
    return index


def align_entity_to_candidates(
    *,
    entity: object,
    entity_type: object,
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    entity_name = str(entity or "").strip()
    normalized_type = normalize_entity_type(entity_type)
    countdown_core = extract_countdown_core_entity(entity_name)
    if countdown_core:
        entity_name = countdown_core
        normalized_type = "occasion"

    if not candidate_index:
        issue = entity_trackability_issue(surface=entity_name, entity_type=normalized_type)
        if issue:
            return None
        return {
            "surface": entity_name,
            "canonical_name": entity_name,
            "entity_type": normalized_type,
            "evidence": "",
            "role_in_unit": "",
        }

    for name in _candidate_aliases(entity_name, entity_name, normalized_type):
        match = candidate_index.get(normalize_entity_name_key(name))
        if match:
            return match
    issue = entity_trackability_issue(surface=entity_name, entity_type=normalized_type)
    if issue:
        return None
    return candidate_index.get(normalize_entity_name_key(entity_name))


def _kg_entity_data(extracted_candidates_or_data: Any) -> Any:
    if not isinstance(extracted_candidates_or_data, dict):
        return {}
    if "entity_mentions" in extracted_candidates_or_data:
        return extracted_candidates_or_data

    kg_payload = extracted_candidates_or_data.get("kg_entity_mentions")
    if isinstance(kg_payload, dict):
        return kg_payload.get("data") if isinstance(kg_payload.get("data"), dict) else {}
    return {}


def _candidate_aliases(surface: str, canonical_name: str, entity_type: str) -> set[str]:
    names = {surface, canonical_name}
    code = embedded_alnum_code_key(surface) or embedded_alnum_code_key(canonical_name)
    if code:
        names.add(code.upper())
        names.add(code.lower())
    if entity_type == "character":
        for name in list(names):
            clean = str(name or "").strip()
            if _contains_cjk(clean) and 3 <= len(clean) <= 4:
                names.add(clean[1:])
            else:
                parts = [part for part in clean.replace(",", " ").replace(".", " ").replace("-", " ").split() if part]
                if len(parts) >= 2:
                    names.update(parts)
                    names.add(" ".join(reversed(parts)))
    return {name for name in names if str(name or "").strip()}


def _contains_cjk(name: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in name or "")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
