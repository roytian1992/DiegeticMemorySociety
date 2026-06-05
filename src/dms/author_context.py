from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.entity_types import normalize_entity_type

_AUTHOR_PROFILE_FIELDS = {
    "role": ("role", "role_in_story", "story_role"),
    "stable_traits": ("stable_traits", "traits", "personality", "personality_traits"),
    "speaking_style": ("speaking_style", "voice", "dialogue_style"),
    "values_or_motivations": ("values_or_motivations", "motivations", "values", "goals"),
    "behavior_constraints": ("behavior_constraints", "constraints", "hard_constraints"),
    "relationship_defaults": ("relationship_defaults", "relationships", "relationship_stances"),
    "private_goals": ("private_goals", "secrets", "hidden_agenda"),
    "notes": ("notes", "profile_notes"),
}

_INITIAL_STATE_FIELDS = {
    "beliefs": ("beliefs", "initial_beliefs"),
    "relationships": ("relationships", "initial_relationships"),
    "knowledge": ("knowledge", "private_knowledge", "initial_knowledge"),
    "goals": ("goals", "initial_goals"),
    "status": ("status", "initial_status", "starting_state"),
}


def load_author_entity_context(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"entities": [], "raw_text": ""}
    context_path = Path(path)
    if not context_path.exists():
        raise FileNotFoundError(f"Author entity context path not found: {context_path}")
    if context_path.is_dir():
        for name in ("author_entities.json", "entities.json", "entities.jsonl"):
            candidate = context_path / name
            if candidate.is_file():
                return load_author_entity_context(candidate)
        return {"entities": [], "raw_text": "\n".join(sorted(item.name for item in context_path.iterdir()))}

    raw_text = context_path.read_text(encoding="utf-8")
    records = _load_records(context_path, raw_text)
    entities = [_normalize_author_entity(record, index=index) for index, record in enumerate(records, start=1)]
    valid_entities = [entity for entity in entities if entity.get("canonical_name")]
    for entity in valid_entities:
        entity["source_path"] = str(context_path)
        sources = entity.setdefault("profile_sources", [])
        if entity.get("author_profile") or entity.get("initial_state"):
            sources.append({"source": "author_context", "path": str(context_path)})
            entity["profile_sources"] = _dedupe_json_list(sources)
    return {
        "source_path": str(context_path),
        "entities": valid_entities,
        "raw_text": raw_text,
    }


def format_author_entity_context_for_prompt(context: dict[str, Any]) -> str:
    entities = context.get("entities") if isinstance(context, dict) else []
    if isinstance(entities, list) and entities:
        return json.dumps(
            {
                "author_defined_entities": entities,
                "policy": (
                    "author_defined_descriptions_and_profiles_are_baselines; "
                    "profiles guide identity/simulation but are not current-unit textual evidence; "
                    "current-unit extraction may only add evidence-supported supplements"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    return str(context.get("raw_text") or "") if isinstance(context, dict) else ""


def author_entities_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    entities = context.get("entities") if isinstance(context, dict) else []
    return [entity for entity in entities if isinstance(entity, dict)]


def _load_records(path: Path, raw_text: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        return records
    if suffix == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("entities", "author_defined_entities", "characters"):
                if isinstance(payload.get(key), list):
                    return [item for item in payload[key] if isinstance(item, dict)]
            return [payload]
    return []


def _normalize_author_entity(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    canonical_name = _first_text(record, "canonical_name", "name", "surface", "entity", "id")
    entity_type = normalize_entity_type(_first_text(record, "entity_type", "type") or "character")
    aliases = _as_text_list(record.get("aliases") or record.get("alias") or record.get("names"))
    description = _first_text(record, "description", "bio", "profile", "summary")
    author_profile = _normalize_author_profile(record, description=description)
    initial_state = _normalize_initial_state(record)
    profile_policy = _normalize_profile_policy(record)
    return {
        "entity_id": _first_text(record, "entity_id", "id") or f"author_{entity_type}_{index:04d}",
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": sorted({alias for alias in [canonical_name, *aliases] if alias}),
        "author_description": description,
        "description": description,
        "author_profile": author_profile,
        "initial_state": initial_state,
        "profile_policy": profile_policy,
        "profile_sources": [{"source": "author_defined", "record_index": index}],
        "source": "author_defined",
    }


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_author_profile(record: dict[str, Any], *, description: str) -> dict[str, Any]:
    profile_sources: list[dict[str, Any]] = []
    for key in ("author_profile", "profile"):
        value = record.get(key)
        if isinstance(value, dict):
            profile_sources.append(value)
    profile_sources.append(record)

    profile: dict[str, Any] = {}
    if description:
        profile["description"] = description
    for canonical_key, aliases in _AUTHOR_PROFILE_FIELDS.items():
        value = _first_profile_value(profile_sources, aliases)
        if value not in ("", [], {}):
            profile[canonical_key] = value
    return profile


def _normalize_initial_state(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("initial_state") if isinstance(record.get("initial_state"), dict) else {}
    sources = [nested, record]
    initial_state: dict[str, Any] = {}
    for canonical_key, aliases in _INITIAL_STATE_FIELDS.items():
        value = _first_profile_value(sources, aliases)
        if value not in ("", [], {}):
            initial_state[canonical_key] = value
    return initial_state


def _normalize_profile_policy(record: dict[str, Any]) -> dict[str, Any]:
    policy = _clean_json_value(record.get("profile_policy")) if isinstance(record.get("profile_policy"), dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    priority = _first_text(policy, "priority") if policy else ""
    visibility = _first_text(policy, "visibility") if policy else ""
    policy.setdefault("priority", priority or _first_text(record, "profile_priority") or "author_locked")
    policy.setdefault("visibility", visibility or _first_text(record, "profile_visibility") or "author_guidance")
    return policy


def _first_profile_value(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key not in source:
                continue
            value = _clean_json_value(source.get(key))
            if value not in (None, "", [], {}):
                return value
    return ""


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        cleaned = [_clean_json_value(item) for item in value]
        return [item for item in cleaned if item not in (None, "", [], {})]
    if isinstance(value, dict):
        cleaned_dict = {str(key): _clean_json_value(item) for key, item in value.items()}
        return {key: item for key, item in cleaned_dict.items() if item not in (None, "", [], {})}
    if isinstance(value, (int, float, bool)):
        return value
    return None


def _dedupe_json_list(items: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped
