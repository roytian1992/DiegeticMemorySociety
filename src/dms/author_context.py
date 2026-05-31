from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.entity_types import normalize_entity_type


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
    return {
        "source_path": str(context_path),
        "entities": [entity for entity in entities if entity.get("canonical_name")],
        "raw_text": raw_text,
    }


def format_author_entity_context_for_prompt(context: dict[str, Any]) -> str:
    entities = context.get("entities") if isinstance(context, dict) else []
    if isinstance(entities, list) and entities:
        return json.dumps(
            {
                "author_defined_entities": entities,
                "policy": "author_defined_descriptions_are_baselines; current-unit extraction may only add evidence-supported supplements",
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
    return {
        "entity_id": _first_text(record, "entity_id", "id") or f"author_{entity_type}_{index:04d}",
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": sorted({alias for alias in [canonical_name, *aliases] if alias}),
        "author_description": description,
        "description": description,
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
