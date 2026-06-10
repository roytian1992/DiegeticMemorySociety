from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_creative_context_packet(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def select_creative_context_notes(
    packet: dict[str, Any],
    entity: dict[str, Any],
    *,
    limit: int = 8,
) -> list[str]:
    if limit <= 0 or not packet:
        return []
    tokens = _entity_tokens(entity)
    sections = (
        "conversation_guidance",
        "artifact_memory",
        "character_visible_knowledge",
        "relationship_context",
        "timeline_context",
        "external_reference_context",
        "style_guidance",
        "open_questions",
        "simulation_context",
        "entity_patch_context",
    )
    notes: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for item in packet.get(section) or []:
            if not isinstance(item, dict):
                continue
            if tokens and not _item_matches_entity(item, tokens):
                continue
            note = compact_creative_context_item(section, item)
            if note and note not in seen:
                notes.append(note)
                seen.add(note)
            if len(notes) >= limit:
                return notes
    return notes


def compact_creative_context_item(section: str, item: dict[str, Any]) -> str:
    statement = str(item.get("statement") or "").strip()
    if not statement:
        return ""
    item_id = str(item.get("item_id") or "").strip()
    source_type = str(item.get("source_type") or item.get("source_role") or "").strip()
    status = str(item.get("status") or "").strip()
    visibility = str(item.get("visibility") or "").strip()
    subject = str(item.get("subject") or "").strip()
    authority = str(item.get("authority") or "").strip()
    prefix_parts = [part for part in (section, source_type, status, authority) if part]
    prefix = " / ".join(prefix_parts)
    subject_text = f"{subject}: " if subject else ""
    suffix_parts = []
    if visibility:
        suffix_parts.append(f"visibility={visibility}")
    if item_id:
        suffix_parts.append(f"id={item_id}")
    suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
    return f"{prefix}: {subject_text}{statement}{suffix}".strip()


def _entity_tokens(entity: dict[str, Any]) -> set[str]:
    raw_values = [
        entity.get("entity_id"),
        entity.get("canonical_name"),
        entity.get("subject"),
        *(entity.get("aliases") or []),
    ]
    tokens = {str(value).strip().lower() for value in raw_values if str(value or "").strip()}
    for token in list(tokens):
        if ":" in token:
            tokens.add(token.split(":", 1)[-1])
    return tokens


def _item_matches_entity(item: dict[str, Any], tokens: set[str]) -> bool:
    item_tokens = {
        str(value).strip().lower()
        for value in (
            item.get("item_id"),
            item.get("subject"),
            *(item.get("entity_ids") or []),
        )
        if str(value or "").strip()
    }
    for token in list(item_tokens):
        if ":" in token:
            item_tokens.add(token.split(":", 1)[-1])
    if item_tokens.intersection(tokens):
        return True
    statement = str(item.get("statement") or "").lower()
    return any(token and token in statement for token in tokens)
