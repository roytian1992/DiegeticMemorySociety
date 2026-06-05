from __future__ import annotations

import re
from typing import Any


DEFAULT_UNIT_TYPE = "scene"
DEFAULT_UNIT_LABEL = "scene"


def normalize_unit_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or DEFAULT_UNIT_TYPE


def normalize_unit_label(value: Any, *, unit_type: str = DEFAULT_UNIT_TYPE) -> str:
    text = str(value or "").strip()
    return text or normalize_unit_type(unit_type) or DEFAULT_UNIT_LABEL


def narrative_unit_identity(
    *,
    scene_id: str,
    discourse_index: int | None = None,
    source_record_id: int | None = None,
    unit_type: str = DEFAULT_UNIT_TYPE,
    unit_label: str | None = None,
    unit_id: str | None = None,
    parent_unit_id: str | None = None,
    chunk_id: str | None = None,
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> dict[str, Any]:
    normalized_type = normalize_unit_type(unit_type)
    label = normalize_unit_label(unit_label, unit_type=normalized_type)
    resolved_unit_id = str(unit_id or scene_id)
    resolved_parent_id = str(parent_unit_id or scene_id)
    resolved_chunk_id = str(chunk_id or resolved_unit_id)
    order = _first_int(discourse_index, source_record_id, _order_from_id(scene_id))
    return {
        "unit_id": resolved_unit_id,
        "unit_type": normalized_type,
        "unit_label": label,
        "unit_order": order,
        "parent_unit_id": resolved_parent_id,
        "parent_unit_type": normalized_type,
        "parent_unit_label": label,
        "parent_unit_order": order,
        "chunk_id": resolved_chunk_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
    }


def narrative_unit_identity_from_record(record: Any) -> dict[str, Any]:
    return narrative_unit_identity(
        scene_id=str(getattr(record, "scene_id", "") or ""),
        discourse_index=getattr(record, "discourse_index", None),
        source_record_id=getattr(record, "source_record_id", None),
        unit_type=getattr(record, "unit_type", DEFAULT_UNIT_TYPE),
        unit_label=getattr(record, "unit_label", None),
        unit_id=getattr(record, "chunk_id", None) or getattr(record, "unit_id", None),
        parent_unit_id=getattr(record, "parent_unit_id", None),
        chunk_id=getattr(record, "chunk_id", None),
        chunk_index=getattr(record, "chunk_index", 1),
        chunk_count=getattr(record, "chunk_count", 1),
    )


def with_narrative_unit_type(record: dict[str, Any], *, unit_type: str, unit_label: str | None = None) -> dict[str, Any]:
    normalized_type = normalize_unit_type(unit_type)
    label = normalize_unit_label(unit_label, unit_type=normalized_type)
    return {
        **record,
        "unit_type": normalized_type,
        "unit_label": label,
        "parent_unit_type": normalized_type,
        "parent_unit_label": label,
    }


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _order_from_id(value: Any) -> int | None:
    for part in str(value or "").split("_"):
        if part.isdigit():
            return int(part)
    return None
