from __future__ import annotations

from typing import Any

from dms.narrative_units import DEFAULT_UNIT_LABEL, DEFAULT_UNIT_TYPE


def unit_metadata(unit_payload: dict[str, Any] | None, unit_id: str) -> dict[str, Any]:
    if not unit_payload:
        return {
            "unit_type": DEFAULT_UNIT_TYPE,
            "unit_label": DEFAULT_UNIT_LABEL,
            "unit_order": None,
            "parent_unit_type": DEFAULT_UNIT_TYPE,
            "parent_unit_label": DEFAULT_UNIT_LABEL,
            "parent_unit_order": None,
            "parent_unit_id": unit_id,
            "chunk_id": unit_id,
            "chunk_index": 1,
            "chunk_count": 1,
            "unit_source_start": None,
            "unit_source_end": None,
            "unit_source_sha256": None,
            "chunk_unit_count": None,
            "max_chunk_units": None,
        }

    source_span = unit_payload.get("source_span") if isinstance(unit_payload.get("source_span"), dict) else {}
    unit_type = str(unit_payload.get("unit_type") or DEFAULT_UNIT_TYPE)
    unit_label = str(unit_payload.get("unit_label") or unit_type)
    return {
        "unit_type": unit_type,
        "unit_label": unit_label,
        "unit_order": optional_int(unit_payload.get("unit_order") or unit_payload.get("order")),
        "parent_unit_type": str(unit_payload.get("parent_unit_type") or unit_type),
        "parent_unit_label": str(unit_payload.get("parent_unit_label") or unit_label),
        "parent_unit_order": optional_int(unit_payload.get("parent_unit_order") or unit_payload.get("unit_order")),
        "parent_unit_id": str(unit_payload.get("parent_unit_id") or source_span.get("parent_unit_id") or unit_id),
        "chunk_id": str(unit_payload.get("chunk_id") or unit_payload.get("unit_id") or unit_id),
        "chunk_index": int_or_default(unit_payload.get("chunk_index"), 1),
        "chunk_count": int_or_default(unit_payload.get("chunk_count"), 1),
        "unit_source_start": optional_int(source_span.get("source_start")),
        "unit_source_end": optional_int(source_span.get("source_end")),
        "unit_source_sha256": source_span.get("source_sha256"),
        "chunk_unit_count": optional_int(source_span.get("chunk_unit_count")),
        "max_chunk_units": optional_int(source_span.get("max_chunk_units")),
    }


def parent_evidence_span(location: dict[str, Any], unit_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not unit_payload:
        return {
            "parent_evidence_start": None,
            "parent_evidence_end": None,
            "parent_source_sha256": None,
        }

    source_span = unit_payload.get("source_span") if isinstance(unit_payload.get("source_span"), dict) else {}
    source_start = optional_int(source_span.get("source_start"))
    evidence_start = optional_int(location.get("evidence_start"))
    evidence_end = optional_int(location.get("evidence_end"))
    if (
        location.get("evidence_source_field") == "content"
        and source_start is not None
        and evidence_start is not None
        and evidence_end is not None
    ):
        return {
            "parent_evidence_start": source_start + evidence_start,
            "parent_evidence_end": source_start + evidence_end,
            "parent_source_sha256": source_span.get("source_sha256"),
        }
    return {
        "parent_evidence_start": evidence_start,
        "parent_evidence_end": evidence_end,
        "parent_source_sha256": location.get("evidence_source_sha256"),
    }


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
