from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.context_kernel.schema import (
    AUTHORITY_VALUES,
    HISTORY_EVENTS,
    ITEM_TYPES,
    LINK_TYPES,
    PATCH_TYPES,
    SOURCE_TYPES,
    STATUS_VALUES,
    TEMPORAL_SCOPE_VALUES,
    VISIBILITY_VALUES,
)


def creative_context_json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "CreativeScope": _creative_scope_schema(),
        "SourceRecord": _source_record_schema(),
        "SourceUnit": _source_unit_schema(),
        "EvidenceRef": _evidence_ref_schema(),
        "CreativeContextItem": _creative_context_item_schema(),
        "ContextLink": _context_link_schema(),
        "EntityPatch": _entity_patch_schema(),
        "CreativeContextPacket": _creative_context_packet_schema(),
    }


def write_creative_context_json_schemas(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schemas = creative_context_json_schemas()
    written = {}
    for name, schema in schemas.items():
        path = output_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written[name] = str(path)
    enums_path = output_dir / "creative_context_enums.json"
    enums = {
        "source_types": sorted(SOURCE_TYPES),
        "item_types": sorted(ITEM_TYPES),
        "authority_values": sorted(AUTHORITY_VALUES),
        "status_values": sorted(STATUS_VALUES),
        "visibility_values": sorted(VISIBILITY_VALUES),
        "temporal_scope_values": sorted(TEMPORAL_SCOPE_VALUES),
        "history_events": sorted(HISTORY_EVENTS),
        "link_types": sorted(LINK_TYPES),
        "patch_types": sorted(PATCH_TYPES),
    }
    enums_path.write_text(json.dumps(enums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"schema_count": len(schemas), "schemas": written, "enums": str(enums_path)}


def _creative_scope_schema() -> dict[str, Any]:
    return _schema(
        "CreativeScope",
        {
            "project_id": _string(required=True),
            "artifact_id": _nullable_string(),
            "artifact_version": _nullable_string(),
            "conversation_id": _nullable_string(),
            "unit_id": _nullable_string(),
            "unit_type": _nullable_string(),
            "before_unit_id": _nullable_string(),
            "before_unit_order": _nullable_integer(),
            "character_id": _nullable_string(),
            "entity_ids": _string_array(),
            "task_mode": _nullable_string(),
            "source_type": _enum_or_null(SOURCE_TYPES),
        },
        required=["project_id"],
    )


def _source_record_schema() -> dict[str, Any]:
    return _schema(
        "SourceRecord",
        {
            "source_id": _string(required=True),
            "project_id": _string(required=True),
            "source_type": _enum(SOURCE_TYPES),
            "title": _string(),
            "status": _enum(STATUS_VALUES),
            "created_at": _nullable_string(),
            "updated_at": _nullable_string(),
            "metadata": _object(),
        },
        required=["source_id", "project_id", "source_type"],
    )


def _source_unit_schema() -> dict[str, Any]:
    return _schema(
        "SourceUnit",
        {
            "unit_id": _string(required=True),
            "source_id": _string(required=True),
            "project_id": _string(required=True),
            "source_type": _enum(SOURCE_TYPES),
            "unit_type": _string(required=True),
            "unit_order": _nullable_integer(),
            "speaker": _nullable_string(),
            "text": _string(),
            "start_offset": _nullable_integer(),
            "end_offset": _nullable_integer(),
            "metadata": _object(),
        },
        required=["unit_id", "source_id", "project_id", "source_type", "unit_type"],
    )


def _evidence_ref_schema() -> dict[str, Any]:
    return _schema(
        "EvidenceRef",
        {
            "evidence_id": _string(required=True),
            "item_id": _string(required=True),
            "source_id": _string(required=True),
            "unit_id": _nullable_string(),
            "text": _string(),
            "start_offset": _nullable_integer(),
            "end_offset": _nullable_integer(),
            "alignment_status": _string(),
            "metadata": _object(),
        },
        required=["evidence_id", "item_id", "source_id"],
    )


def _creative_context_item_schema() -> dict[str, Any]:
    return _schema(
        "CreativeContextItem",
        {
            "item_id": _string(required=True),
            "project_id": _string(required=True),
            "source_type": _enum(SOURCE_TYPES),
            "source_id": _string(required=True),
            "unit_id": _nullable_string(),
            "item_type": {"type": "string", "description": "Known values are listed in creative_context_enums.json."},
            "subject": _string(),
            "statement": _string(required=True),
            "entity_ids": _string_array(),
            "evidence_refs": {"type": "array", "items": {"$ref": "#/$defs/EvidenceRef"}},
            "authority": _enum(AUTHORITY_VALUES),
            "authority_score": _nullable_number(),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "status": _enum(STATUS_VALUES),
            "visibility": _enum(VISIBILITY_VALUES),
            "temporal_scope": _enum(TEMPORAL_SCOPE_VALUES),
            "embedding_text": _string(),
            "payload": _object(),
            "created_at": _nullable_string(),
            "updated_at": _nullable_string(),
        },
        required=["item_id", "project_id", "source_type", "source_id", "item_type", "statement"],
        defs={"EvidenceRef": _evidence_ref_schema()},
    )


def _context_link_schema() -> dict[str, Any]:
    return _schema(
        "ContextLink",
        {
            "link_id": _string(required=True),
            "source_item_id": _string(required=True),
            "target_item_id": _string(required=True),
            "link_type": _enum(LINK_TYPES),
            "created_by": _string(),
            "evidence": _string(),
            "created_at": _nullable_string(),
            "metadata": _object(),
        },
        required=["link_id", "source_item_id", "target_item_id", "link_type"],
    )


def _entity_patch_schema() -> dict[str, Any]:
    return _schema(
        "EntityPatch",
        {
            "patch_id": _string(required=True),
            "project_id": _string(required=True),
            "entity_id": _string(required=True),
            "source_item_id": _string(required=True),
            "patch_type": _enum(PATCH_TYPES),
            "target_field": _string(required=True),
            "patch_statement": _string(required=True),
            "authority": _enum(AUTHORITY_VALUES),
            "status": _enum(STATUS_VALUES),
            "applies_to": _string(),
            "created_at": _nullable_string(),
            "metadata": _object(),
        },
        required=["patch_id", "project_id", "entity_id", "source_item_id", "patch_type", "target_field", "patch_statement"],
    )


def _creative_context_packet_schema() -> dict[str, Any]:
    packet_item = {
        "type": "object",
        "required": ["item_id", "source_type", "statement"],
        "properties": {
            "item_id": _string(),
            "source_type": _string(),
            "source_role": _string(),
            "source_id": _nullable_string(),
            "unit_id": _nullable_string(),
            "item_type": _string(),
            "subject": _string(),
            "statement": _string(required=True),
            "entity_ids": _string_array(),
            "authority": _nullable_string(),
            "confidence": _nullable_number(),
            "status": _nullable_string(),
            "visibility": _nullable_string(),
            "temporal_scope": _nullable_string(),
            "score": _nullable_number(),
            "evidence": _string(),
            "payload": _object(),
        },
        "additionalProperties": True,
    }
    sections = {
        name: {"type": "array", "items": packet_item}
        for name in (
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
    }
    return _schema(
        "CreativeContextPacket",
        {
            "request": _object(),
            "retrieval_boundary": _object(),
            "task_state": _object(),
            "entities": {"type": "array", "items": _object()},
            **sections,
            "source_references": {"type": "array", "items": _object()},
            "trace": _object(),
        },
        required=["request", "retrieval_boundary", "task_state", "trace"],
    )


def _schema(title: str, properties: dict[str, Any], *, required: list[str] | None = None, defs: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def _enum(values: set[str]) -> dict[str, Any]:
    return {"type": "string", "enum": sorted(values)}


def _enum_or_null(values: set[str]) -> dict[str, Any]:
    return {"type": ["string", "null"], "enum": sorted(values) + [None]}


def _string(*, required: bool = False) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if required:
        schema["minLength"] = 1
    return schema


def _nullable_string() -> dict[str, Any]:
    return {"type": ["string", "null"]}


def _nullable_integer() -> dict[str, Any]:
    return {"type": ["integer", "null"]}


def _nullable_number() -> dict[str, Any]:
    return {"type": ["number", "null"]}


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _object() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": True}
