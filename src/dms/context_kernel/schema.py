from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


SOURCE_TYPES = {
    "conversation",
    "narrative_artifact",
    "external_reference",
    "simulation",
}

ITEM_TYPES = {
    "user_preference",
    "creative_decision",
    "story_constraint",
    "character_intent",
    "worldbuilding_note",
    "style_preference",
    "open_question",
    "rejected_option",
    "task_state",
    "correction",
    "event",
    "fact",
    "relation",
    "state",
    "summary",
    "visibility",
    "timeline_event",
    "world_bible",
    "character_profile",
    "relationship_fact",
    "timeline_doc",
    "location_doc",
    "organization_fact",
    "style_guide",
    "author_note",
    "notes",
    "simulation_hypothesis",
}

AUTHORITY_VALUES = {
    "user_explicit",
    "user_confirmed",
    "artifact_canonical",
    "artifact_active",
    "external_source",
    "model_inferred",
    "simulation_hypothesis",
    "system_imported",
}

STATUS_VALUES = {
    "extracted",
    "tentative",
    "active",
    "canonical",
    "superseded",
    "rejected",
    "archived",
    "deleted",
}

VISIBILITY_VALUES = {
    "author_only",
    "character_visible",
    "world_public",
    "style_only",
    "character_private",
    "revealed_by_story",
    "unknown",
}

TEMPORAL_SCOPE_VALUES = {
    "temporal_episode",
    "atemporal_fact",
    "durable_state",
    "uncertain",
    "not_applicable",
}

HISTORY_EVENTS = {
    "ADD",
    "UPDATE",
    "DELETE",
    "PATCH",
    "PROMOTE",
    "SUPERSEDE",
    "REJECT",
    "ARCHIVE",
    "LINK",
}

LINK_TYPES = {
    "promotes",
    "supersedes",
    "supports",
    "contradicts",
    "constrains",
    "derived_from",
    "cites",
}

PATCH_TYPES = {
    "add",
    "revise",
    "constrain",
    "deprecate",
    "rename",
    "merge",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    digest = hashlib.sha1(
        "\n".join(_stable_string(part) for part in parts).encode("utf-8")
    ).hexdigest()[:length]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class CreativeScope:
    project_id: str
    artifact_id: str | None = None
    artifact_version: str | None = None
    conversation_id: str | None = None
    unit_id: str | None = None
    unit_type: str | None = None
    before_unit_id: str | None = None
    before_unit_order: int | None = None
    character_id: str | None = None
    entity_ids: tuple[str, ...] = ()
    task_mode: str | None = None
    source_type: str | None = None

    def normalized(self) -> "CreativeScope":
        return CreativeScope(
            project_id=_required_text(self.project_id, "project_id"),
            artifact_id=_optional_text(self.artifact_id),
            artifact_version=_optional_text(self.artifact_version),
            conversation_id=_optional_text(self.conversation_id),
            unit_id=_optional_text(self.unit_id),
            unit_type=_optional_text(self.unit_type),
            before_unit_id=_optional_text(self.before_unit_id),
            before_unit_order=self.before_unit_order,
            character_id=_optional_text(self.character_id),
            entity_ids=tuple(_dedupe_texts(self.entity_ids)),
            task_mode=_optional_text(self.task_mode),
            source_type=_optional_source_type(self.source_type),
        )

    def model_dump(self) -> dict[str, Any]:
        return _drop_none(asdict(self.normalized()))


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    project_id: str
    source_type: str
    title: str = ""
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "SourceRecord":
        created_at = self.created_at or utc_now()
        return SourceRecord(
            source_id=_required_text(self.source_id, "source_id"),
            project_id=_required_text(self.project_id, "project_id"),
            source_type=_required_choice(self.source_type, SOURCE_TYPES, "source_type"),
            title=str(self.title or "").strip(),
            status=_required_choice(self.status, STATUS_VALUES, "status"),
            created_at=created_at,
            updated_at=self.updated_at or created_at,
            metadata=dict(self.metadata or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    source_id: str
    project_id: str
    source_type: str
    unit_type: str
    unit_order: int | None = None
    speaker: str | None = None
    text: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "SourceUnit":
        return SourceUnit(
            unit_id=_required_text(self.unit_id, "unit_id"),
            source_id=_required_text(self.source_id, "source_id"),
            project_id=_required_text(self.project_id, "project_id"),
            source_type=_required_choice(self.source_type, SOURCE_TYPES, "source_type"),
            unit_type=_required_text(self.unit_type, "unit_type"),
            unit_order=self.unit_order,
            speaker=_optional_text(self.speaker),
            text=str(self.text or ""),
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            metadata=dict(self.metadata or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return _drop_none(asdict(self.normalized()))


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    item_id: str
    source_id: str
    unit_id: str | None = None
    text: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    alignment_status: str = "provided"
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "EvidenceRef":
        return EvidenceRef(
            evidence_id=_required_text(self.evidence_id, "evidence_id"),
            item_id=_required_text(self.item_id, "item_id"),
            source_id=_required_text(self.source_id, "source_id"),
            unit_id=_optional_text(self.unit_id),
            text=str(self.text or ""),
            start_offset=self.start_offset,
            end_offset=self.end_offset,
            alignment_status=str(self.alignment_status or "provided").strip(),
            metadata=dict(self.metadata or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return _drop_none(asdict(self.normalized()))


@dataclass(frozen=True)
class CreativeContextItem:
    item_id: str
    project_id: str
    source_type: str
    source_id: str
    item_type: str
    statement: str
    unit_id: str | None = None
    subject: str = ""
    entity_ids: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    authority: str = "model_inferred"
    authority_score: float | None = None
    confidence: float = 0.0
    status: str = "active"
    visibility: str = "author_only"
    temporal_scope: str = "not_applicable"
    embedding_text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def normalized(self) -> "CreativeContextItem":
        statement = _required_text(self.statement, "statement")
        created_at = self.created_at or utc_now()
        embedding_text = self.embedding_text.strip() if self.embedding_text else default_embedding_text(
            source_type=self.source_type,
            item_type=self.item_type,
            subject=self.subject,
            statement=statement,
            status=self.status,
        )
        return CreativeContextItem(
            item_id=_required_text(self.item_id, "item_id"),
            project_id=_required_text(self.project_id, "project_id"),
            source_type=_required_choice(self.source_type, SOURCE_TYPES, "source_type"),
            source_id=_required_text(self.source_id, "source_id"),
            unit_id=_optional_text(self.unit_id),
            item_type=_normalize_item_type(self.item_type),
            subject=str(self.subject or "").strip(),
            statement=statement,
            entity_ids=tuple(_dedupe_texts(self.entity_ids)),
            evidence_refs=tuple(ref.normalized() for ref in self.evidence_refs),
            authority=_required_choice(self.authority, AUTHORITY_VALUES, "authority"),
            authority_score=self.authority_score,
            confidence=float(self.confidence or 0.0),
            status=_required_choice(self.status, STATUS_VALUES, "status"),
            visibility=_required_choice(self.visibility, VISIBILITY_VALUES, "visibility"),
            temporal_scope=_required_choice(self.temporal_scope, TEMPORAL_SCOPE_VALUES, "temporal_scope"),
            embedding_text=embedding_text,
            payload=dict(self.payload or {}),
            created_at=created_at,
            updated_at=self.updated_at or created_at,
        )

    def model_dump(self) -> dict[str, Any]:
        item = self.normalized()
        data = asdict(item)
        data["evidence_refs"] = [ref.model_dump() for ref in item.evidence_refs]
        return _drop_none(data)


@dataclass(frozen=True)
class ContextLink:
    link_id: str
    source_item_id: str
    target_item_id: str
    link_type: str
    created_by: str = "system"
    evidence: str = ""
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ContextLink":
        return ContextLink(
            link_id=_required_text(self.link_id, "link_id"),
            source_item_id=_required_text(self.source_item_id, "source_item_id"),
            target_item_id=_required_text(self.target_item_id, "target_item_id"),
            link_type=_required_choice(self.link_type, LINK_TYPES, "link_type"),
            created_by=str(self.created_by or "system").strip(),
            evidence=str(self.evidence or ""),
            created_at=self.created_at or utc_now(),
            metadata=dict(self.metadata or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return _drop_none(asdict(self.normalized()))


@dataclass(frozen=True)
class EntityPatch:
    patch_id: str
    project_id: str
    entity_id: str
    source_item_id: str
    patch_type: str
    target_field: str
    patch_statement: str
    authority: str = "user_explicit"
    status: str = "active"
    applies_to: str = "project"
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "EntityPatch":
        return EntityPatch(
            patch_id=_required_text(self.patch_id, "patch_id"),
            project_id=_required_text(self.project_id, "project_id"),
            entity_id=_required_text(self.entity_id, "entity_id"),
            source_item_id=_required_text(self.source_item_id, "source_item_id"),
            patch_type=_required_choice(self.patch_type, PATCH_TYPES, "patch_type"),
            target_field=_required_text(self.target_field, "target_field"),
            patch_statement=_required_text(self.patch_statement, "patch_statement"),
            authority=_required_choice(self.authority, AUTHORITY_VALUES, "authority"),
            status=_required_choice(self.status, STATUS_VALUES, "status"),
            applies_to=str(self.applies_to or "project").strip(),
            created_at=self.created_at or utc_now(),
            metadata=dict(self.metadata or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return _drop_none(asdict(self.normalized()))


def default_embedding_text(
    *,
    source_type: str,
    item_type: str,
    subject: str,
    statement: str,
    status: str,
) -> str:
    lines = [
        f"source: {source_type}",
        f"type: {item_type}",
    ]
    if subject:
        lines.append(f"subject: {subject}")
    lines.extend(
        [
            f"statement: {statement}",
            f"status: {status}",
        ]
    )
    return "\n".join(lines)


def item_from_mapping(payload: dict[str, Any]) -> CreativeContextItem:
    evidence_refs = payload.get("evidence_refs") or ()
    refs = []
    for ref in evidence_refs:
        if isinstance(ref, EvidenceRef):
            refs.append(ref)
        elif isinstance(ref, dict):
            refs.append(EvidenceRef(**ref))
    return CreativeContextItem(
        item_id=str(payload.get("item_id") or payload.get("memory_id") or ""),
        project_id=str(payload.get("project_id") or ""),
        source_type=str(payload.get("source_type") or ""),
        source_id=str(payload.get("source_id") or payload.get("source_doc_id") or ""),
        unit_id=payload.get("unit_id") or payload.get("chunk_id"),
        item_type=str(payload.get("item_type") or payload.get("type") or ""),
        subject=str(payload.get("subject") or ""),
        statement=str(payload.get("statement") or payload.get("memory") or payload.get("text") or ""),
        entity_ids=tuple(payload.get("entity_ids") or ()),
        evidence_refs=tuple(refs),
        authority=str(payload.get("authority") or "model_inferred"),
        authority_score=payload.get("authority_score"),
        confidence=float(payload.get("confidence") or 0.0),
        status=str(payload.get("status") or "active"),
        visibility=str(payload.get("visibility") or payload.get("knowledge_scope") or "author_only"),
        temporal_scope=str(payload.get("temporal_scope") or payload.get("memory_temporal_scope") or "not_applicable"),
        embedding_text=str(payload.get("embedding_text") or ""),
        payload=dict(payload.get("payload") or payload.get("raw") or {}),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
    ).normalized()


def _normalize_item_type(value: str) -> str:
    text = _required_text(value, "item_type")
    return text if text in ITEM_TYPES else text


def _required_choice(value: Any, choices: set[str], field_name: str) -> str:
    text = _required_text(value, field_name)
    if text not in choices:
        raise ValueError(f"Unsupported {field_name}: {text}")
    return text


def _optional_source_type(value: str | None) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return _required_choice(text, SOURCE_TYPES, "source_type")


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe_texts(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _stable_string(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return str(value)

