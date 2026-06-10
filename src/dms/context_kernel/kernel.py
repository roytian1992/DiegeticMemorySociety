from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.context_kernel.providers import ContextLLMProvider
from dms.context_kernel.schema import (
    CreativeContextItem,
    CreativeScope,
    EntityPatch,
    EvidenceRef,
    SourceRecord,
    SourceUnit,
    default_embedding_text,
    item_from_mapping,
    stable_id,
)
from dms.context_kernel.store import CreativeContextStore


class CreativeMemoryKernel:
    """Source-aware memory API for creative context.

    The kernel is intentionally store-first. LLM extraction and vector retrieval can
    sit behind the same API later, but the standard item lifecycle is already
    explicit here: add/search/get/update/delete/history/link/promote/patch.
    """

    def __init__(self, store: CreativeContextStore) -> None:
        self.store = store

    @classmethod
    def from_db(cls, db_path: str | Path, *, reset: bool = False) -> "CreativeMemoryKernel":
        return cls(CreativeContextStore(db_path, reset=reset))

    def add_source(self, source: SourceRecord) -> dict[str, Any]:
        return self.store.upsert_source(source)

    def add_unit(self, unit: SourceUnit) -> dict[str, Any]:
        return self.store.upsert_unit(unit)

    def add_item(self, item: CreativeContextItem, *, actor: str = "system", reason: str = "") -> dict[str, Any]:
        return self.store.add_item(item, actor=actor, reason=reason)

    def add_items(
        self,
        items: list[CreativeContextItem],
        *,
        actor: str = "system",
        reason: str = "",
    ) -> list[dict[str, Any]]:
        return [self.add_item(item, actor=actor, reason=reason) for item in items]

    def search(
        self,
        query: str,
        *,
        scope: CreativeScope,
        source_types: list[str] | None = None,
        item_types: list[str] | None = None,
        statuses: list[str] | None = None,
        entity_ids: list[str] | None = None,
        visibility: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        return self.store.search_items(
            query,
            scope=scope,
            source_types=source_types,
            item_types=item_types,
            statuses=statuses,
            entity_ids=entity_ids,
            visibility=visibility,
            top_k=top_k,
        )

    def list_retrieval_documents(
        self,
        *,
        scope: CreativeScope,
        source_types: list[str] | None = None,
        item_types: list[str] | None = None,
        statuses: list[str] | None = None,
        entity_ids: list[str] | None = None,
        visibility: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        return self.store.list_retrieval_documents(
            scope=scope,
            source_types=source_types,
            item_types=item_types,
            statuses=statuses,
            entity_ids=entity_ids,
            visibility=visibility,
            limit=limit,
        )

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self.store.get_item(item_id)

    def update(self, item_id: str, patch: dict[str, Any], *, actor: str = "system", reason: str = "") -> dict[str, Any]:
        return self.store.update_item(item_id, patch, actor=actor, reason=reason)

    def delete(self, item_id: str, *, actor: str = "system", reason: str = "", soft: bool = True) -> dict[str, Any]:
        return self.store.delete_item(item_id, actor=actor, reason=reason, soft=soft)

    def history(self, item_id: str) -> list[dict[str, Any]]:
        return self.store.history(item_id)

    def link(
        self,
        source_item_id: str,
        target_item_id: str,
        link_type: str,
        *,
        evidence: str = "",
        created_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.store.add_link(
            source_item_id,
            target_item_id,
            link_type,
            evidence=evidence,
            created_by=created_by,
            metadata=metadata,
        )

    def promote(
        self,
        item_id: str,
        target_layer: str,
        *,
        authority: str = "user_confirmed",
        actor: str = "user",
        reason: str = "",
        target_item_id: str | None = None,
        status: str = "canonical",
    ) -> dict[str, Any]:
        return self.store.promote_item(
            item_id,
            target_layer=target_layer,
            authority=authority,
            actor=actor,
            reason=reason,
            target_item_id=target_item_id,
            status=status,
        )

    def add_entity_patch(self, patch: EntityPatch, *, actor: str = "system", reason: str = "") -> dict[str, Any]:
        return self.store.add_entity_patch(patch, actor=actor, reason=reason)

    def entity_view(self, *, project_id: str, entity_id: str) -> dict[str, Any]:
        return self.store.build_entity_view(project_id=project_id, entity_id=entity_id)


class ExternalKnowledgeKernel:
    """External-reference-facing API backed by the same context store."""

    def __init__(self, memory_kernel: CreativeMemoryKernel) -> None:
        self.memory_kernel = memory_kernel

    def add_item(self, item: CreativeContextItem, *, actor: str = "system", reason: str = "") -> dict[str, Any]:
        if item.source_type != "external_reference":
            raise ValueError("ExternalKnowledgeKernel only accepts source_type='external_reference'")
        if item.authority != "external_source":
            item = item_from_mapping(
                {
                    **item.model_dump(),
                    "authority": "external_source",
                    "status": item.status or "active",
                }
            )
        return self.memory_kernel.add_item(item, actor=actor, reason=reason)

    def search(self, query: str, *, scope: CreativeScope, top_k: int = 10) -> list[dict[str, Any]]:
        return self.memory_kernel.search(
            query,
            scope=scope,
            source_types=["external_reference"],
            statuses=["active", "canonical", "tentative"],
            top_k=top_k,
        )

    def promote(
        self,
        item_id: str,
        target_layer: str,
        *,
        actor: str = "user",
        reason: str = "",
    ) -> dict[str, Any]:
        return self.memory_kernel.promote(
            item_id,
            target_layer,
            authority="user_confirmed",
            actor=actor,
            reason=reason,
            status="canonical",
        )

    def qa(
        self,
        question: str,
        *,
        scope: CreativeScope,
        top_k: int = 6,
        llm_provider: ContextLLMProvider | None = None,
    ) -> dict[str, Any]:
        hits = self.search(question, scope=scope, top_k=top_k)
        items = [_external_answer_item(hit) for hit in hits]
        result = {
            "question": question,
            "answer": "",
            "answer_mode": "source_grounded_retrieval_only",
            "source_type": "external_reference",
            "items": items,
            "citations": _external_citations(items),
            "trace": {
                "external_reference_default_canon": False,
                "returned_count": len(hits),
            },
        }
        if llm_provider is None:
            return result
        answer = llm_provider.generate_text(_external_qa_prompt(question, items)).strip()
        result["answer"] = answer
        result["answer_mode"] = "source_grounded_llm"
        result["trace"] = {
            **result["trace"],
            "provider": getattr(llm_provider, "provider", "unknown"),
            "model": getattr(llm_provider, "model", "unknown"),
            "instruction": "answer only from retrieved external_reference records; do not promote to canon",
        }
        return result


def _external_qa_prompt(question: str, items: list[dict[str, Any]]) -> str:
    compact_items = []
    for index, item in enumerate(items, start=1):
        compact_items.append(
            {
                "ref": f"E{index}",
                "item_id": item.get("item_id"),
                "item_type": item.get("item_type"),
                "subject": item.get("subject"),
                "statement": item.get("statement"),
                "evidence": item.get("evidence"),
                "default_canon": False,
            }
        )
    return (
        "You answer questions over external reference knowledge for a creative writing project.\n"
        "Use only the retrieved external reference records below. External references are not canonical story facts by default.\n"
        "If the evidence is insufficient, say what is missing. Cite item refs like [E1].\n\n"
        f"Question:\n{question}\n\n"
        "Retrieved external reference records:\n"
        f"{json.dumps(compact_items, ensure_ascii=False, indent=2)}\n\n"
        "Answer in the user's language, concise and source-grounded."
    )


def _external_citations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for index, item in enumerate(items, start=1):
        citations.append(
            {
                "ref": f"E{index}",
                "item_id": item.get("item_id"),
                "item_type": item.get("item_type"),
                "subject": item.get("subject"),
                "default_canon": False,
            }
        )
    return citations


def import_artifact_store_items(
    kernel: CreativeMemoryKernel,
    *,
    asset_db_path: str | Path,
    project_id: str,
    source_id: str = "artifact_store",
    title: str = "Narrative Artifact Store",
    canonical: bool = False,
    actor: str = "system",
) -> dict[str, Any]:
    from dms.storage.asset_store import get_retrieval_documents

    source = SourceRecord(
        source_id=source_id,
        project_id=project_id,
        source_type="narrative_artifact",
        title=title,
        status="canonical" if canonical else "active",
        metadata={"asset_db_path": str(asset_db_path)},
    )
    kernel.add_source(source)
    records = get_retrieval_documents(asset_db_path)
    item_count = 0
    unit_ids: set[str] = set()
    for record in records:
        unit_id = str(record.get("parent_scene_id") or "")
        if unit_id and unit_id not in unit_ids:
            unit_ids.add(unit_id)
            kernel.add_unit(
                SourceUnit(
                    unit_id=unit_id,
                    source_id=source_id,
                    project_id=project_id,
                    source_type="narrative_artifact",
                    unit_type="scene",
                    unit_order=_optional_int(record.get("scene_order")),
                    metadata={"source": "asset_store"},
                )
            )
        item = artifact_record_to_context_item(
            record,
            project_id=project_id,
            source_id=source_id,
            canonical=canonical,
        )
        kernel.add_item(item, actor=actor, reason="imported from artifact store")
        item_count += 1
    return {
        "project_id": project_id,
        "source_id": source_id,
        "source_type": "narrative_artifact",
        "imported_items": item_count,
        "imported_units": len(unit_ids),
        "canonical": canonical,
    }


def import_reference_library_items(
    kernel: CreativeMemoryKernel,
    *,
    reference_db_path: str | Path,
    project_id: str,
    source_id: str = "external_reference_library",
    title: str = "External Reference Library",
    actor: str = "system",
) -> dict[str, Any]:
    from dms.reference_library import list_reference_context_records

    source = SourceRecord(
        source_id=source_id,
        project_id=project_id,
        source_type="external_reference",
        title=title,
        status="active",
        metadata={"reference_db_path": str(reference_db_path), "default_canon": False},
    )
    kernel.add_source(source)
    records = list_reference_context_records(reference_db_path)
    item_count = 0
    role_counts: dict[str, int] = {}
    for record in records:
        metadata = _metadata(record)
        source_role = str(record.get("source_role") or metadata.get("source_role") or "external_reference")
        role_counts[source_role] = role_counts.get(source_role, 0) + 1
        chunk_id = str(record.get("chunk_id") or "")
        if chunk_id:
            kernel.add_unit(
                SourceUnit(
                    unit_id=f"{source_id}:{chunk_id}",
                    source_id=source_id,
                    project_id=project_id,
                    source_type="external_reference",
                    unit_type="chunk",
                    metadata={
                        "source_doc_id": record.get("source_doc_id"),
                        "reference_db_path": str(reference_db_path),
                    },
                )
            )
        item = reference_record_to_context_item(
            record,
            project_id=project_id,
            source_id=source_id,
        )
        kernel.add_item(item, actor=actor, reason="imported from external reference library")
        item_count += 1
    return {
        "project_id": project_id,
        "source_id": source_id,
        "source_type": "external_reference",
        "imported_items": item_count,
        "default_canon": False,
        "asset_model": "source_local_external_reference_v1",
        "source_roles": role_counts,
    }


def artifact_record_to_context_item(
    record: dict[str, Any],
    *,
    project_id: str,
    source_id: str,
    canonical: bool = False,
) -> CreativeContextItem:
    metadata = _metadata(record)
    memory = _memory_payload_from_record(record)
    doc_id = str(record.get("doc_id") or record.get("source_id") or stable_id("artifact_doc", record))
    unit_id = str(record.get("parent_scene_id") or memory.get("parent_scene_id") or "")
    statement = _artifact_statement(record, metadata, memory)
    item_type = _artifact_item_type(str(record.get("doc_type") or ""), metadata, memory)
    entity_id = str(record.get("entity_id") or "").strip()
    evidence_text = str(memory.get("evidence_text") or metadata.get("evidence") or "")
    item_id = f"artifact:{doc_id}"
    evidence_refs = ()
    if evidence_text:
        evidence_refs = (
            EvidenceRef(
                evidence_id=stable_id("ev", item_id, evidence_text),
                item_id=item_id,
                source_id=source_id,
                unit_id=unit_id or None,
                text=evidence_text,
                start_offset=_optional_int(memory.get("evidence_start")),
                end_offset=_optional_int(memory.get("evidence_end")),
                alignment_status="imported",
            ),
        )
    return CreativeContextItem(
        item_id=item_id,
        project_id=project_id,
        source_type="narrative_artifact",
        source_id=source_id,
        unit_id=unit_id or None,
        item_type=item_type,
        subject=str(metadata.get("subject") or memory.get("entity_name") or entity_id or ""),
        statement=statement,
        entity_ids=(entity_id,) if entity_id else (),
        evidence_refs=evidence_refs,
        authority="artifact_canonical" if canonical else "artifact_active",
        confidence=1.0 if canonical else 0.8,
        status="canonical" if canonical else "active",
        visibility="character_visible" if item_type in {"event", "state", "relation", "fact"} else "author_only",
        temporal_scope=str(memory.get("memory_temporal_scope") or metadata.get("memory_temporal_scope") or "uncertain"),
        payload={
            "source_doc": record,
            "metadata": metadata,
            "memory": memory,
            "canonical_artifact": canonical,
        },
    ).normalized()


def reference_record_to_context_item(
    record: dict[str, Any],
    *,
    project_id: str,
    source_id: str,
) -> CreativeContextItem:
    metadata = _metadata(record)
    item_id = f"external:{record.get('item_id') or record.get('doc_id')}"
    statement = str(record.get("statement") or metadata.get("statement") or record.get("text") or "").strip()
    evidence = str(record.get("evidence") or metadata.get("evidence") or "").strip()
    subject = str(record.get("subject") or metadata.get("subject") or "").strip()
    chunk_id = str(record.get("chunk_id") or metadata.get("chunk_id") or "").strip()
    source_doc_id = str(record.get("source_doc_id") or metadata.get("source_doc_id") or "").strip()
    evidence_refs = ()
    if evidence:
        evidence_refs = (
            EvidenceRef(
                evidence_id=stable_id("ev", item_id, evidence),
                item_id=item_id,
                source_id=source_id,
                unit_id=f"{source_id}:{chunk_id}" if chunk_id else None,
                text=evidence,
                alignment_status="imported",
            ),
        )
    return CreativeContextItem(
        item_id=item_id,
        project_id=project_id,
        source_type="external_reference",
        source_id=source_id,
        unit_id=f"{source_id}:{chunk_id}" if chunk_id else None,
        item_type=str(record.get("item_type") or metadata.get("item_type") or "notes"),
        subject=subject,
        statement=statement,
        entity_ids=tuple(_reference_entity_ids(subject, metadata)),
        evidence_refs=evidence_refs,
        authority="external_source",
        authority_score=_optional_float(record.get("authority") or metadata.get("authority")),
        confidence=float(record.get("confidence") or metadata.get("confidence") or 0.5),
        status="active",
        visibility=str(record.get("knowledge_scope") or metadata.get("knowledge_scope") or "author_only"),
        temporal_scope="not_applicable",
        embedding_text=default_embedding_text(
            source_type="external_reference",
            item_type=str(record.get("item_type") or metadata.get("item_type") or "notes"),
            subject=subject,
            statement=statement,
            status="active",
        ),
        payload={
            "source_doc_id": source_doc_id,
            "chunk_id": chunk_id,
            "metadata": metadata,
            "knowledge_scope": record.get("knowledge_scope") or metadata.get("knowledge_scope"),
            "known_to": record.get("known_to") or metadata.get("known_to") or [],
            "available_from": record.get("available_from") or metadata.get("available_from"),
            "timeline_hint": record.get("timeline_hint") or metadata.get("timeline_hint"),
            "external_reference_default_canon": False,
            "source_record": record,
        },
    ).normalized()


def _external_answer_item(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": hit.get("item_id"),
        "item_type": hit.get("item_type"),
        "subject": hit.get("subject"),
        "statement": hit.get("statement"),
        "evidence": _first_evidence_text(hit),
        "score": hit.get("score"),
        "source_role": "external_reference",
        "default_canon": False,
    }


def _first_evidence_text(hit: dict[str, Any]) -> str:
    refs = hit.get("evidence_refs") or []
    if refs and isinstance(refs[0], dict):
        return str(refs[0].get("text") or "")
    metadata = hit.get("metadata") or {}
    refs = metadata.get("evidence_refs") or []
    if refs and isinstance(refs[0], dict):
        return str(refs[0].get("text") or "")
    return ""


def _artifact_statement(record: dict[str, Any], metadata: dict[str, Any], memory: dict[str, Any]) -> str:
    for key in ("summary", "proposition", "statement", "text"):
        value = memory.get(key) or metadata.get(key) or record.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return str(record.get("text") or "").strip()


def _artifact_item_type(doc_type: str, metadata: dict[str, Any], memory: dict[str, Any]) -> str:
    raw_type = str(memory.get("memory_type") or metadata.get("memory_type") or "").strip()
    if doc_type == "stated_fact":
        return "fact"
    if doc_type in {"scene_summary", "unit_summary"}:
        return "summary"
    if doc_type.startswith("episodic_memory"):
        if raw_type in {"relationship", "relationship_observation"}:
            return "relation"
        if raw_type in {"state", "entity_state"}:
            return "state"
        return "event"
    return "fact"


def _memory_payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("memory"), dict):
        return dict(record["memory"])
    metadata = _metadata(record)
    raw = metadata.get("raw")
    if isinstance(raw, dict):
        return raw
    return metadata


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    metadata_json = record.get("metadata_json")
    if isinstance(metadata_json, str) and metadata_json.strip():
        try:
            parsed = json.loads(metadata_json)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _reference_entity_ids(subject: str, metadata: dict[str, Any]) -> list[str]:
    entity_ids = metadata.get("entity_ids")
    if isinstance(entity_ids, list):
        return [str(item) for item in entity_ids if str(item or "").strip()]
    subject = str(subject or "").strip()
    return [f"external_subject:{subject}"] if subject else []


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
