from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dms.context_kernel.kernel import CreativeMemoryKernel, ExternalKnowledgeKernel
from dms.context_kernel.providers import RerankerProvider
from dms.context_kernel.schema import CreativeScope


@dataclass(frozen=True)
class CreativeContextPacketConfig:
    request: str
    scope: CreativeScope
    task_mode: str = "write"
    top_k: int = 12
    include_conversation_memory: bool = True
    include_artifact_memory: bool = True
    include_external_references: bool = True
    include_simulation: bool = False
    entity_ids: tuple[str, ...] = ()


class ContextAssembler:
    def __init__(
        self,
        memory_kernel: CreativeMemoryKernel,
        external_kernel: ExternalKnowledgeKernel | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self.memory_kernel = memory_kernel
        self.external_kernel = external_kernel or ExternalKnowledgeKernel(memory_kernel)
        self.reranker = reranker

    def build_packet(self, config: CreativeContextPacketConfig) -> dict[str, Any]:
        scope = config.scope.normalized()
        entity_ids = list(config.entity_ids or scope.entity_ids or ())
        query = config.request
        sections: dict[str, list[dict[str, Any]]] = {
            "conversation_guidance": [],
            "artifact_memory": [],
            "character_visible_knowledge": [],
            "relationship_context": [],
            "timeline_context": [],
            "external_reference_context": [],
            "style_guidance": [],
            "open_questions": [],
            "simulation_context": [],
            "entity_patch_context": [],
        }
        trace: dict[str, Any] = {
            "task_mode": config.task_mode,
            "source_roles": {
                "conversation": "guidance/decision/correction, not canon by default",
                "narrative_artifact": "story evidence; canonical only when status is canonical",
                "external_reference": "source-grounded background, not canon by default",
                "simulation": "hypothesis, not canon by default",
            },
            "retrieval": {},
        }

        if config.include_conversation_memory:
            conversation_hits = self.memory_kernel.search(
                query,
                scope=scope,
                source_types=["conversation"],
                statuses=["active", "canonical", "tentative"],
                entity_ids=entity_ids or None,
                top_k=config.top_k,
            )
            conversation_hits = self._rerank(query, conversation_hits, config.top_k)
            self._place_hits(conversation_hits, sections)
            trace["retrieval"]["conversation"] = _trace_hits(conversation_hits)

        if config.include_artifact_memory:
            artifact_hits = self.memory_kernel.search(
                query,
                scope=scope,
                source_types=["narrative_artifact"],
                statuses=["active", "canonical"],
                entity_ids=entity_ids or None,
                top_k=config.top_k,
            )
            artifact_hits = self._rerank(query, artifact_hits, config.top_k)
            self._place_hits(artifact_hits, sections)
            trace["retrieval"]["narrative_artifact"] = _trace_hits(artifact_hits)

        if config.include_external_references:
            external_hits = self.external_kernel.search(
                query,
                scope=scope,
                top_k=config.top_k,
            )
            if entity_ids:
                external_hits = _entity_filter_or_keep_subject_matches(external_hits, entity_ids)
            external_hits = self._rerank(query, external_hits, config.top_k)
            self._place_hits(external_hits, sections)
            trace["retrieval"]["external_reference"] = _trace_hits(external_hits)

        if config.include_simulation:
            simulation_hits = self.memory_kernel.search(
                query,
                scope=scope,
                source_types=["simulation"],
                statuses=["active", "tentative"],
                entity_ids=entity_ids or None,
                top_k=config.top_k,
            )
            simulation_hits = self._rerank(query, simulation_hits, config.top_k)
            self._place_hits(simulation_hits, sections)
            trace["retrieval"]["simulation"] = _trace_hits(simulation_hits)

        entity_views = []
        for entity_id in entity_ids:
            entity_views.append(self.memory_kernel.entity_view(project_id=scope.project_id, entity_id=entity_id))
        sections["entity_patch_context"].extend(_entity_patch_context_items(entity_views))

        return {
            "request": {
                "text": config.request,
                "task_mode": config.task_mode,
            },
            "retrieval_boundary": {
                "before_unit_id": scope.before_unit_id,
                "before_unit_order": scope.before_unit_order,
                "unit_id": scope.unit_id,
                "unit_type": scope.unit_type,
            },
            "task_state": {
                "project_id": scope.project_id,
                "artifact_id": scope.artifact_id,
                "artifact_version": scope.artifact_version,
                "conversation_id": scope.conversation_id,
            },
            "entities": entity_views,
            **sections,
            "source_references": _source_references(sections),
            "trace": trace,
        }

    def _place_hits(self, hits: list[dict[str, Any]], sections: dict[str, list[dict[str, Any]]]) -> None:
        for hit in hits:
            source_type = str(hit.get("source_type") or "")
            item_type = str(hit.get("item_type") or "")
            packet_item = _packet_item(hit)
            if source_type == "conversation":
                if item_type == "open_question":
                    sections["open_questions"].append(packet_item)
                elif item_type in {"style_preference", "user_preference"}:
                    sections["style_guidance"].append(packet_item)
                else:
                    sections["conversation_guidance"].append(packet_item)
            elif source_type == "narrative_artifact":
                if item_type == "relation":
                    sections["relationship_context"].append(packet_item)
                elif item_type in {"timeline_event"}:
                    sections["timeline_context"].append(packet_item)
                else:
                    sections["artifact_memory"].append(packet_item)
            elif source_type == "external_reference":
                if item_type == "style_guide":
                    sections["style_guidance"].append(packet_item)
                elif item_type == "timeline_doc":
                    sections["timeline_context"].append(packet_item)
                else:
                    sections["external_reference_context"].append(packet_item)
            elif source_type == "simulation":
                sections["simulation_context"].append(packet_item)

    def _rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not self.reranker or not hits:
            return hits[:top_k]
        return self.reranker.rerank(query, hits, top_k=top_k)


def format_creative_context_packet_markdown(packet: dict[str, Any]) -> str:
    lines = ["# Creative Context Packet", ""]
    request = packet.get("request") or {}
    if request.get("text"):
        lines.append(f"Request: {request.get('text')}")
        lines.append("")
    _append_section(lines, "Conversation Guidance", packet.get("conversation_guidance") or [])
    _append_section(lines, "Artifact Memory", packet.get("artifact_memory") or [])
    _append_section(lines, "Character Visible Knowledge", packet.get("character_visible_knowledge") or [])
    _append_section(lines, "Relationship Context", packet.get("relationship_context") or [])
    _append_section(lines, "Timeline Context", packet.get("timeline_context") or [])
    _append_section(lines, "External Reference Context", packet.get("external_reference_context") or [])
    _append_section(lines, "Style Guidance", packet.get("style_guidance") or [])
    _append_section(lines, "Open Questions", packet.get("open_questions") or [])
    _append_section(lines, "Simulation Context", packet.get("simulation_context") or [])
    _append_section(lines, "Entity Patch Context", packet.get("entity_patch_context") or [])
    lines.append("## Source References")
    refs = packet.get("source_references") or []
    if not refs:
        lines.append("- none")
    for ref in refs:
        lines.append(f"- {ref.get('item_id')} [{ref.get('source_type')}] {ref.get('source_id')}")
    return "\n".join(lines).rstrip() + "\n"


def _append_section(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append(f"## {title}")
    if not items:
        lines.append("- none")
        lines.append("")
        return
    for item in items:
        source_role = item.get("source_role")
        status = item.get("status")
        source_text = f" [{source_role}; {status}]" if source_role or status else ""
        lines.append(f"- {item.get('statement')}{source_text}")
        if item.get("subject"):
            lines.append(f"  subject: {item.get('subject')}")
        if item.get("evidence"):
            lines.append(f"  evidence: {item.get('evidence')}")
    lines.append("")


def _packet_item(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": hit.get("item_id"),
        "source_type": hit.get("source_type"),
        "source_role": hit.get("source_type"),
        "source_id": hit.get("source_id"),
        "unit_id": hit.get("unit_id"),
        "item_type": hit.get("item_type"),
        "subject": hit.get("subject"),
        "statement": hit.get("statement"),
        "entity_ids": hit.get("entity_ids") or [],
        "authority": hit.get("authority"),
        "confidence": hit.get("confidence"),
        "status": hit.get("status"),
        "visibility": hit.get("visibility"),
        "temporal_scope": hit.get("temporal_scope"),
        "score": hit.get("score"),
        "evidence": _first_evidence_text(hit),
        "payload": hit.get("payload") or {},
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


def _source_references(sections: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    refs = {}
    for items in sections.values():
        for item in items:
            item_id = str(item.get("item_id") or "")
            if not item_id:
                continue
            refs[item_id] = {
                "item_id": item_id,
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "unit_id": item.get("unit_id"),
                "authority": item.get("authority"),
                "status": item.get("status"),
            }
    return list(refs.values())


def _entity_patch_context_items(entity_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entity_view in entity_views:
        entity_id = str(entity_view.get("entity_id") or "")
        for patch in entity_view.get("active_patches") or []:
            if not isinstance(patch, dict):
                continue
            statement = str(patch.get("patch_statement") or "").strip()
            if not statement:
                continue
            items.append(
                {
                    "item_id": patch.get("patch_id"),
                    "source_type": "entity_patch",
                    "source_role": "entity_patch",
                    "source_id": patch.get("source_item_id"),
                    "unit_id": None,
                    "item_type": "entity_patch",
                    "subject": entity_id,
                    "statement": statement,
                    "entity_ids": [entity_id] if entity_id else [],
                    "authority": patch.get("authority"),
                    "confidence": None,
                    "status": patch.get("status"),
                    "visibility": "author_only",
                    "temporal_scope": "not_applicable",
                    "score": None,
                    "evidence": patch.get("source_item_id") or "",
                    "payload": patch,
                }
            )
    return items


def _trace_hits(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "returned_count": len(hits),
        "item_ids": [hit.get("item_id") for hit in hits],
    }


def _entity_filter_or_keep_subject_matches(hits: list[dict[str, Any]], entity_ids: list[str]) -> list[dict[str, Any]]:
    if not entity_ids:
        return hits
    entity_set = set(entity_ids)
    filtered = []
    for hit in hits:
        hit_entities = set(hit.get("entity_ids") or [])
        if hit_entities.intersection(entity_set):
            filtered.append(hit)
            continue
        payload = hit.get("payload") or {}
        subject = str(hit.get("subject") or "").lower()
        if subject and any(str(entity_id).split(":", 1)[-1].lower() in subject for entity_id in entity_ids):
            filtered.append(hit)
            continue
        if payload.get("external_reference_default_canon") is False:
            filtered.append(hit)
    return filtered
