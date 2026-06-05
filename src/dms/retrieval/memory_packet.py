from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.narrative_units import DEFAULT_UNIT_LABEL, DEFAULT_UNIT_TYPE
from dms.reference_library import ReferenceContextQuery, build_reference_context
from dms.storage import (
    get_entity_memories,
    get_one_hop_relationships,
    get_relationship_count,
    get_scene_metadata,
    list_entities,
    resolve_entity_refs,
    search_entity_memories,
    search_retrieval_documents,
)
from dms.relationship_types import soften_formal_relation_type


GLOBAL_SCOPE_VISIBLE_SCOPES = {"atemporal_fact", "durable_state"}


@dataclass(frozen=True)
class MemoryPacketConfig:
    db_path: Path
    chroma_dir: Path
    writing_intent: str
    before_scene_id: str | None = None
    before_scene_order: int | None = None
    unit_type: str = DEFAULT_UNIT_TYPE
    unit_label: str = DEFAULT_UNIT_LABEL
    scene_top_k: int = 5
    entity_memory_top_k: int = 12
    global_scope_memory_top_k: int = 8
    max_entity_memories_before_vector: int = 50
    entity_match_limit: int = 1
    collection_name: str = "dms_retrieval_documents"
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60
    include_reference_context: bool = False
    reference_db_path: Path | None = None
    reference_chroma_dir: Path | None = None
    reference_collection_name: str = "dms_reference_documents"
    reference_top_k: int = 6
    reference_author_top_k: int = 6
    reference_character_top_k: int = 6
    reference_style_top_k: int = 4
    reference_timeline_top_k: int = 4


def decompose_writing_intent(
    writing_intent: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    intent = str(writing_intent or "").strip()
    important_entities = _extract_explicit_entity_refs(intent, db_path=db_path) if db_path else _extract_surface_refs(intent)
    narrative_units = _extract_narrative_units(intent, important_entities)
    return {
        "method": "deterministic_alias_scene_unit_v2",
        "narrative_units": narrative_units or [intent],
        "important_entities": important_entities,
    }


def build_memory_packet(config: MemoryPacketConfig) -> dict[str, Any]:
    decomposition = decompose_writing_intent(config.writing_intent, db_path=config.db_path)
    entity_matches = _resolve_intent_entities(config, decomposition["important_entities"])
    matched_entities = _unique_best_matches(entity_matches)
    related_scene_summaries, scene_retrieval_trace = _retrieve_related_scene_summaries(
        config,
        decomposition["narrative_units"],
    )
    relations = _retrieve_one_hop_relations(config, [entity["entity_id"] for entity in matched_entities])
    relationship_diagnostics = _relationship_diagnostics(config, matched_entities, relations)

    memory_ordinals: dict[str, int] = {}
    indexed_memories: list[dict[str, Any]] = []
    entity_packets: list[dict[str, Any]] = []
    memory_trace: dict[str, Any] = {}
    reference_binder = _ReferenceBinder()

    for entity in matched_entities:
        memories, strategy = _retrieve_entity_memories(config, entity)
        related_indexes: list[int] = []
        for memory in memories:
            memory_id = str(memory.get("memory_id") or "")
            if not memory_id:
                continue
            if memory_id not in memory_ordinals:
                memory_ordinals[memory_id] = len(indexed_memories) + 1
                source_ref = reference_binder.add_memory_evidence(memory)
                indexed_memories.append(_memory_packet_record(memory, memory_ordinals[memory_id], source_ref))
            related_indexes.append(memory_ordinals[memory_id])
        entity_context, entity_source_refs = _entity_context(entity, memories, reference_binder)
        entity_packets.append(
            {
                "entity_id": entity["entity_id"],
                "canonical_name": entity["canonical_name"],
                "entity_type": entity["entity_type"],
                "aliases": entity.get("aliases") or [],
                "initial_description": entity.get("initial_description") or "",
                "author_description": entity.get("author_description") or "",
                "author_profile": entity.get("author_profile") or {},
                "initial_state": entity.get("initial_state") or {},
                "profile_policy": entity.get("profile_policy") or {},
                "profile_sources": entity.get("profile_sources") or [],
                "author_entity_ids": entity.get("author_entity_ids") or [],
                "author_profile_summary": entity_context.get("author_profile_summary", ""),
                "profile": entity_context["profile"],
                "current_state": entity_context["current_state"],
                "source_refs": entity_source_refs,
                "match": entity["match"],
                "related_memory_index": [f"M{index}" for index in sorted(set(related_indexes))],
            }
        )
        memory_trace[str(entity["entity_id"])] = strategy

    global_scope_memories, global_scope_trace = _retrieve_global_scope_memories(
        config,
        decomposition["narrative_units"],
    )
    for memory in global_scope_memories:
        memory_id = str(memory.get("memory_id") or "")
        if not memory_id or memory_id in memory_ordinals:
            continue
        memory_ordinals[memory_id] = len(indexed_memories) + 1
        source_ref = reference_binder.add_memory_evidence(memory)
        indexed_memories.append(_memory_packet_record(memory, memory_ordinals[memory_id], source_ref))

    relations = _bind_relation_refs(relations, reference_binder)
    related_scene_summaries = _bind_scene_summary_refs(related_scene_summaries, reference_binder)
    reference_context, reference_context_trace = _retrieve_reference_context(
        config,
        decomposition["narrative_units"],
        matched_entities,
    )

    return {
        "retrieval_boundary": {
            "before_unit_id": config.before_scene_id,
            "before_unit_order": config.before_scene_order,
            "unit_type": config.unit_type,
            "unit_label": config.unit_label,
            "before_scene_id": config.before_scene_id,
            "before_scene_order": config.before_scene_order,
            "semantics": "strictly_before_unit_with_scene_id_compatibility",
        },
        "entities": entity_packets,
        "relations": relations,
        "relationship_diagnostics": relationship_diagnostics,
        "episodic_memories": indexed_memories,
        "related_scene_summaries": related_scene_summaries,
        **reference_context,
        "references": reference_binder.references,
        "trace": {
            "query_decomposition": {
                "method": decomposition.get("method"),
                "important_entities": decomposition.get("important_entities", []),
                "narrative_units": decomposition.get("narrative_units", []),
            },
            "scene_summary_retrieval": scene_retrieval_trace,
            "entity_matches": entity_matches,
            "entity_memory_retrieval": memory_trace,
            "global_scope_memory_retrieval": global_scope_trace,
            "reference_context_retrieval": reference_context_trace,
            "memory_temporal_scope_policy": {
                "reveal_time_filtering_changed": False,
                "atemporal_fact": "eligible for global retrieval after reveal boundary; no story-time bucket required",
                "durable_state": "eligible for global retrieval after reveal boundary until superseded; no story-time bucket required",
                "temporal_episode": "visible after reveal boundary; future story-time filters may apply",
                "uncertain": "treated conservatively as time-bound",
            },
            "scene_top_k": config.scene_top_k,
            "entity_memory_top_k": config.entity_memory_top_k,
            "global_scope_memory_top_k": config.global_scope_memory_top_k,
            "max_entity_memories_before_vector": config.max_entity_memories_before_vector,
        },
    }


def format_memory_packet_markdown(packet: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Memory Packet")
    lines.append("")
    lines.append("## Entities")
    boundary = packet.get("retrieval_boundary") or {}
    boundary_id = boundary.get("before_unit_id") or boundary.get("before_scene_id")
    unit_label = boundary.get("unit_label") or DEFAULT_UNIT_LABEL
    state_label = f"current state before {_format_unit_boundary(unit_label, boundary_id)}" if boundary_id else "current state"
    for entity in packet.get("entities") or []:
        lines.append("")
        lines.append(f"{entity.get('canonical_name')} ({entity.get('entity_type')})")
        profile = entity.get("profile") or entity.get("description") or ""
        if profile:
            lines.append(f"- profile: {profile}")
        author_profile_summary = entity.get("author_profile_summary") or ""
        if author_profile_summary:
            lines.append(f"- author profile baseline: {author_profile_summary}")
        initial_state_summary = _compact_initial_state_summary(entity.get("initial_state") or {})
        if initial_state_summary:
            lines.append(f"- author initial state: {initial_state_summary}")
        current_state = entity.get("current_state") or ""
        if current_state:
            lines.append(f"- {state_label}: {current_state}")
        memory_indexes = entity.get("related_memory_index") or []
        lines.append(f"- related memory index: {_format_index_list(memory_indexes)}")
        if entity.get("source_refs"):
            lines.append(f"- refs: {_format_ref_list(entity.get('source_refs') or [])}")
    lines.append("")
    lines.append("## Relations")
    relations = packet.get("relations") or []
    if not relations:
        lines.append("- none")
        diagnostics = packet.get("relationship_diagnostics") or {}
        if diagnostics.get("reason"):
            lines.append(f"- diagnostic: {diagnostics.get('reason')}")
    for relation in relations:
        source = relation.get("source_name") or relation.get("source_entity_id")
        target = relation.get("target_name") or relation.get("target_entity_id")
        lines.append(
            f"- {source} -> {target}: {relation.get('relation_type')}"
            f" ({relation.get('status') or 'status_unknown'}, {relation.get('last_updated_scene') or 'scene_unknown'})"
        )
        if relation.get("source_refs"):
            lines.append(f"    refs: {_format_ref_list(relation.get('source_refs') or [])}")
    lines.append("")
    lines.append("## Extracted Episodic Memories")
    memories = packet.get("episodic_memories") or []
    if not memories:
        lines.append("- none")
    for memory in memories:
        lines.append(
            f"[{memory.get('index')}] {memory.get('summary', '')}"
            f" <{memory.get('scene_id')}>"
        )
        if memory.get("memory_temporal_scope"):
            lines.append(f"    scope: {memory.get('memory_temporal_scope')}")
        if memory.get("source_ref"):
            lines.append(f"    ref: [{memory.get('source_ref')}]")
    lines.append("")
    lines.append("## Related Scene Summary")
    scene_summaries = packet.get("related_scene_summaries") or []
    if not scene_summaries:
        lines.append("- none")
    for summary in scene_summaries:
        score = summary.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        metadata = summary.get("metadata") or {}
        title = metadata.get("title") or summary.get("scene_id")
        setting = metadata.get("setting") or {}
        location = setting.get("location")
        setting_text = f" | setting: {location}" if location else ""
        lines.append(f"- <{summary.get('scene_id')}> {title}{score_text}{setting_text}")
        lines.append(f"  summary: {summary.get('summary', '')}")
        facts = metadata.get("stated_facts") or []
        if facts:
            lines.append("  facts:")
            for fact in facts[:3]:
                lines.append(f"  - {fact.get('proposition')}")
        questions = metadata.get("open_questions") or []
        if questions:
            lines.append("  open questions:")
            for question in questions[:2]:
                lines.append(f"  - {question.get('question')}")
        tags = metadata.get("scene_tags") or []
        if tags:
            tag_text = ", ".join(str(tag.get("surface")) for tag in tags[:6] if tag.get("surface"))
            if tag_text:
                lines.append(f"  tags: {tag_text}")
        if summary.get("source_ref"):
            lines.append(f"  ref: [{summary.get('source_ref')}]")
    reference_trace = (packet.get("trace") or {}).get("reference_context_retrieval") or {}
    if reference_trace.get("enabled") or _has_reference_context(packet):
        _append_reference_context_markdown(lines, "Author Reference Context", packet.get("author_reference_context") or [])
        _append_reference_context_markdown(lines, "Character Reference Knowledge", packet.get("character_reference_knowledge") or [])
        _append_reference_context_markdown(lines, "Style Reference Context", packet.get("style_reference_context") or [])
        _append_reference_context_markdown(lines, "Timeline Reference Claims", packet.get("timeline_reference_claims") or [])
    lines.append("")
    lines.append("## References")
    references = packet.get("references") or []
    if not references:
        lines.append("- none")
    for reference in references:
        ref_id = reference.get("ref_id")
        scene_id = reference.get("scene_id")
        text = reference.get("text") or ""
        lines.append(f"[{ref_id}] <{scene_id}>")
        if text:
            for wrapped_line in _format_reference_text_lines(text):
                lines.append(f"    {wrapped_line}")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_intent_entities(config: MemoryPacketConfig, entity_refs: list[str]) -> list[dict[str, Any]]:
    if not entity_refs:
        return []
    return resolve_entity_refs(
        config.db_path,
        entity_refs,
        limit_per_ref=max(config.entity_match_limit, 1),
        min_score=0.55,
    )


def _format_unit_boundary(unit_label: Any, unit_id: Any) -> str:
    text = str(unit_id or "").strip()
    label = str(unit_label or DEFAULT_UNIT_LABEL).strip()
    if not text:
        return ""
    if label and text.lower().startswith(f"{label.lower()}_"):
        return text
    return f"{label} {text}" if label else text


def _unique_best_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[str, dict[str, Any]] = {}
    for match in matches:
        entity_id = str(match.get("entity_id") or "")
        if not entity_id:
            continue
        existing = best_by_id.get(entity_id)
        if existing is None or float(match.get("score") or 0.0) > float(existing.get("match", {}).get("score") or 0.0):
            best_by_id[entity_id] = {
                "entity_id": entity_id,
                "canonical_name": match.get("canonical_name"),
                "entity_type": match.get("entity_type"),
                "aliases": match.get("aliases") or [],
                "first_seen_scene": match.get("first_seen_scene"),
                "first_seen_order": match.get("first_seen_order"),
                "mention_count": match.get("mention_count"),
                "initial_description": match.get("initial_description", ""),
                "author_description": match.get("author_description", ""),
                "descriptions": match.get("descriptions") or [],
                "description_sources": match.get("description_sources") or [],
                "author_profile": match.get("author_profile") or {},
                "initial_state": match.get("initial_state") or {},
                "profile_policy": match.get("profile_policy") or {},
                "profile_sources": match.get("profile_sources") or [],
                "author_entity_ids": match.get("author_entity_ids") or [],
                "match": {
                    "query": match.get("query"),
                    "score": match.get("score"),
                    "match_type": match.get("match_type"),
                    "matched_alias": match.get("matched_alias"),
                },
            }
    return sorted(
        best_by_id.values(),
        key=lambda item: (
            str(item.get("entity_type") or ""),
            int(item.get("first_seen_order") or 0),
            str(item.get("canonical_name") or ""),
        ),
    )


def _retrieve_related_scene_summaries(
    config: MemoryPacketConfig,
    narrative_units: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = "\n".join(unit for unit in narrative_units if str(unit or "").strip()) or config.writing_intent
    summary_result = search_retrieval_documents(
        config.db_path,
        persist_dir=config.chroma_dir,
        query=query,
        collection_name=config.collection_name,
        doc_type="scene_summary",
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
        top_k=max(config.scene_top_k, 1),
        embedding_dim=config.embedding_dim,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_max_tokens=config.embedding_max_tokens,
        embedding_timeout=config.embedding_timeout,
    )
    fact_result = search_retrieval_documents(
        config.db_path,
        persist_dir=config.chroma_dir,
        query=query,
        collection_name=config.collection_name,
        doc_type="stated_fact",
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
        top_k=max(config.scene_top_k, 1),
        embedding_dim=config.embedding_dim,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_max_tokens=config.embedding_max_tokens,
        embedding_timeout=config.embedding_timeout,
    )
    scenes = _merge_scene_hits(summary_result.get("results") or [], fact_result.get("results") or [], top_k=max(config.scene_top_k, 1))
    summaries = [_scene_summary_from_hits(scene_id, hits) for scene_id, hits in scenes]
    scene_metadata = get_scene_metadata(config.db_path, [summary["scene_id"] for summary in summaries])
    for summary in summaries:
        summary["metadata"] = _compact_scene_metadata(scene_metadata.get(summary["scene_id"], {}))
    return summaries, {
        "query": query,
        "sources": ["scene_summary", "stated_fact"],
        "narrative_unit_count": len([unit for unit in narrative_units if str(unit or "").strip()]),
        "scene_summary_hits": _compact_retrieval_hits(summary_result.get("results") or []),
        "stated_fact_hits": _compact_retrieval_hits(fact_result.get("results") or []),
        "returned_scene_ids": [summary["scene_id"] for summary in summaries],
    }


def _merge_scene_hits(
    summary_hits: list[dict[str, Any]],
    fact_hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    best_score: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    rank = 0
    for retrieval_source, hits in (("scene_summary", summary_hits), ("stated_fact", fact_hits)):
        for hit in hits:
            sql = hit.get("sql") or {}
            scene_id = str(sql.get("parent_scene_id") or "")
            if not scene_id:
                continue
            rank += 1
            hit_with_source = dict(hit)
            hit_with_source["retrieval_source"] = retrieval_source
            by_scene.setdefault(scene_id, []).append(hit_with_source)
            score = float(hit.get("score") or 0.0)
            best_score[scene_id] = max(best_score.get(scene_id, score), score)
            first_rank.setdefault(scene_id, rank)
    ordered_scene_ids = sorted(by_scene, key=lambda scene_id: (-best_score.get(scene_id, 0.0), first_rank.get(scene_id, 0)))
    return [(scene_id, by_scene[scene_id]) for scene_id in ordered_scene_ids[:top_k]]


def _scene_summary_from_hits(scene_id: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    summary_hit = next((hit for hit in hits if hit.get("retrieval_source") == "scene_summary"), hits[0] if hits else {})
    sql = summary_hit.get("sql") or {}
    metadata = _json_loads(sql.get("metadata_json"), default={})
    evidence_hits = [_scene_retrieval_evidence(hit) for hit in hits]
    return {
        "scene_id": scene_id,
        "source_id": sql.get("source_id") or scene_id,
        "doc_id": summary_hit.get("doc_id"),
        "summary": metadata.get("summary") or sql.get("text") or summary_hit.get("text") or "",
        "retrieval_text": sql.get("text") or summary_hit.get("text") or "",
        "score": max((float(hit.get("score") or 0.0) for hit in hits), default=None),
        "retrieval_sources": sorted({str(hit.get("retrieval_source")) for hit in hits if hit.get("retrieval_source")}),
        "retrieval_evidence": evidence_hits,
    }


def _scene_retrieval_evidence(hit: dict[str, Any]) -> dict[str, Any]:
    sql = hit.get("sql") or {}
    metadata = _json_loads(sql.get("metadata_json"), default={})
    return {
        "source": hit.get("retrieval_source"),
        "doc_id": hit.get("doc_id"),
        "source_id": sql.get("source_id"),
        "score": hit.get("score"),
        "text": metadata.get("proposition") or metadata.get("summary") or sql.get("text") or hit.get("text") or "",
    }


def _compact_retrieval_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for hit in hits:
        sql = hit.get("sql") or {}
        compact.append(
            {
                "doc_id": hit.get("doc_id"),
                "doc_type": sql.get("doc_type"),
                "source_id": sql.get("source_id"),
                "scene_id": sql.get("parent_scene_id"),
                "score": hit.get("score"),
            }
        )
    return compact


def _retrieve_one_hop_relations(config: MemoryPacketConfig, entity_ids: list[str]) -> list[dict[str, Any]]:
    relations = get_one_hop_relationships(
        config.db_path,
        entity_ids=entity_ids,
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
    )
    allowed_ids = set(entity_ids)
    packed: list[dict[str, Any]] = []
    for relation in relations:
        relation_type = soften_formal_relation_type(
            relation.get("relation_type"),
            evidence="\n".join(str(item) for item in relation.get("evidence") or []),
        )
        packed.append(
            {
                "relationship_id": relation.get("relationship_id"),
                "source_entity_id": relation.get("source_entity_id"),
                "source_name": relation.get("source_name"),
                "source_entity_type": relation.get("source_entity_type"),
                "target_entity_id": relation.get("target_entity_id"),
                "target_name": relation.get("target_name"),
                "target_entity_type": relation.get("target_entity_type"),
                "relation_type": relation_type,
                "direction": relation.get("direction"),
                "status": relation.get("status"),
                "strength": relation.get("strength"),
                "first_seen_scene": relation.get("first_seen_scene"),
                "last_updated_scene": relation.get("last_updated_scene"),
                "is_between_matched_entities": (
                    relation.get("source_entity_id") in allowed_ids and relation.get("target_entity_id") in allowed_ids
                ),
                "evidence": relation.get("evidence") or [],
            }
        )
    return packed


def _relationship_diagnostics(
    config: MemoryPacketConfig,
    matched_entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    total_relationships = get_relationship_count(config.db_path)
    return {
        "matched_entity_count": len(matched_entities),
        "total_relationship_count": total_relationships,
        "returned_relationship_count": len(relations),
        "reason": (
            "No durable relationship records exist in the current SQLite asset store."
            if total_relationships == 0
            else "No one-hop durable relationship records matched the selected entities before the retrieval boundary."
        )
        if not relations
        else "",
    }


def _retrieve_entity_memories(
    config: MemoryPacketConfig,
    entity: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entity_ref = str(entity["entity_id"])
    all_memories = get_entity_memories(
        config.db_path,
        entity_ref=entity_ref,
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
    )
    if len(all_memories) <= config.max_entity_memories_before_vector:
        return all_memories, {
            "strategy": "all_before_boundary",
            "candidate_count": len(all_memories),
            "returned_count": len(all_memories),
        }

    vector_query = f"{config.writing_intent}\n{entity.get('canonical_name')}"
    result = search_entity_memories(
        config.db_path,
        persist_dir=config.chroma_dir,
        query=vector_query,
        collection_name=config.collection_name,
        entity_ref=entity_ref,
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
        top_k=max(config.entity_memory_top_k, 1),
        embedding_dim=config.embedding_dim,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_max_tokens=config.embedding_max_tokens,
        embedding_timeout=config.embedding_timeout,
    )
    vector_memories = [
        hit["memory"]
        for hit in result.get("results") or []
        if isinstance(hit, dict) and isinstance(hit.get("memory"), dict)
    ]
    return vector_memories, {
        "strategy": "vector_top_k_after_sql_filter",
        "candidate_count": len(all_memories),
        "returned_count": len(vector_memories),
        "query": vector_query,
    }


def _retrieve_global_scope_memories(
    config: MemoryPacketConfig,
    narrative_units: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    top_k = max(int(config.global_scope_memory_top_k or 0), 0)
    if top_k <= 0:
        return [], {
            "strategy": "disabled",
            "allowed_scopes": sorted(GLOBAL_SCOPE_VISIBLE_SCOPES),
            "candidate_count": 0,
            "returned_count": 0,
        }

    query = "\n".join(unit for unit in narrative_units if str(unit or "").strip()) or config.writing_intent
    candidate_top_k = max(top_k * 3, top_k, 1)
    result = search_retrieval_documents(
        config.db_path,
        persist_dir=config.chroma_dir,
        query=query,
        collection_name=config.collection_name,
        doc_type="episodic_memory_global",
        before_scene_id=config.before_scene_id,
        before_scene_order=config.before_scene_order,
        top_k=candidate_top_k,
        embedding_dim=config.embedding_dim,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_max_tokens=config.embedding_max_tokens,
        embedding_timeout=config.embedding_timeout,
    )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    scope_counts: dict[str, int] = {}
    for hit in result.get("results") or []:
        if not isinstance(hit, dict) or not isinstance(hit.get("memory"), dict):
            continue
        memory = hit["memory"]
        scope = str(memory.get("memory_temporal_scope") or "temporal_episode")
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        if scope not in GLOBAL_SCOPE_VISIBLE_SCOPES:
            continue
        memory_id = str(memory.get("memory_id") or "")
        if not memory_id or memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)
        selected.append(memory)
        if len(selected) >= top_k:
            break

    return selected, {
        "strategy": "global_scope_vector_top_k_after_sql_filter",
        "allowed_scopes": sorted(GLOBAL_SCOPE_VISIBLE_SCOPES),
        "candidate_count": result.get("count", 0),
        "candidate_top_k": candidate_top_k,
        "returned_count": len(selected),
        "scope_counts": scope_counts,
        "query": query,
    }


def _retrieve_reference_context(
    config: MemoryPacketConfig,
    narrative_units: list[str],
    matched_entities: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    empty = {
        "author_reference_context": [],
        "character_reference_knowledge": [],
        "style_reference_context": [],
        "timeline_reference_claims": [],
    }
    if not config.include_reference_context:
        return empty, {"enabled": False, "reason": "include_reference_context is false"}
    if config.reference_db_path is None:
        return empty, {"enabled": False, "reason": "reference_db_path is not configured"}

    query_text = "\n".join(unit for unit in narrative_units if str(unit or "").strip()) or config.writing_intent
    context, trace = build_reference_context(
        ReferenceContextQuery(
            db_path=config.reference_db_path,
            query=query_text,
            matched_entities=tuple(matched_entities),
            before_scene_id=config.before_scene_id,
            before_scene_order=config.before_scene_order,
            chroma_dir=config.reference_chroma_dir,
            collection_name=config.reference_collection_name,
            top_k=config.reference_top_k,
            author_top_k=config.reference_author_top_k,
            character_top_k=config.reference_character_top_k,
            style_top_k=config.reference_style_top_k,
            timeline_top_k=config.reference_timeline_top_k,
            embedding_dim=config.embedding_dim,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_base_url=config.embedding_base_url,
            embedding_api_key=config.embedding_api_key,
            embedding_max_tokens=config.embedding_max_tokens,
            embedding_timeout=config.embedding_timeout,
        )
    )
    return context, trace


def _memory_packet_record(memory: dict[str, Any], index: int, source_ref: str) -> dict[str, Any]:
    memory_index = f"M{index}"
    return {
        "index": memory_index,
        "ordinal": index,
        "memory_id": memory.get("memory_id"),
        "unit_id": memory.get("chunk_id") or memory.get("parent_scene_id"),
        "unit_type": memory.get("unit_type") or (memory.get("raw") or {}).get("unit_type"),
        "unit_label": memory.get("unit_label") or (memory.get("raw") or {}).get("unit_label"),
        "parent_unit_id": memory.get("parent_scene_id"),
        "scene_id": memory.get("parent_scene_id"),
        "scene_order": memory.get("scene_order"),
        "chunk_index": memory.get("chunk_index"),
        "sequence_index": memory.get("sequence_index"),
        "timeline_index": memory.get("timeline_index"),
        "memory_temporal_scope": memory.get("memory_temporal_scope") or "temporal_episode",
        "memory_temporal_scope_confidence": memory.get("memory_temporal_scope_confidence"),
        "memory_type": memory.get("memory_type"),
        "summary": memory.get("summary"),
        "source_ref": source_ref,
        "evidence_start": memory.get("evidence_start"),
        "evidence_end": memory.get("evidence_end"),
        "parent_evidence_start": memory.get("parent_evidence_start"),
        "parent_evidence_end": memory.get("parent_evidence_end"),
    }


def _entity_context(
    entity: dict[str, Any],
    memories: list[dict[str, Any]],
    reference_binder: "_ReferenceBinder",
) -> tuple[dict[str, str], list[str]]:
    entity_type_label = _entity_type_label(str(entity.get("entity_type") or "entity"))
    name = str(entity.get("canonical_name") or "")
    aliases = [alias for alias in entity.get("aliases") or [] if alias != entity.get("canonical_name")]
    alias_text = f" 别名：{', '.join(aliases[:4])}。" if aliases else ""
    author_profile_summary = _compact_author_profile_summary(entity)
    if not memories:
        profile_parts = [part for part in (author_profile_summary, entity_type_label) if part]
        profile_text = "；".join(profile_parts) or entity_type_label
        return {
            "profile": f"{name}：{profile_text}；检索边界前没有关联到可用的 episodic memory。{alias_text}".strip(),
            "current_state": "检索边界前没有可用状态记录。",
            "author_profile_summary": author_profile_summary,
        }, []

    profile_memories = _select_profile_memories(memories)
    state_memories = memories[-4:] if len(memories) > 4 else memories
    cited_memories = [*profile_memories, *state_memories]
    source_refs = list(dict.fromkeys(reference_binder.add_memory_evidence(memory) for memory in cited_memories))
    profile_text = _join_memory_summaries(profile_memories)
    state_text = _join_memory_summaries(state_memories)
    combined_profile = "；".join(part for part in (author_profile_summary, profile_text or entity_type_label) if part)
    return {
        "profile": f"{name}：{combined_profile}。{alias_text}".replace("。。", "。").strip(),
        "current_state": state_text or "检索边界前没有可用状态记录。",
        "author_profile_summary": author_profile_summary,
    }, source_refs


def _compact_author_profile_summary(entity: dict[str, Any]) -> str:
    profile = entity.get("author_profile") if isinstance(entity.get("author_profile"), dict) else {}
    initial_description = str(entity.get("author_description") or entity.get("initial_description") or "").strip()
    parts: list[str] = []
    if initial_description:
        parts.append(initial_description)
    role = _compact_profile_value(profile.get("role"), limit=2)
    if role:
        parts.append(f"role={role}")
    traits = _compact_profile_value(profile.get("stable_traits"), limit=4)
    if traits:
        parts.append(f"traits={traits}")
    speaking_style = _compact_profile_value(profile.get("speaking_style"), limit=3)
    if speaking_style:
        parts.append(f"speaking_style={speaking_style}")
    values = _compact_profile_value(profile.get("values_or_motivations"), limit=3)
    if values:
        parts.append(f"values={values}")
    constraints = _compact_profile_value(profile.get("behavior_constraints"), limit=3)
    if constraints:
        parts.append(f"constraints={constraints}")
    private_goals = _compact_profile_value(profile.get("private_goals"), limit=2)
    if private_goals:
        parts.append(f"private_goals={private_goals}")
    return "；".join(parts)


def _compact_initial_state_summary(initial_state: dict[str, Any]) -> str:
    if not isinstance(initial_state, dict):
        return ""
    parts: list[str] = []
    for key in ("status", "beliefs", "relationships", "knowledge", "goals"):
        value = _compact_profile_value(initial_state.get(key), limit=3)
        if value:
            parts.append(f"{key}={value}")
    return "；".join(parts)


def _compact_profile_value(value: Any, *, limit: int) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        items = [item for item in (_compact_profile_value(item, limit=limit) for item in value) if item]
        return "、".join(items[:limit])
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            compact = _compact_profile_value(item, limit=limit)
            if compact:
                items.append(f"{key}:{compact}")
        return "、".join(items[:limit])
    return str(value).strip()


def _select_profile_memories(memories: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for memory in memories:
        memory_id = str(memory.get("memory_id") or "")
        summary = str(memory.get("summary") or "")
        if not summary or memory_id in seen_ids:
            continue
        if any(keyword in summary for keyword in ("通过", "前往", "承诺", "负责", "照顾", "认为", "表示", "提醒", "驾驶", "上线", "投放")):
            selected.append(memory)
            seen_ids.add(memory_id)
        if len(selected) >= limit:
            return selected
    for memory in memories:
        memory_id = str(memory.get("memory_id") or "")
        if memory_id and memory_id not in seen_ids and memory.get("summary"):
            selected.append(memory)
            seen_ids.add(memory_id)
        if len(selected) >= limit:
            break
    return selected


def _join_memory_summaries(memories: list[dict[str, Any]]) -> str:
    return "；".join(
        _clean_entity_context_summary(str(memory.get("summary") or ""))
        for memory in memories
        if memory.get("summary")
    )


def _clean_entity_context_summary(summary: str) -> str:
    text = str(summary or "").strip().rstrip("。")
    text = re.sub(r"^张鹏告知逝去的兄嫂，", "", text)
    text = re.sub(r"^一幢六层建筑上覆盖着褪色广告布，显示一群戴墨镜的大猩猩，是", "", text)
    text = text.replace("刘培强在J20C驾驶舱内熟练操作启航按键，准备起飞", "在J20C驾驶舱内熟练操作启航按键，准备起飞")
    text = text.replace("刘培强打开VR头显，战场感知系统上线", "打开VR头显，战场感知系统上线")
    text = text.replace("刘培强表示", "表示")
    text = text.replace("张鹏提醒刘培强", "张鹏提醒他")
    text = text.replace("张鹏询问刘培强", "张鹏询问他")
    text = text.replace("张鹏承诺会照顾好刘培强", "张鹏承诺会照顾好他")
    return text


def _bind_scene_summary_refs(
    summaries: list[dict[str, Any]],
    reference_binder: "_ReferenceBinder",
) -> list[dict[str, Any]]:
    bound = []
    for summary in summaries:
        item = dict(summary)
        item["source_ref"] = reference_binder.add_scene_summary(summary)
        bound.append(item)
    return bound


def _bind_relation_refs(
    relations: list[dict[str, Any]],
    reference_binder: "_ReferenceBinder",
) -> list[dict[str, Any]]:
    bound = []
    for relation in relations:
        item = dict(relation)
        refs = []
        for evidence in relation.get("evidence") or []:
            refs.append(reference_binder.add_relationship_evidence(relation, str(evidence)))
        item["source_refs"] = list(dict.fromkeys(refs))
        bound.append(item)
    return bound


def _compact_scene_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "title": metadata.get("title"),
        "scene_order": metadata.get("scene_order"),
        "setting": metadata.get("setting") or {},
        "stated_facts": (metadata.get("stated_facts") or [])[:5],
        "open_questions": (metadata.get("open_questions") or [])[:3],
        "scene_tags": (metadata.get("scene_tags") or [])[:8],
        "source": metadata.get("source") or {},
    }


class _ReferenceBinder:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], str] = {}
        self.references: list[dict[str, Any]] = []

    def add_memory_evidence(self, memory: dict[str, Any]) -> str:
        memory_id = str(memory.get("memory_id") or "")
        text = str(memory.get("evidence_text") or memory.get("summary") or "")
        key = ("memory", memory_id or text)
        return self._add(
            key,
            {
                "kind": "episodic_memory_evidence",
                "source_id": memory_id,
                "scene_id": memory.get("parent_scene_id"),
                "label": memory_id,
                "text": text,
                "metadata": {
                    "timeline_index": memory.get("timeline_index"),
                    "memory_type": memory.get("memory_type"),
                    "memory_temporal_scope": memory.get("memory_temporal_scope") or "temporal_episode",
                    "evidence_start": memory.get("evidence_start"),
                    "evidence_end": memory.get("evidence_end"),
                    "parent_evidence_start": memory.get("parent_evidence_start"),
                    "parent_evidence_end": memory.get("parent_evidence_end"),
                },
            },
        )

    def add_scene_summary(self, summary: dict[str, Any]) -> str:
        source_id = str(summary.get("source_id") or summary.get("doc_id") or summary.get("scene_id") or "")
        key = ("scene_summary", source_id)
        return self._add(
            key,
            {
                "kind": "scene_summary",
                "source_id": source_id,
                "scene_id": summary.get("scene_id"),
                "label": source_id,
                "text": summary.get("summary") or summary.get("retrieval_text") or "",
                "metadata": {
                    "doc_id": summary.get("doc_id"),
                    "score": summary.get("score"),
                },
            },
        )

    def add_relationship_evidence(self, relation: dict[str, Any], evidence: str) -> str:
        relationship_id = str(relation.get("relationship_id") or "")
        key = ("relationship", f"{relationship_id}:{evidence}")
        return self._add(
            key,
            {
                "kind": "relationship_evidence",
                "source_id": relationship_id,
                "scene_id": relation.get("last_updated_scene") or relation.get("first_seen_scene"),
                "label": relationship_id,
                "text": evidence,
                "metadata": {
                    "source_entity_id": relation.get("source_entity_id"),
                    "target_entity_id": relation.get("target_entity_id"),
                    "relation_type": relation.get("relation_type"),
                },
            },
        )

    def _add(self, key: tuple[str, str], payload: dict[str, Any]) -> str:
        if key in self._by_key:
            return self._by_key[key]
        ref_id = f"R{len(self.references) + 1}"
        self._by_key[key] = ref_id
        record = {"ref_id": ref_id, **payload}
        self.references.append(record)
        return ref_id


def _extract_explicit_entity_refs(intent: str, *, db_path: str | Path | None) -> list[str]:
    refs: list[str] = []
    normalized_intent = _normalize_for_match(intent)
    entities = list_entities(db_path) if db_path else []
    for entity in entities:
        aliases = [entity.get("canonical_name"), *(entity.get("aliases") or [])]
        for alias in aliases:
            alias_text = str(alias or "").strip()
            if not alias_text:
                continue
            normalized_alias = _normalize_for_match(alias_text)
            if len(normalized_alias) < 2:
                continue
            if normalized_alias in normalized_intent:
                _append_unique(refs, alias_text)
                break
            alias_codes = set(_code_tokens(alias_text))
            intent_codes = set(_code_tokens(intent))
            if alias_codes and alias_codes.intersection(intent_codes):
                _append_unique(refs, alias_text)
                break
    for surface in _extract_surface_refs(intent):
        if surface not in refs:
            refs.append(surface)
    return refs


def _extract_surface_refs(intent: str) -> list[str]:
    refs: list[str] = []
    for token in re.findall(r"[A-Za-z]*\d+[A-Za-z0-9]*|[A-Z]{2,}", intent):
        _append_unique(refs, token)
    return refs


def _extract_narrative_units(intent: str, important_entities: list[str]) -> list[str]:
    del important_entities
    units: list[str] = []
    for sentence in re.split(r"[。；;!?！？]", intent):
        for clause in re.split(r"[，,]", sentence):
            unit = _clean_narrative_unit(clause)
            if not unit:
                continue
            _append_unique(units, unit)
    if not units and intent:
        units = [intent.strip()]
    return units[:6]


def _clean_narrative_unit(value: str) -> str:
    unit = str(value or "").strip(" \t\r\n，,。.;；")
    if not unit:
        return ""
    if re.match(r"^为.+(铺垫|埋伏笔|做准备)$", unit):
        return ""
    unit = re.sub(r"^(展现|描写|呈现|表现|刻画)", "", unit).strip()
    unit = re.sub(r"^通过", "", unit).strip()
    unit = unit.replace("则以", "以")
    return unit.strip(" \t\r\n，,。.;；")


def _append_unique(values: list[str], value: str) -> None:
    value = str(value or "").strip()
    if value and value not in values:
        values.append(value)


def _normalize_for_match(value: str) -> str:
    return "".join(
        char.lower()
        for char in str(value or "")
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _code_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]*\d+[A-Za-z0-9]*|[A-Z]{2,}", str(value or ""))]


def _json_loads(value: Any, *, default: Any) -> Any:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return parsed if parsed is not None else default


def _entity_type_label(entity_type: str) -> str:
    return {
        "character": "角色",
        "group": "群体",
        "organization": "组织",
        "location": "地点",
        "object": "重要物品/载具/道具",
        "concept": "概念",
        "occasion": "重要事件/场合",
    }.get(entity_type, entity_type or "实体")


def _format_ref_list(refs: list[str]) -> str:
    return ", ".join(f"[{ref}]" for ref in refs)


def _format_index_list(indexes: list[str]) -> str:
    if not indexes:
        return "none"
    return ", ".join(f"[{index}]" for index in indexes)


def _append_reference_context_markdown(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.append("")
    lines.append(f"## {title}")
    if not items:
        lines.append("- none")
        return
    for item in items:
        subject = str(item.get("subject") or "").strip()
        prefix = f"{subject}: " if subject else ""
        meta = _reference_item_meta(item)
        suffix = f" ({meta})" if meta else ""
        lines.append(f"- {prefix}{item.get('statement') or ''}{suffix}")
        evidence = str(item.get("evidence") or "").strip()
        if evidence:
            lines.append(f"  evidence: {evidence}")


def _reference_item_meta(item: dict[str, Any]) -> str:
    parts = []
    if item.get("item_type"):
        parts.append(str(item.get("item_type")))
    if item.get("knowledge_scope"):
        parts.append(str(item.get("knowledge_scope")))
    if item.get("known_to"):
        parts.append("known_to=" + ", ".join(str(value) for value in item.get("known_to") or []))
    if item.get("available_from") and item.get("available_from") != "unknown":
        parts.append("available_from=" + str(item.get("available_from")))
    return "; ".join(parts)


def _has_reference_context(packet: dict[str, Any]) -> bool:
    return any(
        packet.get(key)
        for key in (
            "author_reference_context",
            "character_reference_knowledge",
            "style_reference_context",
            "timeline_reference_claims",
        )
    )


def _format_reference_text_lines(text: object, *, width: int = 88) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines() or [""]:
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                drop_whitespace=True,
                replace_whitespace=False,
            )
            or [line]
        )
    return lines
