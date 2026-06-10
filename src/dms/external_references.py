from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dms.llm import LLMClient
from dms.reference_library import (
    ChromaReferenceIndexConfig,
    ReferenceFactPropertyExtractionConfig,
    ReferenceKGExtractionConfig,
    ReferenceKnowledgeImportConfig,
    ReferenceKnowledgeQuery,
    ReferenceLibraryIngestConfig,
    build_chroma_reference_index,
    extract_reference_facts_properties,
    extract_reference_kg,
    get_reference_asset_counts,
    import_reference_knowledge,
    ingest_reference_library,
    query_reference_knowledge,
)


@dataclass(frozen=True)
class ExternalReferenceEvidenceBudget:
    """Controls retrieval depth and final evidence packet size."""

    profile: str = "standard"
    retrieval_top_k: int = 8
    retrieval_multiplier: int = 3
    result_limit: int = 8
    max_facts: int = 8
    max_properties: int = 6
    max_relations: int = 6
    max_entities: int = 6
    max_chunks: int = 4
    max_graph_insights: int = 4
    max_source_notes: int = 4
    include_fact_properties: bool = True
    fact_binding_top_k: int = 4
    property_binding_top_k: int = 3
    include_answer_context: bool = True
    legacy_limit: int | None = None


@dataclass(frozen=True)
class ExternalReferencesConfig:
    """mem0-like facade config for source-aware external references."""

    db_path: Path
    work_dir: Path | None = None
    chroma_dir: Path | None = None
    collection_name: str = "dms_reference_knowledge"
    max_chunk_chars: int = 2400
    max_retries: int = 1
    workers: int = 4
    extract_fact_properties: bool = True
    reference_fact_min_entity_degree: int = 2
    reference_fact_max_evidence_chunks_per_job: int = 12
    entity_disambiguation: bool = True
    disambiguation_lexical_threshold: float = 0.88
    auto_index: bool = False
    reset_on_add: bool = True
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalReferences:
    """Small external-reference API inspired by mem0's Memory facade.

    Public callers use add/search/get_all. The LightRAG-style ingest, KG
    extraction, SQLite import, and optional Chroma indexing remain internal
    implementation details.
    """

    def __init__(self, config: ExternalReferencesConfig, *, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client
        self.db_path = Path(config.db_path)
        self.work_dir = Path(config.work_dir) if config.work_dir is not None else None
        self.chroma_dir = Path(config.chroma_dir) if config.chroma_dir is not None else None

    def add(
        self,
        input_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        reset: bool | None = None,
        extract_fact_properties: bool | None = None,
        workers: int | None = None,
        entity_disambiguation: bool | None = None,
        disambiguation_lexical_threshold: float | None = None,
    ) -> dict[str, Any]:
        if not infer:
            raise ValueError("ExternalReferences.add currently requires infer=True to build reference knowledge assets.")
        if self.llm_client is None:
            raise ValueError("llm_client is required for ExternalReferences.add(infer=True)")

        input_path = Path(input_path)
        run_dir, cleanup = self._new_run_dir(input_path)
        library_dir = run_dir / "library"
        kg_dir = run_dir / "kg"
        facts_dir = run_dir / "facts"
        should_extract_fact_properties = (
            self.config.extract_fact_properties
            if extract_fact_properties is None
            else bool(extract_fact_properties)
        )
        worker_count = max(int(self.config.workers if workers is None else workers or 1), 1)
        should_disambiguate_entities = (
            self.config.entity_disambiguation
            if entity_disambiguation is None
            else bool(entity_disambiguation)
        )
        disambiguation_threshold = (
            self.config.disambiguation_lexical_threshold
            if disambiguation_lexical_threshold is None
            else float(disambiguation_lexical_threshold)
        )
        try:
            ingest_summary = ingest_reference_library(
                ReferenceLibraryIngestConfig(
                    input_path=input_path,
                    output_dir=library_dir,
                    max_chunk_chars=self.config.max_chunk_chars,
                    overwrite=True,
                    workers=worker_count,
                )
            )
            kg_summary = extract_reference_kg(
                ReferenceKGExtractionConfig(
                    library_dir=library_dir,
                    output_dir=kg_dir,
                    dry_run=False,
                    overwrite=True,
                    max_retries=self.config.max_retries,
                    workers=worker_count,
                ),
                llm_client=self.llm_client,
            )
            facts_summary = None
            facts_dir_for_import = None
            if should_extract_fact_properties:
                facts_summary = extract_reference_facts_properties(
                    ReferenceFactPropertyExtractionConfig(
                        library_dir=library_dir,
                        kg_dir=kg_dir,
                        output_dir=facts_dir,
                        dry_run=False,
                        overwrite=True,
                        max_retries=self.config.max_retries,
                        workers=worker_count,
                        min_entity_degree=self.config.reference_fact_min_entity_degree,
                        max_evidence_chunks_per_job=self.config.reference_fact_max_evidence_chunks_per_job,
                        entity_disambiguation=should_disambiguate_entities,
                        disambiguation_lexical_threshold=disambiguation_threshold,
                    ),
                    llm_client=self.llm_client,
                )
                facts_dir_for_import = facts_dir
            import_summary = import_reference_knowledge(
                ReferenceKnowledgeImportConfig(
                    library_dir=library_dir,
                    kg_dir=kg_dir,
                    facts_dir=facts_dir_for_import,
                    db_path=self.db_path,
                    reset=self.config.reset_on_add if reset is None else reset,
                    entity_disambiguation=should_disambiguate_entities,
                    disambiguation_lexical_threshold=disambiguation_threshold,
                )
            )
            index_summary = None
            if self.config.auto_index:
                if self.chroma_dir is None:
                    raise ValueError("chroma_dir is required when auto_index=True")
                index_summary = build_chroma_reference_index(
                    ChromaReferenceIndexConfig(
                        db_path=self.db_path,
                        persist_dir=self.chroma_dir,
                        collection_name=self.config.collection_name,
                        reset=True,
                        **self._embedding_kwargs(),
                    )
                )
            return {
                "results": [
                    {
                        "id": str(self.db_path),
                        "event": "ADD",
                        "input_path": str(input_path),
                        "memory": f"Imported {import_summary.get('full_docs', 0)} external reference document(s).",
                        "metadata": {
                            **self.config.metadata,
                            **(metadata or {}),
                            "asset_model": import_summary.get("asset_model"),
                            "work_dir": str(run_dir),
                        },
                    }
                ],
                "summary": {
                    "ingest": ingest_summary,
                    "kg": kg_summary,
                    "facts_properties": facts_summary,
                    "import": import_summary,
                    "index": index_summary,
                    "asset_counts": get_reference_asset_counts(self.db_path),
                },
            }
        finally:
            if cleanup:
                shutil.rmtree(run_dir, ignore_errors=True)

    def search(
        self,
        query: str,
        *,
        evidence_budget: str | dict[str, Any] | ExternalReferenceEvidenceBudget | None = "standard",
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        budget = _resolve_reference_evidence_budget(evidence_budget, limit=limit)
        source_doc_ids, source_paths, source_scope_ids = _source_filters(filters)
        query_plan = _build_reference_query_plan(query, filters=filters)
        pass_specs = _reference_query_passes(query_plan, budget=budget)
        pass_results = []
        for spec in pass_specs:
            result = query_reference_knowledge(
                ReferenceKnowledgeQuery(
                    db_path=self.db_path,
                    query=spec["query"],
                    source_doc_ids=tuple(source_doc_ids),
                    source_paths=tuple(source_paths),
                    source_scope_ids=tuple(source_scope_ids),
                    chroma_dir=self.chroma_dir,
                    collection_name=self.config.collection_name,
                    top_k=spec["top_k"],
                    chunk_top_k=spec["chunk_top_k"],
                    entity_top_k=spec["entity_top_k"],
                    relationship_top_k=spec["relationship_top_k"],
                    include_fact_properties=spec["include_fact_properties"],
                    fact_binding_top_k=spec["fact_binding_top_k"],
                    property_binding_top_k=spec["property_binding_top_k"],
                    **self._embedding_kwargs(),
                )
            )
            pass_results.append({"spec": spec, "result": result})
        result = _fuse_reference_query_results(query=query, pass_results=pass_results, budget=budget)
        evidence_packet = _build_reference_evidence_packet(query=query, query_plan=query_plan, result=result, budget=budget)
        memories = _memory_results_from_reference_query(result, limit=budget.result_limit)
        return {
            "query": query,
            "evidence_budget": _evidence_budget_payload(budget),
            "query_plan": {**query_plan, "passes": pass_specs},
            "source_filter": result.get("source_filter") or {},
            "evidence_packet": evidence_packet,
            "citations": evidence_packet.get("citations") or [],
            "results": memories,
            "relations": result.get("relationships") or [],
            "raw_retrieval": _reference_raw_retrieval_summary(pass_results, result),
            "raw": result,
        }

    def get_all(self) -> dict[str, Any]:
        return {"results": _reference_documents(self.db_path), "counts": get_reference_asset_counts(self.db_path)}

    def reset(self) -> dict[str, str]:
        if self.db_path.exists():
            self.db_path.unlink()
        if self.chroma_dir and self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
        return {"message": "External references reset successfully"}

    def _embedding_kwargs(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.config.embedding_dim,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "embedding_base_url": self.config.embedding_base_url,
            "embedding_api_key": self.config.embedding_api_key,
            "embedding_max_tokens": self.config.embedding_max_tokens,
            "embedding_timeout": self.config.embedding_timeout,
        }

    def _new_run_dir(self, input_path: Path) -> tuple[Path, bool]:
        if self.work_dir is None:
            return Path(tempfile.mkdtemp(prefix="dms_external_refs_")), True
        run_id = _safe_run_id(input_path)
        run_dir = self.work_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, False


def _source_filters(filters: dict[str, Any] | None) -> tuple[list[str], list[str], list[str]]:
    filters = filters or {}
    return (
        _as_list(filters.get("source_doc_id") or filters.get("source_doc_ids")),
        _as_list(filters.get("source_path") or filters.get("source_paths")),
        _as_list(filters.get("source_scope_id") or filters.get("source_scope_ids")),
    )


def _resolve_reference_evidence_budget(
    evidence_budget: str | dict[str, Any] | ExternalReferenceEvidenceBudget | None,
    *,
    limit: int | None = None,
) -> ExternalReferenceEvidenceBudget:
    if isinstance(evidence_budget, ExternalReferenceEvidenceBudget):
        budget = evidence_budget
    else:
        if evidence_budget is None:
            profile = "standard"
            overrides: dict[str, Any] = {}
        elif isinstance(evidence_budget, str):
            profile = evidence_budget.strip().lower() or "standard"
            overrides = {}
        elif isinstance(evidence_budget, dict):
            profile = str(evidence_budget.get("profile") or "standard").strip().lower()
            overrides = dict(evidence_budget)
            overrides.pop("profile", None)
        else:
            raise TypeError("evidence_budget must be 'compact', 'standard', 'deep', a dict, or ExternalReferenceEvidenceBudget")
        if profile not in _REFERENCE_EVIDENCE_BUDGETS:
            raise ValueError(f"Unknown evidence_budget profile: {profile}")
        budget = ExternalReferenceEvidenceBudget(profile=profile, **_REFERENCE_EVIDENCE_BUDGETS[profile])
        if overrides:
            budget = _replace_evidence_budget(budget, overrides)
    if limit is not None:
        legacy_limit = max(int(limit or 0), 0)
        return _replace_evidence_budget(
            budget,
            {
                "profile": "legacy_limit",
                "retrieval_top_k": legacy_limit,
                "result_limit": legacy_limit,
                "max_facts": legacy_limit,
                "max_properties": legacy_limit,
                "max_relations": legacy_limit,
                "max_entities": legacy_limit,
                "max_chunks": legacy_limit,
                "max_graph_insights": legacy_limit,
                "max_source_notes": legacy_limit,
                "legacy_limit": legacy_limit,
            },
        )
    return budget


def _replace_evidence_budget(budget: ExternalReferenceEvidenceBudget, values: dict[str, Any]) -> ExternalReferenceEvidenceBudget:
    data = {
        "profile": budget.profile,
        "retrieval_top_k": budget.retrieval_top_k,
        "retrieval_multiplier": budget.retrieval_multiplier,
        "result_limit": budget.result_limit,
        "max_facts": budget.max_facts,
        "max_properties": budget.max_properties,
        "max_relations": budget.max_relations,
        "max_entities": budget.max_entities,
        "max_chunks": budget.max_chunks,
        "max_graph_insights": budget.max_graph_insights,
        "max_source_notes": budget.max_source_notes,
        "include_fact_properties": budget.include_fact_properties,
        "fact_binding_top_k": budget.fact_binding_top_k,
        "property_binding_top_k": budget.property_binding_top_k,
        "include_answer_context": budget.include_answer_context,
        "legacy_limit": budget.legacy_limit,
    }
    aliases = {
        "max_results": "result_limit",
        "limit": "result_limit",
        "top_k": "retrieval_top_k",
        "answer_context": "include_answer_context",
    }
    for key, value in values.items():
        normalized = aliases.get(str(key), str(key))
        if normalized not in data:
            continue
        data[normalized] = value
    int_fields = {
        "retrieval_top_k",
        "retrieval_multiplier",
        "result_limit",
        "max_facts",
        "max_properties",
        "max_relations",
        "max_entities",
        "max_chunks",
        "max_graph_insights",
        "max_source_notes",
        "fact_binding_top_k",
        "property_binding_top_k",
    }
    for field_name in int_fields:
        data[field_name] = max(int(data[field_name] or 0), 0)
    data["include_fact_properties"] = bool(data["include_fact_properties"])
    data["include_answer_context"] = bool(data["include_answer_context"])
    if data["retrieval_multiplier"] <= 0:
        data["retrieval_multiplier"] = 1
    if data.get("legacy_limit") is not None:
        data["legacy_limit"] = max(int(data["legacy_limit"] or 0), 0)
    data["profile"] = str(data["profile"] or "custom")
    return ExternalReferenceEvidenceBudget(**data)


def _evidence_budget_payload(budget: ExternalReferenceEvidenceBudget) -> dict[str, Any]:
    return {
        "profile": budget.profile,
        "retrieval_top_k": budget.retrieval_top_k,
        "retrieval_multiplier": budget.retrieval_multiplier,
        "result_limit": budget.result_limit,
        "max_facts": budget.max_facts,
        "max_properties": budget.max_properties,
        "max_relations": budget.max_relations,
        "max_entities": budget.max_entities,
        "max_chunks": budget.max_chunks,
        "max_graph_insights": budget.max_graph_insights,
        "max_source_notes": budget.max_source_notes,
        "include_fact_properties": budget.include_fact_properties,
        "fact_binding_top_k": budget.fact_binding_top_k,
        "property_binding_top_k": budget.property_binding_top_k,
        "include_answer_context": budget.include_answer_context,
        "legacy_limit": budget.legacy_limit,
    }


def _build_reference_query_plan(query: str, *, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    low_level_keywords = _extract_low_level_keywords(query)
    high_level_keywords = _extract_high_level_keywords(query, low_level_keywords=low_level_keywords)
    if low_level_keywords and high_level_keywords:
        mode = "mix"
    elif high_level_keywords:
        mode = "high_level"
    else:
        mode = "low_level"
    return {
        "strategy": "dms_source_aware_lightrag_assets_v1",
        "mode": mode,
        "query": str(query or ""),
        "low_level_keywords": low_level_keywords,
        "high_level_keywords": high_level_keywords,
        "source_filter_request": _source_filter_request(filters),
        "execution": "multi_pass_namespace_weighted",
    }


def _extract_low_level_keywords(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    normalized = text
    for word in _LOW_LEVEL_SPLIT_WORDS:
        normalized = normalized.replace(word, " ")
    normalized = re.sub(r"[?？!！,，.。;；:：\[\]【】()（）<>《》\"'“”‘’/\\|]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    candidates: list[str] = []
    for token in normalized.split():
        token = _clean_keyword(token)
        if not token or token in _LOW_LEVEL_STOPWORDS:
            continue
        if _looks_like_query_filler(token):
            continue
        candidates.append(token)
        candidates.extend(_low_level_keyword_subterms(token))
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|\d+[A-Za-z]?", text):
        token = _clean_keyword(token)
        if token and token not in _LOW_LEVEL_STOPWORDS:
            candidates.append(token)
    return _dedupe_ordered(candidates)[:8]


def _low_level_keyword_subterms(token: str) -> list[str]:
    terms: list[str] = []
    if "的" in token:
        terms.extend(part for part in token.split("的") if part)
    for suffix in ("方式", "情况", "信息", "内容", "资料"):
        if token.endswith(suffix) and len(token) > len(suffix):
            terms.append(token[: -len(suffix)])
    return [
        term
        for term in (_clean_keyword(item) for item in terms)
        if term and term not in _LOW_LEVEL_STOPWORDS and not _looks_like_query_filler(term)
    ]


def _extract_high_level_keywords(query: str, *, low_level_keywords: list[str]) -> list[str]:
    text = str(query or "")
    keywords: list[str] = []
    for cue, normalized in _HIGH_LEVEL_CUE_KEYWORDS:
        if cue in text:
            keywords.extend(normalized)
    lowered = text.lower()
    for cue, normalized in _HIGH_LEVEL_ASCII_CUE_KEYWORDS:
        if cue in lowered:
            keywords.extend(normalized)
    if len(low_level_keywords) >= 2 and not keywords:
        keywords.append("关系")
    return _dedupe_ordered(_clean_keyword(item) for item in keywords if _clean_keyword(item))[:8]


def _source_filter_request(filters: dict[str, Any] | None) -> dict[str, list[str]]:
    source_doc_ids, source_paths, source_scope_ids = _source_filters(filters)
    return {
        "source_doc_ids": source_doc_ids,
        "source_paths": source_paths,
        "source_scope_ids": source_scope_ids,
    }


def _reference_query_passes(query_plan: dict[str, Any], *, budget: ExternalReferenceEvidenceBudget) -> list[dict[str, Any]]:
    retrieval_top_k = max(int(budget.retrieval_top_k or 0), 0)
    if retrieval_top_k <= 0:
        return []
    low_keywords = list(query_plan.get("low_level_keywords") or [])
    high_keywords = list(query_plan.get("high_level_keywords") or [])
    query = str(query_plan.get("query") or "")
    narrow = max((retrieval_top_k + 1) // 2, 1)
    wide = max(retrieval_top_k * max(int(budget.retrieval_multiplier or 1), 1), retrieval_top_k, 1)
    passes = [
        {
            "name": "base",
            "focus": "balanced_original_query",
            "query": query,
            "query_terms": [],
            "top_k": retrieval_top_k,
            "chunk_top_k": retrieval_top_k,
            "entity_top_k": retrieval_top_k,
            "relationship_top_k": retrieval_top_k,
            "include_fact_properties": budget.include_fact_properties,
            "fact_binding_top_k": budget.fact_binding_top_k,
            "property_binding_top_k": budget.property_binding_top_k,
        }
    ]
    if low_keywords:
        passes.append(
            {
                "name": "low_level",
                "focus": "entity_and_chunk_anchors",
                "query": _keyword_query(low_keywords, fallback=query),
                "query_terms": low_keywords,
                "top_k": retrieval_top_k,
                "chunk_top_k": retrieval_top_k,
                "entity_top_k": wide,
                "relationship_top_k": narrow,
                "include_fact_properties": budget.include_fact_properties,
                "fact_binding_top_k": budget.fact_binding_top_k,
                "property_binding_top_k": budget.property_binding_top_k,
            }
        )
    if high_keywords:
        passes.append(
            {
                "name": "high_level",
                "focus": "relationship_fact_and_graph_context",
                "query": query,
                "query_terms": high_keywords,
                "top_k": wide,
                "chunk_top_k": narrow,
                "entity_top_k": retrieval_top_k,
                "relationship_top_k": wide,
                "include_fact_properties": budget.include_fact_properties,
                "fact_binding_top_k": budget.fact_binding_top_k,
                "property_binding_top_k": budget.property_binding_top_k,
            }
        )
    return passes


def _keyword_query(keywords: list[str], *, fallback: str) -> str:
    text = " ".join(_dedupe_ordered(_clean_keyword(item) for item in keywords if _clean_keyword(item))).strip()
    return text or str(fallback or "")


def _fuse_reference_query_results(
    *,
    query: str,
    pass_results: list[dict[str, Any]],
    budget: ExternalReferenceEvidenceBudget,
) -> dict[str, Any]:
    field_specs = {
        "matched_clusters": ("cluster_id",),
        "entities": ("source_local_entity_id", "entity_name"),
        "relationships": ("source_local_relation_id", "relation_id", "src_id", "tgt_id"),
        "relationship_facts": ("fact_id", "statement"),
        "atomic_facts": ("fact_id", "statement"),
        "entity_properties": ("property_id", "statement"),
        "pseudo_relationships": ("pseudo_relation_id", "src_display_name", "tgt_display_name"),
        "chunks": ("chunk_id",),
    }
    fused_limit = max(
        budget.retrieval_top_k * budget.retrieval_multiplier,
        budget.max_facts,
        budget.max_properties,
        budget.max_relations,
        budget.max_entities,
        budget.max_chunks,
        1,
    )
    fused: dict[str, Any] = {
        "query": query,
        "mode": "source_local",
        "retrieval_strategy": "query_plan_multi_pass_source_local",
        "count": 0,
    }
    for field, key_fields in field_specs.items():
        fused[field] = _merge_reference_items(pass_results, field=field, key_fields=key_fields, limit=fused_limit)
    source_filter = _merge_reference_source_filters([item.get("result", {}).get("source_filter") for item in pass_results])
    fused["source_filter"] = source_filter
    fused["evidence_board"] = {
        "source_filter": source_filter,
        "matched_clusters": fused["matched_clusters"],
        "source_local_entities": fused["entities"],
        "source_local_relationships": fused["relationships"],
        "relationship_facts": fused["relationship_facts"],
        "atomic_facts": fused["atomic_facts"],
        "entity_properties": fused["entity_properties"],
        "supporting_chunks": fused["chunks"],
        "pseudo_relationships": fused["pseudo_relationships"],
        "graph_insights": _merge_reference_board_items(pass_results, "graph_insights", ("type", "cluster_id")),
        "source_separation_notices": _merge_reference_board_items(
            pass_results,
            "source_separation_notices",
            ("cluster_id", "notice"),
        ),
    }
    fused["count"] = sum(len(fused.get(field) or []) for field in field_specs)
    asset_model = next((item.get("result", {}).get("asset_model") for item in pass_results if item.get("result", {}).get("asset_model")), None)
    if asset_model:
        fused["asset_model"] = asset_model
    return fused


def _merge_reference_items(
    pass_results: list[dict[str, Any]],
    *,
    field: str,
    key_fields: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for pass_index, pass_row in enumerate(pass_results):
        spec = pass_row.get("spec") or {}
        pass_name = str(spec.get("name") or f"pass_{pass_index + 1}")
        for rank, item in enumerate(pass_row.get("result", {}).get(field) or []):
            key = _reference_item_key(item, key_fields)
            if not key:
                continue
            candidate = dict(item)
            candidate_score = _safe_float(candidate.get("score")) + max(0.0, 0.001 * (limit - rank))
            candidate["score"] = round(candidate_score, 6)
            current = selected.get(key)
            if current is None:
                candidate["retrieval_passes"] = [pass_name]
                selected[key] = candidate
                continue
            passes = _dedupe_ordered(list(current.get("retrieval_passes") or []) + [pass_name])
            if candidate_score > _safe_float(current.get("score")):
                candidate["retrieval_passes"] = passes
                selected[key] = candidate
            else:
                current["retrieval_passes"] = passes
                current["score"] = round(_safe_float(current.get("score")) + 0.025, 6)
    return sorted(
        selected.values(),
        key=lambda item: (
            _safe_float(item.get("score")),
            len(item.get("retrieval_passes") or []),
            str(_reference_item_key(item, key_fields)),
        ),
        reverse=True,
    )[: max(int(limit or 0), 0)]


def _reference_item_key(item: dict[str, Any], fields: tuple[str, ...]) -> str:
    values = [str(item.get(field) or "").strip() for field in fields if str(item.get(field) or "").strip()]
    if values:
        return "|".join(values)
    return str(item.get("id") or item.get("statement") or item.get("description") or item.get("content") or "").strip()


def _merge_reference_source_filters(filters: list[dict[str, Any] | None]) -> dict[str, Any]:
    modes = [str(item.get("mode") or "") for item in filters if item]
    mode = "all"
    if "files" in modes:
        mode = "files"
    elif "source_scopes" in modes:
        mode = "source_scopes"
    return {
        "mode": mode,
        "source_scope_ids": _dedupe_ordered(scope for item in filters if item for scope in (item.get("source_scope_ids") or [])),
        "source_doc_ids": _dedupe_ordered(doc for item in filters if item for doc in (item.get("source_doc_ids") or [])),
        "source_paths": _dedupe_ordered(path for item in filters if item for path in (item.get("source_paths") or [])),
    }


def _merge_reference_board_items(
    pass_results: list[dict[str, Any]],
    board_field: str,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for pass_row in pass_results:
        board = pass_row.get("result", {}).get("evidence_board") or {}
        for item in board.get(board_field) or []:
            key = _reference_item_key(item, key_fields)
            if key and key not in selected:
                selected[key] = dict(item)
    return list(selected.values())


def _build_reference_evidence_packet(
    *,
    query: str,
    query_plan: dict[str, Any],
    result: dict[str, Any],
    budget: ExternalReferenceEvidenceBudget,
) -> dict[str, Any]:
    facts = _evidence_items_from_payloads(
        list(result.get("relationship_facts") or []) + list(result.get("atomic_facts") or []),
        prefix="F",
        item_type="fact",
        limit=budget.max_facts,
    )
    properties = _evidence_items_from_payloads(
        _displayable_reference_properties(result.get("entity_properties") or []),
        prefix="P",
        item_type="property",
        limit=budget.max_properties,
    )
    relations = _evidence_items_from_payloads(
        result.get("relationships") or [],
        prefix="R",
        item_type="relationship",
        limit=budget.max_relations,
    )
    entities = _evidence_items_from_payloads(
        result.get("entities") or [],
        prefix="E",
        item_type="entity",
        limit=budget.max_entities,
    )
    chunks = _evidence_items_from_payloads(
        _supporting_reference_packet_chunks(
            result.get("chunks") or [],
            facts=facts,
            relations=relations,
            entities=entities,
            query_plan=query_plan,
        ),
        prefix="C",
        item_type="chunk",
        limit=budget.max_chunks,
    )
    graph_insights = _evidence_items_from_payloads(
        (result.get("evidence_board") or {}).get("graph_insights") or [],
        prefix="G",
        item_type="graph_insight",
        limit=budget.max_graph_insights,
    )
    source_notes = _evidence_items_from_payloads(
        (result.get("evidence_board") or {}).get("source_separation_notices") or [],
        prefix="S",
        item_type="source_note",
        limit=budget.max_source_notes,
    )
    citations = []
    for group in (facts, properties, relations, entities, chunks):
        citations.extend(_citation_from_evidence_item(item) for item in group)
    packet = {
        "query": query,
        "strategy": "query_plan_multi_pass_evidence_packet_v1",
        "evidence_budget": _evidence_budget_payload(budget),
        "query_plan": query_plan,
        "source_filter": result.get("source_filter") or {},
        "facts": facts,
        "properties": properties,
        "relations": relations,
        "entities": entities,
        "chunks": chunks,
        "graph_insights": graph_insights,
        "source_notes": source_notes,
        "citations": citations,
    }
    packet["answer_context"] = _build_reference_answer_context(packet, include=budget.include_answer_context)
    return packet


def _evidence_items_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    prefix: str,
    item_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    if int(limit or 0) <= 0:
        return []
    items = []
    seen = set()
    for raw_payload in payloads:
        payload = _clean_reference_packet_value(raw_payload)
        key = _reference_item_key(
            payload,
            (
                "fact_id",
                "property_id",
                "source_local_relation_id",
                "source_local_entity_id",
                "chunk_id",
                "cluster_id",
                "statement",
                "description",
                "notice",
            ),
        )
        if not key or key in seen:
            continue
        seen.add(key)
        ref_id = f"{prefix}{len(items) + 1}"
        item = {
            "ref_id": ref_id,
            "type": item_type,
            "statement": _reference_statement(payload, item_type=item_type),
            "score": payload.get("score"),
            "retrieval_passes": payload.get("retrieval_passes") or [],
            "source_refs": _source_refs_for_payload(payload),
            "raw": payload,
        }
        if item_type == "chunk":
            content = _clean_reference_display_text(payload.get("content"))
            item["content"] = content
            item["content_chars"] = len(content)
        items.append(item)
        if len(items) >= max(int(limit or 0), 0):
            break
    return items


def _reference_statement(payload: dict[str, Any], *, item_type: str) -> str:
    if item_type == "relationship":
        endpoints = " - ".join(part for part in (str(payload.get("src_id") or "").strip(), str(payload.get("tgt_id") or "").strip()) if part)
        description = _clean_reference_display_text(payload.get("description"))
        return _clean_reference_display_text(f"{endpoints}: {description}" if endpoints and description else description or endpoints)
    if item_type == "entity":
        name = str(payload.get("entity_name") or payload.get("canonical_entity_name") or "").strip()
        entity_type = str(payload.get("entity_type") or "").strip()
        description = _clean_reference_display_text(payload.get("description"))
        label = f"{name} ({entity_type})" if name and entity_type else name
        return _clean_reference_display_text(f"{label}: {description}" if label and description else description or label)
    if item_type == "property":
        entity = str(payload.get("entity_name") or "").strip()
        prop = str(payload.get("property_name") or "").strip()
        value = _clean_reference_display_text(payload.get("property_value") or payload.get("statement"))
        label = ".".join(part for part in (entity, prop) if part)
        return _clean_reference_display_text(f"{label}: {value}" if label and value else value or label)
    if item_type == "chunk":
        return _clean_reference_display_text(payload.get("content"))
    if item_type == "graph_insight":
        name = str(payload.get("canonical_display_name") or payload.get("cluster_id") or "").strip()
        degree = payload.get("global_pseudo_degree")
        coverage = payload.get("source_coverage")
        return f"{name}: graph degree={degree}, source coverage={coverage}"
    if item_type == "source_note":
        name = str(payload.get("canonical_display_name") or payload.get("cluster_id") or "").strip()
        note = _clean_reference_display_text(payload.get("notice"))
        return _clean_reference_display_text(f"{name}: {note}" if name and note else note or name)
    return _clean_reference_display_text(payload.get("statement") or payload.get("description") or payload.get("content"))


def _clean_reference_display_text(value: Any) -> str:
    return str(value or "").replace("<SEP>", "；").strip()


def _clean_reference_packet_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_reference_display_text(value)
    if isinstance(value, list):
        return [_clean_reference_packet_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clean_reference_packet_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clean_reference_packet_value(item) for key, item in value.items()}
    return value


def _displayable_reference_properties(properties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        prop
        for prop in properties
        if str(prop.get("property_name") or "").strip().lower() not in {"entity_type", "type", "category"}
        and str(prop.get("statement") or "").strip()
    ]


def _supporting_reference_packet_chunks(
    chunks: list[dict[str, Any]],
    *,
    facts: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    query_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    query_terms = [
        str(term).strip()
        for term in [*(query_plan.get("low_level_keywords") or []), *(query_plan.get("high_level_keywords") or [])]
        if str(term).strip()
    ]
    primary_items = _query_relevant_evidence_items([*facts, *relations], query_terms)
    fallback_items = _query_relevant_evidence_items(entities, query_terms)
    primary_supported_chunk_ids = _evidence_source_chunk_ids(primary_items)
    supported_chunk_ids = primary_supported_chunk_ids or _evidence_source_chunk_ids(fallback_items)
    selected = []
    bound_chunks = _chunk_payloads_from_evidence_items(primary_items or fallback_items)
    for chunk in bound_chunks:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("source_chunk_id") or "")
        if chunk_id and chunk_id in supported_chunk_ids:
            selected.append(chunk)
    if not selected:
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            if query_terms and any(term in content for term in query_terms):
                selected.append(chunk)
    deduped: dict[str, dict[str, Any]] = {}
    for chunk in selected:
        key = str(chunk.get("chunk_id") or chunk.get("source_chunk_id") or chunk.get("content_sha256") or chunk.get("content") or "")
        if key and key not in deduped:
            deduped[key] = chunk
    return list(deduped.values())


def _chunk_payloads_from_evidence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for item in items:
        for ref in item.get("source_refs") or []:
            chunk_id = str(ref.get("source_chunk_id") or "").strip()
            content = str(ref.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "full_doc_id": ref.get("source_doc_id"),
                    "source_scope_id": ref.get("source_scope_id"),
                    "source_path": ref.get("source_path"),
                    "relative_path": ref.get("relative_path"),
                    "title": ref.get("title"),
                    "heading": ref.get("heading"),
                    "content": content,
                    "score": item.get("score"),
                    "retrieval_passes": item.get("retrieval_passes") or [],
                }
            )
    return chunks


def _query_relevant_evidence_items(items: list[dict[str, Any]], query_terms: list[str]) -> list[dict[str, Any]]:
    if not query_terms:
        return items
    selected = []
    for item in items:
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        haystack = "\n".join(
            str(value or "")
            for value in (
                item.get("statement"),
                raw.get("statement"),
                raw.get("description"),
                raw.get("src_id"),
                raw.get("tgt_id"),
                raw.get("entity_name"),
                raw.get("subject"),
                raw.get("object"),
            )
        )
        if any(term in haystack for term in query_terms):
            selected.append(item)
    return selected


def _evidence_source_chunk_ids(items: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for item in items:
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else item
        for key in ("source_chunk_id", "chunk_id"):
            value = str(raw.get(key) or "").strip()
            if value:
                values.add(value)
        for chunk_id in raw.get("source_chunk_ids") or []:
            value = str(chunk_id or "").strip()
            if value:
                values.add(value)
        for ref in item.get("source_refs") or []:
            value = str(ref.get("source_chunk_id") or "").strip()
            if value:
                values.add(value)
        for chunk in raw.get("evidence_chunks") or []:
            value = str(chunk.get("chunk_id") or "").strip()
            if value:
                values.add(value)
    return values


def _source_refs_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for chunk in payload.get("evidence_chunks") or []:
        refs.append(
            {
                "source_doc_id": chunk.get("source_doc_id"),
                "source_scope_id": chunk.get("source_scope_id"),
                "source_chunk_id": chunk.get("chunk_id"),
                "source_path": chunk.get("source_path"),
                "relative_path": chunk.get("relative_path"),
                "title": chunk.get("title"),
                "heading": chunk.get("heading"),
                "content": chunk.get("content"),
            }
        )
    if payload.get("chunk_id") or payload.get("source_chunk_id"):
        refs.append(
            {
                "source_doc_id": payload.get("full_doc_id") or payload.get("doc_id") or payload.get("source_doc_id"),
                "source_scope_id": payload.get("source_scope_id"),
                "source_chunk_id": payload.get("chunk_id") or payload.get("source_chunk_id"),
                "source_path": payload.get("source_path") or payload.get("file_path"),
                "relative_path": payload.get("relative_path"),
                "title": payload.get("title"),
                "heading": payload.get("heading"),
                "content": payload.get("content") or payload.get("statement") or "",
            }
        )
    for doc_id in payload.get("source_doc_ids") or []:
        refs.append(
            {
                "source_doc_id": doc_id,
                "source_scope_id": payload.get("source_scope_id"),
                "source_chunk_id": None,
                "source_path": payload.get("file_path"),
                "relative_path": None,
                "title": None,
                "heading": None,
                "content": "",
            }
        )
    if payload.get("source_doc_id"):
        refs.append(
            {
                "source_doc_id": payload.get("source_doc_id"),
                "source_scope_id": payload.get("source_scope_id"),
                "source_chunk_id": payload.get("source_chunk_id"),
                "source_path": payload.get("file_path"),
                "relative_path": None,
                "title": None,
                "heading": None,
                "content": payload.get("statement") or payload.get("description") or "",
            }
        )
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for ref in refs:
        key = (
            str(ref.get("source_doc_id") or ""),
            str(ref.get("source_scope_id") or ""),
            str(ref.get("source_chunk_id") or ""),
            str(ref.get("source_path") or ""),
        )
        if any(key) and key not in deduped:
            deduped[key] = {key_: value for key_, value in ref.items() if value not in (None, "")}
    return list(deduped.values())


def _citation_from_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    source_refs = item.get("source_refs") or []
    return {
        "ref_id": item.get("ref_id"),
        "type": item.get("type"),
        "source_doc_ids": _dedupe_ordered(ref.get("source_doc_id") for ref in source_refs if ref.get("source_doc_id")),
        "source_scope_ids": _dedupe_ordered(ref.get("source_scope_id") for ref in source_refs if ref.get("source_scope_id")),
        "source_chunk_ids": _dedupe_ordered(ref.get("source_chunk_id") for ref in source_refs if ref.get("source_chunk_id")),
        "source_paths": _dedupe_ordered(ref.get("source_path") or ref.get("relative_path") for ref in source_refs if ref.get("source_path") or ref.get("relative_path")),
        "statement": item.get("statement"),
    }


def _format_reference_answer_context(packet: dict[str, Any]) -> str:
    sections = []
    section_specs = [
        ("Facts", packet.get("facts") or []),
        ("Properties", packet.get("properties") or []),
        ("Relations", packet.get("relations") or []),
        ("Entities", packet.get("entities") or []),
        ("Chunks", packet.get("chunks") or []),
        ("Graph Insights", packet.get("graph_insights") or []),
        ("Source Notes", packet.get("source_notes") or []),
    ]
    for title, items in section_specs:
        if not items:
            continue
        lines = [f"{title}:"]
        for item in items:
            source_refs = _format_source_refs(item.get("source_refs") or [])
            suffix = f" {source_refs}" if source_refs else ""
            lines.append(f"- [{item.get('ref_id')}] {_format_reference_context_statement(item)}{suffix}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _format_reference_context_statement(item: dict[str, Any]) -> str:
    if item.get("type") == "chunk":
        return str(item.get("content") or item.get("statement") or "").strip()
    return _clean_reference_display_text(item.get("statement"))


def _build_reference_answer_context(packet: dict[str, Any], *, include: bool) -> str:
    return _format_reference_answer_context(packet) if include else ""


def _format_source_refs(source_refs: list[dict[str, Any]]) -> str:
    labels = []
    for ref in source_refs:
        label = ref.get("source_chunk_id") or ref.get("source_doc_id") or ref.get("source_path") or ref.get("relative_path")
        if label:
            labels.append(str(label))
    return f"(sources: {', '.join(_dedupe_ordered(labels))})" if labels else ""


def _reference_raw_retrieval_summary(pass_results: list[dict[str, Any]], fused_result: dict[str, Any]) -> dict[str, Any]:
    passes = []
    for pass_row in pass_results:
        spec = dict(pass_row.get("spec") or {})
        result = pass_row.get("result") or {}
        passes.append(
            {
                **spec,
                "retrieval_strategy": result.get("retrieval_strategy"),
                "source_filter": result.get("source_filter") or {},
                "counts": {
                    "entities": len(result.get("entities") or []),
                    "relationships": len(result.get("relationships") or []),
                    "relationship_facts": len(result.get("relationship_facts") or []),
                    "atomic_facts": len(result.get("atomic_facts") or []),
                    "properties": len(result.get("entity_properties") or []),
                    "chunks": len(result.get("chunks") or []),
                },
            }
        )
    return {
        "strategy": "query_plan_multi_pass_source_local",
        "passes": passes,
        "fused_counts": {
            "entities": len(fused_result.get("entities") or []),
            "relationships": len(fused_result.get("relationships") or []),
            "relationship_facts": len(fused_result.get("relationship_facts") or []),
            "atomic_facts": len(fused_result.get("atomic_facts") or []),
            "properties": len(fused_result.get("entity_properties") or []),
            "chunks": len(fused_result.get("chunks") or []),
        },
    }


def _clean_keyword(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().strip("，,。；;：:？?!！")).strip()


def _looks_like_query_filler(token: str) -> bool:
    return any(filler in token for filler in _LOW_LEVEL_FILLER_FRAGMENTS) and len(token) <= 8


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_ordered(values: Any) -> list[Any]:
    deduped = []
    seen = set()
    for value in values:
        if value is None:
            continue
        key = str(value)
        if not key.strip() or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)] if str(value or "").strip() else []


def _memory_results_from_reference_query(result: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.get("relationship_facts") or []:
        rows.append(_memory_result(item.get("fact_id"), item.get("statement"), "relationship_fact", item))
    for item in result.get("atomic_facts") or []:
        rows.append(_memory_result(item.get("fact_id"), item.get("statement"), "atomic_fact", item))
    for item in result.get("relationships") or []:
        rows.append(_memory_result(item.get("source_local_relation_id"), item.get("description"), "relationship", item))
    for item in result.get("entities") or []:
        rows.append(_memory_result(item.get("source_local_entity_id"), item.get("description"), "entity", item))
    for item in result.get("chunks") or []:
        rows.append(_memory_result(item.get("chunk_id"), item.get("content"), "chunk", item))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("id") or row.get("memory") or "")
        if key and key not in deduped:
            deduped[key] = row
    return list(deduped.values())[: max(int(limit or 0), 0)]


def _memory_result(memory_id: Any, memory: Any, memory_type: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(memory_id or ""),
        "memory": str(memory or ""),
        "type": memory_type,
        "score": raw.get("score"),
        "metadata": {
            "source_type": "external_reference",
            "source_role": memory_type,
            "source_doc_ids": raw.get("source_doc_ids") or ([raw.get("source_doc_id")] if raw.get("source_doc_id") else []),
            "source_scope_id": raw.get("source_scope_id"),
            "source_chunk_ids": raw.get("source_chunk_ids") or ([raw.get("source_chunk_id")] if raw.get("source_chunk_id") else []),
            "raw": raw,
        },
    }


def _reference_documents(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doc_id, source_scope_id, source_scope_name, source_path, relative_path,
                   file_name, format, title, content_sha256, raw_json
            FROM reference_full_docs
            ORDER BY doc_id
            """
        ).fetchall()
    return [_document_result(dict(row)) for row in rows]


def _document_result(row: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads(str(row.pop("raw_json") or "{}"))
    doc_id = str(row.get("doc_id") or "")
    return {
        "id": doc_id,
        "memory": str(row.get("title") or row.get("file_name") or doc_id),
        "type": "source_document",
        "metadata": {**row, "raw": metadata},
    }


def _safe_run_id(input_path: Path) -> str:
    import hashlib
    import re

    key = str(input_path.resolve() if input_path.exists() else input_path)
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", input_path.stem or input_path.name or "reference")
    return f"{slug}_{hashlib.sha1(key.encode('utf-8')).hexdigest()}"


_LOW_LEVEL_SPLIT_WORDS = (
    "和",
    "与",
    "跟",
    "及",
    "以及",
    "还有",
    "关于",
    "有关",
    "之间",
    "是什么",
    "什么",
    "哪些",
    "哪位",
    "是谁",
    "如何",
    "怎么",
    "为什么",
    "请",
    "告诉我",
    "说明",
    "解释",
)

_LOW_LEVEL_STOPWORDS = {
    "关系",
    "关联",
    "相关",
    "资料",
    "信息",
    "内容",
    "外部",
    "参考",
    "reference",
    "refs",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
}

_LOW_LEVEL_FILLER_FRAGMENTS = (
    "什么",
    "哪些",
    "如何",
    "怎么",
    "为什么",
    "是否",
    "有没有",
    "请问",
)

_HIGH_LEVEL_CUE_KEYWORDS = (
    ("关系", ("关系", "关联")),
    ("关联", ("关联",)),
    ("相关", ("相关",)),
    ("指导", ("指导", "训练")),
    ("训练", ("训练",)),
    ("加入", ("加入", "组织关系")),
    ("属于", ("所属",)),
    ("身份", ("身份",)),
    ("性格", ("性格", "人物属性")),
    ("属性", ("属性",)),
    ("设定", ("设定",)),
    ("时间", ("时间", "时间线")),
    ("时间线", ("时间线",)),
    ("什么时候", ("时间", "时间线")),
    ("哪年", ("时间", "时间线")),
    ("何时", ("时间", "时间线")),
    ("先后", ("时间顺序",)),
    ("导致", ("因果",)),
    ("原因", ("因果",)),
    ("地点", ("地点",)),
    ("位置", ("地点",)),
    ("组织", ("组织关系",)),
    ("阵营", ("组织关系",)),
)

_HIGH_LEVEL_ASCII_CUE_KEYWORDS = (
    ("relationship", ("relationship",)),
    ("relation", ("relationship",)),
    ("timeline", ("timeline",)),
    ("when", ("timeline",)),
    ("where", ("location",)),
    ("identity", ("identity",)),
    ("profile", ("profile",)),
    ("trait", ("trait",)),
)

_REFERENCE_EVIDENCE_BUDGETS: dict[str, dict[str, Any]] = {
    "compact": {
        "retrieval_top_k": 4,
        "retrieval_multiplier": 2,
        "result_limit": 4,
        "max_facts": 4,
        "max_properties": 3,
        "max_relations": 3,
        "max_entities": 4,
        "max_chunks": 2,
        "max_graph_insights": 2,
        "max_source_notes": 2,
        "include_fact_properties": True,
        "fact_binding_top_k": 3,
        "property_binding_top_k": 2,
        "include_answer_context": True,
    },
    "standard": {
        "retrieval_top_k": 8,
        "retrieval_multiplier": 3,
        "result_limit": 8,
        "max_facts": 8,
        "max_properties": 6,
        "max_relations": 6,
        "max_entities": 6,
        "max_chunks": 4,
        "max_graph_insights": 4,
        "max_source_notes": 4,
        "include_fact_properties": True,
        "fact_binding_top_k": 4,
        "property_binding_top_k": 3,
        "include_answer_context": True,
    },
    "deep": {
        "retrieval_top_k": 16,
        "retrieval_multiplier": 5,
        "result_limit": 16,
        "max_facts": 16,
        "max_properties": 12,
        "max_relations": 10,
        "max_entities": 12,
        "max_chunks": 8,
        "max_graph_insights": 8,
        "max_source_notes": 8,
        "include_fact_properties": True,
        "fact_binding_top_k": 6,
        "property_binding_top_k": 4,
        "include_answer_context": True,
    },
}
