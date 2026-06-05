from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dms.llm import LLMClient
from dms.parsing import extract_json_value
from dms.progress import print_progress
from dms.prompts import YAMLPromptLoader
from dms.source_evidence import locate_evidence
from dms.storage.chroma_index import build_embedding_function


REFERENCE_SCHEMA_VERSION = 1

AUTHOR_CONTEXT_TYPES = {
    "world_bible",
    "character_profile",
    "relationship_fact",
    "location_doc",
    "organization_fact",
    "author_note",
}
CHARACTER_KNOWLEDGE_TYPES = {
    "world_bible",
    "character_profile",
    "relationship_fact",
    "location_doc",
    "organization_fact",
    "timeline_doc",
}
STYLE_TYPES = {"style_guide"}
TIMELINE_TYPES = {"timeline_doc"}

AUTHOR_VISIBLE_SCOPES = {"author_only", "world_public", "character_private", "revealed_by_story"}
CHARACTER_VISIBLE_SCOPES = {"world_public", "character_private", "revealed_by_story"}
SUPPORTED_REFERENCE_EXTENSIONS = {".md", ".markdown", ".txt", ".json", ".jsonl"}


@dataclass(frozen=True)
class ReferenceLibraryIngestConfig:
    input_path: Path
    output_dir: Path
    max_chunk_chars: int = 2400
    overwrite: bool = False


@dataclass(frozen=True)
class ReferenceItemExtractionConfig:
    library_dir: Path
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    task_settings_path: Path = Path("task_specs/task_settings/reference_items_task.json")
    start: int = 1
    limit: int | None = None
    dry_run: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ReferenceItemImportConfig:
    items_path: Path
    db_path: Path
    reset: bool = False


@dataclass(frozen=True)
class ChromaReferenceIndexConfig:
    db_path: Path
    persist_dir: Path
    collection_name: str = "dms_reference_documents"
    reset: bool = False
    upsert_batch_size: int = 1000
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60


@dataclass(frozen=True)
class ReferenceContextQuery:
    db_path: Path
    query: str
    matched_entities: tuple[dict[str, Any], ...] = ()
    before_scene_id: str | None = None
    before_scene_order: int | None = None
    chroma_dir: Path | None = None
    collection_name: str = "dms_reference_documents"
    top_k: int = 6
    author_top_k: int = 6
    character_top_k: int = 6
    style_top_k: int = 4
    timeline_top_k: int = 4
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60


def ingest_reference_library(config: ReferenceLibraryIngestConfig) -> dict[str, Any]:
    input_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _reference_input_files(input_path)
    raw_documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for file_index, path in enumerate(files, start=1):
        file_records = _load_reference_file(path, root=input_path if input_path.is_dir() else input_path.parent, file_index=file_index)
        for raw_doc in file_records:
            raw_documents.append(raw_doc)
            chunks.extend(_chunk_reference_document(raw_doc, max_chunk_chars=max(int(config.max_chunk_chars or 1), 1)))

    raw_path = output_dir / "raw_documents.jsonl"
    chunks_path = output_dir / "reference_chunks.jsonl"
    summary_path = output_dir / "summary.json"
    _write_jsonl(raw_path, raw_documents)
    _write_jsonl(chunks_path, chunks)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "file_count": len(files),
        "raw_document_count": len(raw_documents),
        "reference_chunk_count": len(chunks),
        "max_chunk_chars": config.max_chunk_chars,
        "artifacts": {
            "raw_documents": str(raw_path),
            "reference_chunks": str(chunks_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def extract_reference_items(
    config: ReferenceItemExtractionConfig,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    if not config.dry_run and llm_client is None:
        raise ValueError("llm_client is required when dry_run is false")
    if config.start < 1:
        raise ValueError("start must be >= 1")

    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    for directory in (
        output_dir,
        output_dir / "inputs",
        output_dir / "prompts",
        output_dir / "raw_outputs",
        output_dir / "parsed",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    chunks = _read_jsonl(Path(config.library_dir) / "reference_chunks.jsonl")
    selected = _select_records(chunks, start=config.start, limit=config.limit)
    task_settings = _read_json(config.task_settings_path)
    prompt_id = str(task_settings["prompt_id"])
    extraction_policy = _format_policy(task_settings.get("extraction_policy", []))
    loader = YAMLPromptLoader(config.prompt_dir)
    prompt_spec = loader.load(prompt_id)

    accepted_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    completed_count = 0
    parsed_count = 0
    failed_count = 0

    for ordinal, chunk in enumerate(selected, start=1):
        print_progress(
            "reference_items:chunk",
            ordinal - 1,
            len(selected),
            detail=f"chunk={chunk.get('chunk_id')} status=start",
        )
        chunk_id = str(chunk.get("chunk_id") or f"chunk_{ordinal:06d}")
        prompt_context = _reference_chunk_prompt_context(chunk)
        prompt_text = loader.render(
            prompt_spec,
            task_values={"reference_chunk_json": prompt_context},
            static_values={"extraction_policy": extraction_policy},
        )
        input_path = output_dir / "inputs" / f"{_safe_path_id(chunk_id)}.json"
        prompt_path = output_dir / "prompts" / f"{_safe_path_id(chunk_id)}.txt"
        raw_path = output_dir / "raw_outputs" / f"{_safe_path_id(chunk_id)}.json"
        parsed_path = output_dir / "parsed" / f"{_safe_path_id(chunk_id)}.json"
        input_path.write_text(json.dumps(prompt_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prompt_path.write_text(prompt_text.rstrip() + "\n", encoding="utf-8")

        if config.dry_run:
            raw_payload = {
                "chunk_id": chunk_id,
                "status": "not_run",
                "reason": "dry_run",
                "raw_text": "",
            }
            parsed_payload = {
                "chunk_id": chunk_id,
                "status": "not_parsed",
                "reason": "dry_run",
                "data": None,
                "accepted_item_count": 0,
                "rejected_item_count": 0,
            }
            status = "dry_run_rendered"
            error = None
        else:
            try:
                assert llm_client is not None
                result = llm_client.complete(prompt_text)
                completed_count += 1
                parse_result = extract_json_value(result.text)
                if not parse_result.ok:
                    failed_count += 1
                    status = "parse_failed"
                    error = parse_result.error
                    parsed_items = []
                else:
                    parsed_items = _extract_items_from_model_payload(parse_result.data)
                    parsed_count += 1
                    status = "completed"
                    error = None
                accepted, rejected = _validate_reference_items_for_chunk(parsed_items, chunk)
                accepted_items.extend(accepted)
                rejected_items.extend(rejected)
                raw_payload = {
                    "chunk_id": chunk_id,
                    "status": "completed",
                    "provider": result.provider,
                    "model": result.model,
                    "raw_text": result.text,
                    "usage": result.usage,
                    "raw_response": result.raw_response,
                }
                parsed_payload = {
                    "chunk_id": chunk_id,
                    "status": "parsed" if parse_result.ok else "parse_failed",
                    "data": parse_result.data if parse_result.ok else None,
                    "parse_error": parse_result.error,
                    "accepted_item_count": len(accepted),
                    "rejected_item_count": len(rejected),
                }
            except Exception as exc:  # noqa: BLE001 - preserve per-chunk extraction failures.
                failed_count += 1
                status = "llm_failed"
                error = str(exc)
                raw_payload = {
                    "chunk_id": chunk_id,
                    "status": "llm_failed",
                    "error": str(exc),
                    "raw_text": "",
                }
                parsed_payload = {
                    "chunk_id": chunk_id,
                    "status": "not_parsed",
                    "reason": "llm_failed",
                    "data": None,
                    "accepted_item_count": 0,
                    "rejected_item_count": 0,
                }

        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parsed_path.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        trace_records.append(
            {
                "ordinal": ordinal,
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "source_path": chunk.get("source_path"),
                "input_path": str(input_path),
                "prompt_path": str(prompt_path),
                "raw_output_path": str(raw_path),
                "parsed_path": str(parsed_path),
                "status": status,
                "error": error,
                "prompt_char_count": len(prompt_text),
                "chunk_char_count": len(str(chunk.get("content") or "")),
            }
        )
        print_progress(
            "reference_items:chunk",
            ordinal,
            len(selected),
            detail=f"chunk={chunk_id} status={status}",
        )

    items_path = output_dir / "reference_items.jsonl"
    rejected_path = output_dir / "rejected_items.jsonl"
    trace_path = output_dir / "trace.jsonl"
    summary_path = output_dir / "summary.json"
    accepted_items = _dedupe_reference_items(accepted_items)
    _write_jsonl(items_path, accepted_items)
    _write_jsonl(rejected_path, rejected_items)
    _write_jsonl(trace_path, trace_records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run_complete" if config.dry_run else "complete",
        "library_dir": str(config.library_dir),
        "output_dir": str(output_dir),
        "prompt_id": prompt_id,
        "prompt_path": str(prompt_spec.path),
        "task_settings_path": str(config.task_settings_path),
        "selection": {
            "start": config.start,
            "limit": config.limit,
            "selected_count": len(selected),
        },
        "llm": {
            "provider": getattr(llm_client, "provider", None) if llm_client else None,
            "model": getattr(llm_client, "model", None) if llm_client else None,
        },
        "llm_completed_count": completed_count,
        "parsed_output_count": parsed_count,
        "failed_count": failed_count,
        "accepted_item_count": len(accepted_items),
        "rejected_item_count": len(rejected_items),
        "artifacts": {
            "reference_items": str(items_path),
            "rejected_items": str(rejected_path),
            "trace": str(trace_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def import_reference_items(config: ReferenceItemImportConfig) -> dict[str, Any]:
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if config.reset and db_path.exists():
        db_path.unlink()

    records = [_normalize_reference_item(row, index=index) for index, row in enumerate(_read_jsonl(config.items_path), start=1)]
    with _connect(db_path) as conn:
        _apply_reference_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO reference_metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(REFERENCE_SCHEMA_VERSION)),
        )
        for record in records:
            _insert_reference_item(conn, record)
            _insert_reference_document(conn, record)
        conn.commit()

    return {
        "db_path": str(db_path),
        "items_path": str(config.items_path),
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_items": len(records),
        "reference_documents": len(records),
    }


def build_chroma_reference_index(config: ChromaReferenceIndexConfig) -> dict[str, Any]:
    chromadb = _import_chromadb()
    persist_dir = Path(config.persist_dir)
    if config.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    if config.reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            pass
    embedding_function = build_embedding_function(
        provider=config.embedding_provider,
        embedding_dim=config.embedding_dim,
        model_name=config.embedding_model,
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        max_tokens=config.embedding_max_tokens,
        timeout=config.embedding_timeout,
    )
    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    records = list_reference_documents(config.db_path)
    batches = _batches(records, max(int(config.upsert_batch_size or 1), 1))
    print_progress(
        "reference_chroma_index:start",
        0,
        len(batches),
        detail=f"documents={len(records)} collection={config.collection_name} persist_dir={persist_dir}",
    )
    batch_count = 0
    for batch in batches:
        batch_count += 1
        collection.upsert(
            ids=[str(record["doc_id"]) for record in batch],
            documents=[str(record["text"]) for record in batch],
            metadatas=[_reference_chroma_metadata(record) for record in batch],
        )
        print_progress(
            "reference_chroma_index:batch",
            batch_count,
            len(batches),
            detail=f"documents={len(batch)}",
        )
    return {
        "db_path": str(config.db_path),
        "persist_dir": str(persist_dir),
        "collection_name": config.collection_name,
        "document_count": len(records),
        "upsert_batch_size": max(int(config.upsert_batch_size or 1), 1),
        "upsert_batch_count": batch_count,
        "embedding": embedding_function.config_summary(),
    }


def build_reference_context(query: ReferenceContextQuery) -> tuple[dict[str, Any], dict[str, Any]]:
    top_k = max(int(query.top_k or 0), 0)
    if top_k <= 0:
        return _empty_reference_context(), {
            "enabled": True,
            "strategy": "disabled_by_top_k",
            "returned_counts": _reference_context_counts(_empty_reference_context()),
        }

    candidate_rows = _search_reference_documents(query)
    matched_names = _matched_entity_names(query.matched_entities)
    grouped = {
        "author_reference_context": _select_reference_items(
            candidate_rows,
            allowed_types=AUTHOR_CONTEXT_TYPES,
            allowed_scopes=AUTHOR_VISIBLE_SCOPES,
            top_k=max(int(query.author_top_k or top_k), 0),
        ),
        "character_reference_knowledge": _select_character_reference_knowledge(
            candidate_rows,
            matched_names=matched_names,
            top_k=max(int(query.character_top_k or top_k), 0),
        ),
        "style_reference_context": _select_reference_items(
            candidate_rows,
            allowed_types=STYLE_TYPES,
            allowed_scopes={"style_only"},
            top_k=max(int(query.style_top_k or top_k), 0),
        ),
        "timeline_reference_claims": _select_reference_items(
            candidate_rows,
            allowed_types=TIMELINE_TYPES,
            allowed_scopes=AUTHOR_VISIBLE_SCOPES | {"world_public"},
            top_k=max(int(query.timeline_top_k or top_k), 0),
        ),
    }
    trace = {
        "enabled": True,
        "strategy": "chroma_sql_filtered" if query.chroma_dir else "sql_ranked",
        "db_path": str(query.db_path),
        "chroma_dir": str(query.chroma_dir) if query.chroma_dir else None,
        "collection_name": query.collection_name if query.chroma_dir else None,
        "query": query.query,
        "before_scene_id": query.before_scene_id,
        "before_scene_order": query.before_scene_order,
        "matched_entities": [entity.get("canonical_name") or entity.get("entity_id") for entity in query.matched_entities],
        "candidate_count": len(candidate_rows),
        "returned_counts": _reference_context_counts(grouped),
        "visibility_policy": {
            "author_reference_context": "author_only, world_public, character_private, and revealed_by_story facts are author-facing context",
            "character_reference_knowledge": "world_public is visible to all; character_private requires known_to match; revealed_by_story respects available_from boundary when set",
            "style_reference_context": "style_only is wording/style guidance, not factual memory",
            "timeline_reference_claims": "timeline_doc claims remain author/system context and may later feed temporal graph construction",
        },
    }
    return grouped, trace


def list_reference_documents(
    db_path: str | Path,
    *,
    item_types: set[str] | None = None,
    knowledge_scopes: set[str] | None = None,
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
) -> list[dict[str, Any]]:
    max_order = before_scene_order
    if max_order is None and before_scene_id:
        max_order = _max_scene_order_before(before_scene_id)
    sql = "SELECT * FROM reference_documents WHERE 1 = 1"
    params: list[Any] = []
    if item_types:
        placeholders = ", ".join("?" for _ in item_types)
        sql += f" AND item_type IN ({placeholders})"
        params.extend(sorted(item_types))
    if knowledge_scopes:
        placeholders = ", ".join("?" for _ in knowledge_scopes)
        sql += f" AND knowledge_scope IN ({placeholders})"
        params.extend(sorted(knowledge_scopes))
    if max_order is not None:
        sql += " AND (available_from_order IS NULL OR available_from_order <= ?)"
        params.append(max_order)
    sql += " ORDER BY authority DESC, confidence DESC, item_id"
    with _connect(db_path) as conn:
        return [_reference_document_payload(row) for row in conn.execute(sql, params)]


def get_reference_item(db_path: str | Path, item_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM reference_items WHERE item_id = ?", (item_id,)).fetchone()
        return _reference_item_payload(row) if row else None


def _search_reference_documents(query: ReferenceContextQuery) -> list[dict[str, Any]]:
    sql_docs = list_reference_documents(
        query.db_path,
        before_scene_order=query.before_scene_order,
        before_scene_id=query.before_scene_id,
    )
    if not sql_docs:
        return []
    limit = max(
        int(query.author_top_k or 0),
        int(query.character_top_k or 0),
        int(query.style_top_k or 0),
        int(query.timeline_top_k or 0),
        int(query.top_k or 0),
        1,
    )
    if not query.chroma_dir:
        return _rank_reference_sql_docs(sql_docs, query.query, query.matched_entities)[: max(limit * 4, limit)]

    chromadb = _import_chromadb()
    docs_by_id = {str(row["doc_id"]): row for row in sql_docs}
    allowed_doc_ids = set(docs_by_id)
    client = chromadb.PersistentClient(path=str(query.chroma_dir))
    embedding_function = build_embedding_function(
        provider=query.embedding_provider,
        embedding_dim=query.embedding_dim,
        model_name=query.embedding_model,
        base_url=query.embedding_base_url,
        api_key=query.embedding_api_key,
        max_tokens=query.embedding_max_tokens,
        timeout=query.embedding_timeout,
    )
    collection = client.get_collection(
        name=query.collection_name,
        embedding_function=embedding_function,
    )
    collection_count = int(collection.count())
    candidate_count = max(min(max(limit * 20, len(sql_docs), 1), max(collection_count, 1)), 1)
    result = collection.query(
        query_texts=[query.query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc_id, distance in zip(ids, distances):
        doc_id = str(doc_id)
        if doc_id not in allowed_doc_ids:
            continue
        row = dict(docs_by_id[doc_id])
        row["score"] = 1.0 - float(distance) if distance is not None else None
        hits.append(row)
        if len(hits) >= max(limit * 4, limit):
            break
    return hits


def _rank_reference_sql_docs(
    docs: list[dict[str, Any]],
    query: str,
    matched_entities: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    entity_tokens = []
    for entity in matched_entities:
        entity_tokens.extend(_tokens(entity.get("canonical_name") or ""))
        for alias in entity.get("aliases") or []:
            entity_tokens.extend(_tokens(alias))
    ranked = []
    for row in docs:
        text = "\n".join(
            str(row.get(key) or "")
            for key in ("subject", "statement", "evidence", "text")
        )
        tokens = set(_tokens(text))
        query_overlap = len(tokens.intersection(query_tokens))
        entity_overlap = len(tokens.intersection(entity_tokens))
        authority = float(row.get("authority") or 0.0)
        confidence = float(row.get("confidence") or 0.0)
        score = query_overlap + entity_overlap * 1.5 + authority * 0.25 + confidence * 0.25
        payload = dict(row)
        payload["score"] = round(score, 4)
        ranked.append(payload)
    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("authority") or 0.0),
            float(item.get("confidence") or 0.0),
            str(item.get("item_id") or ""),
        ),
        reverse=True,
    )


def _select_reference_items(
    rows: list[dict[str, Any]],
    *,
    allowed_types: set[str],
    allowed_scopes: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    selected = []
    seen: set[str] = set()
    for row in rows:
        item_type = str(row.get("item_type") or "")
        scope = str(row.get("knowledge_scope") or "")
        item_id = str(row.get("item_id") or "")
        if item_type not in allowed_types or scope not in allowed_scopes or item_id in seen:
            continue
        selected.append(_packet_reference_item(row))
        seen.add(item_id)
        if len(selected) >= top_k:
            break
    return selected


def _select_character_reference_knowledge(
    rows: list[dict[str, Any]],
    *,
    matched_names: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    selected = []
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id") or "")
        if item_id in seen:
            continue
        item_type = str(row.get("item_type") or "")
        scope = str(row.get("knowledge_scope") or "")
        if item_type not in CHARACTER_KNOWLEDGE_TYPES or scope not in CHARACTER_VISIBLE_SCOPES:
            continue
        if not _is_character_visible(row, matched_names):
            continue
        selected.append(_packet_reference_item(row))
        seen.add(item_id)
        if len(selected) >= top_k:
            break
    return selected


def _is_character_visible(row: dict[str, Any], matched_names: set[str]) -> bool:
    scope = str(row.get("knowledge_scope") or "")
    if scope == "world_public":
        return True
    known_to = {str(item).strip().lower() for item in row.get("known_to") or [] if str(item or "").strip()}
    if "all" in known_to:
        return True
    normalized_names = {name.lower() for name in matched_names}
    if scope == "character_private":
        return bool(known_to.intersection(normalized_names))
    if scope == "revealed_by_story":
        available_from = str(row.get("available_from") or "unknown")
        if available_from != "story_start" and row.get("available_from_order") is None:
            return False
        return bool(not known_to or known_to.intersection(normalized_names))
    return False


def _packet_reference_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": row.get("item_id"),
        "doc_id": row.get("source_doc_id") or row.get("doc_id"),
        "chunk_id": row.get("chunk_id"),
        "item_type": row.get("item_type"),
        "subject": row.get("subject"),
        "statement": row.get("statement"),
        "evidence": row.get("evidence"),
        "authority": row.get("authority"),
        "knowledge_scope": row.get("knowledge_scope"),
        "known_to": row.get("known_to") or [],
        "available_from": row.get("available_from"),
        "timeline_hint": row.get("timeline_hint"),
        "confidence": row.get("confidence"),
        "score": row.get("score"),
    }


def _empty_reference_context() -> dict[str, Any]:
    return {
        "author_reference_context": [],
        "character_reference_knowledge": [],
        "style_reference_context": [],
        "timeline_reference_claims": [],
    }


def _reference_context_counts(context: dict[str, Any]) -> dict[str, int]:
    return {
        "author_reference_context": len(context.get("author_reference_context") or []),
        "character_reference_knowledge": len(context.get("character_reference_knowledge") or []),
        "style_reference_context": len(context.get("style_reference_context") or []),
        "timeline_reference_claims": len(context.get("timeline_reference_claims") or []),
    }


def _apply_reference_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS reference_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reference_items (
            item_id TEXT PRIMARY KEY,
            source_doc_id TEXT,
            chunk_id TEXT,
            item_type TEXT NOT NULL,
            subject TEXT,
            statement TEXT NOT NULL,
            evidence TEXT,
            authority REAL NOT NULL DEFAULT 0,
            knowledge_scope TEXT NOT NULL,
            known_to_json TEXT NOT NULL,
            available_from TEXT,
            available_from_order INTEGER,
            timeline_hint TEXT,
            confidence REAL NOT NULL DEFAULT 0,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reference_documents (
            doc_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            source_doc_id TEXT,
            chunk_id TEXT,
            item_type TEXT NOT NULL,
            knowledge_scope TEXT NOT NULL,
            known_to_json TEXT NOT NULL,
            available_from TEXT,
            available_from_order INTEGER,
            authority REAL NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reference_documents_type_scope
            ON reference_documents(item_type, knowledge_scope, available_from_order);
        CREATE INDEX IF NOT EXISTS idx_reference_items_type_scope
            ON reference_items(item_type, knowledge_scope, available_from_order);
        """
    )


def _insert_reference_item(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO reference_items
        (item_id, source_doc_id, chunk_id, item_type, subject, statement, evidence,
         authority, knowledge_scope, known_to_json, available_from, available_from_order,
         timeline_hint, confidence, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["item_id"],
            row.get("source_doc_id"),
            row.get("chunk_id"),
            row["item_type"],
            row.get("subject"),
            row["statement"],
            row.get("evidence"),
            float(row.get("authority") or 0.0),
            row["knowledge_scope"],
            json.dumps(row.get("known_to") or [], ensure_ascii=False),
            row.get("available_from"),
            row.get("available_from_order"),
            row.get("timeline_hint"),
            float(row.get("confidence") or 0.0),
            json.dumps(row.get("raw") or row, ensure_ascii=False),
        ),
    )


def _insert_reference_document(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    text = _reference_retrieval_text(row)
    metadata = {
        "item_id": row.get("item_id"),
        "source_doc_id": row.get("source_doc_id"),
        "chunk_id": row.get("chunk_id"),
        "item_type": row.get("item_type"),
        "subject": row.get("subject"),
        "statement": row.get("statement"),
        "evidence": row.get("evidence"),
        "authority": row.get("authority"),
        "knowledge_scope": row.get("knowledge_scope"),
        "known_to": row.get("known_to") or [],
        "available_from": row.get("available_from"),
        "available_from_order": row.get("available_from_order"),
        "timeline_hint": row.get("timeline_hint"),
        "confidence": row.get("confidence"),
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO reference_documents
        (doc_id, item_id, source_doc_id, chunk_id, item_type, knowledge_scope,
         known_to_json, available_from, available_from_order, authority, confidence,
         text, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"reference:{row['item_id']}",
            row["item_id"],
            row.get("source_doc_id"),
            row.get("chunk_id"),
            row["item_type"],
            row["knowledge_scope"],
            json.dumps(row.get("known_to") or [], ensure_ascii=False),
            row.get("available_from"),
            row.get("available_from_order"),
            float(row.get("authority") or 0.0),
            float(row.get("confidence") or 0.0),
            text,
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def _normalize_reference_item(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    source_doc_id = str(row.get("doc_id") or row.get("source_doc_id") or row.get("document_id") or "").strip()
    chunk_id = str(row.get("chunk_id") or row.get("source_chunk_id") or "").strip()
    statement = str(row.get("statement") or row.get("text") or row.get("summary") or "").strip()
    evidence = str(row.get("evidence") or row.get("evidence_text") or "").strip()
    item_type = _normalize_item_type(row.get("item_type") or row.get("type"))
    scope = _normalize_knowledge_scope(row.get("knowledge_scope") or row.get("scope"), item_type=item_type)
    known_to = _normalize_known_to(row.get("known_to"))
    item_id = str(row.get("item_id") or row.get("record_id") or "").strip()
    if not item_id:
        digest = hashlib.sha1(
            "\n".join([source_doc_id, chunk_id, item_type, statement, evidence, str(index)]).encode("utf-8")
        ).hexdigest()[:12]
        item_id = f"ref_item_{index:06d}_{digest}"
    if not statement:
        raise ValueError(f"Reference item missing statement at index {index}")
    available_from = str(row.get("available_from") or "").strip() or "unknown"
    return {
        "item_id": item_id,
        "source_doc_id": source_doc_id,
        "chunk_id": chunk_id,
        "item_type": item_type,
        "subject": str(row.get("subject") or "").strip(),
        "statement": statement,
        "evidence": evidence,
        "authority": _float_or(row.get("authority"), 0.5),
        "knowledge_scope": scope,
        "known_to": known_to,
        "available_from": available_from,
        "available_from_order": _available_from_order(available_from),
        "timeline_hint": str(row.get("timeline_hint") or "").strip(),
        "confidence": _float_or(row.get("confidence"), 0.5),
        "raw": dict(row),
    }


def _reference_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_REFERENCE_EXTENSIONS:
            raise ValueError(f"Unsupported reference file extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Reference input path not found: {input_path}")
    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_REFERENCE_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No supported reference files found under: {input_path}")
    return files


def _load_reference_file(path: Path, *, root: Path, file_index: int) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [_raw_reference_document(path, root=root, file_index=file_index, record_index=1, title=path.stem, content=text)]
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(
                    _raw_reference_document(
                        path,
                        root=root,
                        file_index=file_index,
                        record_index=index,
                        title=str(payload.get("heading") or payload.get("title") or payload.get("record_id") or path.stem)
                        if isinstance(payload, dict)
                        else f"{path.stem}_{index}",
                        content=_json_record_to_text(payload),
                        raw_payload=payload,
                    )
                )
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        docs = _flatten_json_reference_payload(payload, path=path, root=root, file_index=file_index)
        return docs or [
            _raw_reference_document(
                path,
                root=root,
                file_index=file_index,
                record_index=1,
                title=path.stem,
                content=_json_record_to_text(payload),
                raw_payload=payload,
            )
        ]
    raise ValueError(f"Unsupported reference file extension: {suffix}")


def _raw_reference_document(
    path: Path,
    *,
    root: Path,
    file_index: int,
    record_index: int,
    title: str,
    content: str,
    raw_payload: Any | None = None,
) -> dict[str, Any]:
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    doc_key = f"{rel.as_posix()}:{record_index}"
    digest = hashlib.sha1(doc_key.encode("utf-8")).hexdigest()[:10]
    content_text = str(content or "")
    return {
        "doc_id": f"ref_doc_{file_index:04d}_{record_index:04d}_{digest}",
        "source_path": str(path),
        "relative_path": rel.as_posix(),
        "file_name": path.name,
        "format": path.suffix.lower().lstrip("."),
        "record_index": record_index,
        "title": str(title or path.stem).strip(),
        "content": content_text,
        "content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
        "raw": raw_payload if raw_payload is not None else {},
    }


def _flatten_json_reference_payload(payload: Any, *, path: Path, root: Path, file_index: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        record_index = 0
        for doc in payload["documents"]:
            if not isinstance(doc, dict):
                continue
            doc_title = str(doc.get("doc_id") or doc.get("title") or path.stem)
            chunks = doc.get("chunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    record_index += 1
                    title = str(chunk.get("heading") or chunk.get("chunk_id") or doc_title)
                    records.append(
                        _raw_reference_document(
                            path,
                            root=root,
                            file_index=file_index,
                            record_index=record_index,
                            title=title,
                            content=_json_record_to_text({**doc, **chunk, "parent_doc": doc_title}),
                            raw_payload={"document": doc, "chunk": chunk},
                        )
                    )
            else:
                record_index += 1
                records.append(
                    _raw_reference_document(
                        path,
                        root=root,
                        file_index=file_index,
                        record_index=record_index,
                        title=doc_title,
                        content=_json_record_to_text(doc),
                        raw_payload=doc,
                    )
                )
        return records
    if isinstance(payload, list):
        return [
            _raw_reference_document(
                path,
                root=root,
                file_index=file_index,
                record_index=index,
                title=str(item.get("title") or item.get("heading") or item.get("record_id") or f"{path.stem}_{index}")
                if isinstance(item, dict)
                else f"{path.stem}_{index}",
                content=_json_record_to_text(item),
                raw_payload=item,
            )
            for index, item in enumerate(payload, start=1)
        ]
    return []


def _json_record_to_text(payload: Any) -> str:
    if isinstance(payload, dict):
        parts = []
        for key, value in payload.items():
            if key in {"raw", "documents", "chunks"}:
                continue
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            if rendered and rendered not in {"None", ""}:
                parts.append(f"{key}: {rendered}")
        return "\n".join(parts)
    if isinstance(payload, list):
        return "\n".join(_json_record_to_text(item) for item in payload)
    return str(payload or "")


def _chunk_reference_document(raw_doc: dict[str, Any], *, max_chunk_chars: int) -> list[dict[str, Any]]:
    content = str(raw_doc.get("content") or "")
    sections = _markdown_like_sections(content) if raw_doc.get("format") in {"md", "markdown", "txt"} else []
    if not sections:
        sections = [{"heading": raw_doc.get("title") or "", "content": content, "section_index": 1}]

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for section in sections:
        for part_index, text in enumerate(_split_text_by_chars(section["content"], max_chars=max_chunk_chars), start=1):
            if not str(text or "").strip():
                continue
            chunk_index += 1
            chunk_id = f"{raw_doc['doc_id']}_chunk_{chunk_index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": raw_doc["doc_id"],
                    "source_path": raw_doc.get("source_path"),
                    "relative_path": raw_doc.get("relative_path"),
                    "format": raw_doc.get("format"),
                    "title": raw_doc.get("title"),
                    "heading": section.get("heading") or raw_doc.get("title") or "",
                    "section_index": section.get("section_index"),
                    "chunk_index": chunk_index,
                    "section_part_index": part_index,
                    "content": text,
                    "content_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                    "parent_content_sha256": raw_doc.get("content_sha256"),
                    "char_count": len(str(text)),
                    "raw_document": {
                        "doc_id": raw_doc.get("doc_id"),
                        "source_path": raw_doc.get("source_path"),
                        "relative_path": raw_doc.get("relative_path"),
                        "title": raw_doc.get("title"),
                    },
                }
            )
    return chunks


def _markdown_like_sections(content: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_lines: list[str] = []
    section_index = 0
    for line in str(content or "").splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            if current_lines:
                section_index += 1
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(current_lines).strip(),
                        "section_index": section_index,
                    }
                )
            current_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        section_index += 1
        sections.append({"heading": current_heading, "content": "\n".join(current_lines).strip(), "section_index": section_index})
    return [section for section in sections if str(section.get("content") or "").strip()]


def _split_text_by_chars(text: str, *, max_chars: int) -> list[str]:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return [value] if value else []
    paragraphs = re.split(r"\n\s*\n", value)
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current.strip())
            current = paragraph
    if current:
        parts.append(current.strip())
    return parts


def _reference_chunk_prompt_context(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "source_path": chunk.get("source_path"),
        "relative_path": chunk.get("relative_path"),
        "format": chunk.get("format"),
        "title": chunk.get("title"),
        "heading": chunk.get("heading"),
        "content": chunk.get("content"),
    }


def _extract_items_from_model_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("reference_items")
        if items is None:
            items = data.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, dict)]
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def _validate_reference_items_for_chunk(items: list[dict[str, Any]], chunk: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        raw_item = dict(item)
        raw_item.setdefault("doc_id", chunk.get("doc_id"))
        raw_item.setdefault("chunk_id", chunk.get("chunk_id"))
        raw_item.setdefault("source_doc_id", chunk.get("doc_id"))
        try:
            normalized = _normalize_reference_item(raw_item, index=index)
        except ValueError as exc:
            rejected.append(
                _rejected_reference_item(
                    {
                        **raw_item,
                        "item_id": raw_item.get("item_id") or raw_item.get("record_id") or f"{chunk.get('chunk_id')}_item_{index:03d}",
                    },
                    chunk,
                    reason=str(exc),
                )
            )
            continue
        if not normalized.get("evidence"):
            rejected.append(_rejected_reference_item(normalized, chunk, reason="missing_evidence"))
            continue
        location = locate_evidence(
            normalized.get("evidence"),
            {
                "title": chunk.get("title") or "",
                "subtitle": chunk.get("heading") or "",
                "content": chunk.get("content") or "",
            },
        )
        if location.get("evidence_verification_status") == "rejected":
            rejected.append(_rejected_reference_item(normalized, chunk, reason="evidence_not_aligned", location=location))
            continue
        normalized["doc_id"] = normalized["source_doc_id"]
        normalized["chunk_id"] = chunk.get("chunk_id")
        normalized["source_path"] = chunk.get("source_path")
        normalized["evidence_text"] = normalized.get("evidence")
        normalized["evidence_aligned_text"] = location.get("evidence_aligned_text")
        normalized["evidence_source_field"] = location.get("evidence_source_field")
        normalized["evidence_start"] = location.get("evidence_start")
        normalized["evidence_end"] = location.get("evidence_end")
        normalized["evidence_source_sha256"] = location.get("evidence_source_sha256")
        normalized["evidence_alignment_score"] = location.get("evidence_alignment_score")
        normalized["evidence_verification_status"] = location.get("evidence_verification_status")
        accepted.append(normalized)
    return _dedupe_reference_items(accepted), rejected


def _rejected_reference_item(
    item: dict[str, Any],
    chunk: dict[str, Any],
    *,
    reason: str,
    location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **item,
        "doc_id": item.get("source_doc_id") or chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "source_path": chunk.get("source_path"),
        "rejection_reason": reason,
        "evidence_location": location or {},
    }


def _dedupe_reference_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _dedupe_key(item)
        existing = selected.get(key)
        if existing is None or _item_quality(item) > _item_quality(existing):
            selected[key] = item
    return list(selected.values())


def _dedupe_key(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("item_type") or ""),
            _normalize_compact(item.get("subject")),
            _normalize_compact(item.get("statement")),
        ]
    )


def _item_quality(item: dict[str, Any]) -> tuple[float, float]:
    return (float(item.get("authority") or 0.0), float(item.get("confidence") or 0.0))


def _normalize_compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _normalize_item_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "character": "character_profile",
        "character_profile_doc": "character_profile",
        "relationship": "relationship_fact",
        "location": "location_doc",
        "organization": "organization_fact",
        "org": "organization_fact",
        "timeline": "timeline_doc",
        "style": "style_guide",
        "note": "author_note",
        "notes": "author_note",
    }
    text = aliases.get(text, text)
    allowed = AUTHOR_CONTEXT_TYPES | CHARACTER_KNOWLEDGE_TYPES | STYLE_TYPES | TIMELINE_TYPES
    return text if text in allowed else "author_note"


def _normalize_knowledge_scope(value: Any, *, item_type: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "public": "world_public",
        "world": "world_public",
        "private": "character_private",
        "character": "character_private",
        "author": "author_only",
        "style": "style_only",
        "revealed": "revealed_by_story",
    }
    text = aliases.get(text, text)
    if item_type in STYLE_TYPES:
        return "style_only"
    if text in {"author_only", "world_public", "character_private", "revealed_by_story", "style_only"}:
        return text
    return "author_only"


def _normalize_known_to(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return ["all"]
        parts = re.split(r"[,，;；\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return [str(value).strip()]


def _reference_retrieval_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("item_type") or ""),
        str(row.get("subject") or ""),
        str(row.get("statement") or ""),
        str(row.get("evidence") or ""),
        str(row.get("timeline_hint") or ""),
        "known_to: " + ", ".join(row.get("known_to") or []) if row.get("known_to") else "",
    ]
    return "\n".join(part for part in parts if part)


def _reference_document_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["known_to"] = _json_loads(payload.pop("known_to_json", "[]"), default=[])
    metadata = _json_loads(payload.get("metadata_json"), default={})
    if isinstance(metadata, dict):
        payload.update({key: value for key, value in metadata.items() if key not in payload or payload.get(key) in (None, "")})
    return payload


def _reference_item_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["known_to"] = _json_loads(payload.pop("known_to_json", "[]"), default=[])
    payload["raw"] = _json_loads(payload.pop("raw_json", "{}"), default={})
    return payload


def _matched_entity_names(entities: tuple[dict[str, Any], ...]) -> set[str]:
    names: set[str] = set()
    for entity in entities:
        for value in (entity.get("entity_id"), entity.get("canonical_name")):
            if str(value or "").strip():
                names.add(str(value).strip())
        for alias in entity.get("aliases") or []:
            if str(alias or "").strip():
                names.add(str(alias).strip())
    return names


def _available_from_order(value: str) -> int | None:
    text = str(value or "").strip()
    if not text or text in {"story_start", "unknown"}:
        return None
    return _scene_order_from_id(text) or None


def _scene_order_from_id(value: Any) -> int:
    text = str(value or "")
    for part in text.split("_"):
        if part.isdigit():
            return int(part)
    return 0


def _max_scene_order_before(value: Any) -> int:
    order = _scene_order_from_id(value)
    return max(order - 1, 0) if order else 0


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokens(text: Any) -> list[str]:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [
        lowered[index : index + 2]
        for index in range(max(len(lowered) - 1, 0))
        if any("\u4e00" <= char <= "\u9fff" for char in lowered[index : index + 2])
    ]
    return tokens + cjk_chars + cjk_bigrams


def _reference_chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "item_id": record.get("item_id"),
        "source_doc_id": record.get("source_doc_id") or "",
        "chunk_id": record.get("chunk_id") or "",
        "item_type": record.get("item_type"),
        "knowledge_scope": record.get("knowledge_scope"),
        "available_from_order": int(record.get("available_from_order") or 0),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def _select_records(records: list[dict[str, Any]], *, start: int, limit: int | None) -> list[dict[str, Any]]:
    offset = max(start - 1, 0)
    return records[offset:] if limit is None else records[offset : offset + max(limit, 0)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
            else:
                raise ValueError(f"JSONL rows must be objects: {path}")
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_policy(policy: Any) -> str:
    if isinstance(policy, list):
        return "\n".join(f"- {item}" for item in policy)
    return str(policy or "")


def _safe_path_id(value: Any) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "item"))
    return safe.strip("._") or "item"


def _json_loads(value: Any, *, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Use the screenplay conda environment or install the optional vector DB dependency."
        ) from exc
    return chromadb
