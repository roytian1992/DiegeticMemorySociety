from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.context_kernel.kernel import CreativeMemoryKernel
from dms.context_kernel.schema import CreativeScope
from dms.progress import print_progress
from dms.storage.chroma_index import build_embedding_function


@dataclass(frozen=True)
class ContextChromaIndexConfig:
    context_db_path: Path
    persist_dir: Path
    project_id: str
    collection_name: str = "dms_context_documents"
    reset: bool = False
    upsert_batch_size: int = 1000
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60


def build_context_chroma_index(config: ContextChromaIndexConfig) -> dict[str, Any]:
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
    embedding_function = _embedding_function(config)
    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    kernel = CreativeMemoryKernel.from_db(config.context_db_path)
    records = kernel.list_retrieval_documents(
        scope=CreativeScope(project_id=config.project_id),
        statuses=["active", "canonical", "tentative"],
        limit=1_000_000,
    )
    batches = _batches(records, max(1, int(config.upsert_batch_size)))
    print_progress(
        "context_chroma_index:start",
        0,
        len(batches),
        detail=f"documents={len(records)} collection={config.collection_name} persist_dir={persist_dir}",
    )
    batch_count = 0
    for batch in batches:
        batch_count += 1
        collection.upsert(
            ids=[str(record["doc_id"]) for record in batch],
            documents=[str(record.get("text") or record.get("statement") or "") for record in batch],
            metadatas=[_metadata(record) for record in batch],
        )
        print_progress("context_chroma_index:batch", batch_count, len(batches), detail=f"documents={len(batch)}")
    return {
        "context_db_path": str(config.context_db_path),
        "persist_dir": str(persist_dir),
        "collection_name": config.collection_name,
        "project_id": config.project_id,
        "document_count": len(records),
        "upsert_batch_size": max(1, int(config.upsert_batch_size)),
        "upsert_batch_count": batch_count,
        "embedding": embedding_function.config_summary(),
    }


def search_context_chroma_index(
    context_db_path: str | Path,
    *,
    persist_dir: str | Path,
    query: str,
    scope: CreativeScope,
    collection_name: str = "dms_context_documents",
    source_types: list[str] | None = None,
    item_types: list[str] | None = None,
    statuses: list[str] | None = None,
    entity_ids: list[str] | None = None,
    visibility: list[str] | None = None,
    top_k: int = 10,
    embedding_dim: int = 384,
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_max_tokens: int = 8192,
    embedding_timeout: int = 60,
) -> dict[str, Any]:
    chromadb = _import_chromadb()
    kernel = CreativeMemoryKernel.from_db(context_db_path)
    sql_docs = kernel.list_retrieval_documents(
        scope=scope,
        source_types=source_types,
        item_types=item_types,
        statuses=statuses,
        entity_ids=entity_ids,
        visibility=visibility,
        limit=1_000_000,
    )
    if not sql_docs:
        return {"query": query, "count": 0, "results": []}
    allowed_doc_ids = {str(record["doc_id"]) for record in sql_docs}
    docs_by_id = {str(record["doc_id"]): record for record in sql_docs}
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_function = build_embedding_function(
        provider=embedding_provider,
        embedding_dim=embedding_dim,
        model_name=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        max_tokens=embedding_max_tokens,
        timeout=embedding_timeout,
    )
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    collection_count = int(collection.count())
    if len(allowed_doc_ids) <= 200:
        candidate_count = max(collection_count, 1)
    else:
        candidate_count = min(max(top_k * 20, len(allowed_doc_ids) + top_k), max(collection_count, 1))
    result = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    for doc_id, distance, document, metadata in zip(ids, distances, documents, metadatas):
        doc_id = str(doc_id)
        if doc_id not in allowed_doc_ids:
            continue
        row = dict(docs_by_id[doc_id])
        row["score"] = 1.0 - float(distance) if distance is not None else None
        row["distance"] = distance
        row["text"] = document
        row["chroma_metadata"] = metadata
        hits.append(row)
        if len(hits) >= max(int(top_k or 0), 0):
            break
    return {
        "query": query,
        "collection_name": collection_name,
        "project_id": scope.project_id,
        "embedding": embedding_function.config_summary(),
        "count": len(hits),
        "results": hits,
    }


def _embedding_function(config: ContextChromaIndexConfig):
    return build_embedding_function(
        provider=config.embedding_provider,
        embedding_dim=config.embedding_dim,
        model_name=config.embedding_model,
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        max_tokens=config.embedding_max_tokens,
        timeout=config.embedding_timeout,
    )


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "item_id": record.get("item_id"),
        "project_id": record.get("project_id"),
        "source_type": record.get("source_type"),
        "source_id": record.get("source_id"),
        "unit_id": record.get("unit_id") or "",
        "unit_order": int(record.get("unit_order") or 0),
        "item_type": record.get("item_type"),
        "subject": record.get("subject") or "",
        "status": record.get("status"),
        "authority": record.get("authority"),
        "visibility": record.get("visibility"),
        "temporal_scope": record.get("temporal_scope"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for context Chroma indexing.") from exc
    return chromadb
