from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.progress import print_progress
from dms.storage.asset_store import get_retrieval_documents


@dataclass(frozen=True)
class ChromaMemoryIndexConfig:
    db_path: Path
    persist_dir: Path
    collection_name: str = "dms_retrieval_documents"
    reset: bool = False
    upsert_batch_size: int = 1000
    embedding_dim: int = 384
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_max_tokens: int = 8192
    embedding_timeout: int = 60


def build_chroma_memory_index(config: ChromaMemoryIndexConfig) -> dict[str, Any]:
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

    records = get_retrieval_documents(config.db_path)
    batch_size = max(1, int(config.upsert_batch_size))
    record_batches = _batches(records, batch_size)
    print_progress(
        "chroma_index:start",
        0,
        len(record_batches),
        detail=f"documents={len(records)} collection={config.collection_name} persist_dir={persist_dir}",
    )
    batch_count = 0
    for batch in record_batches:
        batch_count += 1
        collection.upsert(
            ids=[str(record["doc_id"]) for record in batch],
            documents=[str(record["text"]) for record in batch],
            metadatas=[_chroma_metadata(record) for record in batch],
        )
        print_progress(
            "chroma_index:batch",
            batch_count,
            len(record_batches),
            detail=f"documents={len(batch)}",
        )

    return {
        "db_path": str(config.db_path),
        "persist_dir": str(persist_dir),
        "collection_name": config.collection_name,
        "document_count": len(records),
        "upsert_batch_size": batch_size,
        "upsert_batch_count": batch_count,
        "embedding": embedding_function.config_summary(),
    }


def _batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def search_entity_memories(
    db_path: str | Path,
    *,
    persist_dir: str | Path,
    query: str,
    collection_name: str = "dms_retrieval_documents",
    entity_ref: str | None = None,
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
    top_k: int = 10,
    embedding_dim: int = 384,
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
    embedding_max_tokens: int = 8192,
    embedding_timeout: int = 60,
) -> dict[str, Any]:
    return search_retrieval_documents(
        db_path,
        persist_dir=persist_dir,
        query=query,
        collection_name=collection_name,
        doc_type="episodic_memory_entity" if entity_ref else None,
        entity_ref=entity_ref,
        before_scene_order=before_scene_order,
        before_scene_id=before_scene_id,
        top_k=top_k,
        embedding_dim=embedding_dim,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_max_tokens=embedding_max_tokens,
        embedding_timeout=embedding_timeout,
    )


def search_retrieval_documents(
    db_path: str | Path,
    *,
    persist_dir: str | Path,
    query: str,
    collection_name: str = "dms_retrieval_documents",
    doc_type: str | None = None,
    doc_types: list[str] | None = None,
    entity_ref: str | None = None,
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
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
    sql_docs = get_retrieval_documents(
        db_path,
        doc_type=doc_type,
        doc_types=doc_types,
        entity_ref=entity_ref,
        before_scene_order=before_scene_order,
        before_scene_id=before_scene_id,
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
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )
    # Chroma metadata filters vary by version. Query broadly enough, then apply the
    # authoritative SQL time/entity filter above.
    collection_count = int(collection.count())
    if len(allowed_doc_ids) <= 200:
        candidate_count = max(collection_count, 1)
    else:
        candidate_count = min(
            max(top_k * 20, len(allowed_doc_ids) + top_k),
            max(collection_count, 1),
        )
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
        row = docs_by_id[doc_id]
        hit = {
            "doc_id": doc_id,
            "score": 1.0 - float(distance) if distance is not None else None,
            "distance": distance,
            "text": document,
            "metadata": metadata,
            "sql": row,
        }
        hit.update(_memory_payload(db_path, row))
        hits.append(hit)
        if len(hits) >= top_k:
            break

    return {
        "query": query,
        "doc_type": doc_type,
        "doc_types": doc_types,
        "entity_ref": entity_ref,
        "before_scene_id": before_scene_id,
        "before_scene_order": before_scene_order,
        "embedding": embedding_function.config_summary(),
        "count": len(hits),
        "results": hits,
    }


def build_embedding_function(
    *,
    provider: str,
    embedding_dim: int,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 8192,
    timeout: int = 60,
):
    provider = str(provider or "hash").strip().lower()
    if provider == "hash":
        return HashEmbeddingFunction(dim=embedding_dim)
    if provider == "openai":
        return OpenAICompatibleEmbeddingFunction(
            model_name=model_name or "bge-m3",
            base_url=base_url or "http://localhost:8080/v1",
            api_key=api_key,
            dimensions=embedding_dim,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


class HashEmbeddingFunction:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def name(self) -> str:
        return f"dms-local-hash-{self.dim}"

    def config_summary(self) -> dict[str, Any]:
        return {"provider": "hash", "mode": "local_hash", "dim": self.dim}

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        return [self._embed(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        return self(input)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingFunction:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        max_tokens: int = 8192,
        timeout: int = 60,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.batch_size = max(batch_size, 1)

    def name(self) -> str:
        dim = self.dimensions if self.dimensions is not None else "native"
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        return f"dms-openai-compatible-{safe_model}-{dim}"

    def config_summary(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        texts = [_truncate_for_embedding(str(text or ""), self.max_tokens) for text in input]
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index : index + self.batch_size]
            embeddings.extend(self._embed_batch(batch, include_dimensions=True))
        return embeddings

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API name.
        return self(input)

    def _embed_batch(self, texts: list[str], *, include_dimensions: bool) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        if include_dimensions and self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        try:
            response = self._post_embeddings(payload)
        except RuntimeError as exc:
            if include_dimensions and self.dimensions is not None and "dimension" in str(exc).lower():
                response = self._post_embeddings({"model": self.model_name, "input": texts})
            else:
                raise
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response did not include a data list.")
        records = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        embeddings = [record.get("embedding") for record in records if isinstance(record, dict)]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding response returned {len(embeddings)} embeddings for {len(texts)} inputs.")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _post_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed: HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError("Embedding response was not a JSON object.")
        return parsed


def _tokens(text: str) -> list[str]:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = [lowered[index : index + 2] for index in range(max(len(lowered) - 1, 0))]
    cjk_bigrams = [token for token in cjk_bigrams if any("\u4e00" <= char <= "\u9fff" for char in token)]
    return tokens + cjk_chars + cjk_bigrams


def _truncate_for_embedding(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    # Conservative approximation for OpenAI-compatible local services. Existing
    # DMS retrieval docs are short, so this normally returns the original text.
    max_chars = max_tokens * 4
    return text[:max_chars]


def _chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "doc_type": record.get("doc_type"),
        "source_id": record.get("source_id"),
        "memory_id": record.get("memory_id") or "",
        "entity_id": record.get("entity_id") or "",
        "parent_scene_id": record.get("parent_scene_id"),
        "scene_order": int(record.get("scene_order") or 0),
        "chunk_index": int(record.get("chunk_index") or 1),
        "sequence_index": int(record.get("sequence_index") or 1),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _memory_payload(db_path: str | Path, doc: dict[str, Any]) -> dict[str, Any]:
    memory_id = doc.get("memory_id")
    if not memory_id:
        return {}
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        memory = conn.execute("SELECT * FROM episodic_memories WHERE memory_id = ?", (memory_id,)).fetchone()
    if not memory:
        return {}
    payload = dict(memory)
    raw = _json_loads(payload.get("raw_json"), default={})
    payload["raw"] = raw
    if isinstance(raw, dict):
        payload["memory_temporal_scope"] = raw.get("memory_temporal_scope") or "temporal_episode"
        payload["memory_temporal_scope_confidence"] = raw.get("memory_temporal_scope_confidence")
        payload["memory_temporal_scope_reason"] = raw.get("memory_temporal_scope_reason")
    else:
        payload["memory_temporal_scope"] = "temporal_episode"
        payload["memory_temporal_scope_confidence"] = None
        payload["memory_temporal_scope_reason"] = None
    return {"memory": payload}


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Use the screenplay conda environment or install the optional vector DB dependency."
        ) from exc
    return chromadb
