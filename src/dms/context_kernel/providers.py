from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from dms.llm import LLMClient, LLMResult, OpenAIChatClient
from dms.storage.chroma_index import build_embedding_function


class ContextLLMProvider(Protocol):
    provider: str
    model: str

    def generate_text(self, prompt: str) -> str:
        ...

    def generate_json(self, prompt: str) -> dict[str, Any]:
        ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class RerankerProvider(Protocol):
    def rerank(self, query: str, items: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class OpenAICompatibleLLMConfig:
    model: str
    base_url: str
    api_key: str = "token-abc123"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_seconds: int = 120


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingConfig:
    model: str = "bge-m3"
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = "token-abc123"
    embedding_dim: int = 1024
    max_tokens: int = 8192
    timeout: int = 60


@dataclass(frozen=True)
class FastAPIRerankerConfig:
    base_url: str = "http://127.0.0.1:8090"
    model: str = "bge-reranker-base"
    timeout: int = 60


class LLMClientProvider:
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.provider = client.provider
        self.model = client.model

    def generate_text(self, prompt: str) -> str:
        return self.client.complete(prompt).text

    def generate_json(self, prompt: str) -> dict[str, Any]:
        text = self.generate_text(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise


def openai_compatible_llm_provider(config: OpenAICompatibleLLMConfig) -> LLMClientProvider:
    return LLMClientProvider(
        OpenAIChatClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            enable_thinking=False,
        )
    )


class ChromaEmbeddingProvider:
    def __init__(self, config: OpenAICompatibleEmbeddingConfig | None = None, *, provider: str = "openai") -> None:
        self.config = config or OpenAICompatibleEmbeddingConfig()
        if provider == "hash" and config is None:
            self.config = OpenAICompatibleEmbeddingConfig(model="hash", base_url="", api_key="", embedding_dim=384)
        self.embedding_function = build_embedding_function(
            provider=provider,
            embedding_dim=self.config.embedding_dim,
            model_name=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_function.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_function.embed_query([text])[0]


class ScoreReranker:
    def rerank(self, query: str, items: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:top_k]


class FastAPIReranker:
    def __init__(self, config: FastAPIRerankerConfig | None = None) -> None:
        self.config = config or FastAPIRerankerConfig()

    def rerank(self, query: str, items: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "documents": [str(item.get("statement") or item.get("text") or "") for item in items],
            "model": self.config.model,
            "top_k": top_k,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/rerank",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception:
            return ScoreReranker().rerank(query, items, top_k=top_k)
        scores = result.get("scores") or result.get("results") or []
        scored = []
        for index, item in enumerate(items):
            score = _score_at(scores, index)
            payload_item = dict(item)
            if score is not None:
                payload_item["rerank_score"] = score
            scored.append(payload_item)
        return sorted(
            scored,
            key=lambda item: float(item.get("rerank_score", item.get("score") or 0.0)),
            reverse=True,
        )[:top_k]


def _score_at(scores: Any, index: int) -> float | None:
    if isinstance(scores, list):
        if index >= len(scores):
            return None
        value = scores[index]
        if isinstance(value, dict):
            value = value.get("score")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None
