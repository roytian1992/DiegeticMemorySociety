from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    raw_response: dict[str, Any]
    usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, prompt: str) -> LLMResult:
        ...


class FakeSceneInventoryClient:
    """Deterministic test client for scene inventory runner tests."""

    provider = "fake"
    model = "fake-scene-inventory"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "setting": {
                "location": "FAKE_LOCATION",
                "time_hint": "FAKE_TIME",
                "spatial_context": "FAKE_SPATIAL_CONTEXT",
            },
            "stated_facts": [],
            "open_questions": [],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeSceneSummaryClient:
    """Deterministic test client for scene summary runner tests."""

    provider = "fake"
    model = "fake-scene-summary"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "summary": f"FAKE_SUMMARY {unit_id}",
            "salient_points": [f"FAKE_POINT {unit_id}"],
            "continuity_hooks": [f"FAKE_HOOK {unit_id}"],
            "retrieval_text": f"FAKE_SUMMARY {unit_id}\nFAKE_POINT {unit_id}\nFAKE_HOOK {unit_id}",
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeKGEntityMentionClient:
    """Deterministic test client for KG entity mention runner tests."""

    provider = "fake"
    model = "fake-kg-entity-mentions"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "entity_mentions": [
                {
                    "surface": "FAKE_CHARACTER",
                    "entity_type": "character",
                    "canonical_hint": "FAKE_CHARACTER",
                    "description": "a trackable test character",
                    "role_in_unit": "speaker",
                    "attributes_or_state": "present",
                    "evidence": unit_id,
                },
                {
                    "surface": "FAKE_DEVICE",
                    "entity_type": "object",
                    "canonical_hint": "FAKE_DEVICE",
                    "description": "a trackable test device",
                    "role_in_unit": "device",
                    "attributes_or_state": "active",
                    "evidence": unit_id,
                },
                {
                    "surface": "FAKE_CONCEPT",
                    "entity_type": "concept",
                    "canonical_hint": "FAKE_CONCEPT",
                    "description": "a trackable test concept",
                    "role_in_unit": "concept",
                    "attributes_or_state": "introduced",
                    "evidence": unit_id,
                },
                {
                    "surface": "FAKE_OCCASION",
                    "entity_type": "occasion",
                    "canonical_hint": "FAKE_OCCASION",
                    "description": "a trackable test occasion",
                    "role_in_unit": "trackable occasion",
                    "attributes_or_state": "planned recurring occasion",
                    "evidence": unit_id,
                },
            ],
            "unresolved_mentions": [
                {
                    "surface": "FAKE_UNKNOWN",
                    "reason": "referent unclear in local unit",
                    "evidence": unit_id,
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeKGEntityRefinementClient:
    """Deterministic refinement client that returns the initial KG payload."""

    provider = "fake"
    model = "fake-kg-entity-refinement"

    def complete(self, prompt: str) -> LLMResult:
        payload = _extract_last_json_object_after_marker(prompt, "# Initial Entity Output To Refine")
        data = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        if not isinstance(data, dict):
            data = {"unit_id": _extract_unit_id(prompt), "entity_mentions": [], "scene_tags": [], "unresolved_mentions": []}
        data.setdefault("scene_tags", [])
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeSceneEventClient:
    """Deterministic test client for scene event candidate runner tests."""

    provider = "fake"
    model = "fake-scene-events"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "events": [
                {
                    "event_id_hint": f"{unit_id}_event_001",
                    "summary": "FAKE_EVENT",
                    "participants": ["FAKE_CHARACTER"],
                    "location": "FAKE_LOCATION",
                    "event_type": "action",
                    "evidence": unit_id,
                }
            ],
            "knowledge_transfers": [
                {
                    "source": "FAKE_SOURCE",
                    "receiver": "FAKE_CHARACTER",
                    "content": "FAKE_KNOWLEDGE",
                    "epistemic_status": "knows",
                    "evidence": unit_id,
                }
            ],
            "state_changes": [
                {
                    "entity": "FAKE_OBJECT",
                    "before": "FAKE_BEFORE",
                    "after": "FAKE_AFTER",
                    "evidence": unit_id,
                }
            ],
            "thread_candidates": [
                {
                    "thread_type": "setup",
                    "summary": "FAKE_THREAD",
                    "evidence": unit_id,
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeVisibilityNotesClient:
    """Deterministic test client for visibility notes runner tests."""

    provider = "fake"
    model = "fake-visibility-notes"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "visibility_records": [
                {
                    "fact_or_event": "FAKE_EVENT",
                    "character": "FAKE_CHARACTER",
                    "visibility": "observed",
                    "evidence": unit_id,
                }
            ],
            "hidden_or_future_sensitive_items": [
                {
                    "item": "FAKE_HIDDEN_ITEM",
                    "hidden_from": ["FAKE_CHARACTER"],
                    "reason": "FAKE_REASON",
                    "evidence": unit_id,
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeEpisodicMemoryClient:
    """Deterministic test client for episodic memory runner tests."""

    provider = "fake"
    model = "fake-episodic-memories"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        evidence = _extract_first_source_evidence(prompt)
        payload = {
            "unit_id": unit_id,
            "episodic_memories": [
                {
                    "memory_id_hint": f"{unit_id}_memory_001",
                    "sequence_index": 1,
                    "timeline_label": unit_id,
                    "memory_type": "action",
                    "summary": "FAKE_CHARACTER interacts with FAKE_DEVICE.",
                    "evidence": evidence,
                    "entity_links": [
                        {
                            "entity": "FAKE_CHARACTER",
                            "entity_type": "character",
                            "link_role": "actor",
                            "evidence": evidence,
                        },
                        {
                            "entity": "FAKE_DEVICE",
                            "entity_type": "object",
                            "link_role": "object",
                            "evidence": evidence,
                        },
                        {
                            "entity": "FAKE_CONCEPT",
                            "entity_type": "concept",
                            "link_role": "concept",
                            "evidence": evidence,
                        },
                    ],
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class FakeDurableRelationshipClient:
    """Deterministic test client for durable relationship runner tests."""

    provider = "fake"
    model = "fake-durable-relationships"

    def complete(self, prompt: str) -> LLMResult:
        unit_id = _extract_unit_id(prompt)
        payload = {
            "unit_id": unit_id,
            "relationship_observations": [
                {
                    "source_entity": "FAKE_CHARACTER",
                    "target_entity": "FAKE_DEVICE",
                    "relation_type": "responsible_for",
                    "status_or_change": "FAKE_CHARACTER has a durable responsibility for FAKE_DEVICE.",
                    "evidence": _extract_first_source_evidence(prompt),
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class AnthropicMessagesClient:
    """Small Anthropic Messages API compatible client using urllib.

    Environment defaults:
    - ANTHROPIC_BASE_URL, e.g. http://host/api
    - ANTHROPIC_AUTH_TOKEN
    - ANTHROPIC_MODEL
    """

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        auth_token: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        if not self.auth_token:
            raise ValueError("Missing Anthropic auth token. Set ANTHROPIC_AUTH_TOKEN or pass auth_token.")

    def complete(self, prompt: str) -> LLMResult:
        endpoint = self.base_url
        if not endpoint.endswith("/v1/messages"):
            endpoint = endpoint + "/v1/messages"

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.auth_token or "",
            "anthropic-version": "2023-06-01",
            "authorization": f"Bearer {self.auth_token}",
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc

        text = _extract_anthropic_text(raw)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return LLMResult(text=text, provider=self.provider, model=self.model, raw_response=raw, usage=usage)


class OpenAIChatClient:
    """OpenAI-compatible /v1/chat/completions client."""

    provider = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0,
        timeout_seconds: int = 120,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        include_chat_template_kwargs: bool = True,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL") or "Qwen3.5-397B-A17B-FP8"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8001").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.include_chat_template_kwargs = include_chat_template_kwargs

    def complete(self, prompt: str) -> LLMResult:
        endpoint = self.base_url
        if endpoint.endswith("/v1"):
            endpoint = endpoint + "/chat/completions"
        elif not endpoint.endswith("/v1/chat/completions"):
            endpoint = endpoint + "/v1/chat/completions"

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.include_chat_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

        text = _extract_openai_chat_text(raw)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return LLMResult(text=text, provider=self.provider, model=self.model, raw_response=raw, usage=usage)


def _extract_openai_chat_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(raw.get("text"), str):
        return raw["text"]
    return json.dumps(raw, ensure_ascii=False)


def _extract_anthropic_text(raw: dict[str, Any]) -> str:
    content = raw.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    if isinstance(raw.get("completion"), str):
        return raw["completion"]
    if isinstance(raw.get("text"), str):
        return raw["text"]
    return json.dumps(raw, ensure_ascii=False)


def _extract_unit_id(prompt: str) -> str:
    for marker in ('"unit_id":', '"scene_id":'):
        idx = prompt.rfind(marker)
        if idx == -1:
            continue
        after = prompt[idx + len(marker) :].lstrip()
        if not after.startswith('"'):
            continue
        after = after[1:]
        end = after.find('"')
        if end != -1:
            return after[:end]
    return "unit_unknown"


def _extract_first_source_evidence(prompt: str) -> str:
    payload = _extract_last_json_object_after_marker(prompt, "# Current Narrative Unit")
    if isinstance(payload, dict):
        for field in ("content", "title", "subtitle"):
            value = str(payload.get(field) or "").strip()
            if value:
                return value[: min(len(value), 24)]
    return _extract_unit_id(prompt)


def _extract_last_json_object_after_marker(prompt: str, marker: str) -> Any:
    start = prompt.rfind(marker)
    if start == -1:
        return None
    text = prompt[start + len(marker) :]
    object_start = text.find("{")
    if object_start == -1:
        return None
    absolute_start = start + len(marker) + object_start
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(prompt[absolute_start:], start=absolute_start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(prompt[absolute_start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None
