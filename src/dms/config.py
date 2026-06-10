from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dms.llm import OpenAIChatClient


def load_local_config(path: str | Path = "configs/local_config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML object: {config_path}")
    return payload


def build_openai_client_from_config(
    config: dict[str, Any],
    section: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> OpenAIChatClient:
    block = config.get(section)
    if not isinstance(block, dict):
        raise ValueError(f"Missing config section: {section}")
    provider = str(block.get("provider") or "openai").strip().lower()
    if provider != "openai":
        raise ValueError(f"Only openai-compatible config is supported for {section}: {provider}")
    values = {**block, **{key: value for key, value in (overrides or {}).items() if value is not None}}
    include_chat_template_kwargs = values.get("include_chat_template_kwargs", True)
    return OpenAIChatClient(
        model=values.get("model_name") or values.get("model"),
        base_url=values.get("base_url"),
        api_key=values.get("api_key"),
        max_tokens=int(values.get("max_tokens") or 2048),
        temperature=float(values.get("temperature") or 0),
        timeout_seconds=int(values.get("timeout_seconds") or values.get("timeout") or 120),
        enable_thinking=bool(values.get("enable_thinking", False)),
        reasoning_effort=values.get("reasoning_effort"),
        thinking=_normalise_thinking_config(values.get("thinking")),
        include_chat_template_kwargs=_normalise_bool(include_chat_template_kwargs),
    )


def _normalise_thinking_config(value: Any) -> dict[str, Any] | None:
    if value is None or value is False:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"type": value.strip()}
    raise ValueError("thinking config must be a mapping, string, false, or null")


def _normalise_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def embedding_kwargs_from_config(config: dict[str, Any], section: str = "embedding") -> dict[str, Any]:
    block = config.get(section)
    if not isinstance(block, dict):
        raise ValueError(f"Missing config section: {section}")
    provider = str(block.get("provider") or "hash").strip().lower()
    return {
        "embedding_provider": provider,
        "embedding_model": block.get("model_name") or block.get("model"),
        "embedding_base_url": block.get("base_url"),
        "embedding_api_key": block.get("api_key"),
        "embedding_max_tokens": int(block.get("max_tokens") or 8192),
        "embedding_timeout": int(block.get("timeout_seconds") or block.get("timeout") or 60),
        "embedding_dim": int(block.get("dimensions") or block.get("embedding_dim") or block.get("dim") or 384),
    }


def redact_model_config(block: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(block)
    if "api_key" in redacted:
        redacted["api_key"] = "***"
    return redacted
