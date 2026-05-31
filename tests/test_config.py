from __future__ import annotations

import json
from pathlib import Path

from dms.config import build_openai_client_from_config, embedding_kwargs_from_config, load_local_config, redact_model_config


def test_load_local_config_and_build_writing_client(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  provider: openai
  model_name: qwen
  api_key: qwen-key
  base_url: http://127.0.0.1:8002
embedding:
  provider: openai
  model_name: bge-m3
  api_key: not-needed
  base_url: http://localhost:8080/v1
writing_llm:
  provider: openai
  model_name: gpt-5.5
  api_key: secret
  base_url: https://example.test/v1
  max_tokens: 1200
  temperature: 0.7
  timeout_seconds: 240
  reasoning_effort: high
  include_chat_template_kwargs: false
""",
        encoding="utf-8",
    )

    config = load_local_config(config_path)
    client = build_openai_client_from_config(config, "writing_llm")

    assert client.model == "gpt-5.5"
    assert client.base_url == "https://example.test/v1"
    assert client.reasoning_effort == "high"
    assert client.include_chat_template_kwargs is False
    assert redact_model_config(config["writing_llm"])["api_key"] == "***"
    assert "secret" not in json.dumps(redact_model_config(config["writing_llm"]))

    embedding_kwargs = embedding_kwargs_from_config(config)
    assert embedding_kwargs == {
        "embedding_provider": "openai",
        "embedding_model": "bge-m3",
        "embedding_base_url": "http://localhost:8080/v1",
        "embedding_api_key": "not-needed",
        "embedding_max_tokens": 8192,
        "embedding_timeout": 60,
        "embedding_dim": 384,
    }
