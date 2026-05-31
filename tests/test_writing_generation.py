from __future__ import annotations

import json
from pathlib import Path

import dms.writing as writing
from dms.llm import LLMResult
from dms.writing import SocialWritingGenerationConfig, generate_writing_with_social_simulation


class FakeWritingClient:
    provider = "fake"
    model = "fake-writing"

    def complete(self, prompt: str) -> LLMResult:
        text = "J20C掠过海面，刘培强沉默，张鹏提醒他稳住，UEG基地灯光渐近。"
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_generate_writing_with_social_simulation_uses_config_and_redacts_key(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
writing_llm:
  provider: openai
  model_name: fake-writing
  api_key: secret
  base_url: http://fake.test/v1
""",
        encoding="utf-8",
    )
    memory_path = tmp_path / "memory.md"
    cards_path = tmp_path / "cards.md"
    social_path = tmp_path / "social.md"
    style_path = tmp_path / "style.md"
    memory_path.write_text("# Memory Packet\n", encoding="utf-8")
    cards_path.write_text("# Attribute Cards\n", encoding="utf-8")
    social_path.write_text("# Social Simulation\n", encoding="utf-8")
    style_path.write_text("张鹏：慢点。\n", encoding="utf-8")
    monkeypatch.setattr(writing, "build_openai_client_from_config", lambda config, section: FakeWritingClient())

    summary = generate_writing_with_social_simulation(
        SocialWritingGenerationConfig(
            writing_request="写一段返航。",
            memory_packet_path=memory_path,
            attribute_cards_path=cards_path,
            social_simulation_path=social_path,
            style_reference_path=style_path,
            output_dir=tmp_path / "out",
            model_config_path=config_path,
            length_requirement="127-182字。",
            output_requirements="中文。",
            overwrite=True,
        )
    )

    assert summary["model_config"]["api_key"] == "***"
    assert (tmp_path / "out" / "prompt.md").is_file()
    assert (tmp_path / "out" / "draft.md").read_text(encoding="utf-8").startswith("J20C")
    metadata_text = (tmp_path / "out" / "metadata.json").read_text(encoding="utf-8")
    assert "secret" not in metadata_text
    assert json.loads((tmp_path / "out" / "quick_eval.json").read_text(encoding="utf-8"))["entities_present"]["UEG"]
