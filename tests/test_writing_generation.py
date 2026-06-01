from __future__ import annotations

import json
from pathlib import Path

import dms.writing as writing
from dms.llm import LLMResult
from dms.scripts.wandering_earth import ScriptScene
from dms.writing import (
    SocialWritingGenerationConfig,
    _quick_eval,
    format_previous_scene_context,
    generate_writing_with_social_simulation,
)


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
    previous_path = tmp_path / "previous.md"
    memory_path.write_text("# Memory Packet\n", encoding="utf-8")
    cards_path.write_text("# Attribute Cards\n", encoding="utf-8")
    social_path.write_text("# Social Simulation\n", encoding="utf-8")
    style_path.write_text("张鹏：慢点。\n", encoding="utf-8")
    previous_path.write_text("Previous scene: scene_0004\nFull text:\n上一场景。\n", encoding="utf-8")
    monkeypatch.setattr(writing, "build_openai_client_from_config", lambda config, section: FakeWritingClient())

    summary = generate_writing_with_social_simulation(
        SocialWritingGenerationConfig(
            writing_request="写一段J20C返航，展现刘培强和张鹏互动。",
            memory_packet_path=memory_path,
            attribute_cards_path=cards_path,
            social_simulation_path=social_path,
            previous_scene_context_path=previous_path,
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
    assert (tmp_path / "out" / "previous_scene_context.md").is_file()
    assert "上一场景。" in (tmp_path / "out" / "prompt.md").read_text(encoding="utf-8")
    assert summary["inputs"]["previous_scene_context_chars"] > 0
    assert (tmp_path / "out" / "draft.md").read_text(encoding="utf-8").startswith("J20C")
    quick_eval = json.loads((tmp_path / "out" / "quick_eval.json").read_text(encoding="utf-8"))
    assert quick_eval["writer_packet_artifact_terms_present"] == []
    assert quick_eval["dialogue_risk_phrases_present"] == []
    assert quick_eval["request_anchors_present"]["J20C"]
    assert quick_eval["request_anchors_present"]["刘培强"]
    assert quick_eval["request_anchors_present"]["张鹏"]
    metadata_text = (tmp_path / "out" / "metadata.json").read_text(encoding="utf-8")
    assert "secret" not in metadata_text
    assert quick_eval["entities_present"]["UEG"]


def test_format_previous_scene_context_uses_full_text_for_short_scene() -> None:
    scene = ScriptScene(
        scene_id="scene_0004",
        source_record_id=4,
        discourse_index=4,
        title="4、INT.夜.驾驶舱",
        subtitle="",
        content="刘培强：收到。\n张鹏：稳住。",
        raw_heading_number=4,
        interior_exterior="INT",
        time_of_day="夜",
        location_hint="驾驶舱",
        character_count=16,
    )

    context = format_previous_scene_context(scene, max_chars=800)

    assert "Full text:" in context
    assert "刘培强：收到。" in context
    assert "Summary:" not in context


def test_format_previous_scene_context_uses_summary_and_entities_for_long_scene() -> None:
    scene = ScriptScene(
        scene_id="scene_0004",
        source_record_id=4,
        discourse_index=4,
        title="4、INT.夜.驾驶舱",
        subtitle="",
        content="刘培强：测试。" * 200,
        raw_heading_number=4,
        interior_exterior="INT",
        time_of_day="夜",
        location_hint="驾驶舱",
        character_count=1200,
    )

    context = format_previous_scene_context(scene, max_chars=80, summary="刘培强上一场景情绪紧绷。", entities=["张鹏"])

    assert "Full text:" not in context
    assert "Summary:" in context
    assert "刘培强上一场景情绪紧绷。" in context
    assert "Entities:" in context
    assert "张鹏" in context
    assert "刘培强" in context


def test_quick_eval_extracts_request_anchors_without_long_mixed_phrase() -> None:
    result = _quick_eval(
        "返航航线压过利伯维尔上空，刘培强沉默，张鹏提醒他拉升。",
        prompt_path=Path("prompt.md"),
        draft_path=Path("draft.md"),
        writing_request="描绘J20C返航途中飞越战区废墟，展现张鹏与刘培强互动。",
    )

    assert result["request_anchors"] == ["J20C", "刘培强", "张鹏"]
    assert result["missing_request_anchors"] == ["J20C"]


def test_quick_eval_accepts_supported_vehicle_aliases() -> None:
    result = _quick_eval(
        "歼20C飞越战区废墟，刘培强沉默，张鹏提醒他拉升。",
        prompt_path=Path("prompt.md"),
        draft_path=Path("draft.md"),
        writing_request="描绘J20C返航途中飞越战区废墟，展现张鹏与刘培强互动。",
    )

    assert result["request_anchors_present"]["J20C"] is True
    assert result["missing_request_anchors"] == []
