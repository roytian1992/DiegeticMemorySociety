from __future__ import annotations

import json
from pathlib import Path

import pytest

from dms.llm import LLMResult
from dms.simulation import SocialSimulationConfig, run_social_simulation


class FakeSocialSimulationClient:
    provider = "fake"
    model = "fake-social-simulation"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        if "Simulate how the target character" in prompt:
            character = "刘培强" if "刘培强" in prompt else "张鹏"
            payload = {
                "character": character,
                "prefix_boundary": "before scene_0006",
                "intent_assumptions": ["返航途中是写作请求提供的新场景条件"],
                "likely_internal_state": [
                    {"value": f"{character}处在返航前的紧张状态", "status": "inferred", "refs": ["M1"]}
                ],
                "likely_actions": [
                    {"value": f"{character}会用动作压住情绪", "status": "inferred", "refs": ["M1"]}
                ],
                "likely_dialogue": [
                    {"value": f"{character}说话偏短促", "status": "inferred", "refs": ["M1"]}
                ],
                "interaction_pressure": [
                    {"target": "对方", "pressure": "形成提醒与回避的张力", "status": "inferred", "refs": ["M1"]}
                ],
                "avoid_or_risks": [{"risk": "不要写成知道后续事件", "refs": ["M1"]}],
                "memory_basis": [{"point": "返航前关系张力", "refs": ["M1"]}],
            }
        elif "Coordinate the character simulations" in prompt:
            payload = {
                "simulation_id": "scene6_social_test",
                "prefix_boundary": "before scene_0006",
                "scene_beats": [
                    {
                        "beat": "低空返航动作先制造压迫感，再让提醒与回避浮出。",
                        "participants": ["刘培强", "张鹏"],
                        "purpose": "建立人物张力",
                        "intent_basis": ["低空返航来自写作请求"],
                        "memory_basis": ["M1"],
                        "risks": ["不要透露未来"],
                    }
                ],
                "character_dynamics": [
                    {"source": "张鹏", "target": "刘培强", "dynamic": "提醒和压舱", "refs": ["M1"]}
                ],
                "memory_risks": [{"risk": "避免补写无证据军衔", "refs": ["M1"]}],
                "writer_guidance": [{"guidance": "先动作后对白", "refs": ["M1"]}],
            }
        else:
            raise AssertionError(f"Unexpected prompt:\n{prompt[:500]}")
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_run_social_simulation_from_attribute_cards(tmp_path: Path) -> None:
    cards_path = tmp_path / "attribute_cards.json"
    cards_path.write_text(json.dumps(_cards(), ensure_ascii=False), encoding="utf-8")
    client = FakeSocialSimulationClient()

    summary = run_social_simulation(
        SocialSimulationConfig(
            attribute_cards_path=cards_path,
            social_simulation_intent="刘培强和张鹏在返航途中互动。",
            output_dir=tmp_path / "simulation",
            overwrite=True,
        ),
        llm_client=client,
    )

    assert summary["character_simulation_count"] == 2
    assert summary["inputs"]["social_simulation_intent"] == "刘培强和张鹏在返航途中互动。"
    assert summary["inputs"]["source_isolation"]["target_scene_text_visible"] is False
    assert summary["inputs"]["source_isolation"]["writing_spec_visible"] is False
    assert len(summary["social_simulation"]["scene_beats"]) == 1
    assert summary["algorithmic_social_plan"]["version"] == "asip_v0"
    selected_sequence = summary["algorithmic_social_plan"]["selected_sequence"]
    assert selected_sequence["selection_strategy"] == "multi_candidate_rerank_v1"
    assert selected_sequence["candidate_sequence_count"] >= 1
    assert "score_components" in selected_sequence
    assert summary["social_simulation_metrics"]["candidate_action_count"] >= 1
    assert summary["social_simulation_metrics"]["candidate_sequence_count"] >= 1
    assert summary["verification"]["status"] in {"pass", "warn"}
    assert summary["writer_packet_verification"]["status"] in {"pass", "warn"}
    assert len(client.prompts) == 3
    assert (tmp_path / "simulation" / "character_simulations.json").is_file()
    assert (tmp_path / "simulation" / "social_simulation.json").is_file()
    assert (tmp_path / "simulation" / "algorithmic_social_plan.json").is_file()
    assert (tmp_path / "simulation" / "verification.json").is_file()
    assert (tmp_path / "simulation" / "writer_packet_verification.json").is_file()
    assert (tmp_path / "simulation" / "writer_packet.md").is_file()
    markdown = (tmp_path / "simulation" / "social_simulation.md").read_text(encoding="utf-8")
    assert "# Social Simulation" in markdown
    assert "social simulation intent" in markdown
    assert "target scene text visible: no" in markdown
    assert "intent assumptions" in markdown
    assert "intent basis" in markdown
    assert "dialogue posture" in markdown
    assert "## Algorithmic Social Plan" in markdown
    assert "## Verification" in markdown
    assert "低空返航动作先制造压迫感" in markdown
    writer_packet = (tmp_path / "simulation" / "writer_packet.md").read_text(encoding="utf-8")
    assert "# Social Simulation Writer Packet" in writer_packet
    assert "target scene text visible: no" in writer_packet
    assert "## Optional Interaction Functions" in writer_packet
    assert "posture only" in writer_packet
    for prompt in client.prompts:
        assert "social_simulation_intent" in prompt
        assert '"writing_spec"' not in prompt
        assert "J20C返航途中穿越战区废墟时的紧张氛围" not in prompt


def test_social_simulation_rejects_target_scene_source_fields(tmp_path: Path) -> None:
    cards = _cards()
    cards[0]["content"] = "这是目标 scene 6 原文，模拟阶段不应该看到。"
    cards_path = tmp_path / "attribute_cards.json"
    cards_path.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Forbidden target-scene source field"):
        run_social_simulation(
            SocialSimulationConfig(
                attribute_cards_path=cards_path,
                social_simulation_intent="刘培强和张鹏在返航途中互动。",
                output_dir=tmp_path / "simulation",
                overwrite=True,
            ),
            llm_client=FakeSocialSimulationClient(),
        )


def _cards() -> list[dict]:
    return [
        {
            "entity_id": "character_0011",
            "canonical_name": "刘培强",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "current_state": [{"value": "准备返航", "status": "explicit", "refs": ["M1"]}],
            "stable_traits": [{"trait": "情绪外显", "status": "inferred", "refs": ["M1"]}],
            "relationship_stances": [{"target": "张鹏", "stance": "受其提醒", "status": "inferred", "refs": ["M1"]}],
            "hard_constraints": [{"constraint": "不能知道未来信息", "refs": ["M1"]}],
            "simulation_risks": [{"risk": "不要把焦躁写成永久定性", "refs": ["M1"]}],
        },
        {
            "entity_id": "character_0017",
            "canonical_name": "张鹏",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "current_state": [{"value": "与刘培强同行", "status": "explicit", "refs": ["M1"]}],
            "stable_traits": [{"trait": "务实", "status": "inferred", "refs": ["M1"]}],
            "relationship_stances": [{"target": "刘培强", "stance": "提醒与照看", "status": "inferred", "refs": ["M1"]}],
            "hard_constraints": [],
            "simulation_risks": [],
        },
    ]
