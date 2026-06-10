from __future__ import annotations

import json
from pathlib import Path

from dms.llm import LLMResult
from dms.simulation import AttributeCardConfig, build_entity_attribute_cards, format_attribute_cards_markdown


class FakeAttributeCardClient:
    provider = "fake"
    model = "fake-attribute-card"

    def complete(self, prompt: str) -> LLMResult:
        if "刘培强" in prompt:
            name = "刘培强"
        else:
            name = "张鹏"
        payload = {
            "entity_id": "character_001",
            "canonical_name": name,
            "entity_type": "character",
            "prefix_boundary": "scene_0006",
            "role_in_story": [{"value": "飞行员", "status": "explicit", "refs": ["M1"]}],
            "current_state": [{"value": "正在返航前", "status": "explicit", "refs": ["M1"]}],
            "salient_past_actions": [{"action": "曾驾驶J20C", "status": "explicit", "refs": ["M1"]}],
            "stable_traits": [{"trait": "情绪直接", "status": "inferred", "refs": ["M1", "R1"]}],
            "speaking_style": [],
            "values_or_motivations": [],
            "relationship_stances": [
                {"target": "张鹏", "stance": "接受提醒但可能回避", "status": "inferred", "refs": ["M1"]}
            ],
            "behavior_tendencies": [],
            "hard_constraints": [{"constraint": "不能知道未来信息", "refs": ["M1"]}],
            "simulation_risks": [{"risk": "不要把短暂焦躁写成永久性格定论", "refs": ["M1"]}],
            "uncertain_or_unsupported": [{"claim": "年龄", "reason": "context does not support exact age"}],
        }
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_build_entity_attribute_cards_from_memory_packet(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    creative_context_path = tmp_path / "creative_context_packet.json"
    packet_path.write_text(json.dumps(_packet(), ensure_ascii=False), encoding="utf-8")
    creative_context_path.write_text(json.dumps(_creative_context_packet(), ensure_ascii=False), encoding="utf-8")

    summary = build_entity_attribute_cards(
        AttributeCardConfig(
            memory_packet_path=packet_path,
            creative_context_packet_path=creative_context_path,
            output_dir=tmp_path / "cards",
            overwrite=True,
        ),
        llm_client=FakeAttributeCardClient(),
    )

    assert summary["card_count"] == 1
    assert summary["inputs"]["creative_context_packet_path"] == str(creative_context_path)
    card = summary["cards"][0]
    assert card["canonical_name"] == "刘培强"
    assert card["salient_past_actions"][0]["action"] == "曾驾驶J20C"
    assert card["stable_traits"][0]["status"] == "inferred"
    assert "simulation_constraints" not in card
    assert card["hard_constraints"][0]["constraint"] == "不能知道未来信息"
    assert card["author_profile_baseline"]["stable_traits"] == ["嘴硬", "抗压"]
    assert card["author_initial_state"]["beliefs"] == ["地球处境正在恶化"]
    assert card["author_profile_policy"]["priority"] == "author_locked"
    assert (tmp_path / "cards" / "attribute_cards.json").is_file()
    context = json.loads((tmp_path / "cards" / "inputs" / "character_001.json").read_text(encoding="utf-8"))
    assert context["entity"]["author_profile"]["speaking_style"] == ["短句", "压着情绪说"]
    assert context["character_reference_knowledge"][0]["ref_id"] == "REF:ref_liu_knows_550a"
    assert any("不要把刘培强写成完全放下嘴硬" in note for note in context["creative_context_notes"])
    assert "REF:ref_liu_knows_550a" in context["instructions"]["available_reference_ids"]
    markdown = (tmp_path / "cards" / "attribute_cards.md").read_text(encoding="utf-8")
    assert "## 刘培强 (character)" in markdown
    assert "stable traits" in markdown
    assert "salient past actions" in markdown
    assert "hard constraints" in markdown
    assert "simulation risks" in markdown
    assert "不能知道未来信息" in markdown
    assert "author profile baseline" in markdown
    assert "嘴硬" in markdown


def test_format_attribute_cards_markdown_handles_variable_sections() -> None:
    markdown = format_attribute_cards_markdown(
        {
            "inputs": {"memory_packet_path": "packet.json"},
            "card_count": 1,
            "cards": [
                {
                    "canonical_name": "张鹏",
                    "entity_type": "character",
                    "prefix_boundary": "scene_0006",
                    "stable_traits": [{"trait": "务实", "status": "inferred", "refs": ["M1"]}],
                }
            ],
        }
    )

    assert "# Entity Attribute Cards" in markdown
    assert "## 张鹏 (character)" in markdown
    assert "务实" in markdown


def test_attribute_card_prefix_boundary_uses_narrative_unit_label(tmp_path: Path) -> None:
    packet = _packet()
    packet["retrieval_boundary"] = {
        "before_unit_id": "chapter_0006",
        "unit_type": "chapter",
        "unit_label": "chapter",
        "before_scene_id": "scene_0006",
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_attribute_cards(
        AttributeCardConfig(
            memory_packet_path=packet_path,
            output_dir=tmp_path / "cards",
            overwrite=True,
        ),
        llm_client=FakeAttributeCardClient(),
    )

    assert summary["cards"][0]["prefix_boundary"] == "before chapter_0006"


def test_attribute_cards_demote_unsupported_formal_roles(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    packet = _packet()
    packet["entities"][0]["canonical_name"] = "张鹏"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")

    class Client:
        provider = "fake"
        model = "fake"

        def complete(self, prompt: str) -> LLMResult:
            payload = {
                "canonical_name": "张鹏",
                "entity_type": "character",
                "prefix_boundary": "before scene_0006",
                "role_in_story": [{"value": "刘培强的监护人和教官", "status": "inferred", "refs": ["R1"]}],
                "hard_constraints": [{"constraint": "必须提及太阳危机真实存在", "refs": ["R1"]}],
                "simulation_risks": [{"risk": "不应将张鹏视为法律意义上的监护人或教官", "refs": ["R1"]}],
                "uncertain_or_unsupported": [{"claim": "张鹏是法定监护人", "reason": "unsupported"}],
            }
            text = json.dumps(payload, ensure_ascii=False)
            return LLMResult(text=text, provider=self.provider, model=self.model, raw_response={}, usage={})

    summary = build_entity_attribute_cards(
        AttributeCardConfig(memory_packet_path=packet_path, output_dir=tmp_path / "cards", overwrite=True),
        llm_client=Client(),
    )

    card = summary["cards"][0]
    assert card["role_in_story"] == []
    assert card["hard_constraints"][0]["constraint"] == "太阳危机真实存在"
    assert any(item["claim"] == "刘培强的监护人和教官" for item in card["uncertain_or_unsupported"])


def _packet() -> dict:
    return {
        "retrieval_boundary": {"before_scene_id": "scene_0006"},
        "entities": [
            {
                "entity_id": "character_001",
                "canonical_name": "刘培强",
                "entity_type": "character",
                "profile": "刘培强是飞行员。",
                "current_state": "准备返航。",
                "author_profile": {
                    "stable_traits": ["嘴硬", "抗压"],
                    "speaking_style": ["短句", "压着情绪说"],
                    "behavior_constraints": ["不能提前知道未来剧情"],
                },
                "author_profile_summary": "traits=嘴硬、抗压；speaking_style=短句、压着情绪说",
                "initial_state": {"beliefs": ["地球处境正在恶化"]},
                "profile_policy": {"priority": "author_locked", "visibility": "author_guidance"},
                "source_refs": ["R1"],
                "related_memory_index": ["M1"],
            },
            {
                "entity_id": "object_001",
                "canonical_name": "J20C",
                "entity_type": "object",
                "related_memory_index": [],
            },
        ],
        "relations": [],
        "episodic_memories": [
            {
                "index": "M1",
                "memory_id": "m1",
                "scene_id": "scene_0005",
                "summary": "刘培强驾驶J20C。",
                "source_ref": "R1",
            }
        ],
        "references": [
            {
                "ref_id": "R1",
                "scene_id": "scene_0005",
                "text": "刘培强驾驶J20C。",
            }
        ],
        "character_reference_knowledge": [
            {
                "item_id": "ref_liu_knows_550a",
                "item_type": "character_profile",
                "subject": "刘培强",
                "statement": "刘培强知道550A能够分析脑电波。",
                "knowledge_scope": "character_private",
                "known_to": ["刘培强"],
                "available_from": "story_start",
            },
            {
                "item_id": "ref_zhang_private",
                "item_type": "character_profile",
                "subject": "张鹏",
                "statement": "张鹏知道另一条私密设定。",
                "knowledge_scope": "character_private",
                "known_to": ["张鹏"],
                "available_from": "story_start",
            },
        ],
    }


def _creative_context_packet() -> dict:
    return {
        "conversation_guidance": [
            {
                "item_id": "conv:liu:constraint",
                "source_type": "conversation",
                "status": "active",
                "authority": "user_explicit",
                "subject": "刘培强",
                "statement": "不要把刘培强写成完全放下嘴硬。",
                "entity_ids": ["character_001"],
                "visibility": "author_only",
            }
        ],
        "external_reference_context": [
            {
                "item_id": "external:liu:profile",
                "source_type": "external_reference",
                "status": "active",
                "authority": "external_source",
                "subject": "刘培强",
                "statement": "外部资料称刘培强是飞行员。",
                "entity_ids": ["character_001"],
                "visibility": "author_only",
            }
        ],
    }
