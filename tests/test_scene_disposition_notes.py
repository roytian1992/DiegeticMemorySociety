from __future__ import annotations

import json
from pathlib import Path

from dms.llm import LLMResult
from dms.simulation import (
    SceneDispositionNoteConfig,
    build_scene_disposition_notes,
    format_scene_disposition_notes_markdown,
)


class FakeDispositionClient:
    provider = "fake"
    model = "fake-disposition"

    def complete(self, prompt: str) -> LLMResult:
        name = "刘培强" if "刘培强" in prompt else "张鹏"
        payload = {
            "entity_id": "character_001",
            "canonical_name": name,
            "scene_disposition_note": f"{name}在当前场景里仍保留既有性格基底，但返航互动会让他更收敛；依据 M1 和 AUTHOR_PROFILE。",
        }
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResult(text=text, provider=self.provider, model=self.model, raw_response={}, usage={"prompt_chars": len(prompt)})


def test_build_scene_disposition_notes_keeps_flat_schema(tmp_path: Path) -> None:
    cards_path = tmp_path / "attribute_cards.json"
    memory_packet_path = tmp_path / "memory_packet.json"
    cards_path.write_text(json.dumps(_cards(), ensure_ascii=False), encoding="utf-8")
    memory_packet_path.write_text(json.dumps(_memory_packet(), ensure_ascii=False), encoding="utf-8")

    summary = build_scene_disposition_notes(
        SceneDispositionNoteConfig(
            attribute_cards_path=cards_path,
            memory_packet_path=memory_packet_path,
            social_simulation_intent="刘培强和张鹏在返航途中互动。",
            output_dir=tmp_path / "notes",
            overwrite=True,
        ),
        llm_client=FakeDispositionClient(),
    )

    assert summary["note_count"] == 2
    assert summary["inputs"]["memory_packet_path"] == str(memory_packet_path)
    note = summary["scene_disposition_notes"][0]
    assert set(note) == {"entity_id", "canonical_name", "scene_disposition_note"}
    assert "scene_disposition_note" in note
    assert "返航互动" in note["scene_disposition_note"]
    assert (tmp_path / "notes" / "scene_disposition_notes.json").is_file()
    markdown = (tmp_path / "notes" / "scene_disposition_notes.md").read_text(encoding="utf-8")
    assert "# Scene Disposition Notes" in markdown
    assert "memory_packet.json" in markdown
    assert "刘培强" in markdown
    context = json.loads((tmp_path / "notes" / "inputs" / "disposition_character_001.json").read_text(encoding="utf-8"))
    assert context["relevant_memory_notes"] == [
        "M1 <scene_0004> scope=temporal_episode: 刘培强返航途中情绪焦躁",
        "M2 <scene_0004> scope=atemporal_fact: 月球发动机危机是所有返航任务的背景风险",
    ]
    assert context["relevant_reference_notes"] == [
        "REF:ref_liu_550a 刘培强: 刘培强知道550A能够分析脑电波。 [character_private]"
    ]


def test_format_scene_disposition_notes_markdown() -> None:
    markdown = format_scene_disposition_notes_markdown(
        {
            "inputs": {"attribute_cards_path": "cards.json", "social_simulation_intent": "返航互动"},
            "note_count": 1,
            "scene_disposition_notes": [
                {
                    "entity_id": "character_001",
                    "canonical_name": "刘培强",
                    "scene_disposition_note": "在张鹏提醒下更收敛。",
                }
            ],
        }
    )

    assert "cards.json" in markdown
    assert "返航互动" in markdown
    assert "在张鹏提醒下更收敛" in markdown


def _cards() -> list[dict]:
    return [
        {
            "entity_id": "character_001",
            "canonical_name": "刘培强",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "stable_traits": [{"trait": "嘴硬", "status": "inferred", "refs": ["M1"]}],
            "author_profile_summary": "traits=嘴硬、抗压",
        },
        {
            "entity_id": "character_002",
            "canonical_name": "张鹏",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "stable_traits": [{"trait": "务实", "status": "inferred", "refs": ["M1"]}],
        },
    ]


def _memory_packet() -> dict:
    return {
        "entities": [
            {
                "entity_id": "character_001",
                "canonical_name": "刘培强",
                "related_memory_index": ["M1"],
            },
            {
                "entity_id": "character_002",
                "canonical_name": "张鹏",
                "related_memory_index": [],
            },
        ],
        "episodic_memories": [
            {
                "index": "M1",
                "scene_id": "scene_0004",
                "memory_temporal_scope": "temporal_episode",
                "summary": "刘培强返航途中情绪焦躁",
            },
            {
                "index": "M2",
                "scene_id": "scene_0004",
                "memory_temporal_scope": "atemporal_fact",
                "summary": "月球发动机危机是所有返航任务的背景风险",
            },
            {
                "index": "M3",
                "scene_id": "scene_0002",
                "memory_temporal_scope": "temporal_episode",
                "summary": "无关的普通时间片记忆",
            },
        ],
        "character_reference_knowledge": [
            {
                "item_id": "ref_liu_550a",
                "subject": "刘培强",
                "statement": "刘培强知道550A能够分析脑电波。",
                "knowledge_scope": "character_private",
                "known_to": ["刘培强"],
            },
            {
                "item_id": "ref_zhang_private",
                "subject": "张鹏",
                "statement": "张鹏知道另一条私密设定。",
                "knowledge_scope": "character_private",
                "known_to": ["张鹏"],
            },
        ],
    }
