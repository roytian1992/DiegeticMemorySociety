from dms.parsing import (
    extract_json_value,
    validate_durable_relationships,
    validate_episodic_memories,
    validate_kg_entity_mentions,
    validate_scene_event_candidates,
    validate_scene_inventory,
    validate_visibility_notes,
)
from dms.entity_alignment import sanitize_kg_entity_output


def test_extract_json_value_from_fenced_block() -> None:
    result = extract_json_value('prefix\n```json\n{"scene_id": "scene_0001"}\n```\nsuffix')

    assert result.ok is True
    assert result.data == {"scene_id": "scene_0001"}


def test_extract_json_value_repairs_model_json_when_possible() -> None:
    result = extract_json_value(
        """
        {
          "unit_id": "scene_0005",
          "entity_mentions": [
            {
              "surface": "VR 头显",
              "entity_type": "object",
              "canonical_hint": "",
              "role_in_unit": "device",
              "attributes_or_state": "used for battlefield awareness",
              "evidence": "刘培强打开VR 头显"
              }
            },
            {
              "surface": "战场感知系统",
              "entity_type": "concept",
              "canonical_hint": "",
              "role_in_unit": "capability",
              "attributes_or_state": "online",
              "evidence": "战场感知系统上线"
            }
          ],
          "scene_tags": [],
          "unresolved_mentions": []
        }
        """
    )

    assert result.ok is True
    assert [item["surface"] for item in result.data["entity_mentions"]] == ["VR 头显", "战场感知系统"]


def test_validate_scene_inventory_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "setting": {"location": "", "time_hint": "", "spatial_context": ""},
        "stated_facts": [],
        "open_questions": [],
    }

    assert validate_scene_inventory(data, expected_scene_id="scene_0001") == []


def test_validate_scene_inventory_reports_scene_mismatch() -> None:
    data = {
        "unit_id": "scene_9999",
        "setting": {"location": "", "time_hint": "", "spatial_context": ""},
        "stated_facts": [],
        "open_questions": [],
    }

    errors = validate_scene_inventory(data, expected_scene_id="scene_0001")
    assert "unit_id mismatch" in errors[0]


def test_validate_scene_inventory_rejects_entity_fields() -> None:
    data = {
        "unit_id": "scene_0001",
        "setting": {"location": "", "time_hint": "", "spatial_context": ""},
        "characters": [],
        "objects": [],
        "stated_facts": [],
        "open_questions": [],
    }

    errors = validate_scene_inventory(data, expected_scene_id="scene_0001")

    assert "characters must not be emitted by scene_inventory; use kg_entity_mentions" in errors
    assert "objects must not be emitted by scene_inventory; use kg_entity_mentions" in errors


def test_validate_kg_entity_mentions_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "刘培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": "预备航天员候选人",
                "role_in_unit": "speaker",
                "attributes_or_state": "",
                "evidence": "刘培强：",
            }
        ],
        "unresolved_mentions": [],
    }

    assert validate_kg_entity_mentions(data, expected_scene_id="scene_0001") == []


def test_validate_kg_entity_mentions_accepts_optional_description() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "刘培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": "通过预备航天员考核的人",
                "role_in_unit": "speaker",
                "attributes_or_state": "已通过预备航天员考核",
                "evidence": "刘培强通过预备航天员考核",
            }
        ],
        "unresolved_mentions": [],
    }

    assert validate_kg_entity_mentions(data, expected_scene_id="scene_0001") == []


def test_validate_kg_entity_mentions_rejects_non_string_description() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "刘培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": ["bad"],
                "role_in_unit": "speaker",
                "attributes_or_state": "",
                "evidence": "刘培强",
            }
        ],
        "unresolved_mentions": [],
    }

    errors = validate_kg_entity_mentions(data, expected_scene_id="scene_0001")

    assert "entity_mentions[1].description must be a string" in errors


def test_validate_kg_entity_mentions_accepts_scene_tags() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [],
        "scene_tags": [
            {
                "surface": "一场山火",
                "tag_type": "illustrative_example",
                "reason": "rhetorical example",
                "evidence": "一场山火",
            }
        ],
        "unresolved_mentions": [],
    }

    assert validate_kg_entity_mentions(data, expected_scene_id="scene_0001") == []


def test_sanitize_kg_entity_output_demotes_scene_tag_entity_types_and_unresolved_markers() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "杂草",
                "entity_type": "background_element",
                "canonical_hint": "",
                "role_in_unit": "background_element",
                "attributes_or_state": "半人高",
                "evidence": "杂草已有半人高",
            },
            {
                "surface": "师父",
                "entity_type": "unresolved",
                "canonical_hint": "",
                "role_in_unit": "unresolved",
                "attributes_or_state": "",
                "evidence": "师父，真的有太阳危机吗？",
            },
        ],
        "unresolved_mentions": [],
    }

    sanitized = sanitize_kg_entity_output(data)

    assert sanitized["entity_mentions"] == []
    assert sanitized["scene_tags"][0]["surface"] == "杂草"
    assert sanitized["scene_tags"][0]["tag_type"] == "background_element"
    assert sanitized["unresolved_mentions"][0]["surface"] == "师父"
    assert validate_kg_entity_mentions(sanitized, expected_scene_id="scene_0001") == []


def test_validate_kg_entity_mentions_reports_required_lists() -> None:
    data = {"unit_id": "scene_0001", "entity_mentions": None, "unresolved_mentions": []}

    errors = validate_kg_entity_mentions(data, expected_scene_id="scene_0001")

    assert "entity_mentions must be a list" in errors


def test_validate_kg_entity_mentions_accepts_legacy_type_aliases() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "数字生命",
                "entity_type": "world_rule_or_concept",
                "canonical_hint": "数字生命",
                "role_in_unit": "concept",
                "attributes_or_state": "",
                "evidence": "数字生命",
            }
        ],
        "unresolved_mentions": [],
    }

    assert validate_kg_entity_mentions(data, expected_scene_id="scene_0001") == []


def test_validate_kg_entity_mentions_rejects_unknown_entity_type() -> None:
    data = {
        "unit_id": "scene_0001",
        "entity_mentions": [
            {
                "surface": "数字生命",
                "entity_type": "theme",
                "canonical_hint": "数字生命",
                "role_in_unit": "concept",
                "attributes_or_state": "",
                "evidence": "数字生命",
            }
        ],
        "unresolved_mentions": [],
    }

    errors = validate_kg_entity_mentions(data, expected_scene_id="scene_0001")
    assert any("entity_mentions[1].entity_type must be one of" in error for error in errors)


def test_validate_scene_event_candidates_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "thread_candidates": [],
    }

    assert validate_scene_event_candidates(data, expected_scene_id="scene_0001") == []


def test_validate_scene_event_candidates_reports_required_lists() -> None:
    data = {
        "unit_id": "scene_0001",
        "events": None,
        "knowledge_transfers": [],
        "state_changes": [],
        "thread_candidates": [],
    }

    errors = validate_scene_event_candidates(data, expected_scene_id="scene_0001")
    assert "events must be a list" in errors


def test_validate_visibility_notes_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "visibility_records": [],
        "hidden_or_future_sensitive_items": [],
    }

    assert validate_visibility_notes(data, expected_scene_id="scene_0001") == []


def test_validate_visibility_notes_reports_required_lists() -> None:
    data = {
        "unit_id": "scene_0001",
        "visibility_records": [],
        "hidden_or_future_sensitive_items": None,
    }

    errors = validate_visibility_notes(data, expected_scene_id="scene_0001")
    assert "hidden_or_future_sensitive_items must be a list" in errors


def test_validate_episodic_memories_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "episodic_memories": [
            {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "action",
                "summary": "刘培强进入房间",
                "evidence": "刘培强进入房间",
                "entity_links": [
                    {
                        "entity": "刘培强",
                        "entity_type": "character",
                        "link_role": "actor",
                        "evidence": "刘培强",
                    }
                ],
            }
        ],
    }

    assert validate_episodic_memories(data, expected_scene_id="scene_0001") == []


def test_validate_episodic_memories_accepts_concept_and_occasion_links() -> None:
    data = {
        "unit_id": "scene_0001",
        "episodic_memories": [
            {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "setting",
                "summary": "数字生命实验开始",
                "evidence": "数字生命实验",
                "entity_links": [
                    {
                        "entity": "数字生命",
                        "entity_type": "concept",
                        "link_role": "concept",
                        "evidence": "数字生命",
                    },
                    {
                        "entity": "数字生命实验",
                        "entity_type": "occasion",
                        "link_role": "other",
                        "evidence": "数字生命实验",
                    },
                ],
            }
        ],
    }

    assert validate_episodic_memories(data, expected_scene_id="scene_0001") == []


def test_validate_episodic_memories_reports_bad_sequence_index() -> None:
    data = {
        "unit_id": "scene_0001",
        "episodic_memories": [
            {
                "memory_id_hint": "m1",
                "sequence_index": "1",
                "timeline_label": "scene_0001",
                "memory_type": "action",
                "summary": "刘培强进入房间",
                "evidence": "刘培强进入房间",
                "entity_links": [],
            }
        ],
    }

    errors = validate_episodic_memories(data, expected_scene_id="scene_0001")
    assert "episodic_memories[1].sequence_index must be an integer" in errors


def test_validate_episodic_memories_checks_exact_source_evidence() -> None:
    data = {
        "unit_id": "scene_0001",
        "episodic_memories": [
            {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "action",
                "summary": "刘培强进入房间",
                "evidence": "刘培强进入房间",
                "entity_links": [
                    {
                        "entity": "刘培强",
                        "entity_type": "character",
                        "link_role": "actor",
                        "evidence": "刘培强",
                    }
                ],
            },
            {
                "memory_id_hint": "m2",
                "sequence_index": 2,
                "timeline_label": "scene_0001",
                "memory_type": "action",
                "summary": "模型改写证据",
                "evidence": "刘培强走进屋子",
                "entity_links": [],
            },
        ],
    }
    source_unit = {"title": "1、INT.日.房间", "subtitle": "", "content": "刘培强进入房间。"}

    errors = validate_episodic_memories(data, expected_scene_id="scene_0001", source_unit=source_unit)

    assert "episodic_memories[2].evidence must align to a contiguous span from title, subtitle, or content" in errors
    assert not any("episodic_memories[1].evidence" in error for error in errors)


def test_validate_episodic_memories_rejects_relationship_observations() -> None:
    data = {
        "unit_id": "scene_0001",
        "episodic_memories": [],
        "relationship_observations": [],
    }

    errors = validate_episodic_memories(data, expected_scene_id="scene_0001")

    assert "relationship_observations must not be emitted by episodic_memories; use durable_relationships" in errors


def test_validate_durable_relationships_accepts_minimal_schema() -> None:
    data = {
        "unit_id": "scene_0001",
        "relationship_observations": [
            {
                "source_entity": "甲",
                "target_entity": "乙",
                "relation_type": "alliance",
                "status_or_change": "甲和乙是盟友",
                "evidence": "甲和乙是盟友",
            }
        ],
    }

    assert validate_durable_relationships(data, expected_scene_id="scene_0001") == []


def test_validate_durable_relationships_checks_exact_source_evidence() -> None:
    data = {
        "unit_id": "scene_0001",
        "relationship_observations": [
            {
                "source_entity": "甲",
                "target_entity": "乙",
                "relation_type": "alliance",
                "status_or_change": "甲和乙是盟友",
                "evidence": "甲和乙成为盟友",
            }
        ],
    }
    source_unit = {"title": "", "subtitle": "", "content": "甲和乙是盟友。"}

    errors = validate_durable_relationships(data, expected_scene_id="scene_0001", source_unit=source_unit)

    assert "relationship_observations[1].evidence must align to a contiguous span from title, subtitle, or content" in errors


def test_validators_accept_legacy_scene_id_schema() -> None:
    data = {
        "scene_id": "scene_0001",
        "setting": {"location": "", "time_of_day": "", "interior_exterior": ""},
        "stated_facts": [],
        "open_questions": [],
    }

    assert validate_scene_inventory(data, expected_scene_id="scene_0001") == []
