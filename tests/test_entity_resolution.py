import json
from pathlib import Path

from dms.memory import build_entity_resolution_artifacts, resolve_name_variants


def test_resolve_name_variants_handles_chinese_given_name_alias() -> None:
    variants = resolve_name_variants("刘培强")

    assert "刘培强" in variants
    assert "培强" in variants


def test_resolve_name_variants_does_not_drop_prefix_from_role_name() -> None:
    variants = resolve_name_variants("印度科学家")

    assert "印度科学家" in variants
    assert "度科学家" not in variants


def test_resolve_name_variants_does_not_drop_prefix_from_object_name() -> None:
    variants = resolve_name_variants("电脑屏幕")

    assert "电脑屏幕" in variants
    assert "脑屏幕" not in variants


def test_resolve_name_variants_does_not_drop_prefix_from_concept_or_occasion_name() -> None:
    concept_variants = resolve_name_variants("脑电波")
    occasion_variants = resolve_name_variants("视频录制")
    civilization_variants = resolve_name_variants("人类文明")
    digital_self_variants = resolve_name_variants("数字的你")

    assert "脑电波" in concept_variants
    assert "电波" not in concept_variants
    assert "视频录制" in occasion_variants
    assert "频录制" not in occasion_variants
    assert "人类文明" in civilization_variants
    assert "类文明" not in civilization_variants
    assert "数字的你" in digital_self_variants
    assert "字的你" not in digital_self_variants


def test_resolve_name_variants_handles_latin_first_last_and_titles() -> None:
    variants = resolve_name_variants("Dr. Liu Peiqiang")

    assert "Liu Peiqiang" in variants
    assert "Peiqiang Liu" in variants
    assert "Liu" in variants
    assert "Peiqiang" in variants


def test_build_entity_resolution_artifacts(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [
            {
                "entity_id": "character_0001",
                "canonical_name": "刘培强",
                "aliases": ["刘培强"],
                "scene_ids": ["scene_0001"],
                "mentions": [{"scene_id": "scene_0001", "name": "刘培强", "evidence": "刘培强进入房间"}],
            }
        ],
        "objects": [],
        "events": [
            {
                "scene_id": "scene_0002",
                "record_id": "event_001",
                "participants": ["培强", "韩朵朵"],
                "summary": "培强与韩朵朵争执",
                "evidence": "培强：我必须去。韩朵朵：不行。",
            }
        ],
        "knowledge_transfers": [
            {
                "scene_id": "scene_0002",
                "record_id": "kt_001",
                "source": "培强",
                "receiver": "韩朵朵",
                "content": "他必须离开",
                "epistemic_status": "believes",
                "evidence": "我必须去。",
            }
        ],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    aliases = [json.loads(line) for line in (output_dir / "aliases.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["entity_count"] >= 2
    assert any(entity["canonical_name"] == "刘培强" for entity in entities)
    assert any(alias["alias"] == "培强" for alias in aliases)
    assert summary["relationship_count"] == 0


def test_entity_resolution_merges_character_and_group_role_mentions(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [{"canonical_name": "科研人员", "aliases": ["科研人员"], "scene_ids": ["scene_0001"], "mentions": []}],
        "objects": [],
        "events": [{"scene_id": "scene_0001", "record_id": "event_001", "participants": ["科研人员"], "summary": "", "evidence": ""}],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [{"scene_id": "scene_0001", "character": "科研人员", "evidence": ""}],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]

    matching = [entity for entity in entities if entity["canonical_name"] == "科研人员"]
    assert len(matching) == 1


def test_entity_resolution_uses_kg_entity_mentions_and_canonical_hint(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0001",
                "surface": "刘培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": "通过预备航天员考核的人",
                "evidence": "刘培强进入房间",
            },
            {
                "scene_id": "scene_0002",
                "surface": "培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": "将前往月球受训的人",
                "evidence": "培强抬头",
            },
            {
                "scene_id": "scene_0002",
                "surface": "地下城",
                "entity_type": "location",
                "canonical_hint": "",
                "evidence": "地下城广播响起",
            },
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    traces = [
        json.loads(line) for line in (output_dir / "resolution_traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    liu = [entity for entity in entities if entity["canonical_name"] == "刘培强"]
    assert len(liu) == 1
    assert "培强" in liu[0]["aliases"]
    assert liu[0]["initial_description"] == "通过预备航天员考核的人"
    assert "将前往月球受训的人" in liu[0]["descriptions"]
    assert any(entity["entity_type"] == "location" and entity["canonical_name"] == "地下城" for entity in entities)
    assert summary["resolution_trace_count"] == 3
    assert any(trace["canonical_hint"] == "刘培强" and trace["mention"] == "培强" for trace in traces)


def test_entity_resolution_keeps_author_description_as_baseline(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0002",
                "surface": "培强",
                "entity_type": "character",
                "canonical_hint": "刘培强",
                "description": "将前往月球受训的人",
                "evidence": "培强抬头",
            }
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    build_entity_resolution_artifacts(
        world_model_path,
        output_dir,
        author_entities=[
            {
                "canonical_name": "刘培强",
                "entity_type": "character",
                "aliases": ["培强"],
                "author_description": "作者设定里的年轻预备航天员",
            }
        ],
    )
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]

    liu = [entity for entity in entities if entity["canonical_name"] == "刘培强"]
    assert len(liu) == 1
    assert liu[0]["author_description"] == "作者设定里的年轻预备航天员"
    assert liu[0]["initial_description"] == "作者设定里的年轻预备航天员"
    assert "将前往月球受训的人" in liu[0]["descriptions"]
    assert liu[0]["description_sources"][0]["source"] == "author_defined"


def test_entity_resolution_merges_role_modifier_and_alnum_code_aliases(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0001",
                "surface": "为首的印度科学家",
                "entity_type": "character",
                "canonical_hint": "",
                "evidence": "为首的印度科学家站在镜头前",
            },
            {
                "scene_id": "scene_0001",
                "surface": "印度科学家",
                "entity_type": "character",
                "canonical_hint": "",
                "evidence": "印度科学家：人，本质上就是一堆电信号",
            },
            {
                "scene_id": "scene_0001",
                "surface": "量子计算机550A",
                "entity_type": "object",
                "canonical_hint": "550A",
                "evidence": "量子计算机550A 正在分析",
            },
            {
                "scene_id": "scene_0001",
                "surface": "550A",
                "entity_type": "object",
                "canonical_hint": "",
                "evidence": "550A 正在分析",
            },
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]

    scientists = [entity for entity in entities if "印度科学家" in entity["aliases"]]
    computers = [entity for entity in entities if "550A" in entity["aliases"]]
    assert len(scientists) == 1
    assert "为首的印度科学家" in scientists[0]["aliases"]
    assert len(computers) == 1
    assert "量子计算机550A" in computers[0]["aliases"]


def test_entity_resolution_does_not_create_relationships_from_event_copresence(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {"scene_id": "scene_0001", "surface": "甲", "entity_type": "character", "canonical_hint": "", "evidence": "甲和乙进入"},
            {"scene_id": "scene_0001", "surface": "乙", "entity_type": "character", "canonical_hint": "", "evidence": "甲和乙进入"},
        ],
        "events": [
            {
                "scene_id": "scene_0001",
                "record_id": "event_001",
                "participants": ["甲", "乙"],
                "summary": "甲和乙进入房间",
                "evidence": "甲和乙进入房间",
            }
        ],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)

    assert summary["relationship_count"] == 0


def test_entity_resolution_does_not_create_relationships_from_state_changes(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0001",
                "surface": "受试者",
                "entity_type": "character",
                "canonical_hint": "",
                "evidence": "受试者状态改变",
            }
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [
            {
                "scene_id": "scene_0001",
                "record_id": "state_001",
                "entity": "受试者",
                "before": "连接设备",
                "after": "被分析",
                "evidence": "受试者状态改变",
            }
        ],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)

    assert summary["relationship_count"] == 0


def test_entity_resolution_does_not_create_relationships_from_knowledge_transfer(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {"scene_id": "scene_0001", "surface": "甲", "entity_type": "character", "canonical_hint": "", "evidence": "甲告诉乙"},
            {"scene_id": "scene_0001", "surface": "乙", "entity_type": "character", "canonical_hint": "", "evidence": "甲告诉乙"},
        ],
        "events": [],
        "knowledge_transfers": [
            {
                "scene_id": "scene_0001",
                "record_id": "kt_001",
                "source": "甲",
                "receiver": "乙",
                "content": "秘密",
                "epistemic_status": "knows",
                "evidence": "甲告诉乙秘密",
            }
        ],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)

    assert summary["relationship_count"] == 0


def test_entity_resolution_skips_composite_surface_not_in_evidence(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0001",
                "surface": "镜头",
                "entity_type": "object",
                "canonical_hint": "",
                "evidence": "对着镜头展开介绍",
            }
        ],
        "events": [],
        "knowledge_transfers": [
            {
                "scene_id": "scene_0001",
                "record_id": "kt_001",
                "source": "印度科学家",
                "receiver": "观众/镜头",
                "content": "介绍",
                "epistemic_status": "knows",
                "evidence": "印度科学家对着镜头展开介绍",
            }
        ],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = (output_dir / "entities.jsonl").read_text(encoding="utf-8")

    assert "观众/镜头" not in entities
    assert '"canonical_name": "镜头"' in entities


def test_entity_resolution_keeps_only_durable_relationship_observations(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {"scene_id": "scene_0001", "surface": "甲", "entity_type": "character", "canonical_hint": "", "evidence": "甲和乙是盟友"},
            {"scene_id": "scene_0001", "surface": "乙", "entity_type": "character", "canonical_hint": "", "evidence": "甲和乙是盟友"},
            {"scene_id": "scene_0001", "surface": "丙", "entity_type": "character", "canonical_hint": "", "evidence": "甲看向丙"},
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "relationship_observations": [
            {
                "scene_id": "scene_0001",
                "record_id": "rel_001",
                "source_entity": "甲",
                "target_entity": "乙",
                "relation_type": "alliance",
                "status_or_change": "甲和乙是盟友",
                "evidence": "甲和乙是盟友",
            },
            {
                "scene_id": "scene_0001",
                "record_id": "rel_002",
                "source_entity": "甲",
                "target_entity": "丙",
                "relation_type": "addresses",
                "status_or_change": "甲对丙说话",
                "evidence": "甲对丙说话",
            },
        ],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    summary = build_entity_resolution_artifacts(world_model_path, output_dir)
    relationships = [
        json.loads(line) for line in (output_dir / "relationships.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["relationship_count"] == 1
    assert relationships[0]["relation_type"] == "alliance"


def test_entity_resolution_normalizes_legacy_concept_and_occasion_types(tmp_path: Path) -> None:
    world_model_path = tmp_path / "prefix_world_model.json"
    output_dir = tmp_path / "entity_resolution"
    world_model = {
        "characters": [],
        "objects": [],
        "kg_entity_mentions": [
            {
                "scene_id": "scene_0001",
                "surface": "数字生命",
                "entity_type": "world_rule_or_concept",
                "canonical_hint": "",
                "evidence": "数字生命",
            },
            {
                "scene_id": "scene_0001",
                "surface": "月球危机",
                "entity_type": "event_or_disaster",
                "canonical_hint": "",
                "evidence": "月球危机",
            },
        ],
        "events": [],
        "knowledge_transfers": [],
        "state_changes": [],
        "visibility_records": [],
    }
    world_model_path.write_text(json.dumps(world_model, ensure_ascii=False), encoding="utf-8")

    build_entity_resolution_artifacts(world_model_path, output_dir)
    entities = [json.loads(line) for line in (output_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()]

    assert any(entity["canonical_name"] == "数字生命" and entity["entity_type"] == "concept" for entity in entities)
    assert any(entity["canonical_name"] == "月球危机" and entity["entity_type"] == "occasion" for entity in entities)
