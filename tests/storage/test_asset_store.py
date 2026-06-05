import sqlite3
from pathlib import Path

from dms.storage import (
    AssetStoreImportConfig,
    get_entity_memories,
    get_one_hop_relationships,
    get_relationship_count,
    get_retrieval_documents,
    get_scene_metadata,
    import_run_assets,
    resolve_entity_refs,
)
from tests.helpers import write_jsonl


def test_asset_store_imports_entities_memories_and_time_queries(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"

    summary = import_run_assets(
        AssetStoreImportConfig(
            db_path=db_path,
            ordered_run_dir=run_root,
            reset=True,
        )
    )

    assert summary["entities"] == 2
    assert summary["episodic_memories"] == 2
    assert summary["entity_memory_links"] == 2
    assert summary["scene_summaries"] == 1
    assert summary["unit_summaries"] == 1
    assert summary["relationships"] == 1
    assert summary["scene_metadata"] == 2
    assert summary["stated_fact_documents"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT entity_id, entity_type, original_entity_type FROM entities WHERE canonical_name = ?",
            ("550A",),
        ).fetchone()
    assert dict(row) == {
        "entity_id": "technology_or_facility_0002",
        "entity_type": "object",
        "original_entity_type": "technology_or_facility",
    }

    assert get_entity_memories(db_path, entity_ref="量子计算机550A", before_scene_id="scene_0001") == []

    memories = get_entity_memories(db_path, entity_ref="量子计算机550A", before_scene_id="scene_0002")
    assert [memory["memory_id"] for memory in memories] == ["scene_0001_memory_001"]
    assert memories[0]["summary"] == "550A分析受试者脑电波"
    assert memories[0]["evidence_text"] == "量子计算机550A 正在对他的脑电波进行分析处理"
    assert memories[0]["memory_temporal_scope"] == "atemporal_fact"

    assert get_entity_memories(db_path, entity_ref="培强", before_scene_id="scene_0003") == []
    liu_memories = get_entity_memories(db_path, entity_ref="培强", before_scene_id="scene_0005")
    assert [memory["memory_id"] for memory in liu_memories] == ["scene_0004_memory_001"]

    docs = get_retrieval_documents(db_path, entity_ref="550A", before_scene_id="scene_0002")
    assert [doc["doc_type"] for doc in docs] == ["episodic_memory_entity"]
    assert "550A" in docs[0]["text"]

    fact_docs = get_retrieval_documents(db_path, doc_type="stated_fact", before_scene_id="scene_0002")
    assert [doc["source_id"] for doc in fact_docs] == ["scene_0001_fact_001"]
    assert "550A正在分析受试者脑电波" in fact_docs[0]["text"]

    matches = resolve_entity_refs(db_path, ["培强", "量子计算机550A", "550"], limit_per_ref=1)
    assert [match["canonical_name"] for match in matches] == ["刘培强", "550A", "550A"]
    assert matches[0]["author_profile"]["stable_traits"] == ["嘴硬", "抗压"]
    assert matches[0]["author_profile"]["behavior_constraints"] == ["不能提前知道未来剧情"]
    assert matches[0]["initial_state"]["beliefs"] == ["地球处境正在恶化"]
    assert matches[0]["profile_policy"]["priority"] == "author_locked"

    relationships = get_one_hop_relationships(
        db_path,
        entity_ids=["character_0001"],
        before_scene_id="scene_0005",
    )
    assert len(relationships) == 1
    assert relationships[0]["source_name"] == "刘培强"
    assert relationships[0]["target_name"] == "550A"
    assert get_relationship_count(db_path) == 1

    scene_metadata = get_scene_metadata(db_path, ["scene_0001", "scene_0004"])
    assert scene_metadata["scene_0001"]["title"] == "1、INT.日.印度 数字生命研究室"
    assert scene_metadata["scene_0001"]["setting"]["location"] == "印度 数字生命研究室"
    assert scene_metadata["scene_0001"]["stated_facts"][0]["proposition"] == "550A正在分析受试者脑电波"
    assert scene_metadata["scene_0004"]["scene_tags"][0]["surface"] == "战区废墟"


def _write_sample_ordered_run(tmp_path: Path) -> Path:
    run_root = tmp_path / "ordered"
    kg_dir = run_root / "knowledge_graph"
    memory_dir = run_root / "memories"
    summary_dir = run_root / "summaries"
    scene_context_dir = run_root / "scene_context"
    debug_dir = run_root / "_debug"
    kg_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    scene_context_dir.mkdir(parents=True)
    debug_dir.mkdir(parents=True)

    write_jsonl(
        kg_dir / "entities.jsonl",
        [
            {
                "entity_id": "technology_or_facility_0002",
                "entity_type": "technology_or_facility",
                "canonical_name": "550A",
                "aliases": ["量子计算机550A"],
                "first_seen_scene": "scene_0001",
                "mention_count": 2,
            },
            {
                "entity_id": "character_0001",
                "entity_type": "character",
                "canonical_name": "刘培强",
                "aliases": ["培强"],
                "first_seen_scene": "scene_0004",
                "mention_count": 1,
                "author_description": "作者设定里的年轻飞行员",
                "initial_description": "作者设定里的年轻飞行员",
                "author_profile": {
                    "stable_traits": ["嘴硬", "抗压"],
                    "speaking_style": ["短句", "压着情绪说"],
                    "behavior_constraints": ["不能提前知道未来剧情"],
                },
                "initial_state": {"beliefs": ["地球处境正在恶化"]},
                "profile_policy": {"priority": "author_locked", "visibility": "author_guidance"},
                "profile_sources": [{"source": "author_context", "path": "author_entities.json"}],
                "author_entity_ids": ["author_character_0001"],
            },
        ],
    )
    write_jsonl(
        kg_dir / "aliases.jsonl",
        [
            {
                "entity_id": "technology_or_facility_0002",
                "alias": "550A",
                "normalized_alias": "550a",
                "source": "canonical",
            },
            {
                "entity_id": "character_0001",
                "alias": "培强",
                "normalized_alias": "培强",
                "source": "name_variant",
            },
        ],
    )
    write_jsonl(
        kg_dir / "relationships.jsonl",
        [
            {
                "relationship_id": "relationship_0001",
                "source_entity_id": "character_0001",
                "target_entity_id": "technology_or_facility_0002",
                "relation_type": "responsible_for",
                "direction": "directed",
                "status": "active",
                "first_seen_scene": "scene_0004",
                "last_updated_scene": "scene_0004",
                "strength": 0.8,
                "evidence": ["刘培强负责550A相关返航任务"],
            }
        ],
    )
    write_jsonl(
        memory_dir / "episodic_memories.jsonl",
        [
            {
                "record_id": "scene_0001_memory_001",
                "parent_unit_id": "scene_0001",
                "chunk_id": "scene_0001",
                "chunk_index": 1,
                "sequence_index": 1,
                "timeline_index": "scene_0001:001",
                "memory_type": "observation",
                "memory_temporal_scope": "atemporal_fact",
                "summary": "550A分析受试者脑电波",
                "evidence_text": "量子计算机550A 正在对他的脑电波进行分析处理",
                "evidence_start": 0,
                "evidence_end": 24,
                "parent_evidence_start": 0,
                "parent_evidence_end": 24,
                "parent_source_sha256": "sha1",
            },
            {
                "record_id": "scene_0004_memory_001",
                "parent_unit_id": "scene_0004",
                "chunk_id": "scene_0004",
                "chunk_index": 1,
                "sequence_index": 2,
                "timeline_index": "scene_0004:002",
                "memory_type": "action",
                "memory_temporal_scope": "temporal_episode",
                "summary": "刘培强返航途中情绪焦躁",
                "evidence_text": "刘培强一直沉默地望向窗外",
                "parent_source_sha256": "sha4",
            },
        ],
    )
    write_jsonl(
        memory_dir / "entity_memory_links.jsonl",
        [
            {
                "record_id": "scene_0001_memory_001_entity_001",
                "memory_record_id": "scene_0001_memory_001",
                "parent_unit_id": "scene_0001",
                "chunk_index": 1,
                "entity": "量子计算机550A",
                "canonical_entity": "550A",
                "entity_type": "technology_or_facility",
                "link_role": "actor",
                "evidence_text": "量子计算机550A 正在对他的脑电波进行分析处理",
            },
            {
                "record_id": "scene_0004_memory_001_entity_001",
                "memory_record_id": "scene_0004_memory_001",
                "parent_unit_id": "scene_0004",
                "chunk_index": 1,
                "entity": "培强",
                "canonical_entity": "刘培强",
                "entity_type": "character",
                "link_role": "experiencer",
                "evidence_text": "刘培强一直沉默地望向窗外",
            },
        ],
    )
    write_jsonl(
        summary_dir / "scene_summaries.jsonl",
        [
            {
                "record_id": "scene_0001_summary",
                "parent_unit_id": "scene_0001",
                "summary": "550A正在分析受试者脑电波。",
                "retrieval_text": "550A, 脑电波, 数字生命",
            }
        ],
    )
    write_jsonl(
        summary_dir / "unit_summaries.jsonl",
        [
            {
                "record_id": "scene_0001_summary",
                "unit_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "chunk_index": 1,
                "summary": "550A正在分析受试者脑电波。",
                "retrieval_text": "550A, 脑电波, 数字生命",
            }
        ],
    )
    write_jsonl(
        debug_dir / "chunk_manifest.jsonl",
        [
            {
                "scene_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "unit_id": "scene_0001",
                "chunk_id": "scene_0001",
                "chunk_index": 1,
                "chunk_count": 1,
                "source_record_id": 1,
                "discourse_index": 1,
                "title": "1、INT.日.印度 数字生命研究室",
                "source_sha256": "sha1",
                "character_count": 194,
            },
            {
                "scene_id": "scene_0004",
                "parent_unit_id": "scene_0004",
                "unit_id": "scene_0004",
                "chunk_id": "scene_0004",
                "chunk_index": 1,
                "chunk_count": 1,
                "source_record_id": 4,
                "discourse_index": 4,
                "title": "4、EXT.日.利伯维尔 战区废墟",
                "source_sha256": "sha4",
                "character_count": 567,
            },
        ],
    )
    write_jsonl(
        scene_context_dir / "scenes.jsonl",
        [
            {
                "scene_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "setting": {
                    "location": "印度 数字生命研究室",
                    "time_hint": "日",
                    "spatial_context": "INT",
                },
            },
            {
                "scene_id": "scene_0004",
                "parent_unit_id": "scene_0004",
                "setting": {
                    "location": "利伯维尔 战区废墟",
                    "time_hint": "日",
                    "spatial_context": "EXT",
                },
            },
        ],
    )
    write_jsonl(
        scene_context_dir / "stated_facts.jsonl",
        [
            {
                "scene_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "record_id": "scene_0001_fact_001",
                "proposition": "550A正在分析受试者脑电波",
                "speaker_or_source": "",
                "evidence": "量子计算机550A 正在对他的脑电波进行分析处理",
            }
        ],
    )
    write_jsonl(
        scene_context_dir / "open_questions.jsonl",
        [
            {
                "scene_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "record_id": "scene_0001_question_001",
                "question": "受试者的身份是什么？",
                "evidence": "受试者",
            }
        ],
    )
    write_jsonl(
        scene_context_dir / "scene_tags.jsonl",
        [
            {
                "scene_id": "scene_0004",
                "parent_unit_id": "scene_0004",
                "record_id": "scene_0004_scene_tag_001",
                "surface": "战区废墟",
                "tag_type": "setting_tag",
                "reason": "scene environment",
                "evidence": "战区废墟",
            }
        ],
    )
    return run_root
