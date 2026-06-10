import json
from pathlib import Path

import pytest

from dms.retrieval import (
    MemoryPacketConfig,
    build_memory_packet,
    decompose_writing_intent,
    format_memory_packet_markdown,
)
from dms.llm import FakeReferenceKGClient
from dms.reference_library import (
    ReferenceKGExtractionConfig,
    ReferenceKnowledgeImportConfig,
    ReferenceLibraryIngestConfig,
    extract_reference_kg,
    import_reference_knowledge,
    ingest_reference_library,
)
from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
from tests.helpers import write_jsonl
from tests.storage.test_asset_store import _write_sample_ordered_run


pytest.importorskip("chromadb")


def test_decompose_writing_intent_uses_existing_entity_aliases(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))

    decomposition = decompose_writing_intent("刘培强检查550A脑电波分析结果", db_path=db_path)

    assert "刘培强" in decomposition["important_entities"]
    assert "550A" in decomposition["important_entities"]
    assert decomposition["narrative_units"]
    assert all("Key entities:" not in unit for unit in decomposition["narrative_units"])
    assert all("550A 刘培强" not in unit for unit in decomposition["narrative_units"])


def test_memory_packet_dedupes_entity_memories_and_includes_one_hop_relations(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"
    chroma_dir = tmp_path / "chroma"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))
    build_chroma_memory_index(ChromaMemoryIndexConfig(db_path=db_path, persist_dir=chroma_dir, reset=True, embedding_dim=64))

    packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="刘培强和550A在返航任务中涉及脑电波分析",
            before_scene_id="scene_0005",
            unit_type="chapter",
            unit_label="chapter",
            scene_top_k=2,
            entity_memory_top_k=3,
            embedding_dim=64,
        )
    )

    assert "writing_intent" not in packet
    assert packet["retrieval_boundary"]["before_unit_id"] == "scene_0005"
    assert packet["retrieval_boundary"]["unit_type"] == "chapter"
    assert packet["retrieval_boundary"]["unit_label"] == "chapter"
    assert "query_decomposition" not in packet
    assert packet["trace"]["query_decomposition"]["important_entities"]
    assert packet["trace"]["query_decomposition"]["narrative_units"]
    assert all("Key entities:" not in unit for unit in packet["trace"]["query_decomposition"]["narrative_units"])
    assert packet["trace"]["scene_summary_retrieval"]["sources"] == ["scene_summary", "stated_fact"]
    entity_names = {entity["canonical_name"] for entity in packet["entities"]}
    assert {"刘培强", "550A"}.issubset(entity_names)
    memory_ids = [memory["memory_id"] for memory in packet["episodic_memories"]]
    assert memory_ids == list(dict.fromkeys(memory_ids))
    assert {memory["memory_id"] for memory in packet["episodic_memories"]} == {
        "scene_0001_memory_001",
        "scene_0004_memory_001",
    }
    assert {memory["index"] for memory in packet["episodic_memories"]} == {"M1", "M2"}
    assert all("evidence_text" not in memory for memory in packet["episodic_memories"])
    assert all(memory.get("source_ref") for memory in packet["episodic_memories"])
    assert all(str(memory["source_ref"]).startswith("R") for memory in packet["episodic_memories"])
    assert packet["references"]
    assert all(str(reference["ref_id"]).startswith("R") for reference in packet["references"])
    assert any(reference["kind"] == "episodic_memory_evidence" for reference in packet["references"])
    assert any("量子计算机550A" in reference["text"] for reference in packet["references"])
    liu = next(entity for entity in packet["entities"] if entity["canonical_name"] == "刘培强")
    assert "description" not in liu
    assert "profile" in liu
    assert "current_state" in liu
    assert liu["author_profile"]["stable_traits"] == ["嘴硬", "抗压"]
    assert liu["author_profile"]["speaking_style"] == ["短句", "压着情绪说"]
    assert liu["initial_state"]["beliefs"] == ["地球处境正在恶化"]
    assert liu["profile_policy"]["priority"] == "author_locked"
    assert "嘴硬" in liu["author_profile_summary"]
    assert "作者设定里的年轻飞行员" in liu["profile"]
    assert "刘培强返航途中情绪焦躁" in liu["profile"] or "刘培强返航途中情绪焦躁" in liu["current_state"]
    memory_by_index = {memory["index"]: memory for memory in packet["episodic_memories"]}
    assert {memory["memory_temporal_scope"] for memory in packet["episodic_memories"]} == {
        "atemporal_fact",
        "temporal_episode",
    }
    liu_memory_ids = {memory_by_index[index]["memory_id"] for index in liu["related_memory_index"]}
    assert liu_memory_ids == {"scene_0004_memory_001"}
    assert packet["relations"][0]["relation_type"] == "responsible_for"
    assert packet["relations"][0]["source_refs"]
    assert any(reference["kind"] == "relationship_evidence" for reference in packet["references"])
    assert packet["related_scene_summaries"][0]["scene_id"] == "scene_0001"
    assert "stated_fact" in packet["related_scene_summaries"][0]["retrieval_sources"]
    assert packet["related_scene_summaries"][0]["metadata"]["title"] == "1、INT.日.印度 数字生命研究室"
    assert packet["related_scene_summaries"][0]["metadata"]["setting"]["location"] == "印度 数字生命研究室"
    assert packet["trace"]["memory_temporal_scope_policy"]["reveal_time_filtering_changed"] is False
    markdown = format_memory_packet_markdown(packet)
    assert "author profile baseline" in markdown
    assert "author initial state" in markdown
    assert "current state before chapter scene_0005" in markdown


def test_memory_packet_retrieves_global_atemporal_memories_without_entity_link(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    memories_path = run_root / "memories" / "episodic_memories.jsonl"
    records = [
        json.loads(line)
        for line in memories_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.append(
        {
            "record_id": "scene_0004_memory_002",
            "parent_unit_id": "scene_0004",
            "chunk_id": "scene_0004",
            "chunk_index": 1,
            "sequence_index": 3,
            "timeline_index": "scene_0004:003",
            "memory_type": "observation",
            "memory_temporal_scope": "atemporal_fact",
            "summary": "月球发动机危机是所有返航任务的背景风险",
            "evidence_text": "月球发动机危机正在扩大",
            "parent_source_sha256": "sha4",
        }
    )
    write_jsonl(memories_path, records)

    db_path = tmp_path / "assets.sqlite"
    chroma_dir = tmp_path / "chroma"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))
    build_chroma_memory_index(ChromaMemoryIndexConfig(db_path=db_path, persist_dir=chroma_dir, reset=True, embedding_dim=64))

    packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="写刘培强返航时面对月球发动机危机的背景风险",
            before_scene_id="scene_0005",
            scene_top_k=2,
            entity_memory_top_k=2,
            global_scope_memory_top_k=3,
            embedding_dim=64,
        )
    )

    assert "scene_0004_memory_002" in {memory["memory_id"] for memory in packet["episodic_memories"]}
    assert packet["trace"]["global_scope_memory_retrieval"]["returned_count"] >= 1
    assert packet["trace"]["global_scope_memory_retrieval"]["allowed_scopes"] == ["atemporal_fact", "durable_state"]

    earlier_packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="写刘培强返航时面对月球发动机危机的背景风险",
            before_scene_id="scene_0004",
            scene_top_k=2,
            entity_memory_top_k=2,
            global_scope_memory_top_k=3,
            embedding_dim=64,
        )
    )

    assert "scene_0004_memory_002" not in {memory["memory_id"] for memory in earlier_packet["episodic_memories"]}


def test_memory_packet_optionally_includes_external_reference_context(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"
    chroma_dir = tmp_path / "chroma"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))
    build_chroma_memory_index(ChromaMemoryIndexConfig(db_path=db_path, persist_dir=chroma_dir, reset=True, embedding_dim=64))

    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text(
        "# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强保持稳定。\n",
        encoding="utf-8",
    )
    reference_library_dir = tmp_path / "reference_library"
    reference_kg_dir = tmp_path / "reference_kg"
    reference_db = tmp_path / "reference.sqlite"
    ingest_reference_library(ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=reference_library_dir))
    extract_reference_kg(
        ReferenceKGExtractionConfig(library_dir=reference_library_dir, output_dir=reference_kg_dir, dry_run=False),
        llm_client=FakeReferenceKGClient(),
    )
    import_reference_knowledge(
        ReferenceKnowledgeImportConfig(
            library_dir=reference_library_dir,
            kg_dir=reference_kg_dir,
            db_path=reference_db,
            reset=True,
        )
    )

    packet_without_refs = build_memory_packet(
        MemoryPacketConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="刘培强和550A在返航任务中涉及脑电波分析",
            before_scene_id="scene_0005",
            embedding_dim=64,
        )
    )
    assert packet_without_refs["author_reference_context"] == []
    assert packet_without_refs["trace"]["reference_context_retrieval"]["enabled"] is False
    assert "Author Reference Context" not in format_memory_packet_markdown(packet_without_refs)

    packet = build_memory_packet(
        MemoryPacketConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="刘培强和550A在返航任务中涉及脑电波分析",
            before_scene_id="scene_0005",
            embedding_dim=64,
            include_reference_context=True,
            reference_db_path=reference_db,
            reference_top_k=8,
            reference_author_top_k=8,
            reference_character_top_k=8,
            reference_style_top_k=8,
            reference_timeline_top_k=8,
        )
    )

    assert packet["trace"]["reference_context_retrieval"]["enabled"] is True
    assert packet["trace"]["reference_context_retrieval"]["asset_model"] == "source_local_external_reference_v1"
    assert {item["subject"] for item in packet["author_reference_context"]} >= {"刘培强", "550A", "张鹏"}
    assert {item["subject"] for item in packet["character_reference_knowledge"]} >= {"刘培强"}
    markdown = format_memory_packet_markdown(packet)
    assert "## Author Reference Context" in markdown
    assert "## Character Reference Knowledge" in markdown
    assert "刘培强 appears" in markdown


def test_memory_packet_markdown_wraps_reference_text() -> None:
    packet = {
        "entities": [],
        "relations": [],
        "relationship_diagnostics": {},
        "episodic_memories": [],
        "related_scene_summaries": [],
        "references": [
            {
                "ref_id": "R1",
                "kind": "episodic_memory_evidence",
                "label": "long_memory",
                "scene_id": "scene_0001",
                "text": (
                    "这是一段很长很长的引用文本，用来确认References区块会自动换行，"
                    "不会把所有证据都挤在同一行里影响阅读。"
                    "这里继续补充一段同样很长的中文证据文本，确保没有空格的连续中文也会被折行显示。"
                ),
            }
        ],
    }

    markdown = format_memory_packet_markdown(packet)

    assert "episodic_memory_evidence" not in markdown
    assert "long_memory" not in markdown
    assert "[R1] <scene_0001>" in markdown
    reference_lines = markdown.split("[R1] <scene_0001>")[-1].splitlines()
    indented_text_lines = [line for line in reference_lines if line.startswith("    ")]
    assert len(indented_text_lines) > 1


def test_memory_packet_markdown_omits_internal_reference_ids_from_prompt() -> None:
    packet = {
        "entities": [],
        "relations": [],
        "relationship_diagnostics": {},
        "episodic_memories": [],
        "related_scene_summaries": [],
        "references": [
            {
                "ref_id": "R1",
                "kind": "episodic_memory_evidence",
                "source_id": "scene_0004_chunk_001_memory_006",
                "label": "scene_0004_chunk_001_memory_006",
                "scene_id": "scene_0004",
                "text": "经过一辆被炸翻在路中间的巴士",
            }
        ],
    }

    markdown = format_memory_packet_markdown(packet)

    assert "[R1] <scene_0004>" in markdown
    assert "经过一辆被炸翻在路中间的巴士" in markdown
    assert "episodic_memory_evidence" not in markdown
    assert "scene_0004_chunk_001_memory_006" not in markdown


def test_memory_packet_markdown_uses_entity_profile_fields() -> None:
    packet = {
        "retrieval_boundary": {"before_scene_id": "scene_0006"},
        "entities": [
            {
                "canonical_name": "刘培强",
                "entity_type": "character",
                "profile": "年轻飞行员，已通过预备航天员考核。",
                "current_state": "对地球处境带有逃避情绪。",
                "related_memory_index": ["M1"],
            }
        ],
        "relations": [],
        "relationship_diagnostics": {},
        "episodic_memories": [],
        "related_scene_summaries": [],
        "references": [],
    }

    markdown = format_memory_packet_markdown(packet)

    assert "- profile: 年轻飞行员" in markdown
    assert "- current state before scene_0006: 对地球处境带有逃避情绪。" in markdown
    assert "- description:" not in markdown


def test_memory_packet_markdown_uses_unit_boundary_label_without_scene_duplication() -> None:
    packet = {
        "retrieval_boundary": {
            "before_unit_id": "chapter_0006",
            "unit_type": "chapter",
            "unit_label": "chapter",
            "before_scene_id": "scene_0006",
        },
        "entities": [
            {
                "canonical_name": "刘培强",
                "entity_type": "character",
                "current_state": "正在返航前。",
                "related_memory_index": [],
            }
        ],
        "relations": [],
        "relationship_diagnostics": {},
        "episodic_memories": [],
        "related_scene_summaries": [],
        "references": [],
    }

    markdown = format_memory_packet_markdown(packet)

    assert "- current state before chapter_0006: 正在返航前。" in markdown


def test_memory_packet_markdown_includes_memory_temporal_scope() -> None:
    packet = {
        "entities": [],
        "relations": [],
        "relationship_diagnostics": {},
        "episodic_memories": [
            {
                "index": "M1",
                "summary": "数字生命技术是一种延续文明的技术",
                "scene_id": "scene_0001",
                "memory_temporal_scope": "atemporal_fact",
                "source_ref": "R1",
            }
        ],
        "related_scene_summaries": [],
        "references": [],
    }

    markdown = format_memory_packet_markdown(packet)

    assert "scope: atemporal_fact" in markdown
