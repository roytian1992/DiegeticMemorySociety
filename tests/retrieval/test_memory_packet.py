from pathlib import Path

import pytest

from dms.retrieval import (
    MemoryPacketConfig,
    build_memory_packet,
    decompose_writing_intent,
    format_memory_packet_markdown,
)
from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
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
            scene_top_k=2,
            entity_memory_top_k=3,
            embedding_dim=64,
        )
    )

    assert "writing_intent" not in packet
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
    assert "刘培强返航途中情绪焦躁" in liu["profile"] or "刘培强返航途中情绪焦躁" in liu["current_state"]
    memory_by_index = {memory["index"]: memory for memory in packet["episodic_memories"]}
    liu_memory_ids = {memory_by_index[index]["memory_id"] for index in liu["related_memory_index"]}
    assert liu_memory_ids == {"scene_0004_memory_001"}
    assert packet["relations"][0]["relation_type"] == "responsible_for"
    assert packet["relations"][0]["source_refs"]
    assert any(reference["kind"] == "relationship_evidence" for reference in packet["references"])
    assert packet["related_scene_summaries"][0]["scene_id"] == "scene_0001"
    assert "stated_fact" in packet["related_scene_summaries"][0]["retrieval_sources"]
    assert packet["related_scene_summaries"][0]["metadata"]["title"] == "1、INT.日.印度 数字生命研究室"
    assert packet["related_scene_summaries"][0]["metadata"]["setting"]["location"] == "印度 数字生命研究室"


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
