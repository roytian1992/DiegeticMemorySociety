from __future__ import annotations

import json

from dms.external_references import ExternalReferences, ExternalReferencesConfig
from dms.llm import FakeReferenceKGClient


def test_external_references_mem0_like_add_search_get_all(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(
            db_path=tmp_path / "external_refs.sqlite",
            work_dir=tmp_path / "runs",
        ),
        llm_client=FakeReferenceKGClient(),
    )

    add_result = memory.add(refs_dir, metadata={"project_id": "demo"})
    search_result = memory.search("刘培强 550A 张鹏", evidence_budget="standard")
    all_result = memory.get_all()

    assert add_result["results"][0]["event"] == "ADD"
    assert add_result["summary"]["asset_counts"]["reference_source_local_entities"] >= 3
    assert add_result["summary"]["facts_properties"]["selection"]["min_entity_degree"] == 2
    assert add_result["summary"]["facts_properties"]["selection"]["total_jobs"] >= add_result["summary"]["facts_properties"]["eligible_entity_cluster_count"]
    assert add_result["summary"]["facts_properties"]["entity_property_count"] >= 1
    assert add_result["summary"]["asset_counts"]["reference_entity_properties"] >= 1
    assert search_result["results"]
    assert search_result["evidence_budget"]["profile"] == "standard"
    assert search_result["query_plan"]["execution"] == "multi_pass_namespace_weighted"
    assert {item["name"] for item in search_result["raw_retrieval"]["passes"]} == {"base", "low_level", "high_level"}
    low_pass = next(item for item in search_result["raw_retrieval"]["passes"] if item["name"] == "low_level")
    high_pass = next(item for item in search_result["raw_retrieval"]["passes"] if item["name"] == "high_level")
    assert high_pass["query"] == search_result["query"]
    assert low_pass["entity_top_k"] > low_pass["relationship_top_k"]
    assert high_pass["relationship_top_k"] > high_pass["chunk_top_k"]
    assert search_result["evidence_packet"]["answer_context"]
    assert any("刘培强" in item["memory"] or "550A" in item["memory"] for item in search_result["results"])
    assert all_result["results"][0]["type"] == "source_document"


def test_external_references_search_accepts_source_filters(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强", filters={"source_path": str(refs_dir / "profiles.md")})

    assert result["raw"]["source_filter"]["mode"] == "files"
    assert all(item["source_filter"]["mode"] == "files" for item in result["raw_retrieval"]["passes"])
    assert result["results"]


def test_external_references_add_recurses_directory_and_accepts_workers_override(tmp_path):
    refs_dir = tmp_path / "refs"
    nested_dir = refs_dir / "nested"
    nested_dir.mkdir(parents=True)
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")
    (nested_dir / "timeline.md").write_text("# 时间线\n2044 年爆发太空电梯危机。刘培强和550A参与训练，张鹏复盘。\n", encoding="utf-8")
    (nested_dir / "training.md").write_text("# 训练\n张鹏指导刘培强使用550A完成稳定性训练。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(
            db_path=tmp_path / "external_refs.sqlite",
            workers=1,
        ),
        llm_client=FakeReferenceKGClient(),
    )
    add_result = memory.add(refs_dir, workers=2)

    assert add_result["summary"]["ingest"]["file_count"] == 3
    assert add_result["summary"]["ingest"]["workers"] == 2
    assert add_result["summary"]["kg"]["workers"] == 2
    assert add_result["summary"]["facts_properties"]["selection"]["selected_count"] >= 2
    assert add_result["summary"]["facts_properties"]["workers"] == 2
    assert add_result["summary"]["import"]["entity_disambiguation"] is True
    assert add_result["summary"]["asset_counts"]["reference_full_docs"] == 3


def test_external_references_add_can_disable_entity_disambiguation_per_ingest(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(
            db_path=tmp_path / "external_refs.sqlite",
            entity_disambiguation=True,
        ),
        llm_client=FakeReferenceKGClient(),
    )
    add_result = memory.add(refs_dir, entity_disambiguation=False, disambiguation_lexical_threshold=0.75)

    assert add_result["summary"]["facts_properties"]["selection"]["entity_disambiguation"] is False
    assert add_result["summary"]["facts_properties"]["selection"]["disambiguation_lexical_threshold"] == 0.75
    assert add_result["summary"]["import"]["entity_disambiguation"] is False
    assert add_result["summary"]["import"]["disambiguation_lexical_threshold"] == 0.75


def test_external_references_fact_property_extraction_can_be_disabled_per_ingest(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(
            db_path=tmp_path / "external_refs.sqlite",
            extract_fact_properties=True,
        ),
        llm_client=FakeReferenceKGClient(),
    )
    add_result = memory.add(refs_dir, extract_fact_properties=False)
    result = memory.search("刘培强和张鹏是什么关系？")

    assert add_result["summary"]["facts_properties"] is None
    assert add_result["summary"]["import"]["facts_dir"] is None
    assert result["raw"]["entity_properties"] == []
    assert add_result["summary"]["asset_counts"]["reference_entity_properties"] == 0


def test_external_references_search_returns_bound_evidence_packet_for_relation_queries(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强和张鹏是什么关系？", evidence_budget="standard")

    assert "刘培强" in result["query_plan"]["low_level_keywords"]
    assert "张鹏" in result["query_plan"]["low_level_keywords"]
    assert "关系" in result["query_plan"]["high_level_keywords"]
    low_pass = next(item for item in result["raw_retrieval"]["passes"] if item["name"] == "low_level")
    assert low_pass["query"] == "刘培强 张鹏"
    assert result["evidence_packet"]["relations"] or result["evidence_packet"]["facts"]
    assert result["citations"]
    assert "[R1]" in result["evidence_packet"]["answer_context"] or "[F1]" in result["evidence_packet"]["answer_context"]


def test_external_references_search_supports_evidence_budget_profiles_and_custom_overrides(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    compact = memory.search("刘培强和张鹏是什么关系？", evidence_budget="compact")
    deep = memory.search(
        "刘培强和张鹏是什么关系？",
        evidence_budget={"profile": "deep", "max_facts": 2, "max_chunks": 1, "include_answer_context": True},
    )

    compact_base = next(item for item in compact["raw_retrieval"]["passes"] if item["name"] == "base")
    deep_base = next(item for item in deep["raw_retrieval"]["passes"] if item["name"] == "base")
    assert compact["evidence_budget"]["profile"] == "compact"
    assert deep["evidence_budget"]["profile"] == "deep"
    assert deep_base["top_k"] > compact_base["top_k"]
    assert len(deep["evidence_packet"]["facts"]) <= 2
    assert len(deep["evidence_packet"]["chunks"]) <= 1
    assert deep["evidence_budget"]["include_answer_context"] is True
    assert deep["evidence_packet"]["answer_context"]


def test_external_references_search_keeps_legacy_limit_as_budget_override(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强和张鹏是什么关系？", limit=3)

    assert result["evidence_budget"]["profile"] == "legacy_limit"
    assert result["evidence_budget"]["legacy_limit"] == 3
    assert len(result["results"]) <= 3
    assert len(result["evidence_packet"]["facts"]) <= 3


def test_external_references_evidence_packet_keeps_full_chunk_content(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    long_tail = "很长的补充说明" * 80
    (refs_dir / "profiles.md").write_text(
        f"# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。{long_tail}\n",
        encoding="utf-8",
    )

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强和张鹏是什么关系？", evidence_budget={"profile": "standard", "max_chunks": 1})
    chunk = result["evidence_packet"]["chunks"][0]

    assert "content" in chunk
    assert chunk["content_chars"] == len(chunk["content"])
    assert "很长的补充说明" in chunk["content"]
    assert chunk["content_chars"] > 520


def test_external_references_evidence_packet_filters_unrelated_chunks(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")
    (refs_dir / "unrelated.md").write_text("# 三体参考\n丁仪和汪淼讨论科学边界危机。", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强和张鹏是什么关系？", evidence_budget={"profile": "deep", "max_chunks": 8})
    chunk_text = "\n".join(chunk["content"] for chunk in result["evidence_packet"]["chunks"])

    assert "刘培强" in chunk_text
    assert "张鹏" in chunk_text
    assert "丁仪" not in chunk_text


def test_external_references_evidence_packet_does_not_expose_graph_field_separator(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")

    memory = ExternalReferences(
        ExternalReferencesConfig(db_path=tmp_path / "external_refs.sqlite"),
        llm_client=FakeReferenceKGClient(),
    )
    memory.add(refs_dir)

    result = memory.search("刘培强和张鹏是什么关系？", evidence_budget="deep")

    assert "<SEP>" not in json.dumps(result["evidence_packet"], ensure_ascii=False)
