from __future__ import annotations

from pathlib import Path

import pytest

from dms.context_kernel import (
    ChromaEmbeddingProvider,
    ContextAssembler,
    ContextChromaIndexConfig,
    CreativeContextItem,
    CreativeContextPacketConfig,
    CreativeContextStore,
    CreativeMemoryKernel,
    CreativeScope,
    EntityPatch,
    EvidenceRef,
    ExternalKnowledgeKernel,
    LLMClientProvider,
    SourceRecord,
    SourceUnit,
    ScoreReranker,
    audit_external_vs_artifact_conflicts,
    build_context_chroma_index,
    export_external_entity_links,
    export_external_timeline_claims,
    format_creative_context_packet_markdown,
    import_reference_library_items,
    reconcile_items_deterministic,
    search_context_chroma_index,
    stable_id,
    write_creative_context_json_schemas,
)
from dms.llm import FakeReferenceKGClient, LLMResult
from dms.reference_library import (
    ReferenceKGExtractionConfig,
    ReferenceKnowledgeImportConfig,
    ReferenceLibraryIngestConfig,
    extract_reference_kg,
    import_reference_knowledge,
    ingest_reference_library,
)
from tests.helpers import write_jsonl


def test_context_store_records_history_links_promotion_and_entity_patches(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.add_source(SourceRecord(source_id="conversation:001", project_id="we2", source_type="conversation"))
    kernel.add_unit(
        SourceUnit(
            unit_id="turn_001",
            source_id="conversation:001",
            project_id="we2",
            source_type="conversation",
            unit_type="turn",
            unit_order=1,
            speaker="user",
            text="不要把张鹏写成法定监护人，他更像粗粝的老兵教官。",
        )
    )
    item_id = "conv:zhangpeng:no_guardian"
    kernel.add_item(
        CreativeContextItem(
            item_id=item_id,
            project_id="we2",
            source_type="conversation",
            source_id="conversation:001",
            unit_id="turn_001",
            item_type="correction",
            subject="张鹏",
            statement="用户明确要求不要把张鹏写成法定监护人，而应保留粗粝老兵教官气质。",
            entity_ids=("entity:zhang_peng",),
            evidence_refs=(
                EvidenceRef(
                    evidence_id="ev:turn_001",
                    item_id=item_id,
                    source_id="conversation:001",
                    unit_id="turn_001",
                    text="不要把张鹏写成法定监护人",
                    start_offset=0,
                    end_offset=13,
                ),
            ),
            authority="user_explicit",
            confidence=1.0,
            status="active",
            visibility="author_only",
            temporal_scope="not_applicable",
        ),
        actor="user",
        reason="explicit correction",
    )

    kernel.add_entity_patch(
        EntityPatch(
            patch_id="patch:zhangpeng:no_guardian",
            project_id="we2",
            entity_id="entity:zhang_peng",
            source_item_id=item_id,
            patch_type="constrain",
            target_field="style",
            patch_statement="写张鹏时避免法定监护人口吻，保留老兵教官式稳定。",
            authority="user_explicit",
        ),
        actor="user",
        reason="entity style overlay",
    )

    promoted = kernel.promote(item_id, "story_bible", actor="user", reason="用户确认作为项目约束")
    history_events = [entry["event"] for entry in kernel.history(item_id)]
    entity_view = kernel.entity_view(project_id="we2", entity_id="entity:zhang_peng")

    assert promoted["source_type"] == "narrative_artifact"
    assert promoted["status"] == "canonical"
    assert promoted["payload"]["promoted_from"] == item_id
    assert "ADD" in history_events
    assert "PATCH" in history_events
    assert "PROMOTE" in history_events
    assert entity_view["conversation_layer"][0]["item_id"] == item_id
    assert entity_view["active_patches"][0]["patch_statement"].startswith("写张鹏时避免")


def test_external_reference_import_is_searchable_but_not_canon_until_promoted(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text(
        "# 角色资料\n张鹏是航天员教官。他训练刘培强完成危险任务。\n",
        encoding="utf-8",
    )
    library_dir = tmp_path / "library"
    kg_dir = tmp_path / "kg"
    reference_db = tmp_path / "reference.sqlite"
    ingest_reference_library(ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=library_dir, overwrite=True))
    extract_reference_kg(
        ReferenceKGExtractionConfig(library_dir=library_dir, output_dir=kg_dir, dry_run=False, overwrite=True),
        llm_client=FakeReferenceKGClient(),
    )
    import_reference_knowledge(ReferenceKnowledgeImportConfig(library_dir=library_dir, kg_dir=kg_dir, db_path=reference_db, reset=True))
    kernel = _kernel(tmp_path)
    summary = import_reference_library_items(
        kernel,
        reference_db_path=reference_db,
        project_id="we2",
    )
    external_kernel = ExternalKnowledgeKernel(kernel)

    hits = external_kernel.search("张鹏 教官", scope=CreativeScope(project_id="we2"), top_k=3)
    qa = external_kernel.qa("资料里张鹏是什么身份？", scope=CreativeScope(project_id="we2"), top_k=3)
    promoted = external_kernel.promote(hits[0]["item_id"], "story_bible", actor="user", reason="采纳外部角色资料")

    assert summary["asset_model"] == "source_local_external_reference_v1"
    assert summary["imported_items"] >= 3
    assert summary["source_roles"]["external_reference_entity"] >= 1
    assert summary["source_roles"]["external_reference_fact"] >= 1
    assert hits[0]["source_type"] == "external_reference"
    assert hits[0]["authority"] == "external_source"
    assert hits[0]["status"] == "active"
    assert hits[0]["payload"]["external_reference_default_canon"] is False
    assert hits[0]["payload"]["metadata"]["asset_model"] == "source_local_external_reference_v1"
    assert qa["trace"]["external_reference_default_canon"] is False
    assert promoted["source_type"] == "narrative_artifact"
    assert promoted["status"] == "canonical"
    assert promoted["payload"]["promoted_from"] == hits[0]["item_id"]
    assert promoted["evidence_refs"][0]["text"]


def test_lightrag_reference_knowledge_imports_into_context_kernel(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text(
        "# 角色资料\n张鹏是航天员教官。他训练刘培强完成危险任务。\n",
        encoding="utf-8",
    )
    library_dir = tmp_path / "library"
    kg_dir = tmp_path / "kg"
    reference_db = tmp_path / "reference.sqlite"

    ingest_reference_library(ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=library_dir, overwrite=True))
    extract_reference_kg(
        ReferenceKGExtractionConfig(library_dir=library_dir, output_dir=kg_dir, dry_run=False, overwrite=True),
        llm_client=FakeReferenceKGClient(),
    )
    import_reference_knowledge(ReferenceKnowledgeImportConfig(library_dir=library_dir, kg_dir=kg_dir, db_path=reference_db, reset=True))

    kernel = _kernel(tmp_path)
    summary = import_reference_library_items(kernel, reference_db_path=reference_db, project_id="we2")
    external_kernel = ExternalKnowledgeKernel(kernel)
    hits = external_kernel.search("张鹏 刘培强", scope=CreativeScope(project_id="we2"), top_k=5)

    assert summary["imported_items"] >= 3
    assert any(hit["source_type"] == "external_reference" for hit in hits)
    assert any("张鹏" in hit["subject"] or "张鹏" in hit["statement"] for hit in hits)
    assert all(hit["payload"]["external_reference_default_canon"] is False for hit in hits)


def test_external_qa_can_generate_grounded_answer_with_provider(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    _add_source_and_unit(kernel, "refs:library", "external_reference", "refs:library:chunk_001", "chunk")
    external_kernel = ExternalKnowledgeKernel(kernel)
    external_kernel.add_item(
        CreativeContextItem(
            item_id="external:zhangpeng:instructor",
            project_id="we2",
            source_type="external_reference",
            source_id="refs:library",
            unit_id="refs:library:chunk_001",
            item_type="character_profile",
            subject="张鹏",
            statement="外部资料称张鹏是航天员教官。",
            entity_ids=("entity:zhang_peng",),
            evidence_refs=(
                EvidenceRef(
                    evidence_id="ev:external:zhangpeng",
                    item_id="external:zhangpeng:instructor",
                    source_id="refs:library",
                    unit_id="refs:library:chunk_001",
                    text="张鹏：航天员教官。",
                ),
            ),
            authority="external_source",
            confidence=0.9,
        )
    )

    qa = external_kernel.qa(
        "张鹏是什么身份？",
        scope=CreativeScope(project_id="we2"),
        top_k=3,
        llm_provider=LLMClientProvider(_FakeQAClient()),
    )

    assert qa["answer_mode"] == "source_grounded_llm"
    assert "航天员教官" in qa["answer"]
    assert qa["citations"][0]["ref"] == "E1"
    assert qa["items"][0]["default_canon"] is False
    assert qa["trace"]["provider"] == "fake"


def test_creative_context_packet_keeps_source_roles_separate(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    _add_source_and_unit(kernel, "conversation:001", "conversation", "turn_001", "turn")
    _add_source_and_unit(kernel, "artifact:screenplay", "narrative_artifact", "scene_0006", "scene")
    _add_source_and_unit(kernel, "refs:library", "external_reference", "refs:library:chunk_001", "chunk")

    kernel.add_item(
        CreativeContextItem(
            item_id="conv:no_guardian",
            project_id="we2",
            source_type="conversation",
            source_id="conversation:001",
            unit_id="turn_001",
            item_type="correction",
            subject="张鹏",
            statement="不要把张鹏写成法定监护人。",
            entity_ids=("entity:zhang_peng",),
            authority="user_explicit",
            confidence=1.0,
            visibility="author_only",
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="artifact:scene6:comfort",
            project_id="we2",
            source_type="narrative_artifact",
            source_id="artifact:screenplay",
            unit_id="scene_0006",
            item_type="event",
            subject="张鹏",
            statement="张鹏在返航途中安抚刘培强。",
            entity_ids=("entity:zhang_peng", "entity:liu_peiqiang"),
            authority="artifact_canonical",
            confidence=1.0,
            status="canonical",
            visibility="character_visible",
            temporal_scope="temporal_episode",
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="external:zhangpeng:instructor",
            project_id="we2",
            source_type="external_reference",
            source_id="refs:library",
            unit_id="refs:library:chunk_001",
            item_type="character_profile",
            subject="张鹏",
            statement="外部资料称张鹏是航天员教官。",
            entity_ids=("entity:zhang_peng",),
            authority="external_source",
            confidence=0.9,
            visibility="author_only",
            payload={"external_reference_default_canon": False},
        )
    )

    packet = ContextAssembler(kernel).build_packet(
        CreativeContextPacketConfig(
            request="写张鹏和刘培强返航时的互动",
            scope=CreativeScope(project_id="we2", before_unit_id="scene_0007", before_unit_order=7),
            entity_ids=("entity:zhang_peng",),
            top_k=8,
        )
    )
    markdown = format_creative_context_packet_markdown(packet)

    assert [item["item_id"] for item in packet["conversation_guidance"]] == ["conv:no_guardian"]
    assert [item["item_id"] for item in packet["artifact_memory"]] == ["artifact:scene6:comfort"]
    assert [item["item_id"] for item in packet["external_reference_context"]] == ["external:zhangpeng:instructor"]
    assert packet["trace"]["source_roles"]["external_reference"].endswith("not canon by default")
    assert "## Conversation Guidance" in markdown
    assert "## Artifact Memory" in markdown
    assert "## External Reference Context" in markdown


def test_context_packet_includes_entity_patches(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    _add_source_and_unit(kernel, "conversation:001", "conversation", "turn_001", "turn")
    item_id = "conv:zhangpeng:no_guardian"
    kernel.add_item(
        CreativeContextItem(
            item_id=item_id,
            project_id="we2",
            source_type="conversation",
            source_id="conversation:001",
            unit_id="turn_001",
            item_type="correction",
            subject="张鹏",
            statement="不要把张鹏写成法定监护人。",
            entity_ids=("entity:zhang_peng",),
            authority="user_explicit",
            confidence=1.0,
        )
    )
    kernel.add_entity_patch(
        EntityPatch(
            patch_id="patch:zhangpeng:no_guardian",
            project_id="we2",
            entity_id="entity:zhang_peng",
            source_item_id=item_id,
            patch_type="constrain",
            target_field="role",
            patch_statement="写张鹏时避免法定监护人口吻。",
            authority="user_explicit",
        )
    )

    packet = ContextAssembler(kernel).build_packet(
        CreativeContextPacketConfig(
            request="写张鹏",
            scope=CreativeScope(project_id="we2"),
            entity_ids=("entity:zhang_peng",),
        )
    )
    markdown = format_creative_context_packet_markdown(packet)

    assert packet["entity_patch_context"][0]["item_id"] == "patch:zhangpeng:no_guardian"
    assert "## Entity Patch Context" in markdown
    assert "避免法定监护人口吻" in markdown


def test_context_packet_accepts_reranker_and_hash_embedding_provider(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    _add_source_and_unit(kernel, "conversation:001", "conversation", "turn_001", "turn")
    kernel.add_item(
        CreativeContextItem(
            item_id="conv:brief_dialogue",
            project_id="we2",
            source_type="conversation",
            source_id="conversation:001",
            unit_id="turn_001",
            item_type="style_preference",
            subject="对白",
            statement="对白要短促。",
            authority="user_explicit",
            confidence=1.0,
        )
    )

    packet = ContextAssembler(kernel, reranker=ScoreReranker()).build_packet(
        CreativeContextPacketConfig(
            request="对白",
            scope=CreativeScope(project_id="we2"),
            top_k=1,
        )
    )
    embedding = ChromaEmbeddingProvider(provider="hash")

    assert packet["style_guidance"][0]["item_id"] == "conv:brief_dialogue"
    assert len(embedding.embed_query("对白")) == 384


def test_context_search_respects_boundary_and_status(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.add_source(SourceRecord(source_id="artifact:screenplay", project_id="we2", source_type="narrative_artifact"))
    for unit_id, order in (("scene_0001", 1), ("scene_0010", 10)):
        kernel.add_unit(
            SourceUnit(
                unit_id=unit_id,
                source_id="artifact:screenplay",
                project_id="we2",
                source_type="narrative_artifact",
                unit_type="scene",
                unit_order=order,
            )
        )
    kernel.add_item(
        CreativeContextItem(
            item_id="artifact:past",
            project_id="we2",
            source_type="narrative_artifact",
            source_id="artifact:screenplay",
            unit_id="scene_0001",
            item_type="fact",
            statement="550A已经开始分析脑电波。",
            entity_ids=("entity:550a",),
            authority="artifact_canonical",
            status="canonical",
            visibility="character_visible",
            temporal_scope="atemporal_fact",
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="artifact:future",
            project_id="we2",
            source_type="narrative_artifact",
            source_id="artifact:screenplay",
            unit_id="scene_0010",
            item_type="fact",
            statement="550A未来公开限制。",
            entity_ids=("entity:550a",),
            authority="artifact_canonical",
            status="canonical",
            visibility="character_visible",
            temporal_scope="temporal_episode",
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="artifact:rejected",
            project_id="we2",
            source_type="narrative_artifact",
            source_id="artifact:screenplay",
            unit_id="scene_0001",
            item_type="fact",
            statement="被拒绝的550A解释。",
            entity_ids=("entity:550a",),
            authority="model_inferred",
            status="rejected",
            visibility="author_only",
        )
    )

    hits = kernel.search(
        "550A",
        scope=CreativeScope(project_id="we2", before_unit_id="scene_0005", before_unit_order=5),
        source_types=["narrative_artifact"],
        entity_ids=["entity:550a"],
        top_k=10,
    )

    assert [hit["item_id"] for hit in hits] == ["artifact:past"]


def test_reconciliation_deterministic_handles_none_patch_and_promote(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    kernel.add_source(SourceRecord(source_id="conversation:001", project_id="we2", source_type="conversation"))
    kernel.add_unit(
        SourceUnit(
            unit_id="turn_001",
            source_id="conversation:001",
            project_id="we2",
            source_type="conversation",
            unit_type="turn",
            unit_order=1,
        )
    )
    duplicate = CreativeContextItem(
        item_id="conv:pref:001",
        project_id="we2",
        source_type="conversation",
        source_id="conversation:001",
        unit_id="turn_001",
        item_type="style_preference",
        subject="对白",
        statement="对白要短促。",
        entity_ids=("entity:对白",),
        authority="user_explicit",
        confidence=1.0,
    )
    kernel.add_item(duplicate)

    correction = CreativeContextItem(
        item_id="conv:correction:001",
        project_id="we2",
        source_type="conversation",
        source_id="conversation:001",
        unit_id="turn_001",
        item_type="correction",
        subject="张鹏",
        statement="不要把张鹏写成法定监护人。",
        entity_ids=("entity:zhang_peng",),
        authority="user_explicit",
        confidence=1.0,
    )
    decision = CreativeContextItem(
        item_id="conv:decision:001",
        project_id="we2",
        source_type="conversation",
        source_id="conversation:001",
        unit_id="turn_001",
        item_type="creative_decision",
        subject="张鹏",
        statement="张鹏是刘培强的训练教官。",
        entity_ids=("entity:zhang_peng",),
        authority="user_confirmed",
        confidence=1.0,
        payload={"promote_to": "story_bible"},
    )

    results = reconcile_items_deterministic(kernel, [duplicate, correction, decision], actor="user")

    assert [result["action"] for result in results] == ["NONE", "PATCH", "PROMOTE"]
    assert results[1]["patch"]["entity_id"] == "entity:zhang_peng"
    assert results[2]["promoted"]["status"] == "canonical"


def test_context_schema_export_index_search_and_external_audits(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    kernel = _kernel(tmp_path)
    _add_source_and_unit(kernel, "artifact:screenplay", "narrative_artifact", "scene_0001", "scene")
    _add_source_and_unit(kernel, "refs:library", "external_reference", "refs:library:chunk_001", "chunk")
    kernel.add_item(
        CreativeContextItem(
            item_id="artifact:zhangpeng:guardian",
            project_id="we2",
            source_type="narrative_artifact",
            source_id="artifact:screenplay",
            unit_id="scene_0001",
            item_type="fact",
            subject="张鹏",
            statement="张鹏是刘培强的法定监护人。",
            entity_ids=("entity:zhang_peng",),
            authority="artifact_canonical",
            status="canonical",
            confidence=1.0,
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="external:zhangpeng:not_guardian",
            project_id="we2",
            source_type="external_reference",
            source_id="refs:library",
            unit_id="refs:library:chunk_001",
            item_type="character_profile",
            subject="张鹏",
            statement="外部资料说明张鹏不是刘培强的法定监护人，而是训练教官。",
            entity_ids=("entity:zhang_peng",),
            authority="external_source",
            status="active",
            confidence=0.9,
        )
    )
    kernel.add_item(
        CreativeContextItem(
            item_id="external:timeline:2044",
            project_id="we2",
            source_type="external_reference",
            source_id="refs:library",
            unit_id="refs:library:chunk_001",
            item_type="timeline_doc",
            subject="2044",
            statement="2044年发生太空电梯危机。",
            authority="external_source",
            status="active",
            confidence=0.8,
            payload={"timeline_hint": "2044"},
        )
    )

    schema_summary = write_creative_context_json_schemas(tmp_path / "schemas")
    index_summary = build_context_chroma_index(
        ContextChromaIndexConfig(
            context_db_path=tmp_path / "context.sqlite",
            persist_dir=tmp_path / "context_chroma",
            project_id="we2",
            reset=True,
            embedding_dim=64,
        )
    )
    search_result = search_context_chroma_index(
        tmp_path / "context.sqlite",
        persist_dir=tmp_path / "context_chroma",
        query="训练教官 法定监护人",
        scope=CreativeScope(project_id="we2"),
        source_types=["external_reference"],
        top_k=3,
        embedding_dim=64,
    )
    conflicts = audit_external_vs_artifact_conflicts(kernel, scope=CreativeScope(project_id="we2"))
    links = export_external_entity_links(kernel, scope=CreativeScope(project_id="we2"))
    timeline = export_external_timeline_claims(kernel, scope=CreativeScope(project_id="we2"))

    assert schema_summary["schema_count"] >= 8
    assert (tmp_path / "schemas" / "CreativeContextItem.schema.json").is_file()
    assert index_summary["document_count"] == 3
    assert search_result["count"] >= 1
    assert search_result["results"][0]["source_type"] == "external_reference"
    assert conflicts["conflict_count"] == 1
    assert conflicts["conflicts"][0]["external_item_id"] == "external:zhangpeng:not_guardian"
    assert any(link["entity_id"] == "entity:zhang_peng" for link in links["links"])
    assert timeline["claims"][0]["timeline_hint"] == "2044"


def _kernel(tmp_path: Path) -> CreativeMemoryKernel:
    return CreativeMemoryKernel(CreativeContextStore(tmp_path / "context.sqlite", reset=True))


class _FakeQAClient:
    provider = "fake"
    model = "fake-qa"

    def complete(self, prompt: str) -> LLMResult:
        text = "张鹏在外部资料中是航天员教官。[E1]"
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"prompt": prompt},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def _add_source_and_unit(
    kernel: CreativeMemoryKernel,
    source_id: str,
    source_type: str,
    unit_id: str,
    unit_type: str,
) -> None:
    kernel.add_source(SourceRecord(source_id=source_id, project_id="we2", source_type=source_type))
    kernel.add_unit(
        SourceUnit(
            unit_id=unit_id,
            source_id=source_id,
            project_id="we2",
            source_type=source_type,
            unit_type=unit_type,
            unit_order=1,
        )
    )
