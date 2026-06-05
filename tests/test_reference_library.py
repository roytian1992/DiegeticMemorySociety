from __future__ import annotations

import json
from pathlib import Path

from dms.cli import main
from dms.llm import FakeReferenceItemClient, LLMResult
from dms.reference_library import (
    ReferenceContextQuery,
    ReferenceItemExtractionConfig,
    ReferenceItemImportConfig,
    ReferenceLibraryIngestConfig,
    build_reference_context,
    extract_reference_items,
    ingest_reference_library,
    import_reference_items,
    list_reference_documents,
)
from tests.helpers import write_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_reference_items_import_and_context_visibility(tmp_path: Path) -> None:
    items_path = tmp_path / "reference_items.jsonl"
    write_jsonl(
        items_path,
        [
            {
                "item_id": "world_001",
                "doc_id": "world_bible",
                "item_type": "world_bible",
                "subject": "550A",
                "statement": "550A是数字生命研究使用的量子计算机。",
                "evidence": "550A位于数字生命研究室。",
                "knowledge_scope": "world_public",
                "known_to": "all",
                "authority": 0.9,
                "confidence": 0.95,
            },
            {
                "item_id": "liu_private_001",
                "doc_id": "profiles",
                "item_type": "character_profile",
                "subject": "刘培强",
                "statement": "刘培强知道550A会处理脑电波数据。",
                "evidence": "角色设定：刘培强了解550A的基础功能。",
                "knowledge_scope": "character_private",
                "known_to": ["刘培强"],
                "authority": 0.8,
                "confidence": 0.9,
            },
            {
                "item_id": "author_only_001",
                "doc_id": "notes",
                "item_type": "author_note",
                "subject": "张鹏",
                "statement": "张鹏在作者笔记中承担稳定后辈的功能。",
                "knowledge_scope": "author_only",
                "authority": 0.7,
                "confidence": 0.8,
            },
            {
                "item_id": "style_001",
                "doc_id": "style",
                "item_type": "style_guide",
                "subject": "对白",
                "statement": "驾驶舱对白应短促，避免长篇解释。",
                "knowledge_scope": "style_only",
                "authority": 0.6,
                "confidence": 0.9,
            },
            {
                "item_id": "future_reveal_001",
                "doc_id": "timeline",
                "item_type": "timeline_doc",
                "subject": "UEG",
                "statement": "UEG在后续章节公开550A的限制。",
                "knowledge_scope": "revealed_by_story",
                "known_to": "all",
                "available_from": "scene_0010",
                "authority": 0.7,
                "confidence": 0.8,
            },
        ],
    )
    db_path = tmp_path / "reference.sqlite"

    summary = import_reference_items(ReferenceItemImportConfig(items_path=items_path, db_path=db_path, reset=True))

    assert summary["reference_items"] == 5
    docs_before_scene_5 = list_reference_documents(db_path, before_scene_id="scene_0005")
    assert "future_reveal_001" not in {doc["item_id"] for doc in docs_before_scene_5}

    context, trace = build_reference_context(
        ReferenceContextQuery(
            db_path=db_path,
            query="刘培强和550A在驾驶舱返航任务中讨论脑电波",
            matched_entities=(
                {
                    "entity_id": "character_0001",
                    "canonical_name": "刘培强",
                    "aliases": ["培强"],
                },
            ),
            before_scene_id="scene_0005",
            top_k=8,
            author_top_k=8,
            character_top_k=8,
            style_top_k=8,
            timeline_top_k=8,
        )
    )

    assert trace["enabled"] is True
    assert trace["strategy"] == "sql_ranked"
    assert {item["item_id"] for item in context["author_reference_context"]} >= {
        "world_001",
        "liu_private_001",
        "author_only_001",
    }
    assert {item["item_id"] for item in context["character_reference_knowledge"]} == {
        "world_001",
        "liu_private_001",
    }
    assert context["style_reference_context"][0]["item_id"] == "style_001"
    assert context["timeline_reference_claims"] == []


def test_cli_import_reference_items(tmp_path: Path) -> None:
    items_path = tmp_path / "reference_items.jsonl"
    write_jsonl(
        items_path,
        [
            {
                "item_id": "style_001",
                "item_type": "style_guide",
                "statement": "对白要短促。",
                "knowledge_scope": "style_only",
            }
        ],
    )
    db_path = tmp_path / "reference.sqlite"

    code = main(["import-reference-items", str(items_path), "--output-db", str(db_path), "--overwrite"])

    assert code == 0
    docs = list_reference_documents(db_path)
    assert docs[0]["item_id"] == "style_001"
    assert docs[0]["knowledge_scope"] == "style_only"


def test_ingest_reference_library_chunks_mixed_formats(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "world.md").write_text(
        "# 世界观\n550A位于数字生命研究室。\n\n# 风格\n对白要短促，信息密度高。\n",
        encoding="utf-8",
    )
    (refs_dir / "profiles.txt").write_text("刘培强：年轻飞行员。\n张鹏：教官。", encoding="utf-8")
    (refs_dir / "timeline.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "title": "时间线",
                        "chunks": [
                            {"heading": "2044", "content": "2044 年发生太空电梯危机。"},
                            {"heading": "地点", "content": "加蓬基地负责重要发射任务。"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(refs_dir / "notes.jsonl", [{"title": "笔记", "content": "作者笔记：不要把设定讲成说明书。"}])
    output_dir = tmp_path / "library"

    summary = ingest_reference_library(
        ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=output_dir, max_chunk_chars=80)
    )

    raw_docs = _read_jsonl(output_dir / "raw_documents.jsonl")
    chunks = _read_jsonl(output_dir / "reference_chunks.jsonl")
    assert summary["file_count"] == 4
    assert summary["raw_document_count"] == 5
    assert len(raw_docs) == 5
    assert summary["reference_chunk_count"] == len(chunks)
    assert {chunk["format"] for chunk in chunks} >= {"md", "txt", "json", "jsonl"}
    assert any(chunk["heading"] == "世界观" and "550A" in chunk["content"] for chunk in chunks)


def test_extract_reference_items_fake_client_aligns_evidence(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "mixed.md").write_text(
        "# 世界与人物\n550A位于数字生命研究室。刘培强第一次接触相关训练。\n\n# 时间与风格\n2044 年爆发太空电梯危机。对白要短促，信息密度高。\n",
        encoding="utf-8",
    )
    library_dir = tmp_path / "library"
    ingest_reference_library(ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=library_dir))
    output_dir = tmp_path / "extract"

    summary = extract_reference_items(
        ReferenceItemExtractionConfig(library_dir=library_dir, output_dir=output_dir, dry_run=False),
        llm_client=FakeReferenceItemClient(),
    )

    items = _read_jsonl(output_dir / "reference_items.jsonl")
    assert summary["accepted_item_count"] >= 4
    assert summary["rejected_item_count"] == 0
    assert {item["item_type"] for item in items} >= {"world_bible", "character_profile", "timeline_doc", "style_guide"}
    assert all(item["evidence_verification_status"] in {"exact", "fuzzy_aligned"} for item in items)
    assert all(item["evidence_start"] is not None and item["evidence_end"] is not None for item in items)
    assert [item for item in items if item["item_type"] == "style_guide"][0]["knowledge_scope"] == "style_only"


def test_extract_reference_items_rejects_bad_items_without_failing_chunk(tmp_path: Path) -> None:
    class BadReferenceItemClient:
        provider = "fake"
        model = "bad-reference-items"

        def complete(self, prompt: str) -> LLMResult:
            payload = {
                "chunk_id": "ignored",
                "reference_items": [
                    {
                        "item_type": "world_bible",
                        "subject": "550A",
                        "statement": "550A是设定资料中的设备。",
                        "evidence": "550A",
                        "knowledge_scope": "author_only",
                    },
                    {
                        "item_type": "world_bible",
                        "subject": "错误证据",
                        "statement": "这条证据不在原文中。",
                        "evidence": "不存在的证据",
                        "knowledge_scope": "author_only",
                    },
                    {
                        "item_type": "author_note",
                        "subject": "缺少 statement",
                        "evidence": "550A",
                        "knowledge_scope": "author_only",
                    },
                ],
            }
            text = json.dumps(payload, ensure_ascii=False)
            return LLMResult(text=text, provider=self.provider, model=self.model, raw_response={"fake": True}, usage={})

    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "world.txt").write_text("550A位于数字生命研究室。", encoding="utf-8")
    library_dir = tmp_path / "library"
    ingest_reference_library(ReferenceLibraryIngestConfig(input_path=refs_dir, output_dir=library_dir))
    output_dir = tmp_path / "extract"

    summary = extract_reference_items(
        ReferenceItemExtractionConfig(library_dir=library_dir, output_dir=output_dir, dry_run=False),
        llm_client=BadReferenceItemClient(),
    )

    items = _read_jsonl(output_dir / "reference_items.jsonl")
    rejected = _read_jsonl(output_dir / "rejected_items.jsonl")
    assert summary["accepted_item_count"] == 1
    assert len(items) == 1
    assert len(rejected) == 2
    assert {item["rejection_reason"] for item in rejected} == {
        "evidence_not_aligned",
        "Reference item missing statement at index 3",
    }


def test_cli_reference_ingest_and_extract_fake(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "style.md").write_text("# 风格\n对白要短促，信息密度高。\n", encoding="utf-8")
    library_dir = tmp_path / "library"
    extract_dir = tmp_path / "extract"

    ingest_code = main(
        [
            "ingest-reference-library",
            str(refs_dir),
            "--output-dir",
            str(library_dir),
            "--max-chunk-chars",
            "120",
        ]
    )
    extract_code = main(
        [
            "extract-reference-items",
            str(library_dir),
            "--output-dir",
            str(extract_dir),
            "--no-dry-run",
            "--provider",
            "fake",
        ]
    )

    assert ingest_code == 0
    assert extract_code == 0
    items = _read_jsonl(extract_dir / "reference_items.jsonl")
    assert items[0]["item_type"] == "style_guide"
    assert items[0]["knowledge_scope"] == "style_only"
