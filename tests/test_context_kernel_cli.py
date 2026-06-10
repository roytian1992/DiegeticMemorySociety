from __future__ import annotations

import json
from pathlib import Path

import pytest

from dms.cli import main
from tests.helpers import write_jsonl


def test_context_cli_ingests_conversation_searches_and_builds_packet(tmp_path: Path) -> None:
    transcript = tmp_path / "conversation.jsonl"
    write_jsonl(
        transcript,
        [
            {
                "turn_id": "turn_001",
                "role": "user",
                "content": "不要把张鹏写成法定监护人，他更像粗粝的老兵教官。",
            },
            {
                "turn_id": "turn_002",
                "role": "assistant",
                "content": "明白，我会保留老兵教官气质。",
            },
        ],
    )
    context_db = tmp_path / "context.sqlite"

    ingest_code = main(
        [
            "context-ingest-conversation",
            str(transcript),
            "--context-db",
            str(context_db),
            "--project-id",
            "we2",
            "--conversation-id",
            "dev",
            "--reset-context-store",
        ]
    )
    assert ingest_code == 0

    search_code = main(
        [
            "context-search",
            "--context-db",
            str(context_db),
            "--project-id",
            "we2",
            "--query",
            "张鹏 监护人",
            "--source-type",
            "conversation",
        ]
    )
    assert search_code == 0

    packet_path = tmp_path / "packet.json"
    packet_code = main(
        [
            "build-creative-context-packet",
            "--context-db",
            str(context_db),
            "--project-id",
            "we2",
            "--request",
            "写张鹏返航时安抚刘培强",
            "--entity-id",
            "entity:张鹏",
            "--output",
            str(packet_path),
        ]
    )

    assert packet_code == 0
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["conversation_guidance"]
    assert packet["trace"]["source_roles"]["conversation"].startswith("guidance")


def test_context_cli_imports_references_qa_and_promotes(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n张鹏是航天员教官。他训练刘培强。\n", encoding="utf-8")
    library_dir = tmp_path / "reference_library"
    kg_dir = tmp_path / "reference_kg"
    reference_db = tmp_path / "reference.sqlite"
    context_db = tmp_path / "context.sqlite"

    assert main(["ingest-reference-library", str(refs_dir), "--output-dir", str(library_dir)]) == 0
    assert main(["extract-reference-kg", str(library_dir), "--output-dir", str(kg_dir), "--no-dry-run", "--provider", "fake"]) == 0
    assert (
        main(
            [
                "import-reference-knowledge",
                "--library-dir",
                str(library_dir),
                "--kg-dir",
                str(kg_dir),
                "--output-db",
                str(reference_db),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "context-import-references",
                str(reference_db),
                "--context-db",
                str(context_db),
                "--project-id",
                "we2",
                "--reset-context-store",
            ]
        )
        == 0
    )

    qa_path = tmp_path / "qa.json"
    assert (
        main(
            [
                "context-external-qa",
                "--context-db",
                str(context_db),
                "--project-id",
                "we2",
                "--question",
                "张鹏是什么身份？",
                "--output",
                str(qa_path),
            ]
        )
        == 0
    )
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["items"][0]["default_canon"] is False

    qa_llm_path = tmp_path / "qa_llm.json"
    assert (
        main(
            [
                "context-external-qa",
                "--context-db",
                str(context_db),
                "--project-id",
                "we2",
                "--question",
                "张鹏是什么身份？",
                "--answer-with-llm",
                "--output",
                str(qa_llm_path),
            ]
        )
        == 0
    )
    qa_llm = json.loads(qa_llm_path.read_text(encoding="utf-8"))
    assert qa_llm["answer_mode"] == "source_grounded_llm"
    assert "航天员教官" in qa_llm["answer"]

    item_id = qa["items"][0]["item_id"]
    assert (
        main(
            [
                "context-promote-item",
                "--context-db",
                str(context_db),
                "--item-id",
                item_id,
                "--target-layer",
                "story_bible",
                "--reason",
                "采纳为角色设定",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "context-history",
                "--context-db",
                str(context_db),
                "--item-id",
                item_id,
            ]
        )
        == 0
    )


def test_context_cli_schema_index_search_and_audit(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    transcript = tmp_path / "conversation.jsonl"
    write_jsonl(
        transcript,
        [
            {
                "turn_id": "turn_001",
                "role": "user",
                "content": "记住：张鹏不是刘培强的法定监护人，他是训练教官。",
            }
        ],
    )
    context_db = tmp_path / "context.sqlite"
    schema_dir = tmp_path / "schemas"
    chroma_dir = tmp_path / "context_chroma"

    assert (
        main(
            [
                "context-ingest-conversation",
                str(transcript),
                "--context-db",
                str(context_db),
                "--project-id",
                "we2",
                "--reset-context-store",
            ]
        )
        == 0
    )
    assert main(["export-context-json-schemas", "--output-dir", str(schema_dir)]) == 0
    assert (schema_dir / "CreativeContextPacket.schema.json").is_file()
    assert (
        main(
            [
                "build-context-index",
                "--context-db",
                str(context_db),
                "--persist-dir",
                str(chroma_dir),
                "--project-id",
                "we2",
                "--embedding-dim",
                "64",
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "search-context-index",
                "--context-db",
                str(context_db),
                "--persist-dir",
                str(chroma_dir),
                "--project-id",
                "we2",
                "--query",
                "训练教官",
                "--embedding-dim",
                "64",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "context-export-external-entity-links",
                "--context-db",
                str(context_db),
                "--project-id",
                "we2",
            ]
        )
        == 0
    )
