import json
from pathlib import Path

from dms.llm import (
    FakeDurableRelationshipClient,
    FakeEpisodicMemoryClient,
    FakeKGEntityMentionClient,
    FakeKGEntityRefinementClient,
    FakeSceneInventoryClient,
    FakeSceneSummaryClient,
)
from dms.memory import build_episodic_memory
from dms.runners import SceneOrderedPipelineConfig, run_scene_ordered_pipeline


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_episodic_memory_from_scene_ordered_run(tmp_path: Path) -> None:
    run_root = tmp_path / "ordered"
    memory_dir = tmp_path / "episodic_memory"
    run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=run_root,
            limit=2,
        ),
        llm_clients={
            "scene_summary": FakeSceneSummaryClient(),
            "scene_inventory": FakeSceneInventoryClient(),
            "kg_entity_mentions": FakeKGEntityMentionClient(),
            "kg_entity_refinement": FakeKGEntityRefinementClient(),
            "episodic_memories": FakeEpisodicMemoryClient(),
            "durable_relationships": FakeDurableRelationshipClient(),
        },
    )

    summary = build_episodic_memory(run_root / "_debug" / "extractions" / "episodic_memories", memory_dir)

    assert summary["episodic_memory_count"] == 2
    assert summary["entity_memory_link_count"] == 6
    assert (memory_dir / "episodic_memories.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "entity_memory_links.jsonl").read_text(encoding="utf-8").count("\n") == 6


def test_build_episodic_memory_records_evidence_offsets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        """
        {
          "unit_id": "scene_0001",
          "title": "1、INT.日.房间",
          "subtitle": "",
          "content": "刘培强进入房间。"
        }
        """,
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
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
                    "evidence": "刘培强"
                  }
                ]
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    build_episodic_memory(run_dir, memory_dir)
    memory = (memory_dir / "episodic_memories.jsonl").read_text(encoding="utf-8")
    link = (memory_dir / "entity_memory_links.jsonl").read_text(encoding="utf-8")

    assert '"evidence_exact_match": true' in memory
    assert '"evidence_source_field": "content"' in memory
    assert '"evidence_start": 0' in memory
    assert '"evidence_end": 7' in memory
    assert '"evidence_exact_match": true' in link


def test_build_episodic_memory_infers_temporal_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "unit_id": "scene_0001",
                "title": "1、INT.日.数字生命研究室",
                "subtitle": "",
                "content": "印度科学家说，人本质上就是一堆电信号。刘培强进入房间。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "data": {
                    "unit_id": "scene_0001",
                    "episodic_memories": [
                        {
                            "memory_id_hint": "m1",
                            "sequence_index": 1,
                            "timeline_label": "scene_0001",
                            "memory_type": "observation",
                            "summary": "人本质上就是一堆电信号",
                            "evidence": "人本质上就是一堆电信号",
                            "entity_links": [],
                        },
                        {
                            "memory_id_hint": "m2",
                            "sequence_index": 2,
                            "timeline_label": "scene_0001",
                            "memory_type": "action",
                            "summary": "刘培强进入房间",
                            "evidence": "刘培强进入房间",
                            "entity_links": [],
                        },
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = build_episodic_memory(run_dir, memory_dir)
    records = _read_jsonl(memory_dir / "episodic_memories.jsonl")

    assert summary["memory_temporal_scope_counts"] == {"atemporal_fact": 1, "temporal_episode": 1}
    assert records[0]["memory_temporal_scope"] == "atemporal_fact"
    assert records[1]["memory_temporal_scope"] == "temporal_episode"


def test_build_episodic_memory_records_parent_chunk_offsets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001_chunk_002.json").write_text(
        json.dumps(
            {
                "unit": {
                    "unit_id": "scene_0001_chunk_002",
                    "parent_unit_id": "scene_0001",
                    "chunk_id": "scene_0001_chunk_002",
                    "chunk_index": 2,
                    "chunk_count": 3,
                    "title": "1、INT.日.房间",
                    "subtitle": "",
                    "content": "乙进入房间。丙离开。",
                    "source_span": {
                        "parent_unit_id": "scene_0001",
                        "source_start": 100,
                        "source_end": 110,
                        "source_sha256": "parent-sha",
                        "chunk_unit_count": 10,
                        "max_chunk_units": 800,
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001_chunk_002.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "unit_id": "scene_0001_chunk_002",
                "data": {
                    "unit_id": "scene_0001_chunk_002",
                    "episodic_memories": [
                        {
                            "memory_id_hint": "m1",
                            "sequence_index": 1,
                            "timeline_label": "scene_0001_chunk_002",
                            "memory_type": "action",
                            "summary": "乙进入房间",
                            "evidence": "乙进入房间",
                            "entity_links": [
                                {
                                    "entity": "乙",
                                    "entity_type": "character",
                                    "link_role": "actor",
                                    "evidence": "乙",
                                }
                            ],
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build_episodic_memory(run_dir, memory_dir)
    memory = json.loads((memory_dir / "episodic_memories.jsonl").read_text(encoding="utf-8").strip())
    link = json.loads((memory_dir / "entity_memory_links.jsonl").read_text(encoding="utf-8").strip())

    assert memory["parent_unit_id"] == "scene_0001"
    assert memory["chunk_id"] == "scene_0001_chunk_002"
    assert memory["chunk_index"] == 2
    assert memory["unit_source_start"] == 100
    assert memory["parent_evidence_start"] == 100
    assert memory["parent_evidence_end"] == 105
    assert memory["parent_source_sha256"] == "parent-sha"
    assert link["parent_unit_id"] == "scene_0001"
    assert link["parent_evidence_start"] == 100


def test_build_episodic_memory_rewrites_fuzzy_evidence_to_source_span(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        """
        {
          "unit_id": "scene_0001",
          "title": "1、INT.日.房间",
          "subtitle": "",
          "content": "印度科学家（印度式英语）：人，本质上就是一堆电信号。"
        }
        """,
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0001",
            "episodic_memories": [
              {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "dialogue",
                "summary": "印度科学家称人是电信号",
                "evidence": "印度科学家(印度式英语):人，本质上就是一堆电信号",
                "entity_links": []
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    build_episodic_memory(run_dir, memory_dir)
    memory = (memory_dir / "episodic_memories.jsonl").read_text(encoding="utf-8")

    assert '"evidence_verification_status": "fuzzy_aligned"' in memory
    assert '"evidence": "印度科学家（印度式英语）：人，本质上就是一堆电信号"' in memory
    assert '"model_evidence": "印度科学家(印度式英语):人，本质上就是一堆电信号"' in memory


def test_build_episodic_memory_skips_unaligned_entity_links(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        """
        {
          "unit_id": "scene_0001",
          "title": "",
          "subtitle": "",
          "content": "记忆存在这儿。"
        }
        """,
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0001",
            "episodic_memories": [
              {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "dialogue",
                "summary": "确认记忆存在这儿",
                "evidence": "记忆存在这儿。",
                "entity_links": [
                  {
                    "entity": "印度科学家",
                    "entity_type": "character",
                    "link_role": "speaker",
                    "evidence": "印度科学家（印度式英语）：是的，存在这儿。"
                  },
                  {
                    "entity": "记忆",
                    "entity_type": "concept",
                    "link_role": "concept",
                    "evidence": "记忆存在这儿"
                  }
                ]
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_episodic_memory(run_dir, memory_dir)
    links = (memory_dir / "entity_memory_links.jsonl").read_text(encoding="utf-8")

    assert summary["episodic_memory_count"] == 1
    assert summary["entity_memory_link_count"] == 1
    assert summary["skipped_entity_memory_link_count"] == 1
    assert summary["rejected_evidence_count"] == 1
    assert "印度科学家" not in links
    assert "记忆存在这儿" in links


def test_build_episodic_memory_skips_non_trackable_entity_links(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        """
        {
          "unit": {
            "unit_id": "scene_0001",
            "title": "",
            "subtitle": "",
            "content": "记忆存在脑子里。"
          },
          "extracted_candidates": {
            "kg_entity_mentions": {
              "status": "parsed",
              "data": {
                "unit_id": "scene_0001",
                "entity_mentions": [
                  {
                    "surface": "记忆",
                    "entity_type": "concept",
                    "canonical_hint": "",
                    "role_in_unit": "concept",
                    "attributes_or_state": "",
                    "evidence": "记忆"
                  }
                ],
                "unresolved_mentions": []
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0001",
            "episodic_memories": [
              {
                "memory_id_hint": "m1",
                "sequence_index": 1,
                "timeline_label": "scene_0001",
                "memory_type": "dialogue",
                "summary": "确认记忆存在脑子里",
                "evidence": "记忆存在脑子里",
                "entity_links": [
                  {
                    "entity": "脑子",
                    "entity_type": "concept",
                    "link_role": "concept",
                    "evidence": "脑子"
                  },
                  {
                    "entity": "记忆",
                    "entity_type": "concept",
                    "link_role": "concept",
                    "evidence": "记忆"
                  }
                ]
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_episodic_memory(run_dir, memory_dir, require_entity_candidates=True)
    links = (memory_dir / "entity_memory_links.jsonl").read_text(encoding="utf-8")

    assert summary["entity_memory_link_count"] == 1
    assert summary["skipped_non_trackable_entity_memory_link_count"] == 1
    assert '"entity": "记忆"' in links
    assert "脑子" not in links


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
