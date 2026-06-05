import json
from pathlib import Path

from dms.memory import MemoryTimelineIndexConfig, build_memory_timeline_index


def test_build_memory_timeline_index_enriches_episodic_memories(tmp_path: Path) -> None:
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir()
    _write_jsonl(
        memories_dir / "episodic_memories.jsonl",
        [
            {
                "record_id": "scene_0002_memory_001",
                "scene_id": "scene_0002",
                "timeline_index": "scene_0002:001",
                "sequence_index": 1,
                "summary": "角色A启动发动机",
                "evidence_text": "角色A启动发动机",
            },
            {
                "record_id": "scene_0003_memory_001",
                "scene_id": "scene_0003",
                "timeline_index": "scene_0003:001",
                "sequence_index": 1,
                "summary": "角色A回忆三年前事故",
                "evidence_text": "三年前事故",
            },
            {
                "record_id": "scene_0004_memory_001",
                "scene_id": "scene_0004",
                "timeline_index": "scene_0004:001",
                "sequence_index": 1,
                "summary": "没有对应时间事件",
                "evidence_text": "没有对应时间事件",
            },
            {
                "record_id": "scene_0004_memory_002",
                "scene_id": "scene_0004",
                "timeline_index": "scene_0004:002",
                "sequence_index": 2,
                "summary": "数字生命技术是一种延续文明的技术",
                "evidence_text": "数字生命技术是一种延续文明的技术",
                "memory_temporal_scope": "atemporal_fact",
            },
        ],
    )
    timeline_dir = tmp_path / "timeline"
    timeline_dir.mkdir()
    (timeline_dir / "timeline_graph.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": "scene_0002:event_001",
                        "scene_id": "scene_0002",
                        "source_record_id": 2,
                        "summary": "角色A启动发动机",
                        "evidence": "角色A启动发动机",
                        "event_time_mode": "present_scene",
                        "story_time_hint": "current scene",
                        "granularity": "scene_relative",
                        "confidence": 0.9,
                        "revealed_at_scene_id": "scene_0002",
                        "revealed_at_source_record_id": 2,
                    },
                    {
                        "event_id": "scene_0003:event_001",
                        "scene_id": "scene_0003",
                        "source_record_id": 3,
                        "summary": "角色A回忆三年前事故",
                        "evidence": "三年前事故",
                        "event_time_mode": "past_recalled",
                        "story_time_hint": "三年前",
                        "granularity": "year",
                        "confidence": 0.95,
                        "revealed_at_scene_id": "scene_0003",
                        "revealed_at_source_record_id": 3,
                    },
                ],
                "relations": [
                    {
                        "relation_id": "scene_0003:rel_001",
                        "source_event_id": "scene_0003:event_001",
                        "target_event_id": "scene_0002:event_001",
                        "relation_type": "before",
                    }
                ],
                "timeline_order": [
                    {"timeline_rank": 1, "event_id": "scene_0003:event_001"},
                    {"timeline_rank": 2, "event_id": "scene_0002:event_001"},
                ],
                "timeline_buckets": [
                    {"timeline_bucket": "T001", "event_ids": ["scene_0003:event_001"]},
                    {"timeline_bucket": "T002", "event_ids": ["scene_0002:event_001"]},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_memory_timeline_index(
        MemoryTimelineIndexConfig(
            memory_path=memories_dir,
            timeline_graph_path=timeline_dir,
            output_dir=tmp_path / "indexed",
            overwrite=True,
        )
    )

    assert summary["counts"]["memory_count"] == 4
    assert summary["counts"]["matched_memory_count"] == 2
    assert summary["counts"]["unmatched_memory_count"] == 1
    assert summary["counts"]["not_story_time_bound_memory_count"] == 1
    assert summary["memory_temporal_scope_counts"]["atemporal_fact"] == 1
    enriched = _read_jsonl(tmp_path / "indexed" / "enriched_episodic_memories.jsonl")
    by_id = {item["record_id"]: item for item in enriched}
    assert by_id["scene_0002_memory_001"]["discourse_timeline_index"] == "scene_0002:001"
    assert by_id["scene_0002_memory_001"]["timeline_index_semantics"] == "discourse_scene_sequence"
    assert by_id["scene_0002_memory_001"]["story_time_bucket"] == "T002"
    assert by_id["scene_0002_memory_001"]["story_time_rank"] == 2
    assert by_id["scene_0003_memory_001"]["story_time_bucket"] == "T001"
    assert by_id["scene_0003_memory_001"]["revealed_at_order"] == 3
    assert by_id["scene_0004_memory_001"]["memory_timeline_index_status"] == "no_scene_temporal_event"
    assert by_id["scene_0004_memory_002"]["memory_timeline_index_status"] == "scope_not_story_time_bound"
    assert by_id["scene_0004_memory_002"]["story_time_index"] is None
    assert by_id["scene_0004_memory_002"]["revealed_at_order"] == 4

    story_sorted = _read_jsonl(tmp_path / "indexed" / "story_time_memory_index.jsonl")
    assert [item["record_id"] for item in story_sorted[:2]] == ["scene_0003_memory_001", "scene_0002_memory_001"]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
