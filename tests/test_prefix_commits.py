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
from dms.memory import build_prefix_commits
from dms.runners import SceneOrderedPipelineConfig, run_scene_ordered_pipeline


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_prefix_commits_replays_scene_updates(tmp_path: Path) -> None:
    run_root = tmp_path / "ordered"
    summary = run_scene_ordered_pipeline(
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

    commit_summary = summary["memory_summaries"]["prefix_commits"]
    commits_dir = run_root / "prefix_commits"
    commits = _read_jsonl(commits_dir / "commits.jsonl")
    operations = _read_jsonl(commits_dir / "operations.jsonl")
    first_snapshot = json.loads((commits_dir / "snapshots" / "prefix_after_scene_0001.json").read_text(encoding="utf-8"))
    current = json.loads((commits_dir / "current_snapshot.json").read_text(encoding="utf-8"))

    assert commit_summary["commit_count"] == 2
    assert commit_summary["operation_count"] == len(operations)
    assert commits[0]["parent_unit_id"] == "scene_0001"
    assert commits[1]["parent_unit_id"] == "scene_0002"
    assert first_snapshot["counts"]["memory_count"] == 1
    assert current["after_parent_unit_id"] == "scene_0002"
    assert current["counts"]["memory_count"] == 2
    assert current["counts"]["entity_memory_link_count"] == 6
    assert current["counts"]["relationship_count"] == 1
    assert "object_0003" in current["entity_memory_index"]
    assert any(operation["operation_type"] == "entity_created" for operation in operations)
    assert any(operation["operation_type"] == "entity_memory_linked" for operation in operations)
    assert any(operation["operation_type"] == "relationship_created" for operation in operations)


def test_build_prefix_commits_can_run_standalone(tmp_path: Path) -> None:
    run_root = tmp_path / "ordered"
    standalone = tmp_path / "standalone_commits"
    run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=run_root,
            limit=1,
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

    summary = build_prefix_commits(
        run_root / "_debug" / "intermediate" / "world_model",
        run_root / "knowledge_graph",
        standalone,
    )

    assert summary["commit_count"] == 1
    assert (standalone / "current_snapshot.json").is_file()
    assert (standalone / "operations.jsonl").is_file()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
