import json
from pathlib import Path

from dms.cli import main


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_cli_run_scene_ordered_pipeline_fake_provider(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered"
    code = main(
        [
            "run-scene-ordered-pipeline",
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--limit",
            "2",
            "--provider",
            "fake",
            "--scene-task-concurrency",
            "3",
            "--max-chunk-units",
            "800",
        ]
    )

    assert code == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["task_order"]["cross_unit_order"] == "sequential"
    assert summary["task_order"]["per_unit_parallel"] == ["scene_summary", "scene_inventory", "kg_entity_mentions"]
    assert summary["chunk_count"] == 2
    assert (output_root / "_debug" / "chunk_manifest.jsonl").is_file()
    assert summary["task_summaries"]["scene_summary"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["episodic_memories"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["kg_entity_refinement"]["parsed_output_count"] == 2
    assert summary["memory_summaries"]["world_model"]["episodic_memory_count"] == 2
    assert summary["memory_summaries"]["world_model"]["scene_summary_count"] == 2
    assert (output_root / "README.md").is_file()
    assert (output_root / "summaries" / "scene_summaries.jsonl").is_file()
    assert (output_root / "knowledge_graph" / "entities.jsonl").is_file()
    assert (output_root / "memories" / "episodic_memories.jsonl").is_file()
    assert (output_root / "prefix_commits" / "current_snapshot.json").is_file()
    assert summary["memory_summaries"]["prefix_commits"]["commit_count"] == 2
    assert (output_root / "_debug" / "intermediate" / "world_model" / "prefix_world_model.json").is_file()


def test_cli_run_scene_summary_fake_provider(tmp_path: Path) -> None:
    output_dir = tmp_path / "summary_run"
    memory_dir = tmp_path / "summary_memory"
    code = main(
        [
            "run-scene-summary",
            str(SCRIPT_PATH),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--provider",
            "fake",
            "--no-dry-run",
        ]
    )

    assert code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["parsed_output_count"] == 2

    code = main(
        [
            "build-scene-summary-memory",
            str(output_dir),
            "--output-dir",
            str(memory_dir),
        ]
    )

    assert code == 0
    memory_summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    assert memory_summary["unit_summary_count"] == 2
    assert memory_summary["scene_summary_count"] == 2
    assert (memory_dir / "unit_summaries.jsonl").is_file()
    assert (memory_dir / "scene_summaries.jsonl").is_file()


def test_cli_run_scene_ordered_pipeline_extends_base_run(tmp_path: Path) -> None:
    base_root = tmp_path / "ordered_base"
    extended_root = tmp_path / "ordered_extended"
    main(
        [
            "run-scene-ordered-pipeline",
            str(SCRIPT_PATH),
            "--output-root",
            str(base_root),
            "--limit",
            "1",
            "--provider",
            "fake",
        ]
    )

    code = main(
        [
            "run-scene-ordered-pipeline",
            str(SCRIPT_PATH),
            "--base-output-root",
            str(base_root),
            "--output-root",
            str(extended_root),
            "--start",
            "2",
            "--limit",
            "1",
            "--provider",
            "fake",
        ]
    )

    assert code == 0
    summary = json.loads((extended_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_count"] == 1
    assert summary["chunk_count"] == 1
    assert summary["memory_summaries"]["world_model"]["scene_count"] == 2
    assert summary["memory_summaries"]["world_model"]["scene_summary_count"] == 2
    assert summary["memory_summaries"]["prefix_commits"]["commit_count"] == 2
    assert (extended_root / "_debug" / "extractions" / "kg_entity_mentions" / "parsed" / "scene_0001.json").is_file()
    assert (extended_root / "_debug" / "extractions" / "kg_entity_mentions" / "parsed" / "scene_0002.json").is_file()
    assert (extended_root / "_debug" / "extractions" / "kg_entity_refinement" / "parsed" / "scene_0002.json").is_file()
    unit_trace = [
        json.loads(line)
        for line in (extended_root / "_debug" / "unit_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["scene_id"] for record in unit_trace] == ["scene_0001", "scene_0002"]


def test_cli_build_episodic_memory_from_ordered_run(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered"
    memory_dir = tmp_path / "episodic_memory_copy"
    main(
        [
            "run-scene-ordered-pipeline",
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--limit",
            "1",
            "--provider",
            "fake",
        ]
    )
    code = main(
        [
            "build-episodic-memory",
            str(output_root / "_debug" / "extractions" / "episodic_memories"),
            "--output-dir",
            str(memory_dir),
        ]
    )

    assert code == 0
    summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodic_memory_count"] == 1
    assert summary["entity_memory_link_count"] == 3


def test_cli_build_prefix_commits_from_ordered_run(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered"
    commits_dir = tmp_path / "prefix_commits_copy"
    main(
        [
            "run-scene-ordered-pipeline",
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--limit",
            "1",
            "--provider",
            "fake",
        ]
    )
    code = main(
        [
            "build-prefix-commits",
            str(output_root / "_debug" / "intermediate" / "world_model"),
            "--entity-resolution-dir",
            str(output_root / "knowledge_graph"),
            "--output-dir",
            str(commits_dir),
        ]
    )

    assert code == 0
    summary = json.loads((commits_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["commit_count"] == 1
    assert (commits_dir / "commits.jsonl").is_file()
    assert (commits_dir / "current_snapshot.json").is_file()
