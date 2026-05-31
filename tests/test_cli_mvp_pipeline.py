import json
from pathlib import Path

from dms.cli import main


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_cli_run_mvp_pipeline_fake_provider(tmp_path: Path) -> None:
    output_root = tmp_path / "mvp"
    code = main(
        [
            "run-mvp-pipeline",
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--limit",
            "2",
            "--provider",
            "fake",
        ]
    )

    assert code == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["extraction"]["parsed_output_count"] == 2
    assert summary["canonical_memory"]["character_count"] == 0
    assert (output_root / "canonical_memory" / "canonical_memory.json").is_file()


def test_cli_run_scene_events_and_build_memory_fake_provider(tmp_path: Path) -> None:
    run_dir = tmp_path / "scene_events"
    memory_dir = tmp_path / "event_memory"

    run_code = main(
        [
            "run-scene-events",
            str(SCRIPT_PATH),
            "--output-dir",
            str(run_dir),
            "--limit",
            "2",
            "--no-dry-run",
            "--provider",
            "fake",
        ]
    )
    build_code = main(["build-scene-event-memory", str(run_dir), "--output-dir", str(memory_dir)])

    assert run_code == 0
    assert build_code == 0
    summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["event_count"] == 2
    assert summary["knowledge_transfer_count"] == 2


def test_cli_run_kg_entities_and_build_memory_fake_provider(tmp_path: Path) -> None:
    run_dir = tmp_path / "kg_entities"
    memory_dir = tmp_path / "kg_entity_memory"

    run_code = main(
        [
            "run-kg-entities",
            str(SCRIPT_PATH),
            "--output-dir",
            str(run_dir),
            "--limit",
            "2",
            "--no-dry-run",
            "--provider",
            "fake",
        ]
    )
    build_code = main(["build-kg-entity-memory", str(run_dir), "--output-dir", str(memory_dir)])

    assert run_code == 0
    assert build_code == 0
    summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["entity_mention_count"] == 8
    assert summary["unresolved_mention_count"] == 2
    assert summary["entity_type_counts"]["concept"] == 2
    assert summary["entity_type_counts"]["occasion"] == 2


def test_cli_run_visibility_notes_and_build_memory_fake_provider(tmp_path: Path) -> None:
    event_dir = tmp_path / "scene_events"
    visibility_dir = tmp_path / "visibility_notes"
    memory_dir = tmp_path / "visibility_memory"

    event_code = main(
        [
            "run-scene-events",
            str(SCRIPT_PATH),
            "--output-dir",
            str(event_dir),
            "--limit",
            "2",
            "--no-dry-run",
            "--provider",
            "fake",
        ]
    )
    visibility_code = main(
        [
            "run-visibility-notes",
            str(SCRIPT_PATH),
            "--output-dir",
            str(visibility_dir),
            "--extracted-candidates-dir",
            str(event_dir),
            "--limit",
            "2",
            "--no-dry-run",
            "--provider",
            "fake",
        ]
    )
    build_code = main(["build-visibility-memory", str(visibility_dir), "--output-dir", str(memory_dir)])

    assert event_code == 0
    assert visibility_code == 0
    assert build_code == 0
    summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["visibility_record_count"] == 2
    assert summary["hidden_or_future_sensitive_count"] == 2
