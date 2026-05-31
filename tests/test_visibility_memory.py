from pathlib import Path

from dms.llm import FakeSceneEventClient, FakeVisibilityNotesClient
from dms.memory import build_visibility_memory
from dms.runners import SceneEventRunConfig, VisibilityNotesRunConfig, run_scene_events, run_visibility_notes


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_visibility_memory_from_fake_run(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    run_scene_events(
        SceneEventRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=event_dir,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeSceneEventClient(),
    )

    visibility_dir = tmp_path / "visibility"
    memory_dir = tmp_path / "visibility_memory"
    run_visibility_notes(
        VisibilityNotesRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=visibility_dir,
            extracted_candidates_dir=event_dir,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeVisibilityNotesClient(),
    )

    summary = build_visibility_memory(visibility_dir, memory_dir)

    assert summary["accepted_scene_count"] == 2
    assert summary["skipped_scene_count"] == 0
    assert summary["visibility_record_count"] == 2
    assert summary["hidden_or_future_sensitive_count"] == 2
    assert (memory_dir / "visibility_records.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "hidden_or_future_sensitive_items.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "summary.json").is_file()
