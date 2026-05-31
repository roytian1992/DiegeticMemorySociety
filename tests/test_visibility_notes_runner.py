from pathlib import Path

from dms.llm import FakeSceneEventClient, FakeVisibilityNotesClient
from dms.runners import SceneEventRunConfig, VisibilityNotesRunConfig, run_scene_events, run_visibility_notes


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def _event_candidates_dir(tmp_path: Path) -> Path:
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
    return event_dir


def test_visibility_notes_dry_run_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "visibility"
    summary = run_visibility_notes(
        VisibilityNotesRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            extracted_candidates_dir=_event_candidates_dir(tmp_path),
            start=1,
            limit=2,
        )
    )

    assert summary["status"] == "dry_run_complete"
    assert summary["selected_count"] == 2
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "trace.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "inputs" / "scene_0001.json").is_file()
    assert (output_dir / "prompts" / "scene_0001.txt").is_file()
    assert (output_dir / "raw_outputs" / "scene_0001.json").is_file()
    assert (output_dir / "parsed" / "scene_0001.json").is_file()

    prompt_text = (output_dir / "prompts" / "scene_0001.txt").read_text(encoding="utf-8")
    assert "Fill the visibility JSON" in prompt_text
    assert "FAKE_EVENT" in prompt_text


def test_visibility_notes_fake_llm_run_parses_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_visibility"
    summary = run_visibility_notes(
        VisibilityNotesRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            extracted_candidates_dir=_event_candidates_dir(tmp_path),
            start=1,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeVisibilityNotesClient(),
    )

    assert summary["status"] == "complete"
    assert summary["selected_count"] == 2
    assert summary["llm_completed_count"] == 2
    assert summary["parsed_output_count"] == 2
    assert summary["failed_count"] == 0

    parsed = (output_dir / "parsed" / "scene_0001.json").read_text(encoding="utf-8")
    assert '"status": "parsed"' in parsed
    assert '"FAKE_HIDDEN_ITEM"' in parsed
