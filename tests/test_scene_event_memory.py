from pathlib import Path

from dms.llm import FakeSceneEventClient
from dms.memory import build_scene_event_memory
from dms.runners import SceneEventRunConfig, run_scene_events


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_scene_event_memory_from_fake_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    memory_dir = tmp_path / "memory"
    run_scene_events(
        SceneEventRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=run_dir,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeSceneEventClient(),
    )

    summary = build_scene_event_memory(run_dir, memory_dir)

    assert summary["accepted_scene_count"] == 2
    assert summary["skipped_scene_count"] == 0
    assert summary["event_count"] == 2
    assert summary["knowledge_transfer_count"] == 2
    assert summary["state_change_count"] == 2
    assert summary["thread_candidate_count"] == 2
    assert (memory_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "knowledge_transfers.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "summary.json").is_file()
