from pathlib import Path

from dms.llm import FakeSceneInventoryClient
from dms.memory import build_scene_inventory_memory
from dms.runners import SceneInventoryRunConfig, run_scene_inventory


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_scene_inventory_memory_from_fake_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    memory_dir = tmp_path / "memory"
    run_scene_inventory(
        SceneInventoryRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=run_dir,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeSceneInventoryClient(),
    )

    summary = build_scene_inventory_memory(run_dir, memory_dir)

    assert summary["accepted_scene_count"] == 2
    assert summary["skipped_scene_count"] == 0
    assert summary["character_count"] == 0
    assert (memory_dir / "scenes.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "characters.jsonl").read_text(encoding="utf-8").count("\n") == 0
    assert (memory_dir / "summary.json").is_file()
