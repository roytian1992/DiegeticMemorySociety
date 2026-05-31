from pathlib import Path

from dms.llm import FakeSceneInventoryClient
from dms.runners import SceneInventoryRunConfig, run_scene_inventory


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_scene_inventory_dry_run_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "scene_inventory"
    summary = run_scene_inventory(
        SceneInventoryRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
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
    assert "Fill the inventory JSON" in prompt_text
    assert "scene_0001" in prompt_text


def test_scene_inventory_refuses_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("existing", encoding="utf-8")

    try:
        run_scene_inventory(SceneInventoryRunConfig(script_path=SCRIPT_PATH, output_dir=output_dir))
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("Expected non-empty output dir to fail without overwrite")


def test_scene_inventory_fake_llm_run_parses_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_inventory"
    summary = run_scene_inventory(
        SceneInventoryRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            start=1,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeSceneInventoryClient(),
    )

    assert summary["status"] == "complete"
    assert summary["selected_count"] == 2
    assert summary["llm_completed_count"] == 2
    assert summary["parsed_output_count"] == 2
    assert summary["failed_count"] == 0

    parsed = (output_dir / "parsed" / "scene_0001.json").read_text(encoding="utf-8")
    assert '"status": "parsed"' in parsed
    assert '"scene_id": "scene_0001"' in parsed
    assert '"unit_id": "scene_0001"' in parsed


def test_scene_inventory_requires_llm_client_for_non_dry_run(tmp_path: Path) -> None:
    try:
        run_scene_inventory(
            SceneInventoryRunConfig(
                script_path=SCRIPT_PATH,
                output_dir=tmp_path / "missing_client",
                dry_run=False,
            )
        )
    except ValueError as exc:
        assert "llm_client" in str(exc)
    else:
        raise AssertionError("Expected non-dry-run without client to fail")
