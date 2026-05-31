from pathlib import Path

from dms.llm import FakeSceneSummaryClient
from dms.runners import SceneSummaryRunConfig, run_scene_summary


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_scene_summary_dry_run_writes_chunked_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "scene_summary"
    summary = run_scene_summary(
        SceneSummaryRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            start=1,
            limit=1,
            max_chunk_units=20,
        )
    )

    assert summary["status"] == "dry_run_complete"
    assert summary["selected_count"] == 1
    assert summary["chunk_count"] > 1
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "trace.jsonl").read_text(encoding="utf-8").count("\n") == summary["chunk_count"]
    assert (output_dir / "inputs" / "scene_0001_chunk_001.json").is_file()
    assert (output_dir / "prompts" / "scene_0001_chunk_001.txt").is_file()

    prompt_text = (output_dir / "prompts" / "scene_0001_chunk_001.txt").read_text(encoding="utf-8")
    assert "Write a concise summary record" in prompt_text
    assert "scene_0001_chunk_001" in prompt_text


def test_scene_summary_fake_llm_run_parses_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_summary"
    summary = run_scene_summary(
        SceneSummaryRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            start=1,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeSceneSummaryClient(),
    )

    assert summary["status"] == "complete"
    assert summary["selected_count"] == 2
    assert summary["chunk_count"] == 2
    assert summary["llm_completed_count"] == 2
    assert summary["parsed_output_count"] == 2
    assert summary["failed_count"] == 0

    parsed = (output_dir / "parsed" / "scene_0001.json").read_text(encoding="utf-8")
    assert '"status": "parsed"' in parsed
    assert '"summary": "FAKE_SUMMARY scene_0001"' in parsed


def test_scene_summary_requires_llm_client_for_non_dry_run(tmp_path: Path) -> None:
    try:
        run_scene_summary(
            SceneSummaryRunConfig(
                script_path=SCRIPT_PATH,
                output_dir=tmp_path / "missing_client",
                dry_run=False,
            )
        )
    except ValueError as exc:
        assert "llm_client" in str(exc)
    else:
        raise AssertionError("Expected non-dry-run without client to fail")
