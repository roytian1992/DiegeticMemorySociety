from pathlib import Path

from dms.llm import FakeTemporalExtractionClient, LLMResult
from dms.runners import TemporalExtractionRunConfig, run_temporal_extraction


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_temporal_extraction_dry_run_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "temporal"
    summary = run_temporal_extraction(
        TemporalExtractionRunConfig(
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
    assert (output_dir / "timeline_graph.json").is_file()
    assert (output_dir / "timeline_report.md").is_file()
    assert (output_dir / "temporal_audit.json").is_file()
    assert (output_dir / "temporal_audit_report.md").is_file()

    prompt_text = (output_dir / "prompts" / "scene_0001.txt").read_text(encoding="utf-8")
    assert "Extract a diegetic temporal layer" in prompt_text
    assert "scene_0001" in prompt_text


def test_temporal_extraction_fake_llm_builds_timeline_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_temporal"
    summary = run_temporal_extraction(
        TemporalExtractionRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            start=1,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeTemporalExtractionClient(),
    )

    assert summary["status"] == "complete"
    assert summary["selected_count"] == 2
    assert summary["llm_completed_count"] == 2
    assert summary["parsed_output_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["timeline_counts"]["event_count"] == 2
    assert summary["timeline_counts"]["timeline_bucket_count"] == 2
    assert summary["temporal_audit_counts"]["evidence_rejected_count"] == 0
    assert summary["temporal_audit_counts"]["hard_ordering_unusable_count"] == 0
    assert (output_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "timeline_order.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert "测试时间事件 scene_0001" in (output_dir / "timeline_report.md").read_text(encoding="utf-8")
    assert "No temporal evidence audit issues" in (output_dir / "temporal_audit_report.md").read_text(encoding="utf-8")


def test_temporal_extraction_requires_llm_client_for_non_dry_run(tmp_path: Path) -> None:
    try:
        run_temporal_extraction(
            TemporalExtractionRunConfig(
                script_path=SCRIPT_PATH,
                output_dir=tmp_path / "missing_client",
                dry_run=False,
            )
        )
    except ValueError as exc:
        assert "llm_client" in str(exc)
    else:
        raise AssertionError("Expected non-dry-run without client to fail")


class _TruncatedTemporalClient:
    provider = "fake"
    model = "fake-truncated-temporal"

    def complete(self, prompt: str) -> LLMResult:
        text = (
            '{"unit_id":"scene_0001","temporal_events":[],"temporal_relations":[],'
            '"scene_temporal_index":{"absolute_time_hints":["距太阳氦闪还剩 3'
        )
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"choices": [{"finish_reason": "length"}]},
            usage={"completion_tokens": 1200},
        )


def test_temporal_extraction_marks_length_finish_as_truncated(tmp_path: Path) -> None:
    output_dir = tmp_path / "truncated_temporal"
    summary = run_temporal_extraction(
        TemporalExtractionRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=output_dir,
            start=1,
            limit=1,
            dry_run=False,
        ),
        llm_client=_TruncatedTemporalClient(),
    )

    assert summary["llm_completed_count"] == 1
    assert summary["parsed_output_count"] == 0
    assert summary["failed_count"] == 1
    parsed_text = (output_dir / "parsed" / "scene_0001.json").read_text(encoding="utf-8")
    trace_text = (output_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert '"status": "truncated_output"' in parsed_text
    assert '"status": "truncated_output"' in trace_text
