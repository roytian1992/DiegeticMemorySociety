import json
from pathlib import Path

from dms.memory import build_scene_summary_memory


def test_build_scene_summary_memory_writes_retrieval_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "scene_summary_run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    output_dir = tmp_path / "summaries"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "unit_id": "scene_0001",
                "parent_unit_id": "scene_0001",
                "chunk_id": "scene_0001",
                "chunk_index": 1,
                "chunk_count": 1,
                "title": "1、INT.日.机房",
                "content": "刘培强进入机房。",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "data": {
                    "unit_id": "scene_0001",
                    "summary": "刘培强进入机房。",
                    "salient_points": ["刘培强到达机房"],
                    "continuity_hooks": ["机房后续可能继续承载行动"],
                    "retrieval_text": "刘培强 机房 行动",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_scene_summary_memory(run_dir, output_dir)
    records = [
        json.loads(line)
        for line in (output_dir / "scene_summaries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["scene_summary_count"] == 1
    assert summary["salient_point_count"] == 1
    assert summary["continuity_hook_count"] == 1
    assert records[0]["memory_layer"] == "scene_summary"
    assert records[0]["record_id"] == "scene_0001_summary"
    assert records[0]["retrieval_text"] == "刘培强 机房 行动"
