from __future__ import annotations

import json

from dms.ui.gradio_app import _flatten_target_result, _overview


def test_overview_reads_benchmark_metrics(tmp_path) -> None:
    root = tmp_path / "bench"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"completed_count": 1}), encoding="utf-8")
    row = {
        "scene_id": "scene_0006",
        "title": "6、返航",
        "status": "complete",
        "counts": {"retrieved_entities": 2, "retrieved_memories": 4},
        "metrics": {
            "generated": {
                "writing_intent_consistency": 0.8,
                "writing_quality": 1.0,
                "memory_faithfulness": 1.0,
                "overall": 0.9333,
            },
            "reference": {"overall": 0.9},
            "deltas": {"writing_intent_consistency": -0.1, "overall": 0.0333},
        },
    }
    (root / "metrics.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    table, summary = _overview(str(root))

    assert summary["completed_count"] == 1
    assert table.iloc[0]["scene_id"] == "scene_0006"
    assert table.iloc[0]["delta_intent"] == -0.1


def test_flatten_target_result_for_ui_table() -> None:
    flattened = _flatten_target_result(
        {
            "scene_id": "scene_0006",
            "title": "6、返航",
            "status": "complete",
            "counts": {"retrieved_entities": 2, "retrieved_memories": 4},
            "metrics": {
                "generated": {"overall": 1.0},
                "reference": {"overall": 0.8},
                "deltas": {"overall": 0.2},
            },
        }
    )

    assert flattened["gen_overall"] == 1.0
    assert flattened["ref_overall"] == 0.8
    assert flattened["memories"] == 4
