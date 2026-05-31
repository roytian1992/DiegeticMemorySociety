from __future__ import annotations

import json
from pathlib import Path

import dms.benchmark as benchmark
from dms.benchmark import (
    WritingBenchmarkPrepareConfig,
    WritingBenchmarkRunConfig,
    prepare_writing_benchmark_assets,
    run_writing_benchmark,
)
from dms.llm import LLMResult
from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
from tests.storage.test_asset_store import _write_sample_ordered_run


class FakeBenchmarkClient:
    provider = "fake"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, prompt: str) -> LLMResult:
        if "Extract a sparse exploratory author seed" in prompt:
            payload = {"unit_id": "scene_0005", "writing_intent": "刘培强和张鹏在J20C返航途中互动。"}
            text = json.dumps(payload, ensure_ascii=False)
        elif "Extract a detailed writing intent" in prompt:
            payload = {
                "unit_id": "scene_0005",
                "writing_intent": "写出刘培强和张鹏在J20C返航途中的紧张互动，并体现刘培强的焦躁。",
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Build a writing-facing attribute card" in prompt:
            payload = {
                "entity_id": "character_0001",
                "canonical_name": "刘培强",
                "entity_type": "character",
                "prefix_boundary": "before scene_0005",
                "role_in_story": [{"value": "返航人物", "status": "inferred", "refs": ["M1"]}],
                "current_state": [{"value": "情绪焦躁", "status": "explicit", "refs": ["M1"]}],
                "salient_past_actions": [],
                "stable_traits": [{"trait": "回避压力", "status": "inferred", "refs": ["M1"]}],
                "speaking_style": [],
                "values_or_motivations": [],
                "relationship_stances": [],
                "behavior_tendencies": [],
                "hard_constraints": [{"constraint": "刘培强不是未来信息知情者", "refs": ["M1"]}],
                "simulation_risks": [],
                "uncertain_or_unsupported": [],
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Simulate how the target character" in prompt:
            payload = {
                "character": "刘培强",
                "prefix_boundary": "before scene_0005",
                "intent_assumptions": ["返航来自写作意图"],
                "likely_internal_state": [{"value": "焦躁", "status": "inferred", "refs": ["M1"]}],
                "likely_actions": [{"value": "压低动作", "status": "inferred", "refs": ["M1"]}],
                "likely_dialogue": [],
                "interaction_pressure": [],
                "avoid_or_risks": [],
                "memory_basis": [{"point": "情绪焦躁", "refs": ["M1"]}],
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Coordinate the character simulations" in prompt:
            payload = {
                "simulation_id": "test",
                "prefix_boundary": "before scene_0005",
                "scene_beats": [
                    {
                        "beat": "返航动作承载焦躁。",
                        "participants": ["刘培强"],
                        "purpose": "人物心理",
                        "intent_basis": ["返航"],
                        "memory_basis": ["M1"],
                        "risks": [],
                    }
                ],
                "character_dynamics": [],
                "memory_risks": [],
                "writer_guidance": [],
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Decompose the writing intent" in prompt:
            payload = {
                "requirements": [
                    {
                        "requirement_id": "REQ1",
                        "requirement": "包含刘培强",
                        "category": "entity_anchor",
                        "importance": "core",
                    }
                ]
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Judge writing-intent consistency" in prompt:
            payload = {
                "candidate_label": "generated",
                "requirement_judgments": [
                    {
                        "requirement_id": "REQ1",
                        "status": "satisfied",
                        "evidence_from_candidate": "刘培强",
                        "rationale": "包含刘培强",
                    }
                ],
                "summary": "ok",
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "Evaluate the writing quality" in prompt or "Evaluate memory faithfulness" in prompt:
            payload = {
                "candidate_label": "generated",
                "score": 5,
                "rationale": "ok",
                "strengths": [],
                "weaknesses": [],
            }
            text = json.dumps(payload, ensure_ascii=False)
        else:
            text = "J20C返航，刘培强盯着前方，张鹏的提醒还压在耳边。"
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_prepare_writing_benchmark_assets_dry_run_builds_eligibility(tmp_path: Path) -> None:
    summary = prepare_writing_benchmark_assets(
        WritingBenchmarkPrepareConfig(
            script_path=Path("data/raw/流浪地球2剧本.json"),
            output_dir=tmp_path / "prepare",
            dry_run=True,
            overwrite=True,
        )
    )

    assert summary["status"] == "dry_run_complete"
    assert summary["eligibility"]["writing_eval_target_count"] > 0
    assert (tmp_path / "prepare" / "eligibility" / "writing_eval_targets.jsonl").is_file()


def test_run_writing_benchmark_dry_run_uses_writing_targets(tmp_path: Path) -> None:
    summary = run_writing_benchmark(
        WritingBenchmarkRunConfig(
            script_path=Path("data/raw/流浪地球2剧本.json"),
            db_path=tmp_path / "missing.sqlite",
            chroma_dir=tmp_path / "missing_chroma",
            output_dir=tmp_path / "bench",
            dry_run=True,
            limit=2,
            overwrite=True,
        )
    )

    assert summary["status"] == "dry_run_complete"
    assert summary["target_count"] == 2
    assert (tmp_path / "bench" / "target_manifest.jsonl").is_file()


def test_run_writing_benchmark_one_target_with_fake_clients(tmp_path: Path) -> None:
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"
    chroma_dir = tmp_path / "chroma"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))
    build_chroma_memory_index(ChromaMemoryIndexConfig(db_path=db_path, persist_dir=chroma_dir, reset=True, embedding_dim=64))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  provider: openai
  model_name: fake-llm
  api_key: local
  base_url: http://fake.test/v1
embedding:
  provider: hash
  dimensions: 64
writing_llm:
  provider: openai
  model_name: fake-writing
  api_key: writing
  base_url: http://fake-writing.test/v1
""",
        encoding="utf-8",
    )

    model_config = benchmark.load_local_config(config_path)
    summary = run_writing_benchmark(
        WritingBenchmarkRunConfig(
            script_path=Path("data/raw/流浪地球2剧本.json"),
            db_path=db_path,
            chroma_dir=chroma_dir,
            output_dir=tmp_path / "bench",
            model_config_path=config_path,
            target_scene_ids=("scene_0005",),
            limit=1,
            collection_name="dms_retrieval_documents",
            overwrite=True,
        ),
        llm_client=FakeBenchmarkClient("fake-llm"),
        writing_llm_client=FakeBenchmarkClient("fake-writing"),
        model_config=model_config,
    )

    assert summary["completed_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["aggregate_metrics"]["generated_overall_mean"] == 1.0
    target_summary = json.loads((tmp_path / "bench" / "targets" / "scene_0005" / "summary.json").read_text(encoding="utf-8"))
    assert target_summary["intent_levels"]["memory"] == "sparse"
    assert (tmp_path / "bench" / "targets" / "scene_0005" / "social_simulation" / "social_simulation.md").is_file()
