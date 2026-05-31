from __future__ import annotations

import json
from pathlib import Path

import dms.workflow as workflow
from dms.llm import LLMResult
from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
from dms.workflow import WritingE2EConfig, run_writing_e2e
from tests.storage.test_asset_store import _write_sample_ordered_run


class FakeConfigClient:
    provider = "fake"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, prompt: str) -> LLMResult:
        if "Build a writing-facing attribute card" in prompt:
            text = json.dumps(
                {
                    "entity_id": "character_0001",
                    "canonical_name": "刘培强",
                    "entity_type": "character",
                    "prefix_boundary": "before scene_0005",
                    "role_in_story": [{"value": "返航任务相关人物", "status": "inferred", "refs": ["M1"]}],
                    "current_state": [{"value": "情绪焦躁", "status": "explicit", "refs": ["M1"]}],
                    "salient_past_actions": [],
                    "stable_traits": [{"trait": "会回避压力", "status": "inferred", "refs": ["M1"]}],
                    "speaking_style": [],
                    "values_or_motivations": [],
                    "relationship_stances": [],
                    "behavior_tendencies": [],
                    "hard_constraints": [{"constraint": "刘培强不是未来信息知情者", "refs": ["M1"]}],
                    "simulation_risks": [],
                    "uncertain_or_unsupported": [],
                },
                ensure_ascii=False,
            )
        elif "Simulate how the target character" in prompt:
            text = json.dumps(
                {
                    "character": "刘培强",
                    "prefix_boundary": "before scene_0005",
                    "intent_assumptions": ["返航来自写作意图"],
                    "likely_internal_state": [{"value": "焦躁", "status": "inferred", "refs": ["M1"]}],
                    "likely_actions": [{"value": "压低动作", "status": "inferred", "refs": ["M1"]}],
                    "likely_dialogue": [],
                    "interaction_pressure": [],
                    "avoid_or_risks": [],
                    "memory_basis": [{"point": "情绪焦躁", "refs": ["M1"]}],
                },
                ensure_ascii=False,
            )
        elif "Coordinate the character simulations" in prompt:
            text = json.dumps(
                {
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
                },
                ensure_ascii=False,
            )
        elif "Decompose the writing intent" in prompt:
            text = json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id": "REQ1",
                            "requirement": "包含刘培强",
                            "category": "entity_anchor",
                            "importance": "core",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        elif "Judge writing-intent consistency" in prompt:
            text = json.dumps(
                {
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
                },
                ensure_ascii=False,
            )
        elif "Evaluate the writing quality" in prompt or "Evaluate memory faithfulness" in prompt:
            text = json.dumps(
                {
                    "candidate_label": "generated",
                    "score": 5,
                    "rationale": "ok",
                    "strengths": [],
                    "weaknesses": [],
                },
                ensure_ascii=False,
            )
        else:
            text = "J20C返航，刘培强盯着前方，张鹏的提醒还压在耳边，UEG灯光从雾里浮出来。"
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_run_writing_e2e_keeps_raw_draft_for_evaluation(tmp_path: Path, monkeypatch) -> None:
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
  model_name: fake-judge
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

    def fake_build_client(config, section):
        return FakeConfigClient(str(config[section]["model_name"]))

    monkeypatch.setattr(workflow, "build_openai_client_from_config", fake_build_client)

    summary = run_writing_e2e(
        WritingE2EConfig(
            db_path=db_path,
            chroma_dir=chroma_dir,
            writing_intent="写一段刘培强返航。",
            before_scene_id="scene_0005",
            output_dir=tmp_path / "e2e",
            model_config_path=config_path,
            scene_top_k=1,
            entity_memory_top_k=2,
            max_entity_memories_before_vector=3,
            attribute_entity_types=("character",),
            overwrite=True,
        )
    )

    draft = (tmp_path / "e2e" / "writing" / "draft.md").read_text(encoding="utf-8").strip()
    assert summary["policy"]["post_generation_repair"] == "disabled"
    assert summary["policy"]["evaluation_text"] == "raw writing/draft.md"
    assert summary["writing"]["output"]["draft_path"].endswith("writing/draft.md")
    assert summary["evaluation"]["inputs"]["generated_chars"] == len(draft)
    assert (tmp_path / "e2e" / "memory_packet.md").is_file()
    assert (tmp_path / "e2e" / "attribute_cards" / "attribute_cards.md").is_file()
    assert (tmp_path / "e2e" / "social_simulation" / "social_simulation.md").is_file()
    assert json.loads((tmp_path / "e2e" / "summary.json").read_text(encoding="utf-8"))["model_config"]["writing_llm"][
        "api_key"
    ] == "***"
