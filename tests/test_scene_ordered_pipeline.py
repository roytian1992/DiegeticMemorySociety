import json
import shutil
from pathlib import Path

from dms.llm import (
    FakeDurableRelationshipClient,
    FakeEpisodicMemoryClient,
    FakeKGEntityMentionClient,
    FakeKGEntityRefinementClient,
    FakeSceneInventoryClient,
    FakeSceneSummaryClient,
    LLMResult,
)
from dms.runners import SceneOrderedPipelineConfig, run_scene_ordered_pipeline


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")
DEBUG_EXTRACTIONS = Path("_debug") / "extractions"


def _fake_clients() -> dict[str, object]:
    return {
        "scene_summary": FakeSceneSummaryClient(),
        "scene_inventory": FakeSceneInventoryClient(),
        "kg_entity_mentions": FakeKGEntityMentionClient(),
        "kg_entity_refinement": FakeKGEntityRefinementClient(),
        "episodic_memories": FakeEpisodicMemoryClient(),
        "durable_relationships": FakeDurableRelationshipClient(),
    }


def test_scene_ordered_pipeline_runs_tasks_inside_each_unit(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline"

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=2,
            scene_task_concurrency=3,
        ),
        llm_clients=_fake_clients(),
    )

    assert summary["status"] == "complete"
    assert summary["selected_count"] == 2
    assert summary["chunk_count"] == 2
    assert summary["max_chunk_units"] == 800
    assert summary["task_order"]["cross_unit_order"] == "sequential"
    assert summary["task_order"]["per_unit_parallel"] == [
        "scene_summary",
        "scene_inventory",
        "kg_entity_mentions",
    ]
    assert summary["task_order"]["per_unit_refinement"] == "kg_entity_refinement"
    assert summary["task_order"]["per_unit_after_entities_parallel"] == [
        "episodic_memories",
        "durable_relationships",
    ]
    assert summary["task_summaries"]["scene_summary"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["scene_inventory"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["kg_entity_mentions"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["kg_entity_refinement"]["parsed_output_count"] == 2
    assert "scene_event_candidates" not in summary["task_summaries"]
    assert summary["task_summaries"]["episodic_memories"]["parsed_output_count"] == 2
    assert summary["task_summaries"]["durable_relationships"]["parsed_output_count"] == 2
    assert summary["memory_summaries"]["world_model"]["scene_count"] == 2
    assert summary["memory_summaries"]["world_model"]["scene_summary_count"] == 2
    assert summary["memory_summaries"]["world_model"]["episodic_memory_count"] == 2
    assert summary["memory_summaries"]["entity_resolution"]["entity_count"] >= 1

    assert (output_root / "README.md").is_file()
    assert (output_root / "summaries" / "scene_summaries.jsonl").is_file()
    assert (output_root / "scene_context" / "stated_facts.jsonl").is_file()
    assert (output_root / "knowledge_graph" / "entities.jsonl").is_file()
    assert (output_root / "memories" / "episodic_memories.jsonl").is_file()
    assert not (output_root / "episodic_memories").exists()

    unit_trace = (output_root / "_debug" / "unit_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(unit_trace) == 2
    assert json.loads(unit_trace[0])["scene_id"] == "scene_0001"
    assert json.loads(unit_trace[1])["scene_id"] == "scene_0002"

    episodic_input = (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "inputs" / "scene_0001.json").read_text(
        encoding="utf-8"
    )
    assert "scene_inventory" in episodic_input
    assert "kg_entity_mentions" in episodic_input
    assert "scene_summary" not in episodic_input
    assert "kg_entity_refinement" not in episodic_input
    assert "scene_event_candidates" not in episodic_input
    relationship_input = (
        output_root / DEBUG_EXTRACTIONS / "durable_relationships" / "inputs" / "scene_0001.json"
    ).read_text(encoding="utf-8")
    assert "scene_inventory" in relationship_input
    assert "kg_entity_mentions" in relationship_input
    assert "episodic_memories" not in relationship_input


def test_scene_ordered_pipeline_passes_refined_entities_to_dependent_tasks(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_refined_candidates"
    clients = _fake_clients()
    clients["kg_entity_refinement"] = _RenamingKGEntityRefinementClient()

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
        ),
        llm_clients=clients,
    )

    assert summary["task_summaries"]["kg_entity_refinement"]["parsed_output_count"] == 1
    episodic_input = json.loads(
        (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "inputs" / "scene_0001.json").read_text(
            encoding="utf-8"
        )
    )
    entities = episodic_input["extracted_candidates"]["kg_entity_mentions"]["data"]["entity_mentions"]
    surfaces = [item["surface"] for item in entities]
    assert "REFINED_CHARACTER" in surfaces
    assert "FAKE_CHARACTER" not in surfaces
    assert "REFINED_CHARACTER" in (output_root / "knowledge_graph" / "entities.jsonl").read_text(encoding="utf-8")


def test_scene_ordered_pipeline_uses_author_entity_context_as_description_baseline(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_author_entities"
    author_context_path = tmp_path / "author_entities.json"
    author_context_path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "canonical_name": "FAKE_CHARACTER",
                        "entity_type": "character",
                        "aliases": ["Fake Char"],
                        "description": "作者预设的测试角色",
                        "author_profile": {
                            "stable_traits": ["稳定测试人格"],
                            "speaking_style": ["简短"],
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
            prior_entity_context_path=author_context_path,
        ),
        llm_clients=_fake_clients(),
    )

    prompt = (output_root / DEBUG_EXTRACTIONS / "kg_entity_mentions" / "prompts" / "scene_0001.txt").read_text(
        encoding="utf-8"
    )
    entities = [json.loads(line) for line in (output_root / "knowledge_graph" / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    fake = [entity for entity in entities if entity["canonical_name"] == "FAKE_CHARACTER"]

    assert summary["status"] == "complete"
    assert "author_defined_entities" in prompt
    assert "作者预设的测试角色" in prompt
    assert len(fake) == 1
    assert fake[0]["author_description"] == "作者预设的测试角色"
    assert fake[0]["initial_description"] == "作者预设的测试角色"
    assert fake[0]["author_profile"]["stable_traits"] == ["稳定测试人格"]
    assert "a trackable test character" in fake[0]["descriptions"]


def test_scene_ordered_pipeline_chunks_long_units_when_budget_is_small(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_chunked"

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
            max_chunk_units=20,
        ),
        llm_clients=_fake_clients(),
    )

    assert summary["selected_count"] == 1
    assert summary["chunk_count"] > 1
    assert summary["task_summaries"]["episodic_memories"]["parsed_output_count"] == summary["chunk_count"]
    manifest = [
        json.loads(line)
        for line in (output_root / "_debug" / "chunk_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == summary["chunk_count"]
    assert all(record["chunk_unit_count"] <= 20 for record in manifest)
    assert manifest[0]["unit_id"] == "scene_0001_chunk_001"
    assert (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "inputs" / "scene_0001_chunk_001.json").is_file()


def test_scene_ordered_pipeline_carries_configured_narrative_unit_label(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_chapter_units"

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
            dry_run=True,
            unit_type="chapter",
            unit_label="chapter",
        ),
        llm_clients=None,
    )

    assert summary["narrative_unit"] == {
        "unit_type": "chapter",
        "unit_label": "chapter",
        "legacy_scene_id_compatibility": True,
    }
    manifest = [
        json.loads(line)
        for line in (output_root / "_debug" / "chunk_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest[0]["scene_id"] == "scene_0001"
    assert manifest[0]["unit_type"] == "chapter"
    assert manifest[0]["unit_label"] == "chapter"
    input_payload = json.loads(
        (output_root / DEBUG_EXTRACTIONS / "scene_summary" / "inputs" / "scene_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert input_payload["unit_id"] == "scene_0001"
    assert input_payload["unit_type"] == "chapter"
    assert input_payload["unit_label"] == "chapter"
    assert input_payload["scene_id"] == "scene_0001"


def test_scene_ordered_pipeline_extends_legacy_base_without_refinement_outputs(tmp_path: Path) -> None:
    base_root = tmp_path / "legacy_base"
    extended_root = tmp_path / "extended"
    run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=base_root,
            limit=1,
        ),
        llm_clients=_fake_clients(),
    )
    shutil.rmtree(base_root / DEBUG_EXTRACTIONS / "kg_entity_refinement")
    shutil.rmtree(base_root / "_debug" / "intermediate" / "kg_entity_source")

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            base_output_root=base_root,
            output_root=extended_root,
            start=2,
            limit=1,
        ),
        llm_clients=_fake_clients(),
    )

    assert summary["memory_summaries"]["world_model"]["scene_count"] == 2
    assert summary["memory_summaries"]["world_model"]["scene_summary_count"] == 2
    assert summary["memory_summaries"]["world_model"]["kg_entity_mention_count"] == 8
    assert (extended_root / "_debug" / "intermediate" / "kg_entity_source" / "parsed" / "scene_0001.json").is_file()
    assert (extended_root / "_debug" / "intermediate" / "kg_entity_source" / "parsed" / "scene_0002.json").is_file()


def test_scene_ordered_pipeline_dry_run_skips_memory_build(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_dry"

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
            dry_run=True,
        ),
        llm_clients=None,
    )

    assert summary["status"] == "dry_run_complete"
    assert summary["memory_summaries"] == {}
    assert summary["task_summaries"]["episodic_memories"]["rendered_prompt_count"] == 1


def test_scene_ordered_pipeline_repairs_unaligned_episodic_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_repair"
    clients = _fake_clients()
    clients["episodic_memories"] = _RepairingEpisodicClient()

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
        ),
        llm_clients=clients,
    )

    assert summary["task_summaries"]["episodic_memories"]["parsed_output_count"] == 1
    parsed = json.loads(
        (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "parsed" / "scene_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert parsed["repair_attempted"] is True
    assert parsed["repair_status"] == "parsed"
    assert (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "prompts" / "scene_0001.repair.txt").is_file()
    assert (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "raw_outputs" / "scene_0001.repair.json").is_file()
    assert summary["memory_summaries"]["episodic_memory"]["rejected_evidence_count"] == 0


def test_scene_ordered_pipeline_accepts_soft_link_evidence_warning(tmp_path: Path) -> None:
    output_root = tmp_path / "ordered_pipeline_soft_warning"
    clients = _fake_clients()
    clients["episodic_memories"] = _SoftWarningEpisodicClient()

    summary = run_scene_ordered_pipeline(
        SceneOrderedPipelineConfig(
            script_path=SCRIPT_PATH,
            output_root=output_root,
            limit=1,
        ),
        llm_clients=clients,
    )

    assert summary["task_summaries"]["episodic_memories"]["parsed_output_count"] == 1
    parsed = json.loads(
        (output_root / DEBUG_EXTRACTIONS / "episodic_memories" / "parsed" / "scene_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert parsed["status"] == "parsed"
    assert parsed["validation_errors"] == []
    assert parsed["validation_warnings"]
    assert summary["memory_summaries"]["episodic_memory"]["episodic_memory_count"] == 1
    assert summary["memory_summaries"]["episodic_memory"]["entity_memory_link_count"] == 1
    assert summary["memory_summaries"]["episodic_memory"]["skipped_entity_memory_link_count"] == 1


class _RepairingEpisodicClient:
    provider = "fake"
    model = "fake-repairing-episodic"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> LLMResult:
        self.calls += 1
        evidence = "脑机接口设备上躺着一位满脑贴着电极贴片的受试者" if self.calls > 1 else "模型改写的不存在证据"
        payload = {
            "unit_id": "scene_0001",
            "episodic_memories": [
                {
                    "memory_id_hint": "m1",
                    "sequence_index": 1,
                    "timeline_label": "scene_0001",
                    "memory_type": "action",
                    "summary": "受试者躺在脑机接口设备上。",
                    "evidence": evidence,
                    "entity_links": [
                        {
                            "entity": "FAKE_CHARACTER",
                            "entity_type": "character",
                            "link_role": "experiencer",
                            "evidence": evidence,
                        }
                    ],
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "call": self.calls, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class _RenamingKGEntityRefinementClient:
    provider = "fake"
    model = "fake-renaming-refinement"

    def complete(self, prompt: str) -> LLMResult:
        payload = {
            "unit_id": "scene_0001",
            "entity_mentions": [
                {
                    "surface": "REFINED_CHARACTER",
                    "entity_type": "character",
                    "canonical_hint": "REFINED_CHARACTER",
                    "role_in_unit": "speaker",
                    "attributes_or_state": "present",
                    "evidence": "scene_0001",
                }
            ],
            "scene_tags": [],
            "unresolved_mentions": [],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


class _SoftWarningEpisodicClient:
    provider = "fake"
    model = "fake-soft-warning-episodic"

    def complete(self, prompt: str) -> LLMResult:
        evidence = "脑机接口设备上躺着一位满脑贴着电极贴片的受试者"
        payload = {
            "unit_id": "scene_0001",
            "episodic_memories": [
                {
                    "memory_id_hint": "m1",
                    "sequence_index": 1,
                    "timeline_label": "scene_0001",
                    "memory_type": "action",
                    "summary": "受试者躺在脑机接口设备上。",
                    "evidence": evidence,
                    "entity_links": [
                        {
                            "entity": "FAKE_CHARACTER",
                            "entity_type": "character",
                            "link_role": "experiencer",
                            "evidence": evidence,
                        },
                        {
                            "entity": "印度科学家",
                            "entity_type": "character",
                            "link_role": "speaker",
                            "evidence": "印度科学家（印度式英语）：不存在的拼接证据",
                        },
                    ],
                }
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"fake": True, "text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )
