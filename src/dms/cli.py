from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dms.author_context import author_entities_from_context, load_author_entity_context
from dms.benchmark import (
    WritingBenchmarkPrepareConfig,
    WritingBenchmarkRunConfig,
    prepare_writing_benchmark_assets,
    run_writing_benchmark,
)
from dms.config import build_openai_client_from_config, embedding_kwargs_from_config, load_local_config
from dms.evaluation import WritingEvaluationConfig, build_scene_eligibility_splits, evaluate_writing
from dms.intent_levels import CLI_INTENT_LEVEL_CHOICES
from dms.writing import SocialWritingGenerationConfig, generate_writing_with_social_simulation
from dms.llm import (
    AnthropicMessagesClient,
    FakeDurableRelationshipClient,
    FakeEpisodicMemoryClient,
    FakeKGEntityMentionClient,
    FakeKGEntityRefinementClient,
    FakeReferenceItemClient,
    FakeSceneEventClient,
    FakeSceneInventoryClient,
    FakeSceneSummaryClient,
    FakeTemporalExtractionClient,
    FakeVisibilityNotesClient,
    OpenAIChatClient,
)
from dms.memory import (
    build_canonical_memory,
    build_entity_resolution_artifacts,
    build_episodic_memory,
    build_kg_entity_memory,
    build_memory_timeline_index,
    build_prefix_commits,
    build_prefix_world_model,
    build_scene_event_memory,
    build_scene_inventory_memory,
    build_scene_summary_memory,
    build_visibility_grounded_packet,
    build_visibility_memory,
    build_visibility_packet,
    MemoryTimelineIndexConfig,
    query_memory,
)
from dms.narrative_units import DEFAULT_UNIT_LABEL, DEFAULT_UNIT_TYPE
from dms.reference_library import (
    ChromaReferenceIndexConfig,
    ReferenceItemExtractionConfig,
    ReferenceItemImportConfig,
    ReferenceLibraryIngestConfig,
    build_chroma_reference_index,
    extract_reference_items,
    ingest_reference_library,
    import_reference_items,
)
from dms.retrieval import MemoryPacketConfig, build_memory_packet, format_memory_packet_markdown
from dms.prompts import YAMLPromptLoader
from dms.runners import (
    DurableRelationshipRunConfig,
    KGEntityRunConfig,
    SceneEventRunConfig,
    SceneInventoryRunConfig,
    SceneOrderedPipelineConfig,
    SceneSummaryRunConfig,
    TemporalExtractionRunConfig,
    VisibilityNotesRunConfig,
    run_durable_relationships,
    run_scene_events,
    run_scene_inventory,
    run_kg_entity_mentions,
    run_scene_summary,
    run_scene_ordered_pipeline,
    run_temporal_extraction,
    run_visibility_notes,
)
from dms.scripts.wandering_earth import load_script_scenes, write_jsonl, write_summary
from dms.simulation import (
    AttributeCardConfig,
    SceneDispositionNoteConfig,
    SocialSimulationConfig,
    build_entity_attribute_cards,
    build_scene_disposition_notes,
    run_social_simulation,
)
from dms.storage import (
    AssetStoreImportConfig,
    ChromaMemoryIndexConfig,
    build_chroma_memory_index,
    get_entity_memories,
    import_run_assets,
    search_entity_memories,
)
from dms.workflow import WritingE2EConfig, run_writing_e2e


def _read_writing_intent(args: argparse.Namespace) -> str:
    if getattr(args, "writing_intent", None):
        return str(args.writing_intent)
    path = getattr(args, "writing_intent_file", None)
    if not path:
        raise ValueError("--writing-intent or --writing-intent-file is required")
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and data.get("writing_intent"):
            return str(data["writing_intent"])
        if payload.get("writing_intent"):
            return str(payload["writing_intent"])
    return text.strip()


def _read_social_simulation_intent(args: argparse.Namespace) -> str:
    if getattr(args, "social_simulation_intent", None):
        return str(args.social_simulation_intent)
    path = getattr(args, "social_simulation_intent_file", None)
    if path:
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text.strip()
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict) and data.get("social_simulation_intent"):
                return str(data["social_simulation_intent"])
            if payload.get("social_simulation_intent"):
                return str(payload["social_simulation_intent"])
        return text.strip()
    return _read_writing_intent(args)


def _read_text_arg(value: str | None, path: Path | None, *, label: str) -> str:
    if value is not None:
        return value
    if path is not None:
        return path.read_text(encoding="utf-8").strip()
    raise ValueError(f"--{label} or --{label}-file is required")


def _read_reference_scene_text(script_path: Path | None, scene_id: str | None) -> str | None:
    if script_path is None or scene_id is None:
        return None
    scenes = load_script_scenes(script_path)
    for scene in scenes:
        if scene.scene_id == scene_id:
            return scene.content.strip()
    raise ValueError(f"Scene not found in {script_path}: {scene_id}")


def _add_reference_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-reference-context", action="store_true")
    parser.add_argument("--reference-db", "--reference-db-path", dest="reference_db_path", type=Path, default=None)
    parser.add_argument("--reference-chroma-dir", type=Path, default=None)
    parser.add_argument("--reference-collection-name", default="dms_reference_documents")
    parser.add_argument("--reference-top-k", type=int, default=6)
    parser.add_argument("--reference-author-top-k", type=int, default=6)
    parser.add_argument("--reference-character-top-k", type=int, default=6)
    parser.add_argument("--reference-style-top-k", type=int, default=4)
    parser.add_argument("--reference-timeline-top-k", type=int, default=4)


def _build_llm_client(args: argparse.Namespace, *, fake_task: str = "scene_inventory"):
    model_config_path = getattr(args, "model_config", None)
    if model_config_path:
        model_section = getattr(args, "model_section", None) or "llm"
        return build_openai_client_from_config(
            load_local_config(model_config_path),
            model_section,
            overrides=_explicit_model_overrides(args),
        )
    if args.provider == "fake":
        if fake_task == "scene_summary":
            return FakeSceneSummaryClient()
        if fake_task == "kg_entity_mentions":
            return FakeKGEntityMentionClient()
        if fake_task == "kg_entity_refinement":
            return FakeKGEntityRefinementClient()
        if fake_task == "scene_events":
            return FakeSceneEventClient()
        if fake_task == "temporal_extraction":
            return FakeTemporalExtractionClient()
        if fake_task == "visibility_notes":
            return FakeVisibilityNotesClient()
        if fake_task == "episodic_memories":
            return FakeEpisodicMemoryClient()
        if fake_task == "durable_relationships":
            return FakeDurableRelationshipClient()
        if fake_task == "reference_items":
            return FakeReferenceItemClient()
        return FakeSceneInventoryClient()
    if args.provider == "anthropic":
        return AnthropicMessagesClient(
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
        )
    if args.provider == "openai":
        return OpenAIChatClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.auth_token,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            enable_thinking=False,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


def _explicit_model_overrides(args: argparse.Namespace) -> dict[str, object]:
    raw_args = getattr(args, "_raw_argv", ())
    overrides: dict[str, object] = {}
    if "--max-tokens" in raw_args:
        overrides["max_tokens"] = args.max_tokens
    if "--temperature" in raw_args:
        overrides["temperature"] = args.temperature
    if "--timeout-seconds" in raw_args:
        overrides["timeout_seconds"] = args.timeout_seconds
    return overrides


def _embedding_kwargs(args: argparse.Namespace) -> dict:
    model_config_path = getattr(args, "model_config", None)
    if model_config_path:
        return embedding_kwargs_from_config(
            load_local_config(model_config_path),
            getattr(args, "embedding_section", None) or "embedding",
        )
    return {
        "embedding_dim": args.embedding_dim,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_api_key": args.embedding_api_key,
        "embedding_max_tokens": args.embedding_max_tokens,
        "embedding_timeout": args.embedding_timeout,
    }


def _build_task_llm_clients(args: argparse.Namespace):
    return {
        "scene_summary": _build_llm_client(args, fake_task="scene_summary"),
        "scene_inventory": _build_llm_client(args, fake_task="scene_inventory"),
        "kg_entity_mentions": _build_llm_client(args, fake_task="kg_entity_mentions"),
        "kg_entity_refinement": _build_llm_client(args, fake_task="kg_entity_refinement"),
        "episodic_memories": _build_llm_client(args, fake_task="episodic_memories"),
        "durable_relationships": _build_llm_client(args, fake_task="durable_relationships"),
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="dms")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_script = subparsers.add_parser("inspect-script", help="Inspect a local script JSON fixture.")
    inspect_script.add_argument("path", type=Path)
    inspect_script.add_argument("--limit", type=int, default=3)
    inspect_script.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    inspect_script.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)

    export_script = subparsers.add_parser("export-script-units", help="Export ordered script units as JSONL.")
    export_script.add_argument("path", type=Path)
    export_script.add_argument("--output", type=Path, required=True)
    export_script.add_argument("--summary", type=Path, default=None)
    export_script.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    export_script.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)

    eligibility = subparsers.add_parser(
        "build-scene-eligibility",
        help="Build deterministic memory/evaluation eligibility splits for ordered scenes.",
    )
    eligibility.add_argument("script_path", type=Path)
    eligibility.add_argument("--output-dir", type=Path, required=True)

    render_prompt = subparsers.add_parser("render-prompt", help="Render a YAML prompt by id.")
    render_prompt.add_argument("prompt_id")
    render_prompt.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    render_prompt.add_argument("--task-json", default="{}")
    render_prompt.add_argument("--static-json", default="{}")

    inventory = subparsers.add_parser("run-scene-inventory", help="Render scene inventory prompts for a scene prefix.")
    inventory.add_argument("script_path", type=Path)
    inventory.add_argument("--output-dir", type=Path, required=True)
    inventory.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    inventory.add_argument("--task-settings", type=Path, default=Path("task_specs/task_settings/scene_inventory_task.json"))
    inventory.add_argument("--start", type=int, default=1)
    inventory.add_argument("--limit", type=int, default=5)
    inventory.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    inventory.add_argument("--overwrite", action="store_true")
    inventory.add_argument("--model-config", type=Path, default=None)
    inventory.add_argument("--model-section", default="llm")
    inventory.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    inventory.add_argument("--model", default=None)
    inventory.add_argument("--base-url", default=None)
    inventory.add_argument("--auth-token", default=None)
    inventory.add_argument("--max-tokens", type=int, default=2048)
    inventory.add_argument("--temperature", type=float, default=0.0)
    inventory.add_argument("--timeout-seconds", type=int, default=120)

    scene_summary = subparsers.add_parser("run-scene-summary", help="Run scene summary extraction for a scene prefix.")
    scene_summary.add_argument("script_path", type=Path)
    scene_summary.add_argument("--output-dir", type=Path, required=True)
    scene_summary.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    scene_summary.add_argument("--task-settings", type=Path, default=Path("task_specs/task_settings/scene_summary_task.json"))
    scene_summary.add_argument("--start", type=int, default=1)
    scene_summary.add_argument("--limit", type=int, default=5)
    scene_summary.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    scene_summary.add_argument("--overwrite", action="store_true")
    scene_summary.add_argument("--max-chunk-units", type=int, default=800)
    scene_summary.add_argument("--model-config", type=Path, default=None)
    scene_summary.add_argument("--model-section", default="llm")
    scene_summary.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    scene_summary.add_argument("--model", default=None)
    scene_summary.add_argument("--base-url", default=None)
    scene_summary.add_argument("--auth-token", default=None)
    scene_summary.add_argument("--max-tokens", type=int, default=2048)
    scene_summary.add_argument("--temperature", type=float, default=0.0)
    scene_summary.add_argument("--timeout-seconds", type=int, default=120)

    events = subparsers.add_parser("run-scene-events", help="Run scene event candidate extraction for a scene prefix.")
    events.add_argument("script_path", type=Path)
    events.add_argument("--output-dir", type=Path, required=True)
    events.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    events.add_argument(
        "--task-settings",
        type=Path,
        default=Path("task_specs/task_settings/scene_event_candidates_task.json"),
    )
    events.add_argument("--prior-context", type=Path, default=None)
    events.add_argument("--start", type=int, default=1)
    events.add_argument("--limit", type=int, default=5)
    events.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    events.add_argument("--overwrite", action="store_true")
    events.add_argument("--model-config", type=Path, default=None)
    events.add_argument("--model-section", default="llm")
    events.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    events.add_argument("--model", default=None)
    events.add_argument("--base-url", default=None)
    events.add_argument("--auth-token", default=None)
    events.add_argument("--max-tokens", type=int, default=2048)
    events.add_argument("--temperature", type=float, default=0.0)
    events.add_argument("--timeout-seconds", type=int, default=120)

    temporal = subparsers.add_parser(
        "run-temporal-extraction",
        help="Run diegetic temporal extraction and build an audit timeline graph.",
    )
    temporal.add_argument("script_path", type=Path)
    temporal.add_argument("--output-dir", type=Path, required=True)
    temporal.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    temporal.add_argument(
        "--task-settings",
        type=Path,
        default=Path("task_specs/task_settings/temporal_extraction_task.json"),
    )
    temporal.add_argument("--prior-timeline", type=Path, default=None)
    temporal.add_argument("--start", type=int, default=1)
    temporal.add_argument("--limit", type=int, default=5)
    temporal.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    temporal.add_argument("--overwrite", action="store_true")
    temporal.add_argument("--model-config", type=Path, default=None)
    temporal.add_argument("--model-section", default="llm")
    temporal.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    temporal.add_argument("--model", default=None)
    temporal.add_argument("--base-url", default=None)
    temporal.add_argument("--auth-token", default=None)
    temporal.add_argument("--max-tokens", type=int, default=2048)
    temporal.add_argument("--temperature", type=float, default=0.0)
    temporal.add_argument("--timeout-seconds", type=int, default=120)

    kg_entities = subparsers.add_parser(
        "run-kg-entities",
        help="Run KG entity mention extraction for a scene prefix.",
    )
    kg_entities.add_argument("script_path", type=Path)
    kg_entities.add_argument("--output-dir", type=Path, required=True)
    kg_entities.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    kg_entities.add_argument(
        "--task-settings",
        type=Path,
        default=Path("task_specs/task_settings/kg_entity_mentions_task.json"),
    )
    kg_entities.add_argument("--prior-entity-context", type=Path, default=None)
    kg_entities.add_argument("--start", type=int, default=1)
    kg_entities.add_argument("--limit", type=int, default=5)
    kg_entities.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    kg_entities.add_argument("--overwrite", action="store_true")
    kg_entities.add_argument("--model-config", type=Path, default=None)
    kg_entities.add_argument("--model-section", default="llm")
    kg_entities.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    kg_entities.add_argument("--model", default=None)
    kg_entities.add_argument("--base-url", default=None)
    kg_entities.add_argument("--auth-token", default=None)
    kg_entities.add_argument("--max-tokens", type=int, default=2048)
    kg_entities.add_argument("--temperature", type=float, default=0.0)
    kg_entities.add_argument("--timeout-seconds", type=int, default=120)

    durable_relationships = subparsers.add_parser(
        "run-durable-relationships",
        help="Run durable relationship extraction for a scene prefix.",
    )
    durable_relationships.add_argument("script_path", type=Path)
    durable_relationships.add_argument("--output-dir", type=Path, required=True)
    durable_relationships.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    durable_relationships.add_argument(
        "--task-settings",
        type=Path,
        default=Path("task_specs/task_settings/durable_relationships_task.json"),
    )
    durable_relationships.add_argument("--extracted-candidates-dir", type=Path, default=None)
    durable_relationships.add_argument("--start", type=int, default=1)
    durable_relationships.add_argument("--limit", type=int, default=5)
    durable_relationships.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    durable_relationships.add_argument("--overwrite", action="store_true")
    durable_relationships.add_argument("--model-config", type=Path, default=None)
    durable_relationships.add_argument("--model-section", default="llm")
    durable_relationships.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    durable_relationships.add_argument("--model", default=None)
    durable_relationships.add_argument("--base-url", default=None)
    durable_relationships.add_argument("--auth-token", default=None)
    durable_relationships.add_argument("--max-tokens", type=int, default=2048)
    durable_relationships.add_argument("--temperature", type=float, default=0.0)
    durable_relationships.add_argument("--timeout-seconds", type=int, default=120)

    visibility = subparsers.add_parser("run-visibility-notes", help="Run scene-level visibility note extraction.")
    visibility.add_argument("script_path", type=Path)
    visibility.add_argument("--output-dir", type=Path, required=True)
    visibility.add_argument("--extracted-candidates-dir", type=Path, required=True)
    visibility.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    visibility.add_argument("--task-settings", type=Path, default=Path("task_specs/task_settings/visibility_notes_task.json"))
    visibility.add_argument("--start", type=int, default=1)
    visibility.add_argument("--limit", type=int, default=5)
    visibility.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    visibility.add_argument("--overwrite", action="store_true")
    visibility.add_argument("--model-config", type=Path, default=None)
    visibility.add_argument("--model-section", default="llm")
    visibility.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    visibility.add_argument("--model", default=None)
    visibility.add_argument("--base-url", default=None)
    visibility.add_argument("--auth-token", default=None)
    visibility.add_argument("--max-tokens", type=int, default=2048)
    visibility.add_argument("--temperature", type=float, default=0.0)
    visibility.add_argument("--timeout-seconds", type=int, default=120)

    build_memory = subparsers.add_parser(
        "build-scene-inventory-memory",
        help="Build staged JSONL memory artifacts from parsed scene-inventory outputs.",
    )
    build_memory.add_argument("run_dir", type=Path)
    build_memory.add_argument("--output-dir", type=Path, required=True)

    build_summary_memory = subparsers.add_parser(
        "build-scene-summary-memory",
        help="Build recap and retrieval JSONL artifacts from parsed scene-summary outputs.",
    )
    build_summary_memory.add_argument("run_dir", type=Path)
    build_summary_memory.add_argument("--output-dir", type=Path, required=True)

    build_event_memory = subparsers.add_parser(
        "build-scene-event-memory",
        help="Build staged JSONL memory artifacts from parsed scene-event outputs.",
    )
    build_event_memory.add_argument("run_dir", type=Path)
    build_event_memory.add_argument("--output-dir", type=Path, required=True)

    build_kg_memory = subparsers.add_parser(
        "build-kg-entity-memory",
        help="Build staged KG entity JSONL artifacts from parsed entity outputs.",
    )
    build_kg_memory.add_argument("run_dir", type=Path)
    build_kg_memory.add_argument("--output-dir", type=Path, required=True)

    build_vis_memory = subparsers.add_parser(
        "build-visibility-memory",
        help="Build staged JSONL visibility artifacts from parsed visibility-note outputs.",
    )
    build_vis_memory.add_argument("run_dir", type=Path)
    build_vis_memory.add_argument("--output-dir", type=Path, required=True)

    build_epi_memory = subparsers.add_parser(
        "build-episodic-memory",
        help="Build ordered episodic memory artifacts from parsed episodic-memory outputs.",
    )
    build_epi_memory.add_argument("run_dir", type=Path)
    build_epi_memory.add_argument("--output-dir", type=Path, required=True)

    build_memory_timeline = subparsers.add_parser(
        "build-memory-timeline-index",
        help="Attach diegetic timeline indexes to episodic memory JSONL artifacts.",
    )
    build_memory_timeline.add_argument("memory_path", type=Path)
    build_memory_timeline.add_argument("--timeline-graph", type=Path, required=True)
    build_memory_timeline.add_argument("--output-dir", type=Path, required=True)
    build_memory_timeline.add_argument("--min-match-score", type=float, default=0.25)
    build_memory_timeline.add_argument("--overwrite", action="store_true")

    build_relationship_memory_parser = subparsers.add_parser(
        "build-relationship-memory",
        help="Build durable relationship memory artifacts from parsed relationship outputs.",
    )
    build_relationship_memory_parser.add_argument("run_dir", type=Path)
    build_relationship_memory_parser.add_argument("--output-dir", type=Path, required=True)

    world_model = subparsers.add_parser("build-prefix-world-model", help="Merge canonical, event, entity, and memory layers.")
    world_model.add_argument("--canonical-dir", type=Path, required=True)
    world_model.add_argument("--event-memory-dir", type=Path, required=True)
    world_model.add_argument("--visibility-memory-dir", type=Path, default=None)
    world_model.add_argument("--kg-entity-memory-dir", type=Path, default=None)
    world_model.add_argument("--episodic-memory-dir", type=Path, default=None)
    world_model.add_argument("--relationship-memory-dir", type=Path, default=None)
    world_model.add_argument("--scene-summary-dir", type=Path, default=None)
    world_model.add_argument("--output-dir", type=Path, required=True)

    entity_resolution = subparsers.add_parser(
        "build-entity-resolution",
        help="Build an entity alias registry and relationship timeline from a prefix world model.",
    )
    entity_resolution.add_argument("world_model_path", type=Path)
    entity_resolution.add_argument("--output-dir", type=Path, required=True)
    entity_resolution.add_argument("--prior-entity-context", type=Path, default=None)

    prefix_commits = subparsers.add_parser(
        "build-prefix-commits",
        help="Replay a prefix world model into per-scene commits and snapshots.",
    )
    prefix_commits.add_argument("world_model_path", type=Path)
    prefix_commits.add_argument("--entity-resolution-dir", type=Path, required=True)
    prefix_commits.add_argument("--output-dir", type=Path, required=True)

    canonical = subparsers.add_parser("build-canonical-memory", help="Build canonical prefix memory from staged memory.")
    canonical.add_argument("staged_dir", type=Path)
    canonical.add_argument("--output-dir", type=Path, required=True)

    query = subparsers.add_parser("query-memory", help="Query a canonical memory directory or file.")
    query.add_argument("memory_path", type=Path)
    query.add_argument("--text", default="")
    query.add_argument("--character", default=None)
    query.add_argument("--scene-id", default=None)
    query.add_argument("--limit", type=int, default=10)

    asset_store = subparsers.add_parser(
        "build-asset-store",
        help="Import ordered-run assets into a SQLite store for timeline/entity queries.",
    )
    asset_store.add_argument("--run-root", type=Path, required=True)
    asset_store.add_argument("--output-db", type=Path, required=True)
    asset_store.add_argument("--summary-memory-dir", type=Path, default=None)
    asset_store.add_argument("--overwrite", action="store_true")

    reference_ingest = subparsers.add_parser(
        "ingest-reference-library",
        help="Chunk mixed external reference files into raw reference-library JSONL artifacts.",
    )
    reference_ingest.add_argument("input_path", type=Path)
    reference_ingest.add_argument("--output-dir", type=Path, required=True)
    reference_ingest.add_argument("--max-chunk-chars", type=int, default=2400)
    reference_ingest.add_argument("--overwrite", action="store_true")

    reference_extract = subparsers.add_parser(
        "extract-reference-items",
        help="Extract flat reference_items.jsonl from ingested external reference chunks.",
    )
    reference_extract.add_argument("library_dir", type=Path)
    reference_extract.add_argument("--output-dir", type=Path, required=True)
    reference_extract.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    reference_extract.add_argument(
        "--task-settings",
        type=Path,
        default=Path("task_specs/task_settings/reference_items_task.json"),
    )
    reference_extract.add_argument("--start", type=int, default=1)
    reference_extract.add_argument("--limit", type=int, default=None)
    reference_extract.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    reference_extract.add_argument("--overwrite", action="store_true")
    reference_extract.add_argument("--model-config", type=Path, default=None)
    reference_extract.add_argument("--model-section", default="llm")
    reference_extract.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    reference_extract.add_argument("--model", default=None)
    reference_extract.add_argument("--base-url", default=None)
    reference_extract.add_argument("--auth-token", default=None)
    reference_extract.add_argument("--max-tokens", type=int, default=4096)
    reference_extract.add_argument("--temperature", type=float, default=0.0)
    reference_extract.add_argument("--timeout-seconds", type=int, default=120)

    reference_items = subparsers.add_parser(
        "import-reference-items",
        help="Import flat external reference items JSONL into a standalone SQLite reference library.",
    )
    reference_items.add_argument("items_path", type=Path)
    reference_items.add_argument("--output-db", type=Path, required=True)
    reference_items.add_argument("--overwrite", action="store_true")

    reference_index = subparsers.add_parser(
        "build-reference-index",
        help="Index standalone external reference documents in a persistent Chroma collection.",
    )
    reference_index.add_argument("db_path", type=Path)
    reference_index.add_argument("--persist-dir", type=Path, required=True)
    reference_index.add_argument("--collection-name", default="dms_reference_documents")
    reference_index.add_argument("--model-config", type=Path, default=None)
    reference_index.add_argument("--embedding-section", default="embedding")
    reference_index.add_argument("--embedding-dim", type=int, default=384)
    reference_index.add_argument("--embedding-provider", choices=["hash", "openai"], default="hash")
    reference_index.add_argument("--embedding-model", default=None)
    reference_index.add_argument("--embedding-base-url", default=None)
    reference_index.add_argument("--embedding-api-key", default=None)
    reference_index.add_argument("--embedding-max-tokens", type=int, default=8192)
    reference_index.add_argument("--embedding-timeout", type=int, default=60)
    reference_index.add_argument("--upsert-batch-size", type=int, default=1000)
    reference_index.add_argument("--overwrite", action="store_true")

    query_entity_memories = subparsers.add_parser(
        "query-entity-memories",
        help="List episodic memories linked to an entity before a scene/time point.",
    )
    query_entity_memories.add_argument("db_path", type=Path)
    query_entity_memories.add_argument("--entity", required=True)
    query_entity_memories.add_argument("--before-scene", default=None)
    query_entity_memories.add_argument("--before-scene-order", type=int, default=None)
    query_entity_memories.add_argument("--limit", type=int, default=20)

    chroma_index = subparsers.add_parser(
        "build-chroma-index",
        help="Index SQLite retrieval documents in a persistent Chroma collection.",
    )
    chroma_index.add_argument("db_path", type=Path)
    chroma_index.add_argument("--persist-dir", type=Path, required=True)
    chroma_index.add_argument("--collection-name", default="dms_retrieval_documents")
    chroma_index.add_argument("--model-config", type=Path, default=None)
    chroma_index.add_argument("--embedding-section", default="embedding")
    chroma_index.add_argument("--embedding-dim", type=int, default=384)
    chroma_index.add_argument("--embedding-provider", choices=["hash", "openai"], default="hash")
    chroma_index.add_argument("--embedding-model", default=None)
    chroma_index.add_argument("--embedding-base-url", default=None)
    chroma_index.add_argument("--embedding-api-key", default=None)
    chroma_index.add_argument("--embedding-max-tokens", type=int, default=8192)
    chroma_index.add_argument("--embedding-timeout", type=int, default=60)
    chroma_index.add_argument("--upsert-batch-size", type=int, default=1000)
    chroma_index.add_argument("--overwrite", action="store_true")

    search_entity_memory = subparsers.add_parser(
        "search-entity-memories",
        help="Vector-search retrieval documents, then return source-grounded memory rows.",
    )
    search_entity_memory.add_argument("db_path", type=Path)
    search_entity_memory.add_argument("--chroma-dir", type=Path, required=True)
    search_entity_memory.add_argument("--query", required=True)
    search_entity_memory.add_argument("--entity", default=None)
    search_entity_memory.add_argument("--before-scene", default=None)
    search_entity_memory.add_argument("--before-scene-order", type=int, default=None)
    search_entity_memory.add_argument("--collection-name", default="dms_retrieval_documents")
    search_entity_memory.add_argument("--model-config", type=Path, default=None)
    search_entity_memory.add_argument("--embedding-section", default="embedding")
    search_entity_memory.add_argument("--embedding-dim", type=int, default=384)
    search_entity_memory.add_argument("--embedding-provider", choices=["hash", "openai"], default="hash")
    search_entity_memory.add_argument("--embedding-model", default=None)
    search_entity_memory.add_argument("--embedding-base-url", default=None)
    search_entity_memory.add_argument("--embedding-api-key", default=None)
    search_entity_memory.add_argument("--embedding-max-tokens", type=int, default=8192)
    search_entity_memory.add_argument("--embedding-timeout", type=int, default=60)
    search_entity_memory.add_argument("--top-k", type=int, default=10)

    memory_packet = subparsers.add_parser(
        "build-memory-packet",
        help="Build a writing-time memory packet from intent, scene summaries, entities, relationships, and episodic memories.",
    )
    memory_packet.add_argument("db_path", type=Path)
    memory_packet.add_argument("--chroma-dir", type=Path, required=True)
    memory_packet.add_argument("--writing-intent", default=None)
    memory_packet.add_argument("--writing-intent-file", type=Path, default=None)
    memory_packet.add_argument("--before-scene", default=None)
    memory_packet.add_argument("--before-scene-order", type=int, default=None)
    memory_packet.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    memory_packet.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)
    memory_packet.add_argument("--scene-top-k", type=int, default=5)
    memory_packet.add_argument("--entity-memory-top-k", type=int, default=12)
    memory_packet.add_argument("--global-scope-memory-top-k", type=int, default=8)
    memory_packet.add_argument("--max-entity-memories-before-vector", type=int, default=50)
    memory_packet.add_argument("--entity-match-limit", type=int, default=1)
    memory_packet.add_argument("--collection-name", default="dms_retrieval_documents")
    memory_packet.add_argument("--model-config", type=Path, default=None)
    memory_packet.add_argument("--embedding-section", default="embedding")
    memory_packet.add_argument("--embedding-dim", type=int, default=384)
    memory_packet.add_argument("--embedding-provider", choices=["hash", "openai"], default="hash")
    memory_packet.add_argument("--embedding-model", default=None)
    memory_packet.add_argument("--embedding-base-url", default=None)
    memory_packet.add_argument("--embedding-api-key", default=None)
    memory_packet.add_argument("--embedding-max-tokens", type=int, default=8192)
    memory_packet.add_argument("--embedding-timeout", type=int, default=60)
    _add_reference_context_args(memory_packet)
    memory_packet.add_argument("--format", choices=["json", "markdown"], default="json")
    memory_packet.add_argument("--output", type=Path, default=None)

    writing_eval = subparsers.add_parser(
        "evaluate-writing",
        help="Run LLM-as-judge evaluation for one generated passage against writing intent and memory.",
    )
    writing_eval.add_argument("--writing-intent", default=None)
    writing_eval.add_argument("--writing-intent-file", type=Path, default=None)
    writing_eval.add_argument("--generated-text", default=None)
    writing_eval.add_argument("--generated-text-file", type=Path, default=None)
    writing_eval.add_argument("--memory-packet", default=None)
    writing_eval.add_argument("--memory-packet-file", type=Path, default=None)
    writing_eval.add_argument("--reference-text", default=None)
    writing_eval.add_argument("--reference-text-file", type=Path, default=None)
    writing_eval.add_argument("--reference-script", type=Path, default=None)
    writing_eval.add_argument("--reference-scene-id", default=None)
    writing_eval.add_argument("--output-dir", type=Path, required=True)
    writing_eval.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    writing_eval.add_argument("--overwrite", action="store_true")
    writing_eval.add_argument("--model-config", type=Path, default=None)
    writing_eval.add_argument("--model-section", default="llm")
    writing_eval.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="openai")
    writing_eval.add_argument("--model", default=None)
    writing_eval.add_argument("--base-url", default=None)
    writing_eval.add_argument("--auth-token", default=None)
    writing_eval.add_argument("--max-tokens", type=int, default=4096)
    writing_eval.add_argument("--temperature", type=float, default=0.0)
    writing_eval.add_argument("--timeout-seconds", type=int, default=120)

    attribute_cards = subparsers.add_parser(
        "build-entity-attribute-cards",
        help="Build evidence-bound entity attribute cards from a memory packet.",
    )
    attribute_cards.add_argument("memory_packet_path", type=Path)
    attribute_cards.add_argument("--output-dir", type=Path, required=True)
    attribute_cards.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    attribute_cards.add_argument("--entity-type", action="append", dest="entity_types", default=None)
    attribute_cards.add_argument("--entity", action="append", dest="entity_names", default=None)
    attribute_cards.add_argument("--max-memories-per-entity", type=int, default=16)
    attribute_cards.add_argument("--overwrite", action="store_true")
    attribute_cards.add_argument("--model-config", type=Path, default=None)
    attribute_cards.add_argument("--model-section", default="llm")
    attribute_cards.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="openai")
    attribute_cards.add_argument("--model", default=None)
    attribute_cards.add_argument("--base-url", default=None)
    attribute_cards.add_argument("--auth-token", default=None)
    attribute_cards.add_argument("--max-tokens", type=int, default=2048)
    attribute_cards.add_argument("--temperature", type=float, default=0.0)
    attribute_cards.add_argument("--timeout-seconds", type=int, default=120)

    disposition_notes = subparsers.add_parser(
        "build-scene-disposition-notes",
        help="Build compact scene-conditioned disposition notes from attribute cards and a social simulation intent.",
    )
    disposition_notes.add_argument("attribute_cards_path", type=Path)
    disposition_notes.add_argument("--social-simulation-intent", required=True)
    disposition_notes.add_argument("--memory-packet", type=Path, default=None)
    disposition_notes.add_argument("--output-dir", type=Path, required=True)
    disposition_notes.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    disposition_notes.add_argument("--entity-type", action="append", dest="entity_types", default=None)
    disposition_notes.add_argument("--entity", action="append", dest="entity_names", default=None)
    disposition_notes.add_argument("--max-relevant-memories-per-entity", type=int, default=6)
    disposition_notes.add_argument("--overwrite", action="store_true")
    disposition_notes.add_argument("--model-config", type=Path, default=None)
    disposition_notes.add_argument("--model-section", default="llm")
    disposition_notes.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="openai")
    disposition_notes.add_argument("--model", default=None)
    disposition_notes.add_argument("--base-url", default=None)
    disposition_notes.add_argument("--auth-token", default=None)
    disposition_notes.add_argument("--max-tokens", type=int, default=1024)
    disposition_notes.add_argument("--temperature", type=float, default=0.0)
    disposition_notes.add_argument("--timeout-seconds", type=int, default=120)

    social_simulation = subparsers.add_parser(
        "run-social-simulation",
        help="Run attribute-card-conditioned character social simulation from a low-information social simulation intent.",
    )
    social_simulation.add_argument("attribute_cards_path", type=Path)
    social_simulation.add_argument("--social-simulation-intent", default=None)
    social_simulation.add_argument("--social-simulation-intent-file", type=Path, default=None)
    social_simulation.add_argument("--writing-intent", default=None)
    social_simulation.add_argument("--writing-intent-file", type=Path, default=None)
    social_simulation.add_argument("--scene-disposition-notes", type=Path, default=None)
    social_simulation.add_argument("--output-dir", type=Path, required=True)
    social_simulation.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    social_simulation.add_argument("--overwrite", action="store_true")
    social_simulation.add_argument("--model-config", type=Path, default=None)
    social_simulation.add_argument("--model-section", default="llm")
    social_simulation.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="openai")
    social_simulation.add_argument("--model", default=None)
    social_simulation.add_argument("--base-url", default=None)
    social_simulation.add_argument("--auth-token", default=None)
    social_simulation.add_argument("--max-tokens", type=int, default=2048)
    social_simulation.add_argument("--temperature", type=float, default=0.0)
    social_simulation.add_argument("--timeout-seconds", type=int, default=120)

    social_writing = subparsers.add_parser(
        "generate-writing-social",
        help="Generate final writing from memory packet, attribute cards, social simulation, and local config writing_llm.",
    )
    social_writing.add_argument("--writing-request", required=True)
    social_writing.add_argument("--memory-packet-file", type=Path, required=True)
    social_writing.add_argument("--attribute-cards-file", type=Path, required=True)
    social_writing.add_argument("--social-simulation-file", type=Path, required=True)
    social_writing.add_argument("--output-dir", type=Path, required=True)
    social_writing.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    social_writing.add_argument("--model-section", default="writing_llm")
    social_writing.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    social_writing.add_argument("--previous-scene-context", default="")
    social_writing.add_argument("--previous-scene-context-file", type=Path, default=None)
    social_writing.add_argument("--previous-scene-context-script", type=Path, default=None)
    social_writing.add_argument("--previous-scene-context-scene-id", default=None)
    social_writing.add_argument("--previous-scene-context-max-chars", type=int, default=800)
    social_writing.add_argument("--style-reference-file", type=Path, default=None)
    social_writing.add_argument("--style-reference-script", type=Path, default=None)
    social_writing.add_argument("--style-reference-scene-id", default=None)
    social_writing.add_argument("--length-requirement", default="")
    social_writing.add_argument("--output-requirements", default="")
    social_writing.add_argument("--overwrite", action="store_true")

    writing_e2e = subparsers.add_parser(
        "run-writing-e2e",
        help="Run retrieval, attribute cards, social simulation, raw writing, and optional evaluation from local config.",
    )
    writing_e2e.add_argument("--db-path", type=Path, required=True)
    writing_e2e.add_argument("--chroma-dir", type=Path, required=True)
    writing_e2e.add_argument("--writing-intent", default=None)
    writing_e2e.add_argument("--writing-intent-file", type=Path, default=None)
    writing_e2e.add_argument("--output-dir", type=Path, required=True)
    writing_e2e.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    writing_e2e.add_argument("--llm-section", default="llm")
    writing_e2e.add_argument("--writing-llm-section", default="writing_llm")
    writing_e2e.add_argument("--embedding-section", default="embedding")
    writing_e2e.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    writing_e2e.add_argument("--before-scene", default=None)
    writing_e2e.add_argument("--before-scene-order", type=int, default=None)
    writing_e2e.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    writing_e2e.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)
    writing_e2e.add_argument("--scene-top-k", type=int, default=5)
    writing_e2e.add_argument("--entity-memory-top-k", type=int, default=12)
    writing_e2e.add_argument("--global-scope-memory-top-k", type=int, default=8)
    writing_e2e.add_argument("--max-entity-memories-before-vector", type=int, default=50)
    writing_e2e.add_argument("--entity-match-limit", type=int, default=1)
    writing_e2e.add_argument("--collection-name", default="dms_retrieval_documents")
    _add_reference_context_args(writing_e2e)
    writing_e2e.add_argument("--attribute-entity-type", action="append", dest="attribute_entity_types", default=None)
    writing_e2e.add_argument("--attribute-entity", action="append", dest="attribute_entity_names", default=None)
    writing_e2e.add_argument("--max-memories-per-entity", type=int, default=16)
    writing_e2e.add_argument("--previous-scene-context", default="")
    writing_e2e.add_argument("--previous-scene-context-file", type=Path, default=None)
    writing_e2e.add_argument("--previous-scene-context-script", type=Path, default=None)
    writing_e2e.add_argument("--previous-scene-context-scene-id", default=None)
    writing_e2e.add_argument("--previous-scene-context-max-chars", type=int, default=800)
    writing_e2e.add_argument("--style-reference-file", type=Path, default=None)
    writing_e2e.add_argument("--style-reference-script", type=Path, default=None)
    writing_e2e.add_argument("--style-reference-scene-id", default=None)
    writing_e2e.add_argument("--length-requirement", default="")
    writing_e2e.add_argument("--output-requirements", default="")
    writing_e2e.add_argument("--reference-text", default=None)
    writing_e2e.add_argument("--reference-text-file", type=Path, default=None)
    writing_e2e.add_argument("--reference-script", type=Path, default=None)
    writing_e2e.add_argument("--reference-scene-id", default=None)
    writing_e2e.add_argument("--skip-evaluation", action="store_true")
    writing_e2e.add_argument("--overwrite", action="store_true")

    benchmark_prepare = subparsers.add_parser(
        "prepare-writing-benchmark",
        help="Prepare eligibility splits, optional full extraction, SQLite, and Chroma assets for writing benchmark runs.",
    )
    benchmark_prepare.add_argument("script_path", type=Path)
    benchmark_prepare.add_argument("--output-dir", type=Path, required=True)
    benchmark_prepare.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    benchmark_prepare.add_argument("--llm-section", default="llm")
    benchmark_prepare.add_argument("--embedding-section", default="embedding")
    benchmark_prepare.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    benchmark_prepare.add_argument("--ordered-run-dir", type=Path, default=None)
    benchmark_prepare.add_argument("--extraction-output-root", type=Path, default=None)
    benchmark_prepare.add_argument("--run-extraction", action="store_true")
    benchmark_prepare.add_argument("--start", type=int, default=1)
    benchmark_prepare.add_argument("--limit", type=int, default=None)
    benchmark_prepare.add_argument("--scene-task-concurrency", type=int, default=3)
    benchmark_prepare.add_argument("--max-chunk-units", type=int, default=800)
    benchmark_prepare.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    benchmark_prepare.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)
    benchmark_prepare.add_argument("--db-path", type=Path, default=None)
    benchmark_prepare.add_argument("--chroma-dir", type=Path, default=None)
    benchmark_prepare.add_argument("--collection-name", default="dms_retrieval_documents")
    benchmark_prepare.add_argument("--chroma-upsert-batch-size", type=int, default=1000)
    benchmark_prepare.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    benchmark_prepare.add_argument("--overwrite", action="store_true")

    benchmark_run = subparsers.add_parser(
        "run-writing-benchmark",
        help="Run social simulation intent extraction, writing intent extraction, writing spec extraction, retrieval, social simulation, writing, and evaluation for eligible scenes.",
    )
    benchmark_run.add_argument("script_path", type=Path)
    benchmark_run.add_argument("--db-path", type=Path, required=True)
    benchmark_run.add_argument("--chroma-dir", type=Path, required=True)
    benchmark_run.add_argument("--output-dir", type=Path, required=True)
    benchmark_run.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    benchmark_run.add_argument("--llm-section", default="llm")
    benchmark_run.add_argument("--writing-llm-section", default="writing_llm")
    benchmark_run.add_argument("--embedding-section", default="embedding")
    benchmark_run.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    benchmark_run.add_argument("--eligibility-dir", type=Path, default=None)
    benchmark_run.add_argument("--eligibility-targets-file", type=Path, default=None)
    benchmark_run.add_argument("--target-scene-id", action="append", dest="target_scene_ids", default=None)
    benchmark_run.add_argument("--start-scene-order", type=int, default=None)
    benchmark_run.add_argument("--limit", type=int, default=3)
    benchmark_run.add_argument("--all-targets", action="store_true")
    benchmark_run.add_argument("--dry-run", action="store_true")
    benchmark_run.add_argument("--intent-only", action="store_true")
    benchmark_run.add_argument("--overwrite", action="store_true")
    benchmark_run.add_argument("--stop-on-error", action="store_true")
    benchmark_run.add_argument("--collection-name", default="dms_retrieval_documents")
    intent_level_choices = list(CLI_INTENT_LEVEL_CHOICES)
    benchmark_run.add_argument("--memory-intent-level", choices=intent_level_choices, default="writing_intent")
    benchmark_run.add_argument(
        "--social-simulation-intent-level",
        choices=intent_level_choices,
        default="social_simulation_intent",
    )
    benchmark_run.add_argument("--generation-intent-level", choices=intent_level_choices, default="writing_intent")
    benchmark_run.add_argument("--evaluation-intent-level", choices=intent_level_choices, default="writing_spec")
    benchmark_run.add_argument("--scene-top-k", type=int, default=5)
    benchmark_run.add_argument("--entity-memory-top-k", type=int, default=12)
    benchmark_run.add_argument("--global-scope-memory-top-k", type=int, default=8)
    benchmark_run.add_argument("--max-entity-memories-before-vector", type=int, default=50)
    benchmark_run.add_argument("--entity-match-limit", type=int, default=1)
    _add_reference_context_args(benchmark_run)
    benchmark_run.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    benchmark_run.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)
    benchmark_run.add_argument("--attribute-entity-type", action="append", dest="attribute_entity_types", default=None)
    benchmark_run.add_argument("--attribute-entity", action="append", dest="attribute_entity_names", default=None)
    benchmark_run.add_argument("--max-memories-per-entity", type=int, default=16)
    benchmark_run.add_argument("--previous-scene-context-mode", choices=["previous_scene", "none"], default="previous_scene")
    benchmark_run.add_argument("--previous-scene-context-max-chars", type=int, default=800)
    benchmark_run.add_argument("--style-reference-mode", choices=["previous_scene", "none"], default="none")
    benchmark_run.add_argument("--length-margin", type=float, default=0.2)
    benchmark_run.add_argument("--length-requirement", default="")
    benchmark_run.add_argument("--output-requirements", default=None)

    ui = subparsers.add_parser("launch-ui", help="Launch the Gradio DMS inspection and demo UI.")
    ui.add_argument("--script-path", type=Path, default=Path("data/raw/流浪地球2剧本.json"))
    ui.add_argument("--db-path", type=Path, default=Path("runs/assets/we2_scene12345_7types.sqlite"))
    ui.add_argument("--chroma-dir", type=Path, default=Path("runs/assets/we2_scene12345_7types_chroma_bge_m3"))
    ui.add_argument("--collection-name", default="dms_retrieval_documents_bge_m3")
    ui.add_argument("--benchmark-dir", type=Path, default=Path("runs/benchmark"))
    ui.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    ui.add_argument("--server-name", default="127.0.0.1")
    ui.add_argument("--server-port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")

    packet = subparsers.add_parser("visibility-packet", help="Build a conservative character visibility packet.")
    packet.add_argument("memory_path", type=Path)
    packet.add_argument("--character", required=True)
    packet.add_argument("--scene-id", required=True)
    packet.add_argument("--limit", type=int, default=20)

    grounded_packet = subparsers.add_parser(
        "grounded-visibility-packet",
        help="Build a character packet from explicit visibility records in a prefix world model.",
    )
    grounded_packet.add_argument("world_model_path", type=Path)
    grounded_packet.add_argument("--character", required=True)
    grounded_packet.add_argument("--scene-id", required=True)
    grounded_packet.add_argument("--limit", type=int, default=20)

    mvp = subparsers.add_parser("run-mvp-pipeline", help="Run the current MVP pipeline end to end.")
    mvp.add_argument("script_path", type=Path)
    mvp.add_argument("--output-root", type=Path, required=True)
    mvp.add_argument("--start", type=int, default=1)
    mvp.add_argument("--limit", type=int, default=5)
    mvp.add_argument("--model-config", type=Path, default=None)
    mvp.add_argument("--model-section", default="llm")
    mvp.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    mvp.add_argument("--model", default=None)
    mvp.add_argument("--base-url", default=None)
    mvp.add_argument("--auth-token", default=None)
    mvp.add_argument("--max-tokens", type=int, default=2048)
    mvp.add_argument("--temperature", type=float, default=0.0)
    mvp.add_argument("--timeout-seconds", type=int, default=120)
    mvp.add_argument("--overwrite", action="store_true")

    ordered = subparsers.add_parser(
        "run-scene-ordered-pipeline",
        help="Run ordered units while extracting independent tasks inside each unit concurrently.",
    )
    ordered.add_argument("script_path", type=Path)
    ordered.add_argument("--output-root", type=Path, required=True)
    ordered.add_argument("--base-output-root", type=Path, default=None)
    ordered.add_argument("--prompt-dir", type=Path, default=Path("task_specs/prompts"))
    ordered.add_argument("--scene-summary-task-settings", type=Path, default=Path("task_specs/task_settings/scene_summary_task.json"))
    ordered.add_argument("--scene-inventory-task-settings", type=Path, default=Path("task_specs/task_settings/scene_inventory_task.json"))
    ordered.add_argument("--kg-entity-task-settings", type=Path, default=Path("task_specs/task_settings/kg_entity_mentions_task.json"))
    ordered.add_argument("--kg-entity-refinement-task-settings", type=Path, default=Path("task_specs/task_settings/kg_entity_refinement_task.json"))
    ordered.add_argument("--episodic-memory-task-settings", type=Path, default=Path("task_specs/task_settings/episodic_memories_task.json"))
    ordered.add_argument(
        "--durable-relationship-task-settings",
        type=Path,
        default=Path("task_specs/task_settings/durable_relationships_task.json"),
    )
    ordered.add_argument("--prior-entity-context", type=Path, default=None)
    ordered.add_argument("--start", type=int, default=1)
    ordered.add_argument("--limit", type=int, default=5)
    ordered.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    ordered.add_argument("--scene-task-concurrency", type=int, default=3)
    ordered.add_argument("--max-chunk-units", type=int, default=800)
    ordered.add_argument("--unit-type", default=DEFAULT_UNIT_TYPE)
    ordered.add_argument("--unit-label", default=DEFAULT_UNIT_LABEL)
    ordered.add_argument("--model-config", type=Path, default=None)
    ordered.add_argument("--model-section", default="llm")
    ordered.add_argument("--provider", choices=["fake", "anthropic", "openai"], default="fake")
    ordered.add_argument("--model", default=None)
    ordered.add_argument("--base-url", default=None)
    ordered.add_argument("--auth-token", default=None)
    ordered.add_argument("--max-tokens", type=int, default=2048)
    ordered.add_argument("--temperature", type=float, default=0.0)
    ordered.add_argument("--timeout-seconds", type=int, default=120)
    ordered.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(raw_argv)
    setattr(args, "_raw_argv", tuple(raw_argv))

    if args.command == "inspect-script":
        scenes = load_script_scenes(args.path, unit_type=args.unit_type, unit_label=args.unit_label)
        payload = {
            "path": str(args.path),
            "scene_count": len(scenes),
            "unit_type": args.unit_type,
            "unit_label": args.unit_label,
            "content_char_count": sum(scene.character_count for scene in scenes),
            "first": [scene.to_dict() for scene in scenes[: max(args.limit, 0)]],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-script-units":
        scenes = load_script_scenes(args.path, unit_type=args.unit_type, unit_label=args.unit_label)
        write_jsonl(scenes, args.output)
        if args.summary:
            write_summary(scenes, args.summary, source_path=args.path)
        return 0

    if args.command == "build-scene-eligibility":
        summary = build_scene_eligibility_splits(args.script_path, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "render-prompt":
        loader = YAMLPromptLoader(args.prompt_dir)
        task_values = json.loads(args.task_json)
        static_values = json.loads(args.static_json)
        print(loader.render(args.prompt_id, task_values=task_values, static_values=static_values))
        return 0

    if args.command == "run-scene-inventory":
        llm_client = None if args.dry_run else _build_llm_client(args)
        summary = run_scene_inventory(
            SceneInventoryRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-scene-summary":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="scene_summary")
        summary = run_scene_summary(
            SceneSummaryRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                max_chunk_units=args.max_chunk_units,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-scene-events":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="scene_events")
        summary = run_scene_events(
            SceneEventRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                prior_context_path=args.prior_context,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-kg-entities":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="kg_entity_mentions")
        summary = run_kg_entity_mentions(
            KGEntityRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                prior_entity_context_path=args.prior_entity_context,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-durable-relationships":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="durable_relationships")
        summary = run_durable_relationships(
            DurableRelationshipRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                extracted_candidates_dir=args.extracted_candidates_dir,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-visibility-notes":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="visibility_notes")
        summary = run_visibility_notes(
            VisibilityNotesRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                extracted_candidates_dir=args.extracted_candidates_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-temporal-extraction":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="temporal_extraction")
        summary = run_temporal_extraction(
            TemporalExtractionRunConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                prior_timeline_path=args.prior_timeline,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-scene-inventory-memory":
        summary = build_scene_inventory_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-scene-summary-memory":
        summary = build_scene_summary_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-scene-event-memory":
        summary = build_scene_event_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-kg-entity-memory":
        summary = build_kg_entity_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-visibility-memory":
        summary = build_visibility_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-episodic-memory":
        summary = build_episodic_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-memory-timeline-index":
        summary = build_memory_timeline_index(
            MemoryTimelineIndexConfig(
                memory_path=args.memory_path,
                timeline_graph_path=args.timeline_graph,
                output_dir=args.output_dir,
                min_match_score=args.min_match_score,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-relationship-memory":
        from dms.memory import build_relationship_memory

        summary = build_relationship_memory(args.run_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-prefix-world-model":
        summary = build_prefix_world_model(
            canonical_dir=args.canonical_dir,
            event_memory_dir=args.event_memory_dir,
            visibility_memory_dir=args.visibility_memory_dir,
            kg_entity_memory_dir=args.kg_entity_memory_dir,
            episodic_memory_dir=args.episodic_memory_dir,
            relationship_memory_dir=args.relationship_memory_dir,
            scene_summary_dir=args.scene_summary_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-entity-resolution":
        author_context = load_author_entity_context(args.prior_entity_context)
        summary = build_entity_resolution_artifacts(
            args.world_model_path,
            args.output_dir,
            author_entities=author_entities_from_context(author_context),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-prefix-commits":
        summary = build_prefix_commits(args.world_model_path, args.entity_resolution_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-canonical-memory":
        summary = build_canonical_memory(args.staged_dir, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "query-memory":
        result = query_memory(
            args.memory_path,
            text=args.text,
            character=args.character,
            scene_id=args.scene_id,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-asset-store":
        summary = import_run_assets(
            AssetStoreImportConfig(
                db_path=args.output_db,
                ordered_run_dir=args.run_root,
                summary_memory_dir=args.summary_memory_dir,
                reset=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ingest-reference-library":
        summary = ingest_reference_library(
            ReferenceLibraryIngestConfig(
                input_path=args.input_path,
                output_dir=args.output_dir,
                max_chunk_chars=args.max_chunk_chars,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "extract-reference-items":
        llm_client = None if args.dry_run else _build_llm_client(args, fake_task="reference_items")
        summary = extract_reference_items(
            ReferenceItemExtractionConfig(
                library_dir=args.library_dir,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                task_settings_path=args.task_settings,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import-reference-items":
        summary = import_reference_items(
            ReferenceItemImportConfig(
                items_path=args.items_path,
                db_path=args.output_db,
                reset=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-reference-index":
        embedding_kwargs = _embedding_kwargs(args)
        summary = build_chroma_reference_index(
            ChromaReferenceIndexConfig(
                db_path=args.db_path,
                persist_dir=args.persist_dir,
                collection_name=args.collection_name,
                reset=args.overwrite,
                upsert_batch_size=args.upsert_batch_size,
                **embedding_kwargs,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "query-entity-memories":
        result = get_entity_memories(
            args.db_path,
            entity_ref=args.entity,
            before_scene_id=args.before_scene,
            before_scene_order=args.before_scene_order,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-chroma-index":
        embedding_kwargs = _embedding_kwargs(args)
        summary = build_chroma_memory_index(
            ChromaMemoryIndexConfig(
                db_path=args.db_path,
                persist_dir=args.persist_dir,
                collection_name=args.collection_name,
                reset=args.overwrite,
                upsert_batch_size=args.upsert_batch_size,
                **embedding_kwargs,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search-entity-memories":
        embedding_kwargs = _embedding_kwargs(args)
        result = search_entity_memories(
            args.db_path,
            persist_dir=args.chroma_dir,
            query=args.query,
            collection_name=args.collection_name,
            entity_ref=args.entity,
            before_scene_id=args.before_scene,
            before_scene_order=args.before_scene_order,
            top_k=args.top_k,
            **embedding_kwargs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-memory-packet":
        embedding_kwargs = _embedding_kwargs(args)
        packet_result = build_memory_packet(
            MemoryPacketConfig(
                db_path=args.db_path,
                chroma_dir=args.chroma_dir,
                writing_intent=_read_writing_intent(args),
                before_scene_id=args.before_scene,
                before_scene_order=args.before_scene_order,
                unit_type=args.unit_type,
                unit_label=args.unit_label,
                scene_top_k=args.scene_top_k,
                entity_memory_top_k=args.entity_memory_top_k,
                global_scope_memory_top_k=args.global_scope_memory_top_k,
                max_entity_memories_before_vector=args.max_entity_memories_before_vector,
                entity_match_limit=args.entity_match_limit,
                collection_name=args.collection_name,
                include_reference_context=args.include_reference_context,
                reference_db_path=args.reference_db_path,
                reference_chroma_dir=args.reference_chroma_dir,
                reference_collection_name=args.reference_collection_name,
                reference_top_k=args.reference_top_k,
                reference_author_top_k=args.reference_author_top_k,
                reference_character_top_k=args.reference_character_top_k,
                reference_style_top_k=args.reference_style_top_k,
                reference_timeline_top_k=args.reference_timeline_top_k,
                **embedding_kwargs,
            )
        )
        rendered = (
            format_memory_packet_markdown(packet_result)
            if args.format == "markdown"
            else json.dumps(packet_result, ensure_ascii=False, indent=2) + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "evaluate-writing":
        reference_text = _read_text_arg(
            args.reference_text,
            args.reference_text_file,
            label="reference-text",
        ) if args.reference_text is not None or args.reference_text_file is not None else None
        scene_reference_text = _read_reference_scene_text(args.reference_script, args.reference_scene_id)
        if reference_text is not None and scene_reference_text is not None:
            raise ValueError("Use either explicit reference text or --reference-script/--reference-scene-id, not both")
        reference_text = reference_text if reference_text is not None else scene_reference_text
        summary = evaluate_writing(
            WritingEvaluationConfig(
                writing_intent=_read_text_arg(
                    args.writing_intent,
                    args.writing_intent_file,
                    label="writing-intent",
                ),
                generated_text=_read_text_arg(
                    args.generated_text,
                    args.generated_text_file,
                    label="generated-text",
                ),
                memory_packet=_read_text_arg(
                    args.memory_packet,
                    args.memory_packet_file,
                    label="memory-packet",
                ),
                reference_text=reference_text,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                overwrite=args.overwrite,
            ),
            llm_client=_build_llm_client(args),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-entity-attribute-cards":
        summary = build_entity_attribute_cards(
            AttributeCardConfig(
                memory_packet_path=args.memory_packet_path,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                entity_types=tuple(args.entity_types or ["character"]),
                entity_names=tuple(args.entity_names or []),
                max_memories_per_entity=args.max_memories_per_entity,
                overwrite=args.overwrite,
            ),
            llm_client=_build_llm_client(args),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-scene-disposition-notes":
        summary = build_scene_disposition_notes(
            SceneDispositionNoteConfig(
                attribute_cards_path=args.attribute_cards_path,
                output_dir=args.output_dir,
                social_simulation_intent=args.social_simulation_intent,
                memory_packet_path=args.memory_packet,
                prompt_dir=args.prompt_dir,
                entity_types=tuple(args.entity_types or ["character"]),
                entity_names=tuple(args.entity_names or []),
                max_relevant_memories_per_entity=args.max_relevant_memories_per_entity,
                overwrite=args.overwrite,
            ),
            llm_client=_build_llm_client(args),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-social-simulation":
        summary = run_social_simulation(
            SocialSimulationConfig(
                attribute_cards_path=args.attribute_cards_path,
                social_simulation_intent=_read_social_simulation_intent(args),
                scene_disposition_notes_path=args.scene_disposition_notes,
                output_dir=args.output_dir,
                prompt_dir=args.prompt_dir,
                overwrite=args.overwrite,
            ),
            llm_client=_build_llm_client(args),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-writing-social":
        summary = generate_writing_with_social_simulation(
            SocialWritingGenerationConfig(
                writing_request=args.writing_request,
                memory_packet_path=args.memory_packet_file,
                attribute_cards_path=args.attribute_cards_file,
                social_simulation_path=args.social_simulation_file,
                output_dir=args.output_dir,
                model_config_path=args.model_config,
                model_section=args.model_section,
                prompt_dir=args.prompt_dir,
                previous_scene_context=args.previous_scene_context,
                previous_scene_context_path=args.previous_scene_context_file,
                previous_scene_context_script=args.previous_scene_context_script,
                previous_scene_context_scene_id=args.previous_scene_context_scene_id,
                previous_scene_context_max_chars=args.previous_scene_context_max_chars,
                style_reference_path=args.style_reference_file,
                style_reference_script=args.style_reference_script,
                style_reference_scene_id=args.style_reference_scene_id,
                length_requirement=args.length_requirement,
                output_requirements=args.output_requirements,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-writing-e2e":
        summary = run_writing_e2e(
            WritingE2EConfig(
                db_path=args.db_path,
                chroma_dir=args.chroma_dir,
                writing_intent=_read_writing_intent(args),
                output_dir=args.output_dir,
                model_config_path=args.model_config,
                llm_section=args.llm_section,
                writing_llm_section=args.writing_llm_section,
                embedding_section=args.embedding_section,
                prompt_dir=args.prompt_dir,
                before_scene_id=args.before_scene,
                before_scene_order=args.before_scene_order,
                unit_type=args.unit_type,
                unit_label=args.unit_label,
                scene_top_k=args.scene_top_k,
                entity_memory_top_k=args.entity_memory_top_k,
                global_scope_memory_top_k=args.global_scope_memory_top_k,
                max_entity_memories_before_vector=args.max_entity_memories_before_vector,
                entity_match_limit=args.entity_match_limit,
                collection_name=args.collection_name,
                include_reference_context=args.include_reference_context,
                reference_db_path=args.reference_db_path,
                reference_chroma_dir=args.reference_chroma_dir,
                reference_collection_name=args.reference_collection_name,
                reference_top_k=args.reference_top_k,
                reference_author_top_k=args.reference_author_top_k,
                reference_character_top_k=args.reference_character_top_k,
                reference_style_top_k=args.reference_style_top_k,
                reference_timeline_top_k=args.reference_timeline_top_k,
                attribute_entity_types=tuple(args.attribute_entity_types or ["character"]),
                attribute_entity_names=tuple(args.attribute_entity_names or []),
                max_memories_per_entity=args.max_memories_per_entity,
                previous_scene_context=args.previous_scene_context,
                previous_scene_context_path=args.previous_scene_context_file,
                previous_scene_context_script=args.previous_scene_context_script,
                previous_scene_context_scene_id=args.previous_scene_context_scene_id,
                previous_scene_context_max_chars=args.previous_scene_context_max_chars,
                style_reference_path=args.style_reference_file,
                style_reference_script=args.style_reference_script,
                style_reference_scene_id=args.style_reference_scene_id,
                length_requirement=args.length_requirement,
                output_requirements=args.output_requirements,
                reference_text=args.reference_text,
                reference_text_file=args.reference_text_file,
                reference_script=args.reference_script,
                reference_scene_id=args.reference_scene_id,
                overwrite=args.overwrite,
                skip_evaluation=args.skip_evaluation,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-writing-benchmark":
        summary = prepare_writing_benchmark_assets(
            WritingBenchmarkPrepareConfig(
                script_path=args.script_path,
                output_dir=args.output_dir,
                model_config_path=args.model_config,
                llm_section=args.llm_section,
                embedding_section=args.embedding_section,
                prompt_dir=args.prompt_dir,
                ordered_run_dir=args.ordered_run_dir,
                extraction_output_root=args.extraction_output_root,
                run_extraction=args.run_extraction,
                start=args.start,
                limit=args.limit,
                scene_task_concurrency=args.scene_task_concurrency,
                max_chunk_units=args.max_chunk_units,
                unit_type=args.unit_type,
                unit_label=args.unit_label,
                db_path=args.db_path,
                chroma_dir=args.chroma_dir,
                collection_name=args.collection_name,
                chroma_upsert_batch_size=args.chroma_upsert_batch_size,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-writing-benchmark":
        summary = run_writing_benchmark(
            WritingBenchmarkRunConfig(
                script_path=args.script_path,
                db_path=args.db_path,
                chroma_dir=args.chroma_dir,
                output_dir=args.output_dir,
                model_config_path=args.model_config,
                llm_section=args.llm_section,
                writing_llm_section=args.writing_llm_section,
                embedding_section=args.embedding_section,
                prompt_dir=args.prompt_dir,
                eligibility_dir=args.eligibility_dir,
                eligibility_targets_file=args.eligibility_targets_file,
                target_scene_ids=tuple(args.target_scene_ids or []),
                start_scene_order=args.start_scene_order,
                limit=None if args.all_targets else args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                stop_on_error=args.stop_on_error,
                intent_only=args.intent_only,
                collection_name=args.collection_name,
                memory_intent_level=args.memory_intent_level,
                social_simulation_intent_level=args.social_simulation_intent_level,
                generation_intent_level=args.generation_intent_level,
                evaluation_intent_level=args.evaluation_intent_level,
                scene_top_k=args.scene_top_k,
                entity_memory_top_k=args.entity_memory_top_k,
                global_scope_memory_top_k=args.global_scope_memory_top_k,
                max_entity_memories_before_vector=args.max_entity_memories_before_vector,
                entity_match_limit=args.entity_match_limit,
                include_reference_context=args.include_reference_context,
                reference_db_path=args.reference_db_path,
                reference_chroma_dir=args.reference_chroma_dir,
                reference_collection_name=args.reference_collection_name,
                reference_top_k=args.reference_top_k,
                reference_author_top_k=args.reference_author_top_k,
                reference_character_top_k=args.reference_character_top_k,
                reference_style_top_k=args.reference_style_top_k,
                reference_timeline_top_k=args.reference_timeline_top_k,
                unit_type=args.unit_type,
                unit_label=args.unit_label,
                attribute_entity_types=tuple(args.attribute_entity_types or ["character"]),
                attribute_entity_names=tuple(args.attribute_entity_names or []),
                max_memories_per_entity=args.max_memories_per_entity,
                previous_scene_context_mode=args.previous_scene_context_mode,
                previous_scene_context_max_chars=args.previous_scene_context_max_chars,
                style_reference_mode=args.style_reference_mode,
                length_margin=args.length_margin,
                length_requirement=args.length_requirement,
                output_requirements=args.output_requirements
                if args.output_requirements is not None
                else WritingBenchmarkRunConfig.output_requirements,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "launch-ui":
        from dms.ui.gradio_app import main as launch_gradio_app

        return launch_gradio_app(
            [
                "--script-path",
                str(args.script_path),
                "--db-path",
                str(args.db_path),
                "--chroma-dir",
                str(args.chroma_dir),
                "--collection-name",
                args.collection_name,
                "--benchmark-dir",
                str(args.benchmark_dir),
                "--model-config",
                str(args.model_config),
                "--server-name",
                args.server_name,
                "--server-port",
                str(args.server_port),
                *(["--share"] if args.share else []),
            ]
        )

    if args.command == "visibility-packet":
        result = build_visibility_packet(
            args.memory_path,
            character=args.character,
            scene_id=args.scene_id,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "grounded-visibility-packet":
        result = build_visibility_grounded_packet(
            args.world_model_path,
            character=args.character,
            scene_id=args.scene_id,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-mvp-pipeline":
        output_root = args.output_root
        if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output root exists and is not empty: {output_root}")
        extraction_dir = output_root / "scene_inventory"
        staged_dir = output_root / "staged_memory"
        canonical_dir = output_root / "canonical_memory"
        llm_client = _build_llm_client(args)
        extraction_summary = run_scene_inventory(
            SceneInventoryRunConfig(
                script_path=args.script_path,
                output_dir=extraction_dir,
                start=args.start,
                limit=args.limit,
                dry_run=False,
                overwrite=args.overwrite,
            ),
            llm_client=llm_client,
        )
        staged_summary = build_scene_inventory_memory(extraction_dir, staged_dir)
        canonical_summary = build_canonical_memory(staged_dir, canonical_dir)
        summary = {
            "status": "complete",
            "output_root": str(output_root),
            "extraction": extraction_summary,
            "staged_memory": staged_summary,
            "canonical_memory": canonical_summary,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-scene-ordered-pipeline":
        llm_clients = None if args.dry_run else _build_task_llm_clients(args)
        summary = run_scene_ordered_pipeline(
            SceneOrderedPipelineConfig(
                script_path=args.script_path,
                output_root=args.output_root,
                base_output_root=args.base_output_root,
                prompt_dir=args.prompt_dir,
                scene_summary_task_settings_path=args.scene_summary_task_settings,
                scene_inventory_task_settings_path=args.scene_inventory_task_settings,
                kg_entity_task_settings_path=args.kg_entity_task_settings,
                kg_entity_refinement_task_settings_path=args.kg_entity_refinement_task_settings,
                episodic_memory_task_settings_path=args.episodic_memory_task_settings,
                durable_relationship_task_settings_path=args.durable_relationship_task_settings,
                prior_entity_context_path=args.prior_entity_context,
                start=args.start,
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                scene_task_concurrency=args.scene_task_concurrency,
                max_chunk_units=args.max_chunk_units,
                unit_type=args.unit_type,
                unit_label=args.unit_label,
            ),
            llm_clients=llm_clients,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
