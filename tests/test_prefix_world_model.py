from pathlib import Path

from dms.llm import FakeKGEntityMentionClient, FakeSceneEventClient, FakeSceneInventoryClient, FakeVisibilityNotesClient
from dms.memory import (
    build_canonical_memory,
    build_kg_entity_memory,
    build_prefix_world_model,
    build_scene_event_memory,
    build_scene_inventory_memory,
    build_visibility_grounded_packet,
    build_visibility_memory,
)
from dms.runners import (
    KGEntityRunConfig,
    SceneEventRunConfig,
    SceneInventoryRunConfig,
    VisibilityNotesRunConfig,
    run_scene_events,
    run_scene_inventory,
    run_kg_entity_mentions,
    run_visibility_notes,
)


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_prefix_world_model_and_grounded_packet(tmp_path: Path) -> None:
    inventory_run = tmp_path / "inventory_run"
    staged_inventory = tmp_path / "staged_inventory"
    canonical = tmp_path / "canonical"
    event_run = tmp_path / "event_run"
    event_memory = tmp_path / "event_memory"
    kg_entity_run = tmp_path / "kg_entity_run"
    kg_entity_memory = tmp_path / "kg_entity_memory"
    visibility_run = tmp_path / "visibility_run"
    visibility_memory = tmp_path / "visibility_memory"
    world_model = tmp_path / "world_model"

    run_scene_inventory(
        SceneInventoryRunConfig(script_path=SCRIPT_PATH, output_dir=inventory_run, limit=2, dry_run=False),
        llm_client=FakeSceneInventoryClient(),
    )
    build_scene_inventory_memory(inventory_run, staged_inventory)
    build_canonical_memory(staged_inventory, canonical)

    run_scene_events(
        SceneEventRunConfig(script_path=SCRIPT_PATH, output_dir=event_run, limit=2, dry_run=False),
        llm_client=FakeSceneEventClient(),
    )
    build_scene_event_memory(event_run, event_memory)

    run_kg_entity_mentions(
        KGEntityRunConfig(script_path=SCRIPT_PATH, output_dir=kg_entity_run, limit=2, dry_run=False),
        llm_client=FakeKGEntityMentionClient(),
    )
    build_kg_entity_memory(kg_entity_run, kg_entity_memory)

    run_visibility_notes(
        VisibilityNotesRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=visibility_run,
            extracted_candidates_dir=event_run,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeVisibilityNotesClient(),
    )
    build_visibility_memory(visibility_run, visibility_memory)

    summary = build_prefix_world_model(
        canonical_dir=canonical,
        event_memory_dir=event_memory,
        visibility_memory_dir=visibility_memory,
        kg_entity_memory_dir=kg_entity_memory,
        output_dir=world_model,
    )
    packet = build_visibility_grounded_packet(world_model, character="FAKE_CHARACTER", scene_id="scene_0002")

    assert summary["scene_count"] == 2
    assert summary["event_count"] == 2
    assert summary["kg_entity_mention_count"] == 8
    assert summary["visibility_record_count"] == 2
    assert (world_model / "prefix_world_model.json").is_file()
    assert packet["counts"]["visible_visibility_records"] == 2
    assert packet["counts"]["hidden_or_blocked_items"] == 2
