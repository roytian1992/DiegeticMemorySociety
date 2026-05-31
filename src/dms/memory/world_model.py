from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_prefix_world_model(
    *,
    canonical_dir: str | Path,
    event_memory_dir: str | Path | None = None,
    output_dir: str | Path,
    kg_entity_memory_dir: str | Path | None = None,
    episodic_memory_dir: str | Path | None = None,
    relationship_memory_dir: str | Path | None = None,
    scene_summary_dir: str | Path | None = None,
    visibility_memory_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Merge current staged/canonical layers into one prefix world-model artifact."""

    canonical_path = Path(canonical_dir)
    event_path = Path(event_memory_dir) if event_memory_dir else None
    visibility_path = Path(visibility_memory_dir) if visibility_memory_dir else None
    kg_entity_path = Path(kg_entity_memory_dir) if kg_entity_memory_dir else None
    episodic_path = Path(episodic_memory_dir) if episodic_memory_dir else None
    relationship_path = Path(relationship_memory_dir) if relationship_memory_dir else None
    scene_summary_path = Path(scene_summary_dir) if scene_summary_dir else None
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    canonical = _read_json(canonical_path / "canonical_memory.json")
    events = _read_jsonl(event_path / "events.jsonl") if event_path else []
    knowledge_transfers = _read_jsonl(event_path / "knowledge_transfers.jsonl") if event_path else []
    state_changes = _read_jsonl(event_path / "state_changes.jsonl") if event_path else []
    thread_candidates = _read_jsonl(event_path / "thread_candidates.jsonl") if event_path else []
    visibility_records = _read_jsonl(visibility_path / "visibility_records.jsonl") if visibility_path else []
    hidden_items = _read_jsonl(visibility_path / "hidden_or_future_sensitive_items.jsonl") if visibility_path else []
    kg_entity_mentions = _read_jsonl(kg_entity_path / "entity_mentions.jsonl") if kg_entity_path else []
    scene_tags = _read_jsonl(kg_entity_path / "scene_tags.jsonl") if kg_entity_path else []
    unresolved_kg_mentions = _read_jsonl(kg_entity_path / "unresolved_mentions.jsonl") if kg_entity_path else []
    episodic_memories = _read_jsonl(episodic_path / "episodic_memories.jsonl") if episodic_path else []
    entity_memory_links = _read_jsonl(episodic_path / "entity_memory_links.jsonl") if episodic_path else []
    relationship_observations = (
        _read_jsonl(relationship_path / "relationship_observations.jsonl") if relationship_path else []
    )
    scene_summaries = _read_jsonl(scene_summary_path / "scene_summaries.jsonl") if scene_summary_path else []

    scenes = canonical.get("scenes", []) if isinstance(canonical.get("scenes"), list) else []
    scene_order = {str(scene.get("scene_id")): index for index, scene in enumerate(scenes, start=1)}
    entity_memory_index = _build_entity_memory_index(entity_memory_links, episodic_memories)
    world_model = {
        "memory_layer": "prefix_world_model",
        "source_paths": {
            "canonical_dir": str(canonical_path),
            "event_memory_dir": str(event_path) if event_path else None,
            "visibility_memory_dir": str(visibility_path) if visibility_path else None,
            "kg_entity_memory_dir": str(kg_entity_path) if kg_entity_path else None,
            "episodic_memory_dir": str(episodic_path) if episodic_path else None,
            "relationship_memory_dir": str(relationship_path) if relationship_path else None,
            "scene_summary_dir": str(scene_summary_path) if scene_summary_path else None,
        },
        "scenes": scenes,
        "scene_index": canonical.get("scene_index", {}),
        "scene_order": scene_order,
        "characters": canonical.get("characters", []),
        "objects": canonical.get("objects", []),
        "stated_facts": canonical.get("stated_facts", []),
        "open_questions": canonical.get("open_questions", []),
        "scene_summaries": scene_summaries,
        "events": events,
        "knowledge_transfers": knowledge_transfers,
        "state_changes": state_changes,
        "thread_candidates": thread_candidates,
        "kg_entity_mentions": kg_entity_mentions,
        "scene_tags": scene_tags,
        "unresolved_kg_mentions": unresolved_kg_mentions,
        "episodic_memories": episodic_memories,
        "entity_memory_links": entity_memory_links,
        "entity_memory_index": entity_memory_index,
        "relationship_observations": relationship_observations,
        "visibility_records": visibility_records,
        "hidden_or_future_sensitive_items": hidden_items,
    }

    paths = {
        "world_model": out_path / "prefix_world_model.json",
        "summary": out_path / "summary.json",
    }
    paths["world_model"].write_text(json.dumps(world_model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "output_dir": str(out_path),
        "source_paths": world_model["source_paths"],
        "scene_count": len(scenes),
        "character_count": len(world_model["characters"]),
        "object_count": len(world_model["objects"]),
        "stated_fact_count": len(world_model["stated_facts"]),
        "open_question_count": len(world_model["open_questions"]),
        "scene_summary_count": len(scene_summaries),
        "event_count": len(events),
        "knowledge_transfer_count": len(knowledge_transfers),
        "state_change_count": len(state_changes),
        "thread_candidate_count": len(thread_candidates),
        "kg_entity_mention_count": len(kg_entity_mentions),
        "scene_tag_count": len(scene_tags),
        "unresolved_kg_mention_count": len(unresolved_kg_mentions),
        "episodic_memory_count": len(episodic_memories),
        "entity_memory_link_count": len(entity_memory_links),
        "relationship_observation_count": len(relationship_observations),
        "visibility_record_count": len(visibility_records),
        "hidden_or_future_sensitive_count": len(hidden_items),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_visibility_grounded_packet(
    world_model_path_or_dir: str | Path,
    *,
    character: str,
    scene_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Build a packet from explicit visibility records instead of mention-only visibility."""

    world_model = load_prefix_world_model(world_model_path_or_dir)
    scene_order = _scene_order(world_model)
    target_order = scene_order.get(scene_id)
    if target_order is None:
        raise ValueError(f"Unknown scene_id: {scene_id}")

    character_record = _find_character(world_model, character)
    if character_record is None:
        raise ValueError(f"Unknown character: {character}")

    visible_records = [
        record
        for record in world_model.get("visibility_records", [])
        if _matches_character(str(record.get("character", "")), character)
        and scene_order.get(str(record.get("scene_id")), 10**9) <= target_order
        and str(record.get("visibility", "")) != "unaware"
    ][:limit]
    visible_scene_ids = sorted({str(record.get("scene_id")) for record in visible_records}, key=lambda sid: scene_order.get(sid, 0))
    visible_fact_or_event_texts = [str(record.get("fact_or_event", "")) for record in visible_records]
    visible_events = _records_in_scenes(world_model.get("events", []), visible_scene_ids, limit=limit)
    visible_transfers = [
        record
        for record in _records_in_scenes(world_model.get("knowledge_transfers", []), visible_scene_ids, limit=limit)
        if _matches_character(str(record.get("receiver", "")), character)
        or _matches_character(str(record.get("source", "")), character)
    ][:limit]
    visible_facts = _records_in_scenes(world_model.get("stated_facts", []), visible_scene_ids, limit=limit)

    hidden_items = [
        item
        for item in world_model.get("hidden_or_future_sensitive_items", [])
        if scene_order.get(str(item.get("scene_id")), 10**9) <= target_order
        and _is_hidden_from(item, character)
    ][:limit]
    blocked_future_scenes = [
        scene_id
        for scene_id, order in scene_order.items()
        if order > target_order
    ]

    return {
        "packet_type": "visibility_grounded_character_packet",
        "policy": "explicit_visibility_records_prefix_only",
        "character": character_record,
        "target_scene_id": scene_id,
        "target_scene": world_model.get("scene_index", {}).get(scene_id),
        "visible_scene_ids": visible_scene_ids,
        "visible_visibility_records": visible_records,
        "visible_fact_or_event_texts": visible_fact_or_event_texts,
        "visible_events": visible_events,
        "visible_knowledge_transfers": visible_transfers,
        "visible_stated_facts": visible_facts,
        "hidden_or_blocked_items": hidden_items,
        "blocked": {
            "future_scene_ids": blocked_future_scenes,
            "reason": "after target scene discourse order",
        },
        "counts": {
            "visible_visibility_records": len(visible_records),
            "visible_events": len(visible_events),
            "visible_knowledge_transfers": len(visible_transfers),
            "visible_stated_facts": len(visible_facts),
            "hidden_or_blocked_items": len(hidden_items),
        },
    }


def load_prefix_world_model(world_model_path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(world_model_path_or_dir)
    if path.is_dir():
        path = path / "prefix_world_model.json"
    return _read_json(path)


def _find_character(world_model: dict[str, Any], character: str) -> dict[str, Any] | None:
    query = _normalize_name(character)
    for record in world_model.get("characters", []):
        names = [record.get("canonical_name", ""), *record.get("aliases", [])]
        if any(_normalize_name(str(name)) == query for name in names):
            return record
    return _find_kg_actor(world_model, character)


def _find_kg_actor(world_model: dict[str, Any], character: str) -> dict[str, Any] | None:
    query = _normalize_name(character)
    if not query:
        return None

    matching_mentions: list[dict[str, Any]] = []
    for record in world_model.get("kg_entity_mentions", []):
        if not isinstance(record, dict):
            continue
        entity_type = str(record.get("entity_type") or "")
        if entity_type not in {"character", "group"}:
            continue
        surface = str(record.get("surface") or "")
        canonical_hint = str(record.get("canonical_hint") or "")
        names = [surface, canonical_hint]
        if any(_normalize_name(name) == query for name in names):
            matching_mentions.append(record)

    if not matching_mentions:
        return None

    aliases = sorted(
        {
            str(value)
            for record in matching_mentions
            for value in (record.get("surface"), record.get("canonical_hint"))
            if str(value or "").strip()
        }
    )
    scene_ids = sorted({str(record.get("scene_id")) for record in matching_mentions if record.get("scene_id")})
    canonical_name = next(
        (
            str(record.get("canonical_hint") or record.get("surface"))
            for record in matching_mentions
            if str(record.get("canonical_hint") or record.get("surface") or "").strip()
        ),
        character,
    )
    entity_type = "character" if any(record.get("entity_type") == "character" for record in matching_mentions) else "group"
    return {
        "memory_layer": "prefix_world_model",
        "entity_id": None,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": aliases,
        "scene_ids": scene_ids,
        "mentions": matching_mentions,
        "source": "kg_entity_mentions",
    }
    return None


def _matches_character(value: str, character: str) -> bool:
    value_key = _normalize_name(value)
    character_key = _normalize_name(character)
    return bool(value_key and character_key and (value_key == character_key or character_key in value_key))


def _is_hidden_from(item: dict[str, Any], character: str) -> bool:
    hidden_from = item.get("hidden_from")
    if not isinstance(hidden_from, list):
        return False
    return any(_matches_character(str(value), character) or str(value).strip() in {"所有场景内角色", "所有角色"} for value in hidden_from)


def _records_in_scenes(records: Any, scene_ids: list[str], *, limit: int) -> list[dict[str, Any]]:
    allowed = set(scene_ids)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict) and record.get("scene_id") in allowed][:limit]


def _scene_order(world_model: dict[str, Any]) -> dict[str, int]:
    order = world_model.get("scene_order")
    if isinstance(order, dict):
        return {str(key): int(value) for key, value in order.items()}
    scenes = world_model.get("scenes", []) if isinstance(world_model.get("scenes"), list) else []
    return {str(scene.get("scene_id")): index for index, scene in enumerate(scenes, start=1)}


def _normalize_name(name: str) -> str:
    return "".join(str(name or "").lower().split())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _build_entity_memory_index(
    entity_memory_links: list[dict[str, Any]],
    episodic_memories: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    memory_by_id = {str(record.get("record_id")): record for record in episodic_memories}
    index: dict[str, list[dict[str, Any]]] = {}
    for link in entity_memory_links:
        entity = str(link.get("entity") or "").strip()
        memory_id = str(link.get("memory_record_id") or "")
        if not entity or not memory_id:
            continue
        memory = memory_by_id.get(memory_id, {})
        index.setdefault(entity, []).append(
            {
                "memory_record_id": memory_id,
                "scene_id": link.get("scene_id"),
                "timeline_index": memory.get("timeline_index"),
                "sequence_index": memory.get("sequence_index"),
                "memory_type": memory.get("memory_type"),
                "summary": memory.get("summary"),
                "link_role": link.get("link_role"),
                "evidence": link.get("evidence") or memory.get("evidence", ""),
            }
        )
    for records in index.values():
        records.sort(key=lambda item: (str(item.get("scene_id")), int(item.get("sequence_index") or 0)))
    return index
