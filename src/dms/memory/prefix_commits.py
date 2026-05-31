from __future__ import annotations

import json
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from dms.memory.entity_resolution import resolve_name_variants
from dms.memory.world_model import load_prefix_world_model


def build_prefix_commits(
    world_model_path_or_dir: str | Path,
    entity_resolution_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay extracted artifacts into per-parent-scene commits and snapshots.

    The commit layer is deliberately append-only: it records what each parent
    scene adds or updates, while materialized snapshots provide the current
    prefix view after each scene.
    """

    world_model = load_prefix_world_model(world_model_path_or_dir)
    resolution_path = Path(entity_resolution_dir)
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    snapshots_dir = out_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    entities = _read_jsonl(resolution_path / "entities.jsonl")
    relationship_updates = _read_jsonl(resolution_path / "relationship_updates.jsonl")
    entity_by_id = {str(entity.get("entity_id")): entity for entity in entities if entity.get("entity_id")}
    alias_index = _alias_index(entities, _read_jsonl(resolution_path / "aliases.jsonl"))
    unit_parent_map = _unit_parent_map(world_model)
    parent_order = _parent_order(world_model, unit_parent_map)

    kg_mentions = _group_by_parent(world_model.get("kg_entity_mentions"), unit_parent_map)
    scene_tags = _group_by_parent(world_model.get("scene_tags"), unit_parent_map)
    unresolved_mentions = _group_by_parent(world_model.get("unresolved_kg_mentions"), unit_parent_map)
    episodic_memories = _group_by_parent(world_model.get("episodic_memories"), unit_parent_map)
    entity_memory_links = _group_by_parent(world_model.get("entity_memory_links"), unit_parent_map)
    relationship_updates_by_parent = _group_by_parent(relationship_updates, unit_parent_map)
    stated_facts = _group_by_parent(world_model.get("stated_facts"), unit_parent_map)
    open_questions = _group_by_parent(world_model.get("open_questions"), unit_parent_map)

    all_parents = _merge_parent_order(
        parent_order,
        kg_mentions,
        scene_tags,
        unresolved_mentions,
        episodic_memories,
        entity_memory_links,
        relationship_updates_by_parent,
        stated_facts,
        open_questions,
    )

    state = _empty_state()
    commits: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    snapshot_paths: list[Path] = []

    for commit_index, parent_id in enumerate(all_parents, start=1):
        commit_id = f"commit_{commit_index:04d}"
        before_operation_count = len(operations)
        unit_ids = _unit_ids_for_parent(world_model, unit_parent_map, parent_id)

        for record in kg_mentions.get(parent_id, []):
            _commit_entity_mention(
                state=state,
                operations=operations,
                commit_id=commit_id,
                parent_id=parent_id,
                record=record,
                alias_index=alias_index,
                entity_by_id=entity_by_id,
            )

        for record in unresolved_mentions.get(parent_id, []):
            _append_operation(
                operations,
                commit_id=commit_id,
                parent_id=parent_id,
                operation_type="entity_unresolved",
                payload={
                    "surface": record.get("surface", ""),
                    "reason": record.get("reason", ""),
                    "evidence": record.get("evidence", ""),
                    "source_record_id": record.get("record_id", ""),
                },
            )
            state["unresolved_entity_mentions"].append(_compact_record(record))

        for record in scene_tags.get(parent_id, []):
            compact_tag = _compact_scene_tag(record)
            state["scene_tags"].append(compact_tag)
            _append_operation(
                operations,
                commit_id=commit_id,
                parent_id=parent_id,
                operation_type="scene_tag_added",
                payload=compact_tag,
            )

        for record in stated_facts.get(parent_id, []):
            state["stated_facts"].append(_compact_record(record))
            _append_operation(
                operations,
                commit_id=commit_id,
                parent_id=parent_id,
                operation_type="stated_fact_added",
                payload={
                    "record_id": record.get("record_id", ""),
                    "proposition": record.get("proposition", ""),
                    "evidence": record.get("evidence", ""),
                },
            )

        for record in open_questions.get(parent_id, []):
            state["open_questions"].append({**_compact_record(record), "status": "open"})
            _append_operation(
                operations,
                commit_id=commit_id,
                parent_id=parent_id,
                operation_type="open_question_added",
                payload={
                    "record_id": record.get("record_id", ""),
                    "question": record.get("question", ""),
                    "evidence": record.get("evidence", ""),
                },
            )

        links_by_memory = _links_by_memory(entity_memory_links.get(parent_id, []))
        for record in episodic_memories.get(parent_id, []):
            _commit_memory(
                state=state,
                operations=operations,
                commit_id=commit_id,
                parent_id=parent_id,
                memory=record,
                links=links_by_memory.get(str(record.get("record_id")), []),
                alias_index=alias_index,
                entity_by_id=entity_by_id,
            )

        for record in relationship_updates_by_parent.get(parent_id, []):
            _commit_relationship_update(state, operations, commit_id, parent_id, record)

        snapshot = _snapshot(state, commit_id=commit_id, parent_id=parent_id)
        snapshot_path = snapshots_dir / f"prefix_after_{parent_id}.json"
        _write_json(snapshot_path, snapshot)
        snapshot_paths.append(snapshot_path)

        commit_operations = operations[before_operation_count:]
        commits.append(
            {
                "commit_id": commit_id,
                "parent_unit_id": parent_id,
                "unit_ids": unit_ids,
                "operation_count": len(commit_operations),
                "operation_type_counts": _operation_type_counts(commit_operations),
                "snapshot_path": str(snapshot_path),
                "snapshot_counts": snapshot["counts"],
            }
        )

    current_snapshot_path = out_path / "current_snapshot.json"
    if snapshot_paths:
        current_snapshot_path.write_text(snapshot_paths[-1].read_text(encoding="utf-8"), encoding="utf-8")
    else:
        _write_json(current_snapshot_path, _snapshot(state, commit_id="", parent_id=""))

    commits_path = out_path / "commits.jsonl"
    operations_path = out_path / "operations.jsonl"
    summary_path = out_path / "summary.json"
    _write_jsonl(commits_path, commits)
    _write_jsonl(operations_path, operations)

    summary = {
        "source_world_model": str(Path(world_model_path_or_dir)),
        "source_entity_resolution_dir": str(resolution_path),
        "output_dir": str(out_path),
        "commit_count": len(commits),
        "operation_count": len(operations),
        "current_counts": _read_json(current_snapshot_path).get("counts", {}),
        "artifact_paths": {
            "commits": str(commits_path),
            "operations": str(operations_path),
            "snapshots_dir": str(snapshots_dir),
            "current_snapshot": str(current_snapshot_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def _empty_state() -> dict[str, Any]:
    return {
        "entities": {},
        "entity_aliases": defaultdict(set),
        "entity_memory_index": defaultdict(list),
        "memories": [],
        "relationships": {},
        "stated_facts": [],
        "open_questions": [],
        "unresolved_entity_mentions": [],
        "scene_tags": [],
    }


def _commit_entity_mention(
    *,
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    commit_id: str,
    parent_id: str,
    record: dict[str, Any],
    alias_index: dict[str, dict[str, Any]],
    entity_by_id: dict[str, dict[str, Any]],
) -> str | None:
    surface = str(record.get("surface") or "").strip()
    canonical_hint = str(record.get("canonical_hint") or "").strip()
    entity = _resolve_entity(surface, canonical_hint, alias_index)
    if not entity:
        _append_operation(
            operations,
            commit_id=commit_id,
            parent_id=parent_id,
            operation_type="entity_ambiguous",
            payload={
                "surface": surface,
                "canonical_hint": canonical_hint,
                "reason": "no registry match in final entity-resolution artifacts",
                "evidence": record.get("evidence", ""),
            },
        )
        return None
    entity_id = str(entity.get("entity_id"))
    _ensure_entity_state(
        state=state,
        operations=operations,
        commit_id=commit_id,
        parent_id=parent_id,
        entity_id=entity_id,
        entity_by_id=entity_by_id,
        source="kg_entity_mention",
        evidence=record.get("evidence", ""),
    )
    alias_values = _observed_aliases(surface, canonical_hint)
    for alias in sorted(alias_values):
        _add_alias(
            state=state,
            operations=operations,
            commit_id=commit_id,
            parent_id=parent_id,
            entity_id=entity_id,
            alias=alias,
            source="kg_entity_mention",
        )
    state["entities"][entity_id]["last_seen_scene"] = parent_id
    state["entities"][entity_id]["seen_scene_ids"] = sorted({*state["entities"][entity_id]["seen_scene_ids"], parent_id})
    state["entities"][entity_id]["mention_count"] += 1
    _append_operation(
        operations,
        commit_id=commit_id,
        parent_id=parent_id,
        operation_type="entity_seen",
        payload={
            "entity_id": entity_id,
            "surface": surface,
            "canonical_hint": canonical_hint,
            "source_record_id": record.get("record_id", ""),
            "evidence": record.get("evidence", ""),
        },
    )
    return entity_id


def _ensure_entity_state(
    *,
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    commit_id: str,
    parent_id: str,
    entity_id: str,
    entity_by_id: dict[str, dict[str, Any]],
    source: str,
    evidence: Any,
) -> None:
    if entity_id in state["entities"]:
        return
    entity = entity_by_id.get(entity_id, {"entity_id": entity_id, "canonical_name": entity_id, "entity_type": "other"})
    state["entities"][entity_id] = {
        "entity_id": entity_id,
        "entity_type": entity.get("entity_type", "other"),
        "canonical_name": entity.get("canonical_name", entity_id),
        "first_seen_scene": parent_id,
        "last_seen_scene": parent_id,
        "seen_scene_ids": [parent_id],
        "mention_count": 0,
        "aliases": [],
    }
    _append_operation(
        operations,
        commit_id=commit_id,
        parent_id=parent_id,
        operation_type="entity_created",
        payload={
            "entity_id": entity_id,
            "canonical_name": entity.get("canonical_name", entity_id),
            "entity_type": entity.get("entity_type", "other"),
            "source": source,
            "evidence": evidence,
        },
    )


def _add_alias(
    *,
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    commit_id: str,
    parent_id: str,
    entity_id: str,
    alias: str,
    source: str,
) -> None:
    alias = str(alias or "").strip()
    if not alias:
        return
    normalized = _normalize_alias(alias)
    if normalized in state["entity_aliases"][entity_id]:
        return
    state["entity_aliases"][entity_id].add(normalized)
    aliases = state["entities"][entity_id]["aliases"]
    if alias not in aliases:
        aliases.append(alias)
        aliases.sort()
    _append_operation(
        operations,
        commit_id=commit_id,
        parent_id=parent_id,
        operation_type="entity_alias_added",
        payload={
            "entity_id": entity_id,
            "alias": alias,
            "normalized_alias": normalized,
            "source": source,
        },
    )


def _commit_memory(
    *,
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    commit_id: str,
    parent_id: str,
    memory: dict[str, Any],
    links: list[dict[str, Any]],
    alias_index: dict[str, dict[str, Any]],
    entity_by_id: dict[str, dict[str, Any]],
) -> None:
    memory_id = str(memory.get("record_id") or "")
    compact_memory = _compact_record(memory)
    compact_memory["linked_entity_ids"] = []
    state["memories"].append(compact_memory)
    _append_operation(
        operations,
        commit_id=commit_id,
        parent_id=parent_id,
        operation_type="memory_appended",
        payload={
            "memory_record_id": memory_id,
            "summary": memory.get("summary", ""),
            "memory_type": memory.get("memory_type", ""),
            "evidence": memory.get("evidence", ""),
        },
    )
    for link in links:
        entity = _resolve_entity(str(link.get("entity") or ""), str(link.get("canonical_entity") or ""), alias_index)
        if not entity:
            _append_operation(
                operations,
                commit_id=commit_id,
                parent_id=parent_id,
                operation_type="entity_memory_link_unresolved",
                payload={
                    "memory_record_id": memory_id,
                    "entity": link.get("entity", ""),
                    "evidence": link.get("evidence", ""),
                },
            )
            continue
        entity_id = str(entity.get("entity_id"))
        _ensure_entity_state(
            state=state,
            operations=operations,
            commit_id=commit_id,
            parent_id=parent_id,
            entity_id=entity_id,
            entity_by_id=entity_by_id,
            source="entity_memory_link",
            evidence=link.get("evidence", ""),
        )
        compact_memory["linked_entity_ids"].append(entity_id)
        state["entity_memory_index"][entity_id].append(
            {
                "memory_record_id": memory_id,
                "parent_unit_id": parent_id,
                "scene_id": memory.get("scene_id"),
                "timeline_index": memory.get("timeline_index"),
                "sequence_index": memory.get("sequence_index"),
                "memory_type": memory.get("memory_type"),
                "summary": memory.get("summary"),
                "link_role": link.get("link_role"),
                "evidence": link.get("evidence") or memory.get("evidence", ""),
            }
        )
        _append_operation(
            operations,
            commit_id=commit_id,
            parent_id=parent_id,
            operation_type="entity_memory_linked",
            payload={
                "entity_id": entity_id,
                "memory_record_id": memory_id,
                "link_role": link.get("link_role", ""),
                "evidence": link.get("evidence", ""),
            },
        )


def _commit_relationship_update(
    state: dict[str, Any],
    operations: list[dict[str, Any]],
    commit_id: str,
    parent_id: str,
    update: dict[str, Any],
) -> None:
    key = _relationship_key(update)
    if key not in state["relationships"]:
        state["relationships"][key] = {
            "source_entity_id": update.get("source_entity_id"),
            "source_name": update.get("source_name"),
            "target_entity_id": update.get("target_entity_id"),
            "target_name": update.get("target_name"),
            "relation_type": update.get("relation_type"),
            "direction": update.get("direction", "directed"),
            "status": "active",
            "first_seen_scene": parent_id,
            "last_updated_scene": parent_id,
            "update_ids": [],
            "evidence": [],
            "strength": 0.0,
        }
        operation_type = "relationship_created"
    else:
        operation_type = "relationship_updated"
    relationship = state["relationships"][key]
    relationship["last_updated_scene"] = parent_id
    relationship["update_ids"].append(update.get("update_id", ""))
    relationship["strength"] = round(float(relationship.get("strength", 0.0)) + float(update.get("strength_delta", 0.0)), 3)
    if update.get("evidence"):
        relationship["evidence"].append(update.get("evidence"))
    _append_operation(
        operations,
        commit_id=commit_id,
        parent_id=parent_id,
        operation_type=operation_type,
        payload={
            "update_id": update.get("update_id", ""),
            "source_entity_id": update.get("source_entity_id"),
            "target_entity_id": update.get("target_entity_id"),
            "relation_type": update.get("relation_type"),
            "summary": update.get("summary", ""),
            "evidence": update.get("evidence", ""),
        },
    )


def _snapshot(state: dict[str, Any], *, commit_id: str, parent_id: str) -> dict[str, Any]:
    entities = sorted((deepcopy(entity) for entity in state["entities"].values()), key=lambda item: str(item.get("entity_id")))
    relationships = sorted(
        (deepcopy(record) for record in state["relationships"].values()),
        key=lambda item: (str(item.get("source_entity_id")), str(item.get("target_entity_id")), str(item.get("relation_type"))),
    )
    entity_memory_index = {
        entity_id: records
        for entity_id, records in sorted(state["entity_memory_index"].items())
    }
    return {
        "snapshot_id": f"prefix_after_{parent_id}" if parent_id else "empty_prefix",
        "commit_id": commit_id,
        "after_parent_unit_id": parent_id,
        "entities": entities,
        "relationships": relationships,
        "entity_memory_index": entity_memory_index,
        "memories": deepcopy(state["memories"]),
        "stated_facts": deepcopy(state["stated_facts"]),
        "open_questions": deepcopy(state["open_questions"]),
        "unresolved_entity_mentions": deepcopy(state["unresolved_entity_mentions"]),
        "scene_tags": deepcopy(state["scene_tags"]),
        "counts": {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "memory_count": len(state["memories"]),
            "entity_memory_link_count": sum(len(records) for records in state["entity_memory_index"].values()),
            "stated_fact_count": len(state["stated_facts"]),
            "open_question_count": len(state["open_questions"]),
            "unresolved_entity_mention_count": len(state["unresolved_entity_mentions"]),
            "scene_tag_count": len(state["scene_tags"]),
        },
    }


def _append_operation(
    operations: list[dict[str, Any]],
    *,
    commit_id: str,
    parent_id: str,
    operation_type: str,
    payload: dict[str, Any],
) -> None:
    operations.append(
        {
            "operation_id": f"op_{len(operations) + 1:06d}",
            "commit_id": commit_id,
            "parent_unit_id": parent_id,
            "operation_type": operation_type,
            **payload,
        }
    )


def _observed_aliases(surface: str, canonical_hint: str) -> set[str]:
    names = {surface, canonical_hint}
    for name in list(names):
        names.update(resolve_name_variants(name))
    return {name for name in names if str(name or "").strip()}


def _resolve_entity(surface: str, canonical_hint: str, alias_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for name in (canonical_hint, surface):
        for variant in resolve_name_variants(name):
            match = alias_index.get(_normalize_alias(variant))
            if match:
                return match
        match = alias_index.get(_normalize_alias(name))
        if match:
            return match
    return None


def _alias_index(entities: list[dict[str, Any]], alias_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    entity_by_id = {str(entity.get("entity_id")): entity for entity in entities if entity.get("entity_id")}
    index: dict[str, dict[str, Any]] = {}
    for entity in entities:
        names = [entity.get("canonical_name", ""), *(_as_list(entity.get("aliases")))]
        for name in names:
            normalized = _normalize_alias(str(name))
            if normalized:
                index[normalized] = entity
    for alias in alias_records:
        entity = entity_by_id.get(str(alias.get("entity_id")))
        normalized = str(alias.get("normalized_alias") or _normalize_alias(str(alias.get("alias") or "")))
        if entity and normalized:
            index[normalized] = entity
    return index


def _unit_parent_map(world_model: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for section in (
        "scenes",
        "kg_entity_mentions",
        "scene_tags",
        "unresolved_kg_mentions",
        "episodic_memories",
        "entity_memory_links",
        "relationship_observations",
        "stated_facts",
        "open_questions",
    ):
        for record in _as_list(world_model.get(section)):
            unit_id = str(record.get("unit_id") or record.get("scene_id") or "")
            scene_id = str(record.get("scene_id") or unit_id)
            parent_id = str(record.get("parent_unit_id") or scene_id or unit_id)
            if unit_id:
                mapping[unit_id] = parent_id
            if scene_id:
                mapping[scene_id] = parent_id
    return mapping


def _parent_order(world_model: dict[str, Any], unit_parent_map: dict[str, str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    scenes = _as_list(world_model.get("scenes"))
    for record in scenes:
        unit_id = str(record.get("unit_id") or record.get("scene_id") or "")
        parent_id = str(record.get("parent_unit_id") or unit_parent_map.get(unit_id) or record.get("scene_id") or unit_id)
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            ordered.append(parent_id)
    return ordered


def _merge_parent_order(base_order: list[str], *groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    ordered = list(base_order)
    seen = set(ordered)
    for group in groups:
        for parent_id in sorted(group):
            if parent_id not in seen:
                seen.add(parent_id)
                ordered.append(parent_id)
    return ordered


def _unit_ids_for_parent(world_model: dict[str, Any], unit_parent_map: dict[str, str], parent_id: str) -> list[str]:
    unit_ids: list[str] = []
    for record in _as_list(world_model.get("scenes")):
        unit_id = str(record.get("unit_id") or record.get("scene_id") or "")
        if unit_id and unit_parent_map.get(unit_id, unit_id) == parent_id:
            unit_ids.append(unit_id)
    return unit_ids or [parent_id]


def _group_by_parent(records: Any, unit_parent_map: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _as_list(records):
        unit_id = str(record.get("unit_id") or record.get("scene_id") or "")
        scene_id = str(record.get("scene_id") or unit_id)
        parent_id = str(record.get("parent_unit_id") or unit_parent_map.get(unit_id) or unit_parent_map.get(scene_id) or scene_id)
        if parent_id:
            grouped[parent_id].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: (str(item.get("scene_id")), int(item.get("sequence_index") or 0), str(item.get("record_id"))))
    return dict(grouped)


def _links_by_memory(links: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        grouped[str(link.get("memory_record_id") or "")].append(link)
    return dict(grouped)


def _relationship_key(update: dict[str, Any]) -> str:
    source = str(update.get("source_entity_id") or "")
    target = str(update.get("target_entity_id") or "")
    relation = str(update.get("relation_type") or "")
    if update.get("direction") == "undirected" and source > target:
        source, target = target, source
    return f"{source}|{target}|{relation}"


def _operation_type_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for operation in operations:
        operation_type = str(operation.get("operation_type") or "")
        counts[operation_type] = counts.get(operation_type, 0) + 1
    return counts


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "record_id",
            "scene_id",
            "unit_id",
            "parent_unit_id",
            "chunk_id",
            "chunk_index",
            "chunk_count",
            "timeline_index",
            "sequence_index",
            "memory_type",
            "summary",
            "proposition",
            "question",
            "evidence",
            "parent_evidence_start",
            "parent_evidence_end",
            "parent_source_sha256",
        }
    }


def _compact_scene_tag(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "record_id",
            "scene_id",
            "unit_id",
            "parent_unit_id",
            "chunk_id",
            "chunk_index",
            "chunk_count",
            "sequence_index",
            "surface",
            "tag_type",
            "reason",
            "evidence",
            "parent_evidence_start",
            "parent_evidence_end",
            "parent_source_sha256",
        }
    }


def _normalize_alias(value: str) -> str:
    return "".join(str(value or "").lower().replace("，", ",").replace("。", ".").split()).replace(",", "").replace(".", "").replace("-", "")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
