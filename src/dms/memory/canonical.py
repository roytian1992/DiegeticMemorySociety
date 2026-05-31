from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_canonical_memory(staged_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Promote staged scene inventory artifacts into a simple canonical prefix memory.

    This MVP builder performs conservative deterministic reconciliation:
    names are canonicalized by normalized exact match, with aliases and scene
    mentions retained. It does not infer hidden facts or character beliefs.
    """

    staged_path = Path(staged_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    staged_summary = _read_json(staged_path / "summary.json") if (staged_path / "summary.json").exists() else {}
    scenes = _read_jsonl(staged_path / "scenes.jsonl")
    character_mentions = _read_jsonl(staged_path / "characters.jsonl")
    object_mentions = _read_jsonl(staged_path / "objects.jsonl")
    fact_mentions = _read_jsonl(staged_path / "stated_facts.jsonl")
    question_mentions = _read_jsonl(staged_path / "open_questions.jsonl")

    canonical_characters = _canonicalize_mentions(character_mentions, entity_type="character")
    canonical_objects = _canonicalize_mentions(object_mentions, entity_type="object")

    scene_index = {str(scene.get("scene_id")): scene for scene in scenes}
    memory = {
        "memory_layer": "canonical_prefix",
        "source_staged_dir": str(staged_path),
        "staged_summary": staged_summary,
        "scenes": scenes,
        "scene_index": scene_index,
        "characters": canonical_characters,
        "objects": canonical_objects,
        "stated_facts": fact_mentions,
        "open_questions": question_mentions,
    }

    paths = {
        "memory": out_path / "canonical_memory.json",
        "scenes": out_path / "scenes.jsonl",
        "characters": out_path / "characters.jsonl",
        "objects": out_path / "objects.jsonl",
        "stated_facts": out_path / "stated_facts.jsonl",
        "open_questions": out_path / "open_questions.jsonl",
        "summary": out_path / "summary.json",
    }

    paths["memory"].write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(paths["scenes"], scenes)
    _write_jsonl(paths["characters"], canonical_characters)
    _write_jsonl(paths["objects"], canonical_objects)
    _write_jsonl(paths["stated_facts"], fact_mentions)
    _write_jsonl(paths["open_questions"], question_mentions)

    summary = {
        "source_staged_dir": str(staged_path),
        "output_dir": str(out_path),
        "scene_count": len(scenes),
        "character_count": len(canonical_characters),
        "object_count": len(canonical_objects),
        "stated_fact_count": len(fact_mentions),
        "open_question_count": len(question_mentions),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_canonical_memory(memory_path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(memory_path_or_dir)
    if path.is_dir():
        path = path / "canonical_memory.json"
    return _read_json(path)


def query_memory(
    memory_path_or_dir: str | Path,
    *,
    text: str = "",
    character: str | None = None,
    scene_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    memory = load_canonical_memory(memory_path_or_dir)
    query = (text or "").strip().lower()
    character_query = (character or "").strip().lower()

    scenes = _filter_records(memory.get("scenes", []), query=query, scene_id=scene_id, limit=limit)
    characters = _filter_entities(memory.get("characters", []), query=query or character_query, limit=limit)
    objects = _filter_entities(memory.get("objects", []), query=query, limit=limit)
    facts = _filter_records(memory.get("stated_facts", []), query=query or character_query, scene_id=scene_id, limit=limit)
    questions = _filter_records(memory.get("open_questions", []), query=query, scene_id=scene_id, limit=limit)

    if character_query:
        characters = [
            item for item in memory.get("characters", []) if character_query in str(item.get("canonical_name", "")).lower()
        ][:limit]
        mentioned_scenes = {scene_id for item in characters for scene_id in item.get("scene_ids", [])}
        facts = [item for item in memory.get("stated_facts", []) if item.get("scene_id") in mentioned_scenes][:limit]
        scenes = [item for item in memory.get("scenes", []) if item.get("scene_id") in mentioned_scenes][:limit]

    return {
        "query": {"text": text, "character": character, "scene_id": scene_id, "limit": limit},
        "results": {
            "scenes": scenes,
            "characters": characters,
            "objects": objects,
            "stated_facts": facts,
            "open_questions": questions,
        },
        "counts": {
            "scenes": len(scenes),
            "characters": len(characters),
            "objects": len(objects),
            "stated_facts": len(facts),
            "open_questions": len(questions),
        },
    }


def build_visibility_packet(
    memory_path_or_dir: str | Path,
    *,
    character: str,
    scene_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Build a conservative character-scene memory packet.

    MVP rule: a character can see scene-local setting and facts from scenes
    where the character was explicitly mentioned, up to and including the
    selected scene's discourse order. This is a placeholder policy, but it is
    explicit and auditable.
    """

    memory = load_canonical_memory(memory_path_or_dir)
    target_scene = memory.get("scene_index", {}).get(scene_id)
    if not isinstance(target_scene, dict):
        raise ValueError(f"Unknown scene_id: {scene_id}")

    character_key = _normalize_name(character)
    character_record = None
    for record in memory.get("characters", []):
        if _normalize_name(str(record.get("canonical_name", ""))) == character_key:
            character_record = record
            break
    if character_record is None:
        raise ValueError(f"Unknown character: {character}")

    scene_order = {str(scene.get("scene_id")): idx for idx, scene in enumerate(memory.get("scenes", []), start=1)}
    target_order = scene_order.get(scene_id, 0)
    mentioned_scenes = set(character_record.get("scene_ids", []))
    visible_scene_ids = {
        sid for sid in mentioned_scenes if scene_order.get(sid, 10**9) <= target_order
    }

    visible_scenes = [scene for scene in memory.get("scenes", []) if scene.get("scene_id") in visible_scene_ids][:limit]
    visible_facts = [fact for fact in memory.get("stated_facts", []) if fact.get("scene_id") in visible_scene_ids][:limit]
    visible_questions = [
        question for question in memory.get("open_questions", []) if question.get("scene_id") in visible_scene_ids
    ][:limit]
    blocked_future_scenes = [
        scene.get("scene_id")
        for scene in memory.get("scenes", [])
        if scene_order.get(str(scene.get("scene_id")), 0) > target_order
    ]

    return {
        "packet_type": "character_visibility",
        "policy": "mvp_explicit_mention_prefix_only",
        "character": character_record,
        "target_scene_id": scene_id,
        "target_scene": target_scene,
        "visible_scene_ids": sorted(visible_scene_ids, key=lambda sid: scene_order.get(sid, 0)),
        "visible_scenes": visible_scenes,
        "visible_facts": visible_facts,
        "visible_open_questions": visible_questions,
        "blocked": {
            "future_scene_ids": blocked_future_scenes,
            "reason": "after target scene discourse order",
        },
    }


def _canonicalize_mentions(mentions: list[dict[str, Any]], *, entity_type: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        name = str(mention.get("name") or "").strip()
        if not name:
            continue
        grouped[_normalize_name(name)].append(mention)

    records: list[dict[str, Any]] = []
    for index, (key, items) in enumerate(sorted(grouped.items()), start=1):
        names = [str(item.get("name") or "").strip() for item in items if str(item.get("name") or "").strip()]
        canonical_name = names[0] if names else key
        scene_ids = sorted({str(item.get("scene_id")) for item in items if item.get("scene_id")})
        records.append(
            {
                "memory_layer": "canonical_prefix",
                "entity_id": f"{entity_type}_{index:04d}",
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "aliases": sorted(set(names)),
                "scene_ids": scene_ids,
                "mentions": items,
            }
        )
    return records


def _filter_entities(records: list[dict[str, Any]], *, query: str, limit: int) -> list[dict[str, Any]]:
    if not query:
        return records[:limit]
    return [
        record
        for record in records
        if query in json.dumps(record, ensure_ascii=False).lower()
    ][:limit]


def _filter_records(
    records: list[dict[str, Any]],
    *,
    query: str,
    scene_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        if scene_id and record.get("scene_id") != scene_id:
            continue
        if query and query not in json.dumps(record, ensure_ascii=False).lower():
            continue
        out.append(record)
        if len(out) >= limit:
            break
    return out


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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
