from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.llm import LLMClient, LLMResult
from dms.parsing import extract_json_value
from dms.prompts import YAMLPromptLoader


@dataclass(frozen=True)
class SceneDispositionNoteConfig:
    attribute_cards_path: Path
    output_dir: Path
    social_simulation_intent: str
    memory_packet_path: Path | None = None
    prompt_dir: Path = Path("task_specs/prompts")
    entity_types: tuple[str, ...] = ("character",)
    entity_names: tuple[str, ...] = ()
    max_relevant_memories_per_entity: int = 6
    overwrite: bool = False


def build_scene_disposition_notes(config: SceneDispositionNoteConfig, llm_client: LLMClient) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("inputs", "prompts", "raw_outputs", "parsed"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    cards_payload = json.loads(Path(config.attribute_cards_path).read_text(encoding="utf-8"))
    memory_packet = _load_memory_packet(config.memory_packet_path)
    cards = _select_cards(
        _extract_cards(cards_payload),
        entity_types=config.entity_types,
        entity_names=config.entity_names,
    )
    loader = YAMLPromptLoader(config.prompt_dir)
    notes: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    for card in cards:
        context = _build_note_context(
            card,
            cards=cards,
            social_simulation_intent=config.social_simulation_intent,
            memory_packet=memory_packet,
            max_relevant_memories=config.max_relevant_memories_per_entity,
        )
        call_id = f"disposition_{_safe_id(card.get('entity_id') or card.get('canonical_name'))}"
        (output_dir / "inputs" / f"{call_id}.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        prompt = loader.render("dms/scene_disposition_note", task_values={"disposition_context_json": context})
        prompt_path = output_dir / "prompts" / f"{call_id}.txt"
        raw_path = output_dir / "raw_outputs" / f"{call_id}.json"
        parsed_path = output_dir / "parsed" / f"{call_id}.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        result = llm_client.complete(prompt)
        raw_path.write_text(json.dumps(_llm_result_to_dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parsed = extract_json_value(result.text)
        parsed_payload = {
            "call_id": call_id,
            "status": "parsed" if parsed.ok else "parse_failed",
            "data": parsed.data,
            "parse_error": parsed.error,
        }
        parsed_path.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        calls.append(
            {
                "call_id": call_id,
                "entity_id": card.get("entity_id"),
                "canonical_name": card.get("canonical_name"),
                "prompt_path": str(prompt_path),
                "raw_output_path": str(raw_path),
                "parsed_path": str(parsed_path),
                "status": parsed_payload["status"],
                "usage": result.usage,
            }
        )
        if not parsed.ok:
            raise ValueError(f"Failed to parse scene disposition note for {call_id}: {parsed.error}")
        notes.append(_normalize_note(parsed.data, card))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "provider": llm_client.provider,
            "model": llm_client.model,
        },
        "inputs": {
            "attribute_cards_path": str(config.attribute_cards_path),
            "memory_packet_path": str(config.memory_packet_path) if config.memory_packet_path else None,
            "social_simulation_intent": config.social_simulation_intent,
            "entity_types": list(config.entity_types),
            "entity_names": list(config.entity_names),
            "max_relevant_memories_per_entity": config.max_relevant_memories_per_entity,
        },
        "note_count": len(notes),
        "scene_disposition_notes": notes,
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "notes_json": str(output_dir / "scene_disposition_notes.json"),
            "notes_markdown": str(output_dir / "scene_disposition_notes.md"),
            "calls": str(output_dir / "calls.jsonl"),
            "inputs_dir": str(output_dir / "inputs"),
            "prompts_dir": str(output_dir / "prompts"),
            "raw_outputs_dir": str(output_dir / "raw_outputs"),
            "parsed_dir": str(output_dir / "parsed"),
        },
    }
    (output_dir / "scene_disposition_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "scene_disposition_notes.md").write_text(
        format_scene_disposition_notes_markdown(summary),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "calls.jsonl", calls)
    return summary


def format_scene_disposition_notes_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Scene Disposition Notes", ""]
    inputs = summary.get("inputs") or {}
    if inputs.get("attribute_cards_path"):
        lines.append(f"- source: {inputs.get('attribute_cards_path')}")
    if inputs.get("memory_packet_path"):
        lines.append(f"- memory packet: {inputs.get('memory_packet_path')}")
    if inputs.get("social_simulation_intent"):
        lines.append(f"- social simulation intent: {inputs.get('social_simulation_intent')}")
    lines.append(f"- note count: {summary.get('note_count', 0)}")
    for note in summary.get("scene_disposition_notes") or []:
        lines.append("")
        lines.append(f"## {note.get('canonical_name')}")
        text = str(note.get("scene_disposition_note") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines).rstrip() + "\n"


def _build_note_context(
    card: dict[str, Any],
    *,
    cards: list[dict[str, Any]],
    social_simulation_intent: str,
    memory_packet: dict[str, Any],
    max_relevant_memories: int,
) -> dict[str, Any]:
    target_name = str(card.get("canonical_name") or "")
    packet_entity = _find_packet_entity(card, memory_packet)
    return {
        "social_simulation_intent": social_simulation_intent,
        "target_card": card,
        "relevant_memory_notes": _select_relevant_memory_notes(
            packet_entity,
            memory_packet,
            limit=max(max_relevant_memories, 0),
        ),
        "relevant_reference_notes": _select_relevant_reference_notes(
            packet_entity,
            memory_packet,
            limit=max(max_relevant_memories, 0),
        ),
        "other_visible_cards": [
            _compact_peer_card(peer)
            for peer in cards
            if peer is not card and peer.get("canonical_name") != target_name
        ],
        "instructions": {
            "output_language": "Chinese",
            "schema_policy": "Only entity_id, canonical_name, and scene_disposition_note are business fields.",
            "target_scene_text_visible": False,
            "writing_spec_visible": False,
            "relevant_memory_notes_policy": "Use these compact prefix-memory notes to ground the scene-conditioned prior; do not output separate refs fields.",
            "relevant_reference_notes_policy": "Use these visible external-reference notes only as background knowledge; do not treat them as screenplay events.",
        },
    }


def _load_memory_packet(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _extract_cards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        cards = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        cards = payload["cards"]
    else:
        cards = []
    return [dict(card) for card in cards if isinstance(card, dict)]


def _find_packet_entity(card: dict[str, Any], memory_packet: dict[str, Any]) -> dict[str, Any]:
    entity_id = str(card.get("entity_id") or "")
    canonical_name = str(card.get("canonical_name") or "")
    entities = memory_packet.get("entities") if isinstance(memory_packet, dict) else []
    if not isinstance(entities, list):
        return {}
    for entity in entities:
        if isinstance(entity, dict) and entity_id and entity.get("entity_id") == entity_id:
            return entity
    for entity in entities:
        if isinstance(entity, dict) and canonical_name and entity.get("canonical_name") == canonical_name:
            return entity
    return {}


def _select_relevant_memory_notes(
    packet_entity: dict[str, Any],
    memory_packet: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    if limit <= 0 or not memory_packet:
        return []
    memories = memory_packet.get("episodic_memories")
    if not isinstance(memories, list):
        return []
    memory_by_index = {str(memory.get("index")): memory for memory in memories if isinstance(memory, dict)}
    notes: list[str] = []
    seen: set[str] = set()
    for index in packet_entity.get("related_memory_index") or []:
        memory = memory_by_index.get(str(index))
        note = _compact_memory_note(memory)
        if note and note not in seen:
            notes.append(note)
            seen.add(note)
        if len(notes) >= limit:
            return notes
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        if str(memory.get("memory_temporal_scope") or "") not in {"atemporal_fact", "durable_state"}:
            continue
        note = _compact_memory_note(memory)
        if note and note not in seen:
            notes.append(note)
            seen.add(note)
        if len(notes) >= limit:
            break
    return notes


def _compact_memory_note(memory: dict[str, Any] | None) -> str:
    if not isinstance(memory, dict):
        return ""
    index = str(memory.get("index") or "").strip()
    summary = str(memory.get("summary") or "").strip()
    if not index or not summary:
        return ""
    scene = str(memory.get("scene_id") or "").strip()
    scope = str(memory.get("memory_temporal_scope") or "").strip()
    scene_text = f" <{scene}>" if scene else ""
    scope_text = f" scope={scope}" if scope else ""
    return f"{index}{scene_text}{scope_text}: {summary}"


def _select_relevant_reference_notes(
    packet_entity: dict[str, Any],
    memory_packet: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    if limit <= 0 or not memory_packet:
        return []
    entity_names = {
        str(value).strip().lower()
        for value in (
            packet_entity.get("entity_id"),
            packet_entity.get("canonical_name"),
            *(packet_entity.get("aliases") or []),
        )
        if str(value or "").strip()
    }
    notes: list[str] = []
    seen: set[str] = set()
    for item in memory_packet.get("character_reference_knowledge") or []:
        if not isinstance(item, dict):
            continue
        known_to = {str(value).strip().lower() for value in item.get("known_to") or [] if str(value or "").strip()}
        if "all" not in known_to and known_to and not known_to.intersection(entity_names):
            continue
        note = _compact_reference_note(item)
        if note and note not in seen:
            notes.append(note)
            seen.add(note)
        if len(notes) >= limit:
            break
    return notes


def _compact_reference_note(item: dict[str, Any]) -> str:
    statement = str(item.get("statement") or "").strip()
    if not statement:
        return ""
    ref_id = str(item.get("item_id") or "").strip()
    subject = str(item.get("subject") or "").strip()
    scope = str(item.get("knowledge_scope") or "").strip()
    prefix = f"{subject}: " if subject else ""
    suffix = f" [{scope}]" if scope else ""
    ref_text = f"REF:{ref_id} " if ref_id else ""
    return f"{ref_text}{prefix}{statement}{suffix}".strip()


def _select_cards(
    cards: list[dict[str, Any]],
    *,
    entity_types: tuple[str, ...],
    entity_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed_types = {item.strip() for item in entity_types if item.strip()}
    allowed_names = {item.strip() for item in entity_names if item.strip()}
    selected: list[dict[str, Any]] = []
    for card in cards:
        if allowed_types and card.get("entity_type") not in allowed_types:
            continue
        if allowed_names and card.get("canonical_name") not in allowed_names and card.get("entity_id") not in allowed_names:
            continue
        selected.append(card)
    return selected


def _compact_peer_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": card.get("entity_id"),
        "canonical_name": card.get("canonical_name"),
        "entity_type": card.get("entity_type"),
        "prefix_boundary": card.get("prefix_boundary"),
        "current_state": card.get("current_state") or [],
        "relationship_stances": card.get("relationship_stances") or [],
        "author_profile_summary": card.get("author_profile_summary") or "",
    }


def _normalize_note(data: Any, card: dict[str, Any]) -> dict[str, str]:
    payload = dict(data) if isinstance(data, dict) else {}
    return {
        "entity_id": str(payload.get("entity_id") or card.get("entity_id") or ""),
        "canonical_name": str(payload.get("canonical_name") or card.get("canonical_name") or ""),
        "scene_disposition_note": str(payload.get("scene_disposition_note") or "").strip(),
    }


def _safe_id(value: Any) -> str:
    raw = str(value or "item")
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)
    return safe or "item"


def _llm_result_to_dict(result: LLMResult) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "raw_response": result.raw_response,
        "usage": result.usage,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
