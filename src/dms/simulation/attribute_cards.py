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
class AttributeCardConfig:
    memory_packet_path: Path
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    entity_types: tuple[str, ...] = ("character",)
    entity_names: tuple[str, ...] = ()
    max_memories_per_entity: int = 16
    overwrite: bool = False


def build_entity_attribute_cards(config: AttributeCardConfig, llm_client: LLMClient) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("inputs", "prompts", "raw_outputs", "parsed"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    packet = json.loads(Path(config.memory_packet_path).read_text(encoding="utf-8"))
    loader = YAMLPromptLoader(config.prompt_dir)
    cards: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    for entity in _select_entities(packet, entity_types=config.entity_types, entity_names=config.entity_names):
        context = _build_entity_context(packet, entity, max_memories=config.max_memories_per_entity)
        call_id = _safe_call_id(entity)
        (output_dir / "inputs" / f"{call_id}.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        prompt = loader.render("dms/entity_attribute_card", task_values={"entity_context_json": context})
        prompt_path = output_dir / "prompts" / f"{call_id}.txt"
        raw_path = output_dir / "raw_outputs" / f"{call_id}.json"
        parsed_path = output_dir / "parsed" / f"{call_id}.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        result = llm_client.complete(prompt)
        raw_path.write_text(
            json.dumps(_llm_result_to_dict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
                "entity_id": entity.get("entity_id"),
                "canonical_name": entity.get("canonical_name"),
                "entity_type": entity.get("entity_type"),
                "prompt_path": str(prompt_path),
                "raw_output_path": str(raw_path),
                "parsed_path": str(parsed_path),
                "status": parsed_payload["status"],
                "usage": result.usage,
            }
        )
        if not parsed.ok:
            raise ValueError(f"Failed to parse attribute card for {call_id}: {parsed.error}")
        card = _normalize_card(parsed.data, context)
        cards.append(card)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "provider": llm_client.provider,
            "model": llm_client.model,
        },
        "inputs": {
            "memory_packet_path": str(config.memory_packet_path),
            "entity_types": list(config.entity_types),
            "entity_names": list(config.entity_names),
            "max_memories_per_entity": config.max_memories_per_entity,
        },
        "card_count": len(cards),
        "cards": cards,
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "cards_json": str(output_dir / "attribute_cards.json"),
            "cards_markdown": str(output_dir / "attribute_cards.md"),
            "calls": str(output_dir / "calls.jsonl"),
            "inputs_dir": str(output_dir / "inputs"),
            "prompts_dir": str(output_dir / "prompts"),
            "raw_outputs_dir": str(output_dir / "raw_outputs"),
            "parsed_dir": str(output_dir / "parsed"),
        },
    }
    (output_dir / "attribute_cards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "attribute_cards.md").write_text(format_attribute_cards_markdown(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "calls.jsonl", calls)
    return summary


def format_attribute_cards_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Entity Attribute Cards", ""]
    inputs = summary.get("inputs") or {}
    if inputs.get("memory_packet_path"):
        lines.append(f"- source: {inputs.get('memory_packet_path')}")
    lines.append(f"- card count: {summary.get('card_count', 0)}")
    for card in summary.get("cards") or []:
        lines.append("")
        lines.append(f"## {card.get('canonical_name')} ({card.get('entity_type')})")
        if card.get("prefix_boundary"):
            lines.append(f"- prefix boundary: {card.get('prefix_boundary')}")
        _append_items(lines, "role in story", card.get("role_in_story"), "value")
        _append_items(lines, "current state", card.get("current_state"), "value")
        _append_items(lines, "salient past actions", card.get("salient_past_actions"), "action")
        _append_items(lines, "stable traits", card.get("stable_traits"), "trait")
        _append_items(lines, "speaking style", card.get("speaking_style"), "trait")
        _append_items(lines, "values or motivations", card.get("values_or_motivations"), "value")
        _append_relationships(lines, card.get("relationship_stances"))
        _append_items(lines, "behavior tendencies", card.get("behavior_tendencies"), "tendency")
        _append_constraints(lines, "hard constraints", card.get("hard_constraints"))
        _append_risks(lines, card.get("simulation_risks"))
        unsupported = card.get("uncertain_or_unsupported") or []
        if unsupported:
            lines.append("")
            lines.append("unsupported / uncertain:")
            for item in unsupported:
                claim = item.get("claim") if isinstance(item, dict) else str(item)
                reason = item.get("reason") if isinstance(item, dict) else ""
                suffix = f" - {reason}" if reason else ""
                lines.append(f"- {claim}{suffix}")
    return "\n".join(lines).rstrip() + "\n"


def _select_entities(
    packet: dict[str, Any],
    *,
    entity_types: tuple[str, ...],
    entity_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed_types = {item.strip() for item in entity_types if item.strip()}
    allowed_names = {item.strip() for item in entity_names if item.strip()}
    entities: list[dict[str, Any]] = []
    for entity in packet.get("entities") or []:
        if allowed_types and entity.get("entity_type") not in allowed_types:
            continue
        if allowed_names and entity.get("canonical_name") not in allowed_names and entity.get("entity_id") not in allowed_names:
            continue
        entities.append(entity)
    return entities


def _build_entity_context(packet: dict[str, Any], entity: dict[str, Any], *, max_memories: int) -> dict[str, Any]:
    memory_indexes = set(str(item) for item in (entity.get("related_memory_index") or []))
    memories = [
        memory
        for memory in packet.get("episodic_memories") or []
        if str(memory.get("index")) in memory_indexes
    ][: max(max_memories, 0)]
    refs = _collect_references(packet, entity, memories)
    relations = [
        relation
        for relation in packet.get("relations") or []
        if relation.get("source_entity_id") == entity.get("entity_id")
        or relation.get("target_entity_id") == entity.get("entity_id")
        or relation.get("source_name") == entity.get("canonical_name")
        or relation.get("target_name") == entity.get("canonical_name")
    ]
    return {
        "prefix_boundary": packet.get("retrieval_boundary") or {},
        "prefix_boundary_label": _prefix_boundary_label(packet.get("retrieval_boundary") or {}),
        "output_language": "Chinese",
        "entity": entity,
        "relations": relations,
        "related_memories": memories,
        "references": refs,
        "instructions": {
            "status_labels": ["explicit", "inferred"],
            "available_memory_indexes": [memory.get("index") for memory in memories],
            "available_reference_ids": [ref.get("ref_id") for ref in refs],
        },
    }


def _collect_references(packet: dict[str, Any], entity: dict[str, Any], memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(str(item) for item in (entity.get("source_refs") or []))
    wanted.update(str(memory.get("source_ref")) for memory in memories if memory.get("source_ref"))
    for relation in packet.get("relations") or []:
        if relation.get("source_entity_id") == entity.get("entity_id") or relation.get("target_entity_id") == entity.get("entity_id"):
            wanted.update(str(item) for item in relation.get("source_refs") or [])
    references = []
    for ref in packet.get("references") or []:
        if str(ref.get("ref_id")) in wanted:
            references.append(
                {
                    "ref_id": ref.get("ref_id"),
                    "scene_id": ref.get("scene_id"),
                    "text": ref.get("text"),
                }
            )
    return references


def _normalize_card(data: Any, context: dict[str, Any]) -> dict[str, Any]:
    entity = context.get("entity") or {}
    card = dict(data) if isinstance(data, dict) else {}
    card.setdefault("entity_id", entity.get("entity_id"))
    card.setdefault("canonical_name", entity.get("canonical_name"))
    card.setdefault("entity_type", entity.get("entity_type"))
    card["prefix_boundary"] = context.get("prefix_boundary_label") or card.get("prefix_boundary") or ""
    legacy_constraints = card.get("simulation_constraints")
    if "hard_constraints" not in card and isinstance(legacy_constraints, list):
        card["hard_constraints"] = legacy_constraints
    for key in (
        "role_in_story",
        "current_state",
        "salient_past_actions",
        "stable_traits",
        "speaking_style",
        "values_or_motivations",
        "relationship_stances",
        "behavior_tendencies",
        "hard_constraints",
        "simulation_risks",
        "uncertain_or_unsupported",
    ):
        if not isinstance(card.get(key), list):
            card[key] = []
    card.pop("simulation_constraints", None)
    _demote_unsupported_formal_roles(card)
    _normalize_hard_constraint_wording(card)
    return card


_FORMAL_ROLE_TERMS = (
    "监护",
    "亲属",
    "叔叔",
    "舅舅",
    "教官",
    "教师",
    "老师",
    "上级",
    "下属",
    "指挥官",
    "军人",
    "老兵",
    "飞行员",
    "航天项目相关人员",
    "项目人员",
)


def _demote_unsupported_formal_roles(card: dict[str, Any]) -> None:
    uncertain_terms = _collect_uncertain_formal_terms(card)
    demoted: list[dict[str, Any]] = []
    kept_roles: list[dict[str, Any]] = []
    for item in card.get("role_in_story") or []:
        if not isinstance(item, dict):
            kept_roles.append(item)
            continue
        value = str(item.get("value") or "")
        matched = [term for term in _FORMAL_ROLE_TERMS if term and term in value]
        status = str(item.get("status") or "").strip().lower()
        explicitly_supported = status == "explicit"
        risk_flagged = any(term in uncertain_terms for term in matched)
        if not matched or (explicitly_supported and not risk_flagged):
            kept_roles.append(item)
            continue
        demoted.append(
            {
                "claim": value,
                "reason": "formal role label is not explicitly supported; keep only the underlying evidence-backed behavior or promise",
            }
        )
    if demoted:
        card["role_in_story"] = kept_roles
        card.setdefault("uncertain_or_unsupported", [])
        card["uncertain_or_unsupported"].extend(demoted)


def _collect_uncertain_formal_terms(card: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    risk_text = " ".join(
        str(item.get("risk") or "")
        for item in card.get("simulation_risks") or []
        if isinstance(item, dict)
    )
    uncertain_text = " ".join(
        f"{item.get('claim') or ''} {item.get('reason') or ''}"
        for item in card.get("uncertain_or_unsupported") or []
        if isinstance(item, dict)
    )
    combined = risk_text + " " + uncertain_text
    for term in _FORMAL_ROLE_TERMS:
        if term in combined:
            terms.add(term)
    return terms


def _normalize_hard_constraint_wording(card: dict[str, Any]) -> None:
    for item in card.get("hard_constraints") or []:
        if not isinstance(item, dict):
            continue
        constraint = str(item.get("constraint") or "")
        for prefix in ("必须提及", "必须承认", "必须写出", "必须包含", "必须说明"):
            if constraint.startswith(prefix):
                item["constraint"] = constraint[len(prefix) :].lstrip("：: ，,")
                break


def _prefix_boundary_label(boundary: dict[str, Any]) -> str:
    if boundary.get("before_scene_id"):
        return f"before {boundary.get('before_scene_id')}"
    if boundary.get("before_scene_order") is not None:
        return f"before scene order {boundary.get('before_scene_order')}"
    return ""


def _safe_call_id(entity: dict[str, Any]) -> str:
    raw = str(entity.get("entity_id") or entity.get("canonical_name") or "entity")
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)
    return safe or "entity"


def _append_items(lines: list[str], label: str, items: Any, value_key: str) -> None:
    if not items:
        return
    lines.append("")
    lines.append(f"{label}:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        value = item.get(value_key) or item.get("value") or item.get("trait") or item.get("tendency") or ""
        status = item.get("status")
        refs = _format_refs(item.get("refs") or [])
        meta = f" ({status}; {refs})" if status or refs else ""
        lines.append(f"- {value}{meta}")


def _append_relationships(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("")
    lines.append("relationship stances:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        target = item.get("target") or "unknown"
        stance = item.get("stance") or ""
        status = item.get("status")
        refs = _format_refs(item.get("refs") or [])
        meta = f" ({status}; {refs})" if status or refs else ""
        lines.append(f"- {target}: {stance}{meta}")


def _append_constraints(lines: list[str], label: str, items: Any) -> None:
    if not items:
        return
    lines.append("")
    lines.append(f"{label}:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        refs = _format_refs(item.get("refs") or [])
        suffix = f" ({refs})" if refs else ""
        lines.append(f"- {item.get('constraint') or ''}{suffix}")


def _append_risks(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("")
    lines.append("simulation risks:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        refs = _format_refs(item.get("refs") or [])
        suffix = f" ({refs})" if refs else ""
        lines.append(f"- {item.get('risk') or ''}{suffix}")


def _format_refs(refs: list[Any]) -> str:
    return ", ".join(str(ref) for ref in refs if str(ref).strip())


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
