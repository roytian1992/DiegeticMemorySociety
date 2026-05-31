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
class SocialSimulationConfig:
    attribute_cards_path: Path
    writing_intent: str
    output_dir: Path
    prompt_dir: Path = Path("task_specs/prompts")
    overwrite: bool = False


def run_social_simulation(config: SocialSimulationConfig, llm_client: LLMClient) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("inputs", "prompts", "raw_outputs", "parsed"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    cards_payload = json.loads(Path(config.attribute_cards_path).read_text(encoding="utf-8"))
    cards = _extract_cards(cards_payload)
    loader = YAMLPromptLoader(config.prompt_dir)
    calls: list[dict[str, Any]] = []
    character_simulations: list[dict[str, Any]] = []

    for card in cards:
        context = _build_character_simulation_context(
            card,
            cards=cards,
            writing_intent=config.writing_intent,
        )
        call_id = f"character_{_safe_id(card.get('entity_id') or card.get('canonical_name'))}"
        (output_dir / "inputs" / f"{call_id}.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        parsed = _run_json_prompt(
            loader,
            llm_client,
            output_dir,
            calls,
            call_id=call_id,
            prompt_id="dms/character_social_simulation",
            task_values={"simulation_context_json": context},
        )
        character_simulations.append(_normalize_character_simulation(parsed, card))

    coordinator_context = {
        "writing_intent": config.writing_intent,
        "prefix_boundary": _prefix_boundary(cards),
        "attribute_cards": cards,
        "character_simulations": character_simulations,
    }
    (output_dir / "inputs" / "coordinator.json").write_text(
        json.dumps(coordinator_context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coordinator_payload = _run_json_prompt(
        loader,
        llm_client,
        output_dir,
        calls,
        call_id="coordinator",
        prompt_id="dms/social_simulation_coordinator",
        task_values={"coordinator_context_json": coordinator_context},
    )
    social_simulation = _normalize_coordinator_payload(coordinator_payload, coordinator_context)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "provider": llm_client.provider,
            "model": llm_client.model,
        },
        "inputs": {
            "attribute_cards_path": str(config.attribute_cards_path),
            "writing_intent": config.writing_intent,
            "card_count": len(cards),
        },
        "character_simulation_count": len(character_simulations),
        "character_simulations": character_simulations,
        "social_simulation": social_simulation,
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "character_simulations_json": str(output_dir / "character_simulations.json"),
            "social_simulation_json": str(output_dir / "social_simulation.json"),
            "social_simulation_markdown": str(output_dir / "social_simulation.md"),
            "calls": str(output_dir / "calls.jsonl"),
            "inputs_dir": str(output_dir / "inputs"),
            "prompts_dir": str(output_dir / "prompts"),
            "raw_outputs_dir": str(output_dir / "raw_outputs"),
            "parsed_dir": str(output_dir / "parsed"),
        },
    }
    (output_dir / "character_simulations.json").write_text(
        json.dumps(character_simulations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "social_simulation.json").write_text(
        json.dumps(social_simulation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "social_simulation.md").write_text(format_social_simulation_markdown(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "calls.jsonl", calls)
    return summary


def format_social_simulation_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Social Simulation", ""]
    inputs = summary.get("inputs") or {}
    if inputs.get("attribute_cards_path"):
        lines.append(f"- source: {inputs.get('attribute_cards_path')}")
    if inputs.get("writing_intent"):
        lines.append(f"- writing intent: {inputs.get('writing_intent')}")
    lines.append(f"- character simulations: {summary.get('character_simulation_count', 0)}")

    character_simulations = summary.get("character_simulations") or []
    if character_simulations:
        lines.append("")
        lines.append("## Character Simulations")
        for simulation in character_simulations:
            lines.append("")
            lines.append(f"### {simulation.get('character')}")
            _append_simple_list(lines, "intent assumptions", simulation.get("intent_assumptions"))
            _append_items(lines, "internal state", simulation.get("likely_internal_state"), "value")
            _append_items(lines, "actions", simulation.get("likely_actions"), "value")
            _append_items(lines, "dialogue", simulation.get("likely_dialogue"), "value")
            _append_interaction_pressure(lines, simulation.get("interaction_pressure"))
            _append_risks(lines, "avoid / risks", simulation.get("avoid_or_risks"))

    social = summary.get("social_simulation") or {}
    if social:
        lines.append("")
        lines.append("## Coordinated Beats")
        for index, beat in enumerate(social.get("scene_beats") or [], start=1):
            if not isinstance(beat, dict):
                lines.append(f"{index}. {beat}")
                continue
            participants = ", ".join(str(item) for item in beat.get("participants") or [])
            basis = _format_refs(beat.get("memory_basis") or [])
            suffix = f" [{basis}]" if basis else ""
            lines.append(f"{index}. {beat.get('beat') or ''}{suffix}")
            if participants:
                lines.append(f"   participants: {participants}")
            if beat.get("purpose"):
                lines.append(f"   purpose: {beat.get('purpose')}")
            intent_basis = beat.get("intent_basis") or []
            if intent_basis:
                lines.append(f"   intent basis: {'; '.join(str(item) for item in intent_basis)}")

        _append_dynamics(lines, social.get("character_dynamics"))
        _append_risks(lines, "memory risks", social.get("memory_risks"))
        _append_guidance(lines, social.get("writer_guidance"))

    return "\n".join(lines).rstrip() + "\n"


def _extract_cards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        cards = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        cards = payload["cards"]
    else:
        cards = []
    return [dict(card) for card in cards if isinstance(card, dict)]


def _build_character_simulation_context(
    card: dict[str, Any],
    *,
    cards: list[dict[str, Any]],
    writing_intent: str,
) -> dict[str, Any]:
    target_name = str(card.get("canonical_name") or "")
    other_cards = [
        _compact_peer_card(peer)
        for peer in cards
        if peer is not card and peer.get("canonical_name") != target_name
    ]
    return {
        "writing_intent": writing_intent,
        "target_card": card,
        "other_visible_cards": other_cards,
        "instructions": {
            "output_language": "Chinese",
            "allowed_refs": sorted(_collect_card_refs(card)),
        },
    }


def _compact_peer_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": card.get("entity_id"),
        "canonical_name": card.get("canonical_name"),
        "entity_type": card.get("entity_type"),
        "prefix_boundary": card.get("prefix_boundary"),
        "role_in_story": card.get("role_in_story") or [],
        "current_state": card.get("current_state") or [],
        "relationship_stances": card.get("relationship_stances") or [],
    }


def _collect_card_refs(card: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for value in card.values():
        _collect_refs_from_value(value, refs)
    return refs


def _collect_refs_from_value(value: Any, refs: set[str]) -> None:
    if isinstance(value, dict):
        raw_refs = value.get("refs")
        if isinstance(raw_refs, list):
            refs.update(str(ref) for ref in raw_refs if str(ref).strip())
        for child in value.values():
            _collect_refs_from_value(child, refs)
    elif isinstance(value, list):
        for item in value:
            _collect_refs_from_value(item, refs)


def _run_json_prompt(
    loader: YAMLPromptLoader,
    llm_client: LLMClient,
    output_dir: Path,
    calls: list[dict[str, Any]],
    *,
    call_id: str,
    prompt_id: str,
    task_values: dict[str, Any],
) -> Any:
    prompt = loader.render(prompt_id, task_values=task_values)
    prompt_path = output_dir / "prompts" / f"{call_id}.txt"
    raw_path = output_dir / "raw_outputs" / f"{call_id}.json"
    parsed_path = output_dir / "parsed" / f"{call_id}.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = llm_client.complete(prompt)
    raw_payload = _llm_result_to_dict(result)
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    parsed = extract_json_value(result.text)
    parsed_payload = {
        "call_id": call_id,
        "prompt_id": prompt_id,
        "status": "parsed" if parsed.ok else "parse_failed",
        "data": parsed.data,
        "parse_error": parsed.error,
    }
    parsed_path.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calls.append(
        {
            "call_id": call_id,
            "prompt_id": prompt_id,
            "prompt_path": str(prompt_path),
            "raw_output_path": str(raw_path),
            "parsed_path": str(parsed_path),
            "status": parsed_payload["status"],
            "usage": result.usage,
        }
    )
    if not parsed.ok:
        raise ValueError(f"Failed to parse JSON for {call_id}: {parsed.error}")
    return parsed.data


def _normalize_character_simulation(data: Any, card: dict[str, Any]) -> dict[str, Any]:
    simulation = dict(data) if isinstance(data, dict) else {}
    simulation.setdefault("character", card.get("canonical_name"))
    simulation["prefix_boundary"] = card.get("prefix_boundary") or simulation.get("prefix_boundary") or ""
    for key in (
        "intent_assumptions",
        "likely_internal_state",
        "likely_actions",
        "likely_dialogue",
        "interaction_pressure",
        "avoid_or_risks",
        "memory_basis",
    ):
        if not isinstance(simulation.get(key), list):
            simulation[key] = []
    return simulation


def _normalize_coordinator_payload(data: Any, context: dict[str, Any]) -> dict[str, Any]:
    social = dict(data) if isinstance(data, dict) else {}
    social.setdefault("simulation_id", "social_simulation")
    social["prefix_boundary"] = context.get("prefix_boundary") or social.get("prefix_boundary") or ""
    for key in ("scene_beats", "character_dynamics", "memory_risks", "writer_guidance"):
        if not isinstance(social.get(key), list):
            social[key] = []
    return social


def _prefix_boundary(cards: list[dict[str, Any]]) -> str:
    for card in cards:
        if card.get("prefix_boundary"):
            return str(card["prefix_boundary"])
    return ""


def _safe_id(value: Any) -> str:
    raw = str(value or "item")
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)
    return safe or "item"


def _append_items(lines: list[str], label: str, items: Any, value_key: str) -> None:
    if not items:
        return
    lines.append(f"{label}:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        value = item.get(value_key) or item.get("value") or ""
        refs = _format_refs(item.get("refs") or [])
        status = item.get("status")
        meta = f" ({status}; {refs})" if status or refs else ""
        lines.append(f"- {value}{meta}")


def _append_simple_list(lines: list[str], label: str, items: Any) -> None:
    if not items:
        return
    lines.append(f"{label}:")
    for item in items:
        lines.append(f"- {item}")


def _append_interaction_pressure(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("interaction pressure:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        target = item.get("target") or "unknown"
        pressure = item.get("pressure") or ""
        refs = _format_refs(item.get("refs") or [])
        status = item.get("status")
        meta = f" ({status}; {refs})" if status or refs else ""
        lines.append(f"- {target}: {pressure}{meta}")


def _append_dynamics(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("")
    lines.append("character dynamics:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        source = item.get("source") or "unknown"
        target = item.get("target") or "unknown"
        refs = _format_refs(item.get("refs") or [])
        suffix = f" ({refs})" if refs else ""
        lines.append(f"- {source} -> {target}: {item.get('dynamic') or ''}{suffix}")


def _append_risks(lines: list[str], label: str, items: Any) -> None:
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
        lines.append(f"- {item.get('risk') or ''}{suffix}")


def _append_guidance(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("")
    lines.append("writer guidance:")
    for item in items:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        refs = _format_refs(item.get("refs") or [])
        suffix = f" ({refs})" if refs else ""
        lines.append(f"- {item.get('guidance') or ''}{suffix}")


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
