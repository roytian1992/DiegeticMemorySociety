from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.llm import LLMClient, LLMResult
from dms.parsing import extract_json_value
from dms.prompts import YAMLPromptLoader
from dms.simulation.algorithmic import build_algorithmic_social_plan
from dms.simulation.formatting import format_social_simulation_markdown, format_social_simulation_writer_packet
from dms.simulation.verification import verify_social_simulation, verify_writer_packet

_FORBIDDEN_SOURCE_CONTEXT_KEYS = {
    "content",
    "unit_json",
    "target_scene",
    "target_scene_text",
    "target_text",
    "reference_text",
    "writing_spec",
    "reference_scene_spec",
}


@dataclass(frozen=True)
class SocialSimulationConfig:
    attribute_cards_path: Path
    output_dir: Path
    social_simulation_intent: str = ""
    prompt_dir: Path = Path("task_specs/prompts")
    overwrite: bool = False
    writing_intent: str = ""


def run_social_simulation(config: SocialSimulationConfig, llm_client: LLMClient) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("inputs", "prompts", "raw_outputs", "parsed"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    cards_payload = json.loads(Path(config.attribute_cards_path).read_text(encoding="utf-8"))
    cards = _extract_cards(cards_payload)
    social_simulation_intent = _resolve_social_simulation_intent(config)
    loader = YAMLPromptLoader(config.prompt_dir)
    calls: list[dict[str, Any]] = []
    character_simulations: list[dict[str, Any]] = []

    for card in cards:
        context = _build_character_simulation_context(
            card,
            cards=cards,
            social_simulation_intent=social_simulation_intent,
        )
        _assert_no_forbidden_source_context(context, path=f"character:{card.get('canonical_name') or card.get('entity_id')}")
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
        "social_simulation_intent": social_simulation_intent,
        "prefix_boundary": _prefix_boundary(cards),
        "attribute_cards": cards,
        "character_simulations": character_simulations,
    }
    _assert_no_forbidden_source_context(coordinator_context, path="coordinator")
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
    verification = verify_social_simulation(
        cards=cards,
        character_simulations=character_simulations,
        social_simulation=social_simulation,
        writing_intent=social_simulation_intent,
    )
    algorithmic_plan = build_algorithmic_social_plan(
        cards=cards,
        character_simulations=character_simulations,
        social_simulation=social_simulation,
        writing_intent=social_simulation_intent,
        verification=verification,
    )
    writer_packet_verification = verify_writer_packet(algorithmic_plan.get("writer_packet") or {})

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "provider": llm_client.provider,
            "model": llm_client.model,
        },
        "inputs": {
            "attribute_cards_path": str(config.attribute_cards_path),
            "social_simulation_intent": social_simulation_intent,
            "card_count": len(cards),
            "source_isolation": {
                "target_scene_text_visible": False,
                "writing_spec_visible": False,
                "allowed_new_scene_source": "social_simulation_intent",
            },
        },
        "character_simulation_count": len(character_simulations),
        "character_simulations": character_simulations,
        "social_simulation": social_simulation,
        "algorithmic_social_plan": algorithmic_plan,
        "verification": verification,
        "writer_packet_verification": writer_packet_verification,
        "social_simulation_metrics": algorithmic_plan.get("metrics", {}),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "character_simulations_json": str(output_dir / "character_simulations.json"),
            "social_simulation_json": str(output_dir / "social_simulation.json"),
            "algorithmic_social_plan_json": str(output_dir / "algorithmic_social_plan.json"),
            "verification_json": str(output_dir / "verification.json"),
            "writer_packet_verification_json": str(output_dir / "writer_packet_verification.json"),
            "social_simulation_markdown": str(output_dir / "social_simulation.md"),
            "writer_packet_markdown": str(output_dir / "writer_packet.md"),
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
    (output_dir / "algorithmic_social_plan.json").write_text(
        json.dumps(algorithmic_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "writer_packet_verification.json").write_text(
        json.dumps(writer_packet_verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "social_simulation.md").write_text(format_social_simulation_markdown(summary), encoding="utf-8")
    (output_dir / "writer_packet.md").write_text(format_social_simulation_writer_packet(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "calls.jsonl", calls)
    return summary


def _resolve_social_simulation_intent(config: SocialSimulationConfig) -> str:
    intent = str(config.social_simulation_intent or "").strip()
    legacy = str(config.writing_intent or "").strip()
    if intent:
        return intent
    if legacy:
        return legacy
    raise ValueError("social_simulation_intent is required")


def _assert_no_forbidden_source_context(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in _FORBIDDEN_SOURCE_CONTEXT_KEYS:
                raise ValueError(f"Forbidden target-scene source field in social simulation context: {child_path}")
            _assert_no_forbidden_source_context(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_source_context(child, path=f"{path}[{index}]")


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
    social_simulation_intent: str,
) -> dict[str, Any]:
    target_name = str(card.get("canonical_name") or "")
    other_cards = [
        _compact_peer_card(peer)
        for peer in cards
        if peer is not card and peer.get("canonical_name") != target_name
    ]
    return {
        "social_simulation_intent": social_simulation_intent,
        "target_card": card,
        "other_visible_cards": other_cards,
        "instructions": {
            "output_language": "Chinese",
            "allowed_refs": sorted(_collect_card_refs(card)),
            "target_scene_text_visible": False,
            "writing_spec_visible": False,
            "use_only_intent_and_prefix_cards": True,
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
