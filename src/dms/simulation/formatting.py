from __future__ import annotations

from typing import Any


def format_social_simulation_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Social Simulation", ""]
    inputs = summary.get("inputs") or {}
    if inputs.get("attribute_cards_path"):
        lines.append(f"- source: {inputs.get('attribute_cards_path')}")
    if inputs.get("social_simulation_intent"):
        lines.append(f"- social simulation intent: {inputs.get('social_simulation_intent')}")
    source_isolation = inputs.get("source_isolation") or {}
    if source_isolation:
        visible = "yes" if source_isolation.get("target_scene_text_visible") else "no"
        lines.append(f"- target scene text visible: {visible}")
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
            _append_dialogue_posture_from_examples(lines, simulation.get("likely_dialogue"))
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

    algorithmic_plan = summary.get("algorithmic_social_plan") or {}
    if algorithmic_plan:
        _append_algorithmic_plan(lines, algorithmic_plan)

    verification = summary.get("verification") or {}
    if verification:
        _append_verification(lines, verification)

    return "\n".join(lines).rstrip() + "\n"


def format_social_simulation_writer_packet(summary: dict[str, Any]) -> str:
    lines = ["# Social Simulation Writer Packet", ""]
    inputs = summary.get("inputs") or {}
    if inputs.get("social_simulation_intent"):
        lines.append(f"- social simulation intent: {inputs.get('social_simulation_intent')}")
    source_isolation = inputs.get("source_isolation") or {}
    if source_isolation:
        visible = "yes" if source_isolation.get("target_scene_text_visible") else "no"
        lines.append(f"- target scene text visible: {visible}")
    plan = summary.get("algorithmic_social_plan") or {}
    metrics = plan.get("metrics") or {}
    if metrics:
        lines.append(f"- selected sequence score: {metrics.get('selected_sequence_score', 0)}")
        lines.append(f"- hard violations: {metrics.get('hard_violation_count', 0)}")
        lines.append(f"- soft warnings: {metrics.get('soft_warning_count', 0)}")

    writer_packet = plan.get("writer_packet") or {}
    selected = writer_packet.get("use_as_optional_behavior") or []
    if selected:
        lines.append("")
        lines.append("## Optional Interaction Functions")
        for beat in selected:
            if not isinstance(beat, dict):
                continue
            participants = ", ".join(str(item) for item in beat.get("participants") or [])
            function = beat.get("interaction_function") or ""
            effects = beat.get("state_delta") or {}
            effect_text = ", ".join(f"{key}={value}" for key, value in effects.items())
            lines.append(f"- {beat.get('beat_id')}: {function}; participants: {participants}")
            if effect_text:
                lines.append(f"  effects: {effect_text}")
            action_guidance = beat.get("action_guidance") or []
            for action in action_guidance:
                if not isinstance(action, dict):
                    continue
                lines.append(
                    "  action: "
                    f"{action.get('actor')} / {action.get('action_type')} / {action.get('surface_action')}"
                )
                if action.get("dialogue_intent"):
                    lines.append(f"  dialogue intent: {action.get('dialogue_intent')}")
            verification = beat.get("verification") or {}
            style_risks = verification.get("style_risks") or []
            if style_risks:
                lines.append(f"  risks: {'; '.join(str(item) for item in style_risks)}")

    posture = writer_packet.get("dialogue_posture") or []
    if posture:
        lines.append("")
        lines.append("## Dialogue Posture")
        for item in posture:
            if not isinstance(item, dict):
                continue
            tone = ", ".join(str(value) for value in item.get("tone") or [])
            avoid = ", ".join(str(value) for value in item.get("avoid_phrases") or [])
            lines.append(f"- {item.get('beat_id')}: {item.get('dialogue_function')}; tone: {tone}")
            if avoid:
                lines.append(f"  avoid phrases: {avoid}")
            if item.get("not_canonical_dialogue"):
                lines.append("  note: posture only; do not copy as final dialogue")

    verification = summary.get("writer_packet_verification") or {}
    warnings = verification.get("soft_warnings") or []
    if warnings:
        lines.append("")
        lines.append("## Writer Packet Warnings")
        for warning in warnings[:8]:
            if not isinstance(warning, dict):
                continue
            lines.append(f"- {warning.get('type')}: {warning.get('detail') or warning.get('phrase') or warning.get('term')}")

    lines.append("")
    lines.append("## Use Rules")
    lines.append("- This packet is behavior guidance, not a fact source.")
    lines.append("- Follow the writing request and memory packet before this packet.")
    lines.append("- Treat all beats as optional interaction functions.")
    lines.append("- Do not copy any wording from simulation artifacts as final prose.")
    lines.append("- Prefer concrete action, cockpit/environment pressure, and short dialogue posture.")
    return "\n".join(lines).rstrip() + "\n"


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


def _append_dialogue_posture_from_examples(lines: list[str], items: Any) -> None:
    if not items:
        return
    lines.append("dialogue posture:")
    for item in items:
        if not isinstance(item, dict):
            lines.append("- posture only; avoid treating raw simulation wording as canonical dialogue")
            continue
        value = str(item.get("value") or "").strip()
        refs = _format_refs(item.get("refs") or [])
        status = item.get("status")
        posture = _dialogue_example_to_posture(value)
        meta = f" ({status}; {refs})" if status or refs else ""
        lines.append(f"- {posture}{meta}")


def _dialogue_example_to_posture(value: str) -> str:
    if not value:
        return "posture only; keep dialogue concrete and brief"
    if "短" in value or "慢" in value or "稳" in value:
        return "short, concrete correction or response; do not copy example wording"
    if "地面" in value or "地球" in value or "天" in value:
        return "compressed value resistance or skyward motivation; avoid psychological paraphrase"
    if "知道" in value or "改" in value:
        return "minimal compliance without fully resolving tension"
    return "scene-facing dialogue posture; do not copy example wording"


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


def _append_algorithmic_plan(lines: list[str], plan: dict[str, Any]) -> None:
    lines.append("")
    lines.append("## Algorithmic Social Plan")
    metrics = plan.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append("metrics:")
        for key in (
            "pressure_graph_edge_count",
            "candidate_action_count",
            "selected_beat_count",
            "selected_sequence_score",
            "candidate_sequence_count",
            "selected_sequence_score_margin",
            "rejected_sequence_count",
            "scene_problem_coverage",
            "relationship_coverage",
            "memory_support_rate",
            "hard_violation_count",
            "soft_warning_count",
            "awkward_phrase_risk_count",
            "unique_candidate_memory_ref_count",
        ):
            if key in metrics:
                lines.append(f"- {key}: {metrics[key]}")

    pressure_edges = (plan.get("pressure_graph") or {}).get("edges") or []
    if pressure_edges:
        lines.append("")
        lines.append("pressure graph:")
        for edge in pressure_edges[:5]:
            if not isinstance(edge, dict):
                continue
            lines.append(
                "- "
                f"{edge.get('source')} -> {edge.get('target')}: "
                f"{edge.get('pressure_type')} "
                f"(strength={edge.get('strength')})"
            )

    sequence = plan.get("selected_sequence") or {}
    beats = sequence.get("beats") or []
    if beats:
        lines.append("")
        lines.append("selected beats:")
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            participants = ", ".join(str(item) for item in beat.get("participants") or [])
            actions = ", ".join(str(item) for item in beat.get("actions") or [])
            lines.append(
                "- "
                f"{beat.get('beat_id')}: {beat.get('interaction_function')} "
                f"[{participants}] actions={actions}"
            )
    rejected = sequence.get("rejected_sequences") or []
    if rejected:
        lines.append("")
        lines.append("rejected sequence options:")
        for option in rejected[:3]:
            if not isinstance(option, dict):
                continue
            lines.append(
                "- "
                f"{option.get('sequence_id')}: {option.get('strategy')} "
                f"score={option.get('score')} reason={option.get('rejection_reason')}"
            )

    writer_packet = plan.get("writer_packet") or {}
    posture = writer_packet.get("dialogue_posture") or []
    if posture:
        lines.append("")
        lines.append("dialogue posture:")
        for item in posture:
            if not isinstance(item, dict):
                continue
            tone = ", ".join(str(value) for value in item.get("tone") or [])
            avoid = ", ".join(str(value) for value in item.get("avoid_phrases") or [])
            suffix = f"; avoid: {avoid}" if avoid else ""
            lines.append(f"- {item.get('beat_id')}: {item.get('dialogue_function')} ({tone}){suffix}")


def _append_verification(lines: list[str], verification: dict[str, Any]) -> None:
    lines.append("")
    lines.append("## Verification")
    lines.append(f"- status: {verification.get('status')}")
    metrics = verification.get("metrics") or {}
    for key in (
        "hard_violation_count",
        "soft_warning_count",
        "therapy_phrase_risk_count",
        "unsupported_role_risk_count",
        "final_dialogue_like_guidance_count",
    ):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    hard_violations = verification.get("hard_violations") or []
    if hard_violations:
        lines.append("")
        lines.append("hard violations:")
        for item in hard_violations:
            if isinstance(item, dict):
                lines.append(f"- {item.get('type')}: {item.get('detail') or item.get('phrase') or item.get('ref')}")
    warnings = verification.get("soft_warnings") or []
    if warnings:
        lines.append("")
        lines.append("soft warnings:")
        for item in warnings[:8]:
            if isinstance(item, dict):
                detail = item.get("detail") or item.get("phrase") or item.get("term") or ""
                lines.append(f"- {item.get('type')}: {detail}")


def _format_refs(refs: list[Any]) -> str:
    return ", ".join(str(ref) for ref in refs if str(ref).strip())
