from __future__ import annotations

import math
from typing import Any

from dms.simulation.verification import detect_text_risks


ACTION_TYPES = (
    "risky_operation",
    "safety_correction",
    "minimal_compliance",
    "value_resistance",
    "deflection",
    "care_reframe",
    "information_probe",
    "silence_or_withholding",
    "physical_reaction",
    "environmental_pressure",
)

SAFETY_TERMS = ("危险", "风险", "安全", "稳", "慢", "低空", "告警", "警报", "高度", "姿态", "返航", "飞行")
CARE_TERMS = ("照看", "照顾", "关切", "保护", "提醒", "劝", "承诺", "带着", "后辈", "关心")
RESISTANCE_TERMS = ("抗拒", "不愿", "拒绝", "回避", "压抑", "厌", "地球", "不美好", "别管", "沉默")
RELATION_TERMS = ("关系", "照看", "提醒", "承诺", "同伴", "张鹏", "刘培强")
SETTING_TERMS = ("驾驶舱", "返航", "飞行", "战区", "废墟", "车辆", "基地", "室内", "路上")
ACTION_SUPPORT_TERMS = {
    "risky_operation": SAFETY_TERMS + ("操作", "驾驶", "偏快", "冒险", "压低"),
    "safety_correction": SAFETY_TERMS + CARE_TERMS + ("提醒", "纠正", "拉回"),
    "minimal_compliance": SAFETY_TERMS + RESISTANCE_TERMS + ("配合", "回应", "知道"),
    "value_resistance": RESISTANCE_TERMS + ("价值", "抵触", "沉默"),
    "deflection": RESISTANCE_TERMS + ("回避", "转开", "沉默"),
    "care_reframe": CARE_TERMS + SAFETY_TERMS + ("生存", "返航", "当下"),
    "information_probe": RELATION_TERMS + ("询问", "试探", "状态"),
    "silence_or_withholding": RESISTANCE_TERMS + RELATION_TERMS + ("沉默", "停顿"),
    "physical_reaction": SAFETY_TERMS + ("呼吸", "手", "视线", "姿态", "压力"),
}


def build_algorithmic_social_plan(
    *,
    cards: list[dict[str, Any]],
    character_simulations: list[dict[str, Any]],
    social_simulation: dict[str, Any],
    writing_intent: str,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic ASIP-like artifacts around current prompt output."""

    state_graph = build_social_state_graph(cards, character_simulations, writing_intent=writing_intent)
    pressure_graph = build_pressure_graph(state_graph, writing_intent=writing_intent)
    candidates = build_action_candidates(state_graph, pressure_graph, writing_intent=writing_intent)
    selected_sequence = select_beat_sequence(
        candidates,
        pressure_graph,
        social_simulation=social_simulation,
        verification=verification or {},
    )
    metrics = _build_metrics(state_graph, pressure_graph, candidates, selected_sequence, verification or {})
    writer_packet = _build_writer_packet(selected_sequence, candidates, verification or {})
    writer_metrics = _build_writer_metrics(writer_packet, verification or {})
    metrics.update(writer_metrics)
    return {
        "version": "asip_v0",
        "state_graph": state_graph,
        "pressure_graph": pressure_graph,
        "candidate_actions": candidates,
        "selected_sequence": selected_sequence,
        "writer_packet": writer_packet,
        "metrics": metrics,
    }


def build_social_state_graph(
    cards: list[dict[str, Any]],
    character_simulations: list[dict[str, Any]],
    *,
    writing_intent: str,
) -> dict[str, Any]:
    simulations_by_name = {
        str(item.get("character") or ""): item
        for item in character_simulations
        if isinstance(item, dict)
    }
    characters = []
    for card in cards:
        name = str(card.get("canonical_name") or "")
        simulation = simulations_by_name.get(name, {})
        evidence_texts = _card_texts(card)
        simulation_texts = _simulation_texts(simulation)
        support_items = _support_items_from_value(card) + _support_items_from_value(simulation)
        combined = " ".join(evidence_texts + simulation_texts + [writing_intent])
        relations = []
        for stance in card.get("relationship_stances") or []:
            if not isinstance(stance, dict):
                continue
            target = str(stance.get("target") or "").strip()
            if not target:
                continue
            text = _text_from_item(stance)
            relations.append(
                {
                    "target": target,
                    "stance": text,
                    "refs": _refs_from_item(stance),
                    "trust": _score_terms(text, CARE_TERMS + RELATION_TERMS, default=0.45),
                    "resistance": _score_terms(text, RESISTANCE_TERMS, default=0.25),
                    "care": _score_terms(text, CARE_TERMS, default=0.25),
                }
            )
        characters.append(
            {
                "character_id": card.get("entity_id") or name,
                "name": name,
                "prefix_boundary": card.get("prefix_boundary") or "",
                "public_state": {
                    "roles": _values(card.get("role_in_story"), "value"),
                    "current_state": _values(card.get("current_state"), "value"),
                    "visible_action_state": _infer_visible_action_state(writing_intent, combined),
                },
                "private_state": {
                    "values_or_motivations": _values(card.get("values_or_motivations"), "value"),
                    "likely_internal_state": _values(simulation.get("likely_internal_state"), "value"),
                    "local_goal": _infer_local_goal(card, simulation, writing_intent),
                    "resistance": _score_terms(combined, RESISTANCE_TERMS, default=0.25),
                },
                "affect": {
                    "arousal": _score_terms(combined, ("紧张", "急", "压", "危险", "告警", "警报"), default=0.45),
                    "negative_valence": _score_terms(combined, ("厌", "不美好", "废墟", "危险", "压抑"), default=0.35),
                },
                "control_style": {
                    "risk_tolerance": _score_terms(combined, ("危险", "冒险", "低空", "快", "急", "不稳"), default=0.35),
                    "compliance": 1.0 - _score_terms(combined, RESISTANCE_TERMS, default=0.35),
                },
                "relationship_stances": relations,
                "support_items": support_items,
                "memory_support_refs": _rank_refs(_refs_from_value(card), combined)[:8],
                "hard_constraints": card.get("hard_constraints") or [],
                "risks": (card.get("simulation_risks") or []) + (simulation.get("avoid_or_risks") or []),
            }
        )
    return {
        "characters": characters,
        "scene_frame": {
            "writing_intent": writing_intent,
            "setting_type": _infer_setting_type(writing_intent),
            "interaction_problem": _infer_interaction_problem(writing_intent),
        },
    }


def build_pressure_graph(state_graph: dict[str, Any], *, writing_intent: str) -> dict[str, Any]:
    characters = state_graph.get("characters") or []
    edges: list[dict[str, Any]] = []
    for source in characters:
        for target in characters:
            if source is target:
                continue
            source_text = _state_character_text(source)
            target_text = _state_character_text(target)
            relation = _relation_to(source, target.get("name"))
            pressure_type = _infer_pressure_type(source_text, target_text, relation, writing_intent, source_name=source.get("name"))
            memory_support = _support_score(source.get("memory_support_refs") or [])
            intent_relevance = _score_terms(writing_intent + " " + source_text + " " + target_text, SAFETY_TERMS + RESISTANCE_TERMS)
            relationship_salience = max(float(relation.get("care") or 0), float(relation.get("trust") or 0), 0.25 if relation else 0.0)
            state_urgency = max(
                float(source.get("affect", {}).get("arousal") or 0),
                float(target.get("affect", {}).get("arousal") or 0),
                _score_terms(writing_intent, SAFETY_TERMS),
            )
            strength = round(
                min(
                    1.0,
                    0.35 * memory_support
                    + 0.25 * intent_relevance
                    + 0.20 * relationship_salience
                    + 0.10 * 0.7
                    + 0.10 * state_urgency,
                ),
                3,
            )
            if strength < 0.35:
                continue
            edges.append(
                {
                    "source": source.get("name"),
                    "target": target.get("name"),
                    "pressure_type": pressure_type,
                    "desired_change": _desired_change(pressure_type),
                    "memory_basis": sorted(set((source.get("memory_support_refs") or []) + (relation.get("refs") or []))),
                    "intent_basis": _intent_basis(writing_intent, pressure_type),
                    "strength": strength,
                    "priority": "high" if strength >= 0.75 else "normal",
                    "tone_bounds": _tone_bounds(pressure_type),
                }
            )
    edges.sort(key=lambda item: float(item.get("strength") or 0), reverse=True)
    return {"nodes": [item.get("name") for item in characters], "edges": edges}


def build_action_candidates(
    state_graph: dict[str, Any],
    pressure_graph: dict[str, Any],
    *,
    writing_intent: str,
) -> list[dict[str, Any]]:
    edges = pressure_graph.get("edges") or []
    candidates: list[dict[str, Any]] = []
    for character in state_graph.get("characters") or []:
        name = str(character.get("name") or "")
        related_edges = [edge for edge in edges if edge.get("source") == name or edge.get("target") == name]
        action_types = _candidate_action_types(character, related_edges, writing_intent)
        for action_type in action_types[:6]:
            intent_alignment = _intent_alignment(action_type, writing_intent)
            motivation = _motivation_score(action_type, character, related_edges)
            support_refs = _candidate_support_refs(character, action_type)
            memory_support = _support_score(support_refs)
            productivity = _interaction_productivity(action_type, related_edges)
            novelty = 0.7
            risks = _candidate_risks(action_type, character)
            risk_penalty = 0.1 * len(risks)
            score = round(
                max(
                    0.0,
                    min(
                        1.0,
                        0.30 * intent_alignment
                        + 0.25 * motivation
                        + 0.20 * memory_support
                        + 0.10 * productivity
                        + 0.10 * novelty
                        - risk_penalty,
                    ),
                ),
                3,
            )
            candidates.append(
                {
                    "candidate_id": f"a_{len(candidates) + 1:03d}",
                    "actor": name,
                    "action_type": action_type,
                    "surface_action": _surface_action(action_type, name),
                    "dialogue_intent": _dialogue_intent(action_type, name),
                    "not_final_dialogue": True,
                    "memory_basis": support_refs,
                    "intent_basis": _intent_basis(writing_intent, action_type),
                    "preconditions": _preconditions(action_type),
                    "effects": _effects(action_type),
                    "risks": risks,
                    "score": score,
                }
            )
    candidates.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    return candidates


def select_beat_sequence(
    candidates: list[dict[str, Any]],
    pressure_graph: dict[str, Any],
    *,
    social_simulation: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    options = _build_sequence_options(candidates, pressure_graph, verification=verification)
    if not options:
        options = [_fallback_sequence_option(candidates, verification=verification)]
    options = _dedupe_sequence_options([option for option in options if option.get("beats")])
    options.sort(key=lambda item: (float(item.get("score") or 0.0), _sequence_priority_sum(item)), reverse=True)
    selected = dict(options[0]) if options else _fallback_sequence_option(candidates, verification=verification)
    rejected = [_rejected_sequence_summary(option, selected) for option in options[1:6]]
    selected["selection_strategy"] = "multi_candidate_rerank_v1"
    selected["selected_sequence_id"] = selected.get("sequence_id")
    selected["candidate_sequence_count"] = len(options)
    selected["rejected_sequence_count"] = len(rejected)
    selected["rejected_sequences"] = rejected
    selected["score_margin"] = (
        round(float(selected.get("score") or 0.0) - float(options[1].get("score") or 0.0), 3)
        if len(options) > 1
        else float(selected.get("score") or 0.0)
    )
    prompt_beats = social_simulation.get("scene_beats") or []
    selected["prompt_coordinator_beats"] = prompt_beats
    return selected


def _build_sequence_options(
    candidates: list[dict[str, Any]],
    pressure_graph: dict[str, Any],
    *,
    verification: dict[str, Any],
) -> list[dict[str, Any]]:
    edges = pressure_graph.get("edges") or []
    if not edges:
        return [_fallback_sequence_option(candidates, verification=verification)]
    variant_specs = [
        ("pressure_default", "function_preferred", edges),
        ("relationship_first", "function_preferred", _relationship_first_edges(edges)),
        ("resistance_first", "function_preferred", _resistance_first_edges(edges)),
        ("score_first", "score_preferred", edges),
        ("care_posture", "care_first", edges),
        ("low_directive", "low_directive", edges),
    ]
    options = []
    for index, (strategy, candidate_policy, ordered_edges) in enumerate(variant_specs, start=1):
        beats = _beats_for_edges(candidates, ordered_edges[:5], candidate_policy=candidate_policy)
        if not beats:
            continue
        options.append(_finalize_sequence_option(f"s_{index:03d}", strategy, beats, verification=verification))
    return options


def _beats_for_edges(
    candidates: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    candidate_policy: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_candidates: set[str] = set()
    for edge in edges:
        pair = [edge.get("source"), edge.get("target")]
        pair_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("actor") in pair and candidate.get("candidate_id") not in used_candidates
        ]
        if not pair_candidates:
            continue
        first = _candidate_for_edge_policy(pair_candidates, edge, candidate_policy)
        if not first:
            continue
        used_candidates.add(str(first.get("candidate_id")))
        response = _response_candidate_for_policy(pair_candidates, first, edge, used_candidates, candidate_policy)
        action_ids = [first.get("candidate_id")]
        action_candidates: list[dict[str, Any] | None] = [first]
        if response:
            used_candidates.add(str(response.get("candidate_id")))
            action_ids.append(response.get("candidate_id"))
            action_candidates.append(response)
        selected.append(
            {
                "beat_id": f"b_{len(selected) + 1:03d}",
                "participants": [item for item in pair if item],
                "interaction_function": edge.get("pressure_type"),
                "actions": action_ids,
                "state_delta": _merge_effects(action_candidates),
                "required_in_writing": False,
                "priority": edge.get("strength"),
                "verification": {
                    "memory_supported": bool(edge.get("memory_basis")),
                    "intent_aligned": bool(edge.get("intent_basis")),
                    "hard_constraint_violations": [],
                    "style_risks": _style_risks_for_actions(action_candidates),
                },
            }
        )
        if len(selected) >= 5:
            break
    return selected


def _fallback_sequence_option(candidates: list[dict[str, Any]], *, verification: dict[str, Any]) -> dict[str, Any]:
    beats = []
    for candidate in candidates[:3]:
        beats.append(
            {
                "beat_id": f"b_{len(beats) + 1:03d}",
                "participants": [candidate.get("actor")],
                "interaction_function": candidate.get("action_type"),
                "actions": [candidate.get("candidate_id")],
                "state_delta": candidate.get("effects") or {},
                "required_in_writing": False,
                "priority": candidate.get("score"),
                "verification": {
                    "memory_supported": bool(candidate.get("memory_basis")),
                    "intent_aligned": bool(candidate.get("intent_basis")),
                    "hard_constraint_violations": [],
                    "style_risks": _style_risks_for_actions([candidate]),
                },
            }
        )
    return _finalize_sequence_option("s_fallback", "fallback_top_candidates", beats, verification=verification)


def _finalize_sequence_option(
    sequence_id: str,
    strategy: str,
    beats: list[dict[str, Any]],
    *,
    verification: dict[str, Any],
) -> dict[str, Any]:
    score_payload = _sequence_score_payload(beats, verification)
    return {
        "sequence_id": sequence_id,
        "strategy": strategy,
        "score": score_payload["score"],
        "score_components": score_payload["components"],
        "coverage": _coverage_for_beats(beats),
        "beats": beats,
    }


def _dedupe_sequence_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for option in options:
        signature = tuple(tuple(str(action) for action in beat.get("actions") or []) for beat in option.get("beats") or [])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(option)
    return deduped


def _preferred_candidate_for_edge(candidates: list[dict[str, Any]], edge: dict[str, Any]) -> dict[str, Any]:
    source = edge.get("source")
    pressure_type = edge.get("pressure_type")
    preferred_types = {
        "value_resistance": ["risky_operation", "value_resistance", "deflection", "physical_reaction"],
        "care_guidance": ["safety_correction", "care_reframe", "information_probe", "physical_reaction"],
        "safety_correction": ["safety_correction", "care_reframe", "physical_reaction"],
    }.get(str(pressure_type), [])
    scoped = [
        candidate
        for candidate in candidates
        if candidate.get("actor") == source and candidate.get("action_type") in set(preferred_types)
    ]
    if scoped:
        return _select_by_type_order(scoped, preferred_types)
    source_candidates = [candidate for candidate in candidates if candidate.get("actor") == source]
    return source_candidates[0] if source_candidates else candidates[0]


def _preferred_response_candidate(
    candidates: list[dict[str, Any]],
    first: dict[str, Any],
    edge: dict[str, Any],
    used_candidates: set[str],
) -> dict[str, Any] | None:
    target = edge.get("target")
    pressure_type = edge.get("pressure_type")
    if pressure_type == "value_resistance":
        preferred_types = ["safety_correction", "care_reframe", "minimal_compliance", "physical_reaction"]
    elif pressure_type in {"care_guidance", "safety_correction"}:
        preferred_types = ["minimal_compliance", "value_resistance", "deflection", "physical_reaction"]
    else:
        preferred_types = ["information_probe", "silence_or_withholding", "physical_reaction"]
    target_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("actor") == target
        and candidate.get("candidate_id") != first.get("candidate_id")
    ]
    preferred = [candidate for candidate in target_candidates if candidate.get("action_type") in set(preferred_types)]
    if preferred:
        return _select_by_type_order(preferred, preferred_types)
    unused = [candidate for candidate in target_candidates if candidate.get("candidate_id") not in used_candidates]
    return unused[0] if unused else target_candidates[0] if target_candidates else None


def _candidate_for_edge_policy(
    candidates: list[dict[str, Any]],
    edge: dict[str, Any],
    candidate_policy: str,
) -> dict[str, Any] | None:
    source = edge.get("source")
    source_candidates = [candidate for candidate in candidates if candidate.get("actor") == source]
    if candidate_policy == "score_preferred":
        return source_candidates[0] if source_candidates else candidates[0] if candidates else None
    if candidate_policy == "care_first":
        preferred = [
            candidate
            for candidate in source_candidates
            if candidate.get("action_type") in {"care_reframe", "safety_correction"}
        ]
        if preferred:
            return sorted(preferred, key=lambda item: float(item.get("score") or 0), reverse=True)[0]
    if candidate_policy == "low_directive":
        preferred = [
            candidate
            for candidate in source_candidates
            if candidate.get("action_type") in {"physical_reaction", "silence_or_withholding", "minimal_compliance"}
        ]
        if preferred:
            return sorted(preferred, key=lambda item: float(item.get("score") or 0), reverse=True)[0]
    return _preferred_candidate_for_edge(candidates, edge) if candidates else None


def _response_candidate_for_policy(
    candidates: list[dict[str, Any]],
    first: dict[str, Any],
    edge: dict[str, Any],
    used_candidates: set[str],
    candidate_policy: str,
) -> dict[str, Any] | None:
    if candidate_policy == "low_directive":
        target = edge.get("target")
        target_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("actor") == target
            and candidate.get("candidate_id") != first.get("candidate_id")
            and candidate.get("candidate_id") not in used_candidates
            and candidate.get("action_type") in {"minimal_compliance", "physical_reaction", "silence_or_withholding"}
        ]
        return target_candidates[0] if target_candidates else None
    return _preferred_response_candidate(candidates, first, edge, used_candidates)


def _relationship_first_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        edges,
        key=lambda item: (
            0 if item.get("pressure_type") in {"care_guidance", "safety_correction"} else 1,
            -float(item.get("strength") or 0),
        ),
    )


def _resistance_first_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        edges,
        key=lambda item: (
            0 if item.get("pressure_type") in {"value_resistance", "risky_operation"} else 1,
            -float(item.get("strength") or 0),
        ),
    )


def _select_by_type_order(candidates: list[dict[str, Any]], ordered_types: list[str]) -> dict[str, Any]:
    order = {action_type: index for index, action_type in enumerate(ordered_types)}
    return sorted(
        candidates,
        key=lambda item: (
            order.get(str(item.get("action_type") or ""), len(order)),
            -float(item.get("score") or 0),
        ),
    )[0]


def _build_metrics(
    state_graph: dict[str, Any],
    pressure_graph: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_sequence: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    verification_metrics = verification.get("metrics") or {}
    selected_beats = selected_sequence.get("beats") or []
    candidate_refs = [
        ref
        for candidate in candidates
        for ref in _unique_strings(candidate.get("memory_basis") or [])
    ]
    unique_candidate_refs = _unique_strings(candidate_refs)
    supported_candidates = [candidate for candidate in candidates if candidate.get("memory_basis")]
    rejected_sequences = selected_sequence.get("rejected_sequences") or []
    return {
        "character_count": len(state_graph.get("characters") or []),
        "pressure_graph_edge_count": len(pressure_graph.get("edges") or []),
        "candidate_action_count": len(candidates),
        "candidate_sequence_count": selected_sequence.get("candidate_sequence_count", 1),
        "selected_beat_count": len(selected_beats),
        "selected_sequence_score": selected_sequence.get("score", 0.0),
        "selected_sequence_score_margin": selected_sequence.get("score_margin", selected_sequence.get("score", 0.0)),
        "rejected_sequence_count": len(rejected_sequences),
        "scene_problem_coverage": 1.0 if selected_sequence.get("coverage", {}).get("scene_problem") else 0.0,
        "relationship_coverage": 1.0 if selected_sequence.get("coverage", {}).get("relationship_stance") else 0.0,
        "memory_support_rate": round(len(supported_candidates) / max(len(candidates), 1), 3),
        "candidate_memory_ref_count": len(candidate_refs),
        "unique_candidate_memory_ref_count": len(unique_candidate_refs),
        "hard_violation_count": verification_metrics.get("hard_violation_count", 0),
        "soft_warning_count": verification_metrics.get("soft_warning_count", 0),
        "awkward_phrase_risk_count": verification_metrics.get("therapy_phrase_risk_count", 0),
    }


def _build_writer_metrics(writer_packet: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    action_count = sum(
        len(beat.get("action_guidance") or [])
        for beat in writer_packet.get("use_as_optional_behavior") or []
        if isinstance(beat, dict)
    )
    posture_count = len(writer_packet.get("dialogue_posture") or [])
    avoid_count = len(writer_packet.get("avoid") or [])
    warning_types = {item.get("type") for item in verification.get("soft_warnings") or [] if isinstance(item, dict)}
    return {
        "writer_packet_action_guidance_count": action_count,
        "writer_packet_dialogue_posture_count": posture_count,
        "writer_packet_avoid_phrase_count": avoid_count,
        "raw_simulation_warning_type_count": len(warning_types),
    }


def _build_writer_packet(
    selected_sequence: dict[str, Any],
    candidates: list[dict[str, Any]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    avoid_phrases = []
    for warning in verification.get("soft_warnings") or []:
        if warning.get("type") == "therapy_phrase_risk" and warning.get("phrase"):
            avoid_phrases.append(warning["phrase"])
    candidate_map = {candidate.get("candidate_id"): candidate for candidate in candidates}
    behavior = []
    for beat in selected_sequence.get("beats") or []:
        action_guidance = []
        for action_id in beat.get("actions") or []:
            candidate = candidate_map.get(action_id)
            if not candidate:
                continue
            action_guidance.append(
                {
                    "actor": candidate.get("actor"),
                    "action_type": candidate.get("action_type"),
                    "surface_action": candidate.get("surface_action"),
                    "dialogue_intent": candidate.get("dialogue_intent"),
                    "not_final_dialogue": True,
                    "intent_basis": candidate.get("intent_basis") or [],
                    "memory_basis": _compact_refs(candidate.get("memory_basis") or [], limit=6),
                }
            )
        copied = dict(beat)
        copied["action_guidance"] = action_guidance
        behavior.append(copied)
    dialogue_posture = []
    for beat in behavior:
        function = beat.get("interaction_function") or ""
        dialogue_posture.append(
            {
                "beat_id": beat.get("beat_id"),
                "dialogue_function": function,
                "tone": _tone_bounds(function),
                "avoid_phrases": sorted(set(avoid_phrases)),
                "not_canonical_dialogue": True,
            }
        )
    return {
        "use_as_optional_behavior": behavior,
        "dialogue_posture": dialogue_posture,
        "avoid": sorted(set(avoid_phrases)),
    }


def _card_texts(card: dict[str, Any]) -> list[str]:
    texts = []
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
    ):
        texts.extend(_texts_from_value(card.get(key)))
    return texts


def _simulation_texts(simulation: dict[str, Any]) -> list[str]:
    texts = []
    for key in (
        "intent_assumptions",
        "likely_internal_state",
        "likely_actions",
        "likely_dialogue",
        "interaction_pressure",
        "avoid_or_risks",
        "memory_basis",
    ):
        texts.extend(_texts_from_value(simulation.get(key)))
    return texts


def _texts_from_value(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [text for child in value.values() for text in _texts_from_value(child)]
    if isinstance(value, list):
        return [text for item in value for text in _texts_from_value(item)]
    if isinstance(value, str):
        return [value]
    return []


def _text_from_item(item: dict[str, Any]) -> str:
    for key in ("value", "stance", "trait", "tendency", "constraint", "risk", "action", "point", "pressure"):
        if item.get(key):
            return str(item[key])
    return " ".join(str(value) for value in item.values() if isinstance(value, str))


def _values(items: Any, key: str) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, dict):
            value = item.get(key) or item.get("value") or item.get("trait") or item.get("tendency")
            if value:
                values.append(str(value))
        elif str(item).strip():
            values.append(str(item))
    return values


def _refs_from_item(item: dict[str, Any]) -> list[str]:
    refs = item.get("refs")
    if isinstance(refs, list):
        return [str(ref) for ref in refs if str(ref).strip()]
    return []


def _support_items_from_value(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        refs = _refs_from_item(value)
        text = " ".join(str(child).strip() for child in value.values() if isinstance(child, str) and child.strip())
        if refs and text:
            items.append({"text": text, "refs": refs})
        for child in value.values():
            items.extend(_support_items_from_value(child))
    elif isinstance(value, list):
        for item in value:
            items.extend(_support_items_from_value(item))
    return items


def _refs_from_value(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        raw_refs = value.get("refs")
        if isinstance(raw_refs, list):
            refs.update(str(ref) for ref in raw_refs if str(ref).strip())
        for child in value.values():
            refs.update(_refs_from_value(child))
    elif isinstance(value, list):
        for item in value:
            refs.update(_refs_from_value(item))
    return refs


def _rank_refs(refs: set[str], text: str) -> list[str]:
    def key(ref: str) -> tuple[int, int, str]:
        numeric = "".join(ch for ch in ref if ch.isdigit())
        number = int(numeric) if numeric else 9999
        mentioned = 0 if ref in text else 1
        family = 0 if ref.startswith("M") else 1
        return (mentioned, family, number, ref)

    return sorted((str(ref) for ref in refs if str(ref).strip()), key=key)


def _compact_refs(refs: list[Any], *, limit: int) -> list[str]:
    ordered = []
    for ref in refs:
        value = str(ref)
        if value and value not in ordered:
            ordered.append(value)
    memory_refs = [ref for ref in ordered if ref.startswith("M")]
    reference_refs = [ref for ref in ordered if ref.startswith("R")]
    compact = (memory_refs[: max(1, limit // 2)] + reference_refs[: max(1, limit - len(memory_refs[: max(1, limit // 2)]))])[:limit]
    return compact or ordered[:limit]


def _candidate_support_refs(character: dict[str, Any], action_type: str) -> list[str]:
    refs = list(character.get("memory_support_refs") or [])
    if not refs:
        return []
    support_terms = ACTION_SUPPORT_TERMS.get(action_type, ())
    preferred: list[str] = []
    for item in character.get("support_items") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if support_terms and not any(term in text for term in support_terms):
            continue
        preferred.extend(str(ref) for ref in item.get("refs") or [] if str(ref).strip())
    fallback = [ref for ref in refs if ref not in preferred]
    return _compact_refs(preferred + fallback, limit=6)


def _unique_strings(values: list[Any] | tuple[Any, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _infer_visible_action_state(writing_intent: str, combined: str) -> str:
    text = writing_intent + " " + combined
    if any(term in text for term in ("飞行", "驾驶舱", "返航", "J20")):
        return "处在飞行/返航相关场景中"
    if any(term in text for term in ("对话", "谈", "争执")):
        return "处在对话互动中"
    return "处在写作请求给出的新场景中"


def _infer_local_goal(card: dict[str, Any], simulation: dict[str, Any], writing_intent: str) -> str:
    values = _values(card.get("values_or_motivations"), "value")
    if values:
        return values[0]
    internal = _values(simulation.get("likely_internal_state"), "value")
    if internal:
        return internal[0]
    if "危险" in writing_intent or "紧张" in writing_intent:
        return "在压力中维持自身行为倾向"
    return "按当前关系和场景压力行动"


def _infer_setting_type(writing_intent: str) -> list[str]:
    return [term for term in SETTING_TERMS if term in writing_intent]


def _infer_interaction_problem(writing_intent: str) -> list[str]:
    problems = []
    if any(term in writing_intent for term in ("危险", "紧张", "压力", "告警", "警报")):
        problems.append("risk_or_pressure")
    if any(term in writing_intent for term in ("互动", "劝", "冲突", "张力", "关系")):
        problems.append("relationship_pressure")
    if not problems:
        problems.append("scene_participation")
    return problems


def _state_character_text(character: dict[str, Any]) -> str:
    return " ".join(_texts_from_value(character))


def _relation_to(source: dict[str, Any], target_name: Any) -> dict[str, Any]:
    for relation in source.get("relationship_stances") or []:
        if relation.get("target") == target_name:
            return relation
    return {}


def _infer_pressure_type(
    source_text: str,
    target_text: str,
    relation: dict[str, Any],
    writing_intent: str,
    *,
    source_name: Any = "",
) -> str:
    combined = source_text + " " + target_text + " " + str(relation.get("stance") or "") + " " + writing_intent
    if _relation_expresses_source_care(relation):
        return "care_guidance"
    if any(term in source_text for term in ("不美好", "地球", "抗拒", "不服", "冷硬", "向往", "想上去")):
        return "value_resistance"
    if any(term in combined for term in ("稳", "慢", "安全", "危险", "高度", "姿态", "告警", "警报")):
        if _relation_expresses_source_care(relation) or str(source_name) == "张鹏":
            return "care_guidance"
        return "safety_correction"
    if any(term in combined for term in RESISTANCE_TERMS):
        return "value_resistance"
    if relation:
        return "care_guidance"
    return "information_probe"


def _relation_expresses_source_care(relation: dict[str, Any]) -> bool:
    stance = str(relation.get("stance") or "")
    if not stance:
        return False
    if any(marker in stance for marker in ("受", "被", "接受")):
        return False
    return any(term in stance for term in CARE_TERMS)


def _desired_change(pressure_type: str) -> str:
    return {
        "safety_correction": "降低即时风险并稳住动作",
        "care_guidance": "用关切或经验压住对方的危险倾向",
        "value_resistance": "维持自身价值判断或回避对方劝说",
        "information_probe": "暴露对方真实状态或意图",
    }.get(pressure_type, "推动角色产生可写的回应")


def _intent_basis(writing_intent: str, action_or_pressure: str) -> list[str]:
    basis = []
    for term in SAFETY_TERMS + CARE_TERMS + RESISTANCE_TERMS + SETTING_TERMS:
        if term in writing_intent:
            basis.append(term)
    if not basis:
        basis.append(action_or_pressure)
    return sorted(set(basis))[:6]


def _tone_bounds(pressure_type: str) -> list[str]:
    if pressure_type in {"safety_correction", "risky_operation"}:
        return ["具体", "短促", "场景内动作优先"]
    if pressure_type == "care_guidance":
        return ["关切", "克制", "不心理诊断"]
    if pressure_type in {"value_resistance", "deflection"}:
        return ["压抑", "简短", "避免抽象说教"]
    return ["自然", "简洁", "不解释过度"]


def _candidate_action_types(character: dict[str, Any], edges: list[dict[str, Any]], writing_intent: str) -> list[str]:
    character_text = _state_character_text(character)
    text = character_text + " " + writing_intent
    action_types: list[str] = []
    outgoing_types = {str(edge.get("pressure_type") or "") for edge in edges if edge.get("source") == character.get("name")}
    incoming_types = {str(edge.get("pressure_type") or "") for edge in edges if edge.get("target") == character.get("name")}
    risk_actor = (
        "value_resistance" in outgoing_types
        or any(term in character_text for term in ("偏快", "冒险", "危险行为", "不服", "抗拒", "不美好", "地球", "向往", "想上去", "急"))
    )
    if risk_actor:
        action_types.append("risky_operation")
        action_types.append("physical_reaction")
    if any(edge.get("pressure_type") in {"safety_correction", "care_guidance"} and edge.get("source") == character.get("name") for edge in edges):
        action_types.append("safety_correction")
        action_types.append("care_reframe")
    if any(term in text for term in RESISTANCE_TERMS):
        action_types.append("value_resistance")
        action_types.append("deflection")
        action_types.append("minimal_compliance")
    if any(edge.get("source") == character.get("name") for edge in edges):
        action_types.append("information_probe")
    if "value_resistance" in incoming_types and not risk_actor:
        action_types.append("safety_correction")
        action_types.append("care_reframe")
        action_types.append("physical_reaction")
    if any(term in writing_intent for term in SAFETY_TERMS) and "physical_reaction" not in action_types:
        action_types.append("physical_reaction")
    action_types.append("silence_or_withholding")
    deduped = []
    for action_type in action_types:
        if action_type in ACTION_TYPES and action_type not in deduped:
            deduped.append(action_type)
    return deduped


def _intent_alignment(action_type: str, writing_intent: str) -> float:
    if action_type in {"risky_operation", "safety_correction", "physical_reaction"}:
        return _score_terms(writing_intent, SAFETY_TERMS, default=0.45)
    if action_type in {"value_resistance", "deflection", "minimal_compliance"}:
        return _score_terms(writing_intent, ("互动", "张力", "情绪", "抗拒", "劝"), default=0.45)
    if action_type == "care_reframe":
        return _score_terms(writing_intent, CARE_TERMS + ("互动", "张力"), default=0.45)
    return 0.45


def _motivation_score(action_type: str, character: dict[str, Any], edges: list[dict[str, Any]]) -> float:
    text = _state_character_text(character)
    if action_type in {"risky_operation", "value_resistance", "deflection"}:
        return _score_terms(text, RESISTANCE_TERMS + ("急", "危险", "快"), default=0.45)
    if action_type in {"safety_correction", "care_reframe"}:
        edge_score = max((float(edge.get("strength") or 0) for edge in edges if edge.get("source") == character.get("name")), default=0)
        return max(edge_score, _score_terms(text, CARE_TERMS, default=0.45))
    return _score_terms(text, RELATION_TERMS + SAFETY_TERMS, default=0.45)


def _interaction_productivity(action_type: str, edges: list[dict[str, Any]]) -> float:
    edge_strength = max((float(edge.get("strength") or 0) for edge in edges), default=0.35)
    if action_type in {"risky_operation", "safety_correction", "value_resistance", "care_reframe"}:
        return max(0.65, edge_strength)
    return max(0.35, edge_strength - 0.1)


def _candidate_risks(action_type: str, character: dict[str, Any]) -> list[dict[str, str]]:
    risks = []
    text = _state_character_text(character)
    detected = detect_text_risks(text)
    for phrase in detected["therapy_like_phrases"]:
        risks.append({"risk": f"avoid therapy-like phrase: {phrase}", "severity": "medium"})
    for term in detected["unsupported_formal_role_terms"]:
        risks.append({"risk": f"avoid unsupported formal role: {term}", "severity": "high"})
    if action_type in {"value_resistance", "deflection"}:
        risks.append({"risk": "avoid turning resistance into final psychological diagnosis", "severity": "medium"})
    return risks


def _surface_action(action_type: str, actor: str) -> str:
    return {
        "risky_operation": f"{actor}通过偏快或偏冒险的动作制造场景压力",
        "safety_correction": f"{actor}直接把对方的动作拉回安全边界",
        "minimal_compliance": f"{actor}只做最低限度的配合",
        "value_resistance": f"{actor}用简短反应保留自身价值抵抗",
        "deflection": f"{actor}回避对方劝说的核心",
        "care_reframe": f"{actor}把提醒落到生存、返航或当下动作上",
        "information_probe": f"{actor}用问题试探对方状态",
        "silence_or_withholding": f"{actor}用沉默或停顿保留未说出的压力",
        "physical_reaction": f"{actor}通过手、呼吸、视线或姿态表现压力",
        "environmental_pressure": "环境变化加重互动压力",
    }.get(action_type, f"{actor}产生一个可写的互动动作")


def _dialogue_intent(action_type: str, actor: str) -> str:
    return {
        "risky_operation": "不解释太多，用动作先制造风险",
        "safety_correction": "短促提醒对方稳住当下动作",
        "minimal_compliance": "承认提醒但不完全打开情绪",
        "value_resistance": "表达抵触，但避免抽象心理诊断",
        "deflection": "转开话题或压低回应",
        "care_reframe": "把关切压进具体行动要求",
        "information_probe": "问出对方真正卡住的点",
        "silence_or_withholding": "让停顿承担未说出的态度",
        "physical_reaction": "少说话，让身体反应承载压力",
    }.get(action_type, f"{actor}维持场景内互动")


def _preconditions(action_type: str) -> list[str]:
    if action_type in {"risky_operation", "safety_correction"}:
        return ["场景存在即时动作风险或压力"]
    if action_type in {"value_resistance", "deflection", "minimal_compliance"}:
        return ["对方施加劝说或纠正压力"]
    return ["角色同处一个可互动场景"]


def _effects(action_type: str) -> dict[str, float]:
    return {
        "risky_operation": {"scene_pressure": 0.25, "safety": -0.2, "relationship_tension": 0.15},
        "safety_correction": {"safety": 0.25, "relationship_tension": 0.1, "care_visible": 0.15},
        "minimal_compliance": {"safety": 0.1, "resistance": -0.05, "relationship_tension": -0.05},
        "value_resistance": {"resistance": 0.2, "relationship_tension": 0.15},
        "deflection": {"resistance": 0.1, "information_clarity": -0.1},
        "care_reframe": {"care_visible": 0.25, "relationship_tension": -0.05},
        "information_probe": {"information_clarity": 0.15, "relationship_tension": 0.05},
        "silence_or_withholding": {"information_clarity": -0.1, "scene_pressure": 0.1},
        "physical_reaction": {"scene_pressure": 0.1, "character_pressure_visible": 0.15},
    }.get(action_type, {"scene_pressure": 0.05})


def _merge_effects(candidates: list[dict[str, Any] | None]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for candidate in candidates:
        if not candidate:
            continue
        for key, value in (candidate.get("effects") or {}).items():
            merged[key] = round(merged.get(key, 0.0) + float(value), 3)
    return merged


def _style_risks_for_actions(candidates: list[dict[str, Any] | None]) -> list[str]:
    risks = []
    for candidate in candidates:
        if not candidate:
            continue
        for risk in candidate.get("risks") or []:
            if isinstance(risk, dict) and risk.get("risk"):
                risks.append(str(risk["risk"]))
    return risks


def _coverage_for_beats(beats: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "scene_problem": _coverage_has_problem(beats),
        "pressure_response": any(len(beat.get("participants") or []) >= 2 for beat in beats),
        "relationship_stance": any(
            beat.get("interaction_function") in {"care_guidance", "safety_correction"} for beat in beats
        ),
    }


def _sequence_score_payload(selected: list[dict[str, Any]], verification: dict[str, Any]) -> dict[str, Any]:
    if not selected:
        return {
            "score": 0.0,
            "components": {
                "coverage": 0.0,
                "arc": 0.0,
                "faithfulness": 0.0,
                "base": 0.0,
                "warning_penalty": 0.0,
                "style_risk_penalty": 0.0,
                "support_bonus": 0.0,
            },
        }
    coverage = 0.4 if _coverage_has_problem(selected) else 0.0
    arc = 0.3 if any(len(beat.get("participants") or []) >= 2 for beat in selected) else 0.15
    faithfulness = 0.2 if not verification.get("hard_violations") else 0.0
    warning_penalty = min(0.2, 0.03 * len(verification.get("soft_warnings") or []))
    style_risk_count = sum(
        len((beat.get("verification") or {}).get("style_risks") or [])
        for beat in selected
        if isinstance(beat, dict)
    )
    style_risk_penalty = min(0.15, 0.04 * style_risk_count)
    support_bonus = 0.05 if all((beat.get("verification") or {}).get("memory_supported") for beat in selected) else 0.0
    base = 0.1
    score = round(
        max(0.0, min(1.0, coverage + arc + faithfulness + base + support_bonus - warning_penalty - style_risk_penalty)),
        3,
    )
    return {
        "score": score,
        "components": {
            "coverage": coverage,
            "arc": arc,
            "faithfulness": faithfulness,
            "base": base,
            "warning_penalty": warning_penalty,
            "style_risk_penalty": style_risk_penalty,
            "support_bonus": support_bonus,
        },
    }


def _sequence_priority_sum(option: dict[str, Any]) -> float:
    return sum(float(beat.get("priority") or 0) for beat in option.get("beats") or [] if isinstance(beat, dict))


def _rejected_sequence_summary(option: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_id": option.get("sequence_id"),
        "strategy": option.get("strategy"),
        "score": option.get("score", 0.0),
        "score_margin": round(float(selected.get("score") or 0.0) - float(option.get("score") or 0.0), 3),
        "beat_count": len(option.get("beats") or []),
        "coverage": option.get("coverage") or {},
        "rejection_reason": _sequence_rejection_reason(option, selected),
    }


def _sequence_rejection_reason(option: dict[str, Any], selected: dict[str, Any]) -> str:
    if float(option.get("score") or 0.0) < float(selected.get("score") or 0.0):
        return "lower_sequence_score"
    if not (option.get("coverage") or {}).get("relationship_stance"):
        return "weaker_relationship_coverage"
    if len(option.get("beats") or []) < len(selected.get("beats") or []):
        return "less_interaction_development"
    return "duplicate_or_lower_priority_sequence"


def _coverage_has_problem(selected: list[dict[str, Any]]) -> bool:
    problem_types = {"risky_operation", "safety_correction", "care_guidance", "value_resistance"}
    return any(beat.get("interaction_function") in problem_types for beat in selected)


def _support_score(refs: list[Any]) -> float:
    count = len([ref for ref in refs if str(ref).strip()])
    if count <= 0:
        return 0.2
    return min(1.0, 0.45 + 0.15 * math.log2(count + 1))


def _score_terms(text: str, terms: tuple[str, ...], *, default: float = 0.35) -> float:
    raw = str(text or "")
    if not raw:
        return default
    hits = sum(1 for term in terms if term and term in raw)
    if hits <= 0:
        return default
    return min(1.0, default + 0.12 * hits)
