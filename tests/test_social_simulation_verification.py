from __future__ import annotations

from dms.simulation import verify_social_simulation, verify_writer_packet
from dms.simulation.algorithmic import build_algorithmic_social_plan


def test_verify_social_simulation_flags_therapy_like_phrase() -> None:
    verification = verify_social_simulation(
        cards=_cards(),
        character_simulations=[
            {
                "character": "张鹏",
                "likely_dialogue": [{"value": "张鹏：别跟地球赌气，先稳住。", "refs": ["M1"]}],
            }
        ],
        social_simulation={
            "scene_beats": [
                {
                    "beat": "张鹏用别跟地球赌气的说法劝刘培强。",
                    "participants": ["张鹏", "刘培强"],
                    "memory_basis": ["M1"],
                }
            ],
            "writer_guidance": [{"guidance": "可以写：张鹏：别跟地球赌气。", "refs": ["M1"]}],
        },
        writing_intent="两人在驾驶舱内面对危险飞行行为展开互动。",
    )

    assert verification["status"] == "warn"
    assert verification["metrics"]["therapy_phrase_risk_count"] >= 1
    assert verification["metrics"]["final_dialogue_like_guidance_count"] >= 1


def test_verify_social_simulation_does_not_flag_negated_formal_role_cautions() -> None:
    verification = verify_social_simulation(
        cards=_cards(),
        character_simulations=[
            {
                "character": "刘培强",
                "intent_assumptions": ["不要反推刘培强是正式航天员或正式飞行员。"],
                "likely_dialogue": [{"value": "短促应答型：知道了，慢一点。", "refs": ["M1"]}],
            }
        ],
        social_simulation={"scene_beats": [{"beat": "张鹏提醒刘培强稳住。", "memory_basis": ["M1"]}]},
        writing_intent="两人在驾驶舱内面对危险飞行行为展开互动。",
    )

    assert verification["metrics"]["unsupported_role_risk_count"] == 0
    assert verification["metrics"]["final_dialogue_like_guidance_count"] == 0


def test_verify_social_simulation_flags_unknown_refs_as_hard_violation() -> None:
    verification = verify_social_simulation(
        cards=_cards(),
        character_simulations=[],
        social_simulation={"scene_beats": [{"beat": "无证据动作", "memory_basis": ["M99"]}]},
        writing_intent="测试",
    )

    assert verification["status"] == "fail"
    assert verification["metrics"]["unknown_ref_count"] == 1
    assert verification["hard_violations"][0]["type"] == "unknown_memory_ref"


def test_algorithmic_social_plan_builds_pressure_and_candidate_artifacts() -> None:
    social_simulation = {
        "scene_beats": [
            {
                "beat": "刘培强操作偏快，张鹏提醒他稳住。",
                "participants": ["刘培强", "张鹏"],
                "memory_basis": ["M1"],
                "intent_basis": ["危险飞行"],
            }
        ]
    }
    verification = verify_social_simulation(
        cards=_cards(),
        character_simulations=_character_simulations(),
        social_simulation=social_simulation,
        writing_intent="两名飞行员在返航途中面对危险飞行行为展开互动。",
    )
    plan = build_algorithmic_social_plan(
        cards=_cards(),
        character_simulations=_character_simulations(),
        social_simulation=social_simulation,
        writing_intent="两名飞行员在返航途中面对危险飞行行为展开互动。",
        verification=verification,
    )

    assert plan["version"] == "asip_v0"
    assert plan["state_graph"]["characters"]
    assert plan["pressure_graph"]["edges"]
    assert plan["candidate_actions"]
    assert plan["selected_sequence"]["beats"]
    assert plan["selected_sequence"]["selection_strategy"] == "multi_candidate_rerank_v1"
    assert plan["selected_sequence"]["candidate_sequence_count"] >= 1
    assert "score_components" in plan["selected_sequence"]
    assert "rejected_sequences" in plan["selected_sequence"]
    assert plan["writer_packet"]["dialogue_posture"]
    first_beat = plan["writer_packet"]["use_as_optional_behavior"][0]
    assert first_beat["action_guidance"]
    assert first_beat["action_guidance"][0]["not_final_dialogue"] is True
    assert len(first_beat["action_guidance"][0]["memory_basis"]) <= 6
    assert plan["metrics"]["pressure_graph_edge_count"] >= 1
    assert plan["metrics"]["candidate_sequence_count"] >= 1
    assert plan["metrics"]["unique_candidate_memory_ref_count"] >= 1
    assert plan["metrics"]["writer_packet_action_guidance_count"] >= 1


def test_verify_writer_packet_is_separate_from_raw_simulation_warnings() -> None:
    packet = {
        "use_as_optional_behavior": [
            {
                "beat_id": "b_001",
                "action_guidance": [
                    {
                        "actor": "张鹏",
                        "action_type": "safety_correction",
                        "surface_action": "张鹏把动作拉回安全边界",
                        "dialogue_intent": "短促提醒对方稳住当下动作",
                        "not_final_dialogue": True,
                    }
                ],
            }
        ],
        "dialogue_posture": [{"beat_id": "b_001", "dialogue_function": "safety_correction", "tone": ["具体"]}],
        "avoid": [],
    }

    verification = verify_writer_packet(packet)

    assert verification["status"] == "pass"
    assert verification["metrics"]["final_dialogue_like_count"] == 0


def test_verify_writer_packet_flags_canonical_dialogue() -> None:
    packet = {
        "use_as_optional_behavior": [
            {
                "beat_id": "b_001",
                "action_guidance": [
                    {
                        "actor": "张鹏",
                        "action_type": "safety_correction",
                        "surface_action": "张鹏：别跟地球赌气。",
                        "dialogue_intent": "直接说出最终台词",
                        "not_final_dialogue": True,
                    }
                ],
            }
        ]
    }

    verification = verify_writer_packet(packet)

    assert verification["status"] == "warn"
    assert verification["metrics"]["therapy_phrase_risk_count"] >= 1
    assert verification["metrics"]["final_dialogue_like_count"] >= 1


def _cards() -> list[dict]:
    return [
        {
            "entity_id": "character_liu",
            "canonical_name": "刘培强",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "current_state": [{"value": "准备返航，情绪紧张", "status": "explicit", "refs": ["M1"]}],
            "values_or_motivations": [{"value": "认为地球不美好，向往天上", "status": "inferred", "refs": ["M1"]}],
            "relationship_stances": [{"target": "张鹏", "stance": "受张鹏提醒", "status": "inferred", "refs": ["M1"]}],
            "hard_constraints": [{"constraint": "不能知道未来信息", "refs": ["M1"]}],
            "simulation_risks": [],
        },
        {
            "entity_id": "character_zhang",
            "canonical_name": "张鹏",
            "entity_type": "character",
            "prefix_boundary": "before scene_0006",
            "current_state": [{"value": "与刘培强同行", "status": "explicit", "refs": ["M1"]}],
            "relationship_stances": [{"target": "刘培强", "stance": "提醒与照看", "status": "inferred", "refs": ["M1"]}],
            "hard_constraints": [],
            "simulation_risks": [],
        },
    ]


def _character_simulations() -> list[dict]:
    return [
        {
            "character": "刘培强",
            "likely_internal_state": [{"value": "返航时抗拒劝说", "refs": ["M1"]}],
            "likely_actions": [{"value": "操作偏快", "refs": ["M1"]}],
            "interaction_pressure": [{"target": "张鹏", "pressure": "用沉默和操作形成压力", "refs": ["M1"]}],
            "avoid_or_risks": [],
        },
        {
            "character": "张鹏",
            "likely_internal_state": [{"value": "关切并要求稳住", "refs": ["M1"]}],
            "likely_actions": [{"value": "提醒刘培强放慢", "refs": ["M1"]}],
            "interaction_pressure": [{"target": "刘培强", "pressure": "安全提醒", "refs": ["M1"]}],
            "avoid_or_risks": [],
        },
    ]
