from dms.timeline import TimelineBuildConfig, build_timeline_graph, format_timeline_report, normalize_temporal_scene_output


def test_timeline_graph_keeps_reveal_time_separate_from_story_time() -> None:
    scene_1 = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0001",
            "temporal_events": [
                {
                    "event_id": "scene_0001:event_001",
                    "summary": "角色抵达基地",
                    "participants": ["角色A"],
                    "event_time_mode": "present_scene",
                    "story_time_hint": "current scene",
                    "granularity": "scene_relative",
                    "evidence": "抵达基地",
                    "confidence": 0.9,
                    "revealed_at_scene_id": "scene_0001",
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {"dominant_time_mode": "present_scene"},
            "temporal_warnings": [],
        },
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
    )
    scene_3 = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0003",
            "temporal_events": [
                {
                    "event_id": "scene_0003:event_001",
                    "summary": "角色回忆三年前的事故",
                    "participants": ["角色A"],
                    "event_time_mode": "past_recalled",
                    "story_time_hint": "三年前",
                    "granularity": "year",
                    "evidence": "三年前的事故",
                    "confidence": 0.95,
                    "revealed_at_scene_id": "scene_0003",
                }
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0003:event_001",
                    "target_event_id": "scene_0001:event_001",
                    "relation_type": "before",
                    "evidence": "三年前",
                    "confidence": 0.95,
                    "is_inferred": False,
                }
            ],
            "scene_temporal_index": {
                "dominant_time_mode": "past_recalled",
                "contains_flashback_or_recalled_past": True,
            },
            "temporal_warnings": [],
        },
        scene_id="scene_0003",
        source_record_id=3,
        discourse_index=3,
    )

    graph = build_timeline_graph([scene_1, scene_3], config=TimelineBuildConfig(use_discourse_order_prior=False))

    assert graph["timeline_order"][0]["event_id"] == "scene_0003:event_001"
    assert graph["timeline_order"][0]["revealed_at_scene_id"] == "scene_0003"
    assert graph["events"][1]["is_reveal_of_past"] is True
    assert "角色回忆三年前的事故" in format_timeline_report(graph)


def test_timeline_graph_reports_hard_ordering_conflicts() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0001",
            "temporal_events": [
                {
                    "event_id": "scene_0001:event_001",
                    "summary": "A",
                    "participants": [],
                    "event_time_mode": "present_scene",
                    "evidence": "A",
                    "confidence": 1,
                },
                {
                    "event_id": "scene_0001:event_002",
                    "summary": "B",
                    "participants": [],
                    "event_time_mode": "present_scene",
                    "evidence": "B",
                    "confidence": 1,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0001:event_001",
                    "target_event_id": "scene_0001:event_002",
                    "relation_type": "before",
                    "evidence": "A before B",
                    "confidence": 0.95,
                },
                {
                    "source_event_id": "scene_0001:event_002",
                    "target_event_id": "scene_0001:event_001",
                    "relation_type": "before",
                    "evidence": "B before A",
                    "confidence": 0.95,
                },
            ],
            "scene_temporal_index": {},
            "temporal_warnings": [],
        },
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
    )

    graph = build_timeline_graph([scene])

    assert graph["counts"]["conflict_count"] >= 1
    assert any(conflict["type"] in {"two_node_cycle", "cycle"} for conflict in graph["conflicts"])


def test_timeline_graph_separates_context_and_forecast_tracks() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0008",
            "temporal_events": [
                {
                    "event_id": "scene_0008:event_001",
                    "summary": "太阳活动加剧",
                    "participants": [],
                    "event_track": "context",
                    "event_time_mode": "present_scene",
                    "evidence": "太阳，急速老化",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0008:event_002",
                    "summary": "一百年后太阳吞没地球",
                    "participants": [],
                    "event_track": "forecast",
                    "event_time_mode": "future_anticipated",
                    "evidence": "一百年后，太阳将膨胀到吞没整个地球",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0008:event_001",
                    "target_event_id": "scene_0008:event_002",
                    "relation_type": "anticipates",
                    "evidence": "一百年后",
                    "confidence": 1.0,
                }
            ],
            "scene_temporal_index": {"scene_temporal_role": "montage"},
            "temporal_warnings": [],
        },
        scene_id="scene_0008",
        source_record_id=8,
        discourse_index=8,
    )

    graph = build_timeline_graph([scene])

    assert graph["counts"]["main_story_event_count"] == 0
    assert graph["counts"]["context_event_count"] == 1
    assert graph["counts"]["forecast_event_count"] == 1
    assert [item["event_id"] for item in graph["context_timeline_order"]] == ["scene_0008:event_001"]
    assert [item["event_id"] for item in graph["forecast_timeline_order"]] == ["scene_0008:event_002"]
    assert graph["main_timeline_order"] == []


def test_symmetric_relation_is_not_counted_as_skipped_ordering_edge() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0001",
            "temporal_events": [
                {
                    "event_id": "scene_0001:event_001",
                    "summary": "A",
                    "participants": [],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "A",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0001:event_002",
                    "summary": "B",
                    "participants": [],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "B",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0001:event_001",
                    "target_event_id": "scene_0001:event_002",
                    "relation_type": "overlaps",
                    "evidence": "A and B overlap",
                    "confidence": 0.9,
                }
            ],
            "scene_temporal_index": {},
            "temporal_warnings": [],
        },
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
    )

    graph = build_timeline_graph([scene], config=TimelineBuildConfig(use_discourse_order_prior=False))

    assert graph["counts"]["selected_ordering_relation_count"] == 0
    assert graph["counts"]["skipped_ordering_relation_count"] == 0
    assert graph["timeline_buckets"][0]["event_ids"] == ["scene_0001:event_001", "scene_0001:event_002"]


def test_scene_block_prior_keeps_scene_events_before_next_scene_representative() -> None:
    scene_1 = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0001",
            "temporal_events": [
                {
                    "event_id": "scene_0001:event_001",
                    "summary": "第一场开始",
                    "participants": [],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "开始",
                    "confidence": 1,
                },
                {
                    "event_id": "scene_0001:event_002",
                    "summary": "第一场继续",
                    "participants": [],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "继续",
                    "confidence": 1,
                },
            ],
            "temporal_relations": [],
            "scene_temporal_index": {},
            "temporal_warnings": [],
        },
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
    )
    scene_2 = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0002",
            "temporal_events": [
                {
                    "event_id": "scene_0002:event_001",
                    "summary": "第二场开始",
                    "participants": [],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "第二场",
                    "confidence": 1,
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {},
            "temporal_warnings": [],
        },
        scene_id="scene_0002",
        source_record_id=2,
        discourse_index=2,
    )

    graph = build_timeline_graph([scene_1, scene_2])

    order = [item["event_id"] for item in graph["main_timeline_order"]]
    assert order == ["scene_0001:event_001", "scene_0001:event_002", "scene_0002:event_001"]
    assert any(str(item.get("relation_id", "")).startswith("scene_block_prior:") for item in graph["relations"])


def test_high_confidence_relation_overrides_scene_block_prior_cycle() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0004",
            "temporal_events": [
                {
                    "event_id": "scene_0004:event_001",
                    "summary": "角色抵达废墟",
                    "participants": ["角色"],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "抵达废墟",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0004:event_002",
                    "summary": "角色回忆废墟之前是夜市",
                    "participants": ["角色"],
                    "event_track": "memory",
                    "event_time_mode": "past_recalled",
                    "evidence": "这儿之前，都不带枪",
                    "confidence": 0.95,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0004:event_002",
                    "target_event_id": "scene_0004:event_001",
                    "relation_type": "before",
                    "evidence": "这儿之前，都不带枪",
                    "confidence": 0.95,
                }
            ],
            "scene_temporal_index": {"contains_flashback_or_recalled_past": True},
            "temporal_warnings": [],
        },
        scene_id="scene_0004",
        source_record_id=4,
        discourse_index=4,
    )

    graph = build_timeline_graph([scene])

    order = [item["event_id"] for item in graph["main_timeline_order"]]
    assert order == ["scene_0004:event_002", "scene_0004:event_001"]
    llm_relation = next(item for item in graph["relations"] if item["source"] == "llm")
    prior_relation = next(item for item in graph["relations"] if item["source"] == "scene_block_prior")
    assert llm_relation["selected_for_ordering"] is True
    assert prior_relation["selected_for_ordering"] is False
    assert prior_relation["ordering_skip_reason"] == "would_create_cycle"


def test_visual_montage_without_participants_defaults_to_context_track() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0009",
            "temporal_events": [
                {
                    "event_id": "scene_0009:event_001",
                    "summary": "冰川融化",
                    "participants": [],
                    "event_time_mode": "habitual",
                    "evidence": "8-2、冰川融化。",
                    "confidence": 0.8,
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {"scene_temporal_role": "montage"},
            "temporal_warnings": [],
        },
        scene_id="scene_0009",
        source_record_id=9,
        discourse_index=9,
    )

    graph = build_timeline_graph([scene])

    assert scene["temporal_events"][0]["event_track"] == "context"
    assert graph["main_timeline_order"] == []
    assert [item["event_id"] for item in graph["context_timeline_order"]] == ["scene_0009:event_001"]


def test_environment_visual_with_non_character_participant_defaults_to_context_track() -> None:
    scene = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0008",
            "temporal_events": [
                {
                    "event_id": "scene_0008:event_001",
                    "summary": "太阳急速老化并持续膨胀",
                    "participants": ["太阳"],
                    "event_time_mode": "present_scene",
                    "evidence": "太阳，急速老化，持续膨胀",
                    "confidence": 0.9,
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {"scene_temporal_role": "montage"},
            "temporal_warnings": [],
        },
        scene_id="scene_0008",
        source_record_id=8,
        discourse_index=8,
    )

    graph = build_timeline_graph([scene])

    assert scene["temporal_events"][0]["event_track"] == "context"
    assert graph["main_timeline_order"] == []
