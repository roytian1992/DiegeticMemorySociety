from dms.timeline import (
    TimelineBuildConfig,
    apply_temporal_audit_annotations,
    build_timeline_graph,
    normalize_temporal_scene_output,
    verify_temporal_outputs,
)
from dms.timeline.verification import format_temporal_audit_report


def test_temporal_verifier_aligns_subtitle_evidence_and_flags_bad_evidence() -> None:
    source_unit = {
        "unit_id": "scene_0009",
        "title": "9、EXT.日.全球灾难蒙太奇",
        "subtitle": "8-2、冰川融化。",
        "content": "科学家质问：你们知道这意味着什么吗？",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0009",
            "temporal_events": [
                {
                    "event_id": "scene_0009:event_001",
                    "summary": "冰川融化",
                    "participants": [],
                    "event_track": "context",
                    "event_time_mode": "habitual",
                    "evidence": "8-2、冰川融化。",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0009:event_002",
                    "summary": "Glaciers melt across the world",
                    "participants": ["Global glaciers"],
                    "location": "Earth",
                    "event_track": "context",
                    "event_time_mode": "habitual",
                    "evidence": "冰川融化...",
                    "confidence": 0.7,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0009:event_001",
                    "target_event_id": "scene_0009:event_002",
                    "relation_type": "before",
                    "evidence": "科学家质问暗示冰川融化发生在海啸之前",
                    "confidence": 0.95,
                }
            ],
            "scene_temporal_index": {
                "dominant_time_mode": "habitual",
                "scene_temporal_role": "montage",
                "evidence": "8-2、冰川融化。",
            },
            "temporal_warnings": [],
        },
        scene_id="scene_0009",
        source_record_id=9,
        discourse_index=9,
    )

    audit = verify_temporal_outputs([output], {"scene_0009": source_unit})

    counts = audit["counts"]
    assert counts["evidence_aligned_count"] == 2
    assert counts["evidence_rejected_count"] == 2
    assert counts["elliptical_evidence_count"] == 1
    assert counts["paraphrase_evidence_count"] == 1
    assert counts["non_chinese_output_count"] >= 2
    assert counts["hard_ordering_unusable_count"] == 1
    assert audit["annotations"]["events"]["scene_0009:event_001"]["evidence_source_field"] == "subtitle"
    assert audit["annotations"]["relations"]["scene_0009:rel_001"]["audit_ordering_usable"] is False
    assert "hard_ordering_unusable" in format_temporal_audit_report(audit)


def test_temporal_verifier_suggests_source_spans_for_ellipsis_evidence() -> None:
    source_unit = {
        "unit_id": "scene_0005",
        "title": "5、INT.日.J20C",
        "subtitle": "",
        "content": "刘培强熟练地操作各类启航按键。机舱盖合上。J20C 从战区废墟中垂直升起，驶离。",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0005",
            "temporal_events": [
                {
                    "event_id": "scene_0005:event_001",
                    "summary": "刘培强启动 J20C",
                    "participants": ["刘培强", "J20C"],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "刘培强熟练地操作各类启航按键。机舱盖合上...J20C 从战区废墟中垂直升起，驶离。",
                    "confidence": 0.9,
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {"evidence": "刘培强熟练地操作各类启航按键"},
            "temporal_warnings": [],
        },
        scene_id="scene_0005",
        source_record_id=5,
        discourse_index=5,
    )

    audit = verify_temporal_outputs([output], {"scene_0005": source_unit})
    annotation = audit["annotations"]["events"]["scene_0005:event_001"]

    assert annotation["audit_evidence_aligned"] is True
    assert annotation["gap_filled_evidence"] is True
    assert len(annotation["suggested_evidence_spans"]) == 2
    assert annotation["suggested_evidence_spans"][0]["evidence_aligned_text"] == "刘培强熟练地操作各类启航按键。机舱盖合上"
    assert annotation["evidence_aligned_text"] == source_unit["content"]
    assert "suggested spans:" in format_temporal_audit_report(audit)


def test_temporal_verifier_gap_fills_short_multispan_relation_evidence() -> None:
    source_unit = {
        "unit_id": "scene_0006",
        "title": "6、EXT.日.非洲中部",
        "subtitle": "",
        "content": (
            "J20C 从战区开始返航，飞越非洲中部的热带雨林和小型城镇。"
            "飞过的区域，时不时有爆炸、枪声，硝烟四起。\n"
            "张鹏：能活着就挺美好的了，这傻孩子，我跟你说啊，扛住。"
        ),
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0006",
            "temporal_events": [
                {
                    "event_id": "scene_0006:event_001",
                    "summary": "J20C返航",
                    "participants": ["J20C"],
                    "evidence": "J20C 从战区开始返航，飞越非洲中部的热带雨林和小型城镇",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0006:event_002",
                    "summary": "张鹏鼓励刘培强",
                    "participants": ["张鹏", "刘培强"],
                    "evidence": "张鹏：能活着就挺美好的了，这傻孩子，我跟你说啊，扛住。",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0006:event_001",
                    "target_event_id": "scene_0006:event_002",
                    "relation_type": "before",
                    "evidence": (
                        "J20C 从战区开始返航，飞越非洲中部的热带雨林和小型城镇。 "
                        "张鹏：能活着就挺美好的了，这傻孩子，我跟你说啊，扛住。"
                    ),
                    "confidence": 0.8,
                }
            ],
            "scene_temporal_index": {"evidence": "J20C 从战区开始返航"},
            "temporal_warnings": [],
        },
        scene_id="scene_0006",
        source_record_id=6,
        discourse_index=6,
    )

    audit = verify_temporal_outputs([output], {"scene_0006": source_unit})
    apply_temporal_audit_annotations([output], audit)
    graph = build_timeline_graph([output], config=TimelineBuildConfig(use_discourse_order_prior=False))
    relation = next(item for item in graph["relations"] if item["source"] == "llm")
    annotation = audit["annotations"]["relations"]["scene_0006:rel_001"]

    assert annotation["audit_evidence_aligned"] is True
    assert annotation["gap_filled_evidence"] is True
    assert annotation["gap_filled_evidence_gap_non_ws_char_count"] == 21
    assert "飞过的区域，时不时有爆炸、枪声，硝烟四起。" in annotation["evidence_aligned_text"]
    assert "gap_filled_evidence" in annotation["evidence_issues"]
    assert audit["counts"]["gap_filled_evidence_count"] == 1
    assert audit["counts"]["hard_ordering_unusable_count"] == 0
    assert relation["usable_for_ordering"] is True
    assert "gap fill:" in format_temporal_audit_report(audit)


def test_temporal_verifier_rejects_long_gap_multispan_relation_evidence() -> None:
    long_gap = "这是很长的间隔文本，超过四十个非空白字符，所以不能自动补全为可靠的连续证据。"
    source_unit = {
        "unit_id": "scene_0006",
        "title": "6、EXT.日.非洲中部",
        "subtitle": "",
        "content": f"甲先行动。{long_gap}乙随后行动。",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0006",
            "temporal_events": [
                {
                    "event_id": "scene_0006:event_001",
                    "summary": "甲行动",
                    "participants": ["甲"],
                    "evidence": "甲先行动",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0006:event_002",
                    "summary": "乙行动",
                    "participants": ["乙"],
                    "evidence": "乙随后行动",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0006:event_001",
                    "target_event_id": "scene_0006:event_002",
                    "relation_type": "before",
                    "evidence": "甲先行动。乙随后行动。",
                    "confidence": 0.8,
                }
            ],
            "scene_temporal_index": {"evidence": "甲先行动"},
            "temporal_warnings": [],
        },
        scene_id="scene_0006",
        source_record_id=6,
        discourse_index=6,
    )

    audit = verify_temporal_outputs([output], {"scene_0006": source_unit})
    apply_temporal_audit_annotations([output], audit)
    graph = build_timeline_graph([output], config=TimelineBuildConfig(use_discourse_order_prior=False))
    relation = next(item for item in graph["relations"] if item["source"] == "llm")
    annotation = audit["annotations"]["relations"]["scene_0006:rel_001"]

    assert annotation["audit_evidence_aligned"] is False
    assert annotation.get("gap_filled_evidence") is not True
    assert audit["counts"]["gap_filled_evidence_count"] == 0
    assert audit["counts"]["hard_ordering_unusable_count"] == 1
    assert relation["usable_for_ordering"] is False
    assert relation["ordering_error"] == "audit_unusable_evidence"


def test_temporal_verifier_rejects_reversed_multispan_relation_evidence() -> None:
    source_unit = {
        "unit_id": "scene_0006",
        "title": "6、EXT.日.非洲中部",
        "subtitle": "",
        "content": "甲先行动。短暂间隔。乙随后行动。",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0006",
            "temporal_events": [
                {
                    "event_id": "scene_0006:event_001",
                    "summary": "甲行动",
                    "participants": ["甲"],
                    "evidence": "甲先行动",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0006:event_002",
                    "summary": "乙行动",
                    "participants": ["乙"],
                    "evidence": "乙随后行动",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0006:event_001",
                    "target_event_id": "scene_0006:event_002",
                    "relation_type": "before",
                    "evidence": "乙随后行动。甲先行动。",
                    "confidence": 0.8,
                }
            ],
            "scene_temporal_index": {"evidence": "甲先行动"},
            "temporal_warnings": [],
        },
        scene_id="scene_0006",
        source_record_id=6,
        discourse_index=6,
    )

    audit = verify_temporal_outputs([output], {"scene_0006": source_unit})
    apply_temporal_audit_annotations([output], audit)
    graph = build_timeline_graph([output], config=TimelineBuildConfig(use_discourse_order_prior=False))
    relation = next(item for item in graph["relations"] if item["source"] == "llm")
    annotation = audit["annotations"]["relations"]["scene_0006:rel_001"]

    assert annotation["audit_evidence_aligned"] is False
    assert annotation.get("gap_filled_evidence") is not True
    assert audit["counts"]["gap_filled_evidence_count"] == 0
    assert relation["usable_for_ordering"] is False


def test_temporal_verifier_audits_scene_time_hints() -> None:
    source_unit = {
        "unit_id": "scene_0004",
        "title": "4、EXT.日.利伯维尔 战区废墟",
        "subtitle": "",
        "content": "字卡：距太阳氦闪还剩 34 年。张鹏说：之前这里夜市老好了。",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0004",
            "temporal_events": [
                {
                    "event_id": "scene_0004:event_001",
                    "summary": "刘培强询问太阳危机",
                    "participants": ["刘培强"],
                    "event_track": "forecast",
                    "event_time_mode": "future_anticipated",
                    "evidence": "字卡：距太阳氦闪还剩 34 年",
                    "confidence": 0.9,
                }
            ],
            "temporal_relations": [],
            "scene_temporal_index": {
                "dominant_time_mode": "present_scene",
                "absolute_time_hints": ["距太阳氦闪还剩 34 年"],
                "relative_time_hints": ["之前...夜市老好了"],
                "evidence": "字卡：距太阳氦闪还剩 34 年",
            },
            "temporal_warnings": [],
        },
        scene_id="scene_0004",
        source_record_id=4,
        discourse_index=4,
    )

    audit = verify_temporal_outputs([output], {"scene_0004": source_unit})
    apply_temporal_audit_annotations([output], audit)

    assert audit["counts"]["time_hint_count"] == 2
    assert audit["counts"]["time_hint_issue_count"] == 1
    assert audit["annotations"]["scene_temporal_hints"]["scene_0004:absolute_time_hints[1]"]["audit_evidence_aligned"] is True
    relative_hint = audit["annotations"]["scene_temporal_hints"]["scene_0004:relative_time_hints[1]"]
    assert relative_hint["audit_evidence_aligned"] is True
    assert relative_hint["gap_filled_evidence"] is True
    assert "evidence_not_found" not in relative_hint["evidence_issues"]
    assert "elliptical_evidence" in relative_hint["evidence_issues"]
    assert "relative_time_hints[1]" in output["scene_temporal_index"]["time_hint_audit"]


def test_audited_relation_with_rejected_evidence_does_not_drive_timeline_ordering() -> None:
    source_unit = {
        "unit_id": "scene_0001",
        "title": "1、INT.日.房间",
        "subtitle": "",
        "content": "甲先进入房间。乙随后进入房间。",
    }
    output = normalize_temporal_scene_output(
        {
            "unit_id": "scene_0001",
            "temporal_events": [
                {
                    "event_id": "scene_0001:event_001",
                    "summary": "乙进入房间",
                    "participants": ["乙"],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "乙随后进入房间",
                    "confidence": 0.9,
                },
                {
                    "event_id": "scene_0001:event_002",
                    "summary": "甲进入房间",
                    "participants": ["甲"],
                    "event_track": "plot",
                    "event_time_mode": "present_scene",
                    "evidence": "甲先进入房间",
                    "confidence": 0.9,
                },
            ],
            "temporal_relations": [
                {
                    "source_event_id": "scene_0001:event_002",
                    "target_event_id": "scene_0001:event_001",
                    "relation_type": "before",
                    "evidence": "甲进入明显早于乙进入",
                    "confidence": 0.95,
                }
            ],
            "scene_temporal_index": {"evidence": "甲先进入房间"},
            "temporal_warnings": [],
        },
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
    )

    audit = verify_temporal_outputs([output], {"scene_0001": source_unit})
    apply_temporal_audit_annotations([output], audit)
    graph = build_timeline_graph([output], config=TimelineBuildConfig(use_discourse_order_prior=False))

    relation = next(item for item in graph["relations"] if item["source"] == "llm")
    assert relation["usable_for_ordering"] is False
    assert relation["ordering_error"] == "audit_unusable_evidence"
    assert graph["main_timeline_order"][0]["event_id"] == "scene_0001:event_001"
