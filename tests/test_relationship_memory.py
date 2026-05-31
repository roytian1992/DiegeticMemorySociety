import json
from pathlib import Path

from dms.memory import build_relationship_memory


def test_build_relationship_memory_keeps_only_durable_entity_aligned_relationships(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    output_dir = tmp_path / "relationships"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "unit": {
                    "unit_id": "scene_0001",
                    "title": "",
                    "subtitle": "",
                    "content": "甲和乙是盟友。甲看向丙。",
                },
                "extracted_candidates": {
                    "kg_entity_mentions": {
                        "status": "parsed",
                        "data": {
                            "unit_id": "scene_0001",
                            "entity_mentions": [
                                {
                                    "surface": "甲",
                                    "entity_type": "character",
                                    "canonical_hint": "",
                                    "role_in_unit": "actor",
                                    "attributes_or_state": "",
                                    "evidence": "甲",
                                },
                                {
                                    "surface": "乙",
                                    "entity_type": "character",
                                    "canonical_hint": "",
                                    "role_in_unit": "actor",
                                    "attributes_or_state": "",
                                    "evidence": "乙",
                                },
                            ],
                            "unresolved_mentions": [],
                        },
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "data": {
                    "unit_id": "scene_0001",
                    "relationship_observations": [
                        {
                            "source_entity": "甲",
                            "target_entity": "乙",
                            "relation_type": "alliance",
                            "status_or_change": "甲和乙是盟友",
                            "evidence": "甲和乙是盟友",
                        },
                        {
                            "source_entity": "甲",
                            "target_entity": "乙",
                            "relation_type": "ward_of",
                            "status_or_change": "甲照顾乙",
                            "evidence": "甲和乙是盟友",
                        },
                        {
                            "source_entity": "甲",
                            "target_entity": "乙",
                            "relation_type": "addresses",
                            "status_or_change": "甲对乙说话",
                            "evidence": "甲和乙是盟友",
                        },
                        {
                            "source_entity": "乙",
                            "target_entity": "甲",
                            "relation_type": "guardian_of",
                            "status_or_change": "乙照顾甲",
                            "evidence": "甲和乙是盟友",
                        },
                        {
                            "source_entity": "甲",
                            "target_entity": "丙",
                            "relation_type": "alliance",
                            "status_or_change": "甲和丙是盟友",
                            "evidence": "甲看向丙",
                        },
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_relationship_memory(run_dir, output_dir)
    relationships = [
        json.loads(line) for line in (output_dir / "relationship_observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["relationship_observation_count"] == 2
    assert summary["skipped_non_durable_relationship_count"] == 1
    assert summary["skipped_unresolved_endpoint_count"] == 1
    assert summary["skipped_duplicate_relationship_count"] == 1
    assert relationships[0]["source_entity"] == "甲"
    assert relationships[0]["target_entity"] == "乙"
    assert relationships[0]["relation_type"] == "alliance"
    assert relationships[1]["source_entity"] == "乙"
    assert relationships[1]["target_entity"] == "甲"
    assert relationships[1]["relation_type"] == "care_commitment_to"
    assert relationships[1]["model_relation_type"] == "ward_of"


def test_relationship_memory_softens_care_promise_to_avoid_formal_guardianship(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    inputs_dir = run_dir / "inputs"
    output_dir = tmp_path / "relationships"
    parsed_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "unit": {
                    "unit_id": "scene_0001",
                    "title": "",
                    "subtitle": "",
                    "content": "张鹏说：我一定照顾好他。",
                },
                "extracted_candidates": {
                    "kg_entity_mentions": {
                        "status": "parsed",
                        "data": {
                            "unit_id": "scene_0001",
                            "entity_mentions": [
                                {"surface": "张鹏", "entity_type": "character", "evidence": "张鹏"},
                                {"surface": "刘培强", "entity_type": "character", "evidence": "他"},
                            ],
                        },
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parsed_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "data": {
                    "unit_id": "scene_0001",
                    "relationship_observations": [
                        {
                            "source_entity": "张鹏",
                            "target_entity": "刘培强",
                            "relation_type": "guardian_of",
                            "status_or_change": "张鹏承诺照顾刘培强",
                            "evidence": "我一定照顾好他",
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_relationship_memory(run_dir, output_dir)
    relationships = [
        json.loads(line) for line in (output_dir / "relationship_observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["relationship_observation_count"] == 1
    assert relationships[0]["relation_type"] == "care_commitment_to"
    assert relationships[0]["model_relation_type"] == "guardian_of"
