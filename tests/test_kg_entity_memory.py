from pathlib import Path
import json

from dms.llm import FakeKGEntityMentionClient
from dms.memory import build_kg_entity_memory
from dms.runners import KGEntityRunConfig, run_kg_entity_mentions


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_build_kg_entity_memory_from_fake_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    memory_dir = tmp_path / "memory"
    run_kg_entity_mentions(
        KGEntityRunConfig(
            script_path=SCRIPT_PATH,
            output_dir=run_dir,
            limit=2,
            dry_run=False,
        ),
        llm_client=FakeKGEntityMentionClient(),
    )

    summary = build_kg_entity_memory(run_dir, memory_dir)

    assert summary["accepted_scene_count"] == 2
    assert summary["skipped_scene_count"] == 0
    assert summary["entity_mention_count"] == 8
    assert summary["unresolved_mention_count"] == 2
    assert summary["entity_type_counts"]["character"] == 2
    assert summary["entity_type_counts"]["object"] == 2
    assert summary["entity_type_counts"]["concept"] == 2
    assert summary["entity_type_counts"]["occasion"] == 2
    assert (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8").count("\n") == 8
    assert "a trackable test character" in (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8")
    assert (memory_dir / "unresolved_mentions.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (memory_dir / "summary.json").is_file()


def test_build_kg_entity_memory_demotes_deictic_mentions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0001",
            "entity_mentions": [
              {
                "surface": "这儿",
                "entity_type": "location",
                "canonical_hint": "",
                "role_in_unit": "storage location",
                "attributes_or_state": "ambiguous",
                "evidence": "存到这儿"
              },
              {
                "surface": "脑机接口设备",
                "entity_type": "object",
                "canonical_hint": "",
                "description": "一种脑机交互设备",
                "role_in_unit": "device",
                "attributes_or_state": "",
                "evidence": "脑机接口设备"
              }
            ],
            "unresolved_mentions": []
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_kg_entity_memory(run_dir, memory_dir)
    mentions = (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8")
    unresolved = (memory_dir / "unresolved_mentions.jsonl").read_text(encoding="utf-8")

    assert summary["entity_mention_count"] == 1
    assert summary["unresolved_mention_count"] == 1
    assert "脑机接口设备" in mentions
    assert "一种脑机交互设备" in mentions
    assert "这儿" not in mentions
    assert "这儿" in unresolved


def test_build_kg_entity_memory_filters_non_trackable_entities_and_unsupported_hints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "scene_0001.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0001",
            "entity_mentions": [
              {
                "surface": "脑子",
                "entity_type": "concept",
                "canonical_hint": "大脑",
                "role_in_unit": "concept",
                "attributes_or_state": "",
                "evidence": "脑子"
              },
              {
                "surface": "镜头",
                "entity_type": "object",
                "canonical_hint": "摄像镜头",
                "role_in_unit": "object",
                "attributes_or_state": "",
                "evidence": "镜头"
              },
              {
                "surface": "脑电波",
                "entity_type": "concept",
                "canonical_hint": "",
                "role_in_unit": "data",
                "attributes_or_state": "",
                "evidence": "脑电波"
              }
            ],
            "unresolved_mentions": []
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_kg_entity_memory(run_dir, memory_dir)
    mentions = (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8")
    unresolved = (memory_dir / "unresolved_mentions.jsonl").read_text(encoding="utf-8")

    assert summary["entity_mention_count"] == 2
    assert summary["unresolved_mention_count"] == 1
    assert "脑子" not in mentions
    assert "脑子" in unresolved
    assert '"surface": "镜头"' in mentions
    assert '"canonical_hint": ""' in mentions
    assert "摄像镜头" not in mentions
    assert "脑电波" in mentions


def test_build_kg_entity_memory_demotes_low_value_occasions_to_scene_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "scene_0003.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0003",
            "entity_mentions": [
              {
                "surface": "山火",
                "entity_type": "occasion",
                "canonical_hint": "",
                "role_in_unit": "rhetorical example",
                "attributes_or_state": "",
                "evidence": "这不过是一场山火，一次旱灾，一个物种的灭绝，一座城市的消失"
              },
              {
                "surface": "一次旱灾",
                "entity_type": "occasion",
                "canonical_hint": "",
                "role_in_unit": "rhetorical example",
                "attributes_or_state": "",
                "evidence": "这不过是一场山火，一次旱灾，一个物种的灭绝，一座城市的消失"
              },
              {
                "surface": "一个物种的灭绝",
                "entity_type": "occasion",
                "canonical_hint": "",
                "role_in_unit": "rhetorical example",
                "attributes_or_state": "",
                "evidence": "这不过是一场山火，一次旱灾，一个物种的灭绝，一座城市的消失"
              },
              {
                "surface": "数字生命备份卡",
                "entity_type": "object",
                "canonical_hint": "",
                "role_in_unit": "prop",
                "attributes_or_state": "",
                "evidence": "数字生命备份卡"
              }
            ],
            "scene_tags": [
              {
                "surface": "昏暗",
                "tag_type": "atmosphere",
                "reason": "local ambience",
                "evidence": "昏暗"
              }
            ],
            "unresolved_mentions": []
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_kg_entity_memory(run_dir, memory_dir)
    mentions = (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8")
    scene_tags = (memory_dir / "scene_tags.jsonl").read_text(encoding="utf-8")

    assert summary["entity_mention_count"] == 1
    assert summary["scene_tag_count"] == 4
    assert summary["scene_tag_type_counts"]["illustrative_example"] == 3
    assert "数字生命备份卡" in mentions
    assert "山火" not in mentions
    assert "山火" in scene_tags
    assert "旱灾" in scene_tags
    assert "物种的灭绝" in scene_tags
    assert "昏暗" in scene_tags


def test_build_kg_entity_memory_demotes_visual_background_entities(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    memory_dir = tmp_path / "memory"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "scene_0004_chunk_002.json").write_text(
        """
        {
          "status": "parsed",
          "data": {
            "unit_id": "scene_0004_chunk_002",
            "entity_mentions": [
              {
                "surface": "乌鸦",
                "entity_type": "character",
                "canonical_hint": "",
                "role_in_unit": "wildlife visual detail",
                "attributes_or_state": "",
                "evidence": "一只乌鸦立足其上"
              },
              {
                "surface": "水管",
                "entity_type": "object",
                "canonical_hint": "",
                "role_in_unit": "environment debris",
                "attributes_or_state": "",
                "evidence": "裸露在外的水管孤零零地从弹坑壁上伸出"
              },
              {
                "surface": "积水",
                "entity_type": "object",
                "canonical_hint": "",
                "role_in_unit": "environment",
                "attributes_or_state": "",
                "evidence": "坑内有一些积水"
              },
              {
                "surface": "巴士",
                "entity_type": "object",
                "canonical_hint": "",
                "role_in_unit": "background vehicle",
                "attributes_or_state": "",
                "evidence": "经过一辆被炸翻在路中间的巴士"
              },
              {
                "surface": "太阳",
                "entity_type": "location",
                "canonical_hint": "",
                "role_in_unit": "direction cue",
                "attributes_or_state": "",
                "evidence": "刘培强摘下墨镜，直视太阳的方向"
              },
              {
                "surface": "太阳危机",
                "entity_type": "concept",
                "canonical_hint": "",
                "role_in_unit": "core world concept",
                "attributes_or_state": "",
                "evidence": "师父，真的有太阳危机吗？"
              },
              {
                "surface": "距太阳氦闪还剩34 年",
                "entity_type": "time_or_deadline",
                "canonical_hint": "",
                "role_in_unit": "deadline",
                "attributes_or_state": "34年",
                "evidence": "字卡：距太阳氦闪还剩34 年"
              },
              {
                "surface": "张鹏",
                "entity_type": "character",
                "canonical_hint": "",
                "role_in_unit": "speaker",
                "attributes_or_state": "",
                "evidence": "张鹏：到了，他们就这儿呢。"
              }
            ],
            "scene_tags": [],
            "unresolved_mentions": []
          }
        }
        """,
        encoding="utf-8",
    )

    summary = build_kg_entity_memory(run_dir, memory_dir)
    mentions = (memory_dir / "entity_mentions.jsonl").read_text(encoding="utf-8")
    scene_tags = (memory_dir / "scene_tags.jsonl").read_text(encoding="utf-8")

    assert summary["entity_mention_count"] == 3
    assert summary["scene_tag_count"] == 5
    assert "太阳危机" in mentions
    assert "太阳氦闪" in mentions
    assert "张鹏" in mentions
    mention_surfaces = [
        json.loads(line)["surface"]
        for line in mentions.splitlines()
        if line.strip()
    ]
    assert "距太阳氦闪还剩34 年" not in mention_surfaces
    for surface in ("乌鸦", "水管", "积水", "巴士", "太阳"):
        assert f'"surface": "{surface}"' not in mentions
        assert f'"surface": "{surface}"' in scene_tags
