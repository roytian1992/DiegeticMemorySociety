from pathlib import Path

from dms.evaluation import build_scene_eligibility_splits, classify_scene
from dms.scripts.wandering_earth import ScriptScene, load_script_scenes


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def _scene(scene_id: str, title: str, content: str) -> ScriptScene:
    numeric_id = int(scene_id.removeprefix("scene_"))
    return ScriptScene(
        scene_id=scene_id,
        source_record_id=numeric_id,
        discourse_index=numeric_id,
        title=title,
        subtitle="",
        content=content,
        raw_heading_number=numeric_id,
        interior_exterior="INT",
        time_of_day="日",
        location_hint="测试地点",
        character_count=len(content),
    )


def test_classify_dialogue_conflict_scene_as_writing_target() -> None:
    scene = _scene("scene_0001", "1、INT.日.实验室", "科学家：为什么要被禁止？我决定继续。")

    record = classify_scene(scene)

    assert record["unit_type"] == "conflict_scene"
    assert record["writing_eval_policy"] == "include"
    assert record["memory_policy"] == "include"


def test_classify_visual_vfx_scene_excludes_writing_eval_but_keeps_memory() -> None:
    scene = _scene("scene_0002", "2、EXT.日.城市", "镜头俯瞰城市，火光、爆炸、冲击波、碎片在画面中扩散。")

    record = classify_scene(scene)

    assert record["unit_type"] == "pure_visual_vfx"
    assert record["writing_eval_policy"] == "exclude"
    assert record["memory_policy"] == "include"
    assert record["audit_eval_policy"] == "include"


def test_build_scene_eligibility_splits_for_fixture(tmp_path: Path) -> None:
    summary = build_scene_eligibility_splits(SCRIPT_PATH, tmp_path)
    scenes = load_script_scenes(SCRIPT_PATH)
    non_empty_count = sum(1 for scene in scenes if scene.content)

    assert summary["scene_count"] == len(scenes)
    assert summary["memory_prefix_count"] == non_empty_count
    assert summary["writing_eval_target_count"] > 0
    assert summary["excluded_from_generation_eval_count"] > 0
    assert (tmp_path / "scene_eligibility_all.jsonl").is_file()
    assert (tmp_path / "writing_eval_targets.jsonl").is_file()
    assert (tmp_path / "excluded_from_generation_eval.jsonl").is_file()
