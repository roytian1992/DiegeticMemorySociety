from pathlib import Path

from dms.scripts.wandering_earth import load_script_scenes, write_jsonl, write_summary


SCRIPT_PATH = Path("data/raw/流浪地球2剧本.json")


def test_load_wandering_earth_script_scenes() -> None:
    scenes = load_script_scenes(SCRIPT_PATH)

    assert len(scenes) == 373
    assert scenes[0].scene_id == "scene_0001"
    assert scenes[0].unit_type == "scene"
    assert scenes[0].unit_label == "scene"
    assert scenes[0].raw_heading_number == 1
    assert scenes[0].interior_exterior == "INT"
    assert scenes[0].time_of_day == "日"
    assert "印度" in scenes[0].location_hint
    assert scenes[-1].source_record_id == 373


def test_load_script_scenes_can_label_units_as_chapters() -> None:
    scenes = load_script_scenes(SCRIPT_PATH, unit_type="Chapter", unit_label="chapter")

    assert scenes[0].scene_id == "scene_0001"
    assert scenes[0].unit_type == "chapter"
    assert scenes[0].unit_label == "chapter"


def test_write_units_and_summary(tmp_path: Path) -> None:
    scenes = load_script_scenes(SCRIPT_PATH)[:2]
    units_path = tmp_path / "units.jsonl"
    summary_path = tmp_path / "summary.json"

    write_jsonl(scenes, units_path)
    write_summary(scenes, summary_path, source_path=SCRIPT_PATH)

    assert units_path.read_text(encoding="utf-8").count("\n") == 2
    assert '"scene_count": 2' in summary_path.read_text(encoding="utf-8")
