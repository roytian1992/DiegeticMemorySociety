import json
from pathlib import Path

from dms.memory import build_canonical_memory, build_scene_inventory_memory, build_visibility_packet, query_memory


def _build_fake_canonical(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    parsed_dir = run_dir / "parsed"
    staged_dir = tmp_path / "staged"
    canonical_dir = tmp_path / "canonical"
    parsed_dir.mkdir(parents=True)
    for scene_id in ("scene_0001", "scene_0002"):
        (parsed_dir / f"{scene_id}.json").write_text(
            json.dumps(
                {
                    "status": "parsed",
                    "scene_id": scene_id,
                    "data": {
                        "scene_id": scene_id,
                        "setting": {"location": "FAKE_LOCATION", "time_of_day": "", "interior_exterior": ""},
                        "characters": [{"name": "FAKE_CHARACTER", "evidence": "fixture"}],
                        "objects": [],
                        "stated_facts": [],
                        "open_questions": [],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    build_scene_inventory_memory(run_dir, staged_dir)
    build_canonical_memory(staged_dir, canonical_dir)
    return canonical_dir


def _build_frame_only_canonical(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_frame_only"
    parsed_dir = run_dir / "parsed"
    staged_dir = tmp_path / "staged_frame_only"
    canonical_dir = tmp_path / "canonical_frame_only"
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "scene_0001.json").write_text(
        json.dumps(
            {
                "status": "parsed",
                "scene_id": "scene_0001",
                "data": {
                    "scene_id": "scene_0001",
                    "setting": {"location": "FAKE_LOCATION", "time_of_day": "", "interior_exterior": ""},
                    "stated_facts": [],
                    "open_questions": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    build_scene_inventory_memory(run_dir, staged_dir)
    build_canonical_memory(staged_dir, canonical_dir)
    return canonical_dir


def test_build_canonical_memory_from_staged_inventory(tmp_path: Path) -> None:
    canonical_dir = _build_fake_canonical(tmp_path)

    assert (canonical_dir / "canonical_memory.json").is_file()
    assert (canonical_dir / "summary.json").is_file()
    assert (canonical_dir / "characters.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_query_memory_by_character(tmp_path: Path) -> None:
    canonical_dir = _build_fake_canonical(tmp_path)

    result = query_memory(canonical_dir, character="FAKE_CHARACTER")

    assert result["counts"]["characters"] == 1
    assert result["counts"]["scenes"] == 2


def test_visibility_packet_for_character(tmp_path: Path) -> None:
    canonical_dir = _build_fake_canonical(tmp_path)

    packet = build_visibility_packet(canonical_dir, character="FAKE_CHARACTER", scene_id="scene_0002")

    assert packet["packet_type"] == "character_visibility"
    assert packet["character"]["canonical_name"] == "FAKE_CHARACTER"
    assert packet["visible_scene_ids"] == ["scene_0001", "scene_0002"]


def test_canonical_memory_accepts_frame_only_inventory(tmp_path: Path) -> None:
    canonical_dir = _build_frame_only_canonical(tmp_path)

    summary = (canonical_dir / "summary.json").read_text(encoding="utf-8")

    assert '"scene_count": 1' in summary
    assert '"character_count": 0' in summary
