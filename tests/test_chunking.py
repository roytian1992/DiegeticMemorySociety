from dms.chunking import chunk_scene, chunk_unit_count
from dms.scripts.wandering_earth import ScriptScene


def test_chunk_scene_keeps_short_scene_as_original_unit_id() -> None:
    scene = _scene("刘培强进入房间。")

    chunks = chunk_scene(scene, max_chunk_units=800)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "scene_0001"
    assert chunks[0].parent_unit_id == "scene_0001"
    assert chunks[0].source_start == 0
    assert chunks[0].source_end == len(scene.content)


def test_chunk_scene_splits_long_text_under_budget() -> None:
    content = "。".join(["刘培强进入房间"] * 900) + "。"
    scene = _scene(content)

    chunks = chunk_scene(scene, max_chunk_units=800)

    assert len(chunks) > 1
    assert all(chunk.chunk_unit_count <= 800 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content
    assert chunks[0].chunk_id == "scene_0001_chunk_001"
    assert chunks[-1].chunk_count == len(chunks)


def test_chunk_scene_hard_caps_run_without_natural_boundaries() -> None:
    content = "刘培强" * 401
    scene = _scene(content)

    chunks = chunk_scene(scene, max_chunk_units=800)

    assert len(chunks) == 2
    assert all(chunk.chunk_unit_count <= 800 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content


def test_chunk_unit_count_mixes_cjk_chars_and_english_words() -> None:
    assert chunk_unit_count("刘培强 said hello world 550A") == 7


def _scene(content: str) -> ScriptScene:
    return ScriptScene(
        scene_id="scene_0001",
        source_record_id=1,
        discourse_index=1,
        title="1、INT.日.房间",
        subtitle="",
        content=content,
        raw_heading_number=1,
        interior_exterior="INT",
        time_of_day="日",
        location_hint="房间",
        character_count=len(content),
    )
