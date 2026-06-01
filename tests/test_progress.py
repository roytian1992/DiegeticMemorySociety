from dms.progress import progress_line


def test_progress_line_includes_count_percent_and_detail() -> None:
    line = progress_line("stage", 2, 4, detail="scene=scene_0002")

    assert "stage 2/4 (50.0%) scene=scene_0002" in line


def test_progress_line_handles_zero_total() -> None:
    line = progress_line("stage", 0, 0)

    assert line.endswith("stage 0/0")
