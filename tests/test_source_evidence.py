from dms.source_evidence import locate_evidence


def test_locate_evidence_exact_match() -> None:
    unit = {"title": "1、INT.日.房间", "subtitle": "", "content": "刘培强进入房间。"}

    result = locate_evidence("刘培强进入房间", unit)

    assert result["evidence_verification_status"] == "exact"
    assert result["evidence_source_field"] == "content"
    assert result["evidence_start"] == 0
    assert result["evidence_end"] == 7


def test_locate_evidence_fuzzy_aligns_minor_punctuation_difference() -> None:
    unit = {
        "title": "1、INT.日.印度 数字生命研究室",
        "subtitle": "",
        "content": "印度科学家（印度式英语）：人，本质上就是一堆电信号。",
    }

    result = locate_evidence("印度科学家(印度式英语):人，本质上就是一堆电信号", unit)

    assert result["evidence_verification_status"] == "fuzzy_aligned"
    assert result["evidence_aligned_text"] == "印度科学家（印度式英语）：人，本质上就是一堆电信号"
    assert result["evidence_source_field"] == "content"


def test_locate_evidence_rejects_skipped_middle_clause() -> None:
    unit = {
        "title": "",
        "subtitle": "",
        "content": "印度科学家（印度式英语）：人，本质上就是一堆电信号。你对这个世界的感知。",
    }

    result = locate_evidence("印度科学家（印度式英语）：你对这个世界的感知。", unit)

    assert result["evidence_verification_status"] == "rejected"


def test_locate_evidence_rejects_large_mismatch() -> None:
    unit = {"title": "", "subtitle": "", "content": "刘培强进入房间。"}

    result = locate_evidence("完全无关的模型编造内容", unit)

    assert result["evidence_verification_status"] == "rejected"
