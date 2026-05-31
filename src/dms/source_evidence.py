from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any


SOURCE_FIELDS = ("title", "subtitle", "content")
DEFAULT_FUZZY_THRESHOLD = 0.94
DEFAULT_MAX_NORMALIZED_DELTA_RATIO = 0.08
DEFAULT_MAX_NORMALIZED_DELTA_MIN = 2


def build_source_text_index(unit_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for field in SOURCE_FIELDS:
        text = str(unit_payload.get(field) or "")
        index[field] = {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "char_count": len(text),
        }
    return index


def locate_evidence(evidence: object, unit_payload: dict[str, Any]) -> dict[str, Any]:
    evidence_text = str(evidence or "")
    if not evidence_text:
        return _location("rejected", evidence_text, score=0.0)

    for field in SOURCE_FIELDS:
        source_text = str(unit_payload.get(field) or "")
        start = source_text.find(evidence_text)
        if start >= 0:
            return _location(
                "exact",
                evidence_text,
                aligned_text=evidence_text,
                source_field=field,
                start=start,
                end=start + len(evidence_text),
                source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                score=1.0,
            )
    normalized_match = _normalized_exact_locate_evidence(evidence_text, unit_payload)
    if normalized_match:
        return normalized_match
    return _fuzzy_locate_evidence(evidence_text, unit_payload)


def require_aligned_evidence(
    errors: list[str],
    evidence: object,
    unit_payload: dict[str, Any],
    *,
    label: str,
) -> None:
    location = locate_evidence(evidence, unit_payload)
    if location["evidence_verification_status"] == "rejected":
        errors.append(f"{label}.evidence must align to a contiguous span from title, subtitle, or content")


def require_exact_evidence(
    errors: list[str],
    evidence: object,
    unit_payload: dict[str, Any],
    *,
    label: str,
) -> None:
    require_aligned_evidence(errors, evidence, unit_payload, label=label)


def _normalized_exact_locate_evidence(evidence_text: str, unit_payload: dict[str, Any]) -> dict[str, Any] | None:
    normalized_evidence, _ = _normalize_with_offsets(evidence_text)
    if not normalized_evidence:
        return None

    for field in SOURCE_FIELDS:
        source_text = str(unit_payload.get(field) or "")
        normalized_source, offsets = _normalize_with_offsets(source_text)
        norm_start = normalized_source.find(normalized_evidence)
        if norm_start < 0:
            continue
        norm_end = norm_start + len(normalized_evidence)
        start, end = _source_span_from_normalized_offsets(offsets, norm_start, norm_end)
        aligned_text = source_text[start:end]
        return _location(
            "fuzzy_aligned",
            evidence_text,
            aligned_text=aligned_text,
            source_field=field,
            start=start,
            end=end,
            source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            score=1.0,
        )
    return None


def _fuzzy_locate_evidence(evidence_text: str, unit_payload: dict[str, Any]) -> dict[str, Any]:
    normalized_evidence, _ = _normalize_with_offsets(evidence_text)
    if not normalized_evidence:
        return _location("rejected", evidence_text, score=0.0)

    best: dict[str, Any] | None = None
    for field in SOURCE_FIELDS:
        source_text = str(unit_payload.get(field) or "")
        candidate = _best_window(evidence_text, source_text)
        if candidate is None:
            continue
        score, start, end, aligned_text = candidate
        if best is None or score > float(best["evidence_alignment_score"]):
            best = _location(
                "fuzzy_aligned" if score >= DEFAULT_FUZZY_THRESHOLD else "rejected",
                evidence_text,
                aligned_text=aligned_text,
                source_field=field,
                start=start,
                end=end,
                source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                score=score,
            )

    if best and best["evidence_verification_status"] != "rejected":
        return best
    if best:
        return best
    return _location("rejected", evidence_text, score=0.0)


def _best_window(evidence_text: str, source_text: str) -> tuple[float, int, int, str] | None:
    normalized_evidence, _ = _normalize_with_offsets(evidence_text)
    normalized_source, offsets = _normalize_with_offsets(source_text)
    if not normalized_evidence or not normalized_source:
        return None

    evidence_len = len(normalized_evidence)
    max_delta = max(DEFAULT_MAX_NORMALIZED_DELTA_MIN, round(evidence_len * DEFAULT_MAX_NORMALIZED_DELTA_RATIO))
    min_len = max(1, evidence_len - max_delta)
    max_len = min(len(normalized_source), evidence_len + max_delta)

    best: tuple[float, int, int, str] | None = None
    for norm_start in range(0, len(normalized_source)):
        for window_len in range(min_len, max_len + 1):
            norm_end = norm_start + window_len
            if norm_end > len(normalized_source):
                break
            candidate = normalized_source[norm_start:norm_end]
            score = _similarity(normalized_evidence, candidate)
            if best is None or score > best[0]:
                start, end = _source_span_from_normalized_offsets(offsets, norm_start, norm_end)
                best = (score, start, end, source_text[start:end])
    if best and best[0] >= DEFAULT_FUZZY_THRESHOLD:
        return best
    return best


def _similarity(left: str, right: str) -> float:
    norm_left, _ = _normalize_with_offsets(left)
    norm_right, _ = _normalize_with_offsets(right)
    if not norm_left or not norm_right:
        return 0.0
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def _normalize_for_match(text: str) -> str:
    normalized, _ = _normalize_with_offsets(text)
    return normalized


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(str(text or "")):
        folded = char.lower()
        if re.match(r"[\s，。！？、,.!?；;：:“”\"'（）()\[\]【】…]+", folded):
            continue
        normalized_chars.append(folded)
        offsets.append(index)
    return "".join(normalized_chars), offsets


def _source_span_from_normalized_offsets(offsets: Sequence[int], norm_start: int, norm_end: int) -> tuple[int, int]:
    start = offsets[norm_start]
    end = offsets[norm_end - 1] + 1
    return start, end


def _location(
    status: str,
    evidence_text: str,
    *,
    aligned_text: str = "",
    source_field: str | None = None,
    start: int | None = None,
    end: int | None = None,
    source_sha256: str | None = None,
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "evidence_exact_match": status == "exact",
        "evidence_verification_status": status,
        "evidence_text": evidence_text,
        "evidence_aligned_text": aligned_text,
        "evidence_source_field": source_field,
        "evidence_start": start,
        "evidence_end": end,
        "evidence_source_sha256": source_sha256,
        "evidence_alignment_score": round(score, 4),
    }
