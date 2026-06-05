from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from dms.source_evidence import build_source_text_index, locate_evidence

ORDERING_RELATION_TYPES = {"before", "after", "contains", "causes", "reveals_past"}
SYMMETRIC_RELATION_TYPES = {"overlaps", "same_time"}
GRAPH_RELATION_TYPES = ORDERING_RELATION_TYPES | SYMMETRIC_RELATION_TYPES
MAX_GAP_FILL_SPANS = 2
MAX_GAP_FILL_NON_WS_CHARS = 40

PARAPHRASE_EVIDENCE_TERMS = (
    "暗示",
    "说明",
    "表明",
    "意味着",
    "指向",
    "推测",
    "可见",
    "根据",
    "发生在",
    "先于",
    "晚于",
    "之前",
    "之后",
    "because",
    "suggests",
    "indicates",
    "before",
    "after",
    "event_",
)


def verify_temporal_outputs(
    scene_outputs: list[dict[str, Any]],
    source_units_by_scene: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Audit temporal extraction outputs against source text spans.

    This verifier is deliberately separate from JSON schema validation. Schema
    validation decides whether the model returned a structurally usable object;
    this audit decides whether its cited evidence can be trusted for ordering.
    """

    counter: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    annotations = {
        "events": {},
        "relations": {},
        "scene_temporal_index": {},
        "scene_temporal_hints": {},
        "warnings": {},
    }
    source_index = {
        scene_id: build_source_text_index(source_unit)
        for scene_id, source_unit in source_units_by_scene.items()
        if isinstance(source_unit, dict)
    }

    for output in scene_outputs:
        scene_id = str(output.get("scene_id") or "")
        source_unit = source_units_by_scene.get(scene_id)
        source_is_chinese = _source_is_chinese(source_unit)
        counter["scene_count"] += 1
        if not isinstance(source_unit, dict):
            issues.append(
                _issue(
                    counter,
                    scene_id=scene_id,
                    kind="scene",
                    item_id=scene_id,
                    field="source",
                    issue_type="source_unit_missing",
                    severity="error",
                    detail="No source unit was available for temporal evidence verification.",
                )
            )
            continue

        for event in output.get("temporal_events") or []:
            if not isinstance(event, dict):
                continue
            counter["event_count"] += 1
            event_id = str(event.get("event_id") or "")
            annotation, evidence_issues = _verify_evidence_field(
                counter,
                issues,
                source_unit,
                scene_id=scene_id,
                kind="event",
                item_id=event_id,
                evidence=event.get("evidence"),
            )
            annotations["events"][event_id] = annotation
            if source_is_chinese:
                for field, value in _event_language_fields(event):
                    _add_non_chinese_issue_if_needed(
                        counter,
                        issues,
                        scene_id=scene_id,
                        kind="event",
                        item_id=event_id,
                        field=field,
                        value=value,
                    )
            if evidence_issues:
                counter["event_evidence_issue_count"] += len(evidence_issues)

        for relation in output.get("temporal_relations") or []:
            if not isinstance(relation, dict):
                continue
            counter["relation_count"] += 1
            relation_id = str(relation.get("relation_id") or "")
            relation_type = str(relation.get("relation_type") or "")
            annotation, evidence_issues = _verify_evidence_field(
                counter,
                issues,
                source_unit,
                scene_id=scene_id,
                kind="relation",
                item_id=relation_id,
                evidence=relation.get("evidence"),
            )
            evidence_aligned = bool(annotation.get("audit_evidence_aligned"))
            if relation_type in GRAPH_RELATION_TYPES:
                annotation["audit_ordering_usable"] = evidence_aligned
                if not evidence_aligned:
                    counter["hard_ordering_unusable_count"] += 1
                    issues.append(
                        _issue(
                            counter,
                            scene_id=scene_id,
                            kind="relation",
                            item_id=relation_id,
                            field="evidence",
                            issue_type="hard_ordering_unusable",
                            severity="error",
                            detail="Relation evidence did not align to source text, so this relation must not be used as an ordering edge.",
                            evidence=str(relation.get("evidence") or ""),
                        )
                    )
            annotations["relations"][relation_id] = annotation
            if evidence_issues:
                counter["relation_evidence_issue_count"] += len(evidence_issues)

        scene_index = output.get("scene_temporal_index")
        if isinstance(scene_index, dict):
            counter["scene_index_count"] += 1
            scene_index_id = scene_id
            annotation, _ = _verify_evidence_field(
                counter,
                issues,
                source_unit,
                scene_id=scene_id,
                kind="scene_temporal_index",
                item_id=scene_index_id,
                evidence=scene_index.get("evidence"),
            )
            annotations["scene_temporal_index"][scene_index_id] = annotation
            if source_is_chinese:
                for field in ("absolute_time_hints", "relative_time_hints"):
                    for index, value in enumerate(_as_list(scene_index.get(field)), start=1):
                        field_key = f"{field}[{index}]"
                        hint_id = f"{scene_id}:{field_key}"
                        counter["time_hint_count"] += 1
                        hint_annotation, hint_issues = _verify_evidence_field(
                            counter,
                            issues,
                            source_unit,
                            scene_id=scene_id,
                            kind="scene_temporal_hint",
                            item_id=hint_id,
                            field=field_key,
                            evidence=value,
                        )
                        annotations["scene_temporal_hints"][hint_id] = hint_annotation
                        if hint_issues:
                            counter["time_hint_issue_count"] += len(hint_issues)
                        _add_non_chinese_issue_if_needed(
                            counter,
                            issues,
                            scene_id=scene_id,
                            kind="scene_temporal_hint",
                            item_id=hint_id,
                            field=field_key,
                            value=str(value),
                        )

        for index, warning in enumerate(output.get("temporal_warnings") or [], start=1):
            if not isinstance(warning, dict):
                continue
            counter["warning_count"] += 1
            warning_id = str(warning.get("warning_id") or f"{scene_id}:warning_{index:03d}")
            annotation, _ = _verify_evidence_field(
                counter,
                issues,
                source_unit,
                scene_id=scene_id,
                kind="warning",
                item_id=warning_id,
                evidence=warning.get("evidence"),
            )
            annotations["warnings"][warning_id] = annotation
            if source_is_chinese:
                _add_non_chinese_issue_if_needed(
                    counter,
                    issues,
                    scene_id=scene_id,
                    kind="warning",
                    item_id=warning_id,
                    field="detail",
                    value=str(warning.get("detail") or ""),
                )

    counter["issue_count"] = len(issues)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "temporal_evidence_audit_v0",
        "policy": {
            "source_fields": ["title", "subtitle", "content"],
            "schema_validation_is_separate": True,
            "rejected_relation_evidence_blocks_ordering": True,
            "short_multispan_evidence_gap_fill": True,
            "max_gap_fill_non_ws_chars": MAX_GAP_FILL_NON_WS_CHARS,
            "non_chinese_output_is_warning_only": True,
        },
        "counts": _ordered_counts(counter),
        "source_index": source_index,
        "annotations": annotations,
        "issues": issues,
    }


def apply_temporal_audit_annotations(
    scene_outputs: list[dict[str, Any]],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add audit fields to normalized temporal outputs in place."""

    annotations = audit.get("annotations") if isinstance(audit.get("annotations"), dict) else {}
    event_annotations = annotations.get("events") if isinstance(annotations.get("events"), dict) else {}
    relation_annotations = annotations.get("relations") if isinstance(annotations.get("relations"), dict) else {}
    scene_index_annotations = (
        annotations.get("scene_temporal_index")
        if isinstance(annotations.get("scene_temporal_index"), dict)
        else {}
    )
    scene_hint_annotations = (
        annotations.get("scene_temporal_hints")
        if isinstance(annotations.get("scene_temporal_hints"), dict)
        else {}
    )
    warning_annotations = annotations.get("warnings") if isinstance(annotations.get("warnings"), dict) else {}

    for output in scene_outputs:
        for event in output.get("temporal_events") or []:
            if isinstance(event, dict):
                _merge_annotation(event, event_annotations.get(str(event.get("event_id") or "")))
        for relation in output.get("temporal_relations") or []:
            if isinstance(relation, dict):
                _merge_annotation(relation, relation_annotations.get(str(relation.get("relation_id") or "")))
        scene_index = output.get("scene_temporal_index")
        if isinstance(scene_index, dict):
            scene_id = str(output.get("scene_id") or "")
            _merge_annotation(scene_index, scene_index_annotations.get(scene_id))
            time_hint_audit: dict[str, Any] = {}
            for field in ("absolute_time_hints", "relative_time_hints"):
                for index, _ in enumerate(_as_list(scene_index.get(field)), start=1):
                    field_key = f"{field}[{index}]"
                    annotation = scene_hint_annotations.get(f"{scene_id}:{field_key}")
                    if isinstance(annotation, dict):
                        time_hint_audit[field_key] = annotation
            if time_hint_audit:
                scene_index["time_hint_audit"] = time_hint_audit
        for warning in output.get("temporal_warnings") or []:
            if isinstance(warning, dict):
                _merge_annotation(warning, warning_annotations.get(str(warning.get("warning_id") or "")))
    return scene_outputs


def format_temporal_audit_report(audit: dict[str, Any], *, max_issues: int = 80) -> str:
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    issues = audit.get("issues") if isinstance(audit.get("issues"), list) else []
    lines = [
        "# Temporal Evidence Audit",
        "",
        "## Policy",
        "- Evidence must align to a contiguous span from title, subtitle, or content.",
        "- Short two-fragment evidence may be gap-filled to a contiguous source span and remains annotated.",
        "- Rejected relation evidence blocks that relation from story-time ordering.",
        "- Language and paraphrase checks are audit warnings unless they also fail span alignment.",
        "",
        "## Counts",
    ]
    for key in (
        "scene_count",
        "event_count",
        "relation_count",
        "time_hint_count",
        "evidence_checked_count",
        "evidence_aligned_count",
        "evidence_rejected_count",
        "gap_filled_evidence_count",
        "elliptical_evidence_count",
        "paraphrase_evidence_count",
        "non_chinese_output_count",
        "time_hint_issue_count",
        "hard_ordering_unusable_count",
        "issue_count",
    ):
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.extend(["", "## Issues"])
    if not issues:
        lines.append("- No temporal evidence audit issues.")
    for item in issues[:max_issues]:
        evidence = str(item.get("evidence") or "").replace("\n", " ").strip()
        if len(evidence) > 120:
            evidence = evidence[:117] + "..."
        evidence_suffix = f" | evidence: {evidence}" if evidence else ""
        lines.append(
            "- "
            f"[{item.get('severity')}] {item.get('scene_id')} "
            f"{item.get('kind')} {item.get('item_id')} "
            f"{item.get('field')} {item.get('issue_type')}: {item.get('detail')}"
            f"{evidence_suffix}"
        )
        suggestions = item.get("suggested_evidence_spans")
        if isinstance(suggestions, list) and suggestions:
            preview = []
            for suggestion in suggestions[:3]:
                if not isinstance(suggestion, dict):
                    continue
                text = str(suggestion.get("evidence_aligned_text") or suggestion.get("evidence_text") or "")
                text = text.replace("\n", " ").strip()
                if len(text) > 70:
                    text = text[:67] + "..."
                preview.append(
                    f"{suggestion.get('evidence_source_field')}:{suggestion.get('evidence_start')}-{suggestion.get('evidence_end')} `{text}`"
                )
            if preview:
                lines.append(f"  suggested spans: {'; '.join(preview)}")
        gap_fill = item.get("gap_fill")
        if isinstance(gap_fill, dict):
            lines.append(
                "  gap fill: "
                f"{gap_fill.get('evidence_source_field')}:{gap_fill.get('evidence_start')}-{gap_fill.get('evidence_end')} "
                f"gap_non_ws_chars={gap_fill.get('gap_non_ws_char_count')}"
            )
    if len(issues) > max_issues:
        lines.append(f"- ... {len(issues) - max_issues} additional issues omitted.")
    lines.append("")
    return "\n".join(lines)


def _verify_evidence_field(
    counter: Counter[str],
    issues: list[dict[str, Any]],
    source_unit: dict[str, Any],
    *,
    scene_id: str,
    kind: str,
    item_id: str,
    evidence: Any,
    field: str = "evidence",
) -> tuple[dict[str, Any], list[str]]:
    evidence_text = str(evidence or "").strip()
    evidence_issues: list[str] = []
    counter["evidence_checked_count"] += 1
    location = locate_evidence(evidence_text, source_unit)
    status = str(location.get("evidence_verification_status") or "rejected")
    suspicious_ellipsis = _has_suspicious_ellipsis(evidence_text, source_unit)
    suggested_spans = _suggest_evidence_spans(evidence_text, source_unit) if suspicious_ellipsis or status == "rejected" else []
    gap_fill = _gap_fill_evidence_span(suggested_spans, source_unit)
    if gap_fill:
        location = gap_fill
        status = str(location.get("evidence_verification_status") or status)
        counter["gap_filled_evidence_count"] += 1
    audit_aligned = (status in {"exact", "fuzzy_aligned"} and not suspicious_ellipsis) or bool(gap_fill)
    if status == "exact" and audit_aligned:
        counter["evidence_exact_count"] += 1
    if audit_aligned:
        counter["evidence_aligned_count"] += 1
    else:
        counter["evidence_rejected_count"] += 1

    if not evidence_text:
        evidence_issues.append("empty_evidence")
        counter["empty_evidence_count"] += 1
        issues.append(
            _issue(
                counter,
                scene_id=scene_id,
                kind=kind,
                item_id=item_id,
                field=field,
                issue_type="empty_evidence",
                severity="error",
                detail="Evidence is empty.",
            )
        )
    elif status == "rejected":
        evidence_issues.append("evidence_not_found")
        counter["evidence_not_found_count"] += 1
        issues.append(
            _issue(
                counter,
                scene_id=scene_id,
                kind=kind,
                item_id=item_id,
                field=field,
                issue_type="evidence_not_found",
                severity="error",
                detail="Evidence did not align to title, subtitle, or content.",
                evidence=evidence_text,
                location=location,
                suggested_evidence_spans=suggested_spans,
            )
        )

    if gap_fill and not suspicious_ellipsis:
        evidence_issues.append("gap_filled_evidence")
        issues.append(
            _issue(
                counter,
                scene_id=scene_id,
                kind=kind,
                item_id=item_id,
                field=field,
                issue_type="gap_filled_evidence",
                severity="warning",
                detail="Evidence used multiple source fragments with a short gap; the verifier filled the gap to a contiguous source span.",
                evidence=evidence_text,
                location=location,
                suggested_evidence_spans=suggested_spans,
                gap_fill=gap_fill,
            )
        )

    if suspicious_ellipsis:
        evidence_issues.append("elliptical_evidence")
        counter["elliptical_evidence_count"] += 1
        issues.append(
            _issue(
                counter,
                scene_id=scene_id,
                kind=kind,
                item_id=item_id,
                field=field,
                issue_type="elliptical_evidence",
                severity="warning",
                detail=(
                    "Evidence contains an ellipsis that was gap-filled to a contiguous source span."
                    if gap_fill
                    else "Evidence contains an ellipsis that is not a verbatim source span."
                ),
                evidence=evidence_text,
                location=location,
                suggested_evidence_spans=suggested_spans,
                gap_fill=gap_fill,
            )
        )

    if status == "rejected" and _looks_like_paraphrase_evidence(evidence_text):
        evidence_issues.append("paraphrase_evidence")
        counter["paraphrase_evidence_count"] += 1
        issues.append(
            _issue(
                counter,
                scene_id=scene_id,
                kind=kind,
                item_id=item_id,
                field=field,
                issue_type="paraphrase_evidence",
                severity="warning",
                detail="Evidence looks like an explanatory paraphrase rather than a source quote.",
                evidence=evidence_text,
                location=location,
                suggested_evidence_spans=suggested_spans,
            )
        )

    annotation = {
        **location,
        "audit_evidence_aligned": audit_aligned,
        "evidence_issues": evidence_issues,
        "suggested_evidence_spans": suggested_spans,
    }
    if gap_fill:
        annotation["gap_filled_evidence"] = True
        annotation["gap_filled_evidence_spans"] = suggested_spans[:MAX_GAP_FILL_SPANS]
        annotation["gap_filled_evidence_gap_text"] = gap_fill.get("gap_filled_evidence_gap_text", "")
        annotation["gap_filled_evidence_gap_char_count"] = gap_fill.get("gap_filled_evidence_gap_char_count", 0)
        annotation["gap_filled_evidence_gap_non_ws_char_count"] = gap_fill.get(
            "gap_filled_evidence_gap_non_ws_char_count",
            0,
        )
    return annotation, evidence_issues


def _add_non_chinese_issue_if_needed(
    counter: Counter[str],
    issues: list[dict[str, Any]],
    *,
    scene_id: str,
    kind: str,
    item_id: str,
    field: str,
    value: str,
) -> None:
    text = str(value or "").strip()
    if not _looks_like_non_chinese_output(text):
        return
    counter["non_chinese_output_count"] += 1
    issues.append(
        _issue(
            counter,
            scene_id=scene_id,
            kind=kind,
            item_id=item_id,
            field=field,
            issue_type="non_chinese_output",
            severity="warning",
            detail="Chinese source text produced a likely English output field.",
            evidence=text,
        )
    )


def _merge_annotation(target: dict[str, Any], annotation: Any) -> None:
    if not isinstance(annotation, dict):
        return
    for key in (
        "evidence_exact_match",
        "evidence_verification_status",
        "evidence_aligned_text",
        "evidence_source_field",
        "evidence_start",
        "evidence_end",
        "evidence_source_sha256",
        "evidence_alignment_score",
        "gap_filled_evidence",
        "gap_filled_evidence_spans",
        "gap_filled_evidence_gap_text",
        "gap_filled_evidence_gap_char_count",
        "gap_filled_evidence_gap_non_ws_char_count",
        "audit_evidence_aligned",
        "audit_ordering_usable",
        "evidence_issues",
        "suggested_evidence_spans",
    ):
        if key in annotation:
            target[key] = annotation[key]


def _issue(
    counter: Counter[str],
    *,
    scene_id: str,
    kind: str,
    item_id: str,
    field: str,
    issue_type: str,
    severity: str,
    detail: str,
    evidence: str = "",
    location: dict[str, Any] | None = None,
    suggested_evidence_spans: list[dict[str, Any]] | None = None,
    gap_fill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counter["issue_sequence"] += 1
    payload: dict[str, Any] = {
        "issue_id": f"temporal_audit:{counter['issue_sequence']:04d}",
        "scene_id": scene_id,
        "kind": kind,
        "item_id": item_id,
        "field": field,
        "issue_type": issue_type,
        "severity": severity,
        "detail": detail,
    }
    if evidence:
        payload["evidence"] = evidence
    if location:
        payload["location"] = location
    if suggested_evidence_spans:
        payload["suggested_evidence_spans"] = suggested_evidence_spans
    if gap_fill:
        payload["gap_fill"] = {
            "evidence_source_field": gap_fill.get("evidence_source_field"),
            "evidence_start": gap_fill.get("evidence_start"),
            "evidence_end": gap_fill.get("evidence_end"),
            "gap_char_count": gap_fill.get("gap_filled_evidence_gap_char_count", 0),
            "gap_non_ws_char_count": gap_fill.get("gap_filled_evidence_gap_non_ws_char_count", 0),
        }
    return payload


def _event_language_fields(event: dict[str, Any]) -> list[tuple[str, str]]:
    fields = [
        ("summary", str(event.get("summary") or "")),
        ("location", str(event.get("location") or "")),
        ("story_time_hint", str(event.get("story_time_hint") or "")),
    ]
    for index, participant in enumerate(event.get("participants") or [], start=1):
        fields.append((f"participants[{index}]", str(participant or "")))
    return fields


def _source_is_chinese(source_unit: Any) -> bool:
    if not isinstance(source_unit, dict):
        return False
    text = " ".join(str(source_unit.get(field) or "") for field in ("title", "subtitle", "content"))
    return _cjk_count(text) >= 4


def _looks_like_non_chinese_output(text: str) -> bool:
    if not text:
        return False
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = _cjk_count(text)
    if latin < 10:
        return False
    return latin > max(6, cjk * 0.5)


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))


def _has_suspicious_ellipsis(evidence: str, source_unit: dict[str, Any]) -> bool:
    text = str(evidence or "")
    if "..." not in text and "…" not in text:
        return False
    for field in ("title", "subtitle", "content"):
        if text and text in str(source_unit.get(field) or ""):
            return False
    return True


def _suggest_evidence_spans(evidence: str, source_unit: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[str | None, int | None, int | None, str]] = set()
    for fragment in _evidence_fragments(evidence):
        location = locate_evidence(fragment, source_unit)
        if location.get("evidence_verification_status") not in {"exact", "fuzzy_aligned"}:
            continue
        key = (
            location.get("evidence_source_field"),
            location.get("evidence_start"),
            location.get("evidence_end"),
            str(location.get("evidence_aligned_text") or location.get("evidence_text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(location)
    return suggestions


def _gap_fill_evidence_span(suggested_spans: list[dict[str, Any]], source_unit: dict[str, Any]) -> dict[str, Any] | None:
    spans = suggested_spans[:MAX_GAP_FILL_SPANS]
    if len(spans) != MAX_GAP_FILL_SPANS:
        return None
    if any(span.get("evidence_verification_status") not in {"exact", "fuzzy_aligned"} for span in spans):
        return None
    field = str(spans[0].get("evidence_source_field") or "")
    if not field or any(str(span.get("evidence_source_field") or "") != field for span in spans):
        return None
    try:
        original_first_start = int(spans[0].get("evidence_start"))
        original_second_start = int(spans[1].get("evidence_start"))
        ordered = sorted(
            spans,
            key=lambda span: (
                int(span.get("evidence_start")),
                int(span.get("evidence_end")),
            ),
        )
        first_start = int(ordered[0].get("evidence_start"))
        first_end = int(ordered[0].get("evidence_end"))
        second_start = int(ordered[1].get("evidence_start"))
        second_end = int(ordered[1].get("evidence_end"))
    except (TypeError, ValueError):
        return None
    if original_first_start > original_second_start:
        return None
    if first_start < 0 or second_start < 0 or first_end > second_start or second_end <= second_start:
        return None

    source_text = str(source_unit.get(field) or "")
    gap_text = source_text[first_end:second_start]
    gap_non_ws = re.sub(r"\s+", "", gap_text)
    if len(gap_non_ws) > MAX_GAP_FILL_NON_WS_CHARS:
        return None
    aligned_text = source_text[first_start:second_end]
    source_sha256 = str(ordered[0].get("evidence_source_sha256") or "")
    if not source_sha256:
        source_sha256 = str(ordered[1].get("evidence_source_sha256") or "")
    return {
        "evidence_exact_match": False,
        "evidence_verification_status": "fuzzy_aligned",
        "evidence_text": " ".join(str(span.get("evidence_text") or "") for span in spans if span.get("evidence_text")),
        "evidence_aligned_text": aligned_text,
        "evidence_source_field": field,
        "evidence_start": first_start,
        "evidence_end": second_end,
        "evidence_source_sha256": source_sha256,
        "evidence_alignment_score": min(float(span.get("evidence_alignment_score") or 0.0) for span in ordered),
        "gap_filled_evidence": True,
        "gap_filled_evidence_gap_text": gap_text,
        "gap_filled_evidence_gap_char_count": len(gap_text),
        "gap_filled_evidence_gap_non_ws_char_count": len(gap_non_ws),
    }


def _evidence_fragments(evidence: str) -> list[str]:
    raw_parts = re.split(r"\.{3,}|…+|\n+|\\n+|[；;]+", str(evidence or ""))
    fragments: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        for candidate in _sentence_fragment_candidates(part):
            cleaned = _clean_evidence_fragment(candidate)
            if not _useful_fragment(cleaned) or cleaned in seen:
                continue
            seen.add(cleaned)
            fragments.append(cleaned)
    return fragments


def _sentence_fragment_candidates(text: str) -> list[str]:
    cleaned = _clean_evidence_fragment(text)
    if not cleaned:
        return []
    candidates = [cleaned]
    parts = [part for part in re.split(r"(?<=[。！？!?])\s+", cleaned) if part.strip()]
    if len(parts) > 1:
        candidates.extend(parts)
    return candidates


def _clean_evidence_fragment(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r'^[,{\\[\]}\s"\']+', "", cleaned)
    cleaned = re.sub(r'[,{\[\]}\s"\']+$', "", cleaned)
    cleaned = re.sub(r"^(title|subtitle|content)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _useful_fragment(text: str) -> bool:
    if not text:
        return False
    if len(text) < 4 and _cjk_count(text) < 2:
        return False
    if text in {"content", "title", "subtitle"}:
        return False
    return True


def _looks_like_paraphrase_evidence(evidence: str) -> bool:
    lowered = str(evidence or "").lower()
    return any(term.lower() in lowered for term in PARAPHRASE_EVIDENCE_TERMS)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ordered_counts(counter: Counter[str]) -> dict[str, int]:
    ordered_keys = (
        "scene_count",
        "event_count",
        "relation_count",
        "scene_index_count",
        "warning_count",
        "time_hint_count",
        "evidence_checked_count",
        "evidence_exact_count",
        "evidence_aligned_count",
        "evidence_rejected_count",
        "evidence_not_found_count",
        "empty_evidence_count",
        "gap_filled_evidence_count",
        "elliptical_evidence_count",
        "paraphrase_evidence_count",
        "non_chinese_output_count",
        "event_evidence_issue_count",
        "relation_evidence_issue_count",
        "time_hint_issue_count",
        "hard_ordering_unusable_count",
        "issue_count",
    )
    return {key: int(counter.get(key, 0)) for key in ordered_keys}
