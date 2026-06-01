from __future__ import annotations

import re
from typing import Any


THERAPY_LIKE_PHRASES = (
    "别跟地球赌气",
    "跟地球赌气",
    "和自己和解",
    "你是在逃避",
    "不要逃避",
    "放下过去",
    "放下执念",
)

UNSUPPORTED_FORMAL_ROLE_TERMS = (
    "监护人",
    "法定监护",
    "亲属",
    "叔叔",
    "舅舅",
    "教官",
    "老师",
    "上级",
    "下属",
    "指挥官",
    "正式飞行员",
    "正式航天员",
    "老兵",
)

FUTURE_LEAK_PATTERNS = (
    "后续剧情",
    "后面的场景",
    "下一场",
    "未来会",
    "后来会",
    "目标场景原文",
)

_REF_RE = re.compile(r"^[MR]\d+$")
_NEGATION_HINTS = ("避免", "不要", "不能", "无证据", "不应", "禁止", "风险", "不反推", "不外推", "不是", "不作为")
_SPEAKER_COLON_RE = re.compile(r"(^|[\s\n“”])[\u4e00-\u9fffA-Za-z0-9_]{1,6}[：:][^：:]{2,}")


def verify_social_simulation(
    *,
    cards: list[dict[str, Any]],
    character_simulations: list[dict[str, Any]],
    social_simulation: dict[str, Any],
    writing_intent: str = "",
) -> dict[str, Any]:
    """Verify current social simulation artifacts against basic hard/soft rules."""

    allowed_refs = collect_refs(cards)
    explicit_role_terms = _collect_explicit_role_terms(cards)
    checked_refs: list[str] = []
    unknown_refs: list[dict[str, Any]] = []
    hard_violations: list[dict[str, Any]] = []
    soft_warnings: list[dict[str, Any]] = []

    for path, refs in _iter_ref_lists(
        {
            "character_simulations": character_simulations,
            "social_simulation": social_simulation,
        }
    ):
        for ref in refs:
            checked_refs.append(ref)
            if allowed_refs and _REF_RE.match(ref) and ref not in allowed_refs:
                issue = {
                    "type": "unknown_memory_ref",
                    "path": path,
                    "ref": ref,
                    "detail": f"Reference {ref} is not present in the attribute-card evidence.",
                }
                unknown_refs.append(issue)
                hard_violations.append(issue)

    for path, text in _iter_strings(
        {
            "character_simulations": character_simulations,
            "social_simulation": social_simulation,
            "writing_intent": writing_intent,
        }
    ):
        risks = detect_text_risks(text, explicit_role_terms=explicit_role_terms)
        for phrase in risks["therapy_like_phrases"]:
            soft_warnings.append(
                {
                    "type": "therapy_phrase_risk",
                    "path": path,
                    "phrase": phrase,
                    "detail": "Use concrete scene action or dialogue posture instead of modern psychological paraphrase.",
                }
            )
        for term in risks["unsupported_formal_role_terms"]:
            soft_warnings.append(
                {
                    "type": "unsupported_formal_role_risk",
                    "path": path,
                    "term": term,
                    "detail": "Formal role wording is not explicitly supported by attribute-card role evidence.",
                }
            )
        for phrase in risks["future_leak_phrases"]:
            hard_violations.append(
                {
                    "type": "future_or_target_leak_risk",
                    "path": path,
                    "phrase": phrase,
                    "detail": "The simulation should not refer to future scenes or target-reference text.",
                }
            )
        if _looks_like_final_dialogue(text) and _is_writer_facing_path(path):
            soft_warnings.append(
                {
                    "type": "final_dialogue_like_guidance",
                    "path": path,
                    "detail": "Writer-facing guidance should express dialogue posture, not canonical final dialogue.",
                }
            )

    metrics = {
        "checked_ref_count": len(checked_refs),
        "unknown_ref_count": len(unknown_refs),
        "hard_violation_count": len(hard_violations),
        "soft_warning_count": len(soft_warnings),
        "therapy_phrase_risk_count": sum(1 for item in soft_warnings if item.get("type") == "therapy_phrase_risk"),
        "unsupported_role_risk_count": sum(
            1 for item in soft_warnings if item.get("type") == "unsupported_formal_role_risk"
        ),
        "final_dialogue_like_guidance_count": sum(
            1 for item in soft_warnings if item.get("type") == "final_dialogue_like_guidance"
        ),
    }
    status = "fail" if hard_violations else "warn" if soft_warnings else "pass"
    return {
        "status": status,
        "allowed_refs": sorted(allowed_refs),
        "ref_checks": {
            "checked_refs": checked_refs,
            "unknown_refs": unknown_refs,
        },
        "hard_violations": hard_violations,
        "soft_warnings": soft_warnings,
        "metrics": metrics,
    }


def verify_writer_packet(writer_packet: dict[str, Any]) -> dict[str, Any]:
    hard_violations: list[dict[str, Any]] = []
    soft_warnings: list[dict[str, Any]] = []
    for path, text in _iter_strings(writer_packet):
        risks = detect_text_risks(text)
        for phrase in risks["therapy_like_phrases"]:
            soft_warnings.append(
                {
                    "type": "therapy_phrase_risk",
                    "path": path,
                    "phrase": phrase,
                    "detail": "Writer packet should avoid therapy-like wording and express concrete action/posture.",
                }
            )
        for term in risks["unsupported_formal_role_terms"]:
            soft_warnings.append(
                {
                    "type": "unsupported_formal_role_risk",
                    "path": path,
                    "term": term,
                    "detail": "Writer packet should not introduce unsupported formal role wording.",
                }
            )
        if _looks_like_final_dialogue(text):
            soft_warnings.append(
                {
                    "type": "final_dialogue_like_writer_packet",
                    "path": path,
                    "detail": "Writer packet should contain posture and action guidance, not canonical final dialogue.",
                }
            )
    for path, value in _iter_not_final_dialogue_flags(writer_packet):
        if value is not True:
            soft_warnings.append(
                {
                    "type": "missing_not_final_dialogue_flag",
                    "path": path,
                    "detail": "Action or dialogue guidance should explicitly mark itself as not final dialogue.",
                }
            )
    metrics = {
        "hard_violation_count": len(hard_violations),
        "soft_warning_count": len(soft_warnings),
        "therapy_phrase_risk_count": sum(1 for item in soft_warnings if item.get("type") == "therapy_phrase_risk"),
        "unsupported_role_risk_count": sum(
            1 for item in soft_warnings if item.get("type") == "unsupported_formal_role_risk"
        ),
        "final_dialogue_like_count": sum(
            1 for item in soft_warnings if item.get("type") == "final_dialogue_like_writer_packet"
        ),
        "missing_not_final_dialogue_flag_count": sum(
            1 for item in soft_warnings if item.get("type") == "missing_not_final_dialogue_flag"
        ),
    }
    return {
        "status": "fail" if hard_violations else "warn" if soft_warnings else "pass",
        "hard_violations": hard_violations,
        "soft_warnings": soft_warnings,
        "metrics": metrics,
    }


def detect_text_risks(text: Any, *, explicit_role_terms: set[str] | None = None) -> dict[str, list[str]]:
    raw = str(text or "")
    explicit = explicit_role_terms or set()
    therapy = [phrase for phrase in THERAPY_LIKE_PHRASES if phrase in raw]
    future = [phrase for phrase in FUTURE_LEAK_PATTERNS if phrase in raw]
    formal_roles: list[str] = []
    if not _is_negated_or_risk_text(raw):
        for term in UNSUPPORTED_FORMAL_ROLE_TERMS:
            if term in raw and term not in explicit:
                formal_roles.append(term)
    return {
        "therapy_like_phrases": therapy,
        "unsupported_formal_role_terms": formal_roles,
        "future_leak_phrases": future,
    }


def collect_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    _collect_refs(value, refs)
    return refs


def _collect_refs(value: Any, refs: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"refs", "memory_basis"} and isinstance(child, list):
                refs.update(str(ref) for ref in child if _REF_RE.match(str(ref)))
            else:
                _collect_refs(child, refs)
    elif isinstance(value, list):
        for item in value:
            _collect_refs(item, refs)


def _collect_explicit_role_terms(cards: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for card in cards:
        for item in card.get("role_in_story") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            if status != "explicit":
                continue
            value = str(item.get("value") or "")
            for term in UNSUPPORTED_FORMAL_ROLE_TERMS:
                if term in value:
                    terms.add(term)
    return terms


def _iter_ref_lists(value: Any, path: str = "$") -> list[tuple[str, list[str]]]:
    refs: list[tuple[str, list[str]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"refs", "memory_basis"} and isinstance(child, list):
                string_refs = [str(item) for item in child if _REF_RE.match(str(item))]
                if string_refs:
                    refs.append((child_path, string_refs))
                for index, item in enumerate(child):
                    if isinstance(item, (dict, list)):
                        refs.extend(_iter_ref_lists(item, f"{child_path}[{index}]"))
            else:
                refs.extend(_iter_ref_lists(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            refs.extend(_iter_ref_lists(child, f"{path}[{index}]"))
    return refs


def _iter_strings(value: Any, path: str = "$") -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            strings.extend(_iter_strings(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            strings.extend(_iter_strings(child, f"{path}[{index}]"))
    elif isinstance(value, str) and value.strip():
        strings.append((path, value))
    return strings


def _iter_not_final_dialogue_flags(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    flags: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "not_final_dialogue":
                flags.append((child_path, child))
            else:
                flags.extend(_iter_not_final_dialogue_flags(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flags.extend(_iter_not_final_dialogue_flags(child, f"{path}[{index}]"))
    return flags


def _looks_like_final_dialogue(text: str) -> bool:
    if "“" in text or "”" in text:
        return True
    if _SPEAKER_COLON_RE.search(text):
        return True
    return False


def _is_writer_facing_path(path: str) -> bool:
    return any(part in path for part in (".writer_guidance", ".scene_beats"))


def _is_negated_or_risk_text(text: str) -> bool:
    return any(hint in text for hint in _NEGATION_HINTS)
