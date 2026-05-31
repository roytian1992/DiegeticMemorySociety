from __future__ import annotations

import re


DURABLE_RELATION_TYPES = {
    "ally",
    "alliance",
    "enemy",
    "hostile_to",
    "rival",
    "kinship",
    "family",
    "parent_of",
    "child_of",
    "sibling_of",
    "spouse_of",
    "romantic_partner",
    "friend",
    "trusts",
    "distrusts",
    "mentor",
    "mentorship",
    "mentor_of",
    "teacher_of",
    "instructor_of",
    "student_of",
    "trainee_of",
    "apprentice_of",
    "teammate",
    "teammate_of",
    "mission_partner",
    "pilot_partner",
    "comrade",
    "comrade_of",
    "leader_of",
    "subordinate_of",
    "member_of",
    "belongs_to",
    "owns",
    "owned_by",
    "custodian_of",
    "custody",
    "guardian",
    "guardian_of",
    "ward_of",
    "care_commitment_to",
    "care_responsibility_for",
    "guidance_responsibility_for",
    "protective_responsibility_for",
    "protected_by",
    "protects",
    "responsible_for",
    "duty_to",
    "affiliated_with",
    "works_for",
    "commands",
    "reports_to",
    "dependent_on",
    "long_term_dependency",
}

RELATION_TYPE_ALIASES = {
    "ally": "alliance",
    "mentor": "mentor_of",
    "mentorship": "mentor_of",
    "teacher": "mentor_of",
    "teacher_of": "mentor_of",
    "instructor": "mentor_of",
    "instructor_of": "mentor_of",
    "student": "student_of",
    "trainee": "student_of",
    "trainee_of": "student_of",
    "apprentice": "student_of",
    "apprentice_of": "student_of",
    "guardian": "guardian_of",
    "custodian": "custodian_of",
    "care_commitment": "care_commitment_to",
    "care_commitment_for": "care_commitment_to",
    "care_responsibility": "care_responsibility_for",
    "protective_responsibility": "protective_responsibility_for",
    "team_mate": "teammate",
    "teammates": "teammate",
    "teammate_of": "teammate",
    "mission_partners": "mission_partner",
    "pilot_partners": "pilot_partner",
    "comrades": "comrade",
    "comrade_of": "comrade",
    "romantic_bond": "romantic_partner",
}

INVERSE_RELATION_TYPE_ALIASES = {
    "ward_of": "guardian_of",
    "student_of": "mentor_of",
    "trainee_of": "mentor_of",
    "apprentice_of": "mentor_of",
    "child_of": "parent_of",
    "owned_by": "owns",
    "protected_by": "protects",
    "subordinate_of": "leader_of",
}

UNDIRECTED_RELATION_TYPES = {
    "alliance",
    "enemy",
    "rival",
    "kinship",
    "family",
    "sibling_of",
    "spouse_of",
    "romantic_partner",
    "friend",
    "teammate",
    "mission_partner",
    "pilot_partner",
    "comrade",
    "affiliated_with",
}

MOMENTARY_RELATION_TERMS = {
    "address",
    "addresses",
    "analyzes",
    "analyzing",
    "analyzes_brainwave_of",
    "review",
    "reviews",
    "reviewing",
    "discuss",
    "discusses",
    "demonstrate",
    "demonstrates",
    "demonstrator_to_subject",
    "demonstrator-to-subject",
    "uses_as_model",
    "uses_or_interacts_with",
    "speaks_to",
    "speaking_to",
    "observes",
    "looks_at",
    "connected_to",
    "displayed_on",
    "part_of",
}


def is_durable_relation_type(relation_type: object) -> bool:
    clean = normalize_relation_type(relation_type)
    if not clean:
        return False
    if clean in MOMENTARY_RELATION_TERMS:
        return False
    if clean in DURABLE_RELATION_TYPES:
        return True
    return any(term in clean for term in DURABLE_RELATION_TYPES) and not any(
        term in clean for term in MOMENTARY_RELATION_TERMS
    )


def normalize_relation_type(relation_type: object) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", str(relation_type or "").strip().lower()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return RELATION_TYPE_ALIASES.get(clean, clean)


def is_undirected_relation_type(relation_type: object) -> bool:
    return normalize_relation_type(relation_type) in UNDIRECTED_RELATION_TYPES


def canonicalize_relation_type(relation_type: object) -> tuple[str, bool]:
    clean = normalize_relation_type(relation_type)
    if clean in INVERSE_RELATION_TYPE_ALIASES:
        return INVERSE_RELATION_TYPE_ALIASES[clean], True
    return clean, False


def soften_formal_relation_type(
    relation_type: object,
    *,
    evidence: object = "",
    status_or_change: object = "",
) -> str:
    """Avoid upgrading care or guidance evidence into formal legal/work roles."""

    clean = normalize_relation_type(relation_type)
    text = f"{evidence or ''}\n{status_or_change or ''}".lower()
    if clean in {"guardian_of", "custody", "custodian_of"}:
        if _has_care_language(text) and not _has_explicit_guardian_language(text):
            return "care_commitment_to"
    if clean in {"mentor_of", "teacher_of", "instructor_of"}:
        if not _has_explicit_mentor_language(text):
            return "care_responsibility_for" if _has_care_language(text) else "guidance_responsibility_for"
    return clean


def _has_care_language(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "照顾",
            "看护",
            "照看",
            "关照",
            "保护",
            "take care",
            "look after",
            "care for",
            "protect",
        )
    )


def _has_explicit_guardian_language(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "监护",
            "法定监护",
            "监护人",
            "被监护",
            "抚养权",
            "guardian",
            "legal guardian",
            "ward",
            "custody",
            "custodian",
        )
    )


def _has_explicit_mentor_language(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "导师",
            "老师",
            "教师",
            "教官",
            "师父",
            "师傅",
            "徒弟",
            "学生",
            "学员",
            "mentor",
            "teacher",
            "instructor",
            "student",
            "trainee",
            "apprentice",
        )
    )
