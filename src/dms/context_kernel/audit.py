from __future__ import annotations

import re
from typing import Any

from dms.context_kernel.kernel import CreativeMemoryKernel
from dms.context_kernel.schema import CreativeScope


NEGATION_TERMS = {
    "不",
    "不是",
    "没有",
    "不能",
    "不可",
    "不要",
    "avoid",
    "not",
    "never",
    "no",
}


def audit_external_vs_artifact_conflicts(
    kernel: CreativeMemoryKernel,
    *,
    scope: CreativeScope,
    top_k: int = 100,
) -> dict[str, Any]:
    external_items = kernel.list_retrieval_documents(
        scope=scope,
        source_types=["external_reference"],
        statuses=["active", "canonical", "tentative"],
        limit=top_k,
    )
    artifact_items = kernel.list_retrieval_documents(
        scope=scope,
        source_types=["narrative_artifact"],
        statuses=["active", "canonical"],
        limit=top_k * 5,
    )
    conflicts = []
    for external in external_items:
        for artifact in artifact_items:
            score = _conflict_score(external, artifact)
            if score <= 0:
                continue
            conflicts.append(
                {
                    "external_item_id": external.get("item_id"),
                    "artifact_item_id": artifact.get("item_id"),
                    "subject": external.get("subject") or artifact.get("subject"),
                    "item_type": external.get("item_type"),
                    "score": score,
                    "external_statement": external.get("statement"),
                    "artifact_statement": artifact.get("statement"),
                    "review_policy": "manual_review_required; do not auto-promote external reference",
                }
            )
    conflicts.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return {
        "project_id": scope.project_id,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "trace": {
            "external_count": len(external_items),
            "artifact_count": len(artifact_items),
            "method": "deterministic subject/entity overlap + negation/similarity heuristic",
        },
    }


def export_external_entity_links(
    kernel: CreativeMemoryKernel,
    *,
    scope: CreativeScope,
    top_k: int = 1000,
) -> dict[str, Any]:
    external_items = kernel.list_retrieval_documents(
        scope=scope,
        source_types=["external_reference"],
        statuses=["active", "canonical", "tentative"],
        limit=top_k,
    )
    links = []
    for item in external_items:
        entities = list(item.get("entity_ids") or [])
        if not entities and str(item.get("subject") or "").strip():
            entities = [f"external_subject:{item.get('subject')}"]
        for entity_id in entities:
            links.append(
                {
                    "item_id": item.get("item_id"),
                    "entity_id": entity_id,
                    "subject": item.get("subject"),
                    "source_id": item.get("source_id"),
                    "item_type": item.get("item_type"),
                    "statement": item.get("statement"),
                    "status": "candidate",
                    "review_policy": "external link candidate; not canonical by default",
                }
            )
    return {"project_id": scope.project_id, "link_count": len(links), "links": links}


def export_external_timeline_claims(
    kernel: CreativeMemoryKernel,
    *,
    scope: CreativeScope,
    top_k: int = 1000,
) -> dict[str, Any]:
    items = kernel.list_retrieval_documents(
        scope=scope,
        source_types=["external_reference"],
        item_types=["timeline_doc", "timeline_event"],
        statuses=["active", "canonical", "tentative"],
        limit=top_k,
    )
    claims = []
    for item in items:
        payload = item.get("payload") or {}
        timeline_hint = payload.get("timeline_hint") or _extract_timeline_hint(str(item.get("statement") or ""))
        claims.append(
            {
                "item_id": item.get("item_id"),
                "subject": item.get("subject"),
                "timeline_hint": timeline_hint,
                "statement": item.get("statement"),
                "source_id": item.get("source_id"),
                "status": "candidate",
                "review_policy": "timeline claim candidate; require promotion/linking before canonical timeline use",
            }
        )
    return {"project_id": scope.project_id, "claim_count": len(claims), "claims": claims}


def _conflict_score(external: dict[str, Any], artifact: dict[str, Any]) -> float:
    if not _same_subject_or_entity(external, artifact):
        return 0.0
    external_type = str(external.get("item_type") or "")
    artifact_type = str(artifact.get("item_type") or "")
    compatible_fact_types = {"fact", "state", "character_profile", "world_bible"}
    if external_type != artifact_type and not ({external_type, artifact_type} <= compatible_fact_types):
        return 0.0
    external_text = str(external.get("statement") or "")
    artifact_text = str(artifact.get("statement") or "")
    overlap = _token_overlap(external_text, artifact_text)
    if overlap <= 0:
        return 0.0
    external_negated = _has_negation(external_text)
    artifact_negated = _has_negation(artifact_text)
    if external_negated != artifact_negated:
        return 0.6 + min(overlap, 0.35)
    contradiction_markers = ("冲突", "矛盾", "contradict", "conflict")
    if any(marker in external_text.lower() or marker in artifact_text.lower() for marker in contradiction_markers):
        return 0.5 + min(overlap, 0.3)
    return 0.0


def _same_subject_or_entity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_entities = set(str(item) for item in a.get("entity_ids") or [])
    b_entities = set(str(item) for item in b.get("entity_ids") or [])
    if a_entities and b_entities and a_entities.intersection(b_entities):
        return True
    a_subject = str(a.get("subject") or "").strip().lower()
    b_subject = str(b.get("subject") or "").strip().lower()
    return bool(a_subject and b_subject and a_subject == b_subject)


def _has_negation(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in NEGATION_TERMS)


def _token_overlap(a: str, b: str) -> float:
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens.intersection(b_tokens)) / max(len(a_tokens.union(b_tokens)), 1)


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk_chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    tokens.update(cjk_chars)
    tokens.update(
        lowered[index : index + 2]
        for index in range(max(len(lowered) - 1, 0))
        if any("\u4e00" <= char <= "\u9fff" for char in lowered[index : index + 2])
    )
    return {token for token in tokens if token.strip()}


def _extract_timeline_hint(text: str) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", text)
    if match:
        return match.group(0)
    match = re.search(r"(?:第|scene|chapter)\s*[\w一二三四五六七八九十百千零〇]+", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""
