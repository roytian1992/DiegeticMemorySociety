from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dms.context_kernel.kernel import CreativeMemoryKernel
from dms.context_kernel.schema import CreativeContextItem, EntityPatch, stable_id


RECONCILIATION_ACTIONS = {"ADD", "UPDATE", "DELETE", "NONE", "PATCH", "PROMOTE"}


@dataclass(frozen=True)
class ReconciliationDecision:
    action: str
    candidate_item: CreativeContextItem
    target_item_id: str | None = None
    reason: str = ""
    patch: EntityPatch | None = None
    promote_target_layer: str | None = None

    def normalized(self) -> "ReconciliationDecision":
        action = str(self.action or "").strip().upper()
        if action not in RECONCILIATION_ACTIONS:
            raise ValueError(f"Unsupported reconciliation action: {self.action}")
        return ReconciliationDecision(
            action=action,
            candidate_item=self.candidate_item.normalized(),
            target_item_id=str(self.target_item_id).strip() if self.target_item_id else None,
            reason=str(self.reason or "").strip(),
            patch=self.patch.normalized() if self.patch else None,
            promote_target_layer=str(self.promote_target_layer).strip() if self.promote_target_layer else None,
        )

    def model_dump(self) -> dict[str, Any]:
        record = self.normalized()
        return {
            "action": record.action,
            "candidate_item": record.candidate_item.model_dump(),
            "target_item_id": record.target_item_id,
            "reason": record.reason,
            "patch": record.patch.model_dump() if record.patch else None,
            "promote_target_layer": record.promote_target_layer,
        }


def reconcile_items_deterministic(
    kernel: CreativeMemoryKernel,
    candidates: list[CreativeContextItem],
    *,
    actor: str = "system",
) -> list[dict[str, Any]]:
    """Apply a deterministic ADD/UPDATE/NONE/PATCH/PROMOTE fallback.

    This is deliberately conservative:
    - exact same source/item_type/subject/statement => NONE
    - same source/item_type/subject but different statement => ADD
    - conversation correction with entity links => PATCH
    - user-confirmed creative decision with canonical payload hint => PROMOTE
    """

    results = []
    for candidate in candidates:
        decision = decide_reconciliation_deterministic(kernel, candidate)
        results.append(apply_reconciliation_decision(kernel, decision, actor=actor))
    return results


def decide_reconciliation_deterministic(
    kernel: CreativeMemoryKernel,
    candidate: CreativeContextItem,
) -> ReconciliationDecision:
    item = candidate.normalized()
    existing = kernel.search(
        item.statement,
        scope=_scope_for_item(item),
        source_types=[item.source_type],
        item_types=[item.item_type],
        statuses=["active", "canonical", "tentative"],
        entity_ids=list(item.entity_ids),
        top_k=20,
    )
    for hit in existing:
        if (
            str(hit.get("source_type")) == item.source_type
            and str(hit.get("item_type")) == item.item_type
            and str(hit.get("subject") or "") == item.subject
            and str(hit.get("statement") or "") == item.statement
        ):
            return ReconciliationDecision(
                action="NONE",
                candidate_item=item,
                target_item_id=hit.get("item_id"),
                reason="exact duplicate active item",
            )
    if item.source_type == "conversation" and item.item_type == "correction" and item.entity_ids:
        patch = EntityPatch(
            patch_id=stable_id("patch", item.item_id, item.entity_ids[0], item.statement),
            project_id=item.project_id,
            entity_id=item.entity_ids[0],
            source_item_id=item.item_id,
            patch_type="constrain",
            target_field="profile",
            patch_statement=item.statement,
            authority=item.authority,
            status="active",
            applies_to="project",
            metadata={"reconciliation": "deterministic_patch"},
        )
        return ReconciliationDecision(
            action="PATCH",
            candidate_item=item,
            reason="conversation correction creates entity overlay",
            patch=patch,
        )
    if item.source_type == "conversation" and item.item_type == "creative_decision" and item.payload.get("promote_to"):
        return ReconciliationDecision(
            action="PROMOTE",
            candidate_item=item,
            reason="candidate requests explicit promotion",
            promote_target_layer=str(item.payload["promote_to"]),
        )
    return ReconciliationDecision(action="ADD", candidate_item=item, reason="new context item")


def apply_reconciliation_decision(
    kernel: CreativeMemoryKernel,
    decision: ReconciliationDecision,
    *,
    actor: str = "system",
) -> dict[str, Any]:
    record = decision.normalized()
    if record.action == "NONE":
        return {"action": "NONE", "target_item_id": record.target_item_id, "reason": record.reason}
    if record.action == "ADD":
        item = kernel.add_item(record.candidate_item, actor=actor, reason=record.reason)
        return {"action": "ADD", "item": item, "reason": record.reason}
    if record.action == "UPDATE":
        if not record.target_item_id:
            raise ValueError("UPDATE requires target_item_id")
        item = kernel.update(
            record.target_item_id,
            record.candidate_item.model_dump(),
            actor=actor,
            reason=record.reason,
        )
        return {"action": "UPDATE", "item": item, "reason": record.reason}
    if record.action == "DELETE":
        if not record.target_item_id:
            raise ValueError("DELETE requires target_item_id")
        item = kernel.delete(record.target_item_id, actor=actor, reason=record.reason)
        return {"action": "DELETE", "item": item, "reason": record.reason}
    if record.action == "PATCH":
        item = kernel.add_item(record.candidate_item, actor=actor, reason=record.reason)
        patch = record.patch
        if patch is None:
            raise ValueError("PATCH requires patch payload")
        patch_result = kernel.add_entity_patch(patch, actor=actor, reason=record.reason)
        return {"action": "PATCH", "item": item, "patch": patch_result, "reason": record.reason}
    if record.action == "PROMOTE":
        item = kernel.add_item(record.candidate_item, actor=actor, reason=record.reason)
        promoted = kernel.promote(
            item["item_id"],
            record.promote_target_layer or "story_bible",
            actor=actor,
            reason=record.reason,
        )
        return {"action": "PROMOTE", "item": item, "promoted": promoted, "reason": record.reason}
    raise ValueError(f"Unhandled reconciliation action: {record.action}")


def _scope_for_item(item: CreativeContextItem):
    from dms.context_kernel.schema import CreativeScope

    return CreativeScope(project_id=item.project_id, entity_ids=item.entity_ids, source_type=item.source_type)

