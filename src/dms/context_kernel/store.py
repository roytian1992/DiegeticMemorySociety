from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dms.context_kernel.schema import (
    CreativeContextItem,
    CreativeScope,
    EntityPatch,
    EvidenceRef,
    SourceRecord,
    SourceUnit,
    ContextLink,
    HISTORY_EVENTS,
    STATUS_VALUES,
    utc_now,
    stable_id,
)


SCHEMA_VERSION = 1


class CreativeContextStore:
    """Authoritative sidecar store for source-aware creative context items."""

    def __init__(self, db_path: str | Path, *, reset: bool = False) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.db_path.exists():
            self.db_path.unlink()
        with self._connect() as conn:
            _apply_schema(conn)
            conn.execute(
                "INSERT OR REPLACE INTO context_metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            conn.commit()

    def upsert_source(self, source: SourceRecord) -> dict[str, Any]:
        record = source.normalized()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_sources
                (source_id, project_id, source_type, title, status, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.source_id,
                    record.project_id,
                    record.source_type,
                    record.title,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    _json_dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.model_dump()

    def upsert_unit(self, unit: SourceUnit) -> dict[str, Any]:
        record = unit.normalized()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_units
                (unit_id, source_id, project_id, source_type, unit_type, unit_order,
                 speaker, text, start_offset, end_offset, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.unit_id,
                    record.source_id,
                    record.project_id,
                    record.source_type,
                    record.unit_type,
                    record.unit_order,
                    record.speaker,
                    record.text,
                    record.start_offset,
                    record.end_offset,
                    _json_dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.model_dump()

    def add_item(
        self,
        item: CreativeContextItem,
        *,
        actor: str = "system",
        reason: str = "",
        write_history: bool = True,
    ) -> dict[str, Any]:
        record = item.normalized()
        with self._connect() as conn:
            existing = _get_item_row(conn, record.item_id)
            existing_payload = _item_payload(conn, existing) if existing else None
            _upsert_item(conn, record)
            for evidence in record.evidence_refs:
                _upsert_evidence(conn, evidence)
            _upsert_retrieval_document(conn, record)
            if write_history:
                event = "UPDATE" if existing else "ADD"
                _add_history(
                    conn,
                    item_id=record.item_id,
                    event=event,
                    old_value=existing_payload,
                    new_value=record.model_dump(),
                    actor=actor,
                    reason=reason,
                )
            conn.commit()
        return record.model_dump()

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = _get_item_row(conn, item_id)
            return _item_payload(conn, row) if row else None

    def update_item(
        self,
        item_id: str,
        patch: dict[str, Any],
        *,
        actor: str = "system",
        reason: str = "",
    ) -> dict[str, Any]:
        existing = self.get_item(item_id)
        if existing is None:
            raise ValueError(f"Context item not found: {item_id}")
        protected = {
            "item_id",
            "project_id",
            "source_type",
            "source_id",
            "unit_id",
            "created_at",
        }
        merged = dict(existing)
        for key, value in patch.items():
            if key in protected:
                continue
            merged[key] = value
        merged["updated_at"] = utc_now()
        item = CreativeContextItem(
            item_id=merged["item_id"],
            project_id=merged["project_id"],
            source_type=merged["source_type"],
            source_id=merged["source_id"],
            unit_id=merged.get("unit_id"),
            item_type=merged["item_type"],
            subject=merged.get("subject") or "",
            statement=merged["statement"],
            entity_ids=tuple(merged.get("entity_ids") or ()),
            evidence_refs=tuple(EvidenceRef(**ref) for ref in merged.get("evidence_refs") or ()),
            authority=merged.get("authority") or "model_inferred",
            authority_score=merged.get("authority_score"),
            confidence=float(merged.get("confidence") or 0.0),
            status=merged.get("status") or "active",
            visibility=merged.get("visibility") or "author_only",
            temporal_scope=merged.get("temporal_scope") or "not_applicable",
            embedding_text=merged.get("embedding_text") or "",
            payload=merged.get("payload") or {},
            created_at=merged.get("created_at"),
            updated_at=merged.get("updated_at"),
        )
        return self.add_item(item, actor=actor, reason=reason, write_history=True)

    def set_status(
        self,
        item_id: str,
        status: str,
        *,
        actor: str = "system",
        reason: str = "",
    ) -> dict[str, Any]:
        if status not in STATUS_VALUES:
            raise ValueError(f"Unsupported status: {status}")
        event = {
            "superseded": "SUPERSEDE",
            "rejected": "REJECT",
            "archived": "ARCHIVE",
            "deleted": "DELETE",
        }.get(status, "UPDATE")
        existing = self.get_item(item_id)
        if existing is None:
            raise ValueError(f"Context item not found: {item_id}")
        updated = self.update_item(item_id, {"status": status}, actor=actor, reason=reason)
        with self._connect() as conn:
            _add_history(
                conn,
                item_id=item_id,
                event=event,
                old_value=existing,
                new_value=updated,
                actor=actor,
                reason=reason,
            )
            conn.commit()
        return updated

    def delete_item(self, item_id: str, *, actor: str = "system", reason: str = "", soft: bool = True) -> dict[str, Any]:
        existing = self.get_item(item_id)
        if existing is None:
            raise ValueError(f"Context item not found: {item_id}")
        if soft:
            return self.set_status(item_id, "deleted", actor=actor, reason=reason)
        with self._connect() as conn:
            conn.execute("DELETE FROM context_retrieval_documents WHERE item_id = ?", (item_id,))
            conn.execute("DELETE FROM context_evidence WHERE item_id = ?", (item_id,))
            conn.execute("DELETE FROM context_items WHERE item_id = ?", (item_id,))
            _add_history(
                conn,
                item_id=item_id,
                event="DELETE",
                old_value=existing,
                new_value=None,
                actor=actor,
                reason=reason,
            )
            conn.commit()
        return existing

    def add_link(
        self,
        source_item_id: str,
        target_item_id: str,
        link_type: str,
        *,
        evidence: str = "",
        created_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        link = ContextLink(
            link_id=stable_id("link", source_item_id, target_item_id, link_type, evidence),
            source_item_id=source_item_id,
            target_item_id=target_item_id,
            link_type=link_type,
            evidence=evidence,
            created_by=created_by,
            metadata=metadata or {},
        ).normalized()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_links
                (link_id, source_item_id, target_item_id, link_type, created_by,
                 evidence, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.link_id,
                    link.source_item_id,
                    link.target_item_id,
                    link.link_type,
                    link.created_by,
                    link.evidence,
                    link.created_at,
                    _json_dumps(link.metadata),
                ),
            )
            _add_history(
                conn,
                item_id=source_item_id,
                event="LINK",
                old_value=None,
                new_value=link.model_dump(),
                actor=created_by,
                reason=f"{link_type} -> {target_item_id}",
            )
            conn.commit()
        return link.model_dump()

    def promote_item(
        self,
        item_id: str,
        *,
        target_layer: str,
        authority: str = "user_confirmed",
        actor: str = "user",
        reason: str = "",
        target_item_id: str | None = None,
        status: str = "canonical",
    ) -> dict[str, Any]:
        source = self.get_item(item_id)
        if source is None:
            raise ValueError(f"Context item not found: {item_id}")
        promoted_id = target_item_id or stable_id("ctx_item", item_id, target_layer, source.get("statement"))
        promoted = CreativeContextItem(
            item_id=promoted_id,
            project_id=source["project_id"],
            source_type="narrative_artifact",
            source_id=str(source.get("source_id") or item_id),
            unit_id=source.get("unit_id"),
            item_type=source["item_type"],
            subject=source.get("subject") or "",
            statement=source["statement"],
            entity_ids=tuple(source.get("entity_ids") or ()),
            evidence_refs=tuple(EvidenceRef(**ref) for ref in source.get("evidence_refs") or ()),
            authority=authority,
            confidence=float(source.get("confidence") or 0.0),
            status=status,
            visibility=source.get("visibility") or "author_only",
            temporal_scope=source.get("temporal_scope") or "not_applicable",
            payload={
                **(source.get("payload") or {}),
                "promoted_from": item_id,
                "target_layer": target_layer,
                "promotion_reason": reason,
            },
        )
        result = self.add_item(promoted, actor=actor, reason=reason)
        self.add_link(item_id, promoted_id, "promotes", evidence=reason, created_by=actor, metadata={"target_layer": target_layer})
        with self._connect() as conn:
            _add_history(
                conn,
                item_id=item_id,
                event="PROMOTE",
                old_value=source,
                new_value=result,
                actor=actor,
                reason=reason or f"promoted to {target_layer}",
            )
            conn.commit()
        return result

    def add_entity_patch(self, patch: EntityPatch, *, actor: str = "system", reason: str = "") -> dict[str, Any]:
        record = patch.normalized()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entity_patches
                (patch_id, project_id, entity_id, source_item_id, patch_type, target_field,
                 patch_statement, authority, status, applies_to, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.patch_id,
                    record.project_id,
                    record.entity_id,
                    record.source_item_id,
                    record.patch_type,
                    record.target_field,
                    record.patch_statement,
                    record.authority,
                    record.status,
                    record.applies_to,
                    record.created_at,
                    _json_dumps(record.metadata),
                ),
            )
            _add_history(
                conn,
                item_id=record.source_item_id,
                event="PATCH",
                old_value=None,
                new_value=record.model_dump(),
                actor=actor,
                reason=reason,
            )
            conn.commit()
        return record.model_dump()

    def list_entity_patches(
        self,
        *,
        project_id: str,
        entity_id: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entity_patches WHERE project_id = ?"
        params: list[Any] = [project_id]
        if entity_id:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        if active_only:
            sql += " AND status = 'active'"
        sql += " ORDER BY created_at, patch_id"
        with self._connect() as conn:
            return [_patch_payload(row) for row in conn.execute(sql, params)]

    def search_items(
        self,
        query: str,
        *,
        scope: CreativeScope,
        source_types: list[str] | None = None,
        item_types: list[str] | None = None,
        statuses: list[str] | None = None,
        entity_ids: list[str] | None = None,
        visibility: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        normalized_scope = scope.normalized()
        terms = _tokens(query)
        sql = "SELECT * FROM context_retrieval_documents WHERE project_id = ?"
        params: list[Any] = [normalized_scope.project_id]
        if source_types:
            sql += f" AND source_type IN ({_placeholders(source_types)})"
            params.extend(source_types)
        elif normalized_scope.source_type:
            sql += " AND source_type = ?"
            params.append(normalized_scope.source_type)
        if item_types:
            sql += f" AND item_type IN ({_placeholders(item_types)})"
            params.extend(item_types)
        if statuses:
            sql += f" AND status IN ({_placeholders(statuses)})"
            params.extend(statuses)
        else:
            sql += " AND status IN ('active', 'canonical')"
        if visibility:
            sql += f" AND visibility IN ({_placeholders(visibility)})"
            params.extend(visibility)
        if normalized_scope.before_unit_order is not None:
            sql += " AND (unit_order IS NULL OR unit_order <= ?)"
            params.append(normalized_scope.before_unit_order)
        sql += " ORDER BY updated_at DESC, doc_id"

        requested_entities = set(entity_ids or normalized_scope.entity_ids or ())
        ranked = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                payload = _retrieval_doc_payload(row)
                row_entities = set(payload.get("entity_ids") or [])
                if requested_entities and not requested_entities.intersection(row_entities):
                    continue
                score = _lexical_score(query_terms=terms, payload=payload)
                if requested_entities and requested_entities.intersection(row_entities):
                    score += 2.0
                payload["score"] = round(score, 4)
                ranked.append(payload)
        ranked.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                _status_rank(item.get("status")),
                float(item.get("confidence") or 0.0),
                str(item.get("updated_at") or ""),
            ),
            reverse=True,
        )
        if terms:
            ranked = [item for item in ranked if float(item.get("score") or 0.0) > 0 or requested_entities]
        return ranked[: max(int(top_k or 0), 0)]

    def list_retrieval_documents(
        self,
        *,
        scope: CreativeScope,
        source_types: list[str] | None = None,
        item_types: list[str] | None = None,
        statuses: list[str] | None = None,
        entity_ids: list[str] | None = None,
        visibility: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        normalized_scope = scope.normalized()
        sql = "SELECT * FROM context_retrieval_documents WHERE project_id = ?"
        params: list[Any] = [normalized_scope.project_id]
        if source_types:
            sql += f" AND source_type IN ({_placeholders(source_types)})"
            params.extend(source_types)
        elif normalized_scope.source_type:
            sql += " AND source_type = ?"
            params.append(normalized_scope.source_type)
        if item_types:
            sql += f" AND item_type IN ({_placeholders(item_types)})"
            params.extend(item_types)
        if statuses:
            sql += f" AND status IN ({_placeholders(statuses)})"
            params.extend(statuses)
        else:
            sql += " AND status IN ('active', 'canonical', 'tentative')"
        if visibility:
            sql += f" AND visibility IN ({_placeholders(visibility)})"
            params.extend(visibility)
        if normalized_scope.before_unit_order is not None:
            sql += " AND (unit_order IS NULL OR unit_order <= ?)"
            params.append(normalized_scope.before_unit_order)
        sql += " ORDER BY updated_at DESC, doc_id"

        requested_entities = set(entity_ids or normalized_scope.entity_ids or ())
        documents = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                payload = _retrieval_doc_payload(row)
                row_entities = set(payload.get("entity_ids") or [])
                if requested_entities and not requested_entities.intersection(row_entities):
                    continue
                documents.append(payload)
                if len(documents) >= max(int(limit or 0), 0):
                    break
        return documents

    def build_entity_view(self, *, project_id: str, entity_id: str) -> dict[str, Any]:
        scope = CreativeScope(project_id=project_id, entity_ids=(entity_id,))
        items = self.search_items(
            "",
            scope=scope,
            entity_ids=[entity_id],
            statuses=["active", "canonical", "tentative"],
            top_k=200,
        )
        layers = {
            "artifact_layer": [],
            "conversation_layer": [],
            "external_layer": [],
            "simulation_layer": [],
        }
        for item in items:
            source_type = item.get("source_type")
            if source_type == "narrative_artifact":
                layers["artifact_layer"].append(item)
            elif source_type == "conversation":
                layers["conversation_layer"].append(item)
            elif source_type == "external_reference":
                layers["external_layer"].append(item)
            elif source_type == "simulation":
                layers["simulation_layer"].append(item)
        return {
            "project_id": project_id,
            "entity_id": entity_id,
            **layers,
            "active_patches": self.list_entity_patches(project_id=project_id, entity_id=entity_id),
        }

    def history(self, item_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM context_history
                WHERE item_id = ?
                ORDER BY created_at, history_id
                """,
                (item_id,),
            ).fetchall()
            return [_history_payload(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS context_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_sources (
            source_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_units (
            unit_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            unit_type TEXT NOT NULL,
            unit_order INTEGER,
            speaker TEXT,
            text TEXT,
            start_offset INTEGER,
            end_offset INTEGER,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_items (
            item_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            unit_id TEXT,
            item_type TEXT NOT NULL,
            subject TEXT,
            statement TEXT NOT NULL,
            entity_ids_json TEXT NOT NULL,
            authority TEXT NOT NULL,
            authority_score REAL,
            confidence REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            visibility TEXT NOT NULL,
            temporal_scope TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_evidence (
            evidence_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            unit_id TEXT,
            text TEXT,
            start_offset INTEGER,
            end_offset INTEGER,
            alignment_status TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_links (
            link_id TEXT PRIMARY KEY,
            source_item_id TEXT NOT NULL,
            target_item_id TEXT NOT NULL,
            link_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            evidence TEXT,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_history (
            history_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            event TEXT NOT NULL,
            old_value_json TEXT,
            new_value_json TEXT,
            actor TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entity_patches (
            patch_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_item_id TEXT NOT NULL,
            patch_type TEXT NOT NULL,
            target_field TEXT NOT NULL,
            patch_statement TEXT NOT NULL,
            authority TEXT NOT NULL,
            status TEXT NOT NULL,
            applies_to TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS context_retrieval_documents (
            doc_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            unit_id TEXT,
            unit_order INTEGER,
            item_type TEXT NOT NULL,
            subject TEXT,
            statement TEXT NOT NULL,
            entity_ids_json TEXT NOT NULL,
            status TEXT NOT NULL,
            authority TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            visibility TEXT NOT NULL,
            temporal_scope TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_context_items_project_source
            ON context_items(project_id, source_type, status);
        CREATE INDEX IF NOT EXISTS idx_context_retrieval_scope
            ON context_retrieval_documents(project_id, source_type, item_type, status, unit_order);
        CREATE INDEX IF NOT EXISTS idx_context_links_source
            ON context_links(source_item_id, link_type);
        CREATE INDEX IF NOT EXISTS idx_entity_patches_entity
            ON entity_patches(project_id, entity_id, status);
        """
    )


def _upsert_item(conn: sqlite3.Connection, item: CreativeContextItem) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO context_items
        (item_id, project_id, source_type, source_id, unit_id, item_type, subject,
         statement, entity_ids_json, authority, authority_score, confidence, status,
         visibility, temporal_scope, embedding_text, payload_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.item_id,
            item.project_id,
            item.source_type,
            item.source_id,
            item.unit_id,
            item.item_type,
            item.subject,
            item.statement,
            _json_dumps(list(item.entity_ids)),
            item.authority,
            item.authority_score,
            item.confidence,
            item.status,
            item.visibility,
            item.temporal_scope,
            item.embedding_text,
            _json_dumps(item.payload),
            item.created_at,
            item.updated_at,
        ),
    )


def _upsert_evidence(conn: sqlite3.Connection, evidence: EvidenceRef) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO context_evidence
        (evidence_id, item_id, source_id, unit_id, text, start_offset, end_offset,
         alignment_status, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence.evidence_id,
            evidence.item_id,
            evidence.source_id,
            evidence.unit_id,
            evidence.text,
            evidence.start_offset,
            evidence.end_offset,
            evidence.alignment_status,
            _json_dumps(evidence.metadata),
        ),
    )


def _upsert_retrieval_document(conn: sqlite3.Connection, item: CreativeContextItem) -> None:
    unit_order = None
    if item.unit_id:
        unit = conn.execute("SELECT unit_order FROM context_units WHERE unit_id = ?", (item.unit_id,)).fetchone()
        if unit:
            unit_order = unit["unit_order"]
    metadata = {
        "payload": item.payload,
        "evidence_refs": [ref.model_dump() for ref in item.evidence_refs],
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO context_retrieval_documents
        (doc_id, item_id, project_id, source_type, source_id, unit_id, unit_order,
         item_type, subject, statement, entity_ids_json, status, authority, confidence,
         visibility, temporal_scope, text, metadata_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"context:{item.item_id}",
            item.item_id,
            item.project_id,
            item.source_type,
            item.source_id,
            item.unit_id,
            unit_order,
            item.item_type,
            item.subject,
            item.statement,
            _json_dumps(list(item.entity_ids)),
            item.status,
            item.authority,
            item.confidence,
            item.visibility,
            item.temporal_scope,
            item.embedding_text,
            _json_dumps(metadata),
            item.updated_at,
        ),
    )


def _add_history(
    conn: sqlite3.Connection,
    *,
    item_id: str,
    event: str,
    old_value: dict[str, Any] | None,
    new_value: dict[str, Any] | None,
    actor: str,
    reason: str,
) -> None:
    if event not in HISTORY_EVENTS:
        raise ValueError(f"Unsupported history event: {event}")
    created_at = utc_now()
    conn.execute(
        """
        INSERT INTO context_history
        (history_id, item_id, event, old_value_json, new_value_json, actor, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stable_id("hist", item_id, event, created_at),
            item_id,
            event,
            _json_dumps(old_value) if old_value is not None else None,
            _json_dumps(new_value) if new_value is not None else None,
            actor or "system",
            reason or "",
            created_at,
        ),
    )


def _get_item_row(conn: sqlite3.Connection, item_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM context_items WHERE item_id = ?", (item_id,)).fetchone()


def _item_payload(conn: sqlite3.Connection, row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["entity_ids"] = _json_loads(payload.pop("entity_ids_json", "[]"), [])
    payload["payload"] = _json_loads(payload.pop("payload_json", "{}"), {})
    evidence_rows = conn.execute(
        """
        SELECT * FROM context_evidence
        WHERE item_id = ?
        ORDER BY evidence_id
        """,
        (payload["item_id"],),
    ).fetchall()
    payload["evidence_refs"] = [_evidence_payload(evidence_row) for evidence_row in evidence_rows]
    return payload


def _evidence_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["metadata"] = _json_loads(payload.pop("metadata_json", "{}"), {})
    return payload


def _retrieval_doc_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["entity_ids"] = _json_loads(payload.pop("entity_ids_json", "[]"), [])
    metadata = _json_loads(payload.pop("metadata_json", "{}"), {})
    payload["metadata"] = metadata
    payload["payload"] = metadata.get("payload") or {}
    payload["evidence_refs"] = metadata.get("evidence_refs") or []
    return payload


def _patch_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["metadata"] = _json_loads(payload.pop("metadata_json", "{}"), {})
    return payload


def _history_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["old_value"] = _json_loads(payload.pop("old_value_json", None), None)
    payload["new_value"] = _json_loads(payload.pop("new_value_json", None), None)
    return payload


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _placeholders(values: list[Any]) -> str:
    return ", ".join("?" for _ in values)


def _tokens(text: str) -> set[str]:
    normalized = str(text or "").lower()
    tokens = set()
    current = []
    for char in normalized:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            current.append(char)
        else:
            if current:
                token = "".join(current)
                tokens.add(token)
                if any("\u4e00" <= c <= "\u9fff" for c in token):
                    tokens.update(token[index : index + 2] for index in range(0, max(len(token) - 1, 0)))
                current = []
    if current:
        token = "".join(current)
        tokens.add(token)
        if any("\u4e00" <= c <= "\u9fff" for c in token):
            tokens.update(token[index : index + 2] for index in range(0, max(len(token) - 1, 0)))
    return {token for token in tokens if token}


def _lexical_score(*, query_terms: set[str], payload: dict[str, Any]) -> float:
    if not query_terms:
        return 0.0
    text = "\n".join(
        str(payload.get(key) or "")
        for key in ("subject", "statement", "text")
    )
    terms = _tokens(text)
    overlap = len(query_terms.intersection(terms))
    confidence = float(payload.get("confidence") or 0.0)
    status_bonus = _status_rank(payload.get("status")) * 0.25
    source_bonus = {
        "conversation": 0.4,
        "narrative_artifact": 0.35,
        "external_reference": 0.25,
        "simulation": 0.1,
    }.get(str(payload.get("source_type") or ""), 0.0)
    return float(overlap) + confidence * 0.2 + status_bonus + source_bonus


def _status_rank(status: Any) -> int:
    return {
        "canonical": 5,
        "active": 4,
        "tentative": 2,
        "extracted": 1,
        "superseded": -1,
        "rejected": -2,
        "archived": -3,
        "deleted": -4,
    }.get(str(status or ""), 0)
