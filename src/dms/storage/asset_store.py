from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dms.entity_types import normalize_entity_type


SCHEMA_VERSION = 3


@dataclass(frozen=True)
class AssetStoreImportConfig:
    db_path: Path
    ordered_run_dir: Path
    summary_memory_dir: Path | None = None
    reset: bool = False


def init_asset_store(db_path: str | Path, *, reset: bool = False) -> dict[str, Any]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()

    with _connect(path) as conn:
        _apply_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
    return {"db_path": str(path), "schema_version": SCHEMA_VERSION}


def import_run_assets(config: AssetStoreImportConfig) -> dict[str, Any]:
    init_asset_store(config.db_path, reset=config.reset)
    ordered_run = Path(config.ordered_run_dir)
    summary_dir = Path(config.summary_memory_dir) if config.summary_memory_dir else ordered_run / "summaries"

    counts = {
        "entities": 0,
        "entity_aliases": 0,
        "episodic_memories": 0,
        "entity_memory_links": 0,
        "scene_summaries": 0,
        "unit_summaries": 0,
        "relationships": 0,
        "scene_metadata": 0,
        "stated_fact_documents": 0,
        "retrieval_documents": 0,
    }

    with _connect(config.db_path) as conn:
        entity_id_map = _import_entities(conn, ordered_run, counts)
        _import_aliases(conn, ordered_run, entity_id_map, counts)
        _import_relationships(conn, ordered_run, entity_id_map, counts)
        _import_memories(conn, ordered_run, counts)
        _import_entity_memory_links(conn, ordered_run, entity_id_map, counts)
        _import_scene_metadata(conn, ordered_run, counts)
        _import_summaries(conn, summary_dir, counts)
        conn.commit()

    return {
        "db_path": str(config.db_path),
        "ordered_run_dir": str(ordered_run),
        "summary_memory_dir": str(summary_dir) if summary_dir else None,
        **counts,
    }


def get_entity_memories(
    db_path: str | Path,
    *,
    entity_ref: str,
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        entity = _resolve_entity(conn, entity_ref)
        if entity is None:
            return []
        max_order = before_scene_order
        if max_order is None and before_scene_id:
            max_order = _max_scene_order_before(before_scene_id)
        sql = """
            SELECT
              m.memory_id,
              m.parent_scene_id,
              m.scene_order,
              m.chunk_index,
              m.sequence_index,
              m.timeline_index,
              m.memory_type,
              m.summary,
              m.evidence_text,
              m.evidence_start,
              m.evidence_end,
              m.parent_evidence_start,
              m.parent_evidence_end,
              l.link_role,
              l.entity_id,
              l.entity_name,
              l.entity_type
            FROM entity_memory_links l
            JOIN episodic_memories m ON m.memory_id = l.memory_id
            WHERE l.entity_id = ?
        """
        params: list[Any] = [entity["entity_id"]]
        if max_order is not None:
            sql += " AND l.scene_order <= ?"
            params.append(max_order)
        sql += " ORDER BY m.scene_order, m.chunk_index, m.sequence_index"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in conn.execute(sql, params)]


def list_entities(
    db_path: str | Path,
    *,
    entity_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        sql = "SELECT * FROM entities WHERE 1 = 1"
        params: list[Any] = []
        if entity_types:
            normalized_types = [normalize_entity_type(entity_type) for entity_type in entity_types]
            placeholders = ", ".join("?" for _ in normalized_types)
            sql += f" AND entity_type IN ({placeholders})"
            params.extend(normalized_types)
        sql += " ORDER BY first_seen_order, canonical_name"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_entity_payload(conn, row) for row in conn.execute(sql, params)]


def get_entity_by_id(db_path: str | Path, entity_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
        return _entity_payload(conn, row) if row else None


def resolve_entity_refs(
    db_path: str | Path,
    entity_refs: list[str],
    *,
    entity_types: list[str] | None = None,
    limit_per_ref: int = 3,
    min_score: float = 0.55,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        candidates = _entity_candidates(conn, entity_types=entity_types)

    matches: list[dict[str, Any]] = []
    for entity_ref in entity_refs:
        query = str(entity_ref or "").strip()
        if not query:
            continue
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            score, match_type, matched_alias = _entity_match_score(query, candidate)
            if score < min_score:
                continue
            scored.append(
                {
                    "query": query,
                    "entity_id": candidate["entity_id"],
                    "canonical_name": candidate["canonical_name"],
                    "entity_type": candidate["entity_type"],
                    "score": round(score, 4),
                    "match_type": match_type,
                    "matched_alias": matched_alias,
                    "aliases": candidate["aliases"],
                    "first_seen_scene": candidate["first_seen_scene"],
                    "first_seen_order": candidate["first_seen_order"],
                    "mention_count": candidate["mention_count"],
                }
            )
        scored.sort(
            key=lambda item: (
                float(item["score"]),
                int(item.get("mention_count") or 0),
                -(int(item.get("first_seen_order") or 0)),
            ),
            reverse=True,
        )
        matches.extend(scored[: max(limit_per_ref, 0)])
    return matches


def get_memories_by_ids(db_path: str | Path, memory_ids: list[str]) -> list[dict[str, Any]]:
    ordered_ids = [str(memory_id) for memory_id in memory_ids if str(memory_id or "").strip()]
    if not ordered_ids:
        return []
    placeholders = ", ".join("?" for _ in ordered_ids)
    with _connect(db_path) as conn:
        rows = {
            str(row["memory_id"]): dict(row)
            for row in conn.execute(
                f"SELECT * FROM episodic_memories WHERE memory_id IN ({placeholders})",
                ordered_ids,
            )
        }
    return [rows[memory_id] for memory_id in ordered_ids if memory_id in rows]


def get_one_hop_relationships(
    db_path: str | Path,
    *,
    entity_ids: list[str],
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    unique_ids = sorted({str(entity_id) for entity_id in entity_ids if str(entity_id or "").strip()})
    if not unique_ids:
        return []
    max_order = before_scene_order
    if max_order is None and before_scene_id:
        max_order = _max_scene_order_before(before_scene_id)
    placeholders = ", ".join("?" for _ in unique_ids)
    params: list[Any] = [*unique_ids, *unique_ids]
    sql = f"""
        SELECT
          r.*,
          source.canonical_name AS source_name,
          source.entity_type AS source_entity_type,
          target.canonical_name AS target_name,
          target.entity_type AS target_entity_type
        FROM relationships r
        LEFT JOIN entities source ON source.entity_id = r.source_entity_id
        LEFT JOIN entities target ON target.entity_id = r.target_entity_id
        WHERE (r.source_entity_id IN ({placeholders}) OR r.target_entity_id IN ({placeholders}))
    """
    if max_order is not None:
        sql += " AND r.last_updated_order <= ?"
        params.append(max_order)
    sql += " ORDER BY r.last_updated_order, r.relationship_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect(db_path) as conn:
        rows = []
        for row in conn.execute(sql, params):
            payload = dict(row)
            payload["evidence"] = _json_loads(payload.pop("evidence_json", "[]"), default=[])
            payload["raw"] = _json_loads(payload.pop("raw_json", "{}"), default={})
            rows.append(payload)
        return rows


def get_scene_metadata(db_path: str | Path, scene_ids: list[str]) -> dict[str, dict[str, Any]]:
    ordered_ids = [str(scene_id) for scene_id in scene_ids if str(scene_id or "").strip()]
    if not ordered_ids:
        return {}
    placeholders = ", ".join("?" for _ in ordered_ids)
    with _connect(db_path) as conn:
        rows = {}
        for row in conn.execute(
            f"SELECT * FROM scene_metadata WHERE scene_id IN ({placeholders})",
            ordered_ids,
        ):
            payload = dict(row)
            payload["setting"] = _json_loads(payload.pop("setting_json", "{}"), default={})
            payload["stated_facts"] = _json_loads(payload.pop("stated_facts_json", "[]"), default=[])
            payload["open_questions"] = _json_loads(payload.pop("open_questions_json", "[]"), default=[])
            payload["scene_tags"] = _json_loads(payload.pop("scene_tags_json", "[]"), default=[])
            payload["source"] = _json_loads(payload.pop("source_json", "{}"), default={})
            payload["raw"] = _json_loads(payload.pop("raw_json", "{}"), default={})
            rows[str(payload["scene_id"])] = payload
        return {scene_id: rows[scene_id] for scene_id in ordered_ids if scene_id in rows}


def get_relationship_count(db_path: str | Path) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM relationships").fetchone()
        return int(row["count"] or 0) if row else 0


def get_retrieval_documents(
    db_path: str | Path,
    *,
    doc_type: str | None = None,
    doc_types: list[str] | None = None,
    entity_ref: str | None = None,
    before_scene_order: int | None = None,
    before_scene_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        entity_id = None
        if entity_ref:
            entity = _resolve_entity(conn, entity_ref)
            if entity is None:
                return []
            entity_id = str(entity["entity_id"])

        max_order = before_scene_order
        if max_order is None and before_scene_id:
            max_order = _max_scene_order_before(before_scene_id)

        sql = "SELECT * FROM retrieval_documents WHERE 1 = 1"
        params: list[Any] = []
        if doc_types:
            placeholders = ", ".join("?" for _ in doc_types)
            sql += f" AND doc_type IN ({placeholders})"
            params.extend(doc_types)
        elif doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        if entity_id:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        if max_order is not None:
            sql += " AND scene_order <= ?"
            params.append(max_order)
        sql += " ORDER BY scene_order, chunk_index, sequence_index"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in conn.execute(sql, params)]


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            original_entity_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            original_entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            first_seen_scene TEXT,
            first_seen_order INTEGER,
            mention_count INTEGER NOT NULL DEFAULT 0,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entity_aliases (
            alias_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            source TEXT,
            FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
        );

        CREATE TABLE IF NOT EXISTS episodic_memories (
            memory_id TEXT PRIMARY KEY,
            parent_scene_id TEXT NOT NULL,
            scene_order INTEGER NOT NULL,
            chunk_id TEXT,
            chunk_index INTEGER NOT NULL DEFAULT 1,
            sequence_index INTEGER NOT NULL DEFAULT 1,
            timeline_index TEXT,
            memory_type TEXT,
            summary TEXT NOT NULL,
            evidence_text TEXT,
            evidence_start INTEGER,
            evidence_end INTEGER,
            parent_evidence_start INTEGER,
            parent_evidence_end INTEGER,
            source_sha256 TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entity_memory_links (
            link_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            link_role TEXT,
            parent_scene_id TEXT NOT NULL,
            scene_order INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 1,
            sequence_index INTEGER NOT NULL DEFAULT 1,
            timeline_index TEXT,
            evidence_text TEXT,
            raw_json TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES episodic_memories(memory_id)
        );

        CREATE TABLE IF NOT EXISTS scene_summaries (
            summary_id TEXT PRIMARY KEY,
            parent_scene_id TEXT NOT NULL,
            scene_order INTEGER NOT NULL,
            summary TEXT NOT NULL,
            retrieval_text TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS unit_summaries (
            summary_id TEXT PRIMARY KEY,
            unit_id TEXT NOT NULL,
            parent_scene_id TEXT NOT NULL,
            scene_order INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 1,
            summary TEXT NOT NULL,
            retrieval_text TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_documents (
            doc_id TEXT PRIMARY KEY,
            doc_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            memory_id TEXT,
            entity_id TEXT,
            parent_scene_id TEXT NOT NULL,
            scene_order INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 1,
            sequence_index INTEGER NOT NULL DEFAULT 1,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationships (
            relationship_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            direction TEXT,
            status TEXT,
            first_seen_scene TEXT,
            first_seen_order INTEGER NOT NULL DEFAULT 0,
            last_updated_scene TEXT,
            last_updated_order INTEGER NOT NULL DEFAULT 0,
            strength REAL,
            evidence_json TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scene_metadata (
            scene_id TEXT PRIMARY KEY,
            scene_order INTEGER NOT NULL,
            title TEXT,
            setting_json TEXT NOT NULL,
            stated_facts_json TEXT NOT NULL,
            open_questions_json TEXT NOT NULL,
            scene_tags_json TEXT NOT NULL,
            source_json TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_alias_normalized ON entity_aliases(normalized_alias);
        CREATE INDEX IF NOT EXISTS idx_entity_memory_time
            ON entity_memory_links(entity_id, scene_order, chunk_index, sequence_index);
        CREATE INDEX IF NOT EXISTS idx_memory_time
            ON episodic_memories(scene_order, chunk_index, sequence_index);
        CREATE INDEX IF NOT EXISTS idx_retrieval_scope
            ON retrieval_documents(doc_type, entity_id, scene_order);
        CREATE INDEX IF NOT EXISTS idx_relationship_source
            ON relationships(source_entity_id, last_updated_order);
        CREATE INDEX IF NOT EXISTS idx_relationship_target
            ON relationships(target_entity_id, last_updated_order);
        CREATE INDEX IF NOT EXISTS idx_scene_metadata_order
            ON scene_metadata(scene_order);
        """
    )


def _import_entities(conn: sqlite3.Connection, run_dir: Path, counts: dict[str, int]) -> dict[str, str]:
    path = run_dir / "knowledge_graph" / "entities.jsonl"
    mapping: dict[str, str] = {}
    for row in _read_jsonl(path):
        original_id = str(row.get("entity_id") or "")
        entity_type = normalize_entity_type(row.get("entity_type"))
        canonical = str(row.get("canonical_name") or original_id)
        entity_id = _normalized_entity_id(original_id, entity_type)
        mapping[original_id] = entity_id
        conn.execute(
            """
            INSERT OR REPLACE INTO entities
            (entity_id, original_entity_id, entity_type, original_entity_type, canonical_name,
             first_seen_scene, first_seen_order, mention_count, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                original_id,
                entity_type,
                str(row.get("entity_type") or entity_type),
                canonical,
                row.get("first_seen_scene"),
                _scene_order_from_id(row.get("first_seen_scene")),
                int(row.get("mention_count") or 0),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        counts["entities"] += 1
        _insert_entity_alias(conn, entity_id=entity_id, alias=canonical, source="canonical", counts=counts)
        for alias in row.get("aliases") or []:
            _insert_entity_alias(conn, entity_id=entity_id, alias=str(alias), source="entity_aliases", counts=counts)
    return mapping


def _import_aliases(conn: sqlite3.Connection, run_dir: Path, entity_id_map: dict[str, str], counts: dict[str, int]) -> None:
    path = run_dir / "knowledge_graph" / "aliases.jsonl"
    for row in _read_jsonl(path):
        original_entity_id = str(row.get("entity_id") or "")
        entity_id = entity_id_map.get(original_entity_id, original_entity_id)
        alias = str(row.get("alias") or "")
        _insert_entity_alias(
            conn,
            entity_id=entity_id,
            alias=alias,
            source=str(row.get("source") or "alias_registry"),
            counts=counts,
            normalized_alias=str(row.get("normalized_alias") or _normalize_alias(alias)),
        )


def _import_relationships(
    conn: sqlite3.Connection,
    run_dir: Path,
    entity_id_map: dict[str, str],
    counts: dict[str, int],
) -> None:
    path = run_dir / "knowledge_graph" / "relationships.jsonl"
    for row in _read_jsonl(path):
        relationship_id = str(row.get("relationship_id") or "")
        if not relationship_id:
            continue
        source_entity_id = entity_id_map.get(str(row.get("source_entity_id") or ""), str(row.get("source_entity_id") or ""))
        target_entity_id = entity_id_map.get(str(row.get("target_entity_id") or ""), str(row.get("target_entity_id") or ""))
        first_seen_scene = str(row.get("first_seen_scene") or "")
        last_updated_scene = str(row.get("last_updated_scene") or first_seen_scene)
        conn.execute(
            """
            INSERT OR REPLACE INTO relationships
            (relationship_id, source_entity_id, target_entity_id, relation_type, direction, status,
             first_seen_scene, first_seen_order, last_updated_scene, last_updated_order,
             strength, evidence_json, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relationship_id,
                source_entity_id,
                target_entity_id,
                str(row.get("relation_type") or ""),
                row.get("direction"),
                row.get("status"),
                first_seen_scene,
                _scene_order_from_id(first_seen_scene),
                last_updated_scene,
                _scene_order_from_id(last_updated_scene),
                float(row.get("strength") or 0.0),
                json.dumps(row.get("evidence") or [], ensure_ascii=False),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        counts["relationships"] += 1


def _import_memories(conn: sqlite3.Connection, run_dir: Path, counts: dict[str, int]) -> None:
    path = run_dir / "memories" / "episodic_memories.jsonl"
    for row in _read_jsonl(path):
        memory_id = str(row.get("record_id") or "")
        parent_scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        conn.execute(
            """
            INSERT OR REPLACE INTO episodic_memories
            (memory_id, parent_scene_id, scene_order, chunk_id, chunk_index, sequence_index,
             timeline_index, memory_type, summary, evidence_text, evidence_start, evidence_end,
             parent_evidence_start, parent_evidence_end, source_sha256, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                parent_scene_id,
                _scene_order_from_id(parent_scene_id),
                row.get("chunk_id") or row.get("unit_id"),
                int(row.get("chunk_index") or 1),
                int(row.get("sequence_index") or 1),
                row.get("timeline_index"),
                row.get("memory_type"),
                str(row.get("summary") or ""),
                row.get("evidence_text") or row.get("evidence"),
                _optional_int(row.get("evidence_start")),
                _optional_int(row.get("evidence_end")),
                _optional_int(row.get("parent_evidence_start")),
                _optional_int(row.get("parent_evidence_end")),
                row.get("parent_source_sha256") or row.get("evidence_source_sha256") or row.get("unit_source_sha256"),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        _insert_retrieval_document(
            conn,
            doc_id=f"memory:{memory_id}",
            doc_type="episodic_memory_global",
            source_id=memory_id,
            memory_id=memory_id,
            entity_id=None,
            parent_scene_id=parent_scene_id,
            scene_order=_scene_order_from_id(parent_scene_id),
            chunk_index=int(row.get("chunk_index") or 1),
            sequence_index=int(row.get("sequence_index") or 1),
            text=_memory_retrieval_text(row),
            metadata=row,
        )
        counts["retrieval_documents"] += 1
        counts["episodic_memories"] += 1


def _import_entity_memory_links(
    conn: sqlite3.Connection,
    run_dir: Path,
    entity_id_map: dict[str, str],
    counts: dict[str, int],
) -> None:
    alias_to_entity = _alias_lookup(conn)
    memory_sequences = _memory_sequence_lookup(conn)
    path = run_dir / "memories" / "entity_memory_links.jsonl"
    for row in _read_jsonl(path):
        memory_id = str(row.get("memory_record_id") or "")
        canonical = str(row.get("canonical_entity") or row.get("entity") or "")
        entity_id = _entity_id_for_name(canonical, alias_to_entity) or _entity_id_for_name(str(row.get("entity") or ""), alias_to_entity)
        entity_id = entity_id_map.get(entity_id, entity_id) if entity_id else None
        if not entity_id:
            continue
        parent_scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        sequence_index = memory_sequences.get(memory_id, int(row.get("sequence_index") or 1))
        entity_type = normalize_entity_type(row.get("entity_type"))
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_memory_links
            (link_id, entity_id, memory_id, entity_name, entity_type, link_role, parent_scene_id,
             scene_order, chunk_index, sequence_index, timeline_index, evidence_text, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row.get("record_id") or f"{memory_id}:{entity_id}"),
                entity_id,
                memory_id,
                canonical,
                entity_type,
                row.get("link_role"),
                parent_scene_id,
                _scene_order_from_id(parent_scene_id),
                int(row.get("chunk_index") or 1),
                sequence_index,
                f"{parent_scene_id}:{sequence_index:03d}",
                row.get("evidence_text") or row.get("evidence"),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        link_id = str(row.get("record_id") or f"{memory_id}:{entity_id}")
        doc_id = f"entity_memory:{link_id}"
        _insert_retrieval_document(
            conn,
            doc_id=doc_id,
            doc_type="episodic_memory_entity",
            source_id=link_id,
            memory_id=memory_id,
            entity_id=entity_id,
            parent_scene_id=parent_scene_id,
            scene_order=_scene_order_from_id(parent_scene_id),
            chunk_index=int(row.get("chunk_index") or 1),
            sequence_index=sequence_index,
            text=_entity_memory_retrieval_text(row, _memory_summary(conn, memory_id)),
            metadata=row,
        )
        counts["retrieval_documents"] += 1
        counts["entity_memory_links"] += 1


def _import_summaries(conn: sqlite3.Connection, summary_dir: Path, counts: dict[str, int]) -> None:
    for row in _read_jsonl(summary_dir / "scene_summaries.jsonl"):
        summary_id = str(row.get("record_id") or f"{row.get('scene_id')}_summary")
        parent_scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        conn.execute(
            """
            INSERT OR REPLACE INTO scene_summaries
            (summary_id, parent_scene_id, scene_order, summary, retrieval_text, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary_id,
                parent_scene_id,
                _scene_order_from_id(parent_scene_id),
                str(row.get("summary") or ""),
                str(row.get("retrieval_text") or row.get("summary") or ""),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        _insert_retrieval_document(
            conn,
            doc_id=f"scene_summary:{summary_id}",
            doc_type="scene_summary",
            source_id=summary_id,
            memory_id=None,
            entity_id=None,
            parent_scene_id=parent_scene_id,
            scene_order=_scene_order_from_id(parent_scene_id),
            chunk_index=1,
            sequence_index=1,
            text=str(row.get("retrieval_text") or row.get("summary") or ""),
            metadata=row,
        )
        counts["retrieval_documents"] += 1
        counts["scene_summaries"] += 1

    for row in _read_jsonl(summary_dir / "unit_summaries.jsonl"):
        summary_id = str(row.get("record_id") or f"{row.get('unit_id')}_summary")
        parent_scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        conn.execute(
            """
            INSERT OR REPLACE INTO unit_summaries
            (summary_id, unit_id, parent_scene_id, scene_order, chunk_index, summary, retrieval_text, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary_id,
                str(row.get("unit_id") or row.get("chunk_id") or summary_id),
                parent_scene_id,
                _scene_order_from_id(parent_scene_id),
                int(row.get("chunk_index") or 1),
                str(row.get("summary") or ""),
                str(row.get("retrieval_text") or row.get("summary") or ""),
                json.dumps(row, ensure_ascii=False),
            ),
        )
        _insert_retrieval_document(
            conn,
            doc_id=f"unit_summary:{summary_id}",
            doc_type="unit_summary",
            source_id=summary_id,
            memory_id=None,
            entity_id=None,
            parent_scene_id=parent_scene_id,
            scene_order=_scene_order_from_id(parent_scene_id),
            chunk_index=int(row.get("chunk_index") or 1),
            sequence_index=1,
            text=str(row.get("retrieval_text") or row.get("summary") or ""),
            metadata=row,
        )
        counts["retrieval_documents"] += 1
        counts["unit_summaries"] += 1


def _import_scene_metadata(conn: sqlite3.Connection, run_dir: Path, counts: dict[str, int]) -> None:
    scene_rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "_debug" / "chunk_manifest.jsonl"):
        scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        if not scene_id:
            continue
        record = scene_rows.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "scene_order": _scene_order_from_id(scene_id),
                "title": row.get("title"),
                "chunk_count": int(row.get("chunk_count") or 1),
                "character_count": 0,
                "source_record_id": row.get("source_record_id"),
                "discourse_index": row.get("discourse_index"),
                "source_sha256": row.get("source_sha256"),
            },
        )
        record["title"] = record.get("title") or row.get("title")
        record["chunk_count"] = max(int(record.get("chunk_count") or 1), int(row.get("chunk_count") or 1))
        record["character_count"] = int(record.get("character_count") or 0) + int(row.get("character_count") or 0)
        record["source_record_id"] = record.get("source_record_id") or row.get("source_record_id")
        record["discourse_index"] = record.get("discourse_index") or row.get("discourse_index")
        record["source_sha256"] = record.get("source_sha256") or row.get("source_sha256")

    settings_by_scene: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "scene_context" / "scenes.jsonl"):
        scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        if scene_id and scene_id not in settings_by_scene:
            settings_by_scene[scene_id] = row.get("setting") if isinstance(row.get("setting"), dict) else {}

    facts_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "scene_context" / "stated_facts.jsonl"):
        scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        if scene_id:
            facts_by_scene.setdefault(scene_id, []).append(
                {
                    "record_id": row.get("record_id"),
                    "proposition": row.get("proposition"),
                    "speaker_or_source": row.get("speaker_or_source") or "",
                    "evidence": row.get("evidence"),
                }
            )

    questions_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "scene_context" / "open_questions.jsonl"):
        scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        if scene_id:
            questions_by_scene.setdefault(scene_id, []).append(
                {
                    "record_id": row.get("record_id"),
                    "question": row.get("question"),
                    "evidence": row.get("evidence"),
                }
            )

    tags_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "scene_context" / "scene_tags.jsonl"):
        scene_id = str(row.get("parent_unit_id") or row.get("scene_id") or "")
        if scene_id:
            tags_by_scene.setdefault(scene_id, []).append(
                {
                    "record_id": row.get("record_id"),
                    "surface": row.get("surface"),
                    "tag_type": row.get("tag_type"),
                    "reason": row.get("reason"),
                    "evidence": row.get("evidence"),
                }
            )

    all_scene_ids = sorted(
        set(scene_rows) | set(settings_by_scene) | set(facts_by_scene) | set(questions_by_scene) | set(tags_by_scene),
        key=_scene_order_from_id,
    )
    for scene_id in all_scene_ids:
        source = scene_rows.get(scene_id, {"scene_id": scene_id, "scene_order": _scene_order_from_id(scene_id)})
        conn.execute(
            """
            INSERT OR REPLACE INTO scene_metadata
            (scene_id, scene_order, title, setting_json, stated_facts_json,
             open_questions_json, scene_tags_json, source_json, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scene_id,
                _scene_order_from_id(scene_id),
                source.get("title"),
                json.dumps(settings_by_scene.get(scene_id) or {}, ensure_ascii=False),
                json.dumps(facts_by_scene.get(scene_id) or [], ensure_ascii=False),
                json.dumps(questions_by_scene.get(scene_id) or [], ensure_ascii=False),
                json.dumps(tags_by_scene.get(scene_id) or [], ensure_ascii=False),
                json.dumps(source, ensure_ascii=False),
                json.dumps(
                    {
                        "setting": settings_by_scene.get(scene_id) or {},
                        "stated_facts": facts_by_scene.get(scene_id) or [],
                        "open_questions": questions_by_scene.get(scene_id) or [],
                        "scene_tags": tags_by_scene.get(scene_id) or [],
                        "source": source,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        counts["scene_metadata"] += 1
        for sequence_index, fact in enumerate(facts_by_scene.get(scene_id) or [], start=1):
            source_id = str(fact.get("record_id") or f"{scene_id}_fact_{sequence_index:03d}")
            text = _stated_fact_retrieval_text(fact)
            if not text:
                continue
            _insert_retrieval_document(
                conn,
                doc_id=f"stated_fact:{source_id}",
                doc_type="stated_fact",
                source_id=source_id,
                memory_id=None,
                entity_id=None,
                parent_scene_id=scene_id,
                scene_order=_scene_order_from_id(scene_id),
                chunk_index=1,
                sequence_index=sequence_index,
                text=text,
                metadata=fact,
            )
            counts["retrieval_documents"] += 1
            counts["stated_fact_documents"] += 1


def _insert_retrieval_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    doc_type: str,
    source_id: str,
    memory_id: str | None,
    entity_id: str | None,
    parent_scene_id: str,
    scene_order: int,
    chunk_index: int,
    sequence_index: int,
    text: str,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO retrieval_documents
        (doc_id, doc_type, source_id, memory_id, entity_id, parent_scene_id, scene_order,
         chunk_index, sequence_index, text, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            doc_type,
            source_id,
            memory_id,
            entity_id,
            parent_scene_id,
            scene_order,
            chunk_index,
            sequence_index,
            text,
            json.dumps(metadata, ensure_ascii=False),
        ),
    )


def _insert_entity_alias(
    conn: sqlite3.Connection,
    *,
    entity_id: str,
    alias: str,
    source: str,
    counts: dict[str, int],
    normalized_alias: str | None = None,
) -> None:
    alias = str(alias or "").strip()
    if not alias:
        return
    normalized = normalized_alias or _normalize_alias(alias)
    if not normalized:
        return
    alias_id = f"{entity_id}:{normalized}"
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO entity_aliases(alias_id, entity_id, alias, normalized_alias, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (alias_id, entity_id, alias, normalized, source),
    )
    if cursor.rowcount:
        counts["entity_aliases"] += 1


def _resolve_entity(conn: sqlite3.Connection, entity_ref: str) -> sqlite3.Row | None:
    ref = str(entity_ref or "").strip()
    if not ref:
        return None
    row = conn.execute("SELECT * FROM entities WHERE entity_id = ?", (ref,)).fetchone()
    if row:
        return row
    row = conn.execute(
        "SELECT * FROM entities WHERE canonical_name = ? COLLATE NOCASE LIMIT 1",
        (ref,),
    ).fetchone()
    if row:
        return row
    alias = _normalize_alias(ref)
    return conn.execute(
        """
        SELECT e.* FROM entity_aliases a
        JOIN entities e ON e.entity_id = a.entity_id
        WHERE a.normalized_alias = ?
        LIMIT 1
        """,
        (alias,),
    ).fetchone()


def _entity_payload(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["raw"] = _json_loads(payload.pop("raw_json", "{}"), default={})
    if isinstance(payload["raw"], dict):
        payload["initial_description"] = payload["raw"].get("initial_description", "")
        payload["author_description"] = payload["raw"].get("author_description", "")
        payload["descriptions"] = payload["raw"].get("descriptions", [])
        payload["description_sources"] = payload["raw"].get("description_sources", [])
    else:
        payload["initial_description"] = ""
        payload["author_description"] = ""
        payload["descriptions"] = []
        payload["description_sources"] = []
    payload["aliases"] = _aliases_for_entity(conn, str(payload["entity_id"]))
    return payload


def _aliases_for_entity(conn: sqlite3.Connection, entity_id: str) -> list[str]:
    return [
        str(row["alias"])
        for row in conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY source, alias",
            (entity_id,),
        )
    ]


def _entity_candidates(
    conn: sqlite3.Connection,
    *,
    entity_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM entities WHERE 1 = 1"
    params: list[Any] = []
    if entity_types:
        normalized_types = [normalize_entity_type(entity_type) for entity_type in entity_types]
        placeholders = ", ".join("?" for _ in normalized_types)
        sql += f" AND entity_type IN ({placeholders})"
        params.extend(normalized_types)
    return [_entity_payload(conn, row) for row in conn.execute(sql, params)]


def _entity_match_score(query: str, candidate: dict[str, Any]) -> tuple[float, str, str]:
    query = str(query or "").strip()
    if not query:
        return 0.0, "empty", ""
    if query == candidate["entity_id"]:
        return 1.0, "entity_id", str(candidate["entity_id"])

    normalized_query = _normalize_alias(query)
    match_query = _normalize_for_match(query)

    aliases = [str(candidate.get("canonical_name") or ""), *[str(alias) for alias in candidate.get("aliases") or []]]
    best_score = 0.0
    best_type = "none"
    best_alias = ""
    query_codes = set(_code_tokens(query))

    for alias in aliases:
        normalized_alias = _normalize_alias(alias)
        match_alias = _normalize_for_match(alias)
        if not match_alias:
            continue
        if normalized_query == normalized_alias:
            return 1.0, "exact_alias", alias
        if len(match_query) == 1:
            continue
        alias_codes = set(_code_tokens(alias))
        if query_codes and query_codes.intersection(alias_codes):
            score = 0.95
            match_type = "code_token"
        elif match_query in match_alias or match_alias in match_query:
            coverage = min(len(match_query), len(match_alias)) / max(len(match_query), len(match_alias))
            score = 0.72 + 0.22 * coverage
            match_type = "substring"
        else:
            score = SequenceMatcher(None, match_query, match_alias).ratio()
            match_type = "fuzzy"
        if score > best_score:
            best_score = score
            best_type = match_type
            best_alias = alias
    return best_score, best_type, best_alias


def _alias_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    return {str(row["normalized_alias"]): str(row["entity_id"]) for row in conn.execute("SELECT * FROM entity_aliases")}


def _memory_sequence_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row["memory_id"]): int(row["sequence_index"])
        for row in conn.execute("SELECT memory_id, sequence_index FROM episodic_memories")
    }


def _memory_summary(conn: sqlite3.Connection, memory_id: str) -> str:
    row = conn.execute("SELECT summary FROM episodic_memories WHERE memory_id = ?", (memory_id,)).fetchone()
    return str(row["summary"]) if row else ""


def _entity_id_for_name(name: str, alias_to_entity: dict[str, str]) -> str | None:
    return alias_to_entity.get(_normalize_alias(name))


def _normalized_entity_id(original_id: str, entity_type: str) -> str:
    return original_id or f"{entity_type}_unknown"


def _scene_order_from_id(scene_id: Any) -> int:
    text = str(scene_id or "")
    for part in text.split("_"):
        if part.isdigit():
            return int(part)
    return 0


def _max_scene_order_before(scene_id: Any) -> int:
    order = _scene_order_from_id(scene_id)
    return max(order - 1, 0) if order else 0


def _memory_retrieval_text(row: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            str(row.get("summary") or ""),
            str(row.get("evidence") or row.get("evidence_text") or ""),
            str(row.get("memory_type") or ""),
        )
        if part
    )


def _entity_memory_retrieval_text(row: dict[str, Any], memory_summary: str) -> str:
    return "\n".join(
        part
        for part in (
            str(row.get("canonical_entity") or row.get("entity") or ""),
            str(row.get("link_role") or ""),
            memory_summary,
            str(row.get("evidence") or row.get("evidence_text") or ""),
        )
        if part
    )


def _stated_fact_retrieval_text(row: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            str(row.get("proposition") or ""),
            str(row.get("evidence") or ""),
            str(row.get("speaker_or_source") or ""),
        )
        if part
    )


def _normalize_alias(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _normalize_for_match(value: str) -> str:
    return "".join(
        char.lower()
        for char in str(value or "")
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _code_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]*\d+[A-Za-z0-9]*|[A-Z]{2,}", str(value or ""))]


def _json_loads(value: Any, *, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn
