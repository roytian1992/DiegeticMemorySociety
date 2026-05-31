from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dms.entity_alignment import align_entity_to_candidates, build_entity_candidate_index
from dms.memory.unit_metadata import parent_evidence_span, unit_metadata
from dms.relationship_types import canonicalize_relation_type, is_durable_relation_type, soften_formal_relation_type
from dms.source_evidence import locate_evidence


def build_relationship_memory(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    require_entity_candidates: bool = True,
) -> dict[str, Any]:
    """Build source-grounded durable relationship observations from parsed outputs."""

    run_path = Path(run_dir)
    out_path = Path(output_dir)
    parsed_dir = run_path / "parsed"
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"Parsed dir not found: {parsed_dir}")

    out_path.mkdir(parents=True, exist_ok=True)
    relationships_path = out_path / "relationship_observations.jsonl"
    rejections_path = out_path / "relationship_rejections.jsonl"
    summary_path = out_path / "summary.json"

    counts = {
        "parsed_files": 0,
        "accepted_scene_count": 0,
        "skipped_scene_count": 0,
        "relationship_observation_count": 0,
        "skipped_relationship_observation_count": 0,
        "skipped_non_durable_relationship_count": 0,
        "skipped_unresolved_endpoint_count": 0,
        "skipped_duplicate_relationship_count": 0,
        "rejected_relationship_observation_count": 0,
    }
    evidence_counts: dict[str, int] = {"exact": 0, "fuzzy_aligned": 0, "rejected": 0}
    accepted_keys: set[tuple[str, str, str, str]] = set()

    with (
        relationships_path.open("w", encoding="utf-8") as relationships_handle,
        rejections_path.open("w", encoding="utf-8") as rejections_handle,
    ):
        for parsed_file in sorted(parsed_dir.glob("*.json")):
            counts["parsed_files"] += 1
            payload = _read_json(parsed_file)
            if payload.get("status") != "parsed" or not isinstance(payload.get("data"), dict):
                counts["skipped_scene_count"] += 1
                continue

            data = payload["data"]
            scene_id = _record_scene_id(data, payload, parsed_file)
            unit_payload = _unit_payload_for_run(run_path, scene_id)
            metadata = unit_metadata(unit_payload, scene_id)
            candidate_index = build_entity_candidate_index(_extracted_candidates_for_run(run_path, scene_id))
            counts["accepted_scene_count"] += 1

            for index, item in enumerate(_as_list(data.get("relationship_observations")), start=1):
                record_id = f"{scene_id}_relationship_obs_{index:03d}"
                if not isinstance(item, dict):
                    counts["skipped_relationship_observation_count"] += 1
                    _write_rejection(
                        rejections_handle,
                        scene_id=scene_id,
                        record_id=record_id,
                        reason="relationship observation must be an object",
                        item=item,
                    )
                    continue

                relation_type, reverse_endpoints = canonicalize_relation_type(item.get("relation_type"))
                relation_type = soften_formal_relation_type(
                    relation_type,
                    evidence=item.get("evidence", ""),
                    status_or_change=item.get("status_or_change", ""),
                )
                if not is_durable_relation_type(relation_type):
                    counts["skipped_relationship_observation_count"] += 1
                    counts["skipped_non_durable_relationship_count"] += 1
                    _write_rejection(
                        rejections_handle,
                        scene_id=scene_id,
                        record_id=record_id,
                        reason="relation type is not durable",
                        item=item,
                    )
                    continue

                model_source_entity = item.get("source_entity", "")
                model_target_entity = item.get("target_entity", "")
                source_entity = model_target_entity if reverse_endpoints else model_source_entity
                target_entity = model_source_entity if reverse_endpoints else model_target_entity

                source = align_entity_to_candidates(
                    entity=source_entity,
                    entity_type=item.get("source_entity_type", ""),
                    candidate_index=candidate_index,
                )
                target = align_entity_to_candidates(
                    entity=target_entity,
                    entity_type=item.get("target_entity_type", ""),
                    candidate_index=candidate_index,
                )
                if require_entity_candidates and (source is None or target is None):
                    counts["skipped_relationship_observation_count"] += 1
                    counts["skipped_unresolved_endpoint_count"] += 1
                    _write_rejection(
                        rejections_handle,
                        scene_id=scene_id,
                        record_id=record_id,
                        reason="relationship endpoint did not align to extracted entity candidates",
                        item=item,
                    )
                    continue

                evidence_record = _verified_evidence(item.get("evidence", ""), unit_payload)
                if _evidence_rejected(evidence_record):
                    _count_evidence_status(evidence_counts, evidence_record)
                    counts["skipped_relationship_observation_count"] += 1
                    counts["rejected_relationship_observation_count"] += 1
                    _write_rejection(
                        rejections_handle,
                        scene_id=scene_id,
                        record_id=record_id,
                        reason="evidence did not align to a contiguous source span",
                        item=item,
                        evidence_record=evidence_record,
                    )
                    continue

                canonical_source = _candidate_name(source, source_entity)
                canonical_target = _candidate_name(target, target_entity)
                dedupe_key = (
                    canonical_source,
                    canonical_target,
                    relation_type,
                    str(evidence_record.get("evidence") or ""),
                )
                if dedupe_key in accepted_keys:
                    counts["skipped_relationship_observation_count"] += 1
                    counts["skipped_duplicate_relationship_count"] += 1
                    _write_rejection(
                        rejections_handle,
                        scene_id=scene_id,
                        record_id=record_id,
                        reason="duplicate canonical relationship observation",
                        item=item,
                        evidence_record=evidence_record,
                    )
                    continue
                accepted_keys.add(dedupe_key)
                _count_evidence_status(evidence_counts, evidence_record)

                record = {
                    "memory_layer": "relationship_observation",
                    "scene_id": scene_id,
                    "unit_id": scene_id,
                    **metadata,
                    "record_id": record_id,
                    "source_entity": canonical_source,
                    "target_entity": canonical_target,
                    "source_model_entity": model_source_entity,
                    "target_model_entity": model_target_entity,
                    "relation_type": relation_type,
                    "model_relation_type": item.get("relation_type", ""),
                    "status_or_change": item.get("status_or_change", ""),
                    **evidence_record,
                }
                _write_jsonl(relationships_handle, record)
                counts["relationship_observation_count"] += 1

    summary = {
        "source_run_dir": str(run_path),
        "output_dir": str(out_path),
        "artifact_paths": {
            "relationship_observations": str(relationships_path),
            "relationship_rejections": str(rejections_path),
            "summary": str(summary_path),
        },
        "evidence_verification_counts": evidence_counts,
        "rejected_evidence_count": evidence_counts.get("rejected", 0),
        **counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _candidate_name(candidate: dict[str, Any] | None, fallback: object) -> str:
    if not candidate:
        return str(fallback or "")
    return str(candidate.get("canonical_name") or candidate.get("surface") or fallback or "")


def _verified_evidence(evidence: Any, unit_payload: dict[str, Any] | None) -> dict[str, Any]:
    location = locate_evidence(evidence, unit_payload or {})
    aligned = str(location.get("evidence_aligned_text") or "")
    original = str(location.get("evidence_text") or "")
    status = str(location.get("evidence_verification_status") or "rejected")
    return {
        "evidence": aligned if status in {"exact", "fuzzy_aligned"} and aligned else original,
        "model_evidence": original,
        **location,
        **parent_evidence_span(location, unit_payload),
    }


def _evidence_rejected(record: dict[str, Any]) -> bool:
    return str(record.get("evidence_verification_status") or "rejected") == "rejected"


def _count_evidence_status(counts: dict[str, int], record: dict[str, Any]) -> None:
    status = str(record.get("evidence_verification_status") or "rejected")
    counts[status] = counts.get(status, 0) + 1


def _unit_payload_for_run(run_path: Path, scene_id: str) -> dict[str, Any] | None:
    input_path = run_path / "inputs" / f"{scene_id}.json"
    if not input_path.is_file():
        return None
    payload = _read_json(input_path)
    unit = payload.get("unit") if isinstance(payload.get("unit"), dict) else payload
    return unit if isinstance(unit, dict) else None


def _extracted_candidates_for_run(run_path: Path, scene_id: str) -> dict[str, Any]:
    input_path = run_path / "inputs" / f"{scene_id}.json"
    if not input_path.is_file():
        return {}
    payload = _read_json(input_path)
    candidates = payload.get("extracted_candidates") if isinstance(payload.get("extracted_candidates"), dict) else {}
    return candidates if isinstance(candidates, dict) else {}


def _record_scene_id(data: dict[str, Any], payload: dict[str, Any], parsed_file: Path) -> str:
    return str(data.get("unit_id") or data.get("scene_id") or payload.get("unit_id") or payload.get("scene_id") or parsed_file.stem)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_rejection(
    handle: Any,
    *,
    scene_id: str,
    record_id: str,
    reason: str,
    item: Any,
    evidence_record: dict[str, Any] | None = None,
) -> None:
    handle.write(
        json.dumps(
            {
                "scene_id": scene_id,
                "record_type": "relationship_observation",
                "record_id": record_id,
                "reason": reason,
                "item": item if isinstance(item, dict) else {"value": str(item)},
                "evidence_verification_status": (evidence_record or {}).get("evidence_verification_status"),
                "evidence_alignment_score": (evidence_record or {}).get("evidence_alignment_score"),
                "evidence_source_field": (evidence_record or {}).get("evidence_source_field"),
                "evidence_start": (evidence_record or {}).get("evidence_start"),
                "evidence_end": (evidence_record or {}).get("evidence_end"),
                "parent_evidence_start": (evidence_record or {}).get("parent_evidence_start"),
                "parent_evidence_end": (evidence_record or {}).get("parent_evidence_end"),
                "parent_source_sha256": (evidence_record or {}).get("parent_source_sha256"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
