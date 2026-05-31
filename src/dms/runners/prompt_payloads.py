from __future__ import annotations

from typing import Any

from dms.chunking import NarrativeChunk
from dms.scripts.wandering_earth import ScriptScene


def narrative_unit_payload(scene: ScriptScene | NarrativeChunk) -> dict[str, Any]:
    """Return the genre-neutral payload shown to extraction prompts."""

    unit_id = getattr(scene, "chunk_id", scene.scene_id)
    payload = {
        "unit_id": unit_id,
        "parent_unit_id": getattr(scene, "parent_unit_id", scene.scene_id),
        "chunk_id": getattr(scene, "chunk_id", unit_id),
        "chunk_index": getattr(scene, "chunk_index", 1),
        "chunk_count": getattr(scene, "chunk_count", 1),
        "order": scene.discourse_index,
        "title": scene.title,
        "subtitle": scene.subtitle,
        "source_record_id": scene.source_record_id,
        "content": scene.content,
        "setting_hint": {
            "location": scene.location_hint,
            "time_hint": scene.time_of_day or "",
            "spatial_context": scene.interior_exterior or "",
        },
    }
    if isinstance(scene, NarrativeChunk):
        payload["source_span"] = {
            "parent_unit_id": scene.parent_unit_id,
            "source_start": scene.source_start,
            "source_end": scene.source_end,
            "source_sha256": scene.source_sha256,
            "chunk_unit_count": scene.chunk_unit_count,
            "max_chunk_units": scene.max_chunk_units,
        }
    return payload
