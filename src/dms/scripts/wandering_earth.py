from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScriptScene:
    """Ordered scene unit from the local Wandering Earth 2 JSON fixture."""

    scene_id: str
    source_record_id: int
    discourse_index: int
    title: str
    subtitle: str
    content: str
    raw_heading_number: int | None
    interior_exterior: str | None
    time_of_day: str | None
    location_hint: str
    character_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+)\s*[、.．]\s*"
    r"(?P<ie>EXT\s*/\s*INT|INT\s*/\s*EXT|INT|EXT|内\s*/\s*外|外\s*/\s*内|内|外)?\s*"
    r"[.。]?\s*"
    r"(?P<time>日\s*/\s*夜\s*/\s*无日夜|无日夜|日\s*/\s*夜|夜\s*/\s*日|日|夜|晨|晚|黄昏|清晨|白天|深夜)?\s*"
    r"[.。]?\s*"
    r"(?P<location>.*)$",
    re.IGNORECASE,
)


def load_script_scenes(path: str | Path) -> list[ScriptScene]:
    """Load the source JSON and normalize it into ordered scene records."""

    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in script JSON: {source_path}")

    scenes: list[ScriptScene] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at record {index}: {source_path}")

        record_id = _coerce_int(item.get("_id"), index)
        title = str(item.get("title") or "").strip()
        subtitle = str(item.get("subtitle") or "").strip()
        content = str(item.get("content") or "")
        heading = _parse_heading(title)
        scene_id = f"scene_{record_id:04d}"

        scenes.append(
            ScriptScene(
                scene_id=scene_id,
                source_record_id=record_id,
                discourse_index=index,
                title=title,
                subtitle=subtitle,
                content=content,
                raw_heading_number=heading["number"],
                interior_exterior=heading["interior_exterior"],
                time_of_day=heading["time_of_day"],
                location_hint=heading["location_hint"],
                character_count=len(content),
            )
        )

    return scenes


def write_jsonl(scenes: list[ScriptScene], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for scene in scenes:
            handle.write(json.dumps(scene.to_dict(), ensure_ascii=False) + "\n")


def write_summary(scenes: list[ScriptScene], path: str | Path, *, source_path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    locations = sorted({scene.location_hint for scene in scenes if scene.location_hint})
    payload = {
        "source_path": str(Path(source_path).resolve()),
        "scene_count": len(scenes),
        "content_char_count": sum(scene.character_count for scene in scenes),
        "first_scene_id": scenes[0].scene_id if scenes else None,
        "last_scene_id": scenes[-1].scene_id if scenes else None,
        "location_hint_count": len(locations),
        "sample_location_hints": locations[:20],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_heading(title: str) -> dict[str, Any]:
    match = _HEADING_RE.match(title or "")
    if not match:
        return {
            "number": None,
            "interior_exterior": None,
            "time_of_day": None,
            "location_hint": title.strip(),
        }

    number = _coerce_optional_int(match.group("number"))
    ie = _normalize_slash_value((match.group("ie") or "").upper()) or None
    time = _normalize_slash_value(match.group("time") or "") or None
    location = (match.group("location") or "").strip()
    return {
        "number": number,
        "interior_exterior": ie,
        "time_of_day": time,
        "location_hint": location,
    }


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_slash_value(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", value.strip())
