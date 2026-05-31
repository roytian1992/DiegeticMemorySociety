from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from dms.scripts.wandering_earth import ScriptScene

try:  # LangChain text splitters are useful, but DMS should still have a fallback.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - exercised only when the splitters package is absent.
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


DEFAULT_MAX_CHUNK_UNITS = 800
_RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ";",
    ". ",
    "! ",
    "? ",
    "，",
    ", ",
    "、",
    " ",
    "",
]


@dataclass(frozen=True)
class NarrativeChunk:
    """A source-contiguous processing unit inside one ordered narrative scene."""

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
    chunk_id: str
    parent_unit_id: str
    chunk_index: int
    chunk_count: int
    source_start: int
    source_end: int
    source_sha256: str
    chunk_unit_count: int
    max_chunk_units: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_scene(scene: ScriptScene, *, max_chunk_units: int = DEFAULT_MAX_CHUNK_UNITS) -> list[NarrativeChunk]:
    if max_chunk_units < 1:
        raise ValueError("max_chunk_units must be >= 1")

    content = scene.content or ""
    source_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if not content:
        return [_chunk(scene, index=1, count=1, start=0, end=0, source_sha256=source_sha256, max_chunk_units=max_chunk_units)]

    spans = _chunk_spans(content, max_chunk_units=max_chunk_units)
    if not spans:
        spans = [(0, len(content))]

    chunks: list[NarrativeChunk] = []
    for index, (start, end) in enumerate(spans, start=1):
        chunks.append(
            _chunk(
                scene,
                index=index,
                count=len(spans),
                start=start,
                end=end,
                source_sha256=source_sha256,
                max_chunk_units=max_chunk_units,
            )
        )
    return chunks


def chunk_unit_count(text: str) -> int:
    return sum(1 for _ in _unit_tokens(text))


def _chunk(
    scene: ScriptScene,
    *,
    index: int,
    count: int,
    start: int,
    end: int,
    source_sha256: str,
    max_chunk_units: int,
) -> NarrativeChunk:
    chunk_id = scene.scene_id if count == 1 else f"{scene.scene_id}_chunk_{index:03d}"
    content = (scene.content or "")[start:end]
    return NarrativeChunk(
        scene_id=scene.scene_id,
        source_record_id=scene.source_record_id,
        discourse_index=scene.discourse_index,
        title=scene.title,
        subtitle=scene.subtitle,
        content=content,
        raw_heading_number=scene.raw_heading_number,
        interior_exterior=scene.interior_exterior,
        time_of_day=scene.time_of_day,
        location_hint=scene.location_hint,
        character_count=len(content),
        chunk_id=chunk_id,
        parent_unit_id=scene.scene_id,
        chunk_index=index,
        chunk_count=count,
        source_start=start,
        source_end=end,
        source_sha256=source_sha256,
        chunk_unit_count=chunk_unit_count(content),
        max_chunk_units=max_chunk_units,
    )


def _chunk_spans(content: str, *, max_chunk_units: int) -> list[tuple[int, int]]:
    segments = _recursive_split_spans(content, max_chunk_units=max_chunk_units) or _segments(content)
    normalized_segments: list[tuple[int, int]] = []
    for start, end in segments:
        if chunk_unit_count(content[start:end]) <= max_chunk_units:
            normalized_segments.append((start, end))
        else:
            normalized_segments.extend(_hard_split_span(content, start, end, max_chunk_units=max_chunk_units))

    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    current_units = 0
    for start, end in normalized_segments:
        units = chunk_unit_count(content[start:end])
        if current_start is None:
            current_start = start
            current_end = end
            current_units = units
            continue
        if current_units + units <= max_chunk_units:
            current_end = end
            current_units += units
            continue
        chunks.append((current_start, current_end if current_end is not None else start))
        current_start = start
        current_end = end
        current_units = units

    if current_start is not None:
        chunks.append((current_start, current_end if current_end is not None else len(content)))
    return _trim_empty_spans(content, chunks)


def _segments(content: str) -> list[tuple[int, int]]:
    boundaries = {0, len(content)}
    for match in re.finditer(r"\n\s*\n+", content):
        boundaries.add(match.end())
    for match in re.finditer(r"\n+", content):
        boundaries.add(match.end())
    for match in re.finditer(r"[。！？!?；;]\s*", content):
        boundaries.add(match.end())

    ordered = sorted(boundaries)
    segments: list[tuple[int, int]] = []
    for start, end in zip(ordered, ordered[1:]):
        if end > start:
            segments.append((start, end))
    return segments


def _hard_split_span(content: str, start: int, end: int, *, max_chunk_units: int) -> list[tuple[int, int]]:
    text = content[start:end]
    tokens = list(_unit_tokens(text))
    if not tokens:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    token_index = 0
    current_start = start
    while token_index < len(tokens):
        next_index = min(token_index + max_chunk_units, len(tokens))
        split_end = start + tokens[next_index - 1][1]
        spans.append((current_start, split_end))
        current_start = split_end
        token_index = next_index
    if spans and spans[-1][1] < end:
        last_start, _ = spans[-1]
        spans[-1] = (last_start, end)
    return spans


def _recursive_split_spans(content: str, *, max_chunk_units: int) -> list[tuple[int, int]]:
    if RecursiveCharacterTextSplitter is None:
        return []
    splitter = RecursiveCharacterTextSplitter(
        separators=_RECURSIVE_SEPARATORS,
        keep_separator="end",
        chunk_size=max_chunk_units,
        chunk_overlap=0,
        length_function=chunk_unit_count,
        add_start_index=True,
        strip_whitespace=False,
    )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for document in splitter.create_documents([content]):
        chunk_text = document.page_content
        if not chunk_text:
            continue
        start = document.metadata.get("start_index")
        if not isinstance(start, int) or start < 0:
            return []
        end = start + len(chunk_text)
        if content[start:end] != chunk_text:
            return []
        if start != cursor:
            return []
        spans.append((start, end))
        cursor = end
    if cursor != len(content):
        return []
    return spans


def _unit_tokens(text: str) -> list[tuple[int, int]]:
    tokens: list[tuple[int, int]] = []
    pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[.\-'][A-Za-z0-9]+)*")
    for match in pattern.finditer(text or ""):
        tokens.append((match.start(), match.end()))
    return tokens


def _trim_empty_spans(content: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        if start >= end:
            continue
        if not content[start:end]:
            continue
        trimmed.append((start, end))
    return trimmed
