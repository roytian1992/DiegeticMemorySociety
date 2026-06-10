from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.context_kernel.kernel import CreativeMemoryKernel
from dms.context_kernel.schema import (
    CreativeContextItem,
    EvidenceRef,
    SourceRecord,
    SourceUnit,
    stable_id,
)
from dms.llm import LLMClient
from dms.parsing import extract_json_value


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    role: str
    content: str
    speaker: str = ""
    order: int = 0
    metadata: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "speaker": self.speaker,
            "order": self.order,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class ConversationalMemoryIngestConfig:
    transcript_path: Path
    project_id: str
    context_db_path: Path
    conversation_id: str = "conversation"
    source_id: str | None = None
    title: str = "Creative Conversation"
    reset_store: bool = False
    use_llm: bool = False
    actor: str = "system"


def ingest_conversational_memory(
    config: ConversationalMemoryIngestConfig,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    if config.use_llm and llm_client is None:
        raise ValueError("llm_client is required when use_llm is true")
    source_id = config.source_id or f"conversation:{config.conversation_id}"
    turns = load_conversation_transcript(config.transcript_path)
    kernel = CreativeMemoryKernel.from_db(config.context_db_path, reset=config.reset_store)
    kernel.add_source(
        SourceRecord(
            source_id=source_id,
            project_id=config.project_id,
            source_type="conversation",
            title=config.title,
            status="active",
            metadata={
                "conversation_id": config.conversation_id,
                "transcript_path": str(config.transcript_path),
            },
        )
    )
    for turn in turns:
        kernel.add_unit(
            SourceUnit(
                unit_id=turn.turn_id,
                source_id=source_id,
                project_id=config.project_id,
                source_type="conversation",
                unit_type="turn",
                unit_order=turn.order,
                speaker=turn.speaker or turn.role,
                text=turn.content,
                start_offset=0,
                end_offset=len(turn.content),
                metadata={"role": turn.role, **(turn.metadata or {})},
            )
        )

    if config.use_llm:
        items = extract_conversational_memory_with_llm(
            turns,
            project_id=config.project_id,
            source_id=source_id,
            llm_client=llm_client,
        )
        extraction_mode = "llm"
    else:
        items = extract_conversational_memory_deterministic(
            turns,
            project_id=config.project_id,
            source_id=source_id,
        )
        extraction_mode = "deterministic_v0"

    for item in items:
        kernel.add_item(item, actor=config.actor, reason="conversation memory ingestion")
        if item.item_type == "correction" and item.entity_ids:
            for entity_id in item.entity_ids:
                kernel.add_entity_patch(
                    _patch_from_correction(item, entity_id),
                    actor=config.actor,
                    reason="conversation correction overlay",
                )

    return {
        "context_db_path": str(config.context_db_path),
        "project_id": config.project_id,
        "conversation_id": config.conversation_id,
        "source_id": source_id,
        "transcript_path": str(config.transcript_path),
        "turn_count": len(turns),
        "memory_item_count": len(items),
        "extraction_mode": extraction_mode,
        "item_type_counts": _item_type_counts(items),
    }


def load_conversation_transcript(path: str | Path) -> list[ConversationTurn]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            records = payload.get("turns") or payload.get("messages") or payload.get("conversation") or []
        elif isinstance(payload, list):
            records = payload
        else:
            records = []
    else:
        records = _plain_text_turns(text)
    turns = []
    for index, record in enumerate(records, start=1):
        if isinstance(record, str):
            record = {"role": "user", "content": record}
        if not isinstance(record, dict):
            continue
        content = str(record.get("content") or record.get("text") or record.get("message") or "").strip()
        if not content:
            continue
        role = str(record.get("role") or record.get("speaker") or "user").strip().lower()
        speaker = str(record.get("name") or record.get("speaker") or role).strip()
        turn_id = str(record.get("turn_id") or record.get("id") or f"turn_{index:04d}").strip()
        turns.append(
            ConversationTurn(
                turn_id=turn_id,
                role=role,
                content=content,
                speaker=speaker,
                order=int(record.get("order") or index),
                metadata={k: v for k, v in record.items() if k not in {"content", "text", "message", "role", "speaker", "name", "turn_id", "id", "order"}},
            )
        )
    return turns


def extract_conversational_memory_deterministic(
    turns: list[ConversationTurn],
    *,
    project_id: str,
    source_id: str,
) -> list[CreativeContextItem]:
    items: list[CreativeContextItem] = []
    for turn in turns:
        if turn.role not in {"user", "human"}:
            continue
        content = turn.content.strip()
        for item_type, statement in _deterministic_statements(content):
            subject = _guess_subject(statement)
            item_id = stable_id("conv_item", source_id, turn.turn_id, item_type, statement)
            items.append(
                CreativeContextItem(
                    item_id=item_id,
                    project_id=project_id,
                    source_type="conversation",
                    source_id=source_id,
                    unit_id=turn.turn_id,
                    item_type=item_type,
                    subject=subject,
                    statement=statement,
                    entity_ids=tuple(_entity_ids_for_subject(subject, statement)),
                    evidence_refs=(
                        EvidenceRef(
                            evidence_id=stable_id("ev", item_id, content),
                            item_id=item_id,
                            source_id=source_id,
                            unit_id=turn.turn_id,
                            text=content,
                            start_offset=0,
                            end_offset=len(content),
                            alignment_status="exact_turn",
                            metadata={"speaker": turn.speaker, "role": turn.role},
                        ),
                    ),
                    authority="user_explicit",
                    confidence=0.85,
                    status="active" if item_type != "rejected_option" else "rejected",
                    visibility="author_only",
                    temporal_scope="not_applicable",
                    payload={"conversation_turn": turn.model_dump(), "extraction_mode": "deterministic_v0"},
                ).normalized()
            )
    return _dedupe_items(items)


def extract_conversational_memory_with_llm(
    turns: list[ConversationTurn],
    *,
    project_id: str,
    source_id: str,
    llm_client: LLMClient,
) -> list[CreativeContextItem]:
    prompt = _conversation_extraction_prompt(turns)
    result = llm_client.complete(prompt)
    parsed = extract_json_value(result.text)
    if not parsed.ok:
        return extract_conversational_memory_deterministic(turns, project_id=project_id, source_id=source_id)
    payload = parsed.data
    raw_items = payload.get("memory_items") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return extract_conversational_memory_deterministic(turns, project_id=project_id, source_id=source_id)
    turns_by_id = {turn.turn_id: turn for turn in turns}
    items = []
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        statement = str(raw.get("statement") or "").strip()
        item_type = str(raw.get("item_type") or "creative_decision").strip()
        if not statement:
            continue
        turn_id = str(raw.get("turn_id") or "").strip()
        turn = turns_by_id.get(turn_id) or next((candidate for candidate in turns if candidate.role in {"user", "human"}), None)
        subject = str(raw.get("subject") or _guess_subject(statement)).strip()
        item_id = str(raw.get("item_id") or stable_id("conv_item", source_id, turn_id, item_type, statement, index))
        evidence_text = str(raw.get("evidence") or (turn.content if turn else statement)).strip()
        items.append(
            CreativeContextItem(
                item_id=item_id,
                project_id=project_id,
                source_type="conversation",
                source_id=source_id,
                unit_id=turn.turn_id if turn else None,
                item_type=item_type,
                subject=subject,
                statement=statement,
                entity_ids=tuple(raw.get("entity_ids") or _entity_ids_for_subject(subject, statement)),
                evidence_refs=(
                    EvidenceRef(
                        evidence_id=stable_id("ev", item_id, evidence_text),
                        item_id=item_id,
                        source_id=source_id,
                        unit_id=turn.turn_id if turn else None,
                        text=evidence_text,
                        start_offset=0,
                        end_offset=len(evidence_text),
                        alignment_status="llm_provided",
                    ),
                ),
                authority=str(raw.get("authority") or "user_explicit"),
                confidence=float(raw.get("confidence") or 0.75),
                status=str(raw.get("status") or ("rejected" if item_type == "rejected_option" else "active")),
                visibility=str(raw.get("visibility") or "author_only"),
                temporal_scope="not_applicable",
                payload={"raw_llm_item": raw, "llm": result.to_dict(), "extraction_mode": "llm"},
            ).normalized()
        )
    return _dedupe_items(items)


def _conversation_extraction_prompt(turns: list[ConversationTurn]) -> str:
    transcript = "\n".join(
        f"{turn.turn_id} | {turn.role} | {turn.speaker}: {turn.content}"
        for turn in turns
    )
    return f"""Extract long-term creative context memory from the transcript.

Return JSON only:
{{
  "memory_items": [
    {{
      "turn_id": "turn id",
      "item_type": "user_preference | creative_decision | story_constraint | style_preference | open_question | rejected_option | correction | task_state",
      "subject": "entity/topic",
      "statement": "one durable creative-context statement",
      "evidence": "verbatim user text span",
      "entity_ids": ["entity:..."],
      "authority": "user_explicit | user_confirmed | model_inferred",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Prefer user messages over assistant messages.
- Do not store raw chat unless it expresses durable creative context.
- Conversation memory is not canon by default.
- Corrections and rejected options must be explicit.

Transcript:
{transcript}
"""


def _deterministic_statements(content: str) -> list[tuple[str, str]]:
    normalized = content.strip()
    candidates: list[tuple[str, str]] = []
    sentence_parts = [part.strip() for part in re.split(r"[。！？!?\n]+", normalized) if part.strip()]
    for sentence in sentence_parts:
        if _contains_any(sentence, ["不要", "别", "不能", "避免", "不该", "不应该"]):
            if _contains_any(sentence, ["写", "设定", "角色", "口吻", "风格", "当成", "成为"]):
                candidates.append(("correction", _normalize_statement(sentence)))
            else:
                candidates.append(("story_constraint", _normalize_statement(sentence)))
        elif _contains_any(sentence, ["否掉", "不要了", "废弃", "不采用", "删掉", "算了"]):
            candidates.append(("rejected_option", _normalize_statement(sentence)))
        elif _contains_any(sentence, ["记住", "确定", "确认", "就设定为", "作为正式设定"]):
            candidates.append(("creative_decision", _normalize_statement(sentence)))
        elif _contains_any(sentence, ["我希望", "我想", "偏好", "更喜欢", "最好"]):
            item_type = "style_preference" if _contains_any(sentence, ["风格", "口吻", "对白", "叙述", "节奏", "语气"]) else "user_preference"
            candidates.append((item_type, _normalize_statement(sentence)))
        elif sentence.endswith("吗") or _contains_any(sentence, ["是不是", "如何", "怎么处理", "要不要"]):
            candidates.append(("open_question", _normalize_statement(sentence)))
    return candidates


def _normalize_statement(text: str) -> str:
    return str(text or "").strip(" ，,。；;")


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _guess_subject(statement: str) -> str:
    quoted = re.findall(r"[《“\"]([^”\"]{1,20})[”\"]", statement)
    if quoted:
        return quoted[0].strip()
    for pattern in (
        r"把([\u4e00-\u9fffA-Za-z0-9_]{2,12})写成",
        r"写([\u4e00-\u9fffA-Za-z0-9_]{2,12})时",
        r"([\u4e00-\u9fffA-Za-z0-9_]{2,12})不要",
        r"([\u4e00-\u9fffA-Za-z0-9_]{2,12})不应该",
        r"([\u4e00-\u9fffA-Za-z0-9_]{2,12})应该",
    ):
        match = re.search(pattern, statement)
        if match:
            return match.group(1).strip()
    return ""


def _entity_ids_for_subject(subject: str, statement: str) -> list[str]:
    names = []
    if subject:
        names.append(subject)
    for name in re.findall(r"[\u4e00-\u9fff]{2,4}", statement):
        if name in {"不要", "不能", "应该", "写成", "角色", "设定", "用户", "明确", "希望", "作为", "正式"}:
            continue
        if name not in names:
            names.append(name)
    return [f"entity:{_normalize_entity_key(name)}" for name in names[:4]]


def _normalize_entity_key(name: str) -> str:
    return re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")


def _patch_from_correction(item: CreativeContextItem, entity_id: str):
    from dms.context_kernel.schema import EntityPatch

    return EntityPatch(
        patch_id=stable_id("patch", item.item_id, entity_id, item.statement),
        project_id=item.project_id,
        entity_id=entity_id,
        source_item_id=item.item_id,
        patch_type="constrain",
        target_field="profile",
        patch_statement=item.statement,
        authority=item.authority,
        status="active",
        applies_to="project",
        metadata={"created_from": "conversation_correction"},
    )


def _plain_text_turns(text: str) -> list[dict[str, Any]]:
    records = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        role = "user"
        content = stripped
        if ":" in stripped:
            prefix, rest = stripped.split(":", 1)
            if prefix.strip().lower() in {"user", "assistant", "human", "ai", "用户", "助手"}:
                role = "assistant" if prefix.strip().lower() in {"assistant", "ai", "助手"} else "user"
                content = rest.strip()
        records.append({"turn_id": f"turn_{index:04d}", "role": role, "content": content, "order": index})
    return records


def _dedupe_items(items: list[CreativeContextItem]) -> list[CreativeContextItem]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.source_type, item.item_type, item.subject, item.statement)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _item_type_counts(items: list[CreativeContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.item_type] = counts.get(item.item_type, 0) + 1
    return counts

