from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dms.llm import LLMClient, LLMResult
from dms.parsing import extract_json_value
from dms.prompts import YAMLPromptLoader


@dataclass(frozen=True)
class WritingEvaluationConfig:
    writing_intent: str
    generated_text: str
    memory_packet: str
    output_dir: Path
    reference_text: str | None = None
    prompt_dir: Path = Path("task_specs/prompts")
    overwrite: bool = False


def evaluate_writing(config: WritingEvaluationConfig, llm_client: LLMClient) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("prompts", "raw_outputs", "parsed"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    loader = YAMLPromptLoader(config.prompt_dir)
    calls: list[dict[str, Any]] = []
    requirements = _run_requirements_decomposition(config, loader, llm_client, output_dir, calls)

    candidates = [{"label": "generated", "text": config.generated_text}]
    if config.reference_text is not None:
        candidates.append({"label": "reference", "text": config.reference_text})

    candidate_results: dict[str, Any] = {}
    for candidate in candidates:
        label = str(candidate["label"])
        text = str(candidate["text"])
        intent_result = _run_intent_consistency(
            config,
            loader,
            llm_client,
            output_dir,
            calls,
            candidate_label=label,
            candidate_text=text,
            requirements=requirements,
        )
        quality_result = _run_scalar_judge(
            config,
            loader,
            llm_client,
            output_dir,
            calls,
            metric="writing_quality",
            prompt_id="dms/eval_writing_quality",
            candidate_label=label,
            candidate_text=text,
        )
        faithfulness_result = _run_scalar_judge(
            config,
            loader,
            llm_client,
            output_dir,
            calls,
            metric="memory_faithfulness",
            prompt_id="dms/eval_memory_faithfulness",
            candidate_label=label,
            candidate_text=text,
        )
        candidate_results[label] = {
            "writing_intent_consistency": intent_result,
            "writing_quality": quality_result,
            "memory_faithfulness": faithfulness_result,
            "overall": _mean_scores(
                [
                    intent_result.get("score"),
                    quality_result.get("score"),
                    faithfulness_result.get("score"),
                ]
            ),
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm": {
            "provider": llm_client.provider,
            "model": llm_client.model,
        },
        "inputs": {
            "has_reference_text": config.reference_text is not None,
            "generated_chars": len(config.generated_text),
            "reference_chars": len(config.reference_text or ""),
            "memory_packet_chars": len(config.memory_packet),
            "writing_intent": config.writing_intent,
        },
        "requirements": requirements,
        "candidates": candidate_results,
        "deltas": _compute_reference_deltas(candidate_results),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "calls": str(output_dir / "calls.jsonl"),
            "prompts_dir": str(output_dir / "prompts"),
            "raw_outputs_dir": str(output_dir / "raw_outputs"),
            "parsed_dir": str(output_dir / "parsed"),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "calls.jsonl", calls)
    return summary


def _run_requirements_decomposition(
    config: WritingEvaluationConfig,
    loader: YAMLPromptLoader,
    llm_client: LLMClient,
    output_dir: Path,
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = _run_json_prompt(
        loader,
        llm_client,
        output_dir,
        calls,
        call_id="intent_requirements",
        prompt_id="dms/eval_intent_requirements",
        task_values={"writing_intent": config.writing_intent},
    )
    requirements = payload.get("requirements") if isinstance(payload, dict) else None
    if not isinstance(requirements, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(requirements, start=1):
        if not isinstance(item, dict):
            continue
        requirement = str(item.get("requirement") or "").strip()
        if not requirement:
            continue
        normalized.append(
            {
                "requirement_id": str(item.get("requirement_id") or f"REQ{index}").strip(),
                "requirement": requirement,
                "category": str(item.get("category") or "other").strip(),
                "importance": str(item.get("importance") or "core").strip(),
            }
        )
    return normalized


def _run_intent_consistency(
    config: WritingEvaluationConfig,
    loader: YAMLPromptLoader,
    llm_client: LLMClient,
    output_dir: Path,
    calls: list[dict[str, Any]],
    *,
    candidate_label: str,
    candidate_text: str,
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _run_json_prompt(
        loader,
        llm_client,
        output_dir,
        calls,
        call_id=f"{candidate_label}_intent_consistency",
        prompt_id="dms/eval_intent_consistency",
        task_values={
            "writing_intent": config.writing_intent,
            "requirements_json": {"requirements": requirements},
            "candidate_label": candidate_label,
            "candidate_text": candidate_text,
        },
    )
    judgments = payload.get("requirement_judgments") if isinstance(payload, dict) else None
    if not isinstance(judgments, list):
        judgments = []
    normalized_judgments = _normalize_requirement_judgments(requirements, judgments)
    return {
        "score": _requirement_score(normalized_judgments),
        "satisfied_count": sum(1 for item in normalized_judgments if item["status"] == "satisfied"),
        "partial_count": sum(1 for item in normalized_judgments if item["status"] == "partially_satisfied"),
        "not_satisfied_count": sum(1 for item in normalized_judgments if item["status"] == "not_satisfied"),
        "requirement_count": len(normalized_judgments),
        "requirement_judgments": normalized_judgments,
        "summary": payload.get("summary") if isinstance(payload, dict) else "",
    }


def _run_scalar_judge(
    config: WritingEvaluationConfig,
    loader: YAMLPromptLoader,
    llm_client: LLMClient,
    output_dir: Path,
    calls: list[dict[str, Any]],
    *,
    metric: str,
    prompt_id: str,
    candidate_label: str,
    candidate_text: str,
) -> dict[str, Any]:
    task_values = {
        "writing_intent": config.writing_intent,
        "candidate_label": candidate_label,
        "candidate_text": candidate_text,
    }
    if metric == "memory_faithfulness":
        task_values["memory_packet"] = config.memory_packet
    payload = _run_json_prompt(
        loader,
        llm_client,
        output_dir,
        calls,
        call_id=f"{candidate_label}_{metric}",
        prompt_id=prompt_id,
        task_values=task_values,
    )
    raw_score = _coerce_raw_score(payload.get("score") if isinstance(payload, dict) else None, default=0.0, scale=5.0)
    result = dict(payload) if isinstance(payload, dict) else {"raw_payload": payload}
    result["raw_score"] = raw_score
    result["score"] = round(raw_score / 5.0, 4) if raw_score else 0.0
    return result


def _run_json_prompt(
    loader: YAMLPromptLoader,
    llm_client: LLMClient,
    output_dir: Path,
    calls: list[dict[str, Any]],
    *,
    call_id: str,
    prompt_id: str,
    task_values: dict[str, Any],
) -> Any:
    prompt = loader.render(prompt_id, task_values=task_values)
    prompt_path = output_dir / "prompts" / f"{call_id}.txt"
    raw_path = output_dir / "raw_outputs" / f"{call_id}.json"
    parsed_path = output_dir / "parsed" / f"{call_id}.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = llm_client.complete(prompt)
    raw_payload = _llm_result_to_dict(result)
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    parsed = extract_json_value(result.text)
    parsed_payload = {
        "call_id": call_id,
        "prompt_id": prompt_id,
        "status": "parsed" if parsed.ok else "parse_failed",
        "data": parsed.data,
        "parse_error": parsed.error,
    }
    parsed_path.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calls.append(
        {
            "call_id": call_id,
            "prompt_id": prompt_id,
            "prompt_path": str(prompt_path),
            "raw_output_path": str(raw_path),
            "parsed_path": str(parsed_path),
            "status": parsed_payload["status"],
            "usage": result.usage,
        }
    )
    if not parsed.ok:
        raise ValueError(f"Failed to parse JSON for {call_id}: {parsed.error}")
    return parsed.data


def _normalize_requirement_judgments(
    requirements: list[dict[str, Any]],
    judgments: list[Any],
) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("requirement_id") or "").strip(): item
        for item in judgments
        if isinstance(item, dict)
    }
    normalized: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("requirement_id") or "").strip()
        raw = by_id.get(requirement_id, {})
        status = _normalize_requirement_status(raw.get("status") if isinstance(raw, dict) else None)
        normalized.append(
            {
                "requirement_id": requirement_id,
                "requirement": requirement.get("requirement") or "",
                "category": requirement.get("category") or "other",
                "importance": requirement.get("importance") or "core",
                "status": status,
                "score": _STATUS_SCORE[status],
                "evidence_from_candidate": str(raw.get("evidence_from_candidate") or "") if isinstance(raw, dict) else "",
                "rationale": str(raw.get("rationale") or "") if isinstance(raw, dict) else "",
            }
        )
    return normalized


_STATUS_SCORE = {
    "satisfied": 1.0,
    "partially_satisfied": 0.5,
    "not_satisfied": 0.0,
}


def _normalize_requirement_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _STATUS_SCORE:
        return text
    if "partial" in text or "part" in text:
        return "partially_satisfied"
    if "not" in text or "no" in text or "unsatisfied" in text:
        return "not_satisfied"
    if "satisf" in text or "yes" in text:
        return "satisfied"
    return "not_satisfied"


def _requirement_score(judgments: list[dict[str, Any]]) -> float:
    if not judgments:
        return 0.0
    total_weight = 0.0
    weighted_score = 0.0
    for judgment in judgments:
        weight = 1.5 if judgment.get("importance") == "core" else 1.0
        total_weight += weight
        weighted_score += weight * float(judgment.get("score") or 0.0)
    return round(weighted_score / total_weight, 4) if total_weight else 0.0


def _coerce_raw_score(value: Any, *, default: float, scale: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(1.0, min(scale, score)), 4)


def _mean_scores(scores: list[Any]) -> float:
    values = [float(score) for score in scores if isinstance(score, (int, float))]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _compute_reference_deltas(candidate_results: dict[str, Any]) -> dict[str, float]:
    generated = candidate_results.get("generated")
    reference = candidate_results.get("reference")
    if not isinstance(generated, dict) or not isinstance(reference, dict):
        return {}
    deltas: dict[str, float] = {}
    for metric in ("writing_intent_consistency", "writing_quality", "memory_faithfulness", "overall"):
        generated_score = _metric_score(generated, metric)
        reference_score = _metric_score(reference, metric)
        if generated_score is None or reference_score is None:
            continue
        deltas[metric] = round(generated_score - reference_score, 4)
    return deltas


def _metric_score(candidate_result: dict[str, Any], metric: str) -> float | None:
    if metric == "overall":
        value = candidate_result.get("overall")
    else:
        metric_result = candidate_result.get(metric)
        value = metric_result.get("score") if isinstance(metric_result, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _llm_result_to_dict(result: LLMResult) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "raw_response": result.raw_response,
        "usage": result.usage,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
