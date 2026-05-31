from __future__ import annotations

import json
from pathlib import Path

from dms.evaluation import WritingEvaluationConfig, evaluate_writing
from dms.llm import LLMResult


class FakeWritingJudgeClient:
    provider = "fake"
    model = "fake-writing-judge"

    def complete(self, prompt: str) -> LLMResult:
        if "Decompose the writing intent" in prompt:
            payload = {
                "requirements": [
                    {
                        "requirement_id": "REQ1",
                        "requirement": "Include 刘培强 and 张鹏.",
                        "category": "entity_anchor",
                        "importance": "core",
                    },
                    {
                        "requirement_id": "REQ2",
                        "requirement": "Create tension in a J20C return-flight scene.",
                        "category": "atmosphere",
                        "importance": "core",
                    },
                ]
            }
        elif "Judge writing-intent consistency" in prompt:
            label = _candidate_label(prompt)
            second = "satisfied" if label == "reference" else "partially_satisfied"
            payload = {
                "candidate_label": label,
                "requirement_judgments": [
                    {
                        "requirement_id": "REQ1",
                        "status": "satisfied",
                        "evidence_from_candidate": "刘培强 张鹏",
                        "rationale": "Both names are present.",
                    },
                    {
                        "requirement_id": "REQ2",
                        "status": second,
                        "evidence_from_candidate": "J20C",
                        "rationale": "The flight tension is judged from the candidate.",
                    },
                ],
                "summary": "Intent checklist judged.",
            }
        elif "Evaluate the writing quality" in prompt:
            payload = {
                "candidate_label": _candidate_label(prompt),
                "score": 4,
                "rationale": "Usable scene prose.",
                "strengths": ["clear action"],
                "weaknesses": ["minor thinness"],
            }
        elif "Evaluate memory faithfulness" in prompt:
            payload = {
                "candidate_label": _candidate_label(prompt),
                "score": 5,
                "rationale": "No memory contradiction.",
                "supported_points": ["uses known entities"],
                "issues": [],
            }
        else:
            raise AssertionError(f"Unexpected prompt:\n{prompt[:500]}")
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            raw_response={"text": text},
            usage={"prompt_chars": len(prompt), "completion_chars": len(text)},
        )


def test_evaluate_writing_runs_three_metric_judge_and_reference_delta(tmp_path: Path) -> None:
    summary = evaluate_writing(
        WritingEvaluationConfig(
            writing_intent="写一段刘培强和张鹏驾驶J20C返航的紧张互动。",
            generated_text="刘培强压低J20C，张鹏提醒他稳住。",
            reference_text="J20C返航，刘培强加速，张鹏喊他稳点。",
            memory_packet="# Memory Packet\n\n## Entities\n刘培强\n张鹏\nJ20C",
            output_dir=tmp_path,
        ),
        llm_client=FakeWritingJudgeClient(),
    )

    generated = summary["candidates"]["generated"]
    reference = summary["candidates"]["reference"]
    assert generated["writing_intent_consistency"]["requirement_count"] == 2
    assert generated["writing_intent_consistency"]["score"] == 0.75
    assert reference["writing_intent_consistency"]["score"] == 1.0
    assert generated["writing_quality"]["raw_score"] == 4.0
    assert generated["writing_quality"]["score"] == 0.8
    assert generated["memory_faithfulness"]["raw_score"] == 5.0
    assert generated["memory_faithfulness"]["score"] == 1.0
    assert summary["deltas"]["writing_intent_consistency"] == -0.25
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "calls.jsonl").is_file()
    assert (tmp_path / "prompts" / "intent_requirements.txt").is_file()
    assert (tmp_path / "parsed" / "generated_memory_faithfulness.json").is_file()


def test_eval_prompts_keep_tone_request_separate_from_identity() -> None:
    memory_prompt = Path("task_specs/prompts/dms/eval_memory_faithfulness.yaml").read_text(encoding="utf-8")
    intent_prompt = Path("task_specs/prompts/dms/eval_intent_consistency.yaml").read_text(encoding="utf-8")

    assert "writing intent is not memory evidence" in memory_prompt
    assert "old-soldier voice" in memory_prompt
    assert "old-soldier voice" in intent_prompt
    assert "established identity" in intent_prompt


def _candidate_label(prompt: str) -> str:
    marker = "# Candidate Label"
    after = prompt.split(marker, 1)[1].strip()
    return after.splitlines()[0].strip()
