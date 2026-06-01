# Evaluation Design

## Core Principle

DMS should be evaluated as a writing-time memory system. The generated text does
not need to reproduce the original chapter or scene. It needs to satisfy the
current writing intent, read well as writing, and stay faithful to the supplied
memory packet.

The evaluation input is:

```text
writing intent
+ memory packet built from prefix text
+ optional style / length requirements
+ generated output
```

The original next unit can be evaluated with the same rubric as a calibration
reference, but exact wording or plot duplication is not the goal.

## Three Evaluation Dimensions

### 1. Writing Intent Consistency

Does the output do what the writing intent asked it to do?

Check:

- key entities requested by the intent appear in appropriate roles;
- the requested situation, conflict, transition, or scene function is present;
- the output realizes the intended narrative units, not only matching keywords;
- the ending or handoff matches the requested direction;
- the output avoids major unsupported drift away from the intent.

Suggested score:

```text
1 = misses the intent
2 = covers a few intent anchors but changes the task
3 = partially satisfies the intent with noticeable omissions
4 = mostly satisfies the intent
5 = fully satisfies the intent without distracting drift
```

### 2. Writing Quality

Is the output usable writing under the requested form?

Check:

- prose or screenplay form is appropriate;
- pacing, action/dialogue balance, and sentence rhythm fit the style reference if provided;
- length and formatting requirements are met;
- the text is concrete rather than explanatory notes about memory;
- dialogue and action feel coherent for the scene;
- the draft would be useful for a writer to revise.

Suggested score:

```text
1 = unusable or incoherent
2 = readable but weak, generic, or badly mismatched to the requested form
3 = usable draft with clear quality issues
4 = strong draft with minor issues
5 = polished and directly useful
```

### 3. Memory Faithfulness

Does the output stay faithful to the supplied memory packet?

Check:

- character/entity states do not contradict the memory packet;
- durable relationships are respected unless the intent motivates a change;
- important claims are supported by memory or by the writing intent;
- the output does not invent major new background facts that should have come from memory;
- retrieved memory is used naturally rather than copied as notes;
- references and memory indexes do not leak into the final prose.

Suggested score:

```text
1 = major contradictions or unsupported claims
2 = several memory conflicts or important unsupported additions
3 = mostly faithful with some weakly supported details
4 = faithful with minor ambiguity
5 = fully faithful and naturally grounded
```

## Reference Calibration

For benchmark samples, run the same three-dimension evaluator on both:

```text
generated_output
reference_output
```

Then report:

```text
delta = generated_score - reference_score
```

Use the delta only as calibration. A generated output can be good even if it is
different from the reference, as long as it satisfies the intent, writes well,
and remains memory-faithful.

## Social Intent, Writing Intent, And Writing Spec

The benchmark separates low-information social exploration, realistic author
input, and evaluation ground truth. Social simulation should receive less
target-scene information than generation. Evaluation should use a reference-only
writing specification extracted from the held-out target scene.

Policy:

- `social_simulation_intent`: low-information setup for exploratory social
  simulation. It should be less specific than `writing_intent` and should leave
  behavior, dialogue, emotional turns, and outcomes open.
- `writing_intent`: realistic author input for retrieval and generation. It
  should be much shorter than the source scene and must not become a synopsis
  or beat list.
- `writing_spec`: evaluation ground truth only. It is extracted from
  the held-out target scene, represented as compact required entities,
  narrative units, and state/relationship requirements, and must not be fed into
  generation.
- report the social, generation, and evaluation artifacts in every benchmark
  result.
- benchmark writing prompts may receive `previous_scene_context` for immediate
  continuity. This context is sourced only from the already-visible previous
  scene, not from the held-out target. It is capped at 800 non-whitespace
  characters; longer previous scenes are represented as summary plus entities.
  It must not be confused with the evaluation-only `writing_spec`.

This prevents generated text from receiving inflated scores merely because the
author input was underspecified, while also preventing the generator from seeing
an unrealistically detailed target-scene description.

## Minimal Judge Schema

```json
{
  "writing_intent_consistency": {
    "score": 0,
    "strengths": [],
    "issues": []
  },
  "writing_quality": {
    "score": 0,
    "strengths": [],
    "issues": []
  },
  "memory_faithfulness": {
    "score": 0,
    "strengths": [],
    "issues": []
  },
  "overall": {
    "score": 0,
    "revision_notes": []
  }
}
```

## Prompt Context Rule

Intermediate evidence IDs such as:

```text
episodic_memory_evidence scene_0004_chunk_001_memory_006
```

may be retained in JSON artifacts and traces for debugging. They should not be
included in the final writing prompt. The prompt-facing memory packet should use
short reference labels such as `[R1]` and scene IDs, followed by the original
evidence text only when evidence is needed.
