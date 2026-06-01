# Social Simulation Design

## Purpose

The current social-simulation step is useful but too prompt-driven. It builds
evidence-bound character cards, asks an LLM to infer likely behavior for each
character, and asks another LLM call to coordinate those in-scene beats. That is
not yet a strong simulation mechanism. It has weak control over action choice,
interaction pressure, dialogue texture, and unsupported psychological
translation.

The next version should be an algorithmic interaction planner with LLM calls
used only for typed extraction, controlled candidate generation, and natural
language realization. The planner should decide which interaction beats are
allowed, motivated, useful, and risky before final writing sees them.

## Design Goals

- Keep social simulation less informative than normal writing intent.
- Preserve the author-facing `writing_intent` as the writing target; social
  simulation is behavior guidance only.
- Use prefix-only memory and evidence refs.
- Model interaction as pressure, resistance, compliance, avoidance, and state
  change, not just a list of plausible dialogue lines.
- Separate memory-supported facts from intent-only scene conditions.
- Prevent unsupported role upgrades, modern therapy-like paraphrases, and
  over-literal psychological summaries.
- Produce an inspectable artifact that can be scored, filtered, and compared
  before writing.

## Current Baseline

Current pipeline:

```text
memory packet
  -> per-entity attribute cards
  -> per-character LLM social simulation
  -> LLM coordinator beats
  -> writing prompt as optional guidance
```

Current strengths:

- Prefix-only memory boundary.
- Per-character attribute cards with refs.
- Hard constraints and simulation risks.
- Intent-only mechanics separated from memory refs in prompts.
- Coordinator produces scene beats, dynamics, risks, and writer guidance.

Current weaknesses:

- No explicit social state update.
- No action utility or search.
- No pressure graph.
- No verifier for each beat's support and risk.
- No distinction between dialogue intent and final dialogue wording.
- No automatic detection of modern psychological phrasing such as "别跟地球赌气".
- No multi-candidate simulation or reranking.

## Proposed Mechanism

The next mechanism is `Algorithmic Social Interaction Planner`, or ASIP.

```text
memory packet + social_simulation_intent + attribute cards
  -> social state graph
  -> scene frame
  -> pressure graph
  -> action candidate generation
  -> utility scoring
  -> constrained beat search
  -> beat verification
  -> dialogue posture realization
  -> social simulation packet
```

The planner should be deterministic where possible and use LLM calls behind
typed schemas where natural language inference is needed.

## Data Model

### Character State

Each character gets a compact state vector derived from the attribute card.

```json
{
  "character_id": "character_liu_peiqiang",
  "name": "刘培强",
  "prefix_boundary": "before scene_0006",
  "role": [
    {"value": "预备航天员，即将前往月球受训", "support": ["M6", "R6"], "confidence": 1.0}
  ],
  "current_goals": [
    {"goal": "离开地面环境，靠近太空/天上归属", "support": ["M13", "R13"], "confidence": 0.85}
  ],
  "current_pressures": [
    {"pressure": "正在驾驶或参与J20C飞行操作", "source": "memory", "support": ["M9", "M10"]},
    {"pressure": "写作意图要求返航途中出现危险飞行行为", "source": "intent", "support": []}
  ],
  "affect": {
    "valence": -0.6,
    "arousal": 0.7,
    "dominant_emotions": ["厌弃地面", "急切", "压抑"]
  },
  "control_style": {
    "risk_tolerance": 0.7,
    "impulsivity": 0.6,
    "compliance_with_peer": 0.45
  },
  "relationship_stances": {
    "张鹏": {
      "trust": 0.65,
      "accepts_guidance": 0.5,
      "resistance": 0.45,
      "support": ["M7", "M11", "M12", "R7", "R11", "R12"]
    }
  },
  "hard_constraints": [],
  "risks": []
}
```

These numbers should be coarse ordinal values, not claims of psychological
precision. Suggested scale is `0.0` to `1.0` with buckets:

```text
0.0-0.2 absent/low
0.3-0.4 weak
0.5-0.6 moderate
0.7-0.8 strong
0.9-1.0 explicit/dominant
```

### Scene Frame

The scene frame is extracted from `social_simulation_intent` and the full
`writing_intent`, but labels which parts are available to which stage.

```json
{
  "scene_id": "candidate_scene",
  "social_seed": "两名飞行员在战后返航途中，于驾驶舱内面对危险飞行行为展开互动。",
  "writing_target": "描绘J20C返航途中飞越战区废墟的紧张氛围...",
  "allowed_for_social_simulation": {
    "participants": ["两名飞行员"],
    "setting_type": ["驾驶舱", "返航途中"],
    "interaction_problem": ["危险飞行行为"]
  },
  "not_socially_required": [
    "具体地理锚点",
    "目标场景结局",
    "writing_spec细节"
  ],
  "generation_anchors": ["J20C", "张鹏", "刘培强", "返航", "战区废墟"]
}
```

### Pressure Graph

A pressure graph represents who pressures whom, by what basis, and toward what
behavior.

```json
{
  "nodes": ["刘培强", "张鹏"],
  "edges": [
    {
      "source": "张鹏",
      "target": "刘培强",
      "pressure_type": "safety_correction",
      "desired_change": "降低操作节奏并稳住飞行姿态",
      "memory_basis": ["M11", "R11", "M7", "R7"],
      "intent_basis": ["危险飞行行为"],
      "strength": 0.8,
      "tone_bounds": ["口语化", "关切", "不要升级为正式命令"]
    },
    {
      "source": "刘培强",
      "target": "张鹏",
      "pressure_type": "value_resistance",
      "desired_change": "不把地面安全提醒当作优先事项",
      "memory_basis": ["M13", "R13"],
      "intent_basis": ["驾驶舱互动", "情绪张力"],
      "strength": 0.6,
      "tone_bounds": ["直接", "压抑", "不要心理咨询式抽象"]
    }
  ]
}
```

Pressure edges are not prose. They are typed interaction forces used by the
planner.

### Action Candidate

An action candidate is a possible in-scene move by one character.

```json
{
  "candidate_id": "a_liu_001",
  "actor": "刘培强",
  "action_type": "risky_operation",
  "surface_action": "手指拨得过快，机身轻晃",
  "dialogue_intent": "淡化地面/地球价值，拒绝完全配合",
  "not_final_dialogue": true,
  "memory_basis": ["M9", "M10", "M11", "M13"],
  "intent_basis": ["危险飞行行为", "驾驶舱互动"],
  "preconditions": ["正在飞行操作", "张鹏可观察并提醒"],
  "effects": {
    "flight_stability": -0.2,
    "zhang_peng_concern": 0.2,
    "liu_resistance": 0.1
  },
  "risks": [
    {"risk": "把预备航天员写成正式飞行员或战斗员", "severity": "medium"},
    {"risk": "过度心理化台词", "severity": "medium"}
  ]
}
```

### Beat

A beat is a selected pair or sequence of candidate actions.

```json
{
  "beat_id": "b_001",
  "participants": ["刘培强", "张鹏"],
  "interaction_function": "risk_escalation_and_correction",
  "actions": ["a_liu_001", "a_zhang_002"],
  "state_delta": {
    "flight_stability": -0.1,
    "zhang_peng_concern": 0.2,
    "liu_resistance": 0.1,
    "relationship_tension": 0.2
  },
  "required_in_writing": false,
  "priority": 0.86,
  "verification": {
    "memory_supported": true,
    "intent_aligned": true,
    "hard_constraint_violations": [],
    "style_risks": ["avoid therapy-like wording"]
  }
}
```

## Algorithm

### Step 1. Build Social State Graph

Inputs:

- attribute cards;
- memory packet relations;
- social simulation intent;
- writing intent anchors for generation compatibility.

Process:

1. Normalize character states into coarse numeric and categorical features.
2. Convert relationship stances into directed relationship edges.
3. Convert `hard_constraints` into hard validators.
4. Convert `simulation_risks` into soft validators and penalty features.
5. Keep evidence ids on every state, edge, and risk.

Output:

```text
social_state_graph.json
```

### Step 2. Extract Scene Frame

Use a deterministic parser plus optional LLM schema to extract:

- participants;
- setting type;
- central interaction problem;
- source of pressure;
- generation anchors;
- forbidden evaluator-only details.

The social simulator receives less information than writing. It should not see
`writing_spec`.

### Step 3. Build Pressure Graph

For every ordered character pair `(A, B)`, compute possible pressure edges.

Pressure types:

| Type | Meaning |
| --- | --- |
| `safety_correction` | One character pushes another to reduce immediate risk |
| `value_resistance` | One character resists due to value or goal conflict |
| `care_guidance` | Concern expressed through advice or protective framing |
| `status_challenge` | One character challenges authority, expertise, or competence |
| `avoidance` | One character evades emotional or factual confrontation |
| `information_probe` | One character asks a question to expose a belief/state |
| `comfort_or_repair` | One character reduces tension or repairs trust |

Scoring:

```text
edge_strength =
  0.35 * memory_support
+ 0.25 * intent_relevance
+ 0.20 * relationship_salience
+ 0.10 * recency
+ 0.10 * state_urgency
```

Where:

- `memory_support`: 1.0 if explicit evidence, 0.6 if multi-evidence inference,
  0.3 if weak inference, 0 if unsupported.
- `intent_relevance`: match to social simulation intent.
- `relationship_salience`: relation exists and is relevant.
- `recency`: recent memory scores higher.
- `state_urgency`: current state indicates pressure.

Drop edges below `0.35`. Mark edges `high_priority` above `0.75`.

### Step 4. Generate Action Candidates

For each character, generate candidates from templates and controlled LLM calls.

Template families:

```text
risky_operation
safety_warning
deflection
minimal_compliance
value_statement
care_reframe
question_probe
physical_reaction
silence_or_withholding
```

Candidate generation should be bounded:

- 3-6 candidates per character;
- each candidate cites memory or intent basis;
- final dialogue text is not generated yet;
- dialogue intent is represented abstractly.

Example candidate templates:

```text
Actor has fast-operation tendency + scene has dangerous flight
  -> risky_operation candidate

Other character has care_commitment + safety_correction edge
  -> safety_warning candidate

Actor has negative Earth stance + receives safety warning
  -> deflection/value_statement candidate
```

### Step 5. Score Candidates

Each candidate receives:

```text
candidate_score =
  0.30 * intent_alignment
+ 0.25 * character_motivation
+ 0.20 * memory_support
+ 0.10 * interaction_productivity
+ 0.10 * novelty_without_contradiction
- 0.25 * hard_violation
- 0.15 * unsupported_role_risk
- 0.10 * therapy_phrase_risk
- 0.10 * exact_memory_repetition_risk
```

Definitions:

- `intent_alignment`: does the action help the requested scene problem?
- `character_motivation`: does the action follow the character state?
- `memory_support`: explicit/inferred support from prefix memory.
- `interaction_productivity`: will this create pressure or response?
- `novelty_without_contradiction`: new local action that does not invent major
  facts.
- `hard_violation`: formal role, future knowledge, contradiction, target-scene
  leakage.
- `therapy_phrase_risk`: phrases like `别跟地球赌气`, `放下`, `和自己和解`,
  `你是在逃避`, unless the style explicitly permits modern psychological talk.

### Step 6. Search Beat Sequences

Build a short beat sequence, usually 3-5 beats.

Constraints:

- at least one beat must activate the central interaction problem;
- at least one response beat must show pressure or resistance;
- at least one beat must show relationship stance or care;
- no beat may violate hard constraints;
- total beats should not over-specify final prose.

Search method:

1. Seed with top pressure edge.
2. Select one action from source/target candidates.
3. Apply state delta.
4. Select a response action.
5. Repeat until target coverage is reached or max beats reached.
6. Keep top `N=5` sequences by score.

Sequence score:

```text
sequence_score =
  0.30 * scene_problem_coverage
+ 0.20 * interaction_arc
+ 0.15 * character_consistency
+ 0.15 * memory_faithfulness
+ 0.10 * writing_usefulness
+ 0.10 * diversity
- penalties
```

`interaction_arc` rewards escalation, response, adjustment, and unresolved but
usable tension. It should not require a neat resolution.

### Step 7. Verify Beats

Every selected beat passes through validators.

Hard validators:

- no future or target-scene leakage;
- no unsupported formal role;
- no unsupported kinship;
- no contradiction with current state;
- no reference id leakage into final writer guidance;
- no exact memory repetition unless explicitly needed.

Soft validators:

- therapy-like phrase risk;
- over-specific technical details;
- unsupported worldbuilding;
- excessive biography;
- over-determined plot outcome;
- too much reference-scene specificity.

Beat verification output:

```json
{
  "beat_id": "b_001",
  "status": "pass | revise | reject",
  "hard_violations": [],
  "soft_warnings": [
    {
      "type": "therapy_phrase_risk",
      "detail": "Avoid direct phrasing like 别跟地球赌气; use concrete flight correction instead."
    }
  ],
  "repair_suggestion": "张鹏 should correct the flight action directly: 别较劲，先稳住高度。"
}
```

### Step 8. Dialogue Posture, Not Dialogue Lines

The social simulation packet should not hand final dialogue to the writer as if
it were canonical. It should provide dialogue posture.

Bad:

```text
张鹏: 别跟地球赌气。
```

Good:

```json
{
  "speaker": "张鹏",
  "dialogue_function": "safety warning with care",
  "tone": ["短促", "口语化", "不心理化"],
  "avoid_phrases": ["别跟地球赌气", "你是在逃避", "和自己和解"],
  "example_safe_phrases": ["别较劲，先稳住高度", "先看仪表，别看下面"]
}
```

The writing model can use examples, but the system should mark them as examples
and prefer posture constraints.

## Output Packet

The final social simulation artifact should be structured like this:

```json
{
  "simulation_id": "scene_0006_social_plan_v2",
  "inputs": {
    "social_simulation_intent": "...",
    "memory_boundary": "before scene_0006",
    "characters": ["刘培强", "张鹏"]
  },
  "state_graph_summary": [],
  "pressure_graph": [],
  "selected_sequence": {
    "score": 0.87,
    "coverage": {
      "scene_problem": true,
      "pressure_response": true,
      "relationship_stance": true
    },
    "beats": []
  },
  "rejected_or_repaired": [],
  "writer_guidance": {
    "must_preserve": [],
    "use_as_optional_behavior": [],
    "dialogue_posture": [],
    "avoid": []
  }
}
```

Markdown should show:

- selected beats;
- why each beat was selected;
- support and intent basis;
- warnings and repair suggestions;
- dialogue posture.

## Integration With Writing

Writing prompt should receive:

```text
writing_intent
memory_packet
previous_scene_context
attribute_cards
social_simulation_packet
```

Authority order:

1. Writing intent is the creative target.
2. Memory packet and hard constraints define factual boundaries.
3. Previous scene context provides immediate continuity.
4. Social simulation provides optional behavior guidance.
5. Dialogue posture examples are not required text.

Social simulation must never override writing intent anchors.

## Implementation Plan

### Phase 1. Deterministic Validator Around Current Output

Add a verifier for current `social_simulation.json`.

Files likely needed:

```text
src/dms/simulation/verification.py
tests/test_social_simulation_verification.py
task_specs/prompts/dms/social_simulation_repair.yaml
```

Checks:

- every `memory_basis` ref exists in cards or memory packet;
- intent-only mechanics are not cited as memory;
- formal role terms are flagged;
- therapy-like phrases are flagged;
- exact dialogue suggestions are flagged as examples, not requirements;
- action beats repeating salient past actions are flagged.

This phase can immediately catch the `别跟地球赌气` class of issue if the
phrase appears in simulation guidance.

### Phase 2. Pressure Graph And Candidate Scoring

Add data structures:

```text
src/dms/simulation/social_state.py
src/dms/simulation/pressure_graph.py
src/dms/simulation/action_candidates.py
```

Implement deterministic scoring and output `pressure_graph.json` and
`candidate_actions.json`.

### Phase 3. Beat Search

Add:

```text
src/dms/simulation/beat_search.py
```

Implement top-k sequence search with the coverage constraints above.

### Phase 4. LLM-Bounded Realization

Use LLM only after candidate/beat selection:

- convert candidate actions into concise Chinese writer guidance;
- generate dialogue posture;
- repair flagged phrasing.

The LLM should not decide the whole social plan from scratch.

### Phase 5. Evaluation

Add social simulation metrics:

| Metric | Meaning |
| --- | --- |
| `coverage_scene_problem` | central social problem appears in selected beats |
| `coverage_relationship` | relevant relationship stance appears |
| `memory_support_rate` | share of memory-backed claims with refs |
| `intent_basis_precision` | intent-only details not cited as memory |
| `hard_violation_count` | formal role, contradiction, future leak |
| `therapy_phrase_risk_count` | modern psychological phrasing risk |
| `writing_usefulness_score` | whether guidance helps writing without overdetermining |

For benchmark summaries, report these separately from writing evaluation.

## Example Applied To Scene 6

Inputs:

- Social seed: `两名飞行员在战后返航途中，于驾驶舱内面对危险飞行行为展开互动。`
- Characters: 刘培强, 张鹏.
- Key memory: 刘培强 thinks Earth is not beautiful; 张鹏 reminds him to slow
  down; 张鹏 promised to take care of him; J20C has just left the ruins.

Pressure graph:

```text
张鹏 -> 刘培强: safety_correction / care_guidance
刘培强 -> 张鹏: value_resistance / minimal compliance
```

Good beat:

```text
刘培强操作过快造成轻微不稳 -> 张鹏要求稳住 -> 刘培强短促回应并部分收住 -> 张鹏用生存/返航框架压住情绪
```

Bad realization:

```text
张鹏：别跟地球赌气。
```

Why bad:

- It is a therapy-like abstraction.
- It converts a concrete flight-risk correction into psychological diagnosis.
- It is not close to Zhang Peng's prior terse, concrete speech.

Better posture:

```text
dialogue_function: safety warning with care
tone: short, concrete, cockpit-facing
safe examples:
- 别较劲，先稳住高度。
- 先看仪表，别看下面。
- 慢点，姿态先稳住。
```

The final writer can choose wording, but the simulation makes the safer
interaction function explicit.

## Open Questions

- Should action candidate scoring be globally tuned or per genre/story?
- Should character state numbers be deterministic rules, LLM-estimated, or a
  hybrid?
- Should social simulation run multiple samples and use verifier/reranker, or
  stay deterministic for benchmark comparability?
- How much of the selected beat sequence should be exposed to final writing
  before it becomes over-directive?
- Should previous scene context influence social simulation, or only final
  writing? Current recommendation: use previous scene context only after social
  planning unless the previous scene is already represented in prefix memory.
