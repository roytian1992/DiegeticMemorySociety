# Social Simulation Research Notes

Date: 2026-06-01

This note reviews the local `AgentSims-main.zip` package and recent LLM-agent
social simulation work. The goal is to decide what should influence the next
version of DMS social simulation, especially the proposed algorithmic planner in
`SOCIAL_SIMULATION_DESIGN.md`.

## Local AgentSims Review

Source inspected:

```text
/vepfs-mlp2/c20250513/241404044/users/roytian/downloads/AgentSims-main.zip
```

The archive is the public 2023 AgentSims codebase for "An Open-Source Sandbox
for Large Language Model Evaluation". It contains:

- a Unity WebGL client;
- a Tornado/WebSocket Python server;
- MySQL-backed map, NPC, building, equipment, account, and evaluation models;
- an NPC `Actor` wrapper around an LLM `Agent`;
- a `Mayor` agent that can create buildings and NPCs;
- prompt templates for question answering, planning, acting, chatting,
  critiquing, memory storage, and mayor decisions;
- tick logs and QA-style evaluation logs.

The useful mechanism is a tick-driven embodied loop, not a single generation
prompt:

```text
inited
  -> QA framework asks/answers goal-relevant planning questions
  -> plan chooses a building and purpose
  -> map navigation moves the NPC
  -> act chooses use/chat/experience
  -> use or multi-turn chat executes
  -> critic decides success/fail/not_finished_yet
  -> memory_store updates impressions and episodic memory
  -> next plan
```

Key code paths:

- `agent/actor.py`: dispatches observations such as `inited`,
  `timetick-finishMoving`, `timetick-finishUse`, `chatted`, and
  `timetick-finishChatting` into plan/act/chat/critic/memory steps.
- `agent/agent/agent.py`: implements `plan`, `act`, `chat`, `use`, `critic`,
  `memory_store`, and experience reuse.
- `command/timetick/Tick.py`: advances game time, moves entities, resolves
  use/chat/init queues, executes evaluation, and routes environment feedback
  back to actors.
- `agent/agent/components/memory_store.py`: keeps people impressions,
  building impressions, episodic memories, and reusable experience traces.
- `agent/prompt/*.txt`: defines the LLM schemas. Most outputs are compact JSON.

AgentSims has three ideas worth borrowing:

- **Environment-mediated interaction**: agents do not decide in a vacuum. They
  receive visible people/equipment/cash/time from the environment before acting.
- **Action loop with critic**: plan completion is checked before memory is
  updated, which gives a natural place for validators.
- **Memory as updated state**: conversation and action results change
  impressions and episodic memories, rather than staying as transient prompt
  text.

AgentSims is weaker for our use case in four ways:

- The social reasoning itself is still mostly prompt-only; there is no explicit
  pressure graph, utility function, private goal model, or beat search.
- Dialogue is generated directly, so wording can become canonical too early.
- The memory system stores impressions and episodes, but it does not score
  relevance/recency/importance or separate supported facts from inferred local
  pressures.
- Evaluation is QA-form based; it does not measure social interaction quality,
  action plausibility, or writing usefulness.

For DMS, AgentSims should be treated as an architecture reference for
state/event loops, not as the target social simulation algorithm.

## Recent Representative Work

### Generative Agents / Smallville

Reference: Park et al., "Generative Agents: Interactive Simulacra of Human
Behavior" (2023). Paper: <https://arxiv.org/abs/2304.03442>. Official project:
<https://github.com/joonspk-research/generative_agents>.

Core mechanism:

- every agent has a memory stream;
- retrieval combines recency, importance, and relevance;
- reflection periodically synthesizes higher-level memories;
- planning decomposes daily intent into actions;
- agents observe the world, retrieve memories, decide actions, and interact.

What it contributes to DMS:

- Use a memory scoring model instead of passing all memory uniformly.
- Separate observed facts, reflections, and plans.
- Treat social simulation as a loop over observation, retrieval, planning, and
  action, even if our "world" is only a scene frame.

Limit for DMS:

- Smallville optimizes believable daily behavior. DMS needs short narrative
  scenes with strict prefix-memory boundaries and target writing anchors, so the
  planner needs stronger constraints and less open-ended daily scheduling.

### AgentSims

Reference: Lin et al., "AgentSims: An Open-Source Sandbox for Large Language
Model Evaluation" (2023). Paper: <https://arxiv.org/abs/2308.04026>. Public
repository: <https://github.com/py499372727/AgentSims>.

Core mechanism:

- sandbox environment with NPCs, buildings, equipment, map movement, time ticks,
  mayor decisions, and QA-style evaluations;
- agents can plan, act, chat, use equipment, critique completion, and update
  memory;
- developers can customize tasks by creating agents/buildings and changing
  prompts.

What it contributes to DMS:

- Keep an explicit simulator controller between character state and generated
  action.
- Use typed JSON artifacts for plan/action/critic/memory.
- Store intermediate prompts and outputs for debugging.

Limit for DMS:

- It is closer to an evaluation sandbox than a social reasoning algorithm. It
  does not supply the relationship-pressure scoring we need.

### SOTOPIA

Reference: Zhou et al., "SOTOPIA: Interactive Evaluation for Social
Intelligence in Language Agents" (2023/2024). Paper:
<https://arxiv.org/abs/2310.11667>. Official repository:
<https://github.com/sotopia-lab/sotopia>.

Core mechanism:

- two-agent social episodes with role profiles, public setup, private goals,
  relationship context, and turn-by-turn interaction;
- evaluation measures multiple social dimensions rather than only task success;
- private goals and hidden information make interaction less reducible to
  surface dialogue.

What it contributes to DMS:

- Give each character a public role and private local intention.
- Represent social success as a vector: goal progress, relationship management,
  believability, knowledge consistency, and norm/constraint compliance.
- Do not let the writer see all simulator internals as facts; private
  intentions can shape behavior without becoming exposition.

Limit for DMS:

- SOTOPIA is built for agent evaluation, not prose generation. We should borrow
  its private-goal and multi-dimensional evaluation ideas, then convert outputs
  into writing-facing beat guidance.

### Concordia

Reference: "Generative Agent-Based Modeling with Actions Grounded in Physical,
Social, or Digital Space using Concordia" (2023). Paper:
<https://arxiv.org/abs/2312.03664>. Official repository:
<https://github.com/google-deepmind/concordia>.

Core mechanism:

- a framework for generative agent-based modeling;
- agents are built from components such as memory, observation, identity,
  goals, and action selection;
- a world controller or game master applies rules, mediates observations, and
  records measurements.

What it contributes to DMS:

- Use a game-master/controller layer to decide what each simulated character is
  allowed to know and do.
- Treat measurements as first-class outputs, not afterthoughts.
- Make the simulator modular: state, pressure graph, candidate generation,
  verifier, and writer packet can evolve separately.

Limit for DMS:

- Concordia is a general framework. DMS needs a narrower narrative planner with
  strict authority ordering: writing intent first, memory constraints second,
  social simulation as optional behavior guidance.

### AgentSociety

Reference: "AgentSociety: Large-Scale Simulation of LLM-Driven Generative
Agents Advances Understanding of Human Behaviors and Society" (2025). Paper:
<https://arxiv.org/abs/2502.08691>. Official repository:
<https://github.com/tsinghua-fib-lab/agentsociety>.

Core mechanism:

- large-scale simulation with many LLM-driven agents;
- separates agent need/plan/behavior sequence from environment and societal
  infrastructure;
- emphasizes scalable simulation, social behavior emergence, and city-like
  settings.

What it contributes to DMS:

- Separate internal needs and plans from outward behavior sequences.
- Use structured behavior sequences as intermediate products before natural
  language realization.
- Keep simulation artifacts inspectable at scale.

Limit for DMS:

- Large-scale urban simulation is overkill for scene writing. DMS should borrow
  the need-plan-behavior separation, not the full city simulator.

### OASIS

Reference: "OASIS: Open Agent Social Interaction Simulations with One Million
Agents" (2024/2025). Paper: <https://arxiv.org/abs/2411.11581>. Official
repository: <https://github.com/camel-ai/oasis>.

Core mechanism:

- scalable social-media simulation with many agents;
- combines LLM agents with an environment that controls observation, action
  channels, and interaction propagation;
- focuses on social interaction dynamics at population scale.

What it contributes to DMS:

- Separate action channels from final language. A character may "correct",
  "deflect", "probe", "comply", or "withhold" before we decide exact wording.
- Environment/controller matters: who can observe whom, what state is public,
  and what responses are triggered.

Limit for DMS:

- Population-level diffusion is not our problem. Our problem is dense,
  character-level interaction in short passages.

### Self-Report Grounded Individual Simulations

Reference: Park et al., "LLM Agents Grounded in Self-Reports Enable
General-Purpose Simulation of Individuals" (2024, revised 2026). Paper:
<https://arxiv.org/abs/2411.10109>. The original 2024 title was "Generative
Agent Simulations of 1,000 People".

Core mechanism:

- creates simulation agents from interviews and surveys;
- evaluates whether agents reproduce attitudes and behavioral patterns;
- focuses on fidelity to human-derived profiles and population-level validity.

What it contributes to DMS:

- Character cards should be treated as profile constraints, not optional prompt
  decoration.
- Evaluation should check whether generated actions are consistent with the
  profile evidence, not only whether prose is fluent.

Limit for DMS:

- DMS characters are fictional and evidence comes from prefix narrative memory,
  not interviews. The profile-fidelity idea still applies.

### MACHIAVELLI

Reference: Pan et al., "MACHIAVELLI: Measuring Intelligence and Ethics in AI
Agents" (2023). Paper: <https://arxiv.org/abs/2304.03279>. Project:
<https://aypan17.github.io/machiavelli/>.

Core mechanism:

- evaluates agents in text-adventure-like social decision settings;
- measures reward alongside behavior dimensions such as deception, coercion,
  harm, and ethical/social costs.

What it contributes to DMS:

- Social simulation should not optimize "dramatic progress" alone.
- Candidate actions should carry risk labels: unsupported coercion,
  unsupported hierarchy, relationship distortion, tone mismatch, or future
  leakage.

Limit for DMS:

- MACHIAVELLI is an evaluation benchmark, not a scene planner. We should borrow
  risk-aware scoring rather than its task format.

### Survey View

Reference: "From Individual to Society: A Survey on Social Simulation Driven by
Large Language Model-based Agents" (2024). Paper:
<https://arxiv.org/abs/2412.03563>.

Useful synthesis:

- Modern LLM social simulation tends to separate agent profiling, memory,
  planning, environment, interaction, and evaluation.
- The field is moving from single-agent prompting toward structured multi-agent
  systems with explicit state and measurement.

For DMS, this confirms the direction: social simulation should be a structured
planning layer, not just another prose prompt.

## Implications For DMS

The current DMS social simulation already has some right pieces:

```text
memory packet
  -> evidence-bound attribute cards
  -> per-character simulation
  -> coordinator beats
  -> writing prompt as optional guidance
```

But it is still too close to AgentSims-style prompt orchestration. The next
version should borrow the stronger pieces from recent work:

| Need | Borrow From | DMS Mechanism |
| --- | --- | --- |
| Memory relevance and abstraction | Generative Agents | score memory by relevance/recency/importance/support |
| Environment/controller | AgentSims, Concordia, OASIS | scene controller mediates visibility, allowed actions, and validation |
| Private goals and social dimensions | SOTOPIA | public setup plus private local intentions, relationship pressure, hidden resistance |
| State/action separation | AgentSociety, OASIS | typed action candidates before final wording |
| Risk-aware optimization | MACHIAVELLI | penalties for unsupported roles, future leak, therapy phrasing, relationship distortion |
| Profile fidelity | 1,000-agent simulations | character-card consistency metrics |

The most important design choice is to separate five layers:

```text
1. Evidence layer
   prefix memories, relations, summaries, previous scene context, writing intent

2. Character state layer
   roles, goals, affect, pressures, relationship stances, hard constraints

3. Simulation layer
   pressure graph, private goals, action candidates, state deltas, beat search

4. Verification layer
   support checks, contradiction checks, phrasing/style risk, target leak checks

5. Writing-facing layer
   optional beat posture, dialogue function, avoid phrases, not canonical lines
```

This prevents the current failure mode where a supported memory such as
`地球一点都不美好` becomes an awkward final line like `别跟地球赌气`.

## Recommended ASIP Upgrade

The ASIP design in `SOCIAL_SIMULATION_DESIGN.md` is directionally correct. This
research suggests strengthening it in five places.

### 1. Add Public/Private Scene State

For each character:

```json
{
  "public_state": {
    "visible_role": "驾驶舱内的飞行参与者",
    "visible_action_state": "正在返航飞行中",
    "known_to_others": ["情绪紧张", "操作节奏偏快"]
  },
  "private_state": {
    "local_goal": "靠近天空/离开地面语境",
    "resistance": "不愿完全接受劝稳",
    "withheld_information": []
  }
}
```

Private state should influence behavior, but the writer should not be forced to
explain it directly.

### 2. Add Memory Scoring

Every memory used by the simulator should get:

```text
memory_score =
  0.35 * semantic_relevance_to_scene_problem
+ 0.20 * relationship_relevance
+ 0.20 * recency
+ 0.15 * explicitness
+ 0.10 * importance
```

Use high-scoring memories to form state and pressure. Low-scoring memories can
remain in the packet but should not drive candidate actions.

### 3. Use Action Types Before Dialogue

Candidate actions should be typed:

```text
risky_operation
safety_correction
minimal_compliance
value_resistance
deflection
care_reframe
information_probe
silence_or_withholding
physical_reaction
environmental_pressure
```

The simulator should choose an action type and function first. Dialogue comes
later as posture, not required wording.

### 4. Treat Beat Search As Constrained Optimization

Each beat sequence should be scored:

```text
sequence_score =
  0.25 * writing_intent_coverage
+ 0.20 * pressure_arc
+ 0.15 * character_state_consistency
+ 0.15 * memory_support
+ 0.10 * relationship_expression
+ 0.10 * compactness
+ 0.05 * novelty_without_contradiction
- 0.30 * hard_violation
- 0.15 * unsupported_inference
- 0.10 * awkward_phrase_risk
```

Hard constraints should reject a sequence, not merely lower its score.

### 5. Add Simulation Metrics To Benchmark Output

Report simulation quality separately from writing quality:

```text
pressure_graph_edge_count
selected_sequence_score
scene_problem_coverage
relationship_coverage
private_goal_usage_rate
memory_support_rate
intent_basis_precision
hard_violation_count
soft_warning_count
awkward_phrase_risk_count
writer_overdirective_score
```

This lets us see whether social simulation is improving the interaction layer
even when the final writing judge gives both drafts a high score.

## Concrete Implementation Roadmap

### Phase A: Guard Current Prompt-Based Simulation

Keep the current pipeline, but add a verifier and risk annotator.

Files:

```text
src/dms/simulation/verification.py
tests/test_social_simulation_verification.py
```

Checks:

- all `memory_basis` refs exist;
- intent-only details are not cited as memory;
- unsupported formal roles are flagged;
- future/target-scene leakage is flagged;
- final-dialogue-like text is marked as example only;
- phrases like `别跟地球赌气`, `和自己和解`, `你是在逃避`, `放下` are flagged as
  modern psychological paraphrase risks.

This is the cheapest immediate improvement.

### Phase B: Add State And Pressure Artifacts

Add deterministic/interpretable outputs:

```text
social_state_graph.json
pressure_graph.json
candidate_actions.json
```

The first version can use rules plus optional LLM extraction. It does not need
to replace the current coordinator immediately.

### Phase C: Add Candidate Scoring And Top-K Beat Search

Generate 3-6 action candidates per character, score them, then search for 3-5
beat sequences. Keep top `K=5`, expose the best sequence and rejected/repaired
alternatives.

### Phase D: Convert To Writer Packet

The writer packet should include:

- selected interaction function;
- optional behavior beats;
- dialogue posture;
- avoid phrases;
- support/risk summary.

It should not include canonical dialogue lines unless explicitly labeled as
examples.

### Phase E: Evaluate Social Simulation As Its Own Artifact

Add a benchmark section before final writing:

```json
{
  "social_simulation_metrics": {
    "scene_problem_coverage": 1.0,
    "relationship_coverage": 1.0,
    "memory_support_rate": 0.86,
    "hard_violation_count": 0,
    "awkward_phrase_risk_count": 1
  }
}
```

This gives us a way to compare:

- non-simulation writing;
- prompt-only social simulation;
- ASIP simulation;
- ASIP plus verifier/reranker.

## Bottom Line

AgentSims confirms that a useful simulator needs an event loop, environment
feedback, action execution, critic, and memory update. Recent work pushes that
further: high-quality social simulation needs profile-grounded state, private
goals, controlled action spaces, environment mediation, risk-aware scoring, and
separate evaluation metrics.

For DMS, the right target is not a full city/world simulator. It is a compact
scene-level social planner:

```text
evidence-bound character state
  -> public/private scene frame
  -> pressure graph
  -> typed action candidates
  -> constrained beat search
  -> verifier/reranker
  -> writer-facing posture packet
```

That is the strongest path from the current prompt-based social simulation to a
mechanism that can reliably improve narrative interaction without leaking
unsupported facts or over-determining the final prose.
