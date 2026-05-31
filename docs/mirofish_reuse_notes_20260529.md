# MiroFish Reuse Notes

## Source

- Zip inspected: local `MiroFish-main.zip` archive, not included in this repository.
- Temporary extraction path used for inspection: a disposable `/tmp` directory.
- Inspection date: 2026-05-29

## High-Level Summary

MiroFish is a full-stack multi-agent simulation application rather than a small agent library. Its workflow is:

```text
seed material
  -> ontology generation
  -> graph memory construction with Zep
  -> entity filtering
  -> agent profile generation
  -> simulation config generation
  -> OASIS social simulation
  -> dynamic graph-memory updates
  -> report agent and interactive interviews
```

Its stated goal is "swarm intelligence prediction": use seed materials to build a parallel digital world, generate many social agents, simulate their interactions, and then report on the predicted outcome.

For Diegetic Memory Society, MiroFish is useful mostly as an architectural reference. Direct code reuse is risky because the backend declares `AGPL-3.0` in `backend/pyproject.toml`. Treat the code as inspiration unless license compatibility is explicitly acceptable.

## Relevant Files Read

- `README.md`
- `README-ZH.md`
- `backend/pyproject.toml`
- `backend/app/models/project.py`
- `backend/app/models/task.py`
- `backend/app/services/text_processor.py`
- `backend/app/services/ontology_generator.py`
- `backend/app/services/graph_builder.py`
- `backend/app/services/zep_entity_reader.py`
- `backend/app/services/oasis_profile_generator.py`
- `backend/app/services/simulation_config_generator.py`
- `backend/app/services/simulation_manager.py`
- `backend/app/services/simulation_runner.py`
- `backend/app/services/zep_graph_memory_updater.py`
- `backend/app/services/zep_tools.py`
- `backend/app/services/report_agent.py`
- `backend/app/api/simulation.py`

## What Is Directly Useful For DMS

### 1. Project And Task Lifecycle

MiroFish has a clear persistent project model:

```text
Project
  - uploaded files
  - extracted text
  - ontology
  - graph_id
  - status
  - error
```

It also has a `TaskManager` with:

- task id;
- task type;
- status;
- progress percentage;
- status message;
- result;
- error;
- metadata.

DMS should adopt the same idea:

```text
DMSProject
  - manuscript files
  - prefix checkpoints
  - extracted narrative memory
  - generated plans
  - validation reports

DMSTask
  - ingest_prefix
  - build_memory
  - generate_plan
  - validate_plan
  - evaluate_against_masked_future
```

This matters because DMS experiments will involve long-running LLM extraction, graph building, and generation. A durable task lifecycle will make experiments reproducible and easier to inspect.

### 2. Ontology Generation Before Graph Construction

MiroFish first generates an ontology, then builds a graph. This is useful for DMS, but the ontology should be narrative-specific rather than social-media-specific.

MiroFish ontology:

```text
entity_types
edge_types
analysis_summary
```

DMS ontology should instead generate:

```text
narrative_entity_types:
  - Character
  - Location
  - Object
  - Organization
  - Secret
  - Promise
  - WorldRule
  - Conflict

narrative_edge_types:
  - KNOWS
  - BELIEVES
  - MISBELIEVES
  - WANTS
  - HIDES
  - OBSERVES
  - TELLS
  - CAUSES
  - BLOCKS
  - FORESHADOWS
  - PAYS_OFF
```

The key idea is not the exact prompt, but the **schema-first graph build**. For DMS, this lets us adapt to detective fiction, romance, fantasy, political drama, scripts, or literary novels.

### 3. Entity-To-Agent Profile Generation

MiroFish does not simulate raw graph nodes directly. It converts graph entities into OASIS agent profiles:

```text
entity
  -> enriched context from graph
  -> LLM-generated persona
  -> activity configuration
  -> platform profile
```

DMS should copy this pattern conceptually:

```text
Character entity
  -> visible prefix evidence
  -> CharacterMemoryProfile
  -> planning-time role agent
```

DMS profile fields should be:

```json
{
  "character_id": "...",
  "name": "...",
  "role_in_story": "...",
  "known_facts": [],
  "beliefs": [],
  "misbeliefs": [],
  "goals": [],
  "fears": [],
  "emotional_state": "...",
  "relationship_states": [],
  "secrets_known": [],
  "secrets_hidden_from": [],
  "active_constraints": [],
  "evidence_spans": []
}
```

This is one of the strongest borrowable ideas: **convert extracted knowledge into role-specific agent profiles before generation**.

### 4. Stepwise Configuration Generation

MiroFish avoids asking the LLM to generate one huge configuration. It decomposes config generation into:

1. time config;
2. event config;
3. batched agent configs;
4. platform config.

DMS should similarly decompose next-chapter generation setup:

```text
Step 1: prefix boundary state
Step 2: active narrative constraints
Step 3: character-specific memory profiles
Step 4: unresolved thread inventory
Step 5: planning request configuration
Step 6: continuation plan generation
```

This is better than one huge "summarize everything and write the next chapter" prompt.

### 5. Dynamic Memory Update From Actions

MiroFish converts simulation actions into natural-language episodes and writes them back into Zep graph memory. Example pattern:

```text
AgentAction
  -> natural language episode
  -> batch queue
  -> graph.add(...)
```

DMS can use a safer version:

```text
Generated beat / generated scene / author-accepted edit
  -> narrative episode
  -> sandbox memory graph
```

Important difference:

- Original manuscript memory must remain immutable.
- Generated or simulated continuations should go into a separate sandbox branch.
- Only author-accepted output should be promoted into canonical memory.

Recommended DMS memory layers:

```text
canonical_prefix_memory
candidate_branch_memory
accepted_draft_memory
evaluation_masked_future_memory
```

This prevents generated guesses from contaminating the gold prefix memory.

### 6. Report Agent With Tool-Call Trace Logging

MiroFish's `ReportLogger` records:

- report start;
- planning context;
- outline;
- ReACT thought;
- tool call;
- tool result;
- LLM response;
- section content;
- final report.

DMS should copy the logging discipline for every generation:

```text
generation_log.jsonl
  - selected prefix checkpoint
  - agent packets
  - visibility-gate decisions
  - blocked facts
```

## Source-Verified Details: Agent Profiles

MiroFish separates two related concepts:

1. **OASIS Agent Profile**: who the simulated account/agent is.
2. **Agent Activity Config**: when and how strongly the agent participates in simulation.

### OASIS Agent Profile Fields

`backend/app/services/oasis_profile_generator.py` defines `OasisAgentProfile`.

Core fields:

```text
user_id
user_name / username
name
bio
persona
karma
friend_count
follower_count
statuses_count
age
gender
mbti
country
profession
interested_topics
source_entity_uuid
source_entity_type
created_at
```

The most important fields are:

| Field | Role |
| --- | --- |
| `bio` | short public profile or account description |
| `persona` | detailed behavioral/personality description used to guide the LLM agent |
| `source_entity_uuid` | provenance link back to the graph entity |
| `source_entity_type` | graph/entity type used to build the profile |

Twitter and Reddit require different output shapes:

| Platform | Output file | Notes |
| --- | --- | --- |
| Twitter | `twitter_profiles.csv` | includes `user_char`, which combines `bio + persona` for the LLM agent prompt |
| Reddit | `reddit_profiles.json` | includes `bio`, `persona`, demographics, karma, and interests |

For Twitter export, MiroFish writes:

```text
user_id
name
username
user_char      # full persona text: bio + persona
description    # short profile text
```

For Reddit export, MiroFish writes JSON records like:

```json
{
  "user_id": 0,
  "username": "...",
  "name": "...",
  "bio": "...",
  "persona": "...",
  "karma": 1000,
  "created_at": "...",
  "age": 25,
  "gender": "male",
  "mbti": "INTJ",
  "country": "中国",
  "profession": "...",
  "interested_topics": ["..."]
}
```

### How Profile Content Is Generated

MiroFish builds profile context from Zep graph entities:

```text
entity attributes
related edges / facts / relations
related node summaries
extra graph search results from Zep
```

Then it distinguishes individual entities from group/institution entities.

Individual-like entity types include:

```text
student
alumni
professor
person
publicfigure
expert
faculty
official
journalist
activist
```

Group/institution-like entity types include:

```text
university
governmentagency
organization
ngo
mediaoutlet
company
institution
group
community
```

For individual entities, the generated `persona` is prompted to include:

```text
basic information
background and relation to the event
social relationships
personality and MBTI-like traits
emotion expression style
social-media behavior
stance toward the topic
triggers and motivations
unique habits or catchphrases
personal memory about the event
```

For group/institution entities, the profile becomes a representative account:

```text
institution identity
account positioning
target audience
speaking style
taboo topics
communication strategy
stance toward the event
role in public opinion
```

### Agent Activity Config Fields

`backend/app/services/simulation_config_generator.py` defines `AgentActivityConfig`.
This is not the same as profile. It controls simulation behavior:

```text
agent_id
entity_uuid
entity_name
entity_type
activity_level
posts_per_hour
comments_per_hour
active_hours
response_delay_min
response_delay_max
sentiment_bias
stance
influence_weight
```

For DMS, the split is useful:

```text
CharacterMemoryProfile
  - who the character is at story time t
  - known facts, beliefs, false beliefs, goals, emotions, relationships

CharacterActivityPolicy
  - how likely this character is to act or speak
  - what style of action is plausible now
  - whether the character is central or peripheral in the current scene
```

MiroFish's exact social-media fields should not be copied directly into DMS, but
the two-layer split is worth preserving.

## Source-Verified Details: OASIS Simulation Flow

MiroFish's OASIS workflow has two phases: preparation and execution.

### Preparation Phase

The API endpoint `/api/simulation/prepare` starts a background task. The main
implementation is `SimulationManager.prepare_simulation`.

The preparation chain is:

```text
load simulation state
  -> read entities from Zep graph
  -> filter selected entity types
  -> generate OASIS profiles
  -> save reddit_profiles.json / twitter_profiles.csv
  -> generate simulation_config.json
  -> mark simulation READY
```

The generated files expected by the runner are:

```text
state.json
simulation_config.json
reddit_profiles.json
twitter_profiles.csv
```

`simulation_config.json` contains:

```text
simulation_id
project_id
graph_id
simulation_requirement
time_config
agent_configs
event_config
twitter_config
reddit_config
llm_model
llm_base_url
generation_reasoning
```

`SimulationConfigGenerator` generates this config stepwise:

```text
1. time_config
2. event_config
3. batched agent_configs
4. platform configs
```

The event config includes:

```text
initial_posts
scheduled_events
hot_topics
narrative_direction
```

### Start Phase

`/api/simulation/start` calls `SimulationRunner.start_simulation`.

The runner:

```text
loads simulation_config.json
computes total_rounds from total_simulation_hours and minutes_per_round
chooses a script:
  twitter  -> run_twitter_simulation.py
  reddit   -> run_reddit_simulation.py
  parallel -> run_parallel_simulation.py
starts the script as a subprocess
starts a monitor thread
writes run_state.json
```

Main runtime logs:

```text
sim_xxx/
  twitter/actions.jsonl
  reddit/actions.jsonl
  simulation.log
  run_state.json
```

### OASIS Environment Construction

In `backend/scripts/run_parallel_simulation.py`, each platform builds an OASIS
agent graph and environment.

Twitter:

```text
model = ModelFactory.create(...)
agent_graph = generate_twitter_agent_graph(
  profile_path="twitter_profiles.csv",
  model=model,
  available_actions=TWITTER_ACTIONS
)
env = oasis.make(
  agent_graph=agent_graph,
  platform=oasis.DefaultPlatformType.TWITTER,
  database_path="twitter_simulation.db",
  semaphore=30
)
await env.reset()
```

Reddit:

```text
model = ModelFactory.create(...)
agent_graph = generate_reddit_agent_graph(
  profile_path="reddit_profiles.json",
  model=model,
  available_actions=REDDIT_ACTIONS
)
env = oasis.make(
  agent_graph=agent_graph,
  platform=oasis.DefaultPlatformType.REDDIT,
  database_path="reddit_simulation.db",
  semaphore=30
)
await env.reset()
```

Twitter actions enabled by MiroFish:

```text
CREATE_POST
LIKE_POST
REPOST
FOLLOW
DO_NOTHING
QUOTE_POST
```

Reddit actions enabled by MiroFish:

```text
LIKE_POST
DISLIKE_POST
CREATE_POST
CREATE_COMMENT
LIKE_COMMENT
DISLIKE_COMMENT
SEARCH_POSTS
SEARCH_USER
TREND
REFRESH
DO_NOTHING
FOLLOW
MUTE
```

### Round 0: Initial Event Injection

Before the main loop, MiroFish reads `event_config.initial_posts`.

Each initial post has:

```text
poster_agent_id
content
```

The runner converts these into manual OASIS actions:

```text
ManualAction(
  action_type=ActionType.CREATE_POST,
  action_args={"content": content}
)
```

Then:

```text
await env.step(initial_actions)
```

This is the initial public event that seeds the simulated social environment.

### Main Simulation Loop

For each round:

```text
compute simulated hour
select active agents using time_config and agent_configs
construct actions = {agent: LLMAction()}
await env.step(actions)
read actual actions from OASIS sqlite trace table
write simplified actions to actions.jsonl
record round_start and round_end
```

Active-agent selection uses:

```text
active_hours
activity_level
agents_per_hour_min
agents_per_hour_max
peak/off-peak multipliers
random sampling
```

So the profile controls identity and speaking style, while `agent_configs`
control participation dynamics.

### Action Logging

OASIS writes raw action traces into SQLite. MiroFish reads the trace table and
converts actions into simplified JSONL records:

```json
{
  "round": 3,
  "agent_id": 12,
  "agent_name": "...",
  "action_type": "CREATE_POST",
  "action_args": {
    "content": "..."
  }
}
```

The wrapper enriches actions with context when possible:

```text
liked post content
original repost content
comment content
target user name
quoted post/comment content
```

The monitor thread reads:

```text
twitter/actions.jsonl
reddit/actions.jsonl
```

and updates `run_state.json` for frontend polling.

### Optional Graph Memory Update

If `/start` enables `enable_graph_memory_update`, `SimulationRunner` creates a
`ZepGraphMemoryManager` updater and sends agent activities from `actions.jsonl`
back into Zep graph memory.

For DMS, this pattern is useful but must be adapted:

```text
generated action / beat / scene
  -> candidate_branch_memory
  -> optional author promotion
  -> accepted_draft_memory
```

Generated simulation output must not automatically update canonical prefix
memory.

### Interview Mode

The parallel runner keeps environments alive after simulation unless `--no-wait`
is passed. It then processes IPC commands:

```text
interview
batch_interview
close_env
```

Interview is implemented as:

```text
ManualAction(
  action_type=ActionType.INTERVIEW,
  action_args={"prompt": prompt}
)
await env.step(actions)
```

The result is read back from the OASIS SQLite trace.

This is relevant for DMS because a future prototype could support:

```text
interview character agent at story time t
ask why the character would choose a branch
ask what the character believes or fears
```

but DMS must route that interview through the visibility gate first.

## DMS Adaptation Notes

MiroFish profile:

```text
entity -> enriched graph context -> social-media persona -> OASIS account
```

DMS adaptation:

```text
character/entity
  -> visibility-gated prefix evidence
  -> CharacterMemoryProfile
  -> timeline-constrained narrative simulation
```

Recommended DMS profile:

```json
{
  "character_id": "...",
  "name": "...",
  "story_time": "...",
  "known_facts": [],
  "beliefs": [],
  "false_beliefs": [],
  "goals": [],
  "fears": [],
  "emotional_state": "...",
  "relationship_states": [],
  "secrets_known": [],
  "secrets_hidden_from": [],
  "active_threads": [],
  "world_constraints": [],
  "evidence_spans": []
}
```

Important implementation rule for DMS:

> Profile generation must come after visibility filtering, not before it.

If a profile is generated from omniscient story context, the character agent may
inherit future knowledge or hidden facts. That would break the central DMS
claim.
  - generated beats
  - constraint validation results
  - final plan/prose
```

This is important for research evaluation. Without trace logs, it will be hard to explain why a generated continuation violated a narrative constraint.

### 7. Interactive Agent Interview

MiroFish supports interviewing simulated agents. For DMS, the analogous feature is:

```text
Ask a character at prefix time t.
```

Examples:

- "What does Alice think Basil is hiding at the end of chapter 10?"
- "What would Basil refuse to tell Alice right now?"
- "What does Mara want from the next scene?"

This is useful as a writing interface, but it must pass through the DMS visibility gate.

## What Is Less Useful Or Not Recommended

### 1. OASIS Social-Media Simulation

OASIS is designed for social-platform interaction simulation. DMS is about long-form narrative writing. The event loop, Twitter/Reddit platform logic, likes/reposts/comments, and social-media activity schedules should not be reused directly.

Borrow the idea of simulation artifacts, not the platform simulation itself.

### 2. Zep Cloud As A Hard Dependency

MiroFish depends heavily on Zep Cloud. DMS can optionally support Zep-like graph memory, but should not require it for a research prototype.

For DMS, a local-first stack is better:

- SQLite or DuckDB for structured facts and constraints;
- NetworkX for graph traversal;
- Chroma/Faiss for vector retrieval;
- JSONL artifacts for benchmark reproducibility.

Zep can be a later adapter.

### 3. Social-Media Ontology Prompts

MiroFish ontology prompts are very specific to public opinion simulation. They force exactly 10 entity types and require social-media actors. This does not fit fiction.

DMS needs narrative ontology generation and stable cross-story schema.

### 4. Direct Code Copy

The backend is AGPL-3.0. Unless DMS is intended to be AGPL-compatible, do not copy code directly. Reimplement patterns and data models.

## Recommended DMS Architecture Inspired By MiroFish

```text
DMSProject
  |
  v
Manuscript ingestion
  |
  v
Narrative ontology/schema selection
  |
  v
Prefix graph build
  - events
  - facts
  - character states
  - secrets/reveals
  - threads
  - constraints
  |
  v
Character profile generation
  - profile per major character at prefix boundary
  - evidence-bound
  - visibility-filtered
  |
  v
Planning config generation
  - target chapter
  - active arcs
  - allowed reveals
  - constraints to preserve
  |
  v
Continuation planning agent
  |
  v
Constraint validator
  |
  v
Candidate branch memory
  |
  v
Human selection / author acceptance
```

## Concrete Modules To Implement In DMS

### `dms_project.py`

Persistent project model:

```text
project_id
name
status
source_files
prefix_checkpoints
canonical_memory_path
branch_memory_paths
created_at
updated_at
```

### `dms_task.py`

Task manager modeled after MiroFish:

```text
task_id
task_type
status
progress
message
result
error
metadata
```

### `narrative_schema_generator.py`

Generate or select a narrative schema for a project:

```text
entity types
fact types
edge types
constraint types
beat categories
```

For an MVP, use a fixed schema rather than LLM-generating one for every story.

### `character_profile_generator.py`

Convert DMS memory into character profiles at a prefix boundary.

This is the closest DMS equivalent to MiroFish's `OasisProfileGenerator`.

### `planning_config_generator.py`

Generate a compact config for the continuation agent:

```text
target unit
viewpoint character
active characters
active conflicts
allowed reveal policy
must-satisfy constraints
branch count
output mode: outline/prose
```

### `branch_memory_updater.py`

Convert generated beats or author-accepted scenes into memory episodes.

Rules:

- write generated content to branch memory only;
- never mutate canonical prefix memory during speculative generation;
- allow author promotion from branch to accepted memory.

### `generation_logger.py`

Log all memory packets, visibility decisions, generated plans, and validation results as JSONL.

## Suggested MVP Borrowing Plan

1. Implement DMS project/task lifecycle inspired by MiroFish.
2. Implement a fixed narrative schema; do not build dynamic ontology first.
3. Build prefix memory from a small story using JSONL artifacts.
4. Generate character profiles from prefix memory.
5. Generate next-chapter plans using a coordinator agent.
6. Validate generated plans against active narrative constraints.
7. Store generated plans as candidate branch memory.
8. Add masked-future evaluation only after the pipeline runs end-to-end.

## Bottom Line

MiroFish is not the right codebase to directly fork for Diegetic Memory Society, because its core simulation target is social-media swarm prediction and its backend license is AGPL-3.0.

However, it provides strong architectural ideas:

- project/task lifecycle;
- schema-first graph building;
- entity-to-agent profile conversion;
- stepwise simulation configuration;
- dynamic memory updates from agent actions;
- durable action logs and report logs;
- post-simulation interaction/interview.

The most valuable DMS adaptation is:

> Convert prefix narrative memory into evidence-bound character profiles and planning constraints, generate candidate continuations in a sandbox branch, then validate and log every visibility and consistency decision.
