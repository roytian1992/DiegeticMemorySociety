# Diegetic Memory Society Research Plan

## 0. Working Title

**Diegetic Memory Society: Textual World Models as Memory-Agent Societies for Narrative Writing**

Short version:

**Diegetic Memory Society as a Textual World Model for Long-Form Writing Assistance**

## 1. Core Research Claim

Long-form narrative assistants should not treat story memory as a passive store attached to an agent. In the writing setting, memory is closer to a **textual world model**: an evolving, text-grounded representation of the story world that maintains entity states, event-induced state transitions, epistemic visibility, open narrative threads, and discourse-level reveal status.

The core idea is therefore not traditional **memory for agent**, where an agent retrieves facts from a memory module. The core idea is **memory as agent society**: the memory system itself is an active society of specialized operators that incrementally read the manuscript, update the textual world model, resolve visibility, maintain continuity-bearing entities, and construct task-specific memory packets for writing.

The intended setting is:

```text
The completed novel or script exists for evaluation.
The writing assistant only sees chapters 1..N.
The reference next unit N+1 is used only by the evaluator.
```

For fiction, the key constraint is **diegetic visibility**: whether a fact is available to a specific character, at a specific story-world time, under a specific epistemic status. A useful writing memory should preserve that visibility while supporting plausible continuation, character-state continuity, setup/payoff usage, and world-state consistency.

This project studies whether a **textual world model maintained by a memory-agent society** can improve narrative reasoning by decomposing memory maintenance into specialized operators:

- chapter ingestion operator;
- entity state trackers for characters, places, objects, groups, and world rules;
- timeline and state-transition operator;
- visibility and belief-state resolver;
- secret and reveal operator;
- foreshadowing and payoff operator;
- memory packet builder;
- branch memory simulator;
- consistency validator.

The expected contribution is not merely a new writing tool, but a formal and empirical framework for evaluating whether a memory system improves **prefix-conditioned narrative continuation**: who knows what, what remains unresolved, which arcs are active, and what developments are consistent with the story so far.

Relative to general agent-memory work, DMS should be positioned as a **token-level, evidence-bound textual world model**. It should not rely on parametric memory or latent memory as the primary mechanism, because writing assistance and prefix-only evaluation require provenance, editability, versioning, and explicit access control. The useful memory substrate is therefore external and inspectable: structured text records, JSON/SQL tables, graph edges, state-transition logs, hierarchical summaries, and retrieval packets grounded in source spans.

The textual world model should be built through **chapter-ordered injection**:

```text
read chapter 1
  -> update world model
read chapter 2
  -> update world model from prior state
...
read chapter N
  -> freeze canonical_prefix_memory
  -> generate / evaluate continuation
```

This sequential ingestion policy is a core design decision. Concurrent multi-chapter extraction can be useful for cheap preprocessing, but it should not be the primary memory-building method because it weakens state-transition modeling, risks using future context to interpret earlier chapters, and makes prefix-only evaluation less faithful to the writing scenario.

## 2. Research Questions

### RQ1: Prefix-Conditioned Narrative Consistency

Given only chapters `1..N`, can a chapter-ordered textual world model help produce a next-chapter plan or continuation that remains consistent with the story so far?

Hypothesis:

> A state-transition-, visibility-, and thread-aware textual world model will produce continuations with fewer narrative consistency violations than long-context prefix, vector RAG, summary-memory, and parallel-extraction baselines.

### RQ2: Reference-Trajectory Narrative Alignment

Can the original next unit be used as a reference trajectory to evaluate whether a prefix-only system follows a plausible story trajectory, without treating the original text as the only correct continuation?

Hypothesis:

> DMS will better satisfy writing intent, produce stronger writing, and remain more faithful to prefix memory than baselines, while still allowing alternative valid continuations.

### RQ3: Character State And Diegetic Visibility

Can the system preserve what each character knows, believes, wants, fears, misunderstands, and can plausibly do at the prefix boundary?

Hypothesis:

> Explicit character memory and visibility gates will improve character plausibility and reduce private-knowledge errors in generated plans and continuations.

### RQ4: Memory Faithfulness

Can the memory system preserve constraints established in the prefix, such as unresolved conflicts, object states, relationship states, secrets, promises, foreshadowing, and world rules?

Hypothesis:

> DMS will produce continuations with higher memory faithfulness by respecting prefix-derived entity states, relationships, unresolved threads, and world rules.

### RQ5: Memory-Operator Decomposition

Does the memory-agent society decomposition actually matter?

Hypothesis:

> Ablating state tracking, visibility resolution, secret/reveal handling, and thread tracking will degrade the corresponding consistency, alignment, and constraint-satisfaction metrics.

### RQ6: Chapter-Ordered World-Model Construction

Does sequential chapter injection produce a better writing memory than parallel multi-chapter extraction followed by post-hoc merging?

Hypothesis:

> Chapter-ordered injection will better preserve state transitions, reveal timing, character belief evolution, and prefix-boundary constraints than parallel extraction, especially for stories with delayed reveals, false beliefs, object handoffs, and relationship changes.

## 3. What The System Should Look Like

### 3.1 User-Facing Prototype

The prototype can be built as an authoring and analysis workspace with four main panels.

**Panel A: Manuscript / Script**

- Shows chapter or scene text.
- Supports selecting a story point, e.g. "Chapter 12 ending" or "Scene 34 before reveal".
- Highlights extracted events, characters, secrets, and foreshadowing cues.

**Panel B: Story-World Timeline**

- Displays events by story-world chronology, not just chapter order.
- Supports flashbacks, parallel scenes, uncertain temporal ordering, and delayed reveals.
- Each event links back to textual evidence.

**Panel C: Character Perspective**

- User chooses a character and a story time.
- The system shows:
  - known facts;
  - believed facts;
  - suspected facts;
  - false beliefs;
  - secrets hidden from the character;
  - facts outside the selected prefix or viewpoint.
- For writing support, only prefix-visible and task-appropriate facts should be passed into generation.

**Panel D: Memory Operators, Planning, And Draft Audit Extension**

- Character-state operator: "What would this character plausibly do now?"
- Secret/reveal operator: "Would this scene expose a hidden fact before the character or reader context supports it?"
- Timeline operator: "Does this event contradict earlier ordering?"
- Thread operator: "Which setups remain unresolved?"
- World-rule operator: "Does this violate established rules?"
- Planning coordinator: "Propose next-chapter branches that satisfy active constraints."
- Validator, future extension: "List high-risk consistency issues with evidence."

The interface should always show evidence spans. The author should be able to reject or revise agent claims.

### 3.2 System Architecture

The system is a chapter-ordered textual world-model builder plus a memory-agent society. It should update the model in the same order an author or reader encounters the story, rather than extracting all chapters in parallel and merging after the fact.

```text
Manuscript / script
        |
        v
Chapter-ordered injection loop
  for chapter_or_scene in discourse order:
    parse local text
    extract events and claims
    apply state transitions
    update entity memories
    update visibility and belief states
    update threads, secrets, reveals, and world rules
    write evidence and transition logs
        |
        v
Textual world model
  - event log
  - entity state store
  - character belief and knowledge states
  - object/location/group/world-rule states
  - reveal and discourse-status ledger
  - setup/payoff and open-thread ledger
  - evidence index
        |
        v
Memory-agent society
  - ingestion operator
  - state-transition operator
  - visibility resolver
  - thread/reveal operators
  - packet builder
  - branch simulator
  - consistency validator
        |
        v
Visibility gate
        |
        v
Perspective-constrained writing plan / simulation / auditing
```

The "agents" do not need to be fully autonomous chat agents at the beginning. A practical MVP can implement them as role-specific modules with:

- an operator-specific view of the textual world model;
- a read/write policy;
- a prompt or tool interface;
- structured output schemas;
- evidence requirements.

The important distinction is that DMS agents are not merely consumers of memory. They are **memory operators** that maintain the textual world model.

### 3.3 Implementation Architecture And Reused Ideas

The implementation should separate three layers:

```text
Layer 1: Narrative substrate
  - structured events, facts, entity states, constraints, threads
  - local graph / SQL / JSONL artifacts
  - token-level memory records with source-span provenance
  - chapter-ordered transition logs

Layer 2: Agent runtime
  - tool calling
  - coordinator / router
  - role-specific prompts
  - structured output validation

Layer 3: Writing product workflow
  - project/task lifecycle
  - prefix checkpoint selection
  - next-chapter planning
  - branch memory
  - logs and evaluation artifacts
```

#### Agent Memory Survey Takeaways

The agent-memory survey `Memory in the Age of AI Agents` is useful mainly as a design vocabulary. It organizes memory along three axes:

- `Forms`: what carries memory;
- `Functions`: why the agent needs memory;
- `Dynamics`: how memory is formed, updated, forgotten, and retrieved.

DMS should use this taxonomy, but not import the generic agent-memory objective directly. The project is not trying to build an all-purpose lifelong assistant. It is trying to build a controllable memory system for long-form narrative writing under prefix-only visibility.

Recommended positioning:

| Survey dimension | General agent-memory meaning | DMS interpretation |
| --- | --- | --- |
| Form | token-level, parametric, latent | token-level textual world model for MVP and paper experiments |
| Function | factual, experiential, working | narrative world-state memory, generation working memory, optional writing-strategy memory |
| Dynamics | formation, evolution, retrieval | chapter-ordered injection, state-transition update, visibility-aware packet construction |

Why token-level memory should be the primary form:

- it is externally visible and therefore debuggable;
- it can cite exact chapter/scene evidence;
- it can support story-time validity intervals;
- it can be edited by the author;
- it can be versioned into canonical, speculative, and evaluator-only memory;
- it can enforce explicit prefix and perspective access boundaries.

Why not parametric or latent memory for the main system:

- parametric memory is hard to update, inspect, roll back, or constrain to a single manuscript;
- latent memory is compact but difficult to audit and unsuitable for proving which prefix evidence influenced a continuation;
- both forms weaken the key research claim, because evaluation needs to know exactly which prefix evidence influenced a continuation.

Latent or parametric memory can be future work for efficiency or style adaptation, but the first DMS implementation should remain token-level and provenance-first.

#### Entity-Centric Diegetic Memory

The word `memory` should not mean only a character's subjective recollection. In fiction, many diegetic entities carry continuity constraints. DMS should therefore maintain memory for multiple entity types:

| Memory bearer | What it remembers | Why it matters for writing |
| --- | --- | --- |
| Character | knowledge, beliefs, goals, emotions, relationships, secrets | prevents omniscient behavior, preserves arc continuity |
| Location | past events, accessibility, atmosphere, ownership, hidden areas, damage/state changes | prevents spatial contradictions and supports scene continuity |
| Object / prop | ownership, location, condition, symbolic meaning, promises of later use | preserves Chekhov-style setup/payoff and object continuity |
| Organization / faction | membership, hierarchy, agenda, public knowledge, internal secrets | supports political/social plots and group agency |
| World rule | magic/technology/legal/social constraints, exceptions, costs | prevents violations of established story logic |
| Plot thread | setup, open question, pressure level, promised payoff, current status | keeps continuation from dropping or prematurely resolving arcs |
| Narrator / discourse | what the reader has been told, withheld, misdirected, or invited to infer | separates reader knowledge from character knowledge |

This is the main conceptual difference from generic agent memory: DMS memory is not organized around the assistant's life history. It is organized around the **story world's evolving state** and the **visibility relation** between entities, characters, narrator, and reader.

#### Token-Level Memory Structure

Following the survey's token-level memory taxonomy, DMS should combine three storage shapes:

| Token-level shape | DMS use | Example artifact |
| --- | --- | --- |
| Flat memory | atomic facts, quotes, evidence spans, extracted events | `facts.jsonl`, `events.jsonl`, `quotes.jsonl` |
| Planar memory | entity graph, timeline graph, knowledge-transfer graph, setup/payoff graph | `memory_graph.sqlite` or `graph_edges.jsonl` |
| Hierarchical memory | chapter summaries, episode summaries, storyline summaries, entity dossiers | `summaries/chapter/*.json`, `profiles/*.json` |

Flat memory is good for traceable evidence. Planar graph memory is good for relation and temporal reasoning. Hierarchical memory is good for long manuscripts where the continuation planner needs compressed context without losing links to lower-level evidence.

The key implementation rule is:

```text
summary memory must never replace evidence memory;
it can only point back to lower-level facts, events, and spans.
```

This prevents abstract summaries from becoming ungrounded narrative claims.

#### Qwen-Agent Assessment

`qwen_agent` from `NarrativeSkillAgent_V2/Qwen-Agent-main` is useful as a lightweight agent runtime, not as the DMS memory layer itself.

Useful pieces:

- `Assistant`: general tool-calling agent with optional RAG/file support.
- `FnCallAgent`: function-call loop that can call tools across multiple LLM turns.
- `GroupChat` and `GroupChatAutoRouter`: multi-agent conversation/router primitives.
- `VirtualMemoryAgent`: retrieval-first helper that can inject retrieved knowledge into a system message.
- Writing agents under `qwen_agent/agents/writing`: useful as reference prompts for continuation/outline workflows.

Limitations:

- Its memory is mostly document retrieval, not structured diegetic memory.
- `GroupChat` is closer to role-play discussion than a controlled narrative state machine.
- It does not natively model `character c at story time t can/cannot know fact f`.

Recommended use:

- Do not fork or directly modify Qwen-Agent source.
- Use Qwen-Agent as a dependency or wrapper for tool-calling agents.
- Implement DMS-specific memory, visibility gates, and branch logic in this repository.

#### NarrativeSkillAgent_V2 Assessment

`NarrativeSkillAgent_V2` is more directly relevant than raw Qwen-Agent because it already contains narrative KG and retrieval infrastructure.

Potentially reusable ideas or modules:

- event-first extraction;
- atomic fact index;
- narrative graph with Episode and Storyline aggregation;
- character status extraction;
- graph/vector retrieval tools;
- reading skills, especially character mindstate tracking.

Recommended use:

- Reuse the data-model ideas and prompts where compatible.
- For a clean DMS prototype, reimplement the minimum subset locally:
  - event extraction;
  - atomic facts;
  - character state snapshots;
  - timeline/order fields;
  - constraints and thread records.

#### MiroFish Assessment

MiroFish is not a good direct fork target for DMS. It is a full-stack social-simulation product built around Zep Cloud and OASIS. Its backend declares `AGPL-3.0`, so direct code reuse should be avoided unless license compatibility is intended.

However, it provides several strong implementation ideas:

- persistent project and task lifecycle;
- schema-first graph construction;
- entity-to-agent profile generation;
- stepwise configuration generation instead of one huge LLM prompt;
- dynamic memory update from agent actions;
- durable JSON/JSONL logs for agent actions, reports, and tool calls;
- interactive "interview an agent" workflow.

Recommended DMS adaptation:

```text
manuscript prefix
  -> narrative schema / fixed DMS schema
  -> prefix memory graph
  -> character memory profiles
  -> planning configuration
  -> next-chapter candidate plan
  -> constraint validation
  -> candidate branch memory
  -> optional author promotion to accepted memory
```

The most important MiroFish insight is:

> Do not generate directly from a graph. Convert extracted memory into role-specific, evidence-bound agent profiles and task-specific configuration first.

#### Branch Memory Policy

DMS should distinguish memory layers:

```text
canonical_prefix_memory
  immutable memory extracted from chapters 1..N

candidate_branch_memory
  speculative memory created by generated plans or continuations

accepted_draft_memory
  memory promoted by the author after accepting a branch or draft

reference_evaluation_memory
  evaluator-only records from the original next unit, used for scoring only
```

Generated content must never contaminate canonical prefix memory. This is critical for both writing assistance and prefix-only evaluation.

## 4. Formal Task Definition

Let a story be represented as:

```text
S = (T, C, E, F, K, R)
```

Where:

- `T`: story-world timeline;
- `C`: characters;
- `E`: events;
- `F`: atomic facts;
- `K`: character knowledge and belief states;
- `R`: reveal, secrecy, and permission constraints.

Given:

```text
q = (character c, story time t, task intent i)
```

The system must construct:

```text
VisibleMemory(c, t, i)
```

This memory may include:

- facts that happened before `t`;
- facts directly experienced by `c`;
- facts told to `c`;
- facts inferred by `c`;
- rumors or beliefs held by `c`;
- false beliefs explicitly supported by prior scenes.

It must exclude:

- future events after `t`;
- secrets not yet revealed to `c`;
- facts known only by other characters;
- narrator-only or author-only information;
- correct truths that contradict the character's current false belief, unless the task is auditing rather than simulation.

## 5. Data Sources

The research should separate **raw source texts** from **releasable benchmark artifacts**. Novels and scripts often have different copyright constraints.

### 5.1 Public-Domain Novels

Primary source:

- Project Gutenberg public-domain or no-US-copyright texts.

Use cases:

- full-book timeline construction;
- character knowledge tracking;
- reveal and secret evaluation;
- long-range foreshadowing tracking.

Important constraint:

- Project Gutenberg notes that copyright status and usage freedom can depend on jurisdiction, especially outside the United States. For a public benchmark, release text IDs, offsets, annotations, and preprocessing scripts rather than redistributing unnecessary raw text.

Good candidate genres:

- detective fiction: secrets, reveals, false hypotheses;
- gothic fiction: delayed explanations, unreliable perception;
- adventure fiction: object/location continuity;
- fairy tales: compact causal structure and explicit goals.

Possible examples:

- Sherlock Holmes stories;
- Agatha Christie works that are public domain in the relevant jurisdiction only if verified carefully;
- Dracula;
- Frankenstein;
- Alice's Adventures in Wonderland;
- public-domain fairy tales.

### 5.2 Existing Narrative QA Datasets

Useful as external evaluation or comparison tasks:

- NarrativeQA: long-form QA over books and movie scripts.
- FairytaleQA: explicit and implicit questions over children's stories with narrative-element categories.
- MovieQA: story comprehension over movies with scripts, plots, subtitles, and other modalities.
- DramaQA: character-centered story understanding, especially useful as inspiration for hierarchical QA and character-focused annotation.

These datasets do not directly solve diegetic visibility, but they provide:

- story comprehension questions;
- narrative-level answer targets;
- possible source texts;
- examples of question taxonomies.

### 5.3 Scripts And Screenplays

Scripts are attractive because they already include:

- scene boundaries;
- speakers;
- dialogue turns;
- location headings;
- sometimes explicit action descriptions.

Potential sources:

- public-domain scripts;
- licensed scripts;
- script-derived datasets such as Cornell Movie-Dialogs Corpus;
- MovieQA-like script resources;
- private/local scripts for non-redistributed experiments.

Important constraint:

- Many online screenplays are not safely redistributable. For publishable research, keep raw script text out of released artifacts unless license status is clear. Release derived annotations, source identifiers, and evaluation queries where allowed.

### 5.4 Literary NLP Annotation Resources

Useful for parser evaluation and bootstrapping:

- LitBank: literary entities, events, coreference, and quotation attribution annotations over fiction excerpts.
- BookNLP: practical pipeline for book-scale literary NLP, including character processing and quotation attribution.

These resources are especially useful for evaluating lower-level extraction quality before evaluating the full memory society.

### 5.5 Synthetic And Semi-Synthetic Stories

A strong benchmark should include synthetic or controlled stories because real novels rarely provide complete gold labels for every character's knowledge state.

Design:

- short stories of 1,000-5,000 words;
- 3-8 characters;
- explicit secrets, lies, overheard conversations, false beliefs, and reveals;
- gold timeline and knowledge states authored directly;
- automatic generation followed by human validation.

Why this matters:

- enables exact ground truth;
- supports counterfactual variants;
- supports controlled error injection;
- makes belief-state and constraint-satisfaction evaluation much cleaner.

## 6. Dataset Design: DMS-Bench

The project can define a benchmark called **DMS-Bench**.

### 6.1 Dataset Splits

Split by complete story, not by query, to avoid same-story leakage.

```text
DMS-Bench
  train/
    public_domain_novels/
    scripts_or_scriptlike_texts/
    synthetic_stories/
  dev/
  test/
```

Recommended scale for a first paper:

| Split | Stories | Prefix checkpoints | Future beats | Narrative constraints | Planning/continuation prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev | 5-8 | 30-60 | 300-600 | 300-600 | 50-100 |
| Test | 10-15 | 80-150 | 800-1,500 | 800-1,500 | 100-200 |

The first version should favor annotation quality over size.

### 6.2 Core Annotation Schema

#### Narrative Unit

```json
{
  "unit_id": "ch03_s12_u004",
  "source_id": "story_001",
  "discourse_position": 4312,
  "chapter_or_scene": "Chapter 3",
  "story_time": {
    "start": "T12",
    "end": "T12",
    "certainty": "certain"
  },
  "text_span": {
    "start_char": 12033,
    "end_char": 12301
  },
  "unit_type": "event | dialogue | exposition | flashback | narrator_comment"
}
```

#### Event

```json
{
  "event_id": "ev_00042",
  "unit_id": "ch03_s12_u004",
  "event_type": "discover | tell | observe | infer | deceive | travel | promise | reveal",
  "participants": ["char_alice", "char_basil"],
  "location": "loc_library",
  "summary": "Alice finds the sealed letter in the library.",
  "story_time": "T12",
  "causes": ["ev_00031"],
  "effects": ["fact_00101"],
  "evidence_span": {
    "start_char": 12033,
    "end_char": 12301
  }
}
```

#### Fact

```json
{
  "fact_id": "fact_00101",
  "proposition": "The sealed letter is hidden in the library.",
  "truth_status": "true | false | ambiguous",
  "first_true_time": "T08",
  "first_revealed_in_text": "ch03_s12",
  "fact_type": "object_location | identity | motive | relationship | world_rule | plan | secret",
  "supporting_events": ["ev_00042"]
}
```

#### Entity Memory Record

Entity memory generalizes character memory to all continuity-bearing story entities.

```json
{
  "entity_memory_id": "em_00042",
  "entity_id": "loc_library",
  "entity_type": "character | location | object | organization | faction | world_rule | plot_thread | narrator",
  "memory_function": "factual | experiential | working",
  "memory_shape": "flat | graph | hierarchical",
  "content": "The library has a locked east cabinet that Basil avoids opening.",
  "valid_from": "T08",
  "valid_until": null,
  "truth_status": "true | false | ambiguous | contested",
  "visibility_scope": {
    "known_by": ["char_basil"],
    "hidden_from": ["char_alice"],
    "reader_status": "unrevealed | hinted | revealed | misdirected"
  },
  "supporting_events": ["ev_00031"],
  "evidence_spans": [
    {
      "unit_id": "ch02_s05_u003",
      "start_char": 8831,
      "end_char": 8912
    }
  ],
  "confidence": 0.86,
  "version": "canonical_prefix_memory"
}
```

Required invariants:

- every entity memory record must cite source evidence unless it is explicitly marked as generated branch memory;
- `valid_from` and `valid_until` should be story-world times, not only discourse positions;
- character knowledge and reader/narrator disclosure must be represented separately;
- speculative continuations must use `version = candidate_branch_memory`, never `canonical_prefix_memory`.

#### Character Knowledge State

```json
{
  "knowledge_id": "ks_01021",
  "character_id": "char_alice",
  "fact_id": "fact_00101",
  "valid_from": "T12",
  "valid_until": null,
  "epistemic_status": "knows | believes | suspects | misbelieves | unaware",
  "acquisition_mode": "observed | heard | told_by_character | inferred | read | overheard",
  "source_event": "ev_00042",
  "confidence": 0.9,
  "evidence_span": {
    "start_char": 12033,
    "end_char": 12301
  }
}
```

#### Location State

```json
{
  "location_state_id": "ls_00017",
  "location_id": "loc_library",
  "story_time": "T12",
  "physical_state": "east cabinet locked; window broken; floor wet",
  "access_rules": ["Only Basil has the brass key."],
  "recent_events": ["ev_00031", "ev_00042"],
  "hidden_contents": ["obj_confession_letter"],
  "known_by": ["char_basil"],
  "evidence_spans": []
}
```

#### Object State

```json
{
  "object_state_id": "os_00029",
  "object_id": "obj_old_key",
  "story_time": "T12",
  "owner": "char_alice",
  "location": "loc_alice_room",
  "condition": "tarnished but usable",
  "symbolic_or_plot_role": "possible payoff for locked east cabinet",
  "visibility_scope": {
    "known_by": ["char_alice"],
    "hidden_from": ["char_basil"],
    "reader_status": "hinted"
  },
  "supporting_events": ["ev_00020"],
  "evidence_spans": []
}
```

#### Organization / Faction State

```json
{
  "group_state_id": "gs_00008",
  "group_id": "org_watch",
  "story_time": "T12",
  "members": ["char_mara", "char_basil"],
  "public_goal": "protect the town archive",
  "private_goal": "hide the archive's missing ledger",
  "internal_conflicts": [],
  "known_by": ["char_mara", "char_basil"],
  "hidden_from": ["char_alice"],
  "evidence_spans": []
}
```

#### Secret / Reveal

```json
{
  "secret_id": "sec_00007",
  "fact_id": "fact_00088",
  "hidden_from": ["char_alice", "char_basil"],
  "known_by": ["char_mara"],
  "active_from": "T03",
  "reveal_events": ["ev_00091"],
  "reveal_conditions": "Alice reads the confession letter."
}
```

#### Foreshadowing Thread

```json
{
  "thread_id": "foreshadow_00012",
  "setup_event": "ev_00020",
  "setup_summary": "The old key is mentioned twice but not explained.",
  "promised_payoff_type": "object_use",
  "payoff_event": "ev_00087",
  "status": "paid_off | unresolved | contradicted",
  "strength": "weak | medium | strong"
}
```

### 6.3 Query Types

#### Type A: Writing-Intent-Conditioned Continuation

Input:

```json
{
  "prefix": "chapters_01_10",
  "writing_intent": "Write the next scene where Alice confronts Basil in the library at midnight, using the sealed letter as the pressure point.",
  "output_format": "outline | prose_continuation | branch_plan"
}
```

Gold and references:

- the writing intent, either author-provided or extracted from the reference unit;
- prefix-derived narrative constraints;
- prefix-boundary character and entity states;
- the original next unit as a reference output for calibrated comparison, not as the only valid continuation.

Evaluation target:

- writing intent satisfaction;
- consistency with the prefix;
- character/entity state continuity;
- memory grounding;
- creative plausibility, not exact text reproduction.

#### Type B: Character-State Continuation

Input:

```json
{
  "character": "char_basil",
  "prefix": "chapters_01_10",
  "writing_intent": "Plan Basil's next scene using only what Basil knows and believes."
}
```

Gold:

- Basil's prefix-boundary knowledge, beliefs, goals, emotional state, and relationship state;
- the reference next unit as optional arc-direction calibration, not as a required exact action sequence.

#### Type C: Memory Faithfulness Check

Input:

```json
{
  "prefix": "chapters_01_10",
  "writing_intent": "Generate the next scene while respecting the active narrative constraints."
}
```

Gold:

- constraints extracted from the prefix:
  - character knowledge constraints;
  - relationship states;
  - unresolved conflicts;
  - object/location states;
  - world rules;
  - active secrets and foreshadowing threads.

#### Type D: Diagnostic Memory Retrieval

Input:

```json
{
  "prefix": "chapters_01_10",
  "writing_intent": "Alice confronts Basil in the library about the sealed letter.",
  "task": "Build the memory packet needed to write this scene."
}
```

Gold:

- prefix memories that support the writing intent;
- entity states, relationships, and unresolved threads required to write the requested unit.

### 6.4 Writing-Intent Evaluation

For a writing-assistant memory system, the most natural evaluation setting is:

```text
Full completed novel or script exists for evaluation.
System is only allowed to read a prefix: chapters 1..N.
The evaluator can use the reference next unit to build writing intents and score outputs.
The generator receives only the prefix memory, writing intent, and allowed style/length inputs.
```

This reframes the benchmark from "Can the model understand a finished story?" to:

> Given what an author has written so far and what they intend to write next,
> can the memory system help produce a continuation that satisfies the intent
> and remains consistent with the story state?

#### Prefix Unit

```json
{
  "prefix_id": "story_001_chapters_01_10",
  "source_id": "story_001",
  "visible_range": {
    "start_chapter": 1,
    "end_chapter": 10
  },
  "target_next_unit": "chapter_11",
  "allowed_text": ["chapter_01", "chapter_02", "chapter_03", "chapter_04", "chapter_05", "chapter_06", "chapter_07", "chapter_08", "chapter_09", "chapter_10"],
  "reference_text": ["chapter_11"]
}
```

Each full novel can produce many prefix checkpoints:

```text
chapters 1..3 -> write/support chapter 4 intent
chapters 1..5 -> write/support chapter 6 intent
chapters 1..10 -> write/support chapter 11 intent
chapters 1..15 -> write/support chapter 16 intent
```

This creates a longitudinal evaluation of how memory quality changes as the story gets longer.

#### Writing-Centric Tasks

**Task 1: Writing-Intent-Conditioned Next-Unit Generation**

The primary task should ask the system to produce either:

- a next-chapter outline;
- a next-scene outline;
- a short continuation;
- multiple possible branch plans.

The output should be evaluated as a creative continuation under explicit writing
intent and prefix constraints, not as retrieval.

```json
{
  "task": "intent_conditioned_generation",
  "prefix_id": "story_001_chapters_01_10",
  "target": "chapter_11",
  "writing_intent": "Alice confronts Basil in the library at midnight about the sealed letter.",
  "output_format": "outline | prose_continuation | branch_plan"
}
```

Evaluation:

- writing intent consistency;
- writing quality;
- memory faithfulness.

**Task 2: Reference Delta Calibration**

Run the same intent-based evaluator on both the generated output and the
original reference unit:

```text
score_generated = evaluator(intent, prefix_memory, generated_output)
score_reference = evaluator(intent, prefix_memory, reference_output)
delta = score_generated - score_reference
```

Metrics:

- writing intent consistency delta;
- writing quality delta;
- memory faithfulness delta.

This uses the original text as a calibration point without making exact plot or
wording reproduction the goal.

**Task 3: Memory Faithfulness Check**

Judge whether the generated output remains faithful to the prefix memory packet.
This includes character/entity states, durable relationships, important prior
facts, world constraints, and unresolved threads. The output should not invent
major background facts that should have come from memory, and it should not leak
memory indexes or reference labels into final prose.

Metric:

- memory faithfulness.

**Task 5: Diagnostic Memory Retrieval**

Retrieval is useful as a diagnostic, but should not be the primary claim.

Use the writing intent and reference unit to create gold labels:

1. Identify important intent requirements.
2. Annotate which earlier prefix facts, relationships, promises, or world rules they depend on.
3. Ask the memory system, given only chapters `1..N`, to surface useful prior context.

Metrics:

- antecedent recall@K;
- antecedent precision@K;
- evidence-span accuracy.

These diagnostics explain why a generation succeeds or fails, but they should be reported as secondary diagnostics rather than headline results.

#### Why This Evaluation Is Better For Writing Memory

This setup matches the author's real situation:

- the author has written a prefix;
- the author can provide a concrete writing intent;
- memory must summarize and retrieve prior context;
- the system should help fulfill the intent while preserving continuity;
- the completed reference text can still be used as an evaluation calibration point.

It also avoids a common mistake: treating the final novel as the only valid continuation. The benchmark should distinguish:

- **writing intent satisfaction**, which is the task definition;
- **consistency with prefix**, which is required;
- **reference trajectory alignment**, which is useful but not mandatory;
- **creative plausibility**, which requires human or model-assisted judgment.

## 7. Methods

### 7.1 Parser

The parser is used inside the chapter-ordered injection loop. It should extract local evidence from the current chapter or scene, then pass memory candidates to world-model operators that update the prior state.

For each injected unit, the parser extracts:

- chapter or scene boundaries;
- dialogue turns and speakers;
- character mentions and coreference chains;
- events and event arguments;
- temporal expressions and relative order;
- objects, locations, and relationships;
- knowledge-transfer events, such as tells, overhears, discovers, reads, sees, infers;
- secrets, reveals, promises, and foreshadowing candidates.

Recommended implementation:

- use BookNLP or similar tools for literary preprocessing where useful;
- use LLM extraction with strict JSON schemas for high-level event and belief extraction;
- require every structured claim to carry a source span;
- run incremental parsing by chapter or scene in discourse order;
- reconcile entity aliases and coreference against the current world model before applying updates;
- postpone global reconciliation until after prefix-level memory is frozen, and keep it separate from prefix-only evaluation artifacts.

Parallel per-chapter extraction is allowed only for cheap local preprocessing, such as sentence segmentation, quote detection, or mention candidates. Any operation that affects story truth, belief state, reveal timing, or entity state must be applied sequentially.

### 7.1.1 Chapter-Ordered Injection Protocol

DMS should treat manuscript ingestion as an online update process:

```text
M_0 = empty textual world model

for chapter k in discourse order:
    local_candidates = parse(chapter_k)
    grounded_candidates = align_to_existing_model(local_candidates, M_{k-1})
    transitions = infer_state_transitions(grounded_candidates, M_{k-1})
    visibility_updates = resolve_visibility(transitions, M_{k-1})
    thread_updates = update_threads_and_reveals(transitions, M_{k-1})
    M_k = commit(M_{k-1}, transitions, visibility_updates, thread_updates)
```

At prefix checkpoint `N`, freeze:

```text
canonical_prefix_memory = M_N
```

No operation during generation may read `M_{N+1}` or any later chapter-derived artifact.

Each chapter injection should write an append-only transition log:

```json
{
  "injection_id": "inj_story001_ch05",
  "input_unit": "chapter_05",
  "previous_model": "M_04",
  "new_model": "M_05",
  "events_added": [],
  "state_transitions": [],
  "visibility_updates": [],
  "thread_updates": [],
  "conflicts": [],
  "evidence_spans": []
}
```

This protocol is central to the research claim. It makes the system closer to a textual world model than a static document index.

### 7.2 Timeline Construction

The timeline should support partial ordering rather than force all events into a single total order.

Relations:

- before;
- after;
- overlaps;
- during;
- same_time;
- flashback_to;
- uncertain.

Evaluation can use pairwise temporal relation accuracy and contradiction detection instead of relying only on exact timestamps.

### 7.3 Knowledge Propagation

A character can acquire a fact through:

- direct observation;
- direct participation;
- being told;
- overhearing;
- reading a document;
- inference;
- rumor;
- deception;
- public announcement.

Propagation rules should be conservative:

- if a character is present in a scene, they may observe visible events;
- if a character hears dialogue, they may know the stated proposition;
- if a fact is only implied, mark it as inferred or suspected rather than known;
- if a later reveal corrects a false belief, preserve the old belief state for earlier times.

### 7.4 Visibility Gate

The visibility gate is the core mechanism.

Input:

```text
(character c, story time t, task mode m)
```

Output:

```text
VisibleMemoryPacket(c, t, m)
```

The packet contains:

- visible facts;
- current beliefs;
- current goals and motivations;
- emotional state;
- relevant past events;
- relationship states;
- unresolved local threads;
- evidence spans.

It excludes:

- hidden facts;
- other-character private knowledge;
- author-only annotations.

Task modes:

- `simulate`: strictest, no hidden truths or other-character private knowledge;
- `audit`: may inspect hidden prefix facts but must report them as unavailable for the selected character when appropriate;
- `explain`: can compare character belief with story truth if the user asks for analysis.

### 7.5 Agent Society Coordination

Use a blackboard-style coordinator for memory operators:

```text
Chapter injection or writing query
  -> Coordinator
  -> Ingestion operator: local candidates from current chapter/scene
  -> State-transition operator: entity state changes
  -> Timeline operator: event order and temporal constraints
  -> Visibility resolver: character knowledge and belief updates
  -> Secret/reveal operator: hidden/revealed/discourse status
  -> Thread operator: open setups, promises, payoffs
  -> World-rule operator: stable constraints and exceptions
  -> Visibility gate
  -> Memory packet / branch simulation / audit answer with evidence
```

Each operator should produce structured outputs, not just prose.

Example:

```json
{
  "operator": "secret_reveal_operator",
  "blocked_facts": [
    {
      "fact_id": "fact_00088",
      "reason": "Reveal occurs at T21, query time is T12.",
      "hidden_from": ["char_alice"],
      "evidence": ["ev_00091"]
    }
  ]
}
```

### 7.6 Project, Task, And Artifact Lifecycle

Inspired by MiroFish, DMS should treat every run as a durable project with inspectable artifacts rather than an ephemeral chat.

Recommended project state:

```json
{
  "project_id": "dms_proj_xxx",
  "name": "pilot_story_001",
  "status": "created | memory_built | ready | generating | completed | failed",
  "source_files": [],
  "prefix_checkpoints": [],
  "canonical_memory_path": "data/projects/.../canonical_memory",
  "branch_memory_paths": [],
  "created_at": "...",
  "updated_at": "..."
}
```

Recommended task types:

- `ingest_manuscript`;
- `build_prefix_memory`;
- `generate_character_profiles`;
- `generate_planning_config`;
- `generate_next_chapter_plan`;
- `validate_constraints`;
- `evaluate_reference_alignment`.

Each task should record:

- task id;
- task type;
- status;
- progress;
- message;
- inputs;
- outputs;
- error;
- exact artifact paths.

This makes long-running extraction and generation reproducible and debuggable.

### 7.7 Character Memory Profile Generation

Before generating a continuation, DMS should convert raw memory into character-specific profiles.

This mirrors MiroFish's entity-to-agent profile generation, but the profile is narrative-facing rather than social-media-facing.

Example:

```json
{
  "character_id": "char_alice",
  "prefix_checkpoint": "story_001_chapters_01_10",
  "role_in_story": "viewpoint character",
  "known_facts": [],
  "beliefs": [],
  "misbeliefs": [],
  "goals": [],
  "fears": [],
  "emotional_state": "guarded and suspicious",
  "relationship_states": [],
  "secrets_known": [],
  "secrets_hidden_from": [],
  "active_constraints": [],
  "evidence_spans": []
}
```

Profile generation should be evidence-bound:

- every state item cites prefix evidence;
- only prefix facts are allowed;
- narrator truth and character belief are kept separate;
- uncertain states are marked as `suspects` or `inferred`, not `knows`.

### 7.8 Stepwise Planning Configuration

Following MiroFish's stepwise config generation, DMS should avoid one giant prompt that asks the model to remember everything and write the next chapter.

Instead, build a compact planning configuration:

```json
{
  "target_unit": "chapter_11",
  "prefix_checkpoint": "chapters_01_10",
  "output_mode": "outline | prose | branch_plan",
  "viewpoint_character": "char_alice",
  "active_characters": [],
  "active_conflicts": [],
  "open_threads": [],
  "allowed_reveal_policy": [],
  "must_satisfy_constraints": [],
  "branch_count": 5
}
```

Then pass only this configuration plus evidence packets to the continuation planner.

### 7.9 Candidate Branch Memory

Generated plans or continuations should be stored as branch memory, not written into canonical prefix memory.

```text
canonical_prefix_memory
  -> source of truth from chapters 1..N

candidate_branch_memory
  -> generated beats/scenes for one possible continuation

accepted_draft_memory
  -> author-approved branch promoted into working story memory
```

This mirrors MiroFish's dynamic graph-memory update from actions, but adds a strict separation between canonical and speculative memory.

### 7.10 Generation And Visibility Logs

Every generation should produce a JSONL trace:

```json
{
  "timestamp": "...",
  "run_id": "run_001",
  "action": "visibility_gate",
  "details": {
    "allowed_facts": [],
    "blocked_facts": [],
    "reason": "not visible to the selected character at this story point"
  }
}
```

Recommended logged actions:

- selected prefix checkpoint;
- constructed planning config;
- character profile generated;
- memory packet generated;
- visibility gate decision;
- blocked private or out-of-scope facts;
- generated beat plan;
- constraint validation result;
- final selected output.

This follows MiroFish's report/action logging pattern and is necessary for failure analysis and benchmark reproducibility.

### 7.11 Textual World-Model Formation, Evolution, And Retrieval

The survey's `formation -> evolution -> retrieval` lifecycle should become an explicit DMS engineering contract, but DMS interprets it as textual world-model maintenance.

#### Formation

Formation converts the current chapter or scene into memory candidates conditioned on the prior world model.

Recommended pipeline:

```text
prior textual world model M_{k-1}
  + chapter/scene text
  -> narrative units
  -> events
  -> atomic facts
  -> entity states
  -> state-transition candidates
  -> visibility records
  -> thread/setup records
  -> evidence-bound memory records
  -> updated textual world model M_k
```

Formation should produce both raw trace records and distilled memory records:

- raw trace: event, dialogue, quote, action, description;
- distilled fact: proposition with truth status and valid time;
- entity state: character/location/object/group/world-rule state at a time;
- state transition: `entity_state_before -> event -> entity_state_after`;
- thread state: setup, unresolved pressure, payoff expectation;
- visibility state: who knows, believes, suspects, misbelieves, or is unaware.

This is token-level knowledge distillation, not parametric internalization. The system extracts reusable narrative memory from text while preserving links to the original evidence.

#### Evolution

Evolution controls how the textual world model changes as chapters are injected or as generated branches are explored.

Required operations:

| Operation | DMS behavior |
| --- | --- |
| Sequential commit | apply chapter `k` updates to `M_{k-1}` to produce `M_k` |
| Consolidation | merge local facts into chapter, episode, entity, and thread summaries while preserving evidence links |
| Updating | create time-versioned records when an entity changes state, rather than overwriting the past |
| Conflict handling | mark contradictions as `contested`, `retconned`, `unreliable_narration`, or `extraction_conflict` |
| Forgetting | do not hard-delete canonical evidence; instead archive, down-rank, or compress low-value items |
| Branch isolation | keep generated branch memory separate from canonical prefix memory |

For narrative writing, "forgetting" should mostly mean retrieval down-ranking or summary compression. Rare details can become future payoffs, so deleting old evidence is dangerous.

The system should explicitly avoid this pattern as the main construction route:

```text
extract all chapters independently
  -> merge facts globally
  -> infer prefix state afterward
```

That pattern makes it too easy for future chapters to influence earlier state labels. It can remain as a baseline or ablation, but not as the proposed method.

#### Retrieval

Retrieval should be role-, entity-, time-, and task-aware. It should not be generic top-k vector search.

Recommended retrieval stages:

```text
task intent
  -> query planner
  -> select memory stores
  -> hybrid retrieval
  -> visibility and time filter
  -> contradiction and status check
  -> compression into memory packet
  -> evidence citation
```

Retrieval signals should include:

- lexical match for names, objects, places, promises, and quoted phrases;
- semantic match for related conflicts and motivations;
- graph traversal over entity, timeline, knowledge-transfer, and setup/payoff edges;
- story-time filtering;
- discourse-position filtering for prefix-only evaluation;
- importance and narrative salience;
- role relevance for the target agent or viewpoint character.

Post-retrieval processing should produce **memory packets**, not raw search results. A memory packet is a compact, evidence-cited context object constructed for a specific role and task.

Example:

```json
{
  "packet_id": "packet_ch11_alice_plan",
  "target": {
    "task": "next_chapter_plan",
    "viewpoint_character": "char_alice",
    "prefix_checkpoint": "chapters_01_10"
  },
  "allowed_memory": {
    "character_state": [],
    "location_state": [],
    "object_state": [],
    "open_threads": [],
    "world_rules": []
  },
  "blocked_memory": [
    {
      "memory_id": "em_00931",
      "reason": "not visible to the selected character"
    }
  ],
  "evidence": []
}
```

### 7.12 Working Memory For Generation

DMS also needs a short-lived working memory for the current generation run. This is not a persistent story fact store.

Working memory should contain:

- the current planning objective;
- selected prefix checkpoint;
- active constraints for this generation;
- current candidate outline;
- unresolved validator warnings;
- branch-local decisions made during the current run;
- user edits during interactive writing.

Working memory should be cleared or serialized at the end of a run. If the author accepts a draft, only selected branch-local records are promoted into `accepted_draft_memory`.

### 7.13 Experiential Memory For Writing Strategy

The main DMS paper should focus on narrative factual memory, but an implementation can keep a small experiential memory for system improvement.

Examples:

- which constraint validators frequently catch errors;
- which prompt templates produce fewer visibility errors;
- which retrieval packet sizes work best for next-chapter planning;
- which entity types are often missing from generated continuations;
- common failure modes for a given genre.

This memory is about **how the assistant writes**, not about **what is true in the story world**. It should be stored separately from story memory so it cannot alter canonical narrative facts.

```text
story_world_memory != writing_strategy_memory
```

This separation matters because the evaluation should test whether narrative memory helps writing, not whether the model silently learned benchmark-specific heuristics.

## 8. Baselines

For writing-assistant evaluation, each baseline must be run under a clear access condition:

- **prefix-only**: system can access only chapters `1..N`; this is the fair writing-assistant condition.
- **reference oracle**: system can access the reference next unit, used only as an upper calibration condition, not as a fair writing assistant baseline.

### Baseline 1: Reference-Oracle LLM

Give the model the prefix plus the reference next unit or a reference-unit summary.

Expected weakness:

- not a fair writing assistant baseline because it sees the answer-like reference.

Use:

- upper calibration for style, beat, and intent-satisfaction scores;
- sanity check for evaluator behavior.

### Baseline 2: Chronological Summary Memory

Maintain chapter-by-chapter summaries over chapters `1..N` without character-specific knowledge.

Expected weakness:

- better temporal filtering but weak private knowledge modeling.

### Baseline 3: Vector RAG

Retrieve semantically relevant chunks from chapters `1..N`.

Expected weakness:

- retrieves relevant prefix chunks but may miss causal, character-specific, or unresolved-thread structure.

### Baseline 4: Time-Filtered RAG

Retrieve only chunks before story time `t`.

Expected weakness:

- can still include facts known only by other characters.

### Baseline 5: Character-Filtered RAG

Retrieve chunks mentioning the target character.

Expected weakness:

- misses facts told indirectly to the character;
- may miss relevant non-character entities, locations, objects, or unresolved threads.

### Baseline 6: Single Structured Memory Agent

Use one global structured memory store built from chapters `1..N`, with prompts to respect perspective.

Expected weakness:

- weaker modular control and weaker attribution of why facts are allowed or blocked.

### Baseline 7: Rolling Long-Context Prefix

Give the model as much of the prefix as fits in context, usually the most recent chapters plus a global summary.

Expected weakness:

- good local continuity;
- weak long-range setup retrieval;
- weak character-specific knowledge filtering;
- may forget early promises, object states, or relationship turns.

### Baseline 8: Hierarchical Chapter Summaries

Build chapter, arc, and book-level summaries from the prefix.

Expected weakness:

- strong compression baseline;
- may collapse truth, belief, rumor, and secret into one omniscient summary;
- often loses evidence spans needed for auditing.

### DMS Ablations

Compare full DMS against:

- no timeline operator;
- no secret/reveal operator;
- no belief-state tracking;
- no thread/foreshadowing operator;
- no evidence-span requirement;
- no operator decomposition, only one global memory module;
- no visibility gate, only relevance retrieval.

## 9. Evaluation Plan

The main evaluation should measure whether DMS helps write from a concrete
writing intent while staying faithful to prefix memory. Keep the headline
rubric small:

```text
writing intent consistency
writing quality
memory faithfulness
```

Detailed extraction and retrieval metrics are diagnostics only.

The main writing-assistant setting should be **prefix-only**:

```text
Input to system: chapters/scenes 1..N
Input to system: writing intent for N+1
Input to system: allowed style and length constraints
Input to evaluator: full completed story or script, including the reference N+1 unit
```

The evaluator can use the reference next unit to extract writing intents and to
score the reference output with the same rubric. The generator should receive
only the prefix memory, writing intent, and allowed style or length inputs.

### 9.1 Primary Output Types

Evaluate at least two output granularities:

- **Plan-level output**: next-chapter outline, beat sequence, or branch plan.
- **Prose-level output**: short next-scene or next-chapter continuation.

Plan-level evaluation is cleaner for a first paper because it reduces prose-style noise and focuses on narrative structure. Prose-level evaluation is closer to the final writing-assistant product and should be added once plan-level metrics are stable.

### 9.2 Writing Intent Consistency

Evaluate whether the output does what the writing intent asks. This includes key
entities, requested situation, scene function, narrative turn, and ending
handoff. The evaluator should penalize major unsupported drift, but not require
the wording or exact route of the original text.

### 9.3 Writing Quality

Evaluate whether the output is usable writing in the requested form. This
includes pacing, action/dialogue balance, sentence rhythm, concreteness,
formatting, length compliance, and whether the draft would be useful to revise.

### 9.4 Memory Faithfulness

Evaluate whether the output is faithful to the supplied memory packet. This
includes character/entity states, durable relationships, important prior facts,
world constraints, and whether the output invents major unsupported background.
Memory indexes and reference labels must not leak into the final prose.

### 9.5 Reference Calibration

For benchmark samples, run the same three-dimension evaluator on:

```text
generated_output
reference_output
```

Then report:

```text
delta = generated_score - reference_score
```

This calibrates scores against the original unit without treating the original
as the only valid continuation.

### 9.6 Human Pairwise Evaluation

Use blind pairwise comparisons between DMS and baselines. Evaluators see:

```text
writing intent
prefix summary or relevant prefix excerpts
system A next-chapter plan/continuation
system B next-chapter plan/continuation
optional reference next-unit summary for reference-aware judgments
```

Ask focused questions:

- Which output better satisfies the writing intent?
- Which output is better writing?
- Which output is more faithful to the supplied memory?
- Which output would be more useful to a writer?

Metrics:

- **Writing Intent Consistency Preference**;
- **Writing Quality Preference**;
- **Memory Faithfulness Preference**;
- **Writer Utility Preference**;
- inter-annotator agreement.

### 9.7 Diagnostic Retrieval Evaluation

Retrieval metrics should be secondary diagnostics, not the main proof.

Use the writing intent and reference unit to identify which prefix memories are
useful for writing the target unit, then test whether a memory system surfaces
those memories.

Metrics:

- antecedent recall@K;
- antecedent precision@K;
- evidence-span accuracy.

These metrics are useful for explaining failures. For example, a model may produce an inconsistent continuation because the relevant prefix constraint was never retrieved.

### 9.8 Extraction And Visibility Diagnostics

Low-level extraction remains useful, especially during development.

Tasks:

- character/entity extraction;
- coreference resolution;
- quotation attribution;
- event extraction;
- temporal relation extraction;
- fact extraction;
- location-state extraction;
- object-state extraction;
- organization/faction-state extraction;
- knowledge-transfer event extraction;
- character visibility classification.

Metrics:

- precision, recall, F1;
- evidence-span overlap F1;
- temporal relation accuracy;
- pairwise before/after F1;
- quotation speaker accuracy;
- belief-state classification accuracy;
- visible-fact precision/recall;
- entity-state validity F1;
- object continuity accuracy;
- location continuity accuracy;
- group/faction agenda consistency.

### 9.9 Memory Lifecycle Diagnostics

These diagnostics are secondary to writing-quality metrics, but they help show that DMS is a real agent memory system rather than static RAG.

Formation metrics:

- memory candidate precision/recall against annotated facts, events, entity states, and thread states;
- evidence attachment accuracy;
- entity-type assignment accuracy;
- truth-status and epistemic-status accuracy.

Evolution metrics:

- time-version accuracy for changed entity states;
- conflict detection precision/recall;
- summary faithfulness to lower-level evidence;
- branch contamination rate, i.e. how often generated branch records enter canonical prefix memory;
- compression retention rate for rare but narratively important details.

Retrieval and packet metrics:

- allowed-memory precision under prefix and character visibility filters;
- packet sufficiency judged by whether a continuation can satisfy target constraints;
- packet compactness measured by tokens per satisfied constraint;
- evidence coverage for generated beats and validator decisions.

### 9.10 Future Work: Draft Audit

Draft audit is important for the product direction, but it is not the first proof target in this plan.

Draft audit tasks can use:

```text
chapters/scenes 1..N + candidate chapter/scene N+1 draft
```

and evaluate:

- injected issue detection;
- false alarms per 10,000 words;
- severity-weighted F1;
- evidence quality;
- author-rated usefulness.

For the current research plan, audit should be treated as a later extension after the continuation-consistency metrics are established.

## 10. Experimental Matrix

### Experiment 1: Writing-Intent-Conditioned Planning

Goal:

- test whether DMS produces next-unit plans that satisfy writing intent while remaining consistent with the prefix.

Systems:

- rolling long-context prefix;
- hierarchical summaries;
- prefix-only vector RAG;
- time-filtered RAG;
- single global memory module;
- parallel extraction plus post-hoc merge;
- full DMS.

Metrics:

- writing intent consistency;
- writing quality;
- memory faithfulness;
- three-score delta vs reference;
- human preference on the same three dimensions.

### Experiment 2: Writing-Intent-Conditioned Prose Continuation

Goal:

- test whether DMS improves actual short-form continuation, not only outline planning.

Systems:

- same as Experiment 1.

Metrics:

- writing intent consistency;
- writing quality;
- memory faithfulness;
- three-score delta vs reference;
- human writer utility preference.

### Experiment 3: Reference Delta Calibration

Goal:

- test how close generated outputs are to the reference output under the same writing-intent rubric.

Systems:

- systems from Experiment 1 and Experiment 2;
- reference-oracle output as an upper calibration condition.

Metrics:

- writing intent consistency delta;
- writing quality delta;
- memory faithfulness delta.

### Experiment 4: Ablation Study

Goal:

- show which memory operators are responsible for writing-consistency capabilities.

Conditions:

- full DMS;
- no chapter-ordered injection, replaced by parallel extraction plus post-hoc merge;
- no state-transition operator;
- no timeline operator;
- no visibility resolver;
- no secret/reveal operator;
- no belief-state memory;
- no thread/foreshadowing operator;
- no evidence requirement.

Expected pattern:

- replacing chapter-ordered injection with parallel extraction harms reveal timing, state-transition accuracy, and prefix-boundary constraints;
- removing state-transition tracking increases object/location/relationship continuity errors;
- removing timeline increases chronology and state-continuity errors;
- removing visibility resolution reduces character plausibility and increases private-knowledge errors;
- removing secret/reveal handling increases premature reveal errors;
- removing belief memory reduces belief-state consistency;
- removing thread tracking reduces setup utilization and unresolved-thread preservation.

### Experiment 5: Prefix Length Scaling

Goal:

- test whether DMS degrades more gracefully than long-context and summary baselines as the story prefix grows.

Settings:

```text
chapters 1..3
chapters 1..5
chapters 1..10
chapters 1..20
chapters 1..40
```

Metrics:

- writing intent consistency;
- writing quality;
- memory faithfulness;
- generation latency;
- memory size.

### Experiment 6: Diagnostic Retrieval Analysis

Goal:

- explain whether generation failures come from missing memory, wrong visibility filtering, or weak generation.

Metrics:

- antecedent recall@K;
- antecedent precision@K;
- evidence-span accuracy.

This experiment should be reported after the main continuation-consistency results, not as the primary contribution.

### Experiment 7: Sequential Injection Versus Parallel Extraction

Goal:

- directly test the design claim that a textual world model should be built in chapter order rather than by independent multi-chapter extraction followed by merging.

Systems:

- chapter-ordered DMS world-model construction;
- parallel per-chapter extraction with post-hoc merge;
- chapter summaries accumulated in order;
- vector index over prefix chapters only.

Tasks:

- recover prefix-boundary entity states;
- recover character belief states at chapter `N`;
- identify whether a reveal is hidden, hinted, or revealed at chapter `N`;
- generate next-chapter plan from the constructed memory.

Metrics:

- state-transition accuracy;
- prefix-boundary state accuracy;
- reveal-status accuracy;
- belief-evolution accuracy;
- downstream writing intent consistency;
- downstream writing quality;
- downstream memory faithfulness.

Expected pattern:

- parallel extraction plus merge should miss or blur transitions that depend on prior model state;
- chapter-ordered DMS should better preserve `M_N` as the state available to a writer at chapter `N`.

## 11. Human Annotation Protocol

### 11.1 Annotator Tasks

Annotators should mark:

- events;
- story-world order;
- characters present in scene;
- who observes what;
- who says what to whom;
- facts introduced;
- facts believed, suspected, or misunderstood;
- secrets and reveal points;
- foreshadowing setup and payoff;
- prefix-boundary character states;
- active narrative constraints at each prefix checkpoint;
- writing intents for target unit `N+1`;
- concise reference summaries for target unit `N+1`;
- arc directions between prefix and reference unit;
- setup/payoff links from prefix to reference unit;
- possible consistency violations in generated continuations.

### 11.2 Quality Control

Use:

- double annotation on at least 20-30% of the benchmark;
- adjudication by a senior annotator;
- inter-annotator agreement for:
  - event boundaries;
  - temporal relation labels;
  - knowledge-state labels;
  - visibility labels;
  - reference summary labels;
  - narrative constraint labels;
  - arc direction labels;
  - consistency violation labels.

For subjective labels like foreshadowing strength or character plausibility, report agreement and keep evaluation as human-preference rather than pretending there is one objective answer.

## 12. Reproducibility And Artifact Plan

Recommended project artifacts:

```text
DiegeticMemorySociety/
  README.md
  RESEARCH_PLAN.md
  data/
    manifests/
      sources.jsonl
      splits.json
    annotations/
      events.jsonl
      facts.jsonl
      knowledge_states.jsonl
      secrets.jsonl
      foreshadowing.jsonl
      prefix_checkpoints.jsonl
      reference_beats.jsonl
      narrative_constraints.jsonl
      generated_outputs.jsonl
      queries.jsonl
    derived/
      parser_outputs/
      memory_graphs/
  configs/
    parser.yaml
    retriever.yaml
    eval.yaml
  scripts/
    build_corpus.py
    run_parser.py
    build_memory.py
    run_eval.py
  eval/
    metrics.py
    tasks/
    reports/
  docs/
    annotation_guidelines.md
    experiment_log_YYYYMMDD.md
```

Raw copyrighted text should not be committed or redistributed unless license allows it. Store raw corpora separately and track them through manifests:

```json
{
  "source_id": "pg_00001",
  "title": "Example Novel",
  "author": "Author Name",
  "source_url": "https://www.gutenberg.org/...",
  "license_or_terms": "Project Gutenberg terms checked on YYYY-MM-DD",
  "raw_local_path": "/absolute/local/path/not_committed/example.txt",
  "checksum": "sha256:...",
  "included_in_release": false
}
```

## 13. Implementation Roadmap

### Phase 0: Scope Lock, 1-2 Weeks

Deliverables:

- finalize task definitions;
- choose 3-5 pilot stories;
- write annotation guidelines;
- define JSON schemas for events, facts, entity memories, entity states, visibility, and thread records;
- implement source manifest format;
- define the chapter-ordered injection protocol and transition-log schema;
- decide runtime stack:
  - Qwen-Agent wrapper for tool-calling agents;
  - local DMS memory implementation for narrative state;
  - no direct MiroFish code copy unless license compatibility is explicitly accepted.
- lock memory-form scope:
  - token-level memory only for MVP and paper experiments;
  - no parametric or latent memory as core evidence source;
  - explicit support for character, location, object, organization/faction, world-rule, plot-thread, and narrator/discourse memory.

Success criterion:

- one story can be represented as prefix checkpoints with writing intents, reference outputs, character states, and narrative constraints.

### Phase 1: Parser MVP, 4-6 Weeks

Deliverables:

- chapter/scene segmentation;
- character and speaker extraction;
- event extraction;
- simple fact extraction;
- location, object, and organization/faction state extraction;
- evidence spans;
- chapter-by-chapter ingestion loop;
- append-only transition logs from `M_{k-1}` to `M_k`;
- first flat/graph/hierarchical token-level memory artifacts;
- durable project/task lifecycle;
- canonical prefix memory artifacts in JSONL/SQLite.

Success criterion:

- system can ingest one novel or script in chapter order and produce inspectable world-model states for each prefix checkpoint.

### Phase 2: Visibility Gate, 4-6 Weeks

Deliverables:

- character-time query interface;
- entity-time query interface for locations, objects, groups, world rules, and plot threads;
- prefix-boundary memory packet;
- known/believed/suspected/unknown labels;
- strict prefix-only generation mode;
- evidence-bound character memory profiles;
- evidence-bound location/object/group memory profiles;
- stepwise planning configuration;
- generation and visibility JSONL logs.

Success criterion:

- on pilot stories, generation uses prefix memory, writing intent, and character-private visibility boundaries correctly.

### Phase 3: DMS-Bench v0.1, 6-8 Weeks

Deliverables:

- 5-8 public-domain or synthetic stories;
- prefix checkpoints across chapters;
- 300-600 writing intent and reference-output labels;
- 300-600 narrative constraints;
- 50-100 next-chapter planning prompts;
- 50-100 short continuation prompts;
- baseline scripts.
- candidate branch memory for generated plans;
- constraint validator for branch outputs;
- sequential-vs-parallel extraction comparison script;
- reproducible artifact directories per run.

Success criterion:

- reproducible evaluation table comparing DMS against at least three baselines on intent satisfaction, reference-delta calibration, beat compatibility, constraint satisfaction, and human preference.

### Phase 4: Full Evaluation, 6-10 Weeks

Deliverables:

- DMS-Bench test split;
- next-chapter planning evaluation;
- prose continuation evaluation;
- reference-delta calibration;
- ablation study;
- human pairwise evaluation.

Success criterion:

- clear evidence that DMS improves writing-intent satisfaction and prefix-conditioned narrative consistency.

### Phase 5: Paper / Demo, 4-6 Weeks

Deliverables:

- research paper draft;
- demo interface;
- public benchmark subset;
- artifact documentation;
- examples and failure analysis.

Success criterion:

- a complete submission-ready package with method, benchmark, experiments, and limitations.

## 14. Expected Results And Claims

Strong claims the project can reasonably aim for:

- DMS-generated next-chapter plans satisfy more prefix-derived narrative constraints than long-context, RAG, and summary-memory baselines.
- Chapter-ordered textual world-model construction preserves prefix-boundary entity states, reveal timing, and belief evolution better than parallel extraction with post-hoc merging.
- DMS improves writing-intent satisfaction and reference-beat compatibility without treating the original next unit as the only valid continuation.
- DMS-generated continuations are preferred by human judges for narrative consistency, character plausibility, and setup utilization.
- Operator ablations show that state-transition tracking, visibility resolution, secret/reveal handling, and thread tracking contribute to different consistency dimensions.

Claims to avoid unless strongly proven:

- "Agents are always better than non-agent memory."
- "The system understands literature like humans."
- "The system can automatically fix story problems."
- "The system replaces authorial judgment."

More defensible claim:

> Modeling narrative memory as a chapter-ordered textual world model maintained by specialized memory operators provides a controllable mechanism for writing-intent-conditioned story continuation, especially when combined with explicit visibility gates and reference-delta evaluation.

## 15. Risks And Mitigations

### Risk 1: Copyright Limits Data Release

Mitigation:

- prioritize public-domain texts;
- release annotations, IDs, offsets, and scripts rather than raw copyrighted text;
- keep private or copyrighted scripts as non-released internal evaluation.

### Risk 2: Timeline And Knowledge Labels Are Ambiguous

Mitigation:

- allow uncertain and partial-order labels;
- distinguish story truth from character belief;
- report agreement;
- avoid forcing all scenes into exact timestamps.

### Risk 3: LLM Extraction Hallucinates Structure

Mitigation:

- require evidence spans;
- use schema validation;
- use parser confidence;
- audit a sample manually;
- evaluate extraction separately from reasoning.

### Risk 4: Multi-Agent Architecture Adds Complexity Without Gains

Mitigation:

- include single-memory and ablation baselines;
- show which agent improves which metric;
- keep agents as modular structured memory views, not unnecessary autonomous loops.

### Risk 5: Simulation Evaluation Is Subjective

Mitigation:

- prioritize plan-level beat and constraint metrics before prose-level literary quality;
- separate objective intent/constraint checks from human preference;
- use blind pairwise comparisons;
- ask authors to rate usefulness, not only literary quality.

### Risk 6: Reference Text Is Mistaken As The Only Correct Continuation

Mitigation:

- treat the original next unit as one reference trajectory;
- evaluate writing intent satisfaction, prefix consistency, and constraint satisfaction separately from reference alignment;
- include human judgments for alternative but valid branches;
- report reference calibration as calibration, not exact correctness.

## 16. Minimum Viable Paper Shape

If compressed into one publishable study, the cleanest paper is:

1. Introduce prefix-conditioned narrative consistency as the key writing-memory problem.
2. Define textual world model memory and explain why DMS is memory-as-agent-society rather than memory-for-agent.
3. Define chapter-ordered injection, state-transition logging, visibility gate, and prefix-only generation protocol.
4. Build DMS-Bench with public-domain/synthetic stories, prefix checkpoints, writing intents, reference outputs, and narrative constraints.
5. Compare DMS against rolling long-context prefix, hierarchical summaries, prefix-only RAG, time-filtered RAG, single structured memory, and parallel extraction plus post-hoc merge.
6. Evaluate on:
   - writing intent consistency;
   - writing quality;
   - memory faithfulness;
   - reference-delta calibration against the original unit;
   - human pairwise preference.
7. Show sequential-vs-parallel ingestion ablations and failure cases.

This version is coherent and testable without needing a large authoring product first.

## 17. Reference Starting Points

Useful external resources checked while drafting this plan:

- Project Gutenberg Terms of Use: https://www.gutenberg.org/policy/terms_of_use.html
- Project Gutenberg License: https://www.gutenberg.org/policy/license.html
- NarrativeQA paper: https://arxiv.org/abs/1712.07040
- FairytaleQA paper: https://arxiv.org/abs/2203.13947
- MovieQA paper: https://arxiv.org/abs/1512.02902
- DramaQA dataset page: https://dramaqa.snu.ac.kr/Dataset
- Cornell Movie-Dialogs Corpus via ConvoKit: https://convokit.cornell.edu/documentation/movie.html
- Cornell NLP data page: https://nlp.cornell.edu/data/
- LitBank repository: https://github.com/dbamman/litbank
- BookNLP repository: https://github.com/booknlp/booknlp

Local implementation references inspected:

- Agent-memory survey note: local reading note, not included in this repository.
- Qwen-Agent local source: local NarrativeKnowledgeWeaver archive, not included in this repository.
- Qwen-Agent `Assistant`, `FnCallAgent`, `GroupChat`, and `Memory`: local Qwen-Agent source files inspected as implementation references.
- NarrativeSkillAgent_V2 project: local archive, not included in this repository.
- MiroFish zip: local `MiroFish-main.zip` archive, not included in this repository.
- MiroFish reuse notes: `docs/mirofish_reuse_notes_20260529.md`.
