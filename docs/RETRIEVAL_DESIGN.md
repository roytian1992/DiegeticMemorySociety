# Retrieval Design

## Core Position

DMS retrieval should not be ordinary RAG:

```text
query -> top-k chunks -> prompt
```

For Diegetic Memory Society, retrieval is:

```text
task input + task intent + character perspective + story time
  -> input-specific extraction
  -> entity linking / claim extraction
  -> memory-agent candidate gathering
  -> diegetic visibility filtering
  -> relevance and constraint ranking
  -> task-specific memory packet
```

In short:

> Retrieval is not search. Retrieval is memory-packet construction under
> diegetic constraints.

The system should first ask what is legal for this viewpoint, then ask what is
relevant. This is the reverse of generic RAG, where relevance is usually the
first step.

## Design Principle

The guiding rule is:

> First legal, then relevant. First perspective, then semantics. First packet,
> then generation.

For DMS, the target is not to find text chunks. The target is to construct the
mental state that a character or authoring task is allowed to use at a selected
story moment.

## Retrieval Pipeline

```text
Task input
  -> Intent router
  -> Input parser
     - writing-intent parser for chapter generation
     - candidate claim/action parser for audit
  -> Entity linker / constraint resolver
  -> Retrieval frame builder
  -> Memory-agent candidate gathering around seed entities and claims
  -> Visibility gate
  -> Soft ranking
  -> Task-specific memory packet builder
  -> Simulation / planning / audit
```

### 1. Intent Router

The same memory item can have different access permissions under different
tasks. The first step is therefore to classify the task intent.

| Intent | Retrieval target | Permission model |
| --- | --- | --- |
| `character_simulation` | What this character can think, know, feel, and do now | strict character viewpoint; prefix and character-visible facts only |
| `writing_intent_generation` | Chapter plan or prose from a writing intent plus prefix memory | author view over prefix plus per-character visible packets |
| `next_plot_planning` | Prefix-consistent possible next story branches | author view over prefix only |
| `consistency_audit` | Contradictions, unresolved setups, motivation gaps, viewpoint errors | author/global view over prefix; hidden prefix facts can be read but must be labeled |
| `reference_alignment_eval` | Compare generated output with reference trajectory | evaluator-only; never available to generator |
| `branch_simulation` | Reason inside a speculative branch | candidate branch memory allowed; canonical prefix memory remains immutable |

This step prevents a common failure mode: accidentally using an audit-level or
evaluator-level memory item during character simulation.

### 2. Perspective And Story-Time Resolver

The resolver turns the request into an explicit retrieval frame:

```json
{
  "intent": "character_simulation",
  "character_id": "char_alice",
  "story_time": "T12",
  "memory_layer": "canonical_prefix",
  "branch_id": null
}
```

For writing-intent-conditioned chapter generation, the frame should also
contain intent-derived seed entities and scene constraints:

```json
{
  "intent": "writing_intent_generation",
  "boundary": "chapter_N_ending",
  "target": "chapter_N_plus_1",
  "accessible_layers": ["canonical_prefix"],
  "raw_writing_intent": "Write a scene where Alice finds the sealed letter in the library at midnight.",
  "linked_entities": {
    "characters": ["char_alice"],
    "locations": ["loc_library"],
    "objects": ["obj_sealed_letter"]
  },
  "time_hints": ["midnight"],
  "event_hints": ["find"],
  "scene_goal": "Alice discovers evidence connected to the hidden letter.",
  "conflict_type": "truth_vs_concealment"
}
```

If the request is ambiguous, the system should ask for clarification or default
to the safest frame:

- prefix-only;
- character-local visibility;
- canonical memory only.

## Writing Intent Parsing And Entity Linking

For writing-intent-conditioned generation, retrieval should begin by extracting entities
and scene constraints from the writing intent. The intent is the interface between
author intent and story memory.

### Writing Intent Parser

The system separates low-information social exploration, realistic author
input, and reference-only evaluation ground truth:

| Level | Prompt id | When to use | Output boundary |
| --- | --- | --- | --- |
| Social simulation intent | `dms/social_simulation_intent` | Exploratory character/social simulation | One low-information setup sentence with minimal anchors; less specific than writing intent; no behavior outcome, synopsis, beat list, or retrieval plan |
| Author writing intent | `dms/writing_intent` | Memory-packet construction and writing generation | One concise author-facing sentence with central anchors; much shorter than the source; no target outcome, synopsis, beat list, or retrieval plan |
| Writing spec | `dms/writing_spec` | Benchmark evaluation ground truth only | Compact required entities, narrative units, and state/relationship requirements; never fed into generation; natural-language content must not exceed the source scene length |

The old sparse/detailed writing-intent prompt split has been retired. New
benchmark runs should use `social_simulation_intent` for social exploration,
`writing_intent` for retrieval and generation, and `writing_spec` for
evaluation.

Writing generation also has a separate `previous_scene_context` channel for
immediate continuity. In benchmark runs it defaults to the previous script
scene: if the rendered context fits within 800 non-whitespace characters, the
full previous scene is shown; otherwise the writer sees a compact summary and
entity list. This channel is not used for retrieval scoring and is not a style
reference.

The parser should extract at least:

| Type | Example | Retrieval use |
| --- | --- | --- |
| character mentions | `Alice`, `the detective`, `the old man` | seed character memory profiles |
| location mentions | `library`, `ship`, `underground palace` | retrieve location state and spatial constraints |
| object mentions | `letter`, `key`, `ring` | retrieve object state, ownership, and setup/payoff records |
| organization/faction mentions | `police`, `royal court`, `rebels` | retrieve group goals, alliances, and secrets |
| time hints | `three days later`, `midnight`, `before the trial` | align with story-world time |
| event/action hints | `confront`, `escape`, `discover a body` | retrieve causal chains and related prior events |
| conflict/goal hints | `convince him to stay`, `hide the truth` | retrieve motivations and active threads |
| reveal/payoff hints | `reveal identity`, `use the earlier key` | retrieve secret/reveal ledger and foreshadowing |
| tone/genre cues | `tense`, `comic misunderstanding` | guide style, lower priority than factual constraints |

Example extracted writing intent:

```json
{
  "raw_writing_intent": "Alice confronts Basil in the library at midnight about the sealed letter.",
  "mentions": [
    {"text": "Alice", "type": "character"},
    {"text": "Basil", "type": "character"},
    {"text": "library", "type": "location"},
    {"text": "sealed letter", "type": "object"}
  ],
  "time_hints": ["midnight"],
  "event_hints": ["confront"],
  "scene_goal": "Alice pressures Basil about the sealed letter.",
  "conflict_type": "truth_vs_concealment"
}
```

### Canonical Entity Linking

Writing-intent mentions must be linked to canonical memory IDs before retrieval:

```text
Alice -> char_alice
the old man -> char_basil_father, with uncertainty if ambiguous
library -> loc_library
sealed letter -> obj_sealed_letter
```

Linking should combine:

| Signal | Purpose |
| --- | --- |
| exact / alias match | resolve names, nicknames, titles, and aliases |
| entity type match | prevent character/location/object confusion |
| prefix salience | prefer entities important in prefix memory |
| recency | prefer recently active entities when ambiguous |
| relation to other intent mentions | prefer candidates connected to other linked seeds |
| location/time compatibility | check whether an entity can plausibly appear in the scene |
| embedding similarity | resolve descriptive mentions and paraphrases |
| ambiguity handling | keep multiple candidates, ask the author, or choose safest high-salience candidate |

Important constraint:

> Entity linking must use only the selected prefix memory and the current task
> frame.

If the writing intent says "the true heir", the linker should resolve it only
against information already established in the prefix or keep the mention
ambiguous.

### Constraint Expansion

After linking, the system should expand from seed entities to relevant memory:

```text
seed entities
  -> related entities
  -> active threads
  -> object/location constraints
  -> secret/reveal constraints
  -> per-character visible memories
```

Example:

```text
Writing intent: Alice finds the sealed letter in the library.

Expansion:
  - Alice's current goals, beliefs, false beliefs, and relationships
  - library access/state and prior events there
  - sealed letter location, ownership, and who knows it exists
  - secrets or foreshadowing connected to the letter
  - other characters with motives to intervene
```

This prevents retrieval from becoming vague semantic search. It makes the writing intent a
structured seed for narrative memory construction.

### 3. Memory-Agent Candidate Gathering

This is where the "memory as agent society" idea becomes concrete. DMS should
not use one retriever over all memory. It should ask specialized memory agents
to return candidates from their own domains.

| Memory agent | Candidate memory it returns |
| --- | --- |
| `TimelineMemoryAgent` | prior events, causal chains, state transitions |
| `CharacterMemoryAgent` | known facts, beliefs, misbeliefs, goals, fears, emotions, relationships |
| `SecretRevealAgent` | secret status, reveal timing, who knows what |
| `ThreadMemoryAgent` | unresolved conflicts, promises, foreshadowing, payoff pressure |
| `ObjectLocationAgent` | object ownership, location state, spatial constraints |
| `WorldRuleAgent` | magic, technology, legal, social, or genre rules and exceptions |
| `DiscourseAgent` | reader knowledge, narrator withholding, misdirection, reveal pacing |

These agents return structured candidates, not prompt text. A coordinator later
filters, ranks, deduplicates, and assembles them.

For writing-intent-conditioned generation, agents should retrieve around linked seed
entities and scene constraints rather than around the raw intent string alone.

For consistency audit, agents should retrieve around candidate claims/actions
extracted from the new chapter rather than around the whole chapter as one
undifferentiated query.

### 4. Visibility Gate

The visibility gate is a hard filter. It should run before any generated
reasoning can see the candidate contents.

The core function is:

```text
is_visible(memory_item, character_id, story_time, intent, branch_id) -> decision
```

For character simulation, a memory item is visible only if:

- it is in an allowed memory layer;
- its story time is at or before the selected story time;
- it is not from another speculative branch;
- the selected character directly experienced, was told, inferred, believes,
  suspects, or misbelieves it;
- if it is a secret, it has already been revealed to that character;
- if the character holds a false belief, the false belief is preserved rather
  than silently replaced by the ground truth.

For audit mode, hidden prefix facts may be visible, but the packet must label
them as author/auditor-only. This lets the system detect character-knowledge
and viewpoint errors without letting character simulation use those facts.

Important detail:

> A false belief is not noise. It is a first-class memory item for simulation.

Characters often act from partial or wrong knowledge. Replacing false beliefs
with the true hidden fact would destroy the point of diegetic retrieval.

### 5. Soft Ranking

After hard filtering, candidates should be ranked. Ranking should not rely only
on embedding similarity. DMS needs narrative signals.

Suggested ranking features:

| Signal | Meaning |
| --- | --- |
| semantic relevance | lexical/embedding match to the request |
| intent seed relevance | relation to linked characters, locations, objects, and scene goal |
| recency | closeness to the selected story time |
| causal relevance | whether the item is a cause or consequence of the current state |
| character salience | relation to the selected character's goals, fears, relationships, or arc |
| thread pressure | relation to unresolved setups, promises, conflicts, or payoffs |
| entity overlap | shared characters, locations, objects, factions, or rules |
| emotional weight | strong emotional impact on the character |
| rule constraint | whether it restricts what can plausibly happen next |
| evidence confidence | whether the item is grounded in clear source spans |
| redundancy penalty | whether it repeats information already selected |

An initial scoring shape can be:

```text
score =
  semantic_relevance
  + causal_weight
  + character_salience
  + thread_pressure
  + recency
  + evidence_confidence
  - redundancy
```

The exact weights can start heuristic and later become an ablation target.

### 6. Memory Packet Builder

The final output should be a structured packet, not a blob of chunks.

Suggested packet fields:

```text
VisibleMemoryPacket
  - intent
  - selected_character
  - current_story_time
  - known_facts
  - believed_facts
  - false_beliefs
  - suspected_facts
  - recent_events
  - causal_backstory
  - active_goals
  - fears_and_emotional_state
  - relationship_state
  - unresolved_threads
  - relevant_world_rules
  - object_and_location_constraints
  - reader_or_narrator_state, when needed
  - evidence_spans
  - blocked_memory_log
```

The generation model should receive only fields allowed by the task intent. The
system log should retain `blocked_memory_log` so debugging and evaluation can
explain why private, branch-specific, or out-of-scope facts were withheld.

For chapter generation, the final packet should be a `ChapterGenerationPacket`,
not only a `VisibleMemoryPacket`:

```text
ChapterGenerationPacket
  - writing_intent
  - linked_seed_entities
  - prefix_world_state
  - involved_character_profiles
  - per_character_visible_memory_packets
  - active_threads
  - relevant_secrets_and_reveal_constraints
  - object_and_location_constraints
  - world_rules
  - blocked_private_or_out_of_scope_items_log
  - evidence_spans
```

For auditing, the final packet should be an `AuditContextPacket`:

```text
AuditContextPacket
  - candidate_claim_or_action
  - candidate_span
  - relevant_prefix_constraints
  - character_visibility_state
  - conflicting_or_supporting_evidence
  - issue_hypotheses
  - evidence_spans
```

## Memory Layers

Retrieval must respect memory layer boundaries.

```text
canonical_prefix_memory
  immutable memory extracted from chapters 1..N

candidate_branch_memory
  speculative memory created by generated branches

accepted_draft_memory
  memory promoted by the author after accepting a branch or draft

reference_evaluation_memory
  evaluator-only records from the original next unit, used for scoring only
```

Rules:

- generation never reads `reference_evaluation_memory`;
- character simulation reads only canonical/accepted memory plus the active
  branch, if explicitly requested;
- branch simulation can read its own candidate branch memory;
- generated branches never write into canonical prefix memory automatically;
- audit and evaluation should log which layers were accessible.

## Retrieval Modes

### Writing-Intent-Conditioned Chapter Generation Mode

Used for:

> Given this writing intent, write or plan the next chapter from prefix memory.

Flow:

```text
writing intent
  -> extract mentions, time hints, event hints, scene goal
  -> link mentions to canonical entities
  -> expand seed entities to active constraints
  -> retrieve via memory agents
  -> build author-level generation packet
  -> build per-character visible packets for dialogue/action
```

This mode must keep author context and character context separate:

| Context | Can contain |
| --- | --- |
| author generation packet | prefix-visible hidden facts, active threads, world constraints, reveal planning notes |
| character visible packet | only facts and beliefs visible to that character at the story time |

The model may use author-level context to plan structure, but character actions
and dialogue must be grounded in per-character visible packets.

### Character Simulation Mode

Strictest mode. Used for questions such as:

> What would Character A do at the end of Chapter 12?

Packet should include:

- known and believed facts;
- false beliefs;
- recent events experienced by the character;
- active goals and emotional state;
- relationship state;
- relevant constraints.

Packet should exclude:

- unrevealed secrets;
- facts known only by other characters;
- author-only explanations.

### Next Plot Planning Mode

Used for author support:

> What are plausible next plot branches?

This may use author-visible prefix memory, including hidden facts inside the
prefix. The packet should separate:

- what characters know;
- what the reader knows;
- what only the author/system knows from prefix evidence.

### Consistency Audit Mode

Used for checking:

> Did this scene give a character knowledge or motivation that is not supported yet?

Audit retrieval is claim-centered. The candidate chapter should first be parsed
into candidate events, facts, actions, dialogue claims, state changes, and
reveals. Each candidate item retrieves relevant prefix constraints.

Audit can inspect hidden prefix facts, reveal ledgers, and character knowledge
states. It should produce evidence-backed risk reports rather than character
simulation packets.

Flow:

```text
candidate chapter
  -> candidate_branch_memory
  -> parse candidate claims/actions/state changes
  -> retrieve relevant prefix constraints for each item
  -> compare candidate item against constraints
  -> emit structured issues
```

The candidate chapter must not update canonical prefix memory before audit.

### Reference Alignment Evaluation Mode

Used only in experiments. The evaluator can compare generated plans against
reference beats and reference-output scores, but the generator cannot access
those evaluator artifacts.

This mode should be physically separated in code and logs so the generation
trace remains inspectable.

## Logging Requirements

Every retrieval should write a trace:

```json
{
  "request_id": "...",
  "intent": "character_simulation",
  "character_id": "char_alice",
  "story_time": "T12",
  "raw_writing_intent": "Alice confronts Basil in the library at midnight.",
  "linked_seed_entities": ["char_alice", "char_basil", "loc_library"],
  "accessible_layers": ["canonical_prefix"],
  "candidate_counts_by_agent": {
    "TimelineMemoryAgent": 12,
    "CharacterMemoryAgent": 9
  },
  "blocked_memory": [
    {
      "memory_id": "fact_0087",
      "reason": "outside_selected_prefix_or_branch"
    },
    {
      "memory_id": "secret_0004",
      "reason": "not_revealed_to_character"
    }
  ],
  "selected_memory_ids": ["fact_0001", "event_0007"],
  "packet_path": "..."
}
```

This trace is important for:

- debugging;
- explaining suggestions to the author;
- explaining blocked private or out-of-scope memories;
- evaluating retrieval ablations.

## Implementation Plan

1. Extend `VisibleMemoryPacket` with fields for beliefs, false beliefs, goals,
   relationship state, world rules, object/location constraints, and blocked
   memory logs.
2. Add `RetrievalFrame` and `RetrievalIntent` schemas.
3. Add `WritingIntentFrame`, `EntityMention`, `EntityLinkCandidate`, and
   `LinkedSeedEntities` schemas.
4. Implement writing-intent parser and canonical entity linker that use only prefix
   memory.
5. Split `VisibilityGate` decisions into explicit `allow/block + reason`
   records.
6. Add memory-agent candidate interfaces:
   `TimelineMemoryAgent`, `CharacterMemoryAgent`, `ThreadMemoryAgent`, etc.
7. Add `ChapterGenerationPacket`, `CharacterVisibleMemoryPacket`, and
   `AuditContextPacket` builders.
8. Add retrieval trace JSONL logging.
9. Add ablation switches:
   no visibility gate, no thread memory, no character beliefs, no ranking,
   no entity linking, vector-only retrieval.

## Research Claim

The retrieval claim should be stated sharply:

> DMS improves narrative reasoning not by retrieving more context, but by
> constructing the right context: a perspective-constrained, temporally legal,
> evidence-backed memory packet for the current writing task.
