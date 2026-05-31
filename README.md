# Diegetic Memory Society

## Core Idea

**Diegetic Memory Society** is a concept for long-form creative writing: a narrative memory system where memory is not a passive database serving an agent, but is itself organized as a multi-agent system.

Most agent-memory work asks:

> How can memory serve an agent or a multi-agent system?

This project asks a different question:

> What if memory itself is a society of agents?

In this view, a story's memory is not stored only as flat notes, vectors, or summaries. It is enacted by multiple timeline-bounded narrative agents, each carrying a partial, situated, and role-specific view of the story world.

## One-Sentence Description

Diegetic Memory Society turns a long-form story into a timeline-constrained multi-agent memory system, where character agents can only access what they should know at a given story moment, enabling creative simulation and consistency auditing with explicit prefix and perspective boundaries.

## Why This Is Different

Existing memory-augmented agent systems usually treat memory as an external module:

- an agent writes to memory;
- an agent retrieves from memory;
- a multi-agent system shares or coordinates through memory.

Here, memory is not merely a support module. The memory layer itself becomes a society:

| Common framing | Diegetic Memory Society framing |
| --- | --- |
| Memory for agents | Memory as agents |
| Passive storage | Active perspective-bearing memory |
| Global retrieval | Role- and time-constrained retrieval |
| Omniscient story context | Diegetic, in-world knowledge |
| Single memory bank | Multi-agent memory society |

The central claim is:

> A narrative memory system should not only remember what happened. It should remember who knew what, when they knew it, how they interpreted it, and what they could plausibly do next.

## Motivation

Long-form writing has a persistent memory problem. As the story grows, the author has to track:

- events and chronology;
- character knowledge;
- private secrets;
- misunderstandings;
- promises and foreshadowing;
- world rules;
- changing motivations;
- unresolved conflicts;
- contradictions introduced by later chapters.

A normal writing assistant can read a large manuscript context, but this is also its weakness. If it has an omniscient view, it may generate suggestions that use hidden secrets or facts that a character should not know yet.

For story reasoning, the important unit is not only relevance. It is **visibility**.

At a given story moment, a character should only reason from:

1. events that already happened in the story timeline;
2. information that the character experienced, heard, inferred, or plausibly believes;
3. the character's goals, emotions, biases, and misunderstandings at that moment.

## System Concept

The system runs in the background while an author writes. It continuously parses completed text and turns it into a structured narrative memory society.

### 1. Story Parser

Extracts narrative units from the manuscript:

- events;
- characters;
- locations;
- objects;
- relationships;
- secrets;
- promises;
- foreshadowing;
- world rules;
- conflicts;
- emotional shifts;
- causal links.

### 2. Story-World Timeline

Organizes extracted events by the internal chronology of the story, not merely by chapter order or writing order.

This matters because stories often contain:

- flashbacks;
- parallel timelines;
- unreliable narration;
- delayed revelations;
- non-linear chapter order.

### 3. Memory Agents

Instead of one global memory store, the system creates multiple memory agents. Possible agents include:

| Agent type | What it represents |
| --- | --- |
| Character memory agent | What a specific character knows, believes, wants, and misunderstands |
| Secret memory agent | Hidden facts and their reveal conditions |
| Foreshadowing memory agent | Hints, promises, and later payoffs |
| Timeline memory agent | Event order, causality, and temporal constraints |
| World-rule memory agent | Stable rules of the fictional world |
| Consistency auditor agent | Cross-checks contradictions, viewpoint errors, and unresolved threads |

These agents do not all see the same information. Their access is governed by story time, role, evidence, and narrative permissions.

### 4. Visibility Control

The key mechanism is a visibility gate.

When the author selects a plot point and a character perspective, the system asks:

- Has this event happened before the selected story time?
- Does this character know this fact?
- Is this information secret, inferred, rumored, mistaken, or confirmed?
- Was the knowledge acquired through direct experience, dialogue, observation, or deduction?
- Is this information allowed under the selected prefix, branch, and character perspective?

Only visible memory is passed to the character agent.

### 5. Simulation and Auditing

Once memory is constrained by timeline and perspective, the system can support two main workflows.

#### Creative Simulation

The author can ask:

- What would this character do next from this point?
- How would this character interpret the event without knowing the later truth?
- What possible plot branches follow from this character's current goals and false beliefs?
- What dialogue would be plausible given what both characters know at this moment?

This is not generic continuation. It is perspective-constrained story simulation.

#### Consistency Auditing

The author can also ask:

- Did a character know something too early?
- Did a later chapter contradict an earlier timeline?
- Was a foreshadowed object, promise, or threat ever resolved?
- Did a character's motivation change without enough intermediate evidence?
- Did a character use knowledge that the prefix does not support for that perspective?

This makes the system a plot-hole detector with explicit temporal and perspectival constraints.

## Example Workflow

1. The author writes chapters 1 to 17.
2. The system extracts story events, character states, secrets, and foreshadowing.
3. The author selects "Chapter 12 ending" and "Character A".
4. The system constructs Character A's visible memory at that point.
5. Character A agent simulates likely next actions based on goals, fears, past experiences, and incomplete knowledge.
6. The author writes later chapters.
7. The auditor agents compare later developments against the earlier timeline and character knowledge.
8. The system flags possible issues: character-knowledge mismatch, unclosed foreshadowing, motivation inconsistency, timeline contradiction.

## Relationship to Agent Memory Research

This idea is inspired by the agent memory taxonomy:

- **Forms**: token-level structured memory, timelines, event graphs, character graphs, secret indexes, foreshadowing traces.
- **Functions**: factual memory for story facts, experiential memory for character behavior patterns, working memory for current plot-point simulation.
- **Dynamics**: memory formation from manuscript parsing, memory evolution through updates and contradictions, memory retrieval through time- and perspective-constrained access.

The distinctive part is that memory is not a single substrate used by agents. Memory is decomposed into agents that negotiate, constrain, and audit the story world.

## Naming Candidates

Current preferred name:

> **Diegetic Memory Society**

Why it works:

- **Diegetic** means inside the story world, rather than from the author's omniscient outside view.
- **Memory** emphasizes that the system is about persistent narrative knowledge.
- **Society** captures that memory is organized as many interacting agents, not a single store.

Other possible names:

- Memory-as-Agent-Society
- Narrative Memory Society
- Society of Memories
- Chronology-Aware Memory Society
- Memory as a Cast
- Story Memory Society
- DiegeticMind
- MemoCast

Possible paper-style title:

> **Diegetic Memory Society: Memory-as-Agent-Society for Timeline-Constrained Story Reasoning**

Alternative title:

> **From Memory for Agents to Memory as Agents: A Diegetic Memory Society for Long-Form Story Writing**

## Design Principle

The system should not take authorship away from the writer.

It should provide:

- plausible branches, not one definitive continuation;
- reasons for each simulation, not opaque suggestions;
- risk flags, not automatic rewrites;
- character-bounded perspectives, not omniscient narration;
- traceable evidence, not unsupported claims.

The author remains the creative authority. The system acts as a narrative memory society that can simulate, challenge, and audit the story world.

## Open Questions

| Question | Possible direction |
| --- | --- |
| How to extract a reliable story-world timeline from free-form prose? | event extraction, temporal ordering, chapter-level incremental parsing |
| How to determine what each character knows? | viewpoint modeling, evidence tracking, information propagation |
| How to represent secrets and false beliefs? | private memory, belief states, reveal conditions, confidence labels |
| How to enforce prefix and perspective boundaries? | temporal access control, character-level retrieval filters |
| How should memory agents communicate? | debate, blackboard coordination, role-specific memory exchange |
| How to evaluate story simulation quality? | author preference, plot consistency, character motivation consistency |
| How to avoid over-directing the author? | branch generation, explanation-first suggestions, non-destructive edits |

## Current Project Direction

The immediate goal is to keep this as a concept note and early prototype direction.

The most important next step is to make the core distinction crisp:

> Memory is not a passive module attached to agents.  
> Memory is a society of agents whose partial perspectives collectively maintain and reason over a story world.
