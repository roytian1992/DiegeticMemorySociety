# Diegetic Timeline Design

## Purpose

The current memory system uses script chapter order as the prefix boundary. That
is still the right anti-leak default for writing benchmarks, but it is not the
same thing as story-world chronology. A scene can reveal a past event, anticipate
a future event, or run in parallel with another scene.

The timeline layer therefore separates three clocks:

- `discourse_order`: where a scene appears in the script.
- `story_time`: when an event occurs inside the fictional world.
- `revealed_at`: where the text reveals the event or fact.

The MVP builds an auditable temporal graph. It does not change memory packet
filtering.

## Extraction Surface

Command:

```bash
PYTHONPATH=src python -m dms.cli run-temporal-extraction \
  data/raw/流浪地球2剧本.json \
  --output-dir runs/dev/timeline_smoke \
  --start 1 \
  --limit 6 \
  --no-dry-run \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --overwrite
```

Artifacts:

```text
inputs/
prompts/
raw_outputs/
parsed/
trace.jsonl
events.jsonl
relations.jsonl
scene_temporal_index.jsonl
timeline_graph.json
timeline_order.jsonl
conflicts.json
timeline_report.md
temporal_audit.json
temporal_audit_report.md
manifest.json
summary.json
```

## Model Output

The LLM extracts per-scene candidates:

- `temporal_events`: story-world events, recalled past events, anticipated
  events, habitual events, and uncertain temporal mentions.
- `event_track`: separates `plot`, `context`, `forecast`, `memory`,
  `hypothetical`, and `unknown` events.
- `temporal_relations`: `before`, `after`, `overlaps`, `same_time`, `contains`,
  `causes`, `anticipates`, `claims`, `reveals_past`, or `uncertain`.
- `scene_temporal_index`: scene-level temporal cues and ambiguity flags.
- `temporal_warnings`: montage, flashback ambiguity, parallel action, or
  contradictory cues.

Every item must include short textual evidence from the current scene.

## Solver Policy

LLM output is treated as candidate evidence. Code builds the graph:

- explicit high-confidence relations are hard ordering edges;
- explicit relations whose evidence cannot be aligned to the current source
  text are kept for audit but are not used as ordering edges;
- low-confidence relations remain visible but weak;
- script chapter order is a weak prior only;
- cycles are reported in `conflicts.json`, not silently repaired;
- overlapping or same-time relations form timeline buckets.

This avoids pretending that a single linear timeline is always recoverable.

## Evidence Audit

`run-temporal-extraction` now runs a temporal evidence audit after parsing and
before graph solving. The audit checks each event, relation, scene index, and
warning evidence against the current unit's `title`, `subtitle`, and `content`.

Audit policy:

- exact or normalized/fuzzy contiguous source spans pass;
- missing, empty, paraphrased, or ellipsis-compressed evidence is reported;
- rejected relation evidence sets `audit_ordering_usable=false`, so the solver
  does not use that relation for story-time ordering;
- likely English output on Chinese source text is reported as a warning for
  manual cleanup, not as a parse failure.

## Non-Plot Scenes

Visual-only scenes, subtitle-only montage panels, establishing shots, and broad
world-state images should not be forced into the main plot chain.

Policy:

- concrete character action and dialogue action use `event_track=plot`;
- visual/world-state panels such as "冰川融化" use `event_track=context`;
- plans and predictions such as "一百年后太阳吞没地球" use
  `event_track=forecast` and connect through `anticipates` or `claims`, not
  ordinary `before`;
- memories and backstory use `event_track=memory`;
- empty-content scenes should use the most specific subtitle as evidence before
  falling back to the title.

The graph exposes `main_timeline_order`, `context_timeline_order`, and
`forecast_timeline_order` separately. Memory retrieval should continue to rely
on reveal time unless a caller explicitly asks for story-time display.

## Memory Policy

The benchmark anti-leak rule stays unchanged:

```text
visible_to_target = revealed_at is before target discourse scene
```

Even if an event happened earlier in world time, it must not enter a target
scene's writing context if it is only revealed in a later script scene.

Future integration should add a world-model display mode that can sort memories
by `story_time`, while memory retrieval still filters by `revealed_at` and
visibility.

## Memory Timeline Index

`build-memory-timeline-index` enriches episodic memory JSONL with the temporal
graph. It preserves the existing `timeline_index` semantics as discourse scene
sequence and adds separate story-time fields.

Command:

```bash
PYTHONPATH=src python -m dms.cli build-memory-timeline-index \
  runs/scene_ordered/example/memories \
  --timeline-graph runs/dev/timeline_smoke/timeline_graph.json \
  --output-dir runs/dev/memory_timeline_index \
  --overwrite
```

Outputs:

```text
enriched_episodic_memories.jsonl
story_time_memory_index.jsonl
memory_timeline_links.jsonl
unmatched_memories.jsonl
summary.json
```

Added fields include:

```text
discourse_timeline_index
timeline_index_semantics
memory_timeline_index_status
story_event_id
story_time_index
story_time_bucket
story_time_rank
story_time_mode
story_time_hint
story_time_granularity
story_time_confidence
revealed_at_scene_id
revealed_at_order
temporal_relation_basis
story_time_match_score
```

The builder uses same-scene event candidates and evidence/summary similarity.
Unmatched memories remain in the enriched output with `story_time_* = null` and
are also written to `unmatched_memories.jsonl`.
