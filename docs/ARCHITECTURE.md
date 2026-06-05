# Code Architecture

This repository now has a real, prompt-managed prefix-memory prototype for
**Diegetic Memory Society**. It is still not the full paper system, but it now
does more than a fake MVP: it can use a local OpenAI-compatible Qwen endpoint to
extract scene frames/facts/questions, KG entity mentions, event candidates,
visibility notes, and merge them into a prefix world model.

## Current Layers

```text
src/dms/
  scripts/            # local source adapters
  prompts/            # YAML prompt loading and rendering
  llm/                # injectable model clients
  parsing/            # JSON output extraction and validation
  runners/            # reproducible extraction runs
  memory/             # staged, canonical, visibility, and world-model builders
  evaluation/         # dataset eligibility and evaluation split builders
  benchmark.py        # full-script writing benchmark orchestration
  timeline/           # diegetic temporal graph normalization, solving, reporting
  ui/                 # Gradio inspection/demo UI
  cli.py              # command-line workflows
```

## Input Adapter

`src/dms/scripts/wandering_earth.py` loads the local
`data/raw/流浪地球2剧本.json` fixture and normalizes it into ordered scene
records with source id, discourse index, heading fields, location hint, title,
and content.

## Prompt Management

Prompts live under `task_specs/prompts/dms/` as YAML files with declared
`task_variables` and `static_variables`, following the reference
NarrativeKnowledgeWeaver style.

Current prompts:

- `dms/scene_inventory`;
- `dms/kg_entity_mentions`;
- `dms/scene_event_candidates`;
- `dms/visibility_notes`.

`src/dms/prompts/loader.py` safely renders only declared variables, so JSON
schema braces inside prompts are preserved.

## Model Providers

`src/dms/llm/client.py` supports:

- deterministic fake clients for tests;
- Anthropic Messages-compatible calls;
- OpenAI-compatible `/v1/chat/completions` calls.

The currently exposed local Qwen model is:

```text
provider: openai
base_url: http://127.0.0.1:<port>
model: Qwen3-235B-FP8
chat_template_kwargs.enable_thinking: false
```

## Extraction Runners

Each runner writes the same reproducibility surface:

```text
inputs/<scene_id>.json
prompts/<scene_id>.txt
raw_outputs/<scene_id>.json
parsed/<scene_id>.json
trace.jsonl
manifest.json
summary.json
```

Implemented runners:

- `run-scene-inventory`: setting/frame, stated facts, open questions only;
- `run-kg-entities`: KG entity mentions and unresolved mentions;
- `run-scene-events`: events, knowledge transfers, state changes, thread candidates;
- `run-visibility-notes`: per-character visibility records and hidden/future-sensitive items.

## Memory Artifacts

The current memory flow is:

```text
scene inventory parsed outputs
  -> staged scene/frame/fact/question JSONL
  -> canonical prefix memory

KG entity parsed outputs
  -> staged KG entity JSONL

scene event parsed outputs
  -> staged event JSONL

visibility note parsed outputs
  -> staged visibility JSONL

canonical prefix memory + event JSONL + visibility JSONL + optional KG entity JSONL
  -> prefix world model
  -> visibility-grounded character packet

prefix world model
  -> entity registry / alias table / resolution traces
  -> relationship updates / relationship timeline
```

Implemented builders:

- `build-scene-inventory-memory`;
- `build-kg-entity-memory`;
- `build-canonical-memory`;
- `build-scene-event-memory`;
- `build-visibility-memory`;
- `build-prefix-world-model`.
- `build-entity-resolution`.

`scene_inventory` is deliberately light now: it does not accept model-emitted
`characters`, `objects`, or `entity_mentions`. Entities are extracted by
`kg_entity_mentions` into broader KG node candidates such as character, group,
organization, location, object, technology/facility, event/disaster,
world-rule/concept, media/document, and time/deadline.

The canonical layer still uses exact-name reconciliation for legacy staged
character/object files. The newer visibility-grounded packet uses explicit
visibility records and can fall back to KG character/group mentions when the
canonical layer has no character records.

`build-entity-resolution` is the first explicit entity/relationship layer. It
generates:

- `entities.jsonl`;
- `aliases.jsonl`;
- `resolution_traces.jsonl`;
- `relationship_updates.jsonl`;
- `relationships.jsonl`.

The current name rules handle deterministic normalization, Chinese short-name
variants for likely person names, English first/last and last/first variants,
and common title stripping such as `Dr.` or `Captain`. Relationship artifacts
are time-aware: each relationship has first/last scene fields and points back to
scene-level update records with evidence. The resolver also consumes
`kg_entity_mentions`; `canonical_hint` is used as an auditable alias grouping
signal when the extractor can justify the alias from context.

## Evaluation Eligibility

`src/dms/evaluation/eligibility.py` builds a deterministic first-pass split over
ordered scenes. The split does not decide whether a scene enters memory; it
decides whether a scene is a good target for writing-generation evaluation.

Generated split files:

- `scene_eligibility_all.jsonl`;
- `memory_prefix.jsonl`;
- `writing_eval_targets.jsonl`;
- `audit_eval_targets.jsonl`;
- `excluded_from_generation_eval.jsonl`.

Policy:

- empty scenes are excluded from memory and evaluation;
- pure visual/VFX, montage, transition, and establishing-description scenes are
  excluded from writing generation evaluation;
- visually driven scenes can still enter memory and audit evaluation when they
  contain state changes, world-state information, VO, or exposition;
- dialogue, conflict, and character-action scenes are writing-eval candidates.

## Writing Benchmark

`src/dms/benchmark.py` is the control layer for running the system over a full
script or a bounded subset. It has two stages:

1. `prepare-writing-benchmark`: builds eligibility splits, optionally runs the
   scene-ordered extractor, imports assets into SQLite, and builds a Chroma
   retrieval index.
2. `run-writing-benchmark`: for each eligible target scene, extracts a
   low-information `social_simulation_intent`, a concise author-facing
   `writing_intent`, and a reference-only `writing_spec`, builds a
   prefix-only memory packet, builds character attribute cards, runs social
   simulation, generates a raw draft, and evaluates the draft against the
   writing specification.

The default benchmark policy is deliberately conservative:

- social simulation uses `social_simulation_intent`, a deliberately
  under-specified setup for character exploration;
- generation uses `writing_intent`, a concise author brief that must stay much
  shorter than the target source scene;
- evaluation uses `writing_spec`, a compact ground-truth requirement
  specification extracted from the masked target scene;
- retrieval uses `before_scene_id=<target_scene>` and never reads target/future
  scene memories;
- writing receives a separate `previous_scene_context` for immediate local
  continuity by default. If the previous scene fits within the configured
  limit, currently 800 non-whitespace characters, the context contains the
  full previous scene; otherwise it contains a compact summary and entity list.
  This is distinct from `style_reference`, which defaults to disabled and can
  still be enabled explicitly for surface-form experiments;
- generated drafts are not repaired before evaluation.

## Diegetic Timeline

`src/dms/timeline/` is an independent temporal audit layer. It separates script
chapter order, story-world time, and reveal time. The current command
`run-temporal-extraction` extracts per-scene temporal events and relations, then
builds `timeline_graph.json`, `timeline_order.jsonl`, `conflicts.json`,
`timeline_report.md`, `temporal_audit.json`, and
`temporal_audit_report.md`. Relation evidence that cannot be aligned to the
current scene source is kept for audit but blocked from story-time ordering.

This layer is not yet wired into memory packet filtering. Benchmark retrieval
continues to use prefix visibility by discourse/reveal time so that past-world
events revealed in later scenes cannot leak into earlier target scenes.

`build-memory-timeline-index` can enrich `memories/episodic_memories.jsonl` with
story-time bucket/rank fields from `timeline_graph.json`. This produces an
auditable sidecar index and does not change the SQLite asset store or retrieval
defaults.

For existing scene-1-to-5 assets, start target runs from scene 6:

```bash
PYTHONPATH=src python3 -m dms.cli run-writing-benchmark \
  data/raw/流浪地球2剧本.json \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --output-dir runs/benchmark/we2_scene6_smoke \
  --start-scene-order 6 \
  --limit 1 \
  --overwrite
```

For a full-script benchmark, first build full-prefix assets, then run:

```bash
PYTHONPATH=src python3 -m dms.cli run-writing-benchmark \
  data/raw/流浪地球2剧本.json \
  --db-path runs/assets/<full_script>.sqlite \
  --chroma-dir runs/assets/<full_script>_chroma \
  --collection-name dms_retrieval_documents_bge_m3 \
  --output-dir runs/benchmark/<full_script_run> \
  --all-targets \
  --overwrite
```

Because the full script currently has 194 writing-eval target scenes, use
`--limit` for pilot batches before running `--all-targets`.

## Gradio UI

The lightweight UI is in `src/dms/ui/gradio_app.py` and can be launched through:

```bash
PYTHONPATH=src python3 -m dms.cli launch-ui \
  --benchmark-dir runs/benchmark/we2_scene6_smoke \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --server-port 7860
```

It exposes four views:

- benchmark overview table and aggregate summary;
- scene inspector for social simulation intent, writing intent, writing spec, memory packet, attribute cards,
  social simulation, draft, reference, and scores;
- one-scene run button for demos;
- eligibility split preview.

## CLI Examples

Build scene eligibility splits:

```bash
PYTHONPATH=src python3 -m dms.cli build-scene-eligibility \
  data/raw/流浪地球2剧本.json \
  --output-dir data/evaluation_splits/we2_scene_eligibility_20260530
```

Run real Qwen scene inventory under the split schema:

```bash
PYTHONPATH=src python3 -m dms.cli run-scene-inventory \
  data/raw/流浪地球2剧本.json \
  --output-dir runs/scene_inventory/we2_prefix2_qwen235_split_20260530 \
  --start 1 --limit 2 --no-dry-run \
  --provider openai \
  --base-url http://127.0.0.1:<port> \
  --auth-token <local-token> \
  --model Qwen3-235B-FP8
```

Run real Qwen KG entity mentions:

```bash
PYTHONPATH=src python3 -m dms.cli run-kg-entities \
  data/raw/流浪地球2剧本.json \
  --output-dir runs/kg_entities/we2_prefix2_qwen235_split_20260530 \
  --start 1 --limit 2 --no-dry-run \
  --provider openai \
  --base-url http://127.0.0.1:<port> \
  --auth-token <local-token> \
  --model Qwen3-235B-FP8
```

Build the merged prefix world model:

```bash
PYTHONPATH=src python3 -m dms.cli build-prefix-world-model \
  --canonical-dir runs/canonical_memory/we2_prefix2_qwen235_split_20260530 \
  --event-memory-dir runs/event_memory/we2_prefix2_fake_split_20260530 \
  --visibility-memory-dir runs/visibility_memory/we2_prefix2_fake_split_20260530 \
  --kg-entity-memory-dir runs/kg_entity_memory/we2_prefix2_qwen235_split_20260530 \
  --output-dir runs/world_model/we2_prefix2_qwen235_split_20260530
```

Build an explicit-visibility packet:

```bash
PYTHONPATH=src python3 -m dms.cli grounded-visibility-packet \
  runs/world_model/we2_prefix2_fake_with_kg_20260530 \
  --character FAKE_CHARACTER \
  --scene-id scene_0002
```

Build entity and relationship artifacts:

```bash
PYTHONPATH=src python3 -m dms.cli build-entity-resolution \
  runs/world_model/we2_prefix2_qwen235_split_20260530 \
  --output-dir runs/entity_resolution/we2_prefix2_qwen235_split_20260530
```

## Verified Split Run

The current split-schema Qwen smoke run is:

```text
runs/scene_inventory/we2_prefix2_qwen235_split_20260530/
runs/kg_entities/we2_prefix2_qwen235_split_20260530/
```

It used model `Qwen3-235B-FP8` from a local OpenAI-compatible endpoint and reports:

- scenes: 2;
- split inventory accepted scenes: 2;
- split inventory character/object counts: 0/0;
- stated facts: 14;
- open questions: 7;
- KG entity mentions: 22;
- unresolved KG mentions: 1;
- KG entity types: technology/facility 3, object 9, character 3, group 1, world-rule/concept 3, location 2, organization 1.

The split downstream wiring smoke is:

```text
runs/world_model/we2_prefix2_qwen235_split_20260530/
runs/entity_resolution/we2_prefix2_qwen235_split_20260530/
```

That world model intentionally combines Qwen split inventory/KG outputs with
fake event/visibility layers to verify wiring. It reports 2 scenes, 14 facts, 7
open questions, 22 KG entity mentions, 1 unresolved KG mention, and the entity
resolution pass reports 21 entities, 32 aliases, and 32 resolution traces.

The older pre-split real Qwen prefix run remains at
`runs/world_model/we2_prefix3_qwen_20260530/`. It used the previous one-shot
inventory schema where characters and objects were extracted inside
`scene_inventory`, so it should not be mixed with split-schema result tables.

The current full-script eligibility split is:

```text
data/evaluation_splits/we2_scene_eligibility_20260530/
```

It reports 373 scenes, 367 memory-prefix scenes, 194 writing-eval targets, 337
audit-eval targets, and 179 scenes excluded from generation evaluation.

## Known Limits

- The split schema is verified on the first 2 scenes with Qwen and fake
  downstream wiring; the older 3-scene Qwen world model is pre-split.
- Entity reconciliation now has a rule-based registry, but it is not yet robust
  enough for all cross-lingual, nickname, title, and ambiguous alias cases.
- Relationship extraction is a first-pass rule layer over events and knowledge
  transfers, not final semantic relationship understanding.
- Scene eligibility is heuristic and should be reviewed or LLM-adjudicated for
  boundary cases before final paper experiments.
- Visibility extraction is LLM-derived and not yet calibrated against human labels.
- Belief state, secret/reveal ledger, thread lifecycle, HLI planning, generation,
  audit evaluation, and ablation experiments are not complete.
- Long Qwen calls are slow; interrupted long runs can leave per-scene artifacts
  without final `summary.json`, so merged run directories must record provenance.
