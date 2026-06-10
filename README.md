# Diegetic Memory Society

Diegetic Memory Society is a prototype memory system for long-form writing. It turns completed narrative units into prefix-bounded assets: a small knowledge graph, evidence-grounded episodic memories, diegetic timeline audits, external reference context, entity attribute cards, social simulation outputs, and writing/evaluation artifacts.

![DMS overview](assets/DMS_overview.png)

## Current Status

The repository now contains a working Python prototype rather than only the original concept note.

- Narrative-unit processing defaults to script scenes, but CLI commands also accept `--unit-type` and `--unit-label` for chapter/passage-style inputs while keeping legacy `scene_id` storage compatibility.
- Ordered processing runs by narrative unit, while independent extraction tasks inside one unit can run concurrently.
- Long scenes are split with a configurable maximum chunk size, currently defaulting to `800` English words or Chinese characters.
- Extracted entity types are intentionally limited to seven graph-facing categories: `character`, `group`, `organization`, `location`, `object`, `concept`, and `occasion`.
- Episodic memories keep source evidence for traceability, evidence is aligned back to the original unit text, and each memory is tagged as `temporal_episode`, `atemporal_fact`, `durable_state`, or `uncertain`.
- A diegetic timeline audit layer can extract story-world temporal events and relations, align temporal evidence, block unsupported ordering edges, and build an auditable timeline graph.
- Author-provided entity profiles can initialize character/entity baselines without being treated as screenplay-derived evidence.
- Durable relations are reserved for longer-lived state-like relationships rather than momentary actions.
- SQLite stores entities, aliases, relations, memories, scene summaries, metadata, and retrieval documents.
- Chroma provides vector retrieval over memory and summary documents.
- External reference documents are managed as LightRAG-style source-aware assets: full docs, text chunks, doc status, extracted entities, relationships, graph indexes, and vector documents can be stored in a standalone reference SQLite DB and optionally indexed in Chroma.
- The retrieval pipeline builds a prefix-only memory packet for a target unit, so generation cannot read target or future unit memories.
- External reference context is opt-in and is kept separate from screenplay memory and canonical KG edges.
- Social-simulation intent extraction produces a low-information `social_simulation_intent` for exploratory character behavior.
- Scene disposition notes provide a lightweight, natural-language, memory-grounded soft prior before social simulation.
- Writing-intent extraction produces a concise author-facing `writing_intent` for retrieval and generation.
- Writing-spec extraction produces `writing_spec` as evaluation ground truth only; it is not fed into writing prompts.
- Social simulation is positioned as a low-spec ideation layer: it uses the low-information social intent plus retrieved memory and character cards to suggest plausible character behavior.
- Writing generation uses a separate `writing_llm` config section, receives an explicit previous-scene continuity context by default in benchmark runs, and does not apply automatic post-generation repair.
- Evaluation is LLM-as-judge over three dimensions: writing intent consistency, writing quality, and memory faithfulness.
- A Gradio UI is available for inspecting benchmark runs, scene artifacts, memory packets, social simulation, drafts, and scores.

## Repository Layout

```text
src/dms/
  benchmark.py              # full writing benchmark orchestration
  cli.py                    # command line entry points
  config.py                 # local YAML model config loader
  evaluation/               # scene eligibility and writing evaluation
  llm/                      # fake and OpenAI-compatible model clients
  memory/                   # staged memory, KG, relations, world model
  reference_library.py      # external reference ingest, extraction, indexing, retrieval
  parsing/                  # JSON extraction with json_repair fallback
  prompts/                  # YAML prompt loading
  retrieval/                # memory packet construction
  runners/                  # extraction runners and ordered pipeline
  scripts/                  # source script adapters
  simulation/               # attribute cards and social simulation
  storage/                  # SQLite import and Chroma indexing
  timeline/                 # diegetic temporal extraction audit and timeline solver
  ui/                       # Gradio app
task_specs/
  prompts/dms/              # prompt-managed extraction, simulation, writing, eval tasks
  task_settings/            # task-level schemas and policies
data/reference_library/     # small public-source external reference test corpus
data/evaluation_splits/     # committed deterministic eligibility splits
scripts/
  run_full_benchmark.sh     # long-running full-script tmux entry point
```

Local-only files are intentionally ignored by git:

- `configs/local_config.yaml`
- `configs/local_model_config.yaml`
- `data/raw/`
- `runs/`
- `outputs/`
- `logs/`
- `docs/experiment_log_*.md`

## Setup

The current development environment is the local conda environment named `screenplay`.

```bash
conda activate screenplay
cd DiegeticMemorySociety
export PYTHONPATH=src
```

Install editable package metadata if needed:

```bash
pip install -e .
```

For Chroma indexing and the UI, the environment also needs `chromadb` and `gradio`.

## Local Config

Use a local YAML config at `configs/local_config.yaml`. This file is ignored and should not be committed.

```yaml
llm:
  provider: openai
  model_name: Qwen3-235B-FP8
  api_key: <local-token>
  base_url: http://127.0.0.1:8002
  max_tokens: 4096
  timeout: 240
  temperature: 0
  enable_thinking: false
  include_chat_template_kwargs: true

embedding:
  provider: openai
  model_name: bge-m3
  api_key: <local-token>
  base_url: http://127.0.0.1:8081/v1
  max_tokens: 8192
  dimensions: 1024
  timeout: 60

writing_llm:
  provider: openai
  model_name: gpt-5.5
  api_key: <local-secret>
  base_url: <writing-llm-openai-compatible-base-url>
  max_tokens: 4096
  timeout: 240
  temperature: 0.7
  reasoning_effort: high
  # Some OpenAI-compatible reasoning endpoints need this instead of
  # chat_template_kwargs.enable_thinking=false.
  # thinking:
  #   type: disabled
  # include_chat_template_kwargs: false
```

The current code uses the LLM and embedding sections. Creative Context Kernel
commands can also call an OpenAI-compatible LLM. For local Qwen/vLLM servers,
use `include_chat_template_kwargs: true` with `enable_thinking: false`; for
OpenAI-style reasoning endpoints, use `thinking: {type: disabled}` and
`include_chat_template_kwargs: false`. Commands can optionally use a FastAPI
reranker for source-aware packet ranking.

## Common Commands

Inspect the raw script:

```bash
python -m dms.cli inspect-script data/raw/流浪地球2剧本.json --limit 3
```

Build deterministic scene eligibility splits:

```bash
python -m dms.cli build-scene-eligibility \
  data/raw/流浪地球2剧本.json \
  --output-dir data/evaluation_splits/we2_scene_eligibility_20260530
```

Run ordered extraction for a small pilot batch:

```bash
python -m dms.cli run-scene-ordered-pipeline \
  data/raw/流浪地球2剧本.json \
  --output-root runs/scene_ordered/we2_scene12345_qwen235_8002_7types \
  --start 1 \
  --limit 5 \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --scene-task-concurrency 3 \
  --max-chunk-units 800 \
  --unit-type scene \
  --unit-label scene \
  --overwrite
```

Run temporal extraction and build an auditable diegetic timeline graph:

```bash
python -m dms.cli run-temporal-extraction \
  data/raw/流浪地球2剧本.json \
  --output-dir runs/timeline/we2_scene1_10_qwen235_8002 \
  --start 1 \
  --limit 10 \
  --no-dry-run \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --max-tokens 1800 \
  --timeout-seconds 240 \
  --overwrite
```

External references expose a small mem0-like facade for normal application code:
`add()` ingests a source and builds the internal LightRAG-style assets, while
`search()` retrieves source-grounded reference memories. The lower-level CLI
pipeline remains available for debugging and offline inspection.
See [External References Usage](docs/external_references_usage.md) for the
Python facade, FastAPI facade, CLI pipeline, and fact/property configuration
switches.

```python
from dms.external_references import ExternalReferences, ExternalReferencesConfig
from dms.llm import OpenAIChatClient

refs = ExternalReferences(
    ExternalReferencesConfig(
        db_path="runs/reference_library/we2_refs.sqlite",
        work_dir="runs/reference_library/we2_refs_work",
        chroma_dir="runs/reference_library/we2_refs_chroma_bge_m3",
        auto_index=True,
        embedding_provider="openai",
        embedding_model="bge-m3",
        embedding_base_url="http://127.0.0.1:8081/v1",
        embedding_api_key="token-abc123",
        embedding_dim=1024,
    ),
    llm_client=OpenAIChatClient(
        base_url="http://127.0.0.1:8002",
        api_key="token-abc123",
        model="Qwen3-235B-FP8",
        include_chat_template_kwargs=True,
        enable_thinking=False,
    ),
)

refs.add("data/reference_library/we2_web_refs_20260605")
hits = refs.search("张鹏和刘培强的关系", evidence_budget="standard")
context = hits["evidence_packet"]["answer_context"]
```

### FastAPI Service

Install the service extra when running the HTTP API:

```bash
pip install -e '.[service]'
```

Start the service with local paths. The default provider is `fake`; set
`DMS_PROVIDER=openai` and `DMS_MODEL_CONFIG=configs/local_config.yaml` for a
real OpenAI-compatible LLM.

```bash
export DMS_REFERENCE_DB=runs/reference_library/we2_refs.sqlite
export DMS_REFERENCE_WORK_DIR=runs/reference_library/service_work
export DMS_WORKERS=8
dms-service --host 127.0.0.1 --port 8000
```

Memory facade endpoints:

- `GET /health`
- `POST /add`
- `POST /search`
- `GET /get_all`

Example:

```bash
curl -X POST http://127.0.0.1:8000/add \
  -H 'Content-Type: application/json' \
  -d '{"input_path":"data/reference_library/we2_web_refs_20260605","workers":8,"include_fact_properties":false}'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"张鹏和刘培强的关系","evidence_budget":"standard","fact_binding_top_k":4}'

curl http://127.0.0.1:8000/get_all
```

`refs.search()` uses DMS's source-aware LightRAG asset flow rather than a single
keyword-concatenated query. It builds a compact query plan with
`low_level_keywords` for entity/chunk anchors and `high_level_keywords` for
relationship/fact intent, runs balanced, low-level, and high-level retrieval
passes with different namespace weights, then fuses source-local entities,
relationships, facts, properties, chunks, pseudo-graph signals, and source
notes into an `evidence_packet`. The packet contains stable evidence IDs such as
`[F1]`, `[R1]`, and `[C1]`, plus `answer_context` for final answer prompts.
`source_doc_ids`, `source_paths`, and `source_scope_ids` filters are applied to
every retrieval pass.

Use `evidence_budget` instead of a flat hit limit. Built-in profiles are
`compact` for realtime dialogue, `standard` for normal writing context, and
`deep` for broader reference inspection. A dict can override section budgets,
for example `{"profile": "deep", "max_facts": 12, "max_chunks": 6,
"include_answer_context": true}`. The older `limit=` argument is kept only as a
legacy quantity-budget override.

The internal asset path follows the LightRAG asset model: ingest files, extract a
chunk-level KG, import source-local entities/relationships into SQLite, then
optionally build a vector index over chunks, source-local entities/relationships,
atomic facts, entity properties, and pseudo graph assets.

```bash
python -m dms.cli ingest-reference-library \
  data/reference_library/we2_web_refs_20260605 \
  --output-dir runs/reference_library/we2_refs_ingest \
  --overwrite

python -m dms.cli extract-reference-kg \
  runs/reference_library/we2_refs_ingest \
  --output-dir runs/reference_library/we2_refs_kg_qwen235_8002 \
  --no-dry-run \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --overwrite

python -m dms.cli import-reference-knowledge \
  --library-dir runs/reference_library/we2_refs_ingest \
  --kg-dir runs/reference_library/we2_refs_kg_qwen235_8002 \
  --output-db runs/reference_library/we2_refs.sqlite \
  --overwrite
```

The imported DB uses the source-local external reference asset model
(`source_local_external_reference_v1`). Each ingested raw document is a default
`source_scope`; entities with the same name are not truly merged across scopes.
Cross-source alignment is represented only as pseudo clusters and pseudo
relationships, so reference facts remain tied to their original source.

Core imported assets:

- `reference_full_docs`, `reference_text_chunks`, `reference_doc_status`, and `reference_llm_response_cache`
- `reference_extracted_entities` and `reference_extracted_relationships`
- `reference_source_scopes`
- `reference_source_local_entities` and `reference_source_local_relationships`
- `reference_entity_clusters`, `reference_entity_cluster_members`, and `reference_pseudo_relationships`
- `reference_atomic_facts` and `reference_entity_properties`
- `reference_full_entities`, `reference_full_relations`, `reference_entity_chunks`, and `reference_relation_chunks`
- `reference_vector_documents` for chunk, source-local KG, claim/property, and pseudo graph retrieval

Optionally build a reference Chroma index. Source-local entity vector text uses
`entity_name + entity_type + description + facts + attributes`; source-local
relationship vector text uses `keywords + source/target + description + facts`.
Pseudo clusters and pseudo relationships are also indexed for query-time graph
insights, but they do not rewrite source-local facts.

```bash
python -m dms.cli build-reference-index \
  runs/reference_library/we2_refs.sqlite \
  --persist-dir runs/reference_library/we2_refs_chroma_bge_m3 \
  --collection-name dms_reference_knowledge_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --overwrite
```

You can query all references or restrict retrieval to specific source files:

```bash
python -m dms.cli query-reference-knowledge \
  runs/reference_library/we2_refs.sqlite \
  --query "张鹏和刘培强的关系" \
  --source-path profiles.md \
  --chroma-dir runs/reference_library/we2_refs_chroma_bge_m3 \
  --collection-name dms_reference_knowledge_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding
```

Query output includes an `evidence_board` with matched source-local entities,
source-local relationships, atomic facts, entity properties, supporting chunks,
matched pseudo clusters, pseudo relationships, graph insights, and source
separation notices. File-restricted queries filter by `source_doc_id` and derived
`source_scope_id`; all-reference queries can still use pseudo graph signals for
degree and source-aware navigation.

Import an ordered run into SQLite:

```bash
python -m dms.cli build-asset-store \
  --run-root runs/scene_ordered/we2_scene12345_qwen235_8002_7types \
  --output-db runs/assets/we2_scene12345_7types.sqlite \
  --overwrite
```

## Creative Context Kernel

`build-memory-packet` remains the legacy-compatible writing retrieval path. The
new `context_kernel` sidecar is the source-aware standardization layer for
conversation memory, narrative artifact memory, external reference knowledge,
promotion/history, entity patches, and `creative_context_packet` assembly.

Ingest conversation memory from a transcript:

```bash
python -m dms.cli context-ingest-conversation \
  runs/conversations/session.jsonl \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --conversation-id dev_session \
  --reset-context-store
```

Import existing screenplay/artifact memory and external references into the
same sidecar:

```bash
python -m dms.cli context-import-artifacts \
  runs/assets/we2_scene12345_7types.sqlite \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2

python -m dms.cli context-import-references \
  runs/reference_library/we2_refs.sqlite \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2
```

Build a source-aware packet. External references and conversation guidance are
kept separate from canonical narrative artifact memory; entity patches are
included as author-facing constraints.

```bash
python -m dms.cli build-creative-context-packet \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --request "写一段返航途中刘培强和张鹏的互动。" \
  --before-unit-id scene_0006 \
  --before-unit-order 6 \
  --entity-id character_0011 \
  --entity-id character_0017 \
  --reranker fastapi \
  --reranker-base-url http://127.0.0.1:8090 \
  --format markdown \
  --output runs/context/scene6_creative_context_packet.md
```

Ask external references only. Add `--answer-with-llm` to generate a concise
source-grounded answer; without it, the command returns retrieval evidence only.

```bash
python -m dms.cli context-external-qa \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --question "张鹏在外部资料里是什么身份？" \
  --answer-with-llm \
  --provider openai \
  --model Qwen3-235B-FP8 \
  --base-url http://127.0.0.1:8002 \
  --auth-token token-abc123
```

Promote a conversation or external item only after explicit acceptance:

```bash
python -m dms.cli context-promote-item \
  --context-db runs/context/we2_context.sqlite \
  --item-id external:ref_zhangpeng_profile \
  --target-layer story_bible \
  --reason "采纳为角色设定"
```

Export schemas, build a statement-level context index, and run review audits:

```bash
python -m dms.cli export-context-json-schemas \
  --output-dir docs/generated/context_schemas

python -m dms.cli build-context-index \
  --context-db runs/context/we2_context.sqlite \
  --persist-dir runs/context/we2_context_chroma_bge_m3 \
  --project-id we2 \
  --collection-name dms_context_documents_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --overwrite

python -m dms.cli search-context-index \
  --context-db runs/context/we2_context.sqlite \
  --persist-dir runs/context/we2_context_chroma_bge_m3 \
  --project-id we2 \
  --collection-name dms_context_documents_bge_m3 \
  --query "张鹏 训练教官 法定监护人" \
  --source-type external_reference \
  --model-config configs/local_config.yaml \
  --embedding-section embedding

python -m dms.cli context-audit-conflicts \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --output runs/context/external_artifact_conflicts.json

python -m dms.cli context-export-external-entity-links \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --output runs/context/external_entity_links.json

python -m dms.cli context-export-external-timeline-claims \
  --context-db runs/context/we2_context.sqlite \
  --project-id we2 \
  --output runs/context/external_timeline_claims.json
```

Social simulation and final writing can optionally consume the same packet:

```bash
python -m dms.cli build-entity-attribute-cards \
  runs/benchmark/scene6/memory_packet.json \
  --creative-context-packet runs/context/scene6_creative_context_packet.json \
  --output-dir runs/benchmark/scene6/attribute_cards \
  --model-config configs/local_config.yaml

python -m dms.cli build-scene-disposition-notes \
  runs/benchmark/scene6/attribute_cards/attribute_cards.json \
  --memory-packet runs/benchmark/scene6/memory_packet.json \
  --creative-context-packet runs/context/scene6_creative_context_packet.json \
  --social-simulation-intent "刘培强和张鹏在返航途中互动。" \
  --output-dir runs/benchmark/scene6/disposition_notes \
  --model-config configs/local_config.yaml

python -m dms.cli run-social-simulation \
  runs/benchmark/scene6/attribute_cards/attribute_cards.json \
  --scene-disposition-notes runs/benchmark/scene6/disposition_notes/scene_disposition_notes.json \
  --creative-context-packet runs/context/scene6_creative_context_packet.json \
  --social-simulation-intent "刘培强和张鹏在返航途中互动。" \
  --output-dir runs/benchmark/scene6/social_simulation \
  --model-config configs/local_config.yaml

python -m dms.cli generate-writing-social \
  --writing-request "写一段返航途中刘培强和张鹏的互动。" \
  --memory-packet-file runs/benchmark/scene6/memory_packet.md \
  --creative-context-packet-file runs/context/scene6_creative_context_packet.md \
  --attribute-cards-file runs/benchmark/scene6/attribute_cards/attribute_cards.md \
  --social-simulation-file runs/benchmark/scene6/social_simulation/writer_packet.md \
  --output-dir runs/benchmark/scene6/writing \
  --model-config configs/local_config.yaml
```

Build a Chroma index with the configured embedding service:

```bash
python -m dms.cli build-chroma-index \
  runs/assets/we2_scene12345_7types.sqlite \
  --persist-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --overwrite
```

Run the current scene-6 smoke benchmark against scene-1-to-5 assets:

```bash
python -m dms.cli run-writing-benchmark \
  data/raw/流浪地球2剧本.json \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --output-dir runs/benchmark/we2_scene6_new_pipeline \
  --target-scene-id scene_0006 \
  --limit 1 \
  --overwrite
```

Build a standalone memory packet with optional external reference context:

```bash
python -m dms.cli build-memory-packet \
  runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --writing-intent "写一段返航途中刘培强和张鹏的互动。" \
  --before-scene scene_0006 \
  --include-reference-context \
  --reference-db runs/reference_library/we2_refs.sqlite \
  --reference-chroma-dir runs/reference_library/we2_refs_chroma_bge_m3 \
  --reference-collection-name dms_reference_knowledge_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --format markdown \
  --output runs/reference_library/scene6_memory_packet_with_refs.md
```

Launch the UI:

```bash
python -m dms.cli launch-ui \
  --benchmark-dir runs/benchmark/we2_scene6_new_pipeline \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --server-port 7860
```

## Full Script Run

The full-script workflow can take a long time. Use the provided script inside `tmux`:

```bash
tmux new-session -d -s dms_full \
  'cd /path/to/DiegeticMemorySociety && bash scripts/run_full_benchmark.sh'
```

The script performs:

1. full ordered extraction over all scenes;
2. SQLite asset-store import;
3. Chroma vector index build;
4. full writing benchmark over all eligible writing targets.

Outputs are written under `runs/`, and logs are written under `logs/`.

Useful overrides:

```bash
RUN_ID=we2_full_qwen235_8002_20260531 \
PYTHON_BIN=/path/to/conda/envs/screenplay/bin/python \
SCENE_TASK_CONCURRENCY=3 \
MAX_CHUNK_UNITS=800 \
bash scripts/run_full_benchmark.sh
```

For a bounded dry pilot, set `SCENE_LIMIT` and `BENCHMARK_LIMIT`.

## Retrieval Flow

The main writing-time retrieval entry point is `build-memory-packet`.

1. `writing_intent` is decomposed into important entities and narrative-unit
   phrases.
2. Entities are resolved against the SQLite entity registry.
3. Related scene summaries are retrieved with the narrative-unit phrases.
4. For each matched entity, already revealed entity-linked episodic memories are
   retrieved. If the SQL candidate set is large, Chroma is used to rank them.
5. Already revealed `atemporal_fact` and `durable_state` memories can also enter
   as global context even when they are not linked to a matched entity.
6. One-hop canonical relations are retrieved from the screenplay-derived
   relationship layer.
7. If `--include-reference-context` is set, external reference records are
   retrieved from the standalone source-local reference DB and optional
   reference Chroma index. They are grouped into `author_reference_context`,
   `character_reference_knowledge`, `style_reference_context`, and
   `timeline_reference_claims`.

Reference visibility is conservative: `author_only` stays author-facing,
`style_only` is never character knowledge, `world_public` may enter character
knowledge, `character_private` requires a `known_to` match, and
`revealed_by_story` respects `available_from` when configured.

External source-local relationships and atomic facts are reference evidence, not
canonical KG edges. They can be retrieved into memory packets, but they are not
automatically merged into the screenplay-derived relationship graph.

## External Reference Quality Notes

The public-source test corpus under `data/reference_library/we2_web_refs_20260605`
is intentionally small and stores source-grounded notes/chunks rather than
mirrored webpages. A real 8002 smoke run on that corpus produced:

- `28` ingested chunks;
- source-local entity/relationship records plus atomic facts and entity
  properties after KG import;
- pseudo entity clusters and pseudo relationships for query-time graph insight;
- rejected or failed chunk parses remain inspectable through extraction traces
  and raw outputs.

The generated run artifacts live under `runs/` and are not committed.

## Evaluation

The writing benchmark compares generated text and the original reference scene under the same judge prompt. The reported metrics are:

- `writing_intent_consistency`
- `writing_quality`
- `memory_faithfulness`

Reference scores are used as calibration, not as a demand that generation copy the original scene.

## Verified Smoke Tests

The standardized scene-6 workflow has been run with scene-1-to-5 assets. The run completed one target scene with:

- 4 retrieved entities;
- 17 retrieved episodic memories;
- 1 durable relation;
- 5 related scene summaries;
- 2 attribute cards;
- 2 character simulations.

Recent targeted regression command:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_writing_benchmark.py \
  tests/test_gradio_app.py \
  tests/test_writing_e2e_workflow.py \
  tests/test_writing_evaluation.py \
  tests/test_config.py \
  tests/test_prompt_loader.py \
  -q
```

Result: `22 passed`.
