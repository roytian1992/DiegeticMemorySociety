# External References Usage

External references are exposed through a small memory-style facade:
`add`, `search`, and `get_all`. Internally, `add` builds source-aware
LightRAG-style assets: documents, chunks, extracted entities, relationships,
optional facts/properties, source-local graph records, and optional vector
documents.

## Install

```bash
pip install -r requirements.txt
```

For a smaller install, use the package metadata:

```bash
pip install -e .
pip install -e '.[service]'  # only needed for FastAPI
```

## Python Facade

```python
from dms.external_references import ExternalReferences, ExternalReferencesConfig
from dms.llm import OpenAIChatClient

refs = ExternalReferences(
    ExternalReferencesConfig(
        db_path="runs/reference_library/we2_refs.sqlite",
        work_dir="runs/reference_library/we2_refs_work",
        workers=8,
        extract_fact_properties=True,
        entity_disambiguation=True,
        auto_index=False,
    ),
    llm_client=OpenAIChatClient(
        base_url="http://127.0.0.1:8002/v1",
        api_key="token-abc123",
        model="Qwen3-235B-FP8",
        include_chat_template_kwargs=True,
        enable_thinking=False,
    ),
)

add_result = refs.add(
    "data/reference_library/we2_web_refs_20260605",
    metadata={"project_id": "we2"},
    workers=8,
    extract_fact_properties=False,
)

hits = refs.search(
    "张鹏和刘培强的关系",
    evidence_budget={
        "profile": "standard",
        "include_fact_properties": True,
        "fact_binding_top_k": 4,
        "property_binding_top_k": 3,
    },
)

all_docs = refs.get_all()
```

Important `add` options:

- `input_path`: a file or directory. Directories are processed recursively for supported files.
- `workers`: parallel worker count for ingest, KG extraction, and fact/property extraction.
- `extract_fact_properties`: ingest-stage switch for facts/properties. Set `False` when the corpus is large or only KG retrieval is needed.
- `entity_disambiguation`: enables type-aware entity canonicalization during facts/properties selection and import.
- `disambiguation_lexical_threshold`: lexical similarity threshold for by-type disambiguation.

Important `search` options:

- `evidence_budget`: built-in profiles are `compact`, `standard`, and `deep`; a dict can override budgets.
- `include_fact_properties`: query-stage switch for returning facts/properties.
- `fact_binding_top_k`: for each matched entity/cluster, bind only the top-k facts most similar to the query.
- `property_binding_top_k`: same mechanism for entity properties.
- `filters`: supports `source_path`, `source_paths`, `source_doc_id`, `source_doc_ids`, `source_scope_id`, and `source_scope_ids`.

## FastAPI Facade

Start the service:

```bash
export DMS_REFERENCE_DB=runs/reference_library/we2_refs.sqlite
export DMS_REFERENCE_WORK_DIR=runs/reference_library/service_work
export DMS_PROVIDER=openai
export DMS_MODEL_CONFIG=configs/local_config.yaml
export DMS_WORKERS=8

dms-service --host 127.0.0.1 --port 8000
```

For local smoke tests without a real LLM:

```bash
export DMS_PROVIDER=fake
dms-service --host 127.0.0.1 --port 8000
```

Endpoints:

- `POST /add`
- `POST /search`
- `GET /get_all`
- `GET /health`

Examples:

```bash
curl -X POST http://127.0.0.1:8000/add \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "data/reference_library/we2_web_refs_20260605",
    "metadata": {"project_id": "we2"},
    "workers": 8,
    "include_fact_properties": false
  }'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "张鹏和刘培强的关系",
    "include_fact_properties": true,
    "fact_binding_top_k": 4,
    "property_binding_top_k": 3
  }'

curl http://127.0.0.1:8000/get_all
```

`include_fact_properties` in `/add` is accepted as an HTTP-facing alias for
`extract_fact_properties`.

## CLI Pipeline

The facade is the normal application entry point. The CLI commands are useful
for debugging and inspecting intermediate artifacts.

```bash
python -m dms.cli ingest-reference-library \
  data/reference_library/we2_web_refs_20260605 \
  --output-dir runs/reference_library/we2_refs_ingest \
  --workers 8 \
  --overwrite

python -m dms.cli extract-reference-kg \
  runs/reference_library/we2_refs_ingest \
  --output-dir runs/reference_library/we2_refs_kg \
  --no-dry-run \
  --workers 8 \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --overwrite

python -m dms.cli extract-reference-facts-properties \
  runs/reference_library/we2_refs_ingest \
  --kg-dir runs/reference_library/we2_refs_kg \
  --output-dir runs/reference_library/we2_refs_facts \
  --no-dry-run \
  --workers 8 \
  --entity-disambiguation \
  --model-config configs/local_config.yaml \
  --model-section llm \
  --overwrite

python -m dms.cli import-reference-knowledge \
  --library-dir runs/reference_library/we2_refs_ingest \
  --kg-dir runs/reference_library/we2_refs_kg \
  --facts-dir runs/reference_library/we2_refs_facts \
  --output-db runs/reference_library/we2_refs.sqlite \
  --entity-disambiguation \
  --overwrite
```

Optional Chroma index:

```bash
python -m dms.cli build-reference-index \
  runs/reference_library/we2_refs.sqlite \
  --persist-dir runs/reference_library/we2_refs_chroma_bge_m3 \
  --collection-name dms_reference_knowledge_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --overwrite
```

Query from CLI:

```bash
python -m dms.cli query-reference-knowledge \
  runs/reference_library/we2_refs.sqlite \
  --query "张鹏和刘培强的关系" \
  --source-path profiles.md \
  --include-fact-properties \
  --fact-binding-top-k 4 \
  --property-binding-top-k 3
```

## Outputs

`search` returns:

- `results`: memory-style compact hits.
- `evidence_packet`: source-grounded entities, relationships, facts, properties, chunks, graph insights, citations, and answer context.
- `query_plan`: low-level/high-level keyword plan and retrieval passes.
- `raw_retrieval`: per-pass retrieval summary.
- `raw`: full structured retrieval output.

`get_all` returns source documents and table counts for the reference DB.
