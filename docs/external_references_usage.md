# External References 使用说明

External References 对外只暴露一层 memory-style facade：
`add`、`search`、`get_all`。应用侧不需要直接关心底层的 ingest、KG
抽取、fact/property 抽取、SQLite import、Chroma indexing 等步骤；这些都由
`add` 内部串起来。

核心原则：

- 模型、embedding、API 地址、token 都从本地配置文件读取。
- 文档路径、DB 路径、work dir、并发数这类运行参数由调用方传入或用环境变量配置。
- FastAPI 服务只提供 `add/search/get_all`，不暴露内部 pipeline 或 context kernel。
- fact/property 抽取是 ingest 阶段开关；检索阶段可单独控制是否返回 facts/properties，并按 query 相似度绑定 top-k。

## 安装依赖

完整开发/服务依赖：

```bash
pip install -r requirements.txt
```

只安装基础包和 FastAPI 服务：

```bash
pip install -e .
pip install -e '.[service]'
```

## 本地配置文件

`configs/local_config.yaml` 是本地私有配置，已在 `.gitignore` 中，不应提交。
可以参考 `configs/default.yaml` 创建，并补充 LLM 和 embedding 配置：

```yaml
llm:
  provider: openai
  model_name: Qwen3.5-397B-A17B-FP8
  api_key: YOUR_LOCAL_API_KEY
  base_url: http://127.0.0.1:8001
  max_tokens: 3072
  temperature: 0
  timeout_seconds: 240
  enable_thinking: false
  include_chat_template_kwargs: true

embedding:
  provider: openai
  model_name: bge-m3
  api_key: YOUR_LOCAL_API_KEY
  base_url: http://127.0.0.1:8081
  max_tokens: 8192
  dimensions: 1024
  timeout_seconds: 60
```

如果使用不需要 `chat_template_kwargs` 的 OpenAI-compatible reasoning 服务，可以在
对应 section 中设置：

```yaml
thinking:
  type: disabled
include_chat_template_kwargs: false
```

## Python 调用

产品代码里不要手写 model/base_url/api_key。推荐用
`load_local_config`、`build_openai_client_from_config` 和
`embedding_kwargs_from_config` 从配置文件构造 facade。

```python
from pathlib import Path

from dms.config import (
    build_openai_client_from_config,
    embedding_kwargs_from_config,
    load_local_config,
)
from dms.external_references import ExternalReferences, ExternalReferencesConfig


def build_external_refs(
    *,
    config_path: str | Path = "configs/local_config.yaml",
    db_path: str | Path = "runs/reference_library/we2_refs.sqlite",
    work_dir: str | Path = "runs/reference_library/we2_refs_work",
    chroma_dir: str | Path | None = None,
    workers: int = 8,
    extract_fact_properties: bool = True,
    entity_disambiguation: bool = True,
) -> ExternalReferences:
    config = load_local_config(config_path)
    llm_client = build_openai_client_from_config(config, "llm")
    embedding_kwargs = embedding_kwargs_from_config(config, "embedding")

    return ExternalReferences(
        ExternalReferencesConfig(
            db_path=Path(db_path),
            work_dir=Path(work_dir),
            chroma_dir=Path(chroma_dir) if chroma_dir else None,
            workers=workers,
            extract_fact_properties=extract_fact_properties,
            entity_disambiguation=entity_disambiguation,
            auto_index=bool(chroma_dir),
            **embedding_kwargs,
        ),
        llm_client=llm_client,
    )


refs = build_external_refs(
    db_path="runs/reference_library/we2_refs.sqlite",
    work_dir="runs/reference_library/we2_refs_work",
    chroma_dir=None,
    workers=8,
    extract_fact_properties=False,
)

add_result = refs.add(
    "data/reference_library/we2_web_refs_20260605",
    metadata={"project_id": "we2"},
    workers=8,
)

search_result = refs.search(
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

`add` 的关键参数：

- `input_path`：文件或目录；目录会递归处理支持的文档类型。
- `workers`：覆盖默认并发数，会传到 ingest、KG 抽取、fact/property 抽取阶段。
- `extract_fact_properties`：ingest 阶段是否抽取 facts/properties。大语料或只需要 KG 检索时可以关掉。
- `entity_disambiguation`：是否在实体类型内做消歧和 canonical 聚合。
- `disambiguation_lexical_threshold`：实体消歧的词面相似度阈值。

`search` 的关键参数：

- `evidence_budget`：可用 `compact`、`standard`、`deep`，也可以传 dict 覆盖细项。
- `include_fact_properties`：检索阶段是否返回 facts/properties。
- `fact_binding_top_k`：对命中的实体/cluster，只绑定与 query 最相似的 top-k facts。
- `property_binding_top_k`：对命中的实体/cluster，只绑定与 query 最相似的 top-k properties。
- `filters`：可按 `source_path/source_paths`、`source_doc_id/source_doc_ids`、`source_scope_id/source_scope_ids` 限定来源。

## FastAPI 服务

服务读取环境变量构造 `ExternalReferencesConfig`。模型和 embedding 从
`DMS_MODEL_CONFIG` 指向的 YAML 配置读取。

```bash
export DMS_PROVIDER=openai
export DMS_MODEL_CONFIG=configs/local_config.yaml
export DMS_MODEL_SECTION=llm
export DMS_EMBEDDING_SECTION=embedding

export DMS_REFERENCE_DB=runs/reference_library/we2_refs.sqlite
export DMS_REFERENCE_WORK_DIR=runs/reference_library/service_work
export DMS_REFERENCE_CHROMA_DIR=
export DMS_REFERENCE_COLLECTION=dms_reference_knowledge

export DMS_WORKERS=8
export DMS_EXTRACT_FACT_PROPERTIES=true
export DMS_ENTITY_DISAMBIGUATION=true
export DMS_DISAMBIGUATION_LEXICAL_THRESHOLD=0.88

dms-service --host 127.0.0.1 --port 8000
```

本地 smoke test 可以不用真实 LLM：

```bash
export DMS_PROVIDER=fake
export DMS_REFERENCE_DB=runs/reference_library/smoke_refs.sqlite
dms-service --host 127.0.0.1 --port 8000
```

接口：

- `POST /add`
- `POST /search`
- `GET /get_all`
- `GET /health`

`/add` 示例：

```bash
curl -X POST http://127.0.0.1:8000/add \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "data/reference_library/we2_web_refs_20260605",
    "metadata": {"project_id": "we2"},
    "workers": 8,
    "include_fact_properties": false,
    "entity_disambiguation": true
  }'
```

说明：HTTP 层接受 `include_fact_properties` 作为 ingest 阶段
`extract_fact_properties` 的别名。两者不要同时传冲突值。

`/search` 示例：

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "张鹏和刘培强的关系",
    "evidence_budget": "standard",
    "include_fact_properties": true,
    "fact_binding_top_k": 4,
    "property_binding_top_k": 3,
    "filters": {
      "source_path": "profiles.md"
    }
  }'
```

`/get_all` 示例：

```bash
curl http://127.0.0.1:8000/get_all
```

## CLI 调试 pipeline

正常产品调用建议用 Python facade 或 FastAPI。CLI 主要用于检查中间产物。

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

可选 Chroma index：

```bash
python -m dms.cli build-reference-index \
  runs/reference_library/we2_refs.sqlite \
  --persist-dir runs/reference_library/we2_refs_chroma_bge_m3 \
  --collection-name dms_reference_knowledge_bge_m3 \
  --model-config configs/local_config.yaml \
  --embedding-section embedding \
  --overwrite
```

CLI 查询：

```bash
python -m dms.cli query-reference-knowledge \
  runs/reference_library/we2_refs.sqlite \
  --query "张鹏和刘培强的关系" \
  --source-path profiles.md \
  --include-fact-properties \
  --fact-binding-top-k 4 \
  --property-binding-top-k 3
```

## 返回结构

`search` 返回的主要字段：

- `results`：memory-style 的紧凑命中结果。
- `evidence_packet`：给回答或写作上下文使用的证据包，包含实体、关系、facts、properties、chunks、graph insights、citations 和 answer context。
- `query_plan`：low-level/high-level keyword plan 和多路检索 passes。
- `raw_retrieval`：每个检索 pass 的摘要。
- `raw`：完整结构化检索结果。

`get_all` 返回：

- `results`：已导入的 source documents。
- `counts`：reference DB 内各资产表的数量。

## 推荐默认值

小规模调试：

```text
workers=2
extract_fact_properties=false
include_fact_properties=false
```

正常服务：

```text
workers=8
extract_fact_properties=true
entity_disambiguation=true
fact_binding_top_k=4
property_binding_top_k=3
```

大语料首次导入：

```text
workers=8 或更高
extract_fact_properties=false
```

先把 KG 和 source-local entity/relationship 跑通，再按需要开启
fact/property 抽取；否则 facts 数量容易膨胀，调试时也更难定位问题。
