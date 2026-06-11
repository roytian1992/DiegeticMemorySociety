# External References 使用说明

External References 提供一组 memory-style 接口：
`add`、`search`、`get_all`。应用侧不需要直接关心底层的 ingest、KG
抽取、fact/property 抽取、SQLite import、Chroma indexing 等步骤；这些都由
`add` 内部串起来。

使用方式概览：

- 模型、embedding、API 地址、token 都从本地配置文件读取。
- 文档路径、DB 路径、work dir、并发数这类运行参数由配置文件或请求参数提供。
- FastAPI 服务提供 `add/search/get_all` 三个主要接口。
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

`configs/local_config.yaml` 用于保存本机模型服务和 embedding 服务配置，已在 `.gitignore` 中。
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

Python 侧通常先读取本地配置，再创建 `ExternalReferences` 实例。
`load_local_config` 负责读取 YAML，`build_openai_client_from_config` 负责创建
LLM client，`embedding_kwargs_from_config` 负责生成 embedding 参数。

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

服务直接读取 YAML 配置。第一次使用可以生成模板：

```bash
dms-service --init-config configs/local_config.yaml
```

然后编辑 `configs/local_config.yaml` 中的 `llm`、`embedding` 和
`external_references` section。服务相关配置放在 `service` section：

```yaml
external_references:
  provider: openai
  model_section: llm
  embedding_section: embedding
  db_path: runs/reference_library/we2_refs.sqlite
  work_dir: runs/reference_library/service_work
  chroma_dir:
  collection_name: dms_reference_knowledge
  workers: 8
  max_chunk_chars: 2400
  max_retries: 1
  extract_fact_properties: true
  reference_fact_min_entity_degree: 2
  reference_fact_max_evidence_chunks_per_job: 12
  entity_disambiguation: true
  disambiguation_lexical_threshold: 0.88
  auto_index: false
  reset_on_add: true

service:
  host: 127.0.0.1
  port: 8000
```

启动服务：

```bash
dms-service --config configs/local_config.yaml
```

如果当前目录存在 `configs/local_config.yaml`，也可以直接运行 `dms-service`。
`--host` 和 `--port` 只作为临时覆盖项使用。

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

## CLI 调试流程

Python 接口和 FastAPI 服务适合应用集成；CLI 适合检查中间产物、定位抽取或导入问题。

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

## 常用配置组合

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
