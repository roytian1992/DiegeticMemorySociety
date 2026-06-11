from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.config import build_openai_client_from_config, embedding_kwargs_from_config, load_local_config
from dms.external_references import ExternalReferences, ExternalReferencesConfig
from dms.llm import FakeReferenceKGClient, LLMClient

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    try:
        from pydantic import ConfigDict
    except ImportError:  # pydantic v1
        ConfigDict = None  # type: ignore[assignment]
except ImportError as exc:  # pragma: no cover - exercised when service extra is not installed.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment]

    def Field(default: Any = None, *, default_factory: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        if default_factory is not None:
            return default_factory()
        return default

    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


DEFAULT_SERVICE_CONFIG_TEMPLATE = """\
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
"""


@dataclass(frozen=True)
class DMSServiceSettings:
    reference_db_path: Path
    work_dir: Path | None = None
    reference_chroma_dir: Path | None = None
    reference_collection_name: str = "dms_reference_knowledge"
    model_config_path: Path | None = None
    model_section: str = "llm"
    embedding_section: str = "embedding"
    provider: str = "fake"
    max_chunk_chars: int = 2400
    max_retries: int = 1
    workers: int = 4
    extract_fact_properties: bool = True
    reference_fact_min_entity_degree: int = 2
    reference_fact_max_evidence_chunks_per_job: int = 12
    entity_disambiguation: bool = True
    disambiguation_lexical_threshold: float = 0.88
    auto_index: bool = False
    reset_on_add: bool = True


class _Model(BaseModel):  # type: ignore[misc]
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


class AddRequest(_Model):
    input_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    infer: bool = True
    reset: bool | None = None
    extract_fact_properties: bool | None = None
    include_fact_properties: bool | None = None
    workers: int | None = None
    entity_disambiguation: bool | None = None
    disambiguation_lexical_threshold: float | None = None


class SearchRequest(_Model):
    query: str
    evidence_budget: str | dict[str, Any] | None = "standard"
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = None
    include_fact_properties: bool | None = None
    fact_binding_top_k: int | None = None
    property_binding_top_k: int | None = None


def create_app(settings: DMSServiceSettings | None = None) -> FastAPI:
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError("FastAPI service dependencies are not installed. Install with `pip install -e .[service]`.") from _FASTAPI_IMPORT_ERROR
    settings = settings or settings_from_env()
    app = FastAPI(title="Diegetic Memory Society Memory API", version="0.1.0")

    def references() -> ExternalReferences:
        return ExternalReferences(_reference_config(settings), llm_client=_llm_client(settings))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "reference_db_path": str(settings.reference_db_path),
        }

    @app.post("/add")
    def add(request: AddRequest) -> dict[str, Any]:
        extract_fact_properties = _resolve_fact_property_add_flag(request)
        try:
            return references().add(
                request.input_path,
                metadata=request.metadata,
                infer=request.infer,
                reset=request.reset,
                extract_fact_properties=extract_fact_properties,
                workers=request.workers,
                entity_disambiguation=request.entity_disambiguation,
                disambiguation_lexical_threshold=request.disambiguation_lexical_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            return references().search(
                request.query,
                evidence_budget=_search_evidence_budget(request),
                filters=request.filters,
                limit=request.limit,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/get_all")
    def get_all() -> dict[str, Any]:
        return references().get_all()

    return app


def settings_from_env() -> DMSServiceSettings:
    reference_db = _env_path("DMS_REFERENCE_DB", Path("data/reference_library/reference.sqlite"))
    model_config_raw = os.environ.get("DMS_MODEL_CONFIG")
    return DMSServiceSettings(
        reference_db_path=reference_db,
        work_dir=_optional_env_path("DMS_REFERENCE_WORK_DIR"),
        reference_chroma_dir=_optional_env_path("DMS_REFERENCE_CHROMA_DIR"),
        reference_collection_name=os.environ.get("DMS_REFERENCE_COLLECTION", "dms_reference_knowledge"),
        model_config_path=Path(model_config_raw) if model_config_raw else None,
        model_section=os.environ.get("DMS_MODEL_SECTION", "llm"),
        embedding_section=os.environ.get("DMS_EMBEDDING_SECTION", "embedding"),
        provider=os.environ.get("DMS_PROVIDER", "fake"),
        max_chunk_chars=int(os.environ.get("DMS_MAX_CHUNK_CHARS", "2400")),
        max_retries=int(os.environ.get("DMS_MAX_RETRIES", "1")),
        workers=int(os.environ.get("DMS_WORKERS", "4")),
        extract_fact_properties=_env_bool("DMS_EXTRACT_FACT_PROPERTIES", True),
        reference_fact_min_entity_degree=int(os.environ.get("DMS_REFERENCE_FACT_MIN_ENTITY_DEGREE", "2")),
        reference_fact_max_evidence_chunks_per_job=int(os.environ.get("DMS_REFERENCE_FACT_MAX_EVIDENCE_CHUNKS_PER_JOB", "12")),
        entity_disambiguation=_env_bool("DMS_ENTITY_DISAMBIGUATION", True),
        disambiguation_lexical_threshold=float(os.environ.get("DMS_DISAMBIGUATION_LEXICAL_THRESHOLD", "0.88")),
        auto_index=_env_bool("DMS_AUTO_INDEX", False),
        reset_on_add=_env_bool("DMS_RESET_ON_ADD", True),
    )


def settings_from_config(path: str | Path = "configs/local_config.yaml") -> DMSServiceSettings:
    config_path = Path(path)
    config = load_local_config(config_path)
    block = _external_references_block(config)
    model_section = str(block.get("model_section") or "llm")
    embedding_section = str(block.get("embedding_section") or "embedding")
    provider = str(
        block.get("provider")
        or block.get("llm_provider")
        or (config.get(model_section) if isinstance(config.get(model_section), dict) else {}).get("provider")
        or "openai"
    ).strip().lower()
    return DMSServiceSettings(
        reference_db_path=_config_path(block.get("db_path"), Path("data/reference_library/reference.sqlite")),
        work_dir=_optional_config_path(block.get("work_dir")),
        reference_chroma_dir=_optional_config_path(block.get("chroma_dir")),
        reference_collection_name=str(block.get("collection_name") or block.get("reference_collection_name") or "dms_reference_knowledge"),
        model_config_path=config_path,
        model_section=model_section,
        embedding_section=embedding_section,
        provider=provider,
        max_chunk_chars=_config_int(block.get("max_chunk_chars"), 2400),
        max_retries=_config_int(block.get("max_retries"), 1),
        workers=_config_int(block.get("workers"), 4),
        extract_fact_properties=_config_bool(block.get("extract_fact_properties"), True),
        reference_fact_min_entity_degree=_config_int(block.get("reference_fact_min_entity_degree"), 2),
        reference_fact_max_evidence_chunks_per_job=_config_int(block.get("reference_fact_max_evidence_chunks_per_job"), 12),
        entity_disambiguation=_config_bool(block.get("entity_disambiguation"), True),
        disambiguation_lexical_threshold=_config_float(block.get("disambiguation_lexical_threshold"), 0.88),
        auto_index=_config_bool(block.get("auto_index"), False),
        reset_on_add=_config_bool(block.get("reset_on_add"), True),
    )


def service_host_port_from_config(path: str | Path = "configs/local_config.yaml") -> tuple[str, int]:
    config = load_local_config(path)
    service = config.get("service")
    if not isinstance(service, dict):
        service = {}
    return str(service.get("host") or "127.0.0.1"), _config_int(service.get("port"), 8000)


def write_service_config_template(path: str | Path, *, overwrite: bool = False) -> dict[str, str]:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(DEFAULT_SERVICE_CONFIG_TEMPLATE, encoding="utf-8")
    return {"config_path": str(output_path), "status": "created"}


def _reference_config(settings: DMSServiceSettings) -> ExternalReferencesConfig:
    embedding_kwargs = _embedding_kwargs(settings)
    return ExternalReferencesConfig(
        db_path=settings.reference_db_path,
        work_dir=settings.work_dir,
        chroma_dir=settings.reference_chroma_dir,
        collection_name=settings.reference_collection_name,
        max_chunk_chars=settings.max_chunk_chars,
        max_retries=settings.max_retries,
        workers=settings.workers,
        extract_fact_properties=settings.extract_fact_properties,
        reference_fact_min_entity_degree=settings.reference_fact_min_entity_degree,
        reference_fact_max_evidence_chunks_per_job=settings.reference_fact_max_evidence_chunks_per_job,
        entity_disambiguation=settings.entity_disambiguation,
        disambiguation_lexical_threshold=settings.disambiguation_lexical_threshold,
        auto_index=settings.auto_index,
        reset_on_add=settings.reset_on_add,
        **embedding_kwargs,
    )


def _llm_client(settings: DMSServiceSettings) -> LLMClient:
    if settings.provider == "fake":
        return FakeReferenceKGClient()
    if settings.provider == "openai":
        if settings.model_config_path is None:
            raise HTTPException(status_code=400, detail="model_config_path is required when provider=openai")
        return build_openai_client_from_config(load_local_config(settings.model_config_path), settings.model_section)
    raise HTTPException(status_code=400, detail=f"Unsupported external reference provider: {settings.provider}")


def _embedding_kwargs(settings: DMSServiceSettings) -> dict[str, Any]:
    if settings.model_config_path is None:
        return {}
    try:
        return embedding_kwargs_from_config(load_local_config(settings.model_config_path), settings.embedding_section)
    except Exception:
        return {}


def _resolve_fact_property_add_flag(request: AddRequest) -> bool | None:
    if request.include_fact_properties is None:
        return request.extract_fact_properties
    if request.extract_fact_properties is not None and request.extract_fact_properties != request.include_fact_properties:
        raise HTTPException(
            status_code=400,
            detail="extract_fact_properties and include_fact_properties conflict; set only one value.",
        )
    return request.include_fact_properties


def _search_evidence_budget(request: SearchRequest) -> str | dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if request.include_fact_properties is not None:
        overrides["include_fact_properties"] = request.include_fact_properties
    if request.fact_binding_top_k is not None:
        overrides["fact_binding_top_k"] = request.fact_binding_top_k
    if request.property_binding_top_k is not None:
        overrides["property_binding_top_k"] = request.property_binding_top_k
    if not overrides:
        return request.evidence_budget
    if isinstance(request.evidence_budget, dict):
        return {**request.evidence_budget, **overrides}
    return {"profile": request.evidence_budget or "standard", **overrides}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _external_references_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("external_references")
    if block is None:
        block = config.get("reference_service")
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError("Config section external_references must be a YAML object")
    return block


def _config_path(value: Any, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    return Path(str(value))


def _optional_config_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _config_int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _config_float(value: Any, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DMS memory FastAPI service.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config containing llm, embedding, external_references, and service sections. Defaults to configs/local_config.yaml when present.")
    parser.add_argument("--init-config", type=Path, default=None, help="Write a starter YAML config and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Allow --init-config to overwrite an existing file.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    if args.init_config is not None:
        try:
            result = write_service_config_template(args.init_config, overwrite=args.overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
        print(f"Wrote DMS service config: {result['config_path']}")
        return

    default_config = Path("configs/local_config.yaml")
    config_path = args.config or (default_config if default_config.is_file() else None)
    if config_path is not None:
        settings = settings_from_config(config_path)
        config_host, config_port = service_host_port_from_config(config_path)
        host = args.host or config_host
        port = args.port if args.port is not None else config_port
    else:
        settings = settings_from_env()
        host = args.host or os.environ.get("DMS_HOST", "127.0.0.1")
        port = args.port if args.port is not None else int(os.environ.get("DMS_PORT", "8000"))

    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, reload=args.reload)


app = create_app() if _FASTAPI_IMPORT_ERROR is None else None


if __name__ == "__main__":
    main()
