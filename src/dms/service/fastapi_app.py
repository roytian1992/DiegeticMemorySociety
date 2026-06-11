from __future__ import annotations

import argparse
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


DEFAULT_SERVICE_CONFIG_PATH = Path("configs/default.yaml")


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
    settings = settings or default_service_settings()
    app = FastAPI(title="Diegetic Memory Society Memory API", version="0.1.0")

    def references() -> ExternalReferences:
        return references_from_settings(settings)

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


def default_service_settings() -> DMSServiceSettings:
    default_config = Path("configs/local_config.yaml")
    if default_config.is_file():
        return settings_from_config(default_config)
    if DEFAULT_SERVICE_CONFIG_PATH.is_file():
        return settings_from_config(DEFAULT_SERVICE_CONFIG_PATH)
    raise FileNotFoundError(
        "No DMS service config found. Run `dms-service --init-config configs/local_config.yaml` "
        "or pass `--config path/to/config.yaml`."
    )


def references_from_config(path: str | Path = "configs/local_config.yaml") -> ExternalReferences:
    return references_from_settings(settings_from_config(path))


def references_from_settings(settings: DMSServiceSettings) -> ExternalReferences:
    return ExternalReferences(_reference_config(settings), llm_client=_llm_client(settings))


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


def write_service_config_template(
    path: str | Path,
    *,
    overwrite: bool = False,
    template_path: str | Path = DEFAULT_SERVICE_CONFIG_PATH,
) -> dict[str, str]:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {output_path}")
    source_path = Path(template_path)
    if source_path.is_file():
        template = source_path.read_text(encoding="utf-8")
    else:
        template = _fallback_service_config_template()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template, encoding="utf-8")
    return {"config_path": str(output_path), "status": "created"}


def _fallback_service_config_template() -> str:
    return """\
llm:
  provider: openai
  model_name: YOUR_CHAT_MODEL
  api_key: YOUR_LOCAL_API_KEY
  base_url: http://127.0.0.1:8001
  max_tokens: 3072
  temperature: 0
  timeout_seconds: 240
  enable_thinking: false
  include_chat_template_kwargs: true

embedding:
  provider: openai
  model_name: YOUR_EMBEDDING_MODEL
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
    parser.add_argument("--config", type=Path, default=None, help="YAML config containing llm, embedding, external_references, and service sections. Defaults to configs/local_config.yaml when present, otherwise configs/default.yaml.")
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
    config_path = args.config or (default_config if default_config.is_file() else (DEFAULT_SERVICE_CONFIG_PATH if DEFAULT_SERVICE_CONFIG_PATH.is_file() else None))
    if config_path is None:
        parser.error("No config found. Run `dms-service --init-config configs/local_config.yaml` or pass `--config`.")
    settings = settings_from_config(config_path)
    config_host, config_port = service_host_port_from_config(config_path)
    host = args.host or config_host
    port = args.port if args.port is not None else config_port

    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, reload=args.reload)


def _create_module_app() -> FastAPI | None:
    if _FASTAPI_IMPORT_ERROR is not None:
        return None
    try:
        return create_app()
    except FileNotFoundError as exc:
        detail = str(exc)
        missing_config_app = FastAPI(title="Diegetic Memory Society Memory API", version="0.1.0")

        @missing_config_app.get("/health")
        def health() -> dict[str, Any]:
            return {"status": "config_missing", "detail": detail}

        return missing_config_app


app = _create_module_app()


if __name__ == "__main__":
    main()
