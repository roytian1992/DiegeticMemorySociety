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
            raise HTTPException(status_code=400, detail="DMS_MODEL_CONFIG is required when DMS_PROVIDER=openai")
        return build_openai_client_from_config(load_local_config(settings.model_config_path), settings.model_section)
    raise HTTPException(status_code=400, detail=f"Unsupported DMS_PROVIDER: {settings.provider}")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DMS memory FastAPI service.")
    parser.add_argument("--host", default=os.environ.get("DMS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DMS_PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("dms.service.fastapi_app:app", host=args.host, port=args.port, reload=args.reload)


app = create_app() if _FASTAPI_IMPORT_ERROR is None else None
