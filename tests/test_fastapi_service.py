from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dms.service.fastapi_app import (
    DMSServiceSettings,
    DEFAULT_SERVICE_CONFIG_PATH,
    _create_module_app,
    create_app,
    default_service_settings,
    references_from_config,
    service_host_port_from_config,
    settings_from_config,
    write_service_config_template,
)


def test_fastapi_memory_add_search_get_all(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    nested_dir = refs_dir / "nested"
    nested_dir.mkdir(parents=True)
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")
    (nested_dir / "timeline.md").write_text("# 时间线\n张鹏指导刘培强使用550A完成稳定性训练。\n", encoding="utf-8")
    app = create_app(
        DMSServiceSettings(
            reference_db_path=tmp_path / "reference.sqlite",
            work_dir=tmp_path / "runs",
            provider="fake",
            workers=1,
        )
    )
    client = TestClient(app)

    health = client.get("/health")
    add = client.post(
        "/add",
        json={
            "input_path": str(refs_dir),
            "metadata": {"project_id": "demo"},
            "workers": 2,
            "include_fact_properties": False,
        },
    )
    search = client.post(
        "/search",
        json={
            "query": "刘培强 张鹏",
            "include_fact_properties": False,
            "fact_binding_top_k": 2,
            "property_binding_top_k": 1,
        },
    )
    all_items = client.get("/get_all")

    assert health.status_code == 200
    assert health.json()["reference_db_path"] == str(tmp_path / "reference.sqlite")
    assert add.status_code == 200
    assert add.json()["summary"]["ingest"]["file_count"] == 2
    assert add.json()["summary"]["ingest"]["workers"] == 2
    assert add.json()["summary"]["kg"]["workers"] == 2
    assert add.json()["summary"]["facts_properties"] is None
    assert search.status_code == 200
    assert search.json()["results"]
    assert search.json()["evidence_budget"]["include_fact_properties"] is False
    assert search.json()["evidence_budget"]["fact_binding_top_k"] == 2
    assert search.json()["evidence_budget"]["property_binding_top_k"] == 1
    assert all_items.status_code == 200
    assert all_items.json()["counts"]["reference_full_docs"] == 2
    assert all_items.json()["results"][0]["type"] == "source_document"


def test_fastapi_add_rejects_conflicting_fact_property_flags(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")
    app = create_app(DMSServiceSettings(reference_db_path=tmp_path / "reference.sqlite", provider="fake"))
    client = TestClient(app)

    response = client.post(
        "/add",
        json={
            "input_path": str(refs_dir),
            "extract_fact_properties": True,
            "include_fact_properties": False,
        },
    )

    assert response.status_code == 400
    assert "conflict" in response.json()["detail"]


def test_fastapi_service_only_exposes_memory_facade(tmp_path: Path) -> None:
    app = create_app(DMSServiceSettings(reference_db_path=tmp_path / "reference.sqlite", provider="fake"))
    client = TestClient(app)

    assert client.get("/jobs").status_code == 404
    assert client.get("/references/counts").status_code == 404
    assert client.post("/context/search", json={}).status_code == 404


def test_fastapi_service_settings_can_be_loaded_from_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "local_config.yaml"
    config_path.write_text(
        f"""
llm:
  provider: openai
  model_name: fake-model
  api_key: token
  base_url: http://127.0.0.1:8001
embedding:
  provider: hash
  dimensions: 64
external_references:
  provider: fake
  model_section: llm
  embedding_section: embedding
  db_path: {tmp_path / "refs.sqlite"}
  work_dir: {tmp_path / "work"}
  collection_name: configured_collection
  workers: 3
  extract_fact_properties: false
  entity_disambiguation: false
service:
  host: 0.0.0.0
  port: 8765
""",
        encoding="utf-8",
    )

    settings = settings_from_config(config_path)
    host, port = service_host_port_from_config(config_path)
    app = create_app(settings)
    client = TestClient(app)

    assert settings.reference_db_path == tmp_path / "refs.sqlite"
    assert settings.work_dir == tmp_path / "work"
    assert settings.reference_collection_name == "configured_collection"
    assert settings.workers == 3
    assert settings.extract_fact_properties is False
    assert settings.entity_disambiguation is False
    assert settings.provider == "fake"
    assert host == "0.0.0.0"
    assert port == 8765
    assert client.get("/health").json()["reference_db_path"] == str(tmp_path / "refs.sqlite")


def test_external_reference_facade_can_be_built_from_yaml_config(tmp_path: Path) -> None:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    (refs_dir / "profiles.md").write_text("# 人物\n刘培强第一次接触550A训练。张鹏指导刘培强。\n", encoding="utf-8")
    config_path = tmp_path / "local_config.yaml"
    config_path.write_text(
        f"""
llm:
  provider: openai
  model_name: fake-model
  api_key: token
  base_url: http://127.0.0.1:8001
embedding:
  provider: hash
  dimensions: 64
external_references:
  provider: fake
  model_section: llm
  embedding_section: embedding
  db_path: {tmp_path / "refs.sqlite"}
  work_dir: {tmp_path / "work"}
  workers: 2
  extract_fact_properties: false
service:
  host: 127.0.0.1
  port: 8000
""",
        encoding="utf-8",
    )
    refs = references_from_config(config_path)

    add_result = refs.add(refs_dir)

    assert refs.config.db_path == tmp_path / "refs.sqlite"
    assert refs.config.work_dir == tmp_path / "work"
    assert refs.config.workers == 2
    assert refs.config.extract_fact_properties is False
    assert add_result["summary"]["facts_properties"] is None


def test_fastapi_service_can_write_starter_config(tmp_path: Path) -> None:
    config_path = tmp_path / "generated_config.yaml"

    result = write_service_config_template(config_path)

    assert result == {"config_path": str(config_path), "status": "created"}
    text = config_path.read_text(encoding="utf-8")
    assert "external_references:" in text
    assert "service:" in text
    assert "llm:" in text


def test_default_config_is_external_reference_service_ready() -> None:
    settings = settings_from_config(DEFAULT_SERVICE_CONFIG_PATH)
    host, port = service_host_port_from_config(DEFAULT_SERVICE_CONFIG_PATH)

    assert settings.provider == "openai"
    assert settings.model_section == "llm"
    assert settings.embedding_section == "embedding"
    assert settings.reference_db_path == Path("runs/reference_library/we2_refs.sqlite")
    assert settings.work_dir == Path("runs/reference_library/service_work")
    assert settings.reference_collection_name == "dms_reference_knowledge"
    assert settings.workers == 8
    assert settings.extract_fact_properties is True
    assert settings.entity_disambiguation is True
    assert host == "127.0.0.1"
    assert port == 8000


def test_default_service_settings_requires_yaml_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="--init-config"):
        default_service_settings()


def test_module_app_import_path_does_not_require_yaml_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    app = _create_module_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "config_missing"
