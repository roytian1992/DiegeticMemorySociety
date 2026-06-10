from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dms.service.fastapi_app import DMSServiceSettings, create_app


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
