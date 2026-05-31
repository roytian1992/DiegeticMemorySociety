import pytest

from dms.storage import AssetStoreImportConfig, ChromaMemoryIndexConfig, build_chroma_memory_index, import_run_assets
from dms.storage.chroma_index import search_entity_memories, search_retrieval_documents
from tests.storage.test_asset_store import _write_sample_ordered_run


pytest.importorskip("chromadb")


def test_chroma_index_searches_sqlite_filtered_entity_memories(tmp_path):
    run_root = _write_sample_ordered_run(tmp_path)
    db_path = tmp_path / "assets.sqlite"
    chroma_dir = tmp_path / "chroma"
    import_run_assets(AssetStoreImportConfig(db_path=db_path, ordered_run_dir=run_root, reset=True))

    summary = build_chroma_memory_index(
        ChromaMemoryIndexConfig(
            db_path=db_path,
            persist_dir=chroma_dir,
            reset=True,
            embedding_dim=64,
        )
    )
    assert summary["document_count"] == 7

    result = search_entity_memories(
        db_path,
        persist_dir=chroma_dir,
        query="550A 脑电波 分析",
        entity_ref="550A",
        before_scene_id="scene_0002",
        top_k=3,
        embedding_dim=64,
    )
    assert result["count"] == 1
    assert result["results"][0]["memory"]["memory_id"] == "scene_0001_memory_001"

    scene_result = search_retrieval_documents(
        db_path,
        persist_dir=chroma_dir,
        query="脑电波 数字生命",
        doc_type="scene_summary",
        before_scene_id="scene_0002",
        top_k=3,
        embedding_dim=64,
    )
    assert scene_result["count"] == 1
    assert scene_result["results"][0]["sql"]["parent_scene_id"] == "scene_0001"

    fact_result = search_retrieval_documents(
        db_path,
        persist_dir=chroma_dir,
        query="脑电波 分析",
        doc_type="stated_fact",
        before_scene_id="scene_0002",
        top_k=3,
        embedding_dim=64,
    )
    assert fact_result["count"] == 1
    assert fact_result["results"][0]["sql"]["source_id"] == "scene_0001_fact_001"
