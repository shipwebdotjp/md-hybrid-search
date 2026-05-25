import pytest
import json
from pathlib import Path
from src.index import SearchIndex, DirectorySource
from src.exceptions import ConfigMismatchError
from typing import List

class MockEmbedder:
    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.embedding_dim = 128
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 128

@pytest.fixture
def index_setup(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("content", encoding="utf-8")

    params = {
        "collection_name": "test-maint",
        "sources": [DirectorySource(str(vault))],
        "sqlite_path": str(tmp_path / "test.sqlite"),
        "chroma_path": str(tmp_path / "chroma"),
        "embedder": MockEmbedder(),
        "chunk_size": 100,
        "chunk_overlap": 10
    }
    return params, vault

def test_rebuild_and_clear(index_setup):
    params, vault = index_setup
    index = SearchIndex(**params)

    # Sync first
    index.sync()
    assert index.db.get_collection(index.collection_name) is not None
    assert len(index.db.get_files_by_collection(index.collection_name)) == 1

    # Clear
    index.clear()
    assert index.db.get_collection(index.collection_name) is None
    assert len(index.db.get_files_by_collection(index.collection_name)) == 0
    # Chroma should be empty too (verified by search not finding anything)
    assert len(index.search("content", mode="similarity")) == 0

    # Rebuild
    index.rebuild()
    assert index.db.get_collection(index.collection_name) is not None
    assert len(index.db.get_files_by_collection(index.collection_name)) == 1
    assert len(index.search("content", mode="similarity")) == 1

def test_config_mismatch_chunk_size(index_setup):
    params, vault = index_setup
    index = SearchIndex(**params)
    index.sync()

    # Change chunk_size
    params["chunk_size"] = 200
    index2 = SearchIndex(**params)

    with pytest.raises(ConfigMismatchError, match="chunk_size"):
        index2.sync()

    with pytest.raises(ConfigMismatchError, match="chunk_size"):
        index2.search("query")

def test_config_mismatch_embedder(index_setup):
    params, vault = index_setup
    index = SearchIndex(**params)
    index.sync()

    # Change embedder model name
    params["embedder"] = MockEmbedder(model_name="new-model")
    index2 = SearchIndex(**params)

    with pytest.raises(ConfigMismatchError, match="embedder_fingerprint"):
        index2.sync()

def test_schema_version_check(index_setup):
    params, vault = index_setup
    index = SearchIndex(**params)
    index.sync()

    # Manually corrupt schema version in DB
    import sqlite3
    conn = sqlite3.connect(params["sqlite_path"])
    conn.execute("UPDATE schema_meta SET value = 'v999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    from src.exceptions import IndexCorruptionError
    with pytest.raises(IndexCorruptionError, match="Invalid or corrupt schema_version"):
        SearchIndex(**params)
