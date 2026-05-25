import pytest
import os
import time
from pathlib import Path
from src.index import SearchIndex, DirectorySource, SyncReport
from typing import List

class MockEmbedder:
    def __init__(self):
        self.model_name = "test-model"
        self.embedding_dim = 128
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls += 1
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls += 1
        return [0.1] * 128

@pytest.fixture
def index_params(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return {
        "collection_name": "test-sync-coll",
        "sources": [DirectorySource(str(vault))],
        "sqlite_path": str(tmp_path / "test.sqlite"),
        "chroma_path": str(tmp_path / "chroma"),
        "embedder": MockEmbedder(),
        "chunk_size": 100,
        "chunk_overlap": 10
    }

def test_sync_new_file(index_params):
    vault_path = Path(index_params["sources"][0].path)
    (vault_path / "note1.md").write_text("Hello World", encoding="utf-8")

    index = SearchIndex(**index_params)
    report = index.sync()

    assert report.scanned_files == 1
    assert report.new_files == 1
    assert report.inserted_chunks == 1
    assert report.deleted_files == 0
    assert report.updated_files == 0
    assert report.unchanged_files == 0

def test_sync_unchanged_file(index_params):
    vault_path = Path(index_params["sources"][0].path)
    note1 = vault_path / "note1.md"
    note1.write_text("Hello World", encoding="utf-8")

    index = SearchIndex(**index_params)
    index.sync() # First sync

    report = index.sync() # Second sync
    assert report.scanned_files == 1
    assert report.unchanged_files == 1
    assert report.new_files == 0
    assert report.updated_files == 0

def test_sync_updated_file(index_params):
    vault_path = Path(index_params["sources"][0].path)
    note1 = vault_path / "note1.md"
    note1.write_text("Hello World", encoding="utf-8")

    index = SearchIndex(**index_params)
    index.sync()

    # Update file
    note1.write_text("Hello World Updated", encoding="utf-8")
    # Deterministic mtime bump
    forced_mtime = note1.stat().st_mtime + 2.0
    os.utime(note1, (forced_mtime, forced_mtime))

    report = index.sync()
    assert report.scanned_files == 1
    assert report.updated_files == 1
    assert report.inserted_chunks == 1
    assert report.deleted_chunks == 1

def test_sync_deleted_file(index_params):
    vault_path = Path(index_params["sources"][0].path)
    note1 = vault_path / "note1.md"
    note1.write_text("Hello World", encoding="utf-8")

    index = SearchIndex(**index_params)
    index.sync()

    # Delete file
    note1.unlink()

    report = index.sync()
    assert report.scanned_files == 0
    assert report.deleted_files == 1
    assert report.deleted_chunks == 1

def test_sync_source_removal(tmp_path):
    vault1 = tmp_path / "vault1"
    vault1.mkdir()
    (vault1 / "note1.md").write_text("Note 1", encoding="utf-8")

    vault2 = tmp_path / "vault2"
    vault2.mkdir()
    (vault2 / "note2.md").write_text("Note 2", encoding="utf-8")

    params = {
        "collection_name": "test-source-removal",
        "sources": [DirectorySource(str(vault1)), DirectorySource(str(vault2))],
        "sqlite_path": str(tmp_path / "test.sqlite"),
        "chroma_path": str(tmp_path / "chroma"),
        "embedder": MockEmbedder(),
    }

    index = SearchIndex(**params)
    index.sync()

    # Remove one source
    params["sources"] = [DirectorySource(str(vault1))]
    index2 = SearchIndex(**params)
    report = index2.sync()

    assert report.deleted_files == 1 # note2.md should be deleted
    assert report.scanned_files == 1
    assert report.unchanged_files == 1

def test_sync_missing_source_raises(index_params):
    vault_path = Path(index_params["sources"][0].path)
    (vault_path / "note1.md").write_text("Hello", encoding="utf-8")

    index = SearchIndex(**index_params)

    # Remove the directory from disk
    import shutil
    shutil.rmtree(vault_path)

    with pytest.raises(FileNotFoundError):
        index.sync()

def test_sync_embed_failure_rolls_back(index_params):
    vault_path = Path(index_params["sources"][0].path)
    (vault_path / "note1.md").write_text("Hello", encoding="utf-8")

    class FailingEmbedder:
        embedding_dim = 128
        def embed_documents(self, texts):
            raise RuntimeError("Embed fail")
        def embed_query(self, text):
            return [0.1] * 128

    index_params["embedder"] = FailingEmbedder()
    index = SearchIndex(**index_params)

    # Try sync, should fail
    with pytest.raises(RuntimeError, match="Embed fail"):
        index.sync()

    # Verify SQLite is empty (except for collection which might have been created but we moved it into transaction)
    import sqlite3
    conn = sqlite3.connect(index_params["sqlite_path"])
    conn.row_factory = sqlite3.Row

    # Files table should be empty if transaction rolled back
    row = conn.execute("SELECT count(*) as count FROM files").fetchone()
    assert row["count"] == 0

    # Chunks table should be empty
    row = conn.execute("SELECT count(*) as count FROM chunks").fetchone()
    assert row["count"] == 0

    # Collection metadata should NOT exist if it was the first sync and it rolled back
    row = conn.execute("SELECT count(*) as count FROM collections").fetchone()
    assert row["count"] == 0

    conn.close()
