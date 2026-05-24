import pytest
from pathlib import Path
from src.index import SearchIndex, DirectorySource
from typing import List

class MockEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 128

def test_index_process_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    content = "Hello world! This is a test note."
    note.write_text(content, encoding="utf-8")

    index = SearchIndex(
        collection_name="test-coll",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=MockEmbedder(),
        chunk_size=10,
        chunk_overlap=2
    )

    chunks = index._process_file(str(vault), "note.md")

    assert len(chunks) > 1
    assert chunks[0].content == content[:10]
    assert chunks[0].metadata["relative_path"] == "note.md"
    assert chunks[0].metadata["collection_name"] == "test-coll"
    assert "content_hash" in chunks[0].metadata

def test_index_sync_wiring(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text("Note 1 content", encoding="utf-8")
    (vault / "note2.md").write_text("Note 2 content", encoding="utf-8")
    (vault / "other.txt").write_text("Not a markdown", encoding="utf-8")

    index = SearchIndex(
        collection_name="test-sync",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=MockEmbedder(),
        chunk_size=100,
        chunk_overlap=10
    )

    report = index.sync()

    assert report.scanned_files == 2
    assert report.inserted_chunks == 2
