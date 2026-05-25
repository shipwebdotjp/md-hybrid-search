import pytest
from pathlib import Path
from src.index import SearchIndex, DirectorySource
from src.exceptions import EmbeddingError
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

    chunks = index._process_file(str(note.resolve()), str(vault), "note.md")

    assert len(chunks) > 1
    assert chunks[0].content == content[:10]
    assert chunks[0].metadata["relative_path"] == "note.md"
    assert chunks[0].metadata["collection_name"] == "test-coll"
    assert "content_hash" in chunks[0].metadata
    assert chunks[0].embedding == [0.1] * 128
    assert index._embedding_dim == 128

    # embedding cache is not used; repeated calls should hit the embedder again
    index._process_file(str(note.resolve()), str(vault), "note.md")
    assert index.embedder.document_calls == 2

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


def test_index_query_embedding_is_generated_for_similarity_modes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    index = SearchIndex(
        collection_name="test-query-embedding",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=MockEmbedder(),
    )

    index.search(query="hello", mode="keyword")
    assert index.embedder.query_calls == 0

    index.search(query="hello", mode="similarity")
    assert index.embedder.query_calls == 1

    index.search(query="hello", mode="hybrid")
    assert index.embedder.query_calls == 2


def test_index_embedding_dimension_mismatch_raises(tmp_path):
    class BadEmbedder:
        embedding_dim = 128

        def embed_documents(self, texts):
            return [[0.1] * 64 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 64

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("Hello world", encoding="utf-8")

    index = SearchIndex(
        collection_name="test-embedding-mismatch",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=BadEmbedder(),
    )

    with pytest.raises(ValueError):
        index._process_file(str((vault / "note.md").resolve()), str(vault), "note.md")


def test_index_embedding_errors_propagate(tmp_path):
    class FailingEmbedder:
        embedding_dim = 128

        def embed_documents(self, texts):
            raise RuntimeError("document embedding failed")

        def embed_query(self, text):
            raise RuntimeError("query embedding failed")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("Hello world", encoding="utf-8")

    index = SearchIndex(
        collection_name="test-embedding-errors",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=FailingEmbedder(),
    )

    with pytest.raises(EmbeddingError, match="document embedding failed"):
        index._process_file(str((vault / "note.md").resolve()), str(vault), "note.md")

    with pytest.raises(EmbeddingError, match="query embedding failed"):
        index.search(query="hello", mode="similarity")
