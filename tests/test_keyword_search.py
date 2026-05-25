import pytest
from pathlib import Path
from src.index import SearchIndex, DirectorySource, SearchHit
from typing import List

class MockEmbedder:
    def __init__(self):
        self.model_name = "test-model"
        self.embedding_dim = 128

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 128

@pytest.fixture
def index_factory(tmp_path):
    def _create(collection_name="test-coll"):
        vault = tmp_path / collection_name
        vault.mkdir(exist_ok=True)
        return SearchIndex(
            collection_name=collection_name,
            sources=[DirectorySource(str(vault))],
            sqlite_path=str(tmp_path / "test.sqlite"),
            chroma_path=str(tmp_path / "chroma"),
            embedder=MockEmbedder(),
            chunk_size=100,
            chunk_overlap=0
        ), vault
    return _create

def test_keyword_search_basic(index_factory):
    index, vault = index_factory("basic")
    (vault / "note.md").write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    index.sync()

    # Search for a word present in the text
    results = index.search(query="fox", mode="keyword")

    assert len(results) == 1
    hit = results[0]
    assert isinstance(hit, SearchHit)
    assert hit.mode == "keyword"
    assert "fox" in hit.content
    assert hit.score == 1.0  # First rank

    # Check metadata
    assert hit.metadata["collection_name"] == "basic"
    assert "note.md" in hit.metadata["file_path"]
    assert hit.metadata["relative_path"] == "note.md"
    assert hit.metadata["chunk_index"] == 0
    assert "mtime" in hit.metadata
    assert "content_hash" in hit.metadata

def test_keyword_search_no_match(index_factory):
    index, vault = index_factory("no-match")
    (vault / "note.md").write_text("Hello world", encoding="utf-8")
    index.sync()

    results = index.search(query="nonexistent", mode="keyword")
    assert len(results) == 0

def test_keyword_search_multiple_results_scoring(index_factory):
    index, vault = index_factory("scoring")
    (vault / "note1.md").write_text("apple apple apple", encoding="utf-8")
    (vault / "note2.md").write_text("apple", encoding="utf-8")
    index.sync()

    results = index.search(query="apple", mode="keyword")

    assert len(results) == 2
    # Scores should be 1/1, 1/2
    assert results[0].score == 1.0
    assert results[1].score == 0.5
    assert results[0].score > results[1].score

def test_keyword_search_collection_scoping(index_factory):
    # Collection A
    index_a, vault_a = index_factory("coll-a")
    (vault_a / "note.md").write_text("Unique content in collection A", encoding="utf-8")
    index_a.sync()

    # Collection B
    index_b, vault_b = index_factory("coll-b")
    (vault_b / "note.md").write_text("Unique content in collection B", encoding="utf-8")
    index_b.sync()

    # Search in A should not find B
    results_a = index_a.search(query="collection", mode="keyword")
    assert len(results_a) == 1
    assert "collection A" in results_a[0].content

    # Search in B should not find A
    results_b = index_b.search(query="collection", mode="keyword")
    assert len(results_b) == 1
    assert "collection B" in results_b[0].content

def test_keyword_search_sync_updates(index_factory):
    index, vault = index_factory("sync-updates")
    note = vault / "note.md"
    note.write_text("Old content with keyword apple", encoding="utf-8")
    index.sync()

    assert len(index.search(query="apple", mode="keyword")) == 1

    # Update file content
    note.write_text("New content with keyword banana", encoding="utf-8")
    index.sync()

    # Keyword 'apple' should no longer match
    assert len(index.search(query="apple", mode="keyword")) == 0
    # Keyword 'banana' should now match
    assert len(index.search(query="banana", mode="keyword")) == 1

def test_keyword_search_sync_deletions(index_factory):
    index, vault = index_factory("sync-deletions")
    note = vault / "note.md"
    note.write_text("Content with keyword cherry", encoding="utf-8")
    index.sync()

    assert len(index.search(query="cherry", mode="keyword")) == 1

    # Delete file
    note.unlink()
    index.sync()

    # Keyword 'cherry' should no longer match
    assert len(index.search(query="cherry", mode="keyword")) == 0
