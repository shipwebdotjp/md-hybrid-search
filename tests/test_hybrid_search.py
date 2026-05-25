import pytest
from pathlib import Path
from src.index import SearchIndex, DirectorySource, SearchHit
from typing import List

class MockEmbedder:
    def __init__(self):
        self.embedding_dim = 128

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Deterministic embeddings based on content to simulate similarity
        # "apple" -> [1.0, 0, ...]
        # "banana" -> [0, 1.0, ...]
        embeddings = []
        for text in texts:
            vec = [0.0] * 128
            if "apple" in text.lower():
                vec[0] = 1.0
            if "banana" in text.lower():
                vec[1] = 1.0
            embeddings.append(vec)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * 128
        if "apple" in text.lower():
            vec[0] = 1.0
        if "banana" in text.lower():
            vec[1] = 1.0
        return vec

@pytest.fixture
def hybrid_index(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    # note1: apple apple apple (good for keyword)
    (vault / "note1.md").write_text("apple apple apple", encoding="utf-8")
    # note2: apple (okay for keyword, good for similarity)
    (vault / "note2.md").write_text("apple", encoding="utf-8")
    # note3: banana (irrelevant for apple search)
    (vault / "note3.md").write_text("banana", encoding="utf-8")

    index = SearchIndex(
        collection_name="test-hybrid",
        sources=[DirectorySource(str(vault))],
        sqlite_path=str(tmp_path / "test.sqlite"),
        chroma_path=str(tmp_path / "chroma"),
        embedder=MockEmbedder(),
        chunk_size=100,
        chunk_overlap=0
    )
    index.sync()
    return index

def test_hybrid_search_rrf(hybrid_index):
    # Search for apple
    # Keyword: note1 (higher rank due to frequency/BM25), note2
    # Similarity: note1 and note2 should be close (both have "apple")

    results = hybrid_index.search(query="apple", mode="hybrid", limit=5)

    assert len(results) >= 2
    assert "apple" in results[0].content
    assert results[0].mode == "hybrid"

    # Verify RRF score is calculated
    # For a hit in both at rank 1: 1/(60+1) + 1/(60+1) = 2/61 approx 0.0327
    # Our implementation uses 0-based index for rank in loop: 1/(k + i + 1)
    # so rank 1 (i=0) is 1/(60+1).
    assert results[0].score > 0
    assert results[0].score > results[-1].score

def test_hybrid_search_deduplication(hybrid_index):
    # note1 matches both keyword and similarity
    results = hybrid_index.search(query="apple", mode="hybrid", limit=10)

    # Check for duplicate chunk_ids
    chunk_ids = [hit.chunk_id for hit in results]
    assert len(chunk_ids) == len(set(chunk_ids))

def test_hybrid_search_limit(hybrid_index):
    results = hybrid_index.search(query="apple", mode="hybrid", limit=1)
    assert len(results) == 1

def test_hybrid_search_empty_query(hybrid_index):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        hybrid_index.search(query="  ", mode="hybrid")

def test_hybrid_search_invalid_mode(hybrid_index):
    with pytest.raises(ValueError, match="Invalid search mode"):
        hybrid_index.search(query="apple", mode="invalid")
