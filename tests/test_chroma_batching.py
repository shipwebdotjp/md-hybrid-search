import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from md_hybrid_search import SearchIndex, DirectorySource
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
def index_params(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return {
        "collection_name": "test-batch-coll",
        "sources": [DirectorySource(str(vault))],
        "sqlite_path": str(tmp_path / "test.sqlite"),
        "chroma_path": str(tmp_path / "chroma"),
        "embedder": MockEmbedder(),
        "chunk_size": 10,
        "chunk_overlap": 0
    }

def test_sync_upsert_per_file(index_params):
    vault_path = Path(index_params["sources"][0].path)
    (vault_path / "note1.md").write_text("file1 content", encoding="utf-8")
    (vault_path / "note2.md").write_text("file2 content", encoding="utf-8")

    index = SearchIndex(**index_params)

    with patch.object(index.chroma_collection, "upsert") as mock_upsert:
        index.sync()
        # Should be called twice (one for each file)
        assert mock_upsert.call_count == 2

def test_sync_upsert_batching(index_params):
    vault_path = Path(index_params["sources"][0].path)
    # 15 characters, chunk_size=10 -> 2 chunks
    (vault_path / "note1.md").write_text("1234567890abcde", encoding="utf-8")

    index = SearchIndex(**index_params)

    # Mock max_batch_size to 1
    with patch.object(index.chroma_client, "get_max_batch_size", return_value=1):
        with patch.object(index.chroma_collection, "upsert") as mock_upsert:
            index.sync()
            # 2 chunks / batch_size 1 -> 2 upserts for one file
            assert mock_upsert.call_count == 2

def test_chroma_delete_batching(index_params):
    index = SearchIndex(**index_params)

    ids = ["id1", "id2", "id3"]

    # Mock max_batch_size to 2
    with patch.object(index.chroma_client, "get_max_batch_size", return_value=2):
        with patch.object(index.chroma_collection, "delete") as mock_delete:
            index._chroma_delete(ids)
            # 3 IDs / batch_size 2 -> 2 delete calls
            assert mock_delete.call_count == 2

            # Check calls
            # Call 1: id1, id2 (or any 2 from set)
            # Call 2: id3 (remaining 1)
            args1, kwargs1 = mock_delete.call_args_list[0]
            args2, kwargs2 = mock_delete.call_args_list[1]

            assert len(kwargs1["ids"]) == 2
            assert len(kwargs2["ids"]) == 1
