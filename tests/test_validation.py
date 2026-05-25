import pytest
from pathlib import Path
from md_hybrid_search import SearchIndex, DirectorySource
from md_hybrid_search import SourceNotFoundError
from typing import List

class MockEmbedder:
    embedding_dim = 128
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 128

@pytest.fixture
def base_params(tmp_path):
    return {
        "sqlite_path": str(tmp_path / "test.sqlite"),
        "chroma_path": str(tmp_path / "chroma"),
        "embedder": MockEmbedder(),
    }

def test_collection_name_validation(base_params):
    # Valid names
    for name in ["valid-name", "valid_name", "v123", "a-b_c", "A1B2"]:
        SearchIndex(collection_name=name, sources=[], **base_params)

    # Too short
    with pytest.raises(ValueError, match="between 3 and 63 characters"):
        SearchIndex(collection_name="ab", sources=[], **base_params)

    # Too long
    with pytest.raises(ValueError, match="between 3 and 63 characters"):
        SearchIndex(collection_name="a" * 64, sources=[], **base_params)

    # Invalid characters
    for name in ["invalid.name", "invalid space", "invalid!", "@start", "end@"]:
        with pytest.raises(ValueError, match="collection_name must start and end with alphanumeric"):
            SearchIndex(collection_name=name, sources=[], **base_params)

def test_directory_source_path_normalization(tmp_path):
    # We can't easily test ~ expansion in sandbox without side effects,
    # but we can test absolute and resolve.
    subdir = tmp_path / "sub"
    subdir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(subdir, target_is_directory=True)

    source = DirectorySource(str(link))
    # DirectorySource uses resolve(), so link should resolve to subdir
    assert source.path == str(subdir.resolve())
    assert Path(source.path).is_absolute()

def test_source_redundancy_and_hierarchy(base_params, tmp_path):
    p1 = tmp_path / "p1"
    p1.mkdir()
    p1_sub = p1 / "sub"
    p1_sub.mkdir()
    p2 = tmp_path / "p2"
    p2.mkdir()

    # Duplicate paths (should be deduplicated silently)
    index = SearchIndex(
        collection_name="test-dup",
        sources=[DirectorySource(str(p1)), DirectorySource(str(p1))],
        **base_params
    )
    assert len(index.sources) == 1

    # Parent-child relationship
    with pytest.raises(ValueError, match="parent-child relationship"):
        SearchIndex(
            collection_name="test-hierarchy",
            sources=[DirectorySource(str(p1)), DirectorySource(str(p1_sub))],
            **base_params
        )

def test_sync_validation(base_params, tmp_path):
    # Empty sources
    index = SearchIndex(collection_name="test-empty", sources=[], **base_params)
    with pytest.raises(ValueError, match="sources cannot be empty for sync"):
        index.sync()

    # Missing source path
    missing_path = tmp_path / "missing"
    index2 = SearchIndex(
        collection_name="test-missing",
        sources=[DirectorySource(str(missing_path))],
        **base_params
    )
    # Note: DirectorySource(str(missing_path)) won't raise if the path doesn't exist yet,
    # it just resolves it.
    with pytest.raises(SourceNotFoundError):
        index2.sync()
