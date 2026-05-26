import os
import pytest
from md_hybrid_search.processor import load_markdown, chunk_text, generate_chunk_id, normalize_text, Chunk

def test_load_markdown(tmp_path):
    content = "---\ntitle: Test\n---\nBody content."
    p = tmp_path / "test.md"
    p.write_text(content, encoding="utf-8")

    loaded = load_markdown(str(p))
    assert loaded == content

def test_chunk_text():
    text = "abcdefghij" # 10 chars

    # No overlap
    chunks = chunk_text(text, chunk_size=3, chunk_overlap=0)
    assert chunks == ["abc", "def", "ghi", "j"]

    # With overlap
    chunks = chunk_text(text, chunk_size=3, chunk_overlap=1)
    assert chunks == ["abc", "cde", "efg", "ghi", "ij"]

    # Large chunk size
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert chunks == ["abcdefghij"]

    # Empty text
    assert chunk_text("", 10, 2) == []

def test_chunk_text_errors():
    with pytest.raises(ValueError):
        chunk_text("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 11)
    with pytest.raises(ValueError):
        chunk_text("abc", 2, -1)

def test_generate_chunk_id():
    id1 = generate_chunk_id("coll", "/path/file.md", 0, "hash1")
    id2 = generate_chunk_id("coll", "/path/file.md", 0, "hash1")
    assert id1 == id2

    id3 = generate_chunk_id("coll", "/path/file.md", 1, "hash1")
    assert id1 != id3

def test_normalize_text():
    text = "  Hello   WORLD! \n New line. "
    normalized = normalize_text(text)

    # Relaxed assertion: lowercase, no punctuation, and contains expected words
    assert normalized == normalized.lower()
    assert "!" not in normalized
    assert "." not in normalized

    tokens = normalized.split()
    assert "hello" in tokens
    assert "world" in tokens
    assert "new" in tokens
    assert "line" in tokens
    # Ensure no empty tokens
    assert all(tokens)

    # CJK and mixed
    text_cjk = "こんにちは  世界\nHello WORLD"
    normalized_cjk = normalize_text(text_cjk)
    assert "こんにちは" in normalized_cjk
    assert "世界" in normalized_cjk
    assert "hello" in normalized_cjk.lower()
    assert "world" in normalized_cjk.lower()

def test_chunk_dataclass():
    c = Chunk(
        chunk_id="id1",
        content="content",
        content_hash="hash",
        chunk_index=0,
        metadata={"key": "value"}
    )
    assert c.chunk_id == "id1"
    assert c.metadata["key"] == "value"
