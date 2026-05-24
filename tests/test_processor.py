import os
import pytest
from src.processor import load_markdown, chunk_text, generate_chunk_id, normalize_text

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
    # 0:3 -> abc
    # (3-1):5 -> cde
    # (5-1):7 -> efg
    # (7-1):9 -> ghi
    # (9-1):11 -> ij
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

def test_generate_chunk_id():
    id1 = generate_chunk_id("coll", "/path/file.md", 0, "hash1")
    id2 = generate_chunk_id("coll", "/path/file.md", 0, "hash1")
    assert id1 == id2

    id3 = generate_chunk_id("coll", "/path/file.md", 1, "hash1")
    assert id1 != id3

def test_normalize_text():
    text = "  Hello   WORLD! \n New line. "
    normalized = normalize_text(text)
    assert normalized == "hello world! new line."

    # CJK and mixed
    text_cjk = "こんにちは  世界\nHello WORLD"
    normalized_cjk = normalize_text(text_cjk)
    assert normalized_cjk == "こんにちは 世界 hello world"
