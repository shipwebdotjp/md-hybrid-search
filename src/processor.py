import hashlib
import re
from typing import List
from pathlib import Path

def load_markdown(filepath: str) -> str:
    """Reads a Markdown file in UTF-8. YAML frontmatter is included in the body."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Fixed-length chunking based on character count.
    Reflects chunk_size and chunk_overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += (chunk_size - chunk_overlap)

    return chunks

def generate_chunk_id(collection_name: str, file_path: str, chunk_index: int, content_hash: str) -> str:
    """
    Generates a deterministic chunk_id from collection_name, file_path, chunk_index, and content_hash.
    """
    data = f"{collection_name}|{file_path}|{chunk_index}|{content_hash}"
    return hashlib.sha256(data.encode()).hexdigest()

def normalize_text(text: str) -> str:
    """
    Basic normalization for FTS indexing.
    Lowercases and collapses whitespace.
    """
    # Simple normalization: lowercase and whitespace collapse
    # In a real-world scenario, this might involve more complex CJK tokenization
    normalized = text.lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized
