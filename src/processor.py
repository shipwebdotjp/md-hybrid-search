import hashlib
import re
import json
from typing import List, Any
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    content_hash: str
    chunk_index: int
    metadata: dict[str, Any]

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
    Uses JSON serialization to avoid collisions.
    """
    data = {
        "collection_name": collection_name,
        "file_path": file_path,
        "chunk_index": chunk_index,
        "content_hash": content_hash
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()

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
