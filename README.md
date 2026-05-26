# md-hybrid-search

A library for local hybrid search (Keyword + Vector) optimized for Markdown files, specifically designed for Obsidian-style workflows.

## Features

- **Differential Sync**: Scans local directories for `.md` files and only indexes changes using `mtime`, `size`, and `content_hash`.
- **Hybrid Search**: Combines SQLite FTS5 (BM25) and ChromaDB (Vector Similarity) using Reciprocal Rank Fusion (RRF).
- **No Over-Engineering**: Focused on being a library, not a full application. You bring the embedder and manage the storage paths.
- **Obsidian-Friendly**: Handles YAML frontmatter and maintains file metadata.

## Installation

```bash
pip install md-hybrid-search
```

## Getting Started

### 1. Define an Embedder

You need to provide an object that follows the `Embedder` protocol. Here's an example using `sentence-transformers`:

```python
from typing import List

class MyEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_dim = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()
```

### 2. Initialize and Sync

```python
from md_hybrid_search import SearchIndex, DirectorySource

index = SearchIndex(
    collection_name="my-vault",
    sources=[
        DirectorySource("/path/to/obsidian-vault"),
    ],
    sqlite_path="data/search.sqlite",
    chroma_path="data/chroma",
    embedder=MyEmbedder(),
    chunk_size=1000,    # Optional: characters per chunk
    chunk_overlap=100,  # Optional: overlap between chunks
)

# Synchronize index with local files
report = index.sync()
print(f"Synced {report.scanned_files} files, inserted {report.inserted_chunks} chunks.")
```

### 3. Search

```python
results = index.search("How to use hybrid search?", limit=5, mode="hybrid")

for hit in results:
    print(f"[{hit.score:.4f}] {hit.metadata['relative_path']}")
    print(hit.content[:100] + "...")
```

## Core API

### `SearchIndex`

- `sync()`: Scans sources and updates the index.
- `search(query, limit=10, mode="hybrid")`: Searches the collection. Modes: `keyword`, `similarity`, `hybrid`.
- `rebuild()`: Clears the index and re-indexes everything. Required when `chunk_size` or `embedder` changes.
- `clear()`: Removes all data for the collection from the index.

### `DirectorySource`

- `path`: The root directory to scan for `.md` files. Subdirectories are scanned recursively.

## Rules and Responsibilities

- **Collection Name**: Must be 3-63 characters, start/end with alphanumeric, and contain only `[a-zA-Z0-9_-]`.
- **Caller Responsibilities**:
    - Manage and persist the `sqlite_path` and `chroma_path`.
    - Provide a consistent `Embedder` implementation.
    - Call `rebuild()` if configuration (chunk size, model, etc.) changes.
- **Rebuild Requirements**: `SearchIndex` detects configuration mismatches and raises `ConfigMismatchError`. You must call `rebuild()` to recover.

## License

MIT
