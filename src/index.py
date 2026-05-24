from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, List, Optional, Any, Literal
import re
import os
import json
import hashlib
from .db import Database, SCHEMA_VERSION

@dataclass(frozen=True)
class DirectorySource:
    path: str

    def __post_init__(self):
        # 1. ~ expansion -> 2. Absolute -> 3. Resolve
        p = Path(self.path).expanduser().absolute().resolve()
        object.__setattr__(self, "path", str(p))

class Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

@dataclass(frozen=True)
class SyncReport:
    collection_name: str
    scanned_files: int
    new_files: int
    updated_files: int
    unchanged_files: int
    deleted_files: int
    inserted_chunks: int
    deleted_chunks: int

@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    mode: Literal["keyword", "similarity", "hybrid"]
    content: str
    metadata: dict[str, Any]

class SearchIndex:
    def __init__(
        self,
        collection_name: str,
        sources: List[DirectorySource],
        sqlite_path: str,
        chroma_path: str,
        embedder: Embedder,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self._validate_collection_name(collection_name)
        self.collection_name = collection_name
        self.sources = self._normalize_and_validate_sources(sources)
        self.sqlite_path = sqlite_path
        self.chroma_path = chroma_path
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize database
        self.db = Database(sqlite_path)

        # Upsert collection and metadata
        self._sync_collection_metadata()

        # NOTE: _sync_sources_to_db() is moved to sync() to avoid destructive operations during init.

    def _get_embedder_fingerprint(self) -> str:
        # Gather properties for a deterministic fingerprint
        props = {
            "class": self.embedder.__class__.__name__,
            "model_name": getattr(self.embedder, "model_name", None),
            "embedding_dim": getattr(self.embedder, "embedding_dim", getattr(self.embedder, "dim", None)),
        }
        # Serialize to stable JSON
        serialized = json.dumps(props, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _get_tokenizer_fingerprint(self) -> str:
        # Gather properties for a deterministic fingerprint
        # Since tokenizer is internal and not yet fully exposed, we use placeholder logic for now.
        props = {
            "name": "standard",
            "version": "v1"
        }
        serialized = json.dumps(props, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _sync_collection_metadata(self):
        metadata = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder_fingerprint": self._get_embedder_fingerprint(),
            "tokenizer_fingerprint": self._get_tokenizer_fingerprint(),
            "schema_version": SCHEMA_VERSION,
        }
        self.db.upsert_collection(self.collection_name, json.dumps(metadata))

    def _sync_sources_to_db(self):
        # Add current sources
        active_paths = []
        for source in self.sources:
            self.db.upsert_source(self.collection_name, source.path)
            active_paths.append(source.path)

        # Remove sources that are no longer in the list for this collection
        self.db.delete_sources_except(self.collection_name, active_paths)

    def _validate_collection_name(self, name: str):
        # collection_name rules:
        # - alphanumeric, underscore, hyphen
        # - 3-63 characters
        # - start/end with alphanumeric
        if not (3 <= len(name) <= 63):
            raise ValueError(f"collection_name must be between 3 and 63 characters: {name}")

        pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$"
        if not re.match(pattern, name):
            raise ValueError(
                f"collection_name must start and end with alphanumeric characters and "
                f"contain only alphanumeric, underscores, or hyphens: {name}"
            )

    def _normalize_and_validate_sources(self, sources: List[DirectorySource]) -> List[DirectorySource]:
        # Deduplicate sources based on normalized path
        unique_paths = {}
        for s in sources:
            unique_paths[s.path] = s

        sorted_sources = sorted(unique_paths.values(), key=lambda x: x.path)

        # Check for parent-child relationship
        for i, s1 in enumerate(sorted_sources):
            p1 = Path(s1.path)
            for s2 in sorted_sources[i+1:]:
                p2 = Path(s2.path)
                # Check if p1 is a parent of p2
                if p1 in p2.parents or p1 == p2:
                     raise ValueError(f"Sources cannot have parent-child relationship: {p1} and {p2}")

        return sorted_sources

    def sync(self) -> SyncReport:
        if not self.sources:
            raise ValueError("sources cannot be empty for sync()")

        for source in self.sources:
            if not Path(source.path).exists():
                raise FileNotFoundError(f"Source path does not exist: {source.path}")

        # Sync sources to DB now that we're actually syncing
        self._sync_sources_to_db()

        # Implementation out of scope for this session
        return SyncReport(
            collection_name=self.collection_name,
            scanned_files=0,
            new_files=0,
            updated_files=0,
            unchanged_files=0,
            deleted_files=0,
            inserted_chunks=0,
            deleted_chunks=0
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: Literal["keyword", "similarity", "hybrid"] = "hybrid"
    ) -> List[SearchHit]:
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        if limit < 1:
            raise ValueError(f"Limit must be at least 1: {limit}")

        valid_modes = ["keyword", "similarity", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Implementation out of scope for this session
        return []

    def rebuild(self) -> SyncReport:
        # Sync sources to DB for rebuild too
        self._sync_sources_to_db()
        # Implementation out of scope for this session
        return self.sync()

    def clear(self) -> None:
        # Implementation out of scope for this session
        pass
