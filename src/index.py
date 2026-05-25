from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, List, Optional, Any, Literal
import re
import os
import json
import hashlib
import time
import chromadb
from .db import Database, SCHEMA_VERSION
from . import processor
from .exceptions import (
    MdHybridSearchError,
    ConfigMismatchError,
    IndexCorruptionError,
    EmbeddingError,
    SourceNotFoundError,
)

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
        # This tracks the observed embedding dimension for validation only.
        self._embedding_dim: Optional[int] = None

        # Initialize database
        self.db = Database(sqlite_path)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

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

    def _check_config_mismatch(self) -> None:
        """
        Checks if the current configuration matches the persisted metadata in the database.
        Raises ConfigMismatchError if there is a divergence.
        """
        coll_data = self.db.get_collection(self.collection_name)
        if not coll_data:
            return

        try:
            stored_meta = json.loads(coll_data["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            # If metadata is corrupted, we should probably treat it as a mismatch
            # but for now let's just assume it's a mismatch that requires rebuild.
            raise ConfigMismatchError(
                f"Collection '{self.collection_name}' has corrupted metadata. "
                "Please run rebuild() to re-index."
            )

        current_meta = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder_fingerprint": self._get_embedder_fingerprint(),
            "tokenizer_fingerprint": self._get_tokenizer_fingerprint(),
        }

        mismatches = []
        for key, current_val in current_meta.items():
            stored_val = stored_meta.get(key)
            if stored_val != current_val:
                mismatches.append(f"{key} (stored: {stored_val}, current: {current_val})")

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
        # This will cascade delete files and chunks in SQLite
        # But we need to handle ChromaDB cleanup before that if possible,
        # or find which files were deleted.
        self.db.delete_sources_except(self.collection_name, active_paths)

    def _declared_embedding_dim(self) -> Optional[int]:
        for attr in ("embedding_dim", "dim"):
            value = getattr(self.embedder, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _record_embedding_dim(self, observed_dim: int) -> int:
        if observed_dim <= 0:
            raise ValueError("Embedding dimension must be a positive integer")

        declared_dim = self._declared_embedding_dim()
        expected_dim = declared_dim if declared_dim is not None else self._embedding_dim

        if expected_dim is None:
            self._embedding_dim = observed_dim
            return observed_dim

        if observed_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {observed_dim}"
            )

        self._embedding_dim = expected_dim

        return expected_dim

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            embeddings = list(self.embedder.embed_documents(texts))
        except Exception as e:
            raise EmbeddingError(f"Error generating document embeddings: {e}") from e

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"embed_documents returned {len(embeddings)} embeddings for {len(texts)} texts"
            )

        normalized_embeddings: List[List[float]] = []
        for embedding in embeddings:
            vector = list(embedding)
            self._record_embedding_dim(len(vector))
            normalized_embeddings.append(vector)

        return normalized_embeddings

    def _embed_query(self, text: str) -> List[float]:
        try:
            embedding = list(self.embedder.embed_query(text))
        except Exception as e:
            raise EmbeddingError(f"Error generating query embedding: {e}") from e

        self._record_embedding_dim(len(embedding))
        return embedding

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
                raise SourceNotFoundError(f"Source path does not exist: {source.path}")

    def _get_files_on_disk(self) -> dict[str, dict[str, Any]]:
        """Scans sources and returns a map of resolved file_path to its metadata."""
        disk_files = {}
        for source in self.sources:
            source_path = Path(source.path)
            # rglob doesn't follow directory symlinks by default
            for p in source_path.rglob("*.md"):
                if p.is_file():
                    try:
                        resolved_path = p.resolve()
                        stat = resolved_path.stat()
                        disk_files[str(resolved_path)] = {
                            "source_path": str(source_path),
                            "relative_path": str(p.relative_to(source_path)),
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                    except (FileNotFoundError, PermissionError):
                        continue
        return disk_files

    def _calculate_content_hash(self, file_path: str) -> str:
        content = processor.load_markdown(file_path)
        return hashlib.sha256(content.encode()).hexdigest()

    def sync(self) -> SyncReport:
        self._check_config_mismatch()
        self._validate_source_existence()

        # Identify files that WILL be deleted due to source removal
        active_source_paths = [s.path for s in self.sources]

        # 1. Get current state
        disk_files = self._get_files_on_disk()
        db_files = {f["file_path"]: f for f in self.db.get_files_by_collection(self.collection_name)}

        # 2. Determine changes
        to_delete = []
        to_index = []  # List of (file_path, disk_meta, is_update)
        unchanged_files = 0

        # Add files that were orphaned by source removal BUT are not on disk anymore
        # (This is a subset of "Files on DB but not on disk")
        for fp in db_files:
            if fp not in disk_files:
                to_delete.append(fp)

        # Files that were orphaned by source removal BUT ARE still on disk via another source
        # will be handled in the disk loop below (treated as is_update if is_orphaned is True)

        # Files on disk
        for fp, disk_meta in disk_files.items():
            if fp not in db_files:
                to_index.append((fp, disk_meta, False))
            else:
                db_meta = db_files[fp]
                # If the file's stored source_path is no longer active,
                # we MUST re-index it to re-attach it to an active source.
                # Otherwise, _sync_sources_to_db() will delete it via cascade.
                is_orphaned = db_meta["source_path"] not in active_source_paths

                if not is_orphaned and disk_meta["mtime"] == db_meta["mtime"] and disk_meta["size"] == db_meta["size"]:
                    unchanged_files += 1
                else:
                    # Content hash check
                    content_hash = self._calculate_content_hash(fp)
                    if not is_orphaned and content_hash == db_meta["content_hash"]:
                        unchanged_files += 1
                    else:
                        to_index.append((fp, disk_meta, True))

        # 3. Prepare data and collect Chroma IDs to delete
        chroma_ids_to_delete = []
        # Pre-collect IDs for files to be deleted (including orphaned by source removal)
        # We must do this BEFORE the database transaction executes deletions
        for fp in to_delete:
            chroma_ids_to_delete.extend(self.db.get_chunk_ids_for_file(self.collection_name, fp))

        all_new_chunks = []
        indexed_files_data = []  # List of (fp, meta, content_hash, chunks, is_update)
        new_files_count = 0
        updated_files_count = 0
        deleted_chunks_count = len(chroma_ids_to_delete)

        for fp, meta, is_update in to_index:
            if is_update:
                old_chunk_ids = self.db.get_chunk_ids_for_file(self.collection_name, fp)
                chroma_ids_to_delete.extend(old_chunk_ids)
                deleted_chunks_count += len(old_chunk_ids)
                updated_files_count += 1
            else:
                new_files_count += 1

            chunks = self._process_file(fp, meta["source_path"], meta["relative_path"])
            content_hash = self._calculate_content_hash(fp)
            indexed_files_data.append((fp, meta, content_hash, chunks, is_update))
            all_new_chunks.extend(chunks)

        # 4. SQLite Transaction (Unified)
        with self.db.conn:
            # Metadata & Sources
            if not self.db.get_collection(self.collection_name):
                self._sync_collection_metadata()
            self._sync_sources_to_db()

            # Execute deletions
            for fp in to_delete:
                self.db.delete_file(self.collection_name, fp)

            # Update unchanged but mtime/size changed files
            # This is safe to do because content_hash matched
            for fp, disk_meta in disk_files.items():
                if fp in db_files:
                    db_meta = db_files[fp]
                    if disk_meta["mtime"] != db_meta["mtime"] or disk_meta["size"] != db_meta["size"]:
                        # We only do this if it's considered unchanged (hash matched)
                        # Re-calculate hash to be absolutely sure if it's not in to_index
                        content_hash = self._calculate_content_hash(fp)
                        if content_hash == db_meta["content_hash"]:
                            self.db.upsert_file({
                                **db_meta,
                                "mtime": disk_meta["mtime"],
                                "size": disk_meta["size"],
                                "last_indexed_at": time.time()
                            })

            # Execute indexing
            for fp, meta, content_hash, chunks, is_update in indexed_files_data:
                if is_update:
                    self.db.delete_file(self.collection_name, fp)

                now = time.time()
                self.db.upsert_file({
                    "collection_name": self.collection_name,
                    "file_path": fp,
                    "source_path": meta["source_path"],
                    "relative_path": meta["relative_path"],
                    "mtime": meta["mtime"],
                    "size": meta["size"],
                    "content_hash": content_hash,
                    "last_indexed_at": now
                })

                for chunk in chunks:
                    self.db.upsert_chunk({
                        "chunk_id": chunk.chunk_id,
                        "collection_name": self.collection_name,
                        "file_path": fp,
                        "source_path": meta["source_path"],
                        "relative_path": meta["relative_path"],
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "content_normalized": processor.normalize_text(chunk.content),
                        "content_hash": chunk.content_hash,
                        "token_count": None,
                        "mtime": meta["mtime"],
                        "created_at": now
                    })

        # 5. Chroma Sync (after SQLite commit)
        if chroma_ids_to_delete:
            # deduplicate IDs to avoid chromadb.errors.DuplicateIDError
            unique_ids_to_delete = list(set(chroma_ids_to_delete))
            self.chroma_collection.delete(ids=unique_ids_to_delete)

        if all_new_chunks:
            self.chroma_collection.upsert(
                ids=[c.chunk_id for c in all_new_chunks],
                embeddings=[c.embedding for c in all_new_chunks],
                metadatas=[c.metadata for c in all_new_chunks],
                documents=[c.content for c in all_new_chunks]
            )

        return SyncReport(
            collection_name=self.collection_name,
            scanned_files=len(disk_files),
            new_files=new_files_count,
            updated_files=updated_files_count,
            unchanged_files=unchanged_files,
            deleted_files=len(to_delete),
            inserted_chunks=len(all_new_chunks),
            deleted_chunks=deleted_chunks_count
        )

    def _keyword_search_raw(self, query: str, limit: int) -> List[dict[str, Any]]:
        tokenized_query = processor.tokenize_query(query)
        if not tokenized_query:
            return []
        return self.db.keyword_search(self.collection_name, tokenized_query, limit)

    def _similarity_search_raw(self, query: str, limit: int) -> dict[str, Any]:
        embedding = self._embed_query(query)
        return self.chroma_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas"]
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

        if mode == "keyword":
            rows = self._keyword_search_raw(query, limit)
            hits = []
            for i, row in enumerate(rows):
                hits.append(SearchHit(
                    chunk_id=row["chunk_id"],
                    score=1.0 / (i + 1),
                    mode="keyword",
                    content=row["content"],
                    metadata={
                        "collection_name": row["collection_name"],
                        "source_path": row["source_path"],
                        "file_path": row["file_path"],
                        "relative_path": row["relative_path"],
                        "chunk_index": row["chunk_index"],
                        "mtime": row["mtime"],
                        "content_hash": row["content_hash"],
                    }
                ))
            return hits

        if mode == "similarity":
            results = self._similarity_search_raw(query, limit)
            hits = []
            if results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                for i, (chunk_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                    hits.append(SearchHit(
                        chunk_id=chunk_id,
                        score=1.0 / (i + 1),
                        mode="similarity",
                        content=doc,
                        metadata=meta
                    ))
            return hits

        if mode == "hybrid":
            candidate_limit = max(limit * 5, 50)

            # Keyword candidates
            kw_rows = self._keyword_search_raw(query, candidate_limit)

            # Similarity candidates
            sim_results = self._similarity_search_raw(query, candidate_limit)

            # RRF
            k = 60
            scores = {}  # chunk_id -> score
            chunk_data = {}  # chunk_id -> (content, metadata)

            for i, row in enumerate(kw_rows):
                cid = row["chunk_id"]
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + i + 1)
                if cid not in chunk_data:
                    chunk_data[cid] = (row["content"], {
                        "collection_name": row["collection_name"],
                        "source_path": row["source_path"],
                        "file_path": row["file_path"],
                        "relative_path": row["relative_path"],
                        "chunk_index": row["chunk_index"],
                        "mtime": row["mtime"],
                        "content_hash": row["content_hash"],
                    })

            if sim_results["ids"] and sim_results["ids"][0]:
                ids = sim_results["ids"][0]
                documents = sim_results["documents"][0]
                metadatas = sim_results["metadatas"][0]
                for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                    scores[cid] = scores.get(cid, 0) + 1.0 / (k + i + 1)
                    if cid not in chunk_data:
                        chunk_data[cid] = (doc, meta)

            # Sort and return
            sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
            top_ids = sorted_ids[:limit]

            hits = []
            for cid in top_ids:
                content, metadata = chunk_data[cid]
                hits.append(SearchHit(
                    chunk_id=cid,
                    score=scores[cid],
                    mode="hybrid",
                    content=content,
                    metadata=metadata
                ))
            return hits

        return []

    def rebuild(self) -> SyncReport:
        """
        Deletes all index state for the collection in SQLite and ChromaDB,
        then re-indexes from the current sources.
        """
        self._validate_source_existence()

        # Clear existing data for this collection
        self.db.delete_collection(self.collection_name)
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            # If collection didn't exist in Chroma, ignore error
            pass

        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )

        # Re-initialize collection and sources via sync()
        # sync() will call _sync_collection_metadata() because get_collection() will be None
        return self.sync()

    def _process_file(self, file_path: str, source_path: str, relative_path: str) -> List[processor.Chunk]:
        """Loads and chunks a single file, preparing it for indexing."""
        content = processor.load_markdown(file_path)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        mtime = os.path.getmtime(file_path)

        raw_chunks = processor.chunk_text(content, self.chunk_size, self.chunk_overlap)

        chunk_specs = []
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
                "mtime": mtime,
                "content_hash": content_hash,
            }
            chunk_specs.append((chunk_id, text, metadata, i))

        embeddings = self._embed_documents([text for _, text, _, _ in chunk_specs])

        chunks = []
        for (chunk_id, text, metadata, i), embedding in zip(chunk_specs, embeddings):
            chunks.append(
                processor.Chunk(
                    chunk_id=chunk_id,
                    content=text,
                    content_hash=content_hash,
                    chunk_index=i,
                    metadata=metadata,
                    embedding=embedding,
                )
            )
        return chunks

    def clear(self) -> None:
        """
        Removes all index state for the collection from SQLite and ChromaDB.
        Does NOT delete any Markdown files from disk.
        """
        self.db.delete_collection(self.collection_name)
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
