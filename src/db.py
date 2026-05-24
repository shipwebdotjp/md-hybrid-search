import sqlite3
import json
import time
from typing import Optional, List, Dict, Any

SCHEMA_VERSION = 2

class Database:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_connection()
        self.check_schema_version()
        self.initialize_schema()

    def _setup_connection(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def initialize_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Set schema version if not exists
            self.conn.execute("""
                INSERT OR IGNORE INTO schema_meta (key, value)
                VALUES ('schema_version', ?)
            """, (str(SCHEMA_VERSION),))

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    collection_name TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    collection_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (collection_name, source_path),
                    FOREIGN KEY (collection_name) REFERENCES collections(collection_name) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    collection_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_indexed_at REAL NOT NULL,
                    PRIMARY KEY (collection_name, file_path),
                    FOREIGN KEY (collection_name) REFERENCES collections(collection_name) ON DELETE CASCADE,
                    FOREIGN KEY (collection_name, source_path) REFERENCES sources(collection_name, source_path) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    token_count INTEGER,
                    mtime REAL NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (collection_name, file_path) REFERENCES files(collection_name, file_path) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_collection_file
                ON chunks(collection_name, file_path)
            """)

            # FTS5 table
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    collection_name UNINDEXED,
                    content
                )
            """)

    def check_schema_version(self):
        # Check if schema_meta table exists
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'")
        if not cursor.fetchone():
            # If schema_meta is missing, ensure the DB is empty.
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            if cursor.fetchone():
                 raise RuntimeError(
                    "Existing database found without schema version information. "
                    "To prevent corruption, automatic initialization is blocked. Please rebuild the index."
                )
            return

        cursor = self.conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(
                "schema_meta table exists but schema_version is missing. "
                "The database may be corrupted. Please rebuild the index."
            )

        version = int(row['value'])
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch: database has v{version}, but code expects v{SCHEMA_VERSION}. "
                "Automatic migration is not supported. Please rebuild the index."
            )

    # Collection CRUD
    def upsert_collection(self, name: str, metadata_json: str):
        now = time.time()
        with self.conn:
            self.conn.execute("""
                INSERT INTO collections (collection_name, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(collection_name) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
            """, (name, now, now, metadata_json))

    def get_collection(self, name: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM collections WHERE collection_name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_collection(self, name: str):
        with self.conn:
            # Delete from chunks_fts (virtual table, no CASCADE)
            self.conn.execute("DELETE FROM chunks_fts WHERE collection_name = ?", (name,))
            # This will cascade delete sources, files, and chunks
            self.conn.execute("DELETE FROM collections WHERE collection_name = ?", (name,))

    # Source CRUD
    def upsert_source(self, collection_name: str, source_path: str):
        now = time.time()
        with self.conn:
            self.conn.execute("""
                INSERT INTO sources (collection_name, source_path, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(collection_name, source_path) DO UPDATE SET
                    updated_at = excluded.updated_at
            """, (collection_name, source_path, now, now))

    def delete_sources_except(self, collection_name: str, active_paths: List[str]):
        with self.conn:
            if not active_paths:
                # Clean up FTS5 first
                self.conn.execute("""
                    DELETE FROM chunks_fts WHERE chunk_id IN (
                        SELECT chunk_id FROM chunks WHERE collection_name = ?
                    )
                """, (collection_name,))
                self.conn.execute("DELETE FROM sources WHERE collection_name = ?", (collection_name,))
            else:
                placeholders = ','.join(['?'] * len(active_paths))
                # Clean up FTS5 for sources being deleted
                self.conn.execute(f"""
                    DELETE FROM chunks_fts WHERE chunk_id IN (
                        SELECT chunk_id FROM chunks
                        WHERE collection_name = ? AND source_path NOT IN ({placeholders})
                    )
                """, [collection_name] + active_paths)

                query = f"DELETE FROM sources WHERE collection_name = ? AND source_path NOT IN ({placeholders})"
                self.conn.execute(query, [collection_name] + active_paths)

    def get_sources(self, collection_name: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM sources WHERE collection_name = ?", (collection_name,))
        return [dict(row) for row in cursor.fetchall()]

    # File CRUD
    def upsert_file(self, file_record: Dict[str, Any]):
        with self.conn:
            self.conn.execute("""
                INSERT INTO files (
                    collection_name, file_path, source_path, relative_path,
                    mtime, size, content_hash, last_indexed_at
                )
                VALUES (:collection_name, :file_path, :source_path, :relative_path,
                        :mtime, :size, :content_hash, :last_indexed_at)
                ON CONFLICT(collection_name, file_path) DO UPDATE SET
                    source_path = excluded.source_path,
                    relative_path = excluded.relative_path,
                    mtime = excluded.mtime,
                    size = excluded.size,
                    content_hash = excluded.content_hash,
                    last_indexed_at = excluded.last_indexed_at
            """, file_record)

    def delete_file(self, collection_name: str, file_path: str):
        with self.conn:
            # Delete from chunks_fts using subquery
            self.conn.execute("""
                DELETE FROM chunks_fts WHERE chunk_id IN (
                    SELECT chunk_id FROM chunks WHERE collection_name = ? AND file_path = ?
                )
            """, (collection_name, file_path))
            # Delete file, chunks are deleted by CASCADE
            self.conn.execute("DELETE FROM files WHERE collection_name = ? AND file_path = ?", (collection_name, file_path))

    def get_file(self, collection_name: str, file_path: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM files WHERE collection_name = ? AND file_path = ?", (collection_name, file_path))
        row = cursor.fetchone()
        return dict(row) if row else None

    # Chunk CRUD
    def upsert_chunk(self, chunk_record: Dict[str, Any]):
        with self.conn:
            self.conn.execute("""
                INSERT INTO chunks (
                    chunk_id, collection_name, file_path, source_path, relative_path,
                    chunk_index, content, content_hash, token_count, mtime, created_at
                )
                VALUES (:chunk_id, :collection_name, :file_path, :source_path, :relative_path,
                        :chunk_index, :content, :content_hash, :token_count, :mtime, :created_at)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    collection_name = excluded.collection_name,
                    file_path = excluded.file_path,
                    source_path = excluded.source_path,
                    relative_path = excluded.relative_path,
                    chunk_index = excluded.chunk_index,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    token_count = excluded.token_count,
                    mtime = excluded.mtime,
                    created_at = excluded.created_at
            """, chunk_record)

            # Sync to FTS
            self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_record['chunk_id'],))
            self.conn.execute("""
                INSERT INTO chunks_fts (chunk_id, collection_name, content)
                VALUES (?, ?, ?)
            """, (chunk_record['chunk_id'], chunk_record['collection_name'], chunk_record['content']))

    def delete_chunks_for_file(self, collection_name: str, file_path: str):
        with self.conn:
            # Delete from chunks_fts using subquery
            self.conn.execute("""
                DELETE FROM chunks_fts WHERE chunk_id IN (
                    SELECT chunk_id FROM chunks WHERE collection_name = ? AND file_path = ?
                )
            """, (collection_name, file_path))
            # Delete from chunks
            self.conn.execute("DELETE FROM chunks WHERE collection_name = ? AND file_path = ?", (collection_name, file_path))

    def get_chunks_for_file(self, collection_name: str, file_path: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM chunks WHERE collection_name = ? AND file_path = ? ORDER BY chunk_index", (collection_name, file_path))
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()
