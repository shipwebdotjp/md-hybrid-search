"""SQLite FTS5 index manager."""
from __future__ import annotations

import logging
import sqlite3
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).resolve().parents[2] / "db" / "fts_index.db"
_SEARCHABLE_RE = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)


class FTS5Manager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._ensure_db()

    def _get_connection(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(chroma_id UNINDEXED, content)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fts_metadata (
                    chroma_id TEXT PRIMARY KEY,
                    original_text TEXT,
                    tokenized_text TEXT,
                    collection_name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fts_sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chroma_id TEXT NOT NULL,
                    operation TEXT CHECK(operation IN ('insert', 'update', 'delete')),
                    sync_status TEXT DEFAULT 'pending' CHECK(sync_status IN ('pending', 'completed', 'failed')),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fts_sync_log_chroma_id ON fts_sync_log(chroma_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fts_metadata_collection ON fts_metadata(collection_name)")
            conn.commit()
        finally:
            conn.close()

    def insert_document(self, chroma_id: str, original_text: str, tokenized_text: str, collection_name: str):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO fts_docs(chroma_id, content) VALUES (?, ?)", (chroma_id, tokenized_text))
            cur.execute("""
                INSERT OR REPLACE INTO fts_metadata (chroma_id, original_text, tokenized_text, collection_name, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (chroma_id, original_text, tokenized_text, collection_name))
            conn.commit()
        finally:
            conn.close()

    def delete_document(self, chroma_id: str):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM fts_docs WHERE chroma_id = ?", (chroma_id,))
            cur.execute("DELETE FROM fts_metadata WHERE chroma_id = ?", (chroma_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_collection(self, collection_name: str):
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM fts_docs WHERE chroma_id IN (SELECT chroma_id FROM fts_metadata WHERE collection_name = ?)", (collection_name,))
            cur.execute("DELETE FROM fts_metadata WHERE collection_name = ?", (collection_name,))
            conn.commit()
        finally:
            conn.close()

    def _fts5_quote(self, term: str) -> str:
        term = term.strip()
        if not term:
            return ""
        return '"' + term.replace('"', '""') + '"'

    def _build_fts5_query(self, query: str, tokenizer) -> str:
        raw_tokens = tokenizer.tokenize(query)

        terms = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue
            # MeCab/fugashi が返す punctuation-only token を除外
            if not _SEARCHABLE_RE.search(t):
                continue
            terms.append(self._fts5_quote(t))

        return " AND ".join(terms)

    def search_bm25(self, query: str, tokenizer, collection_name: str, top_n: int = 10) -> List[Tuple[str, float, str]]:
        if not collection_name:
            raise ValueError("collection_name is required for FTS search")

        fts_query = self._build_fts5_query(query, tokenizer)
        if not fts_query:
            return []

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT d.chroma_id, bm25(fts_docs) AS score, m.original_text
                FROM fts_docs d
                JOIN fts_metadata m ON d.chroma_id = m.chroma_id
                WHERE m.collection_name = ?
                AND d.content MATCH ?
                ORDER BY score
                LIMIT ?
            """, (collection_name, fts_query, top_n))
            rows = cur.fetchall()
            return [(row["chroma_id"], float(row["score"]), row["original_text"]) for row in rows]
        finally:
            conn.close()
