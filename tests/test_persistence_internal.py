import os
import time
import json
import tempfile
from pathlib import Path
from src.index import SearchIndex, DirectorySource, Embedder

class MockEmbedder:
    def __init__(self):
        self.model_name = "test-model"
        self.embedding_dim = 128
    def embed_documents(self, texts):
        return [[0.1] * 128 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 128

def test_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sqlite_path = str(tmp_path / "test_search.sqlite")
        chroma_path = str(tmp_path / "test_chroma")
        collection_name = "test_collection"

        # Create dummy source directory
        source_dir = str(tmp_path / "test_source")
        os.makedirs(source_dir, exist_ok=True)

        embedder = MockEmbedder()
        sources = [DirectorySource(source_dir)]

        # Initialize SearchIndex
        index = SearchIndex(
            collection_name=collection_name,
            sources=sources,
            sqlite_path=sqlite_path,
            chroma_path=chroma_path,
            embedder=embedder,
            chunk_size=500,
            chunk_overlap=50
        )

        db = index.db

        # Check collection metadata - should be None initially as it is only synced during sync()
        coll = db.get_collection(collection_name)
        assert coll is None

        # Trigger sync to populate sources and metadata
        index.sync()

        coll = db.get_collection(collection_name)
        assert coll is not None
        assert coll['collection_name'] == collection_name
        metadata = json.loads(coll['metadata_json'])
        assert metadata['chunk_size'] == 500
        assert metadata['chunk_overlap'] == 50
        assert len(metadata['embedder_fingerprint']) == 64  # sha256 hex

        db_sources = db.get_sources(collection_name)
        assert len(db_sources) == 1
        assert db_sources[0]['source_path'] == sources[0].path

        # Test File CRUD
        file_record = {
            "collection_name": collection_name,
            "file_path": "/path/to/file.md",
            "source_path": sources[0].path,
            "relative_path": "file.md",
            "mtime": time.time(),
            "size": 1024,
            "content_hash": "hash123",
            "last_indexed_at": time.time()
        }
        db.upsert_file(file_record)
        fetched_file = db.get_file(collection_name, "/path/to/file.md")
        assert fetched_file is not None
        assert fetched_file['content_hash'] == "hash123"

        # Test Chunk CRUD
        chunk_record = {
            "chunk_id": "chunk1",
            "collection_name": collection_name,
            "file_path": "/path/to/file.md",
            "source_path": sources[0].path,
            "relative_path": "file.md",
            "chunk_index": 0,
            "content": "Hello, WORLD!!",
            "normalized_content": "hello world",
            "content_hash": "hash456",
            "token_count": 2,
            "mtime": time.time(),
            "created_at": time.time()
        }
        db.upsert_chunk(chunk_record)
        chunks = db.get_chunks_for_file(collection_name, "/path/to/file.md")
        assert len(chunks) == 1
        assert chunks[0]['content'] == "Hello, WORLD!!"

        # Check FTS
        cursor = db.conn.execute("SELECT * FROM chunks_fts WHERE content MATCH 'world'")
        fts_row = cursor.fetchone()
        assert fts_row is not None
        assert fts_row['chunk_id'] == "chunk1"
        assert fts_row['content'] == "hello world"

        # Test deletion
        db.delete_file(collection_name, "/path/to/file.md")
        assert db.get_file(collection_name, "/path/to/file.md") is None
        assert len(db.get_chunks_for_file(collection_name, "/path/to/file.md")) == 0

        cursor = db.conn.execute("SELECT * FROM chunks_fts WHERE chunk_id = 'chunk1'")
        assert cursor.fetchone() is None

        # Test Source CASCADE to File/Chunk
        db.upsert_file(file_record)
        db.upsert_chunk(chunk_record)
        assert db.get_file(collection_name, file_record['file_path']) is not None

        db.delete_sources_except(collection_name, [])
        assert db.get_file(collection_name, file_record['file_path']) is None
        cursor = db.conn.execute("SELECT * FROM chunks WHERE chunk_id = 'chunk1'")
        assert cursor.fetchone() is None
        cursor = db.conn.execute("SELECT * FROM chunks_fts WHERE chunk_id = 'chunk1'")
        assert cursor.fetchone() is None

        print("Persistence tests passed!")

        # Test schema version mismatch
        db.conn.execute("UPDATE schema_meta SET value = '999' WHERE key = 'schema_version'")
        db.conn.commit()

        try:
            from src.db import Database
            Database(sqlite_path)
        except RuntimeError as e:
            print(f"Caught expected error: {e}")
        else:
            raise AssertionError("Should have raised RuntimeError for schema mismatch")

if __name__ == "__main__":
    test_persistence()
