"""FTS5 sync manager for ChromaDB operations."""

from __future__ import annotations

import logging
from typing import Optional

from src.search.fts_index import FTS5Manager
from src.search.mecab_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


class FTS5SyncManager:
    """Manages synchronization between ChromaDB and FTS5 index."""

    def __init__(self, fts_manager: FTS5Manager, tokenizer):
        """
        Initialize the sync manager.

        Args:
            fts_manager: FTS5Manager instance for FTS5 operations
            tokenizer: Tokenizer instance for text tokenization
        """
        self.fts_manager = fts_manager
        self.tokenizer = tokenizer

    def sync_insert(self, chroma_id: str, text: str, collection_name: str):
        """
        Insert a document into FTS5 index.

        Args:
            chroma_id: ChromaDB document ID
            text: Document text content
            collection_name: Collection name for metadata
        """
        try:
            tokenized_text = self.tokenizer.tokenize_to_string(text)
            # Ensure collection_name is provided when syncing to FTS
            if not collection_name:
                raise ValueError("collection_name is required for FTS sync_insert")
            self.fts_manager.insert_document(
                chroma_id=chroma_id,
                original_text=text,
                tokenized_text=tokenized_text,
                collection_name=collection_name,
            )
            logger.debug(f"FTS5 sync: inserted document {chroma_id}")
        except Exception as e:
            logger.error(f"FTS5 sync insert failed for {chroma_id}: {e}")

    def sync_update(self, chroma_id: str, new_text: str, collection_name: str):
        """
        Update a document in FTS5 index.

        Args:
            chroma_id: ChromaDB document ID
            new_text: New document text content
            collection_name: Collection name for metadata
        """
        try:
            # Ensure collection_name is provided when syncing updates to FTS
            if not collection_name:
                raise ValueError("collection_name is required for FTS sync_update")
            # Delete and re-insert for update
            self.fts_manager.delete_document(chroma_id)
            tokenized_text = self.tokenizer.tokenize_to_string(new_text)
            self.fts_manager.insert_document(
                chroma_id=chroma_id,
                original_text=new_text,
                tokenized_text=tokenized_text,
                collection_name=collection_name,
            )
            logger.debug(f"FTS5 sync: updated document {chroma_id}")
        except Exception as e:
            logger.error(f"FTS5 sync update failed for {chroma_id}: {e}")

    def sync_delete(self, chroma_id: str):
        """
        Delete a document from FTS5 index.

        Args:
            chroma_id: ChromaDB document ID to delete
        """
        try:
            self.fts_manager.delete_document(chroma_id)
            logger.debug(f"FTS5 sync: deleted document {chroma_id}")
        except Exception as e:
            logger.error(f"FTS5 sync delete failed for {chroma_id}: {e}")

    def sync_delete_collection(self, collection_name: str):
        """
        Delete all documents in a collection from FTS5 index.

        Args:
            collection_name: Collection name to delete
        """
        try:
            self.fts_manager.delete_collection(collection_name)
            logger.debug(f"FTS5 sync: deleted collection {collection_name}")
        except Exception as e:
            logger.error(
                f"FTS5 sync delete_collection failed for {collection_name}: {e}"
            )


# Global sync manager instance (singleton)
_fts_sync_manager: Optional[FTS5SyncManager] = None


def get_fts_sync_manager():
    """Get or create the global FTS5 sync manager (singleton)."""
    global _fts_sync_manager
    if _fts_sync_manager is None:
        fts_manager = FTS5Manager()
        tokenizer = get_tokenizer()
        _fts_sync_manager = FTS5SyncManager(fts_manager, tokenizer)
    return _fts_sync_manager


def reset_fts_sync_manager():
    """Reset the global sync manager (useful for testing)."""
    global _fts_sync_manager
    _fts_sync_manager = None
