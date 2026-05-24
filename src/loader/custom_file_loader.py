import os
import datetime
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.cleaners.core import group_broken_paragraphs
from src.loader.CustomHTMLLoader import CustomHTMLLoader
from src.loader.CustomJSONLoader import CustomJSONLoader
from loader.custom_text_loader import CustomTextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import tiktoken
import json
import logging
from FileIndex import FileIndex
from src.embedding.util import get_token_count, normalize_for_diff
from src.search.fts_sync import get_fts_sync_manager


class ChromaFTSWrapper:
    """ChromaDBとFTS5を同時に操作するラッパークラス"""

    def __init__(self, chroma_db, collection_name: str):
        """
        Initialize the wrapper.

        Args:
            chroma_db: ChromaDB instance (must implement add_documents, update_document, delete)
            collection_name: Collection name for FTS5 metadata
        """
        self.chroma_db = chroma_db
        self.collection_name = collection_name
        self.fts_sync = get_fts_sync_manager()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        ChromaDBにドキュメントを追加し、FTS5にも同期する

        Args:
            documents: List of Document objects to add

        Returns:
            List of ChromaDB document IDs
        """
        # 1. ChromaDBに追加
        ids = self.chroma_db.add_documents(documents)

        # 2. FTS5に同期
        # Also persist chroma_id into Chroma's stored metadatas so that later
        # similarity_search results include chroma_id in Document.metadata.
        update_ids = []
        update_metadatas = []
        for doc_id, doc in zip(ids, documents):
            # ensure the in-memory Document has the chroma_id
            try:
                doc.metadata = doc.metadata or {}
            except Exception:
                # if metadata is not a dict, coerce to dict
                doc.metadata = {}
            doc.metadata["chroma_id"] = doc_id

            update_ids.append(doc_id)
            update_metadatas.append(doc.metadata)

            # sync to FTS (keep existing behavior)
            self.fts_sync.sync_insert(
                chroma_id=doc_id,
                text=doc.page_content,
                collection_name=self.collection_name,
            )

            # KGのキューに追加する場合はここで行う（例: enqueue_for_kg(doc)）
            try:
                queue_item = {
                    "file_path": doc.metadata.get("file_path", ""),
                    "source_path": doc.metadata.get("file_path", ""),
                    "chunk_id": doc_id,
                    "text": doc.page_content,
                    "added_at": datetime.datetime.now(datetime.timezone.utc)
                    .astimezone()
                    .isoformat(),
                }
                with open("/Users/ship/project/vectorsearch/memory_sample/vault-llm-memory/processing_queue.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(queue_item, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.error(f"Failed to append queue item: {e}")
                raise

        # Persist metadatas back into Chroma collection in a single batch if possible.
        try:
            collection = getattr(self.chroma_db, "_collection", None)
            if collection is not None and len(update_ids) > 0:
                # collection.update accepts lists of ids and metadatas
                collection.update(ids=update_ids, metadatas=update_metadatas)
        except Exception:
            # don't fail the whole add if metadata update fails; it's non-critical
            pass

        return ids

    def update_document(self, doc_id: str, new_document: Document):
        """
        ChromaDBのドキュメントを更新し、FTS5も更新する

        Args:
            doc_id: Document ID to update
            new_document: New Document object
        """
        # ChromaDBはupsertで更新
        # Ensure chroma_id is present in the document metadata and persist it
        try:
            new_document.metadata = new_document.metadata or {}
        except Exception:
            new_document.metadata = {}
        new_document.metadata["chroma_id"] = doc_id

        self.chroma_db.update_document(doc_id, new_document)

        # Persist updated metadata into Chroma collection
        try:
            collection = getattr(self.chroma_db, "_collection", None)
            if collection is not None:
                collection.update(ids=[doc_id], metadatas=[new_document.metadata])
        except Exception:
            pass

        # FTS5も同期
        self.fts_sync.sync_update(
            chroma_id=doc_id,
            new_text=new_document.page_content,
            collection_name=self.collection_name,
        )

    def delete(self, doc_ids: List[str]):
        """
        ChromaDBからドキュメントを削除し、FTS5も削除する

        Args:
            doc_ids: List of document IDs to delete
        """
        # ChromaDBから削除
        self.chroma_db.delete(doc_ids)

        # FTS5も同期
        for doc_id in doc_ids:
            self.fts_sync.sync_delete(chroma_id=doc_id)


class CustomFileLoader:
    def __init__(
        self,
        file_path: str,
        db=None,
        fileIndex: Optional[FileIndex] = None,
        directory_path: Optional[str] = None,
        mode: str = "single",
    ):
        """
        Initialize the loader with a file path.
        :param file_path: Path to the file to load.
        :param db: vector DB wrapper (must implement add_documents)
        :param fileIndex: FileIndex instance for registering resulting document ids
        :param directory_path: directory containing the file (used for metadata)
        :param mode: "single" (legacy - add to db immediately) or "prepare" (do not add, return chunks)
        """
        self.file_path = file_path
        self.db = db
        self.fileIndex = fileIndex
        self.directory_path = directory_path
        self.mode = mode

    def load(self) -> dict:
        """
        Load and process a single file.

        Modes:
        - "prepare": chunk the file and return chunks + per-chunk token counts and totals.
        - any other: legacy behavior - chunk and immediately add to DB (self.db.add_documents) and register via fileIndex.

        :return: In prepare mode:
                 {
                   'file_path': str,
                   'chunks': List[Document],
                   'chunk_tokens': List[int],
                   'total_tokens': int,
                   'chunk_count': int
                 }
                 In legacy mode:
                 {'count': int, 'total_tokens': int}
        """
        total_tokens = 0
        total_docs = 0

        from main import get_collection_settings

        collection_name = (
            getattr(self.fileIndex, "collection_name", None) if self.fileIndex else None
        )
        collection_settings = get_collection_settings(collection_name)
        provider = (
            collection_settings.get("embedder", {"provider": "openai"}).get(
                "provider", "openai"
            )
            if collection_settings
            else "openai"
        )
        para_split_re = re.compile(r"(\s*\n\s*){3}")
        # print(f"target: {self.file_path}")
        # choose loader based on extension
        if self.file_path.endswith(".xhtml") or self.file_path.endswith(".html"):
            loader = CustomHTMLLoader(file_path=self.file_path)
        elif self.file_path.endswith(".json"):
            loader = CustomJSONLoader(file_path=self.file_path)
        elif self.file_path.endswith(".md") or self.file_path.endswith(".txt"):
            loader = CustomTextLoader(file_path=self.file_path)
        else:
            loader = UnstructuredFileLoader(file_path=self.file_path, mode="paged")

        # load raw documents from file (may already be chunked by loader)
        try:
            docs = loader.load()
        except Exception as e:
            loader = CustomTextLoader(file_path=self.file_path)
            docs = loader.load()

        if len(docs) == 0:
            if self.mode == "prepare":
                return {
                    "file_path": self.file_path,
                    "chunks": [],
                    "chunk_tokens": [],
                    "total_tokens": 0,
                    "chunk_count": 0,
                }
            if self.fileIndex is not None:
                self.fileIndex.addFile(self.file_path, [])
            return {"count": 0, "total_tokens": 0}

        # normalize paragraphs and attach metadata
        for doc in docs:
            doc.page_content = normalize_for_diff(
                group_broken_paragraphs(doc.page_content, paragraph_split=para_split_re)
            )
            doc.metadata["file_directory"] = self.directory_path
            if doc.metadata.get("filename") is None:
                doc.metadata["filename"] = os.path.basename(self.file_path)
            if doc.metadata.get("last_modified") is None:
                doc.metadata["last_modified"] = datetime.datetime.fromtimestamp(
                    os.path.getmtime(self.file_path)
                ).strftime("%Y-%m-%dT%H:%M:%S")

        # Count tokens on the raw per-doc pieces (before further splitting)
        # We'll compute accurate token counts later per final chunk.
        total_docs += len(docs)

        # final splitting into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                " ",
            ],
        )
        chunks = filter_complex_metadata(text_splitter.split_documents(docs))

        # compute token counts per chunk (this is what will be used for throttling & batching)
        chunk_tokens = []
        for chunk in chunks:
            try:
                token_count = get_token_count(provider, chunk.page_content)
                chunk_tokens.append(token_count)
            except Exception:
                # fallback coarse estimate
                chunk_tokens.append(max(1, int(len(chunk.page_content) / 4)))

        total_tokens = sum(chunk_tokens)

        if self.mode == "prepare":
            return {
                "file_path": self.file_path,
                "chunks": chunks,
                "chunk_tokens": chunk_tokens,
                "total_tokens": total_tokens,
                "chunk_count": len(chunks),
            }

        # legacy behavior: immediately add to DB and register file index
        if self.fileIndex is not None and self.db is not None:
            if len(chunks) != 0:
                # ChromaFTSWrapperを使用してChromaDBとFTS5に同時に追加
                collection_name = getattr(self.fileIndex, "collection_name", None)
                if collection_name:
                    wrapped_db = ChromaFTSWrapper(self.db, collection_name)
                    document_ids = wrapped_db.add_documents(chunks)
                else:
                    # collection_nameがない場合はFTS5同期なし
                    document_ids = self.db.add_documents(chunks)
                self.fileIndex.addFile(self.file_path, document_ids)
            else:
                self.fileIndex.addFile(self.file_path, [])

        return {"count": total_docs, "total_tokens": total_tokens}
