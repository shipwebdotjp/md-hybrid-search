from langchain_openai import OpenAIEmbeddings
from src.module.genai.embeddings import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

# Simple wrapper to mimic the minimal Embeddings interface expected by Chroma/langchain
# Methods:
#  - embed_documents(list[str]) -> list[list[float]]
#  - embed_query(str) -> list[float]
import torch
from transformers import AutoTokenizer, AutoModel

from src.embedding.simple_sbert_embeddings import SimpleSbertEmbeddings

# Adapter that provides a consistent Embeddings-like interface.
# This adapter ensures we always expose embed_documents(list[str]) -> list[list[float]]
# and embed_query(str) -> list[float], regardless of the underlying embedder object.
from src.embedding.embeddings_adapter import EmbeddingsAdapter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
# from sentence_transformers import SentenceTransformer
# from huggingface_hub import snapshot_download
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
import chromadb
from chromadb.config import Settings as chromaSettings
import os
import json
import argparse
from pathlib import Path
from loader.custom_directory_loader import CustomDirectoryLoader
import config
from FileIndex import FileIndex
import ai
from src.search.fts_index import FTS5Manager
from src.search.hybrid_search import HybridSearch
from src.search.mecab_tokenizer import get_tokenizer
from src.search.reranker import ReRanker

# プロジェクトルートディレクトリを取得
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = str(BASE_DIR / "db")


def get_embeddings(collection_name):
    api_keys = {
        "openai": config.settings.get("openai_api_key"),
        "google": config.settings.get("google_api_key"),
    }
    collection_settings = get_collection_settings(collection_name)
    if collection_settings is None:
        embedder = {
            "provider": "openai",
            "model": "text-embedding-3-small",
        }
    else:
        embedder = collection_settings.get(
            "embedder",
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
        )

    if embedder["provider"] == "openai" or embedder.get("provider") is None:
        underlying_embeddings = OpenAIEmbeddings(
            model=embedder.get("model", "text-embedding-3-small"),
            api_key=api_keys["openai"],
        )
    elif embedder["provider"] == "google":
        underlying_embeddings = GoogleGenerativeAIEmbeddings(
            model=embedder.get("model", "models/gemini-embedding-001"),
            google_api_key=api_keys["google"],
        )
    elif embedder["provider"] == "huggingface":
        model_name = embedder.get("model", "intfloat/multilingual-e5-base")
        underlying_embeddings = SimpleSbertEmbeddings(model_name)
    else:
        raise ValueError(f"Unknown embedder provider: {embedder['provider']}")
    return underlying_embeddings

    # store = LocalFileStore("./cache/")

    # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings, store, namespace=underlying_embeddings.model
    # )
    # return cached_embedder


def get_client():
    persist_directory = DB_DIR
    settings = chromaSettings(
        persist_directory=persist_directory, anonymized_telemetry=False
    )

    client = chromadb.PersistentClient(settings=settings, path=persist_directory)
    return client


def get_collection_settings(collection_name):
    if config.settings.get("collection") is None:
        return None
    for collection in config.settings["collection"]:
        if collection["name"] == collection_name:
            return collection
    return None


def get_documents(db, collection_name):
    collection_settings = get_collection_settings(collection_name)
    if collection_settings is None:
        return []
    documents_loader_list = collection_settings.get("source")

    # all_documents = []
    results = []
    for loader in documents_loader_list:
        if loader["type"] == "directory":
            loader = CustomDirectoryLoader(
                directory_path=loader["value"],
                glob_pattern=".+\.(pdf|txt|doc|docx|md|csv|tsv|eml|html|xml|epub|json)$",
                mode="single",
                db=db,
                collection_name=collection_name,
            )  # elements
        result = loader.load()
        results.append(result)
    return results


def make_embeddings(collection_name):
    db = get_db(collection_name)
    results = get_documents(db, collection_name)

    total_tokens = 0
    total_docs = 0
    for result in results:
        total_docs += result["count"]
        total_tokens += result["total_tokens"]
    return f"Loaded {total_docs} documents. Total tokens: {total_tokens} Total cost: ${total_tokens / 1000000 * 0.02}"


def get_db(collection_name):
    embedder = get_embeddings(collection_name)
    client = get_client()

    # Ensure we pass an object with .embed_documents/.embed_query to Chroma.
    # Wrap the embedder in EmbeddingsAdapter so Chroma can call .embed_documents(...)
    embedding_adapter = EmbeddingsAdapter(embedder)

    db = Chroma(
        embedding_function=embedding_adapter,
        client=client,
        collection_name=collection_name,
    )

    return db

def query(
    query,
    k=10,
    collection_name="default",
    useai=False,
    model="gemini-flash-latest",
    modeai="simple",
    max_iterations: int = 3,
    search_mode="similarity",
):
    db = get_db(collection_name)
    # print(f'There are {db._collection.count()} in the collection')

    if search_mode == "keyword":
        fts = FTS5Manager()
        tokenizer = get_tokenizer()
        # Enforce collection_name is provided by passing through; FTS5Manager will raise if missing
        bm25_results = fts.search_bm25(query, tokenizer, collection_name, top_n=k)
        # chroma_id を使って ChromaDB からメタデータを取得
        chroma_ids = [doc_id for doc_id, _, _ in bm25_results]
        if chroma_ids:
            chroma_data = db._collection.get(
                ids=chroma_ids, include=["metadatas", "documents"]
            )
            id_to_metadata = dict(zip(chroma_data["ids"], chroma_data["metadatas"]))
            id_to_document = dict(zip(chroma_data["ids"], chroma_data["documents"]))
        else:
            id_to_metadata = {}
            id_to_document = {}
        # メタデータを含めた Document を作成
        docs = []
        for doc_id, score, text in bm25_results:
            metadata = id_to_metadata.get(doc_id, {}).copy()
            metadata["chroma_id"] = doc_id
            metadata["search_method"] = "keyword"
            metadata["bm25_score"] = score
            # ChromaDB に保存されている元のテキストを使用（FTS の方が信頼性が高い場合は text を使用）
            page_content = id_to_document.get(doc_id, text)
            docs.append(Document(page_content=page_content, metadata=metadata))
    elif search_mode == "hybrid":
        fts = FTS5Manager()
        tokenizer = get_tokenizer()
        reranker = ReRanker()
        hybrid = HybridSearch(db, fts, tokenizer, reranker, collection_name)
        docs = [
            doc
            for _, _, doc in hybrid.search(query, top_n=k)
        ]
    else:
        # 既存の類似度検索
        docs = db.similarity_search(query, k=k)

    if useai:
        if modeai == "simple":
            summary = ai.get_summary(query=query, docs=docs, model=model)
        elif modeai == "deep":
            # Deep Research Agent mode
            from src.deep_research.graph import run_deep_research

            summary = run_deep_research(
                query=query,
                collection_name=collection_name,
                max_iterations=max_iterations,
            )
        else:
            summary = ""
    else:
        summary = ""
    return docs, summary


def make_index(collection_name):
    db = get_db(collection_name)
    collection_settings = get_collection_settings(collection_name)
    if collection_settings is None:
        return []
    documents_loader_list = collection_settings.get("source")
    results = []
    for loader in documents_loader_list:
        if loader["type"] == "directory":
            fileIndex = FileIndex(
                db=db,
                directory_path=loader["value"],
                glob_pattern=".+\.(pdf|txt|doc|docx|xls|xlsx|ppt|pptx|md|csv|tsv|eml|html|xml|epub|json)$",
                collection_name=collection_name,
            )
            allFIles = fileIndex.initializeFileTable()
            if allFIles:
                results.extend(allFIles)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="", help="query for search")
    parser.add_argument(
        "-e",
        "--embeddings",
        metavar="COLLECTION_NAME",
        help="create embeddings for specified collection",
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="list all collections"
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=10,
        help="number of results to return (default: 10)",
    )
    parser.add_argument(
        "-c",
        "--collection",
        default="publication",
        help="collection name (default: publication)",
    )
    parser.add_argument("--useai", action="store_true", help="enable AI summarization")
    parser.add_argument(
        "--model",
        default="gemini-flash-latest",
        help="AI model to use (default: gemini-flash-latest)",
    )
    parser.add_argument(
        "--modeai",
        choices=["simple", "deep"],
        default="simple",
        help="AI mode: simple or deep (default: simple)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="max iterations for deep mode (default: 3)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["similarity", "keyword", "hybrid"],
        default="hybrid",
        help="search mode: similarity, keyword, or hybrid (default: hybrid)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output results as JSON",
    )
    args = parser.parse_args()

    if args.embeddings:
        res = make_embeddings(args.embeddings)
        print(res)
    elif args.list:
        client = get_client()
        collections = client.list_collections()
        print("Collections:")
        for collection in collections:
            print(f"- {collection.name} (count: {collection.count()})")
    elif args.query:
        docs, summary = query(
            query=args.query,
            k=args.k,
            collection_name=args.collection,
            useai=args.useai,
            model=args.model,
            modeai=args.modeai,
            max_iterations=args.max_iterations,
            search_mode=args.search_mode,
        )
        if args.json:
            import json
            output = {
                "query": args.query,
                "collection": args.collection,
                "results": [
                    {
                        "metadata": doc.metadata,
                        "content": doc.page_content,
                    }
                    for doc in docs
                ],
                "summary": summary if summary else None,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            for doc in docs:
                print("-------------------")
                print(doc.metadata)
                print(doc.page_content)
            if summary:
                print("Summary:")
                print(summary)
    else:
        parser.print_help()
