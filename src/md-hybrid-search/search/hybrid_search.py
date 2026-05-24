"""Hybrid search combining Chroma similarity and FTS5 BM25."""
from __future__ import annotations

from collections import defaultdict
from langchain_core.documents import Document


class HybridSearch:
    def __init__(self, chroma_db, fts_manager, tokenizer, reranker, collection_name: str, rrf_k: int = 60, rerank_pool_size: int | None = None):
        """Hybrid search combining Chroma similarity and FTS BM25 for a specific collection.

        collection_name is required — FTS searches are scoped to this collection.
        rrf_k: parameter used for Reciprocal Rank Fusion scoring.
        rerank_pool_size: if provided, limit the number of candidates sent to the reranker.
        """
        self.chroma_db = chroma_db
        self.fts = fts_manager
        self.tokenizer = tokenizer
        self.reranker = reranker
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.rerank_pool_size = rerank_pool_size

    def _doc_id(self, doc: Document):
        return doc.metadata.get("id") or doc.metadata.get("chroma_id") or doc.metadata.get("source")

    def _rrf_score(self, rank: int) -> float:
        # rank is 1-based
        return 1.0 / (self.rrf_k + rank)

    def _fetch_missing_docs(self, ids: list[str]) -> dict[str, Document]:
        if not ids:
            return {}

        extra = self.chroma_db.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        fetched: dict[str, Document] = {}
        for i, doc_id in enumerate(extra["ids"]):
            fetched[doc_id] = Document(
                page_content=extra["documents"][i],
                metadata={**(extra["metadatas"][i] or {})},
            )
        return fetched

    def search(self, query: str, top_n: int = 5):
        # retrieve a bit wider than final top_n
        fts_top_n = int(top_n * 2.5)
        chroma_top_n = fts_top_n
        rerank_pool_size = self.rerank_pool_size or max(top_n * 8, 30)

        # -------------------------
        # 1) Dense retrieval
        # -------------------------
        chroma_docs = self.chroma_db.similarity_search(query, k=chroma_top_n)

        doc_store: dict[str, Document] = {}
        dense_rank: dict[str, int] = {}
        dense_ids: set[str] = set()

        for rank, doc in enumerate(chroma_docs, start=1):
            doc_id = self._doc_id(doc)
            if not doc_id:
                continue
            dense_ids.add(doc_id)
            dense_rank[doc_id] = rank
            doc_store[doc_id] = doc

        # -------------------------
        # 2) Sparse retrieval
        # fts_results: list[(doc_id, bm25_score, ...)]
        # -------------------------
        fts_results = self.fts.search_bm25(
            query, self.tokenizer, self.collection_name, top_n=fts_top_n
        )

        sparse_rank: dict[str, int] = {}
        sparse_ids: set[str] = set()

        for rank, (doc_id, *_rest) in enumerate(fts_results, start=1):
            if not doc_id:
                continue
            sparse_ids.add(doc_id)
            sparse_rank[doc_id] = rank

        # fetch docs that appeared only in FTS
        missing_ids = [doc_id for doc_id in sparse_ids if doc_id not in doc_store]
        doc_store.update(self._fetch_missing_docs(missing_ids))

        if not doc_store:
            return []

        # -------------------------
        # 3) RRF fusion
        # -------------------------
        rrf_scores: dict[str, float] = defaultdict(float)

        for doc_id, rank in dense_rank.items():
            rrf_scores[doc_id] += self._rrf_score(rank)

        for doc_id, rank in sparse_rank.items():
            rrf_scores[doc_id] += self._rrf_score(rank)

        fused_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )

        # keep only docs we can actually rerank
        candidate_ids = [doc_id for doc_id in fused_ids if doc_id in doc_store][:rerank_pool_size]
        candidate_docs = [doc_store[doc_id] for doc_id in candidate_ids]

        if not candidate_docs:
            return []

        # -------------------------
        # 4) CrossEncoder rerank
        # -------------------------
        reranked = self.reranker.rerank(
            query,
            candidate_docs,
            top_k=top_n,
        )

        # -------------------------
        # 5) Final output
        # -------------------------
        results: list[tuple[str, float, Document]] = []

        for doc, score in reranked:
            doc_id = self._doc_id(doc)
            if not doc_id:
                continue

            # doc.metadata["rrf_score"] = float(rrf_scores.get(doc_id, 0.0))
            reranker_score = float(score)
            # doc.metadata["reranker_score"] = reranker_score
            # doc.metadata["dense_rank"] = dense_rank.get(doc_id)
            # doc.metadata["sparse_rank"] = sparse_rank.get(doc_id)

            if doc_id in dense_ids and doc_id in sparse_ids:
                doc.metadata["search_method"] = "hybrid"
            elif doc_id in dense_ids:
                doc.metadata["search_method"] = "chroma"
            elif doc_id in sparse_ids:
                doc.metadata["search_method"] = "keyword"
            else:
                doc.metadata["search_method"] = "unknown"

            results.append((doc_id, reranker_score, doc))

        return results