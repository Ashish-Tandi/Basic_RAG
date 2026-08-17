import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class Retriever:
    """
    Thin wrapper around a LangChain vector store that performs similarity
    search for the RAG query pipeline, with content-based deduplication.

    search_type:
      - "similarity": dense vector similarity search.
      - "mmr": max marginal relevance (diversity-aware dense search).
      - "hybrid": dense + BM25 sparse search, fused with Reciprocal Rank
        Fusion (RRF). Dense catches semantic/paraphrase matches, BM25
        catches exact keywords/IDs embeddings tend to blur — fusing both
        beats either alone on mixed query workloads.

    Built on the vector store's native LangChain API
    (`similarity_search_with_score` / `max_marginal_relevance_search`) plus
    LangChain's `BM25Retriever` for the sparse side, so it stays compatible
    with any LangChain vector store. Hybrid mode needs the raw document
    corpus (`documents=`) since vector stores don't reliably expose it.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 4,
        search_type: str = "similarity",
        documents: Optional[Sequence[Document]] = None,
        rrf_k: int = 60,
    ):
        if search_type not in ("similarity", "mmr", "hybrid"):
            raise ValueError(f"Unsupported search_type: {search_type!r}")
        if search_type == "hybrid" and not documents:
            raise ValueError(
                "search_type='hybrid' requires `documents` (the full chunk "
                "corpus) to build the BM25 sparse index."
            )

        self.vector_store = vector_store
        self.k = k
        self.search_type = search_type
        self.rrf_k = rrf_k
        self.bm25_retriever = (
            BM25Retriever.from_documents(list(documents))
            if search_type == "hybrid"
            else None
        )

    @staticmethod
    def _content_hash(doc: Document) -> str:
        """Extract a stable hash of a document's text content for dedup."""
        content = getattr(doc, "page_content", str(doc))
        return hashlib.md5(content.strip().encode("utf-8")).hexdigest()

    @classmethod
    def _deduplicate(
        cls, results: List[Tuple[Document, Optional[float]]]
    ) -> List[Tuple[Document, Optional[float]]]:
        """Removes duplicate chunks based on content hash, preserving order."""
        seen_hashes = set()
        unique_results = []
        for doc, score in results:
            content_hash = cls._content_hash(doc)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append((doc, score))
        return unique_results

    def _rrf_fuse(
        self,
        dense_results: List[Tuple[Document, Optional[float]]],
        sparse_docs: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Fuse dense + BM25 ranked lists via Reciprocal Rank Fusion: each doc
        scores sum(1 / (rrf_k + rank)) over every list it appears in. Rank
        position is used instead of raw scores since vector distances and
        BM25 scores aren't on comparable scales. A doc found by both lists
        accumulates both contributions and floats to the top.
        """
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}
        for doc_list in (
            [doc for doc, _ in dense_results],
            sparse_docs,
        ):
            for rank, doc in enumerate(doc_list, start=1):
                h = self._content_hash(doc)
                rrf_scores[h] = rrf_scores.get(h, 0.0) + 1.0 / (self.rrf_k + rank)
                doc_lookup.setdefault(h, doc)

        ranked_hashes = sorted(rrf_scores, key=lambda h: rrf_scores[h], reverse=True)
        return [(doc_lookup[h], rrf_scores[h]) for h in ranked_hashes]

    def search(
        self, query: str, k: Optional[int] = None, dedup_fetch_multiplier: int = 2
    ) -> List[Tuple[Document, Optional[float]]]:
        """
        Retrieves the most relevant chunks for a query, deduplicated.
        Score meaning depends on search_type: similarity distance (lower =
        better) for "similarity", None for "mmr", fused RRF score (higher =
        better) for "hybrid".
        """
        k = k or self.k
        if k <= 0:
            return []

        fetch_k = k * dedup_fetch_multiplier

        if self.search_type == "hybrid":
            dense_results = self.vector_store.similarity_search_with_score(
                query, k=fetch_k
            )
            self.bm25_retriever.k = fetch_k
            sparse_docs = self.bm25_retriever.invoke(query)
            fused = self._rrf_fuse(dense_results, sparse_docs)
            return self._deduplicate(fused)[:k]

        if self.search_type == "mmr":
            docs = self.vector_store.max_marginal_relevance_search(query, k=fetch_k)
            results: List[Tuple[Document, Optional[float]]] = [
                (doc, None) for doc in docs
            ]
        else:
            results = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        return self._deduplicate(results)[:k]