import hashlib
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class Retriever:
    """
    Thin wrapper around a LangChain vector store that performs similarity
    search for the RAG query pipeline, with content-based deduplication.

    Built on top of the vector store's native LangChain API
    (`similarity_search_with_score` / `max_marginal_relevance_search`)
    rather than a custom retrieval implementation, so it stays compatible
    with any LangChain-compatible vector store (Chroma, FAISS, etc.).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 4,
        search_type: str = "similarity",
    ):
        """
        Args:
            vector_store: Any LangChain VectorStore instance (e.g. Chroma,
                           from Embedding.sbert_embeder() or
                           Embedding.load_vector_store()).
            k (int): Default number of chunks to retrieve per query.
            search_type (str): "similarity" or "mmr" (max marginal relevance,
                                which reduces redundancy among results).
        """
        if search_type not in ("similarity", "mmr"):
            raise ValueError(f"Unsupported search_type: {search_type!r}")
        self.vector_store = vector_store
        self.k = k
        self.search_type = search_type

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

    def search(
        self, query: str, k: Optional[int] = None, dedup_fetch_multiplier: int = 2
    ) -> List[Tuple[Document, Optional[float]]]:
        """
        Retrieves the most relevant chunks for a query, along with a
        similarity distance score for each (lower = more similar).
        Filters out duplicate chunks automatically.

        Note: if more than half the candidates are duplicates, fewer than
        `k` results may be returned. Increase `dedup_fetch_multiplier` if
        this happens frequently for your corpus.

        Args:
            query (str): The natural language search query.
            k (int, optional): Override the default number of results.
            dedup_fetch_multiplier (int): How many extra candidates to fetch
                (as a multiple of k) to compensate for duplicates removed
                during deduplication.

        Returns:
            list[tuple[Document, float | None]]: (chunk, score) pairs.
                Scores are None for MMR search, which doesn't return them.
        """
        k = k or self.k
        if k <= 0:
            return []

        fetch_k = k * dedup_fetch_multiplier

        if self.search_type == "mmr":
            docs = self.vector_store.max_marginal_relevance_search(query, k=fetch_k)
            results: List[Tuple[Document, Optional[float]]] = [
                (doc, None) for doc in docs
            ]
        else:
            results = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        unique_results = self._deduplicate(results)
        return unique_results[:k]