"""Hybrid retrieval: FAISS dense + BM25 sparse + Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.config import Settings
from app.data.processor import QADocument
from app.rag.bm25 import BM25Store
from app.rag.embeddings import EmbeddingClient, FAISSStore
from app.utils.logger import logger


@dataclass
class RetrievalResult:
    """A single retrieval hit with fused score and per-branch provenance."""

    document: QADocument
    fused_score: float
    dense_score: float
    sparse_score: float


class HybridRetriever:
    """Combines dense (FAISS) and sparse (BM25) via Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_store: BM25Store,
        embedder: EmbeddingClient,
        settings: Settings,
    ):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.settings = settings

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        tag_filter: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval and return fused top-k results.

        Args:
            query: User query text.
            k: Final number of results (defaults to settings.final_k).
            tag_filter: If provided, only return documents matching any of these tags.
        """
        if k is None:
            k = self.settings.final_k

        branch_k = self.settings.retrieval_k

        # --- Dense (FAISS) ---
        query_vec = self.embedder.embed_query(query)
        dense_hits = self.faiss_store.search(query_vec, k=branch_k)
        logger.info("Dense retrieval: %d hits", len(dense_hits))

        # --- Sparse (BM25) ---
        sparse_hits = self.bm25_store.search(query, k=branch_k)
        logger.info("Sparse retrieval: %d hits", len(sparse_hits))

        # --- Reciprocal Rank Fusion ---
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        logger.info("Fused: %d unique documents", len(fused))

        # --- Tag filtering (post-fusion) ---
        if tag_filter:
            tag_set = {t.lower() for t in tag_filter}
            filtered = [
                r for r in fused
                if any(t.lower() in tag_set for t in r.document.tags)
            ]
            # If filtering removes everything, fall back to unfiltered
            if filtered:
                logger.info(
                    "Tag filter (%s): %d → %d results",
                    tag_filter, len(fused), len(filtered),
                )
                fused = filtered
            else:
                logger.warning(
                    "Tag filter (%s) removed all results — falling back to unfiltered",
                    tag_filter,
                )

        # Sort by fused score and return top-k
        fused.sort(key=lambda r: r.fused_score, reverse=True)
        return fused[:k]


def reciprocal_rank_fusion(
    dense_hits: list[tuple[QADocument, float]],
    sparse_hits: list[tuple[QADocument, float]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse two ranked lists via Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank)) for each list where the doc appears.
    """
    doc_scores: dict[str, dict] = defaultdict(
        lambda: {"doc": None, "dense": 0.0, "sparse": 0.0, "fused": 0.0}
    )

    # Dense ranks (1-based)
    for rank, (doc, score) in enumerate(dense_hits, 1):
        key = doc.doc_id
        doc_scores[key]["doc"] = doc
        doc_scores[key]["dense"] = score
        doc_scores[key]["fused"] += 1.0 / (rrf_k + rank)

    # Sparse ranks
    for rank, (doc, score) in enumerate(sparse_hits, 1):
        key = doc.doc_id
        if doc_scores[key]["doc"] is None:
            doc_scores[key]["doc"] = doc
        doc_scores[key]["sparse"] = score
        doc_scores[key]["fused"] += 1.0 / (rrf_k + rank)

    results = [
        RetrievalResult(
            document=info["doc"],
            fused_score=info["fused"],
            dense_score=info["dense"],
            sparse_score=info["sparse"],
        )
        for info in doc_scores.values()
        if info["doc"] is not None
    ]
    return results
