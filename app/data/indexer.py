"""Index builder: process CSVs → build FAISS + BM25 indices."""

from __future__ import annotations

from app.config import Settings
from app.data.processor import build_documents
from app.rag.bm25 import BM25Store
from app.rag.embeddings import EmbeddingClient, FAISSStore
from app.utils.logger import logger


def build_index(settings: Settings | None = None) -> None:
    """Build both FAISS and BM25 indices from CSV data."""
    if settings is None:
        settings = Settings()
        settings.index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building RAG indices from CSV data")
    logger.info("=" * 60)

    # Step 1: Process CSVs into documents
    documents = build_documents(
        questions_csv=settings.questions_csv,
        answers_csv=settings.answers_csv,
        tags_csv=settings.tags_csv,
    )

    if not documents:
        logger.error("No documents built — check CSV files")
        return

    logger.info("Total documents: %d", len(documents))

    # Step 2: Build FAISS index
    embedder = EmbeddingClient(settings)
    faiss_store = FAISSStore(settings)
    faiss_store.build(documents, embedder)
    faiss_store.save()

    # Step 3: Build BM25 index
    bm25_store = BM25Store(settings)
    bm25_store.build(documents)
    bm25_store.save()

    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info("  FAISS: %s (%d vectors)", settings.faiss_db_path, faiss_store.size)
    logger.info("  BM25:  %s (%d docs)", settings.bm25_path, bm25_store.documents.__len__())
    logger.info("=" * 60)


if __name__ == "__main__":
    build_index()
