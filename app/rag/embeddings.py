"""Embedding client + FAISS vector store.

Uses sentence-transformers for local embeddings (no API key needed),
FAISS for dense vector retrieval.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from app.config import Settings
from app.data.processor import QADocument
from app.utils.logger import logger

# Default model — small, fast, good quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingClient:
    """Local embedding model via sentence-transformers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.settings.embedding_model or DEFAULT_MODEL
            logger.info("Loading embedding model: %s", model_name)
            self._model = SentenceTransformer(model_name)
            logger.info("Model loaded: dim=%d", self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts → (N, D) float32 array."""
        logger.info("Embedding %d texts...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query → (D,) float32 array."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return np.array(embedding, dtype=np.float32)


class FAISSStore:
    """Persistent FAISS index for dense retrieval."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index: faiss.IndexFlatIP | None = None  # inner product (cosine on normalized)
        self.documents: list[QADocument] = []

    # ---- Build ----

    def build(self, documents: list[QADocument], embedder: EmbeddingClient) -> None:
        """Build FAISS index from documents."""
        logger.info("Building FAISS index for %d documents...", len(documents))
        texts = [doc.text_for_embedding for doc in documents]
        embeddings = embedder.embed_texts(texts)

        # L2-normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.documents = documents
        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

    # ---- Search ----

    def search(
        self, query_vec: np.ndarray, k: int = 10
    ) -> list[tuple[QADocument, float]]:
        """Return top-k (document, score) pairs."""
        if self.index is None:
            raise RuntimeError("FAISS index not built or loaded")
        vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, min(k, self.index.ntotal))
        results: list[tuple[QADocument, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((self.documents[idx], float(score)))
        return results

    # ---- Persistence ----

    def save(self) -> None:
        """Save FAISS index + document metadata to disk."""
        path = Path(self.settings.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.settings.faiss_db_path))
        with open(self.settings.meta_path, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info("Saved FAISS index to %s", path)

    def load(self) -> bool:
        """Load FAISS index + documents from disk. Returns True if loaded."""
        if not self.settings.index_exists:
            return False
        logger.info("Loading FAISS index from %s ...", self.settings.faiss_index_path)
        self.index = faiss.read_index(str(self.settings.faiss_db_path))
        with open(self.settings.meta_path, "rb") as f:
            self.documents = pickle.load(f)
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)
        return True

    @property
    def size(self) -> int:
        return len(self.documents)
