"""BM25 sparse retriever."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.data.processor import QADocument
from app.utils.logger import logger


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, keep tokens >= 2 chars."""
    tokens = re.findall(r"[a-z0-9_]+|[A-Z][a-z]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


class BM25Store:
    """BM25 sparse retriever built from QADocument corpus."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bm25: BM25Okapi | None = None
        self.documents: list[QADocument] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, documents: list[QADocument]) -> None:
        """Build BM25 index from documents."""
        logger.info("Building BM25 index for %d documents...", len(documents))
        self.documents = documents
        self.tokenized_corpus = [tokenize(doc.text_for_embedding) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built: %d documents", len(documents))

    def search(self, query: str, k: int = 10) -> list[tuple[QADocument, float]]:
        """Return top-k (document, bm25_score) pairs."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built or loaded")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Get top-k indices
        k = min(k, len(scores))
        top_indices = np_topk(scores, k)
        results: list[tuple[QADocument, float]] = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        return results

    def save(self) -> None:
        """Persist BM25 index + documents to disk."""
        path = Path(self.settings.bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "documents": [d.to_dict() for d in self.documents],
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )
        logger.info("Saved BM25 index to %s", path)

    def load(self) -> bool:
        """Load BM25 index from disk. Returns True if loaded."""
        path = Path(self.settings.bm25_path)
        if not path.exists():
            return False
        logger.info("Loading BM25 index from %s ...", path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = [QADocument.from_dict(d) for d in data["documents"]]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index loaded: %d documents", len(self.documents))
        return True


def np_topk(scores, k: int) -> list[int]:
    """Return indices of top-k scores (descending). Pure numpy."""
    import numpy as np

    if k >= len(scores):
        # Sort all descending
        return list(np.argsort(scores)[::-1])
    # argpartition for efficiency, then sort the top-k
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.tolist()
