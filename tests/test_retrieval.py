"""Tests for hybrid retrieval (BM25, FAISS, RRF fusion)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.data.processor import QADocument
from app.rag.bm25 import BM25Store, tokenize
from app.rag.retrieval import HybridRetriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("How to iterate over a dict")
        assert "how" in tokens
        assert "iterate" in tokens
        assert "dict" in tokens

    def test_preserves_underscored_tokens(self):
        tokens = tokenize("my_variable_name function_name")
        assert "my_variable_name" in tokens
        assert "function_name" in tokens

    def test_filters_short_tokens(self):
        tokens = tokenize("a b c de fg")
        assert "de" in tokens
        assert "fg" in tokens
        # single-char tokens filtered out
        assert "a" not in tokens

    def test_preserves_numbers(self):
        tokens = tokenize("python3 version 2.7")
        # Numbers should appear
        assert any("python" in t for t in tokens)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class TestBM25Store:
    @pytest.fixture
    def bm25_store(self, mock_settings):
        return BM25Store(mock_settings)

    @pytest.fixture
    def docs_with_text(self):
        return [
            QADocument(
                doc_id="a1", question_id=1, answer_id=1,
                question_title="How to iterate dict",
                question_body="iterate over dictionary",
                answer_body="Use items() to iterate over a dictionary",
                tags=["python", "dictionary"],
            ),
            QADocument(
                doc_id="a2", question_id=2, answer_id=2,
                question_title="How to sort a list",
                question_body="sorting algorithm",
                answer_body="Use sorted() to sort a list in Python",
                tags=["python", "list"],
            ),
            QADocument(
                doc_id="a3", question_id=3, answer_id=3,
                question_title="How to parse JSON",
                question_body="json parsing",
                answer_body="Use json.loads() to parse JSON data",
                tags=["python", "json"],
            ),
        ]

    def test_build_and_search(self, bm25_store, docs_with_text):
        bm25_store.build(docs_with_text)
        results = bm25_store.search("iterate dictionary", k=3)
        assert len(results) > 0
        # The dict question should rank high
        assert results[0][0].doc_id == "a1"

    def test_search_returns_scores(self, bm25_store, docs_with_text):
        bm25_store.build(docs_with_text)
        results = bm25_store.search("sort list", k=2)
        assert len(results) > 0
        assert isinstance(results[0][1], float)

    def test_empty_query_returns_empty(self, bm25_store, docs_with_text):
        bm25_store.build(docs_with_text)
        results = bm25_store.search("??? !!!", k=3)
        assert results == []

    def test_save_and_load(self, bm25_store, docs_with_text, mock_settings):
        bm25_store.build(docs_with_text)
        bm25_store.save()

        # Load into a new store
        store2 = BM25Store(mock_settings)
        loaded = store2.load()
        assert loaded is True
        assert len(store2.documents) == 3
        results = store2.search("iterate dictionary", k=3)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRF:
    @pytest.fixture
    def docs(self):
        return [
            QADocument(doc_id="d1", question_id=1, answer_id=1,
                       question_title="T1", question_body="B1", answer_body="A1"),
            QADocument(doc_id="d2", question_id=2, answer_id=2,
                       question_title="T2", question_body="B2", answer_body="A2"),
            QADocument(doc_id="d3", question_id=3, answer_id=3,
                       question_title="T3", question_body="B3", answer_body="A3"),
        ]

    def test_fuses_two_lists(self, docs):
        dense = [(docs[0], 0.9), (docs[1], 0.7)]
        sparse = [(docs[1], 5.2), (docs[2], 3.1)]
        results = reciprocal_rank_fusion(dense, sparse)
        assert len(results) == 3

    def test_doc_in_both_lists_ranks_higher(self, docs):
        """A doc that appears in both dense and sparse should get a higher fused score."""
        dense = [(docs[0], 0.9), (docs[1], 0.7)]
        sparse = [(docs[1], 5.2), (docs[2], 3.1)]
        results = reciprocal_rank_fusion(dense, sparse)
        # docs[1] is in both lists — should have higher fused score
        sorted_results = sorted(results, key=lambda r: r.fused_score, reverse=True)
        assert sorted_results[0].document.doc_id == "d2"

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_single_list(self, docs):
        dense = [(docs[0], 0.9)]
        results = reciprocal_rank_fusion(dense, [])
        assert len(results) == 1
        assert results[0].document.doc_id == "d1"


# ---------------------------------------------------------------------------
# Hybrid Retriever (with mocked embedder + stores)
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    @pytest.fixture
    def retriever(self):
        settings = MagicMock()
        settings.retrieval_k = 5
        settings.final_k = 3

        faiss_store = MagicMock()
        bm25_store = MagicMock()
        embedder = MagicMock()
        embedder.embed_query.return_value = np.zeros(8, dtype=np.float32)

        return HybridRetriever(faiss_store, bm25_store, embedder, settings)

    def test_retrieve_combines_dense_and_sparse(self, retriever):
        doc1 = QADocument(doc_id="d1", question_id=1, answer_id=1,
                          question_title="T1", question_body="B1",
                          answer_body="A1", tags=["python", "django"])
        doc2 = QADocument(doc_id="d2", question_id=2, answer_id=2,
                          question_title="T2", question_body="B2",
                          answer_body="A2", tags=["python", "flask"])

        retriever.faiss_store.search.return_value = [(doc1, 0.9), (doc2, 0.7)]
        retriever.bm25_store.search.return_value = [(doc2, 5.0), (doc1, 3.0)]

        results = retriever.retrieve("test query")
        assert len(results) > 0
        assert len(results) <= 3  # final_k

    def test_tag_filter_works(self, retriever):
        """Tag filter should remove non-matching docs."""
        doc1 = QADocument(doc_id="d1", question_id=1, answer_id=1,
                          question_title="T1", question_body="B1",
                          answer_body="A1", tags=["python", "django"])
        doc2 = QADocument(doc_id="d2", question_id=2, answer_id=2,
                          question_title="T2", question_body="B2",
                          answer_body="A2", tags=["python", "flask"])

        retriever.faiss_store.search.return_value = [(doc1, 0.9), (doc2, 0.7)]
        retriever.bm25_store.search.return_value = [(doc1, 5.0), (doc2, 3.0)]

        results = retriever.retrieve("test", tag_filter=["django"])
        assert all(r.document.doc_id == "d1" for r in results)

    def test_tag_filter_fallback_when_empty(self, retriever):
        """If tag filter removes everything, fall back to unfiltered."""
        doc1 = QADocument(doc_id="d1", question_id=1, answer_id=1,
                          question_title="T1", question_body="B1",
                          answer_body="A1", tags=["python", "django"])

        retriever.faiss_store.search.return_value = [(doc1, 0.9)]
        retriever.bm25_store.search.return_value = [(doc1, 5.0)]

        # django tags but we filter for "numpy" — no match, should fall back
        results = retriever.retrieve("test", tag_filter=["numpy"])
        assert len(results) > 0  # fell back to unfiltered
