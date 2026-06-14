"""Tests for API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked pipeline."""
    # We need to bypass lifespan to avoid building the real index
    from fastapi import FastAPI

    from app.api.routes import router, set_globals

    app = FastAPI()
    app.include_router(router, tags=["api"])

    # Mock the pipeline globals
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "generation": "Use .items() to iterate over a dict.",
        "tags": ["python", "dictionary"],
        "graded_docs": [
            MagicMock(
                document=MagicMock(
                    question_id=1,
                    question_title="How to iterate dict",
                    answer_id=101,
                    score=40,
                    tags=["python", "dictionary"],
                    answer_body="Use items() to iterate.",
                )
            )
        ],
        "trace": ["query_analysis", "retrieve", "grade_documents", "generate"],
        "answer_grounded": True,
    }

    mock_retriever = MagicMock()
    mock_retriever.faiss_store.size = 100

    set_globals(mock_retriever, mock_graph, MagicMock())

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["index_size"] == 100
        assert "embedding_model" in data
        assert "llm_model" in data


class TestAskEndpoint:
    def test_ask_returns_answer(self, client):
        resp = client.post("/ask", json={"question": "How to iterate a dict?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_ask_returns_sources(self, client):
        resp = client.post("/ask", json={"question": "How to iterate a dict?"})
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_ask_returns_tags(self, client):
        resp = client.post("/ask", json={"question": "How to iterate a dict?"})
        data = resp.json()
        assert "tags" in data
        assert "python" in data["tags"]

    def test_ask_returns_trace(self, client):
        resp = client.post("/ask", json={"question": "How to iterate a dict?"})
        data = resp.json()
        assert "trace" in data
        assert len(data["trace"]) > 0

    def test_ask_returns_grounded_flag(self, client):
        resp = client.post("/ask", json={"question": "How to iterate a dict?"})
        data = resp.json()
        assert "grounded" in data
        assert data["grounded"] is True

    def test_ask_empty_question_returns_422(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422


class TestUIEndpoint:
    def test_ui_returns_html(self, client):
        resp = client.get("/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Python Q&A" in resp.text
