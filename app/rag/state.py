"""LangGraph state definitions."""

from __future__ import annotations

from typing import TypedDict

from app.data.processor import QADocument
from app.rag.retrieval import RetrievalResult


class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline."""

    # Input
    question: str

    # Query analysis
    tags: list[str]
    rewritten_question: str

    # Retrieval
    retrieved_docs: list[RetrievalResult]

    # Grading
    graded_docs: list[RetrievalResult]
    docs_relevant: bool

    # Generation
    generation: str
    answer_grounded: bool

    # Trace / control
    attempt: int
    max_attempts: int
    trace: list[str]
