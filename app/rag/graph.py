"""LangGraph state machine assembly — the adaptive RAG pipeline."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.rag.nodes import RAGNodes
from app.rag.retrieval import HybridRetriever
from app.rag.state import GraphState
from app.rag.tags import TagPredictor
from app.utils.logger import logger


def build_rag_graph(
    settings: Settings,
    retriever: HybridRetriever,
    tag_predictor: TagPredictor,
) -> Any:
    """Build and compile the LangGraph adaptive RAG state machine.

    Graph topology:
        START → query_analysis → retrieve → grade_documents
          grade_documents ──relevant──→ generate → hallucination_check
          grade_documents ──irrelevant──→ query_analysis (increment attempt)
          hallucination_check ──grounded──→ END
          hallucination_check ──not grounded──→ retrieve (increment attempt)
    """
    nodes = RAGNodes(settings, retriever, tag_predictor)

    def increment_attempt(state: GraphState) -> dict:
        """Increment the attempt counter (called on loop-back edges)."""
        attempt = state.get("attempt", 0)
        return {"attempt": attempt + 1}

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("query_analysis", nodes.query_analysis)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("hallucination_check", nodes.hallucination_check)
    graph.add_node("increment_retry", increment_attempt)
    graph.add_node("increment_rewrite", increment_attempt)

    # Set entry point
    graph.set_entry_point("query_analysis")

    # query_analysis → retrieve
    graph.add_edge("query_analysis", "retrieve")

    # retrieve → grade_documents
    graph.add_edge("retrieve", "grade_documents")

    # grade_documents → conditional: generate or rewrite (back to query_analysis)
    def route_after_grading(state: GraphState) -> str:
        if state.get("docs_relevant", False):
            return "generate"
        return "rewrite"

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "increment_rewrite",
        },
    )
    graph.add_edge("increment_rewrite", "query_analysis")

    # generate → hallucination_check
    graph.add_edge("generate", "hallucination_check")

    # hallucination_check → conditional: end or retry
    def route_after_hallucination(state: GraphState) -> str:
        if state.get("answer_grounded", False):
            return "done"
        attempt = state.get("attempt", 0)
        max_attempts = state.get("max_attempts", settings.max_attempts)
        if attempt >= max_attempts:
            return "done"
        return "retry"

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            "done": END,
            "retry": "increment_retry",
        },
    )
    graph.add_edge("increment_retry", "retrieve")

    # Compile
    app_graph = graph.compile()
    logger.info("LangGraph RAG pipeline compiled")
    return app_graph


def run_rag_pipeline(
    graph: Any,
    question: str,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Execute the full RAG pipeline for a single question.

    Returns final state dict with answer, sources, trace.
    """
    initial_state: GraphState = {
        "question": question,
        "tags": [],
        "rewritten_question": "",
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": "",
        "answer_grounded": False,
        "attempt": 0,
        "max_attempts": max_attempts,
        "trace": [],
    }

    # Recursion limit high enough for max_attempts * (rewrite + retrieve + grade + generate + check)
    config = {"recursion_limit": 30}

    final_state = graph.invoke(initial_state, config=config)
    return final_state
