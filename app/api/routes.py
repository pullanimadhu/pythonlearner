"""API route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceInfo,
)
from app.config import get_settings
from app.rag.graph import run_rag_pipeline
from app.utils.logger import logger

router = APIRouter()

# --- Globals set by main.py lifespan ---
_retriever = None
_graph = None
_tag_predictor = None


def set_globals(retriever, graph, tag_predictor) -> None:
    """Set shared pipeline objects (called by main.py at startup)."""
    global _retriever, _graph, _tag_predictor
    _retriever = retriever
    _graph = graph
    _tag_predictor = tag_predictor


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — returns service status and index info."""
    settings = get_settings()
    index_size = _retriever.faiss_store.size if _retriever else 0
    return HealthResponse(
        status="ok" if _graph is not None else "initializing",
        index_size=index_size,
        embedding_model=settings.embedding_model,
        llm_model=settings.model_name,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a Python question using the RAG pipeline."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    logger.info("Question: %s", req.question)
    try:
        state = run_rag_pipeline(_graph, req.question)
    except RecursionError:
        logger.error("Pipeline recursion limit reached")
        raise HTTPException(status_code=504, detail="Query processing timed out — please try a more specific question")
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail="An error occurred while processing your question. Please try again.")

    # Build source list from graded docs
    sources: list[SourceInfo] = []
    graded = state.get("graded_docs") or state.get("retrieved_docs") or []
    for doc_result in graded[:5]:
        doc = doc_result.document
        sources.append(
            SourceInfo(
                question_id=doc.question_id,
                question_title=doc.question_title,
                answer_id=doc.answer_id,
                score=doc.score,
                tags=doc.tags,
                snippet=doc.answer_body[:300],
            )
        )

    return AskResponse(
        answer=state.get("generation", ""),
        sources=sources,
        tags=state.get("tags", []),
        trace=state.get("trace", []),
        grounded=state.get("answer_grounded", False),
    )


@router.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    """Serve the web UI."""
    from pathlib import Path

    ui_path = Path(__file__).parent.parent.parent / "static" / "index.html"
    return HTMLResponse(ui_path.read_text())
