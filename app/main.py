"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_globals
from app.config import get_settings
from app.data.indexer import build_index
from app.rag.bm25 import BM25Store
from app.rag.embeddings import EmbeddingClient, FAISSStore
from app.rag.graph import build_rag_graph
from app.rag.retrieval import HybridRetriever
from app.rag.tags import TagPredictor
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load or build indices, initialize RAG pipeline."""
    settings = get_settings()
    logger.info("Starting RAG Q&A Assistant...")
    logger.info("Index path: %s", settings.faiss_index_path)

    # --- Load or build FAISS index ---
    faiss_store = FAISSStore(settings)
    bm25_store = BM25Store(settings)

    if not faiss_store.load():
        logger.warning("FAISS index not found — building from scratch...")
        build_index(settings)
        faiss_store.load()
        bm25_store.load()
    else:
        bm25_store.load()

    logger.info("FAISS: %d vectors | BM25: %d docs",
                faiss_store.size, len(bm25_store.documents))

    # --- Initialize pipeline ---
    embedder = EmbeddingClient(settings)
    retriever = HybridRetriever(faiss_store, bm25_store, embedder, settings)
    tag_predictor = TagPredictor(settings)
    graph = build_rag_graph(settings, retriever, tag_predictor)

    set_globals(retriever, graph, tag_predictor)
    logger.info("RAG pipeline ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Python Q&A Assistant",
    description="Adaptive RAG pipeline with LangGraph + Hybrid Retrieval (FAISS + BM25 + RRF)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, tags=["api"])

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Redirect to the web UI."""
    return RedirectResponse(url="/ui")
