# Python Q&A Assistant — LangGraph Hybrid RAG Pipeline

An adaptive Retrieval-Augmented Generation (RAG) system that answers Python programming questions using Stack Overflow Q&A data, orchestrated by **LangGraph** with **hybrid retrieval** (FAISS dense + BM25 sparse + Reciprocal Rank Fusion).

## Architecture

```
                    ┌──────────────┐
  User Query ──────►│  Query Node  │── tag prediction + query rewrite
                    └──────┬───────┘
                           ▼
              ┌─────────────────────────┐
              │    Retrieve Node        │
              │  BM25 (top-K) + Dense   │
              │  (top-K) → RRF Fusion   │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │  Grade Docs Node    │── irrelevant? ──► Query Rewrite (loop)
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Generate Answer    │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Hallucination      │── fails? ──► Retrieve (loop)
                │  Check              │
                └──────────┬──────────┘
                           ▼
                      Output
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Orchestration** | LangGraph stateful graph with cycles (adaptive RAG) |
| **Dense Retrieval** | FAISS IndexFlatIP with cosine similarity |
| **Sparse Retrieval** | BM25Okapi (rank-bm25) |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (384-dim, local) |
| **LLM** | z-ai/glm-5 via Drytis OpenAI-compatible gateway |
| **Tag Filtering** | LLM predicts relevant SO tags → metadata pre-filter |
| **Document Grading** | LLM batch-grades retrieved docs as relevant/irrelevant |
| **Hallucination Check** | LLM verifies generated answer is grounded in context |
| **Self-Correction** | Loops back on irrelevant docs or ungrounded answers (max 3) |

## Tech Stack

- **Python 3.13** · **FastAPI + Uvicorn** · **LangGraph** · **FAISS** · **BM25**
- **sentence-transformers** for local embeddings
- **BeautifulSoup** for HTML cleaning
- **Pydantic v2** for validation

## Project Structure

```
app/
├── main.py                  # FastAPI entry point + lifespan
├── config.py                # Pydantic Settings
├── api/
│   ├── routes.py            # API routes (/health, /ask, /ui)
│   └── schemas.py           # Request/response models
├── rag/
│   ├── graph.py             # LangGraph state machine
│   ├── nodes.py             # Graph nodes (query, retrieve, grade, generate)
│   ├── state.py             # GraphState TypedDict
│   ├── retrieval.py         # Hybrid retriever + RRF
│   ├── embeddings.py        # Local embeddings + FAISS store
│   ├── bm25.py              # BM25 sparse retriever
│   └── tags.py              # Tag prediction
├── data/
│   ├── processor.py         # CSV parsing, HTML cleaning, chunking
│   └── indexer.py           # Index builder
└── utils/
    └── logger.py

scripts/
└── build_index.py           # CLI index builder

tests/                       # 53 unit + integration tests
static/
└── index.html               # Web UI (chat interface)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirects to web UI |
| `GET` | `/health` | Health check (status, index size, models) |
| `POST` | `/ask` | Answer a question `{question: str}` → `{answer, sources, tags, trace, grounded}` |
| `GET` | `/ui` | Interactive chat UI |
| `GET` | `/docs` | Swagger API docs |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build the FAISS + BM25 indices from CSV data
python scripts/build_index.py

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Data Format

The system ingests three CSV files:

| File | Columns | Description |
|------|---------|-------------|
| `sampleQuestion.csv` | `Id, OwnerUserId, CreationDate, Score, Title, Body` | Questions (Body is HTML) |
| `sampleAnswer.csv` | `Id, OwnerUserId, CreationDate, ParentId, Score, Body` | Answers (ParentId → Question Id) |
| `Tags.csv` | `Id, Tag` | Question tags (1.8M rows, 16,897 unique tags) |

## RAG Strategy Decisions

1. **Per-answer chunking**: Each answer = 1 retrievable document with question metadata (title, body, tags) attached. Gives granular retrieval.

2. **Hybrid retrieval (dense + sparse + RRF)**: SO Q&A has both code-heavy content (BM25 advantage) and conceptual queries (dense advantage). RRF combines both without needing score calibration.

3. **Tag-based metadata filtering**: The LLM predicts relevant tags from the query, then filters the candidate set. Falls back to unfiltered if the tag filter removes all results.

4. **Adaptive self-correction via LangGraph**: If all retrieved docs are graded irrelevant, the pipeline rewrites the query and retries. If the hallucination check fails, it re-retrieves. Max 3 attempts per loop.

## Tests

```bash
python -m pytest tests/ -v
# 53 tests covering data processing, retrieval (BM25/FAISS/RRF), graph nodes, and API
```
