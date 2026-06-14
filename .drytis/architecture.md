# Architecture

## Directory Structure
```
/workspace/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # Settings (Pydantic Settings)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # API route handlers
│   │   └── schemas.py           # Pydantic request/response models
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph orchestration (state machine)
│   │   ├── state.py             # TypedDict state definitions
│   │   ├── nodes.py             # Graph nodes (query, retrieve, grade, generate)
│   │   ├── retrieval.py         # Hybrid retriever (FAISS + BM25 + RRF)
│   │   ├── embeddings.py        # OpenAI embedding client + FAISS store
│   │   ├── bm25.py              # BM25 sparse retriever
│   │   └── tags.py              # Tag prediction + metadata filtering
│   ├── data/
│   │   ├── __init__.py
│   │   ├── processor.py         # CSV parsing, HTML cleaning, per-answer chunking
│   │   └── indexer.py           # Build FAISS index + BM25 store from processed data
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging configuration
├── scripts/
│   └── build_index.py           # CLI: build index from CSVs
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_data.py             # Data processing tests
│   ├── test_retrieval.py        # Hybrid retrieval tests
│   ├── test_graph.py            # LangGraph node tests
│   └── test_api.py              # API endpoint tests
├── data/                        # Processed data + FAISS index (gitignored)
│   └── faiss_index/
├── static/
│   └── index.html               # Web UI
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

## Data Flow
1. **Index Build** (offline): CSVs → parse → clean HTML → chunk per-answer →
   embed → FAISS index + BM25 corpus → persist to disk
2. **Query Pipeline** (runtime via LangGraph):
   - Query Node: LLM predicts tags → metadata filter
   - Retrieve Node: BM25 top-K + FAISS top-K → RRF fusion → top-K candidates
   - Grade Node: LLM grades each doc relevant/irrelevant → loop back if all irrelevant
   - Generate Node: LLM synthesizes answer from graded context
   - Hallucination Check: LLM verifies answer is grounded in context → loop back if fails

## Key Data Structures
- **Document**: `{id, question_id, question_title, question_body, answer_body,
  tags, score, source}`
- **GraphState**: `{question, tags, retrieved_docs, graded_docs, generation,
  attempt, max_attempts}`

## Infrastructure
- **Background Service**: Uvicorn serving FastAPI on port 8000
- **Caddy Proxy**: Reverse proxy at `/` → port 8000
- **Env**: OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, EMBEDDING_MODEL, etc.
