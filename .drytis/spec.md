# Python Q&A Assistant — Project Specification

## Overview
Build a Python Programming Q&A Assistant using an adaptive RAG pipeline grounded
in Stack Overflow Python Q&A data. The system uses **LangGraph** for orchestration,
**hybrid retrieval** (FAISS dense + BM25 sparse with RRF fusion), **tag-based
metadata filtering**, and **self-correcting generation** (document grading +
hallucination checking).

## Tech Stack
- **Language**: Python 3.13
- **Web Framework**: FastAPI + Uvicorn (ASGI)
- **Orchestration**: LangGraph (stateful, cyclic adaptive RAG graph)
- **Retrieval**: Hybrid — FAISS (dense) + BM25 (sparse) + Reciprocal Rank Fusion
- **Embeddings**: OpenAI text-embedding-3-small (via Drytis OpenAI-compatible API)
- **LLM**: OpenAI GPT-4o-mini (via Drytis OpenAI-compatible API)
- **Vector Store**: FAISS (persistent, local)
- **Data Source**: Stack Overflow Python Q&A (provided CSV files)
- **Testing**: Pytest + httpx
- **Deployment**: Drytis (Caddy reverse proxy → Uvicorn)

## Key Decisions
1. **LangGraph over plain LangChain**: Stateful graph with cycles enables adaptive
   RAG — document grading, query rewriting, hallucination checks, and self-correction
   loops.
2. **Hybrid retrieval (FAISS + BM25 + RRF)**: SO Q&A is heavy on code snippets and
   exact error messages (favors BM25) but also conceptual queries (favors dense).
   Hybrid gives best overall recall + precision.
3. **Per-answer chunking**: Each answer is a document with question metadata. Gives
   granular retrieval and natural document units.
4. **Tag-based metadata filtering**: Use LLM to predict relevant tags from the query,
   then pre-filter the candidate set before retrieval. The 16,897-tag taxonomy is a
   precision goldmine for 600K+ docs.
5. **FAISS over ChromaDB**: Lighter, faster, no external server process, ideal for
   local persistent storage. Simpler deployment.
6. **Drytis OpenAI-compatible API**: Key minted via `create_openai_api_key` — no
   external API keys, spend billed to project owner.

## RAG Pipeline Architecture (LangGraph)
```
                    ┌──────────────┐
  User Query ──────►│  Query Node  │── tag prediction + query analysis
                    └──────┬───────┘
                           ▼
              ┌─────────────────────────┐
              │    Retrieve Node        │
              │  BM25 (top-K) + Dense   │
              │  (top-K) → RRF Fusion   │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │  Grade Docs Node    │── all irrelevant? ──► Query Rewrite (loop)
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Generate Answer    │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Hallucination      │── fails? ──► back to Retrieve (loop)
                │  Check              │
                └──────────┬──────────┘
                           ▼
                        Output
```

## API Endpoints
- `GET /health` — Health check (status, index size, config info)
- `POST /ask` — Accept question, return answer + sources + pipeline trace
- `POST /ask/stream` — SSE streaming response
- `GET /` — Web UI (simple chat interface)
- `GET /docs` — Auto-generated Swagger UI

## Acceptance Criteria
- [ ] FastAPI app runs with production Uvicorn server
- [ ] `GET /health` returns 200 with status JSON
- [ ] `POST /ask` accepts a question and returns a grounded answer with sources
- [ ] LangGraph pipeline: query → retrieve → grade → generate → hallucination check
- [ ] Hybrid retrieval: FAISS dense + BM25 sparse + RRF fusion
- [ ] Tag prediction pre-filters candidate set
- [ ] Document grading rejects irrelevant context (loop back if needed)
- [ ] Hallucination check validates answer against context
- [ ] FAISS index is pre-built and persistent
- [ ] Web UI for interactive Q&A
- [ ] Pytest test suite passes (unit + integration)
- [ ] App deployed to public URL via Drytis
- [ ] README with setup instructions and architecture docs
