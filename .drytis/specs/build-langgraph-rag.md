# Task Spec: Build LangGraph Hybrid RAG Q&A Assistant

## Overview
Build a Python Q&A assistant using LangGraph orchestration with hybrid retrieval
(FAISS + BM25 + RRF), tag-based metadata filtering, document grading, and
hallucination checking. Grounded in Stack Overflow Python Q&A CSV data.

## Files to Create

### Config & Web
- `app/__init__.py`, `app/main.py`, `app/config.py`
- `app/api/__init__.py`, `app/api/routes.py`, `app/api/schemas.py`
- `static/index.html` — web UI

### Data Processing
- `app/data/__init__.py`, `app/data/processor.py` — CSV parsing, HTML cleaning, per-answer chunking
- `app/data/indexer.py` — Build FAISS index + BM25 corpus from processed data

### RAG Pipeline
- `app/rag/__init__.py`
- `app/rag/state.py` — TypedDict graph state
- `app/rag/embeddings.py` — OpenAI embedding client + FAISS vector store
- `app/rag/bm25.py` — BM25 sparse retriever
- `app/rag/retrieval.py` — Hybrid retrieval (dense + sparse + RRF fusion)
- `app/rag/tags.py` — Tag prediction + metadata filtering
- `app/rag/nodes.py` — LangGraph nodes (query, retrieve, grade, generate, hallucination check, route)
- `app/rag/graph.py` — LangGraph state machine assembly

### Utils
- `app/utils/__init__.py`, `app/utils/logger.py`

### Scripts
- `scripts/build_index.py` — CLI entry point for index building

### Tests
- `tests/__init__.py`, `tests/conftest.py`
- `tests/test_data.py` — data processing tests
- `tests/test_retrieval.py` — hybrid retrieval tests
- `tests/test_graph.py` — LangGraph node tests
- `tests/test_api.py` — API endpoint tests

### Config
- `.env.example`, `README.md`, `.gitignore`, `pyproject.toml`

## Acceptance Criteria

### Data Processing
- [ ] Parse Tags.csv (1.8M rows), sampleQuestion.csv, sampleAnswer.csv
- [ ] Clean HTML from question/answer bodies (BeautifulSoup → plain text + code blocks)
- [ ] Chunk per-answer: each answer = 1 document with question metadata
- [ ] Build tag → question_id mapping from Tags.csv
- [ ] Each document has: id, question_id, question_title, answer_text, tags, score

### Embeddings & FAISS
- [ ] Embed all documents via OpenAI text-embedding-3-small
- [ ] Persist FAISS index to disk (FAISS_INDEX_PATH)
- [ ] Support loading index at startup (fast cold start)

### BM25 Sparse Retrieval
- [ ] Build BM25Okapi index from document corpus
- [ ] Persist tokenized corpus for fast reload
- [ ] Return top-K documents with BM25 scores

### Hybrid Retrieval (RRF)
- [ ] Query FAISS (top-K) + BM25 (top-K) in parallel
- [ ] Fuse results via Reciprocal Rank Fusion (RRF)
- [ ] Return final top-K merged results

### Tag Prediction + Filter
- [ ] LLM extracts/predicts relevant tags from user query
- [ ] Use predicted tags as metadata pre-filter on candidate set
- [ ] Fall back to unfiltered if no tag match

### LangGraph Orchestration
- [ ] Graph state: question, tags, retrieved_docs, graded_docs, generation, attempt
- [ ] Node: query_analysis → predict tags + rewrite query if needed
- [ ] Node: retrieve → hybrid retrieval with tag filter
- [ ] Node: grade_documents → LLM grades each doc relevant/irrelevant
- [ ] Edge: if all docs irrelevant → query rewrite → re-retrieve (max 3 attempts)
- [ ] Node: generate → LLM synthesizes answer from graded context
- [ ] Node: hallucination_check → verify answer grounded in context
- [ ] Edge: if hallucination check fails → re-retrieve (max 3 attempts)
- [ ] Compile graph with recursion limit

### FastAPI Service
- [ ] `GET /health` → 200 JSON with status, index size
- [ ] `POST /ask` → {question} → {answer, sources, trace}
- [ ] `GET /` → web UI (interactive chat)
- [ ] CORS enabled
- [ ] Index loaded at startup (lifespan event)

### Tests
- [ ] test_data: HTML cleaning, chunking, tag mapping
- [ ] test_retrieval: BM25, FAISS, RRF fusion correctness
- [ ] test_graph: node execution, routing logic
- [ ] test_api: health, ask endpoint with mocked LLM
- [ ] All tests pass without network (mock LLM/embeddings)

## Edge Cases
- Empty question → 422 validation error
- Question unrelated to Python → graceful "no relevant docs found" response
- All retrieved docs graded irrelevant → query rewrite loop (max 3)
- LLM hallucination detected → re-retrieve loop (max 3)
- FAISS index missing → auto-build on first startup
- API timeout → error handling with user-friendly message
