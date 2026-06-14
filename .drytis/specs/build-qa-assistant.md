# Task Spec: Build Python Q&A Assistant

## Files to Create
- `requirements.txt` — Python dependencies
- `app/__init__.py`, `app/main.py`, `app/config.py`
- `app/api/__init__.py`, `app/api/routes.py`, `app/api/schemas.py`
- `app/rag/__init__.py`, `app/rag/pipeline.py`, `app/rag/vectorstore.py`
- `app/utils/__init__.py`, `app/utils/logger.py`
- `scripts/fetch_data.py` — Fetch SO Python Q&A data
- `scripts/build_index.py` — Build ChromaDB index
- `tests/conftest.py`, `tests/test_api.py`, `tests/test_rag.py`
- `.env.example`, `README.md`, `.gitignore`
- `pyproject.toml`

## Acceptance Criteria
- [ ] FastAPI app serves on port 8000 via Uvicorn
- [ ] `GET /health` returns 200 JSON with status
- [ ] `POST /ask` accepts `{question: str}` returns `{answer: str, sources: list}`
- [ ] RAG pipeline retrieves relevant SO Q&A and synthesizes answers via LangChain + ChromaDB
- [ ] ChromaDB index pre-built from SO Python Q&A data
- [ ] 8+ test queries documented with responses
- [ ] Pytest test suite passes (unit + integration)
- [ ] App accessible at public preview URL
- [ ] README with setup instructions
- [ ] .env.example documents all required variables
- [ ] Background service registered as production Uvicorn command

## Edge Cases to Handle
- Empty question string → 422 validation error
- Question unrelated to Python → graceful response without hallucination
- API timeout → error handling with user-friendly message
