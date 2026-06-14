# Infrastructure

## Proxy Routes
- Caddy reverse proxy at `/` → port 8000 (Uvicorn/FastAPI)

## Background Services
- `app` (Uvicorn): `cd /workspace && uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Environment Variables
- `OPENAI_API_KEY` (secret) — Drytis OpenAI-compatible API key
- `OPENAI_BASE_URL` — Drytis OpenAI-compatible base URL (https://llm.drytis.ai)
- `MODEL_NAME` — LLM model (default: gpt-4o-mini)
- `EMBEDDING_MODEL` — Embedding model (default: text-embedding-3-small)
- `EMBEDDING_DIM` — Embedding vector dimension (default: 1536)
- `RETRIEVAL_K` — Documents to retrieve per branch (default: 10)
- `FINAL_K` — Final documents after fusion (default: 5)
- `FAISS_INDEX_PATH` — Path to persistent FAISS index
- `DATA_DIR` — Directory for processed data files
- `QUESTIONS_CSV` — Path to questions CSV file
- `ANSWERS_CSV` — Path to answers CSV file
- `TAGS_CSV` — Path to tags CSV file
- `APP_ENV` — environment (development/production)
- `APP_DEBUG` — debug flag

## Ports
- 8000: FastAPI/Uvicorn

## Setup Script (deploy-time)
1. `pip install -r requirements.txt`
2. `python scripts/build_index.py` — builds FAISS index + BM25 store from CSVs
3. Start Uvicorn background service
