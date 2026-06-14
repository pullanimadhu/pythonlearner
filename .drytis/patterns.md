# Patterns & Coding Standards

## Code Style
- Python 3.13+, fully type-hinted
- `async def` for route handlers
- Pydantic v2 for request/response validation
- Functions <= 50 lines, single responsibility
- No hardcoded secrets, URLs, or credentials

## Naming
- snake_case for functions, variables, modules
- PascalCase for classes
- UPPER_CASE for constants/env vars

## Error Handling
- API routes use FastAPI HTTPException with appropriate status codes
- RAG pipeline raises custom exceptions (not bare ValueError)
- Logger used throughout — no print() in production code

## Testing
- Unit tests: test functions/classes in isolation
- Integration tests: test API endpoints with httpx TestClient
- Tests must not depend on external API calls (mock LLM/resolver)
- `pytest tests/` should pass without network access

## Env Handling
- All config via Pydantic Settings reading from environment
- `.env.example` documents all required variables
- Never commit `.env` files
