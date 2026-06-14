"""Application configuration via Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / Embeddings ---
    openai_api_key: str = ""
    openai_base_url: str = "https://llm.drytis.ai"
    model_name: str = "z-ai/glm-5"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Retrieval ---
    retrieval_k: int = 10
    final_k: int = 5

    # --- Paths ---
    faiss_index_path: str = "/workspace/data/faiss_index"
    data_dir: str = "/workspace/data"
    questions_csv: str = "/workspace/sampleQuestion.csv"
    answers_csv: str = "/workspace/sampleAnswer.csv"
    tags_csv: str = "/workspace/Tags.csv"

    # --- App ---
    app_env: str = "development"
    app_debug: bool = True
    max_attempts: int = 3

    @property
    def index_dir(self) -> Path:
        return Path(self.faiss_index_path)

    @property
    def faiss_db_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "docs.pkl"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def index_exists(self) -> bool:
        return self.faiss_db_path.exists() and self.meta_path.exists()


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    # Ensure DATA_DIR exists
    s = Settings()
    Path(s.data_dir).mkdir(parents=True, exist_ok=True)
    s.index_dir.mkdir(parents=True, exist_ok=True)
    return s
