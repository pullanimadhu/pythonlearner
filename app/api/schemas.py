"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")


class SourceInfo(BaseModel):
    """Source reference for an answer."""

    question_id: int
    question_title: str
    answer_id: int
    score: int
    tags: list[str]
    snippet: str = Field("", description="Excerpt of the answer")


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str
    sources: list[SourceInfo]
    tags: list[str]
    trace: list[str]
    grounded: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    index_size: int
    embedding_model: str
    llm_model: str
