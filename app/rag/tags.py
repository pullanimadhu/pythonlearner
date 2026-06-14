"""Tag prediction + metadata filtering.

Uses the LLM to predict relevant Stack Overflow tags from the user's query.
These tags are then used to pre-filter the candidate document set.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from app.config import Settings
from app.utils.logger import logger

# Top tags from Tags.csv — used as the vocabulary for tag prediction.
# This keeps the LLM's predictions grounded in the actual tag space.
KNOWN_TAGS_TOP_100 = [
    "python", "django", "python-2.7", "pandas", "python-3.x", "numpy",
    "list", "matplotlib", "regex", "dictionary", "tkinter", "string",
    "flask", "google-app-engine", "csv", "arrays", "json", "mysql",
    "linux", "html", "scipy", "multithreading", "sqlalchemy", "windows",
    "beautifulsoup", "django-models", "javascript", "selenium", "xml",
    "pyqt", "file", "unicode", "class", "oop", "datetime", "performance",
    "exception", "sql", "parsing", "unit-testing", "function", "loops",
    "sorting", "inheritance", "sockets", "http", "subprocess", "linux",
    "debugging", "encoding", "dictionary", "django-templates", "git",
    "bash", "email", "url", "generators", "decorators", "database",
    "api", "security", "logging", "ubuntu", "qt", "gtk", "pygtk",
    "object", "tuple", "set", "lambda", "iterator", "import",
    "module", "type-hinting", "serialization", "pickle", "yield",
    "static-methods", "class-method", "metaclass", "reflection",
    "csv", "excel", "image", "pdf", "web-scraping", "automation",
    "testing", "pytest", "deployment", "docker", "aws", "lambda",
    "postgresql", "sqlite", "mongodb", "redis", "celery", "rest",
]

TAG_PREDICTION_PROMPT = """You are a tag classifier for a Python Q&A system.
Given a user's question, predict which of the following Stack Overflow tags are most relevant.
Return ONLY a JSON list of tag strings (max 5 tags). Pick from the known tags list when possible.

Known tags: {known_tags}

Question: {question}

Respond with ONLY a JSON array of strings, e.g. ["django", "authentication"]. No other text."""


class TagPredictor:
    """Predicts relevant SO tags from a user query using LLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def predict(self, question: str) -> list[str]:
        """Return predicted tags for a question (max 5)."""
        prompt = TAG_PREDICTION_PROMPT.format(
            known_tags=", ".join(KNOWN_TAGS_TOP_100),
            question=question,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful Python tag classifier."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            tags = self._parse_tags(raw)
            logger.info("Predicted tags for query: %s → %s", question[:60], tags)
            return tags
        except Exception as e:
            logger.warning("Tag prediction failed: %s — returning empty", e)
            return []

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        """Parse LLM output into a clean list of tags."""
        # Try JSON parse first
        try:
            tags = json.loads(raw)
            if isinstance(tags, list):
                return [str(t).strip().lower() for t in tags][:5]
        except json.JSONDecodeError:
            pass

        # Fallback: extract quoted strings
        matches = re.findall(r'["\']([^"\']+)["\']', raw)
        if matches:
            return [m.strip().lower() for m in matches][:5]

        # Last resort: comma-separated
        parts = raw.replace("[", "").replace("]", "").split(",")
        return [p.strip().lower() for p in parts if p.strip()][:5]
