"""Data processor: parse CSVs, clean HTML, chunk per-answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class QADocument:
    """A single retrievable document (one answer + question context)."""

    doc_id: str
    question_id: int
    answer_id: int
    question_title: str
    question_body: str
    answer_body: str
    tags: list[str] = field(default_factory=list)
    score: int = 0
    creation_date: str = ""

    @property
    def text_for_embedding(self) -> str:
        """Combined text used for embedding and BM25 indexing."""
        tag_str = ", ".join(self.tags) if self.tags else ""
        parts = [
            f"Tags: {tag_str}" if tag_str else "",
            f"Title: {self.question_title}",
            f"Question: {self.question_body[:500]}",
            f"Answer: {self.answer_body}",
        ]
        return "\n".join(p for p in parts if p)

    @property
    def text_for_display(self) -> str:
        """Human-readable text for the LLM context."""
        tag_str = ", ".join(self.tags) if self.tags else "none"
        return (
            f"Question (id={self.question_id}): {self.question_title}\n"
            f"Tags: {tag_str}\n"
            f"Question body: {self.question_body[:800]}\n"
            f"Answer (score={self.score}):\n{self.answer_body}"
        )

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "question_title": self.question_title,
            "question_body": self.question_body,
            "answer_body": self.answer_body,
            "tags": self.tags,
            "score": self.score,
            "creation_date": self.creation_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QADocument:
        return cls(**d)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def clean_html(html: str) -> str:
    """Convert HTML body to clean plain text, preserving code blocks."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")

    # Extract and mark code blocks before stripping tags
    for pre in soup.find_all("pre"):
        code_text = pre.get_text()
        pre.replace_with(f"\n```\n{code_text}\n```\n")

    # Extract inline code
    for code in soup.find_all("code"):
        code_text = code.get_text()
        code.replace_with(f"`{code_text}`")

    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def load_tags(tags_csv: str | Path) -> dict[int, list[str]]:
    """Load Tags.csv → {question_id: [tag1, tag2, ...]}"""
    logger.info("Loading tags from %s ...", tags_csv)
    df = pd.read_csv(tags_csv, names=["Id", "Tag"], header=0)
    tags_map: dict[int, list[str]] = {}
    for qid, tag in zip(df["Id"], df["Tag"]):
        tags_map.setdefault(int(qid), []).append(str(tag).strip())
    logger.info("Loaded %d questions with tags", len(tags_map))
    return tags_map


def load_questions(questions_csv: str | Path) -> dict[int, dict]:
    """Load questions CSV → {question_id: {title, body, score, date}}"""
    logger.info("Loading questions from %s ...", questions_csv)
    df = pd.read_csv(questions_csv)
    questions: dict[int, dict] = {}
    for _, row in df.iterrows():
        qid = int(row["Id"])
        questions[qid] = {
            "title": str(row.get("Title", "")),
            "body": clean_html(str(row.get("Body", ""))),
            "score": int(row.get("Score", 0)),
            "creation_date": str(row.get("CreationDate", "")),
        }
    logger.info("Loaded %d questions", len(questions))
    return questions


def load_answers(answers_csv: str | Path) -> list[dict]:
    """Load answers CSV → list of answer dicts."""
    logger.info("Loading answers from %s ...", answers_csv)
    df = pd.read_csv(answers_csv)
    answers: list[dict] = []
    for _, row in df.iterrows():
        answers.append(
            {
                "answer_id": int(row["Id"]),
                "parent_id": int(row["ParentId"]),
                "body": clean_html(str(row.get("Body", ""))),
                "score": int(row.get("Score", 0)),
                "creation_date": str(row.get("CreationDate", "")),
            }
        )
    logger.info("Loaded %d answers", len(answers))
    return answers


# ---------------------------------------------------------------------------
# Build documents
# ---------------------------------------------------------------------------

def build_documents(
    questions_csv: str | Path,
    answers_csv: str | Path,
    tags_csv: str | Path,
) -> list[QADocument]:
    """Parse all CSVs and build per-answer QADocument list."""
    questions = load_questions(questions_csv)
    answers = load_answers(answers_csv)
    tags_map = load_tags(tags_csv)

    documents: list[QADocument] = []
    for ans in answers:
        qid = ans["parent_id"]
        q = questions.get(qid)
        if q is None:
            continue

        doc = QADocument(
            doc_id=f"a_{ans['answer_id']}",
            question_id=qid,
            answer_id=ans["answer_id"],
            question_title=q["title"],
            question_body=q["body"],
            answer_body=ans["body"],
            tags=tags_map.get(qid, []),
            score=ans["score"],
            creation_date=ans["creation_date"],
        )
        documents.append(doc)

    logger.info("Built %d QADocuments from data", len(documents))
    return documents
