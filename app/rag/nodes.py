"""LangGraph nodes: the building blocks of the adaptive RAG pipeline.

Each node takes GraphState, does work, and returns a partial state update.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.config import Settings
from app.rag.retrieval import HybridRetriever, RetrievalResult
from app.rag.state import GraphState
from app.rag.tags import TagPredictor
from app.utils.logger import logger

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _make_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _llm_call(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Single-turn LLM call. System message ensures clean output."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant for a Python programming Q&A system. Follow instructions precisely."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Node factory container
# ---------------------------------------------------------------------------


class RAGNodes:
    """Container for all graph nodes, holding shared dependencies."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        tag_predictor: TagPredictor,
    ):
        self.settings = settings
        self.retriever = retriever
        self.tag_predictor = tag_predictor
        self.client = _make_client(settings)

    # ------------------------------------------------------------------
    # Node 1: Query analysis — predict tags + optionally rewrite
    # ------------------------------------------------------------------

    def query_analysis(self, state: GraphState) -> dict[str, Any]:
        """Predict tags from the user's question."""
        question = state.get("question", "")
        attempt = state.get("attempt", 0)

        trace = state.get("trace", [])
        trace.append(f"query_analysis (attempt {attempt})")

        if attempt > 0:
            prompt = (
                f"You are a search query optimizer. Rewrite this question to be more "
                f"specific and search-friendly for a technical Q&A database. "
                f"Keep it as a question. Return ONLY the rewritten question, nothing else.\n\n"
                f"Original: {question}"
            )
            rewritten = _llm_call(self.client, self.settings.model_name, prompt)
            question = rewritten
            trace.append(f"query rewritten → {rewritten[:80]}...")

        tags = self.tag_predictor.predict(question)
        logger.info("Query analysis: tags=%s", tags)

        return {
            "tags": tags,
            "rewritten_question": question,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 2: Retrieve — hybrid retrieval with tag filter
    # ------------------------------------------------------------------

    def retrieve(self, state: GraphState) -> dict[str, Any]:
        """Run hybrid retrieval and return top documents."""
        question = state.get("rewritten_question", state.get("question", ""))
        tags = state.get("tags", [])

        trace = state.get("trace", [])
        trace.append(f"retrieve: query='{question[:60]}...', tags={tags}")

        docs = self.retriever.retrieve(query=question, tag_filter=tags)

        trace.append(f"retrieved {len(docs)} documents")
        logger.info("Retrieved %d documents", len(docs))

        return {
            "retrieved_docs": docs,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 3: Grade documents — batch LLM relevance judgment
    # ------------------------------------------------------------------

    def grade_documents(self, state: GraphState) -> dict[str, Any]:
        """Grade retrieved documents in a single LLM call for efficiency."""
        question = state.get("rewritten_question", state.get("question", ""))
        docs = state.get("retrieved_docs", [])
        attempt = state.get("attempt", 0)

        trace = state.get("trace", [])

        if not docs:
            return {"graded_docs": [], "docs_relevant": False, "trace": trace + ["no docs to grade"]}

        # If we've retried too many times, accept whatever we have as relevant
        if attempt >= 2:
            trace.append(f"max retry reached — accepting all {len(docs)} docs")
            return {
                "graded_docs": docs,
                "docs_relevant": True,
                "trace": trace,
            }

        # Build a batch prompt — list all docs, ask for relevant indices
        doc_list = []
        for i, doc_result in enumerate(docs):
            doc = doc_result.document
            doc_list.append(f"[{i}] Title: {doc.question_title}\n    Answer: {doc.answer_body[:300]}")

        prompt = (
            f"You are a relevance grader for a Python Q&A retrieval system.\n"
            f"Given a question and a list of documents, determine which documents are relevant.\n"
            f"Return ONLY a comma-separated list of relevant document indices (e.g. '0,2,3').\n"
            f"If none are relevant, return 'none'.\n\n"
            f"Question: {question}\n\n"
            f"Documents:\n" + "\n".join(doc_list)
        )

        verdict = _llm_call(self.client, self.settings.model_name, prompt)
        verdict_lower = verdict.lower().strip()

        # Extract digits from the response (handles reasoning model output noise)
        import re as _re
        number_matches = _re.findall(r'\d+', verdict_lower)
        is_none = "none" in verdict_lower and not number_matches

        graded: list[RetrievalResult] = []
        if not is_none and number_matches:
            for num_str in number_matches:
                idx = int(num_str)
                if 0 <= idx < len(docs):
                    graded.append(docs[idx])

        # If grading produced nothing but we have docs, accept top-2 as fallback
        if not graded and attempt >= 1:
            graded = docs[:2]

        docs_relevant = len(graded) > 0
        trace.append(f"graded: {len(graded)}/{len(docs)} relevant (verdict: {verdict_lower[:60]})")
        logger.info("Grade verdict: %s → %d relevant", verdict_lower[:80], len(graded))

        return {
            "graded_docs": graded,
            "docs_relevant": docs_relevant,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 4: Generate — synthesize answer from graded docs
    # ------------------------------------------------------------------

    def generate(self, state: GraphState) -> dict[str, Any]:
        """Generate answer from graded documents."""
        question = state.get("rewritten_question", state.get("question", ""))
        docs = state.get("graded_docs", state.get("retrieved_docs", []))

        trace = state.get("trace", [])

        context_parts = []
        for i, doc_result in enumerate(docs, 1):
            doc = doc_result.document
            context_parts.append(f"[Source {i}] {doc.text_for_display}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"Use ONLY the provided context to answer the question. "
            f"If the context doesn't contain the answer, say you don't have enough information. "
            f"Cite source numbers [Source N] when referencing specific information.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:"
        )

        answer = _llm_call(
            self.client,
            self.settings.model_name,
            prompt,
            temperature=0.3,
        )

        trace.append(f"generated answer ({len(answer)} chars)")
        logger.info("Generated answer: %s...", answer[:100])

        return {
            "generation": answer,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Node 5: Hallucination check — is the answer grounded in context?
    # ------------------------------------------------------------------

    def hallucination_check(self, state: GraphState) -> dict[str, Any]:
        """Verify the generated answer is grounded in the retrieved context."""
        generation = state.get("generation", "")
        docs = state.get("graded_docs", state.get("retrieved_docs", []))
        attempt = state.get("attempt", 0)

        trace = state.get("trace", [])

        # If we've retried enough, accept the answer as grounded
        if attempt >= 2:
            trace.append("hallucination check → SKIPPED (max retries)")
            return {"answer_grounded": True, "trace": trace}

        context_parts = [dr.document.text_for_display for dr in docs]
        context = "\n\n".join(context_parts)

        prompt = (
            f"Determine if the answer is fully supported by the provided context. "
            f"Respond with exactly one word: 'grounded' or 'not grounded'.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer to check:\n{generation}\n\n"
            f"Verdict:"
        )

        verdict = _llm_call(self.client, self.settings.model_name, prompt)
        is_grounded = verdict.lower().strip() == "grounded"

        trace.append(
            f"hallucination check → {'PASSED' if is_grounded else 'FAILED'}"
        )

        return {
            "answer_grounded": is_grounded,
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Routing functions (edges)
    # ------------------------------------------------------------------

    def decide_after_grading(self, state: GraphState) -> str:
        """Route after document grading: generate if relevant, else rewrite."""
        if state.get("docs_relevant", False):
            return "generate"
        return "rewrite_query"

    def decide_after_hallucination(self, state: GraphState) -> str:
        """Route after hallucination check."""
        attempt = state.get("attempt", 0)
        max_attempts = state.get("max_attempts", self.settings.max_attempts)

        if state.get("answer_grounded", False):
            return "end"
        if attempt >= max_attempts:
            return "end"
        return "retrieve"
