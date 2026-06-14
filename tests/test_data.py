"""Tests for data processing module."""

from __future__ import annotations

from app.data.processor import (
    QADocument,
    build_documents,
    clean_html,
    load_answers,
    load_questions,
    load_tags,
)


class TestCleanHtml:
    def test_strips_html_tags(self):
        html = "<p>Hello <strong>world</strong></p>"
        result = clean_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<strong>" not in result

    def test_preserves_code_blocks(self):
        html = "<pre><code>print('hello')</code></pre>"
        result = clean_html(html)
        assert "```" in result
        assert "print('hello')" in result

    def test_preserves_inline_code(self):
        html = "<p>Use <code>items()</code> method</p>"
        result = clean_html(html)
        assert "`items()`" in result

    def test_empty_input(self):
        assert clean_html("") == ""
        assert clean_html(None) == ""  # type: ignore

    def test_collapses_whitespace(self):
        html = "<p>Line 1</p><br><br><br><br><p>Line 2</p>"
        result = clean_html(html)
        assert "\n\n\n\n" not in result

    def test_strips_links(self):
        html = '<p>See <a href="http://example.com">this link</a></p>'
        result = clean_html(html)
        assert "this link" in result
        assert "<a" not in result


class TestLoadTags:
    def test_loads_tags(self, sample_tags_csv):
        tags_map = load_tags(sample_tags_csv)
        assert 100 in tags_map
        assert 200 in tags_map
        assert "python" in tags_map[100]
        assert "dictionary" in tags_map[100]
        assert "itertools" in tags_map[200]

    def test_multiple_tags_per_question(self, sample_tags_csv):
        tags_map = load_tags(sample_tags_csv)
        assert len(tags_map[100]) == 3  # python, dictionary, iteration


class TestLoadQuestions:
    def test_loads_questions(self, sample_questions_csv):
        questions = load_questions(sample_questions_csv)
        assert 100 in questions
        assert 200 in questions
        assert "iterate" in questions[100]["title"].lower()

    def test_cleans_body_html(self, sample_questions_csv):
        questions = load_questions(sample_questions_csv)
        body = questions[100]["body"]
        assert "<p>" not in body
        assert "dict" in body.lower() or "iterate" in body.lower()


class TestLoadAnswers:
    def test_loads_answers(self, sample_answers_csv):
        answers = load_answers(sample_answers_csv)
        assert len(answers) == 3
        assert answers[0]["parent_id"] == 100
        assert answers[2]["parent_id"] == 200


class TestBuildDocuments:
    def test_creates_per_answer_docs(self, sample_documents):
        assert len(sample_documents) == 3  # 3 answers → 3 docs

    def test_doc_has_question_metadata(self, sample_documents):
        doc = sample_documents[0]
        assert doc.question_id == 100
        assert doc.answer_id == 101
        assert "iterate" in doc.question_title.lower()

    def test_doc_has_tags(self, sample_documents):
        doc = sample_documents[0]
        assert "python" in doc.tags
        assert "dictionary" in doc.tags

    def test_doc_has_clean_text(self, sample_documents):
        doc = sample_documents[0]
        assert "<p>" not in doc.answer_body
        assert "items()" in doc.answer_body or "items" in doc.answer_body

    def test_text_for_embedding(self, sample_documents):
        doc = sample_documents[0]
        text = doc.text_for_embedding
        assert "Title:" in text
        assert "Answer:" in text
        assert "Tags:" in text

    def test_to_dict_from_dict_roundtrip(self, sample_documents):
        doc = sample_documents[0]
        d = doc.to_dict()
        doc2 = QADocument.from_dict(d)
        assert doc2.doc_id == doc.doc_id
        assert doc2.question_id == doc.question_id
        assert doc2.tags == doc.tags
