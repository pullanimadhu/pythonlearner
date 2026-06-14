"""Pytest fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure workspace is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<p>I want to <strong>iterate</strong> over a <code>dict</code> in Python.</p>
<pre><code>for k, v in my_dict.items():
    print(k, v)
</code></pre>
<p>What's the best way?</p>
"""

SAMPLE_QUESTIONS_CSV = """Id,OwnerUserId,CreationDate,Score,Title,Body
100,1,2008-08-02T15:11:16Z,50,How to iterate over a dict,"<p>I want to iterate over a dict.</p><pre><code>for k, v in d.items():</code></pre><p>What is the best way?</p>"
200,2,2008-08-02T17:01:58Z,30,How to use itertools.groupby,"<p>I cannot understand itertools.groupby.</p>"
"""

SAMPLE_ANSWERS_CSV = """Id,OwnerUserId,CreationDate,ParentId,Score,Body
101,5,2008-08-02T16:00:00Z,100,40,"<p>Use the <code>.items()</code> method:</p><pre><code>for k, v in my_dict.items():\n    print(k, v)</code></pre>"
102,6,2008-08-02T17:00:00Z,100,20,"<p>You can also use <code>.keys()</code> and <code>.values()</code> separately.</p>"
201,7,2008-08-02T18:00:00Z,200,35,"<p>itertools.groupby groups consecutive elements by a key function.</p><pre><code>from itertools import groupby\nfor key, group in groupby(data, key_func):\n    print(key, list(group))</code></pre>"
"""

SAMPLE_TAGS_CSV = """Id,Tag
100,python
100,dictionary
100,iteration
200,python
200,itertools
200,iteration
"""


@pytest.fixture
def sample_html():
    return SAMPLE_HTML


@pytest.fixture
def sample_questions_csv(tmp_path):
    p = tmp_path / "questions.csv"
    p.write_text(SAMPLE_QUESTIONS_CSV)
    return p


@pytest.fixture
def sample_answers_csv(tmp_path):
    p = tmp_path / "answers.csv"
    p.write_text(SAMPLE_ANSWERS_CSV)
    return p


@pytest.fixture
def sample_tags_csv(tmp_path):
    p = tmp_path / "tags.csv"
    p.write_text(SAMPLE_TAGS_CSV)
    return p


@pytest.fixture
def mock_settings(tmp_path):
    """Settings with mock values — no real API calls."""
    from app.config import Settings

    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://test.example.com",
        embedding_dim=8,
        faiss_index_path=str(tmp_path / "index"),
        data_dir=str(tmp_path / "data"),
    )


@pytest.fixture
def sample_documents(sample_questions_csv, sample_answers_csv, sample_tags_csv):
    """Build documents from sample CSVs."""
    from app.data.processor import build_documents

    return build_documents(
        questions_csv=sample_questions_csv,
        answers_csv=sample_answers_csv,
        tags_csv=sample_tags_csv,
    )
