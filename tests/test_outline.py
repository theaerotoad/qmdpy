import pytest
from unittest.mock import MagicMock
from qmd.docparse.parser import extract_outline
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.web import app

TICK3 = "`" * 3

SAMPLE_MD = f"""# Python Guide

Introductory section.

## Installation

Run pip install.

{TICK3}bash
# This is a comment inside a code block, not a heading
echo "Hello"
{TICK3}

### Windows

Windows instructions here.

## Usage

Usage instructions here.
"""


def test_extract_outline_basic():
    headings = extract_outline(content=SAMPLE_MD)
    assert len(headings) == 4

    # Check Heading 1
    assert headings[0]["level"] == 1
    assert headings[0]["text"] == "Python Guide"
    assert headings[0]["line_num"] == 1

    # Check Heading 2
    assert headings[1]["level"] == 2
    assert headings[1]["text"] == "Installation"

    # Check Heading 3 (Code block comment should be ignored)
    assert headings[2]["level"] == 3
    assert headings[2]["text"] == "Windows"

    # Check Heading 4
    assert headings[3]["level"] == 2
    assert headings[3]["text"] == "Usage"


def test_extract_outline_empty():
    assert extract_outline(content="") == []
    assert extract_outline(content=None, file_path=None) == []


def test_store_get_document_outline(db_conn, temp_db_path, tmp_path):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "guide.md"
    doc_path.write_text(SAMPLE_MD, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("test_code", coll_cfg, force=True)

    outline = store.get_document_outline("test_code", "guide.md")
    assert outline is not None
    assert outline["collection"] == "test_code"
    assert outline["path"] == "guide.md"
    assert outline["title"] == "Guide"
    assert outline["total_chunks"] > 0
    assert len(outline["headings"]) == 4

    # Verify heading details
    h0 = outline["headings"][0]
    assert h0["level"] == 1
    assert h0["text"] == "Python Guide"
    assert h0["start_seq"] <= h0["end_seq"]


def test_store_get_document_outline_missing(db_conn, temp_db_path):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    outline = store.get_document_outline("nonexistent_coll", "nonexistent_file.md")
    assert outline is None


def test_api_outline_endpoint(db_conn, temp_db_path, tmp_path, monkeypatch):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "guide.md"
    doc_path.write_text(SAMPLE_MD, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("test_code", coll_cfg, force=True)

    app.config['config'] = config
    app.config['TESTING'] = True

    monkeypatch.setattr("qmd.web.get_store", lambda: store)

    with app.test_client() as client:
        # Valid outline request
        res = client.get("/api/outline?collection=test_code&path=guide.md")
        assert res.status_code == 200
        data = res.get_json()
        assert data["title"] == "Guide"
        assert len(data["headings"]) == 4

        # Missing path param
        res_bad = client.get("/api/outline")
        assert res_bad.status_code == 400

        # Nonexistent doc
        res_404 = client.get("/api/outline?collection=test_code&path=missing.md")
        assert res_404.status_code == 404

