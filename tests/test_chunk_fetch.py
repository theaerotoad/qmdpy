import pytest
from unittest.mock import MagicMock
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.web import app

SAMPLE_MD_MULTI_CHUNK = """# Document Title

Chunk zero content goes here with enough text to form chunk 0.

## Section 1

Chunk one content goes here with enough text to form chunk 1.

## Section 2

Chunk two content goes here with enough text to form chunk 2.

## Section 3

Chunk three content goes here with enough text to form chunk 3.
"""


def test_chunk_fetch_by_seq_and_window(db_conn, temp_db_path, tmp_path):
    config = Config()
    config.db_path = temp_db_path
    config.max_chunk_size = 80
    config.target_chunk_size = 50
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "multi.md"
    doc_path.write_text(SAMPLE_MD_MULTI_CHUNK, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("test_coll", coll_cfg, force=True)

    # 1. Fetch exact single chunk (window=0)
    results = store.get_chunk_by_seq("test_coll", "multi.md", seq_id=1, window=0)
    assert len(results) == 1
    assert results[0].seq_id == 1

    # 2. Fetch chunk with surrounding context window (window=1)
    results_window = store.get_chunk_by_seq("test_coll", "multi.md", seq_id=1, window=1)
    assert len(results_window) >= 2
    seq_ids = [r.seq_id for r in results_window]
    assert 1 in seq_ids

    # 3. Test lower boundary condition (seq_id=0 with window=2 -> no negative seq_ids)
    results_boundary = store.get_chunk_by_seq("test_coll", "multi.md", seq_id=0, window=2)
    assert all(r.seq_id >= 0 for r in results_boundary)


def test_chunk_fetch_by_rowid(db_conn, temp_db_path, tmp_path):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "multi.md"
    doc_path.write_text(SAMPLE_MD_MULTI_CHUNK, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("test_coll", coll_cfg, force=True)

    # Retrieve vector rowid for seq_id 0
    cursor = db_conn.cursor()
    cursor.execute("SELECT rowid FROM chunk_metadata WHERE seq_id = 0 LIMIT 1")
    rowid = cursor.fetchone()[0]

    results = store.get_chunk_by_id(rowid, window=1)
    assert len(results) >= 1
    assert results[0].seq_id == 0


def test_chunk_fetch_nonexistent(db_conn, temp_db_path):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    assert store.get_chunk_by_id(999999) == []
    assert store.get_chunk_by_seq("test_coll", "missing.md", seq_id=0) == []


def test_api_chunk_endpoint(db_conn, temp_db_path, tmp_path, monkeypatch):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "multi.md"
    doc_path.write_text(SAMPLE_MD_MULTI_CHUNK, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("test_coll", coll_cfg, force=True)

    app.config['config'] = config
    app.config['TESTING'] = True

    monkeypatch.setattr("qmd.web.get_store", lambda: store)

    with app.test_client() as client:
        # 1. Fetch by path & seq_id
        res = client.get("/api/chunk?collection=test_coll&path=multi.md&seq_id=0&window=1")
        assert res.status_code == 200
        data = res.get_json()
        assert "chunks" in data
        assert len(data["chunks"]) >= 1

        # 2. Fetch by rowid
        cursor = db_conn.cursor()
        cursor.execute("SELECT rowid FROM chunk_metadata WHERE seq_id = 0 LIMIT 1")
        rowid = cursor.fetchone()[0]

        res_rowid = client.get(f"/api/chunk?rowid={rowid}&window=1")
        assert res_rowid.status_code == 200
        data_rowid = res_rowid.get_json()
        assert len(data_rowid["chunks"]) >= 1

        # 3. Missing parameters
        res_bad = client.get("/api/chunk")
        assert res_bad.status_code == 400

        # 4. Invalid rowid type
        res_invalid = client.get("/api/chunk?rowid=abc")
        assert res_invalid.status_code == 400

