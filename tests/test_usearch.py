import pytest
import sqlite3
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Skip the entire test suite gracefully if usearch is not installed
usearch = pytest.importorskip("usearch")

from qmd.store import Store, encode_vector
from qmd.config import Config, CollectionConfig
from qmd.utils import compress_text
from qmd.db import set_db_meta, ensure_vector_table

class MockLLM:
    """Mock LLM to return predictable vectors for distance testing."""
    def format_query_for_embedding(self, q): return q
    def embed_batch(self, texts):
        res = []
        for t in texts:
            if "match" in t.lower():
                res.append([1.0, 0.0])
            else:
                res.append([0.0, 1.0])
        return res

def test_usearch_backward_compatibility(db_conn, tmp_path):
    """Test that if usearch is missing or disabled, sqlite-vec fallback is used cleanly."""
    config = Config(db_path=str(tmp_path / "test.db"))
    store = Store(config, connection=db_conn)
    store.llm = MockLLM()
    
    # 1. Setup DB with a vector
    set_db_meta(db_conn, "vector_dim", "2")
    ensure_vector_table(db_conn, dim=2, quant_type="none")
    vec_blob = encode_vector([1.0, 0.0], quant_type="none")
    
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'match', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'fallback.md', 'F', 'h1', 'now')")
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_blob,))
    rowid = cursor.lastrowid
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (?, 'h1', 0, ?)", (rowid, compress_text("match text")))
    db_conn.commit()

    # 2. Force usearch_index to None to simulate missing library/index
    store.usearch_index = None

    # 3. Search and ensure usearch is NOT called, but results are still returned via sqlite-vec
    with patch("usearch.index.Index.search") as mock_search:
        results = store.search_vec("match query")
        mock_search.assert_not_called()
        
        assert len(results) == 1
        assert results[0].path == "fallback.md"
        assert results[0].source == "vec"


def test_build_ann_index(db_conn, tmp_path):
    """Test non-destructive migration building usearch from the SQLite vectors table."""
    config = Config(db_path=str(tmp_path / "test.db"))
    store = Store(config, connection=db_conn)
    
    set_db_meta(db_conn, "vector_dim", "2")
    ensure_vector_table(db_conn, dim=2, quant_type="none")
    
    # Insert 10 raw vectors into sqlite-vec directly
    cursor = db_conn.cursor()
    for i in range(10):
        vec_blob = encode_vector([float(i), 0.0], quant_type="none")
        cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_blob,))
    db_conn.commit()
    
    # Assert no index exists initially
    assert getattr(store, "usearch_index", None) is None
    
    # Build it
    success = store.build_usearch_index()
    assert success is True
    assert store.usearch_index is not None
    assert len(store.usearch_index) == 10
    assert Path(store.usearch_path).exists()


def test_search_routing_and_accuracy(db_conn, tmp_path):
    """Test that vector search correctly routes to usearch when available."""
    config = Config(db_path=str(tmp_path / "test.db"))
    store = Store(config, connection=db_conn)
    store.llm = MockLLM()
    
    set_db_meta(db_conn, "vector_dim", "2")
    ensure_vector_table(db_conn, dim=2, quant_type="none")
    store.build_usearch_index()
    
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'match', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'match.md', 'M', 'h1', 'now')")
    
    vec_blob = encode_vector([1.0, 0.0], quant_type="none")
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_blob,))
    rowid = cursor.lastrowid
    
    # Add to usearch index
    store.usearch_index.add(rowid, np.array([1.0, 0.0], dtype=np.float32))
    
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (?, 'h1', 0, ?)", (rowid, compress_text("match text")))
    db_conn.commit()

    # Spy on the search method to ensure usearch was actually executed
    with patch.object(store.usearch_index, 'search', wraps=store.usearch_index.search) as spy:
        results = store.search_vec("match query")
        spy.assert_called_once()
        
        assert len(results) == 1
        assert results[0].path == "match.md"
        assert results[0].score > 0.9


def test_lifecycle_sync(db_conn, tmp_path):
    """Test write-path synchronization (Insert, Update, Delete) with usearch."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(tmp_path / "test.db")
    )
    store = Store(config, connection=db_conn)
    store.llm = MockLLM()
    
    # Force metadata so chunking logic can properly initialize tables
    set_db_meta(db_conn, "vector_dim", "2")
    store.build_usearch_index() 
    
    # 1. Add file
    f1 = notes_dir / "doc1.md"
    f1.write_text("match content chunk 1")
    store.index_collection("test", config.collections["test"])
    
    # One file, one chunk = 1 vector
    assert len(store.usearch_index) == 1
    
    # 2. Modify file (simulating an update that might change chunks)
    f1.write_text("match content chunk 1\n\n# New Header\n\nmatch content chunk 2")
    store.index_collection("test", config.collections["test"])
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT count(*) FROM vectors")
    db_vec_count = cursor.fetchone()[0]
    
    # Usearch index should be strictly in sync with SQLite row count
    assert len(store.usearch_index) == db_vec_count
    
    # 3. Delete file
    f1.unlink()
    store.index_collection("test", config.collections["test"]) 
    
    cursor.execute("SELECT count(*) FROM vectors")
    db_vec_count = cursor.fetchone()[0]
    assert db_vec_count == 0
    assert len(store.usearch_index) == 0


def test_post_filtering(db_conn, tmp_path):
    """Test over-fetching and SQLite post-filtering for exact query matches."""
    config = Config(db_path=str(tmp_path / "test.db"))
    store = Store(config, connection=db_conn)
    store.llm = MockLLM()
    
    set_db_meta(db_conn, "vector_dim", "2")
    ensure_vector_table(db_conn, dim=2, quant_type="none")
    store.build_usearch_index()
    
    cursor = db_conn.cursor()
    
    # Insert 5 identical matches, 4 in Coll_B, 1 in Coll_A
    vec = [1.0, 0.0]
    vec_blob = encode_vector(vec, quant_type="none")
    
    for i in range(5):
        coll = "Coll_A" if i == 0 else "Coll_B"
        doc_hash = f"h{i}"
        cursor.execute(f"INSERT INTO content (hash, body, created_at) VALUES ('{doc_hash}', 'match', 'now')")
        cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES (?, ?, ?, ?, 'now')", 
                       (coll, f"doc{i}.md", f"Doc {i}", doc_hash))
        
        cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_blob,))
        rowid = cursor.lastrowid
        store.usearch_index.add(rowid, np.array(vec, dtype=np.float32))
        
        cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (?, ?, 0, ?)", 
                       (rowid, doc_hash, compress_text("match text")))
        
    db_conn.commit()

    # Search with limit=2, collection="Coll_A"
    # usearch should return all 5 (or over-fetch limit), and SQLite should filter down to just the 1 in Coll_A
    with patch.object(store.usearch_index, 'search', wraps=store.usearch_index.search) as spy:
        results = store.search_vec("match", limit=2, collection="Coll_A")
        
        spy.assert_called_once()
        # Verify k_val is high enough for overfetching (at least 500 in the implemented logic)
        call_args = spy.call_args
        assert call_args[0][1] >= 500 
        
        assert len(results) == 1
        assert results[0].collection == "Coll_A"
        assert results[0].path == "doc0.md"