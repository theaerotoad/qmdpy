import pytest
from unittest.mock import MagicMock
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.db import init_schema, get_seen_chunks_for_session

def test_session_creation_and_exclude_seen(db_conn, tmp_path):
    history_db_path = tmp_path / "history.db"
    config = Config(
        db_path=str(tmp_path / "main.db"),
        history_db_path=str(history_db_path)
    )
    
    init_schema(db_conn)
    
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.md").write_text("# Doc 1\n\nFirst chunk of document 1.\n\nSecond chunk of document 1.")
    (doc_dir / "doc2.md").write_text("# Doc 2\n\nFirst chunk of document 2.")
    
    store = Store(config=config, connection=db_conn)
    store.llm.embed_batch = MagicMock(return_value=[[0.1] * 768, [0.2] * 768])
    
    col_cfg = CollectionConfig(path=str(doc_dir))
    store.index_collection("test_col", col_cfg)
    
    session_id = "test_sess_123"
    
    # 1. First search without exclude_seen
    results1 = store.hybrid_search("document", limit=2)
    assert len(results1) > 0
    
    # Record results in session
    from qmd.db import record_session_event, record_session_results
    event_id = record_session_event(store.history_conn, session_id, "search", "document", None, str(config.db_path), "2026-08-13T12:00:00Z")
    record_session_results(store.history_conn, session_id, event_id, results1)
    
    # Verify seen chunks set
    seen = get_seen_chunks_for_session(store.history_conn, session_id)
    assert len(seen) == len(results1)
    
    # 2. Second search with exclude_seen
    results2 = store.hybrid_search("document", limit=2, exclude_seen_set=seen)
    
    # Previously seen chunks must not be in results2
    for r in results2:
        key = (r.collection, r.path, r.seq_id)
        assert key not in seen
        
    assert store.last_exclusion_stats["excluded_chunks"] > 0
