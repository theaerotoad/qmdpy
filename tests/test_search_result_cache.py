import pytest
from unittest.mock import MagicMock
from qmd.config import Config, CollectionConfig
from qmd.store import Store, Result
from qmd.db import init_schema, update_db_last_updated

def test_full_search_result_caching_and_invalidation(db_conn, tmp_path):
    history_db_path = tmp_path / "history.db"
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "test.md").write_text("# Title\n\nSearch result caching content.")

    config = Config(
        db_path=str(tmp_path / "main.db"),
        history_db_path=str(history_db_path),
        collections={"test": CollectionConfig(path=str(doc_dir))}
    )
    
    init_schema(db_conn)
    store = Store(config=config, connection=db_conn)
    
    mock_rerank = MagicMock(return_value=[{"index": 0, "score": 0.95}])
    store.llm.rerank = mock_rerank
    store.llm.embed_batch = MagicMock(return_value=[[0.1] * 768])
    
    store.index_collection("test", config.collections["test"])

    query = "Search result caching"
    
    # First search with rerank=True: Cache Miss -> Calls reranker
    res1 = store.hybrid_search(query, rerank=True)
    assert len(res1) > 0
    assert mock_rerank.call_count == 1
    
    # Second identical search: Cache Hit -> Returns cached results without calling reranker
    res2 = store.hybrid_search(query, rerank=True)
    assert len(res2) == len(res1)
    assert res2[0].path == res1[0].path
    assert mock_rerank.call_count == 1  # Unchanged
    
    # Invalidate cache by updating database timestamp (simulating new/updated file index)
    update_db_last_updated(db_conn)
    
    # Third search: Cache Miss due to updated DB -> Calls reranker again
    res3 = store.hybrid_search(query, rerank=True)
    assert len(res3) > 0
    assert mock_rerank.call_count == 2


def test_search_result_caching_disabled_via_config(db_conn, tmp_path):
    history_db_path = tmp_path / "history_disabled.db"
    doc_dir = tmp_path / "docs_disabled"
    doc_dir.mkdir()
    (doc_dir / "test.md").write_text("# Title\n\nDisabled caching search content.")

    config = Config(
        db_path=str(tmp_path / "main_disabled.db"),
        history_db_path=str(history_db_path),
        cache_search_results=False,
        collections={"test": CollectionConfig(path=str(doc_dir))}
    )
    
    init_schema(db_conn)
    store = Store(config=config, connection=db_conn)
    
    mock_rerank = MagicMock(return_value=[{"index": 0, "score": 0.95}])
    store.llm.rerank = mock_rerank
    store.llm.embed_batch = MagicMock(return_value=[[0.1] * 768])
    
    store.index_collection("test", config.collections["test"])

    query = "Disabled caching search"
    
    # First search: calls reranker
    res1 = store.hybrid_search(query, rerank=True)
    assert len(res1) > 0
    assert mock_rerank.call_count == 1
    
    # Second search with cache_search_results=False: bypasses cache and calls reranker again
    res2 = store.hybrid_search(query, rerank=True)
    assert len(res2) > 0
    assert mock_rerank.call_count == 2
