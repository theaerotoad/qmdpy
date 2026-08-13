import pytest
from unittest.mock import MagicMock
from qmd.config import Config
from qmd.store import Store
from qmd.db import get_cached_query_embedding, save_query_embedding, init_history_schema

def test_query_embedding_cache_miss_and_hit(db_conn, tmp_path):
    history_db_path = tmp_path / "history.db"
    config = Config(
        db_path=str(tmp_path / "main.db"),
        history_db_path=str(history_db_path),
        embed_model="TestEmbedModel"
    )
    
    store = Store(config=config, connection=db_conn)
    
    sample_vector = [0.1, 0.2, 0.3, 0.4]
    mock_embed = MagicMock(return_value=[sample_vector])
    store.llm.embed_batch = mock_embed
    
    query = "task management in python"
    
    # 1. First retrieval: Cache Miss -> Calls embedder and saves to cache
    vec1 = store.get_query_embedding(query)
    assert vec1 == sample_vector
    assert mock_embed.call_count == 1
    
    # Verify vector was persisted in history_conn
    cached = get_cached_query_embedding(store.history_conn, query, "TestEmbedModel")
    assert cached is not None
    assert len(cached) == len(sample_vector)
    for a, b in zip(cached, sample_vector):
        assert abs(a - b) < 1e-5
        
    # 2. Second retrieval: Cache Hit -> Returns cached vector without calling embedder again
    vec2 = store.get_query_embedding(query)
    assert vec2 == pytest.approx(sample_vector)
    assert mock_embed.call_count == 1  # Call count remains 1

def test_save_and_get_query_embedding_direct(tmp_path):
    from qmd.db import get_history_connection
    history_conn = get_history_connection(tmp_path / "hist.db")
    
    query = "direct query test"
    model = "ModelA"
    vec = [0.5, -0.5, 1.0]
    
    # Should be None before saving
    assert get_cached_query_embedding(history_conn, query, model) is None
    
    save_query_embedding(history_conn, query, model, vec)
    
    retrieved = get_cached_query_embedding(history_conn, query, model)
    assert retrieved is not None
    assert len(retrieved) == len(vec)
    for a, b in zip(retrieved, vec):
        assert abs(a - b) < 1e-5