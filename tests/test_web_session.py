import pytest
from unittest.mock import MagicMock, patch
from qmd.web import app
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.db import init_schema

@pytest.fixture
def web_client(tmp_path, db_conn):
    app.config['TESTING'] = True
    history_db_path = tmp_path / "history.db"
    
    notes_dir = tmp_path / "web_notes"
    notes_dir.mkdir()
    (notes_dir / "doc1.md").write_text("# Doc 1\n\nFirst chunk of document 1.\n\nSecond chunk of document 1.")
    
    config = Config(
        db_path=str(tmp_path / "web.db"),
        history_db_path=str(history_db_path),
        collections={"test": CollectionConfig(path=str(notes_dir))}
    )
    
    init_schema(db_conn)
    store = Store(config=config, connection=db_conn)
    store.llm.embed_batch = MagicMock(return_value=[[0.1] * 768, [0.2] * 768])
    store.index_collection("test", config.collections["test"])
    
    app.config['config'] = config
    
    with patch("qmd.web.get_store", return_value=store):
        with app.test_client() as client:
            yield client

def test_web_search_session_and_exclude_seen(web_client):
    # 1. Initial search
    res = web_client.post('/api/search', json={
        'query': 'document',
        'limit': 1,
        'session_id': 'web_sess_1'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data['session_id'] == 'web_sess_1'
    assert len(data['results']) == 1
    assert data['excluded_count'] == 0
    
    # 2. Search again with exclude_seen
    res2 = web_client.post('/api/search', json={
        'query': 'document',
        'limit': 1,
        'session_id': 'web_sess_1',
        'exclude_seen': True
    })
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2['session_id'] == 'web_sess_1'
    assert data2['excluded_count'] > 0

    # 3. Search with multi-path parameter
    res3 = web_client.post('/api/search', json={
        'query': 'document',
        'limit': 5,
        'paths': ['doc1.md', 'another/path.md']
    })
    assert res3.status_code == 200
    data3 = res3.get_json()
    assert len(data3['results']) > 0

def test_web_discover_endpoint(web_client):
    res = web_client.post('/api/discover', json={
        'query': 'document',
        'limit': 5,
        'session_id': 'web_disc_1'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data['type'] == 'discover'
    assert data['session_id'] == 'web_disc_1'
    assert len(data['results']) == 1
    assert data['results'][0]['match_count'] >= 1
    assert '<discover_results' in data['xml']
