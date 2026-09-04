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

def test_web_subpath_x_forwarded_prefix(web_client):
    # 1. Verify index template renders static assets and window.QMD_BASE_URL with the prefix
    res = web_client.get('/', headers={'X-Forwarded-Prefix': '/upstream/qmd'})
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'href="/upstream/qmd/static/css/app.css"' in html
    assert 'src="/upstream/qmd/static/js/state.js"' in html
    assert 'window.QMD_BASE_URL = "/upstream/qmd"' in html

    # 2. Verify API requests routed with the subpath prefix in PATH_INFO are correctly handled
    res2 = web_client.post('/upstream/qmd/api/search', json={
        'query': 'document',
        'limit': 1,
        'session_id': 'web_prefix_1'
    }, headers={'X-Forwarded-Prefix': '/upstream/qmd'})
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2['session_id'] == 'web_prefix_1'
    assert len(data2['results']) == 1

def test_web_subpath_script_name(web_client):
    # Verify behavior when SCRIPT_NAME is supplied directly by WSGI environment
    res = web_client.get('/', base_url='http://localhost/upstream/qmd')
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'href="/upstream/qmd/static/css/app.css"' in html
    assert 'src="/upstream/qmd/static/js/state.js"' in html
    assert 'window.QMD_BASE_URL = "/upstream/qmd"' in html

def test_web_subpath_fallback_relative_assets(web_client):
    # When no prefix headers or script_root are supplied, verify relative fallback paths
    res = web_client.get('/')
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'href="static/css/app.css"' in html
    assert 'src="static/js/state.js"' in html
    assert 'src="static/js/modals.js"' in html
    assert 'src="static/js/search.js"' in html


def test_web_document_download(web_client):
    # 1. Download existing document file
    res = web_client.get('/api/document/download?collection=test&path=doc1.md')
    assert res.status_code == 200
    assert b"First chunk of document 1" in res.data
    disp = res.headers.get("Content-Disposition", "")
    assert "attachment" in disp
    assert "doc1.md" in disp

    # 2. Missing path parameter
    res_missing = web_client.get('/api/document/download')
    assert res_missing.status_code == 400

    # 3. Path traversal rejection or missing file
    res_traversal = web_client.get('/api/document/download?collection=test&path=../../outside.txt')
    assert res_traversal.status_code == 404

    res_not_found = web_client.get('/api/document/download?collection=test&path=nonexistent.md')
    assert res_not_found.status_code == 404


def test_web_document_open_system(web_client):
    with patch("subprocess.Popen") as mock_popen, patch("os.startfile", create=True) as mock_startfile:
        # 1. Successful open via POST
        res = web_client.post('/api/document/open', json={'collection': 'test', 'path': 'doc1.md'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'success'
        assert 'doc1.md' in data['message']

        # 2. Successful open via GET query params
        res_get = web_client.get('/api/document/open?collection=test&path=doc1.md')
        assert res_get.status_code == 200
        data_get = res_get.get_json()
        assert data_get['status'] == 'success'

        # 3. Missing path
        res_no_path = web_client.post('/api/document/open', json={'collection': 'test'})
        assert res_no_path.status_code == 400

        # 4. Nonexistent file
        res_missing_file = web_client.post('/api/document/open', json={'collection': 'test', 'path': 'not_there.docx'})
        assert res_missing_file.status_code == 404


def test_web_document_open_system_error(web_client):
    with patch("subprocess.Popen", side_effect=OSError("Command failed")):
        with patch("os.startfile", side_effect=OSError("Command failed"), create=True):
            res = web_client.post('/api/document/open', json={'collection': 'test', 'path': 'doc1.md'})
            assert res.status_code == 500
            data = res.get_json()
            assert data['status'] == 'error'
            assert 'Command failed' in data['message']
