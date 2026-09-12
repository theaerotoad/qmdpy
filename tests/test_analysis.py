import pytest
from qmd.store import Store
from qmd.config import Config
from qmd.db import get_connection, init_schema

class MockLLMForAnalysis:
    def __init__(self, *args, **kwargs):
        pass

    def format_query_for_embedding(self, *args, **kwargs):
        return "mock"
        
    def embed_batch(self, *args, **kwargs):
        return []

    def analyze_document(self, title, content, path=""):
        return {
            "altTitle": f"Better {title}",
            "doc_type": "academicPaper",
            "summary": f"Mock summary for {title}",
            "authors": ["Mock Author"],
            "tags": ["mock", "test"],
            "dates": ["2026-09-11"],
            "questions": ["Q1 in?", "Q2 in?", "Q3 in?", "Q4 out?", "Q5 out?"]
        }

@pytest.fixture
def mock_store(tmp_path):
    db_path = tmp_path / "test_analysis.db"
    config = Config(db_path=str(db_path))
    store = Store(config)
    
    # Intercept LLM with our mock to avoid network calls
    store.llm = MockLLMForAnalysis()
    
    # Insert some dummy chunks into the database so we have something to analyze
    cursor = store.conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('hash1', 'compressed_mock_body', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at, active) VALUES ('test_coll', 'test_doc.md', 'Test Doc', 'hash1', 'now', 1)")
    
    # Needs a corresponding vector and chunk metadata row
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (1, x'00')")
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (1, 'hash1', 0, 'mock chunk text')")
    store.conn.commit()
    
    return store

def test_analyze_target(mock_store):
    # 1. First run, should successfully analyze the document
    results = mock_store.analyze_target(limit=5, time_limit=1.0, verbose=True)
    
    assert len(results) == 1
    res = results[0]
    assert res["path"] == "test_doc.md"
    assert res["hash"] == "hash1"
    
    analysis = res["analysis"]
    assert analysis["summary"] == "Mock summary for Test Doc"
    assert analysis["altTitle"] == "Better Test Doc"
    assert analysis["doc_type"] == "academicPaper"
    assert len(analysis["questions"]) == 5
    assert "Mock Author" in analysis["authors"]
    
    # 2. Run again without force, should return empty because it's already analyzed
    results_second = mock_store.analyze_target(limit=5)
    assert len(results_second) == 0
    
    # 3. Run with force, should override and re-analyze
    results_forced = mock_store.analyze_target(limit=5, force=True)
    assert len(results_forced) == 1