import pytest
from qmd.store import Store
from qmd.config import Config
from qmd.db import get_connection, init_schema

class MockLLMForAnalysis:
    ANALYSIS_PROMPT_VERSION = "1.0"

    def __init__(self, *args, **kwargs):
        self.raise_exception = False
        self.next_response = None

    def format_query_for_embedding(self, *args, **kwargs):
        return "mock"
        
    def embed_batch(self, *args, **kwargs):
        return []

    def analyze_document(self, title, content, path=""):
        if self.raise_exception:
            raise ValueError("Rate limit exceeded")
        if self.next_response is not None:
            resp = self.next_response
            self.next_response = None
            return resp
            
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

    # 4. Bump the version, test --outdated flag
    mock_store.llm.ANALYSIS_PROMPT_VERSION = "1.1"
    
    # Running without outdated should still yield 0 hits
    assert len(mock_store.analyze_target(limit=5)) == 0
    
    # Running with outdated should yield 1 hit since version 1.1 != 1.0
    results_outdated = mock_store.analyze_target(limit=5, outdated=True)
    assert len(results_outdated) == 1
    
    # Running again with outdated should yield 0 hits (now it's 1.1)
    assert len(mock_store.analyze_target(limit=5, outdated=True)) == 0

def test_analyze_robustness_and_empty_retries(mock_store):
    cursor = mock_store.conn.cursor()
    
    # Doc 2: for testing exception failure
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('hash2', 'mock_body2', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at, active) VALUES ('test_coll', 'doc2.md', 'Doc 2', 'hash2', 'now', 1)")
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (2, 'hash2', 0, 'text2')")

    # Doc 3: for testing invalid/empty response
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('hash3', 'mock_body3', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at, active) VALUES ('test_coll', 'doc3.md', 'Doc 3', 'hash3', 'now', 1)")
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (3, 'hash3', 0, 'text3')")
    
    # Doc 4: will simulate a document that has a completely empty analysis in DB
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('hash4', 'mock_body4', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at, active) VALUES ('test_coll', 'doc4.md', 'Doc 4', 'hash4', 'now', 1)")
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (4, 'hash4', 0, 'text4')")
    cursor.execute("""
        INSERT INTO document_analysis (doc_hash, summary, authors, tags, dates, questions, doc_type, alt_title, prompt_version, analyzed_at)
        VALUES ('hash4', '', '[]', '[]', '[]', '[]', '', '', '1.0', 'now')
    """)
    mock_store.conn.commit()

    # 1. Test empty fields auto-retry
    # Doc 4 is already in the DB but empty, so analyze_target should pick it up without `force=True`
    res4 = mock_store.analyze_target(path='doc4.md', verbose=True)
    assert len(res4) == 1
    assert res4[0]['hash'] == 'hash4'
    assert res4[0]['analysis']['summary'] == 'Mock summary for Doc 4'
    
    # 2. Test exception handling during analysis
    mock_store.llm.raise_exception = True
    res2 = mock_store.analyze_target(path='doc2.md', verbose=True)
    assert len(res2) == 0  # Should yield 0 successful analyses
    
    # Verify it was not saved (meaning it gets picked up on retry)
    mock_store.llm.raise_exception = False
    res2_retry = mock_store.analyze_target(path='doc2.md')
    assert len(res2_retry) == 1
    assert res2_retry[0]['hash'] == 'hash2'

    # 3. Test empty/error dict response handling
    mock_store.llm.next_response = {"summary": "   ", "error": "rate limit or block"}
    res3 = mock_store.analyze_target(path='doc3.md', verbose=True)
    assert len(res3) == 0
    
    # Verify it was not saved (meaning it gets picked up on retry)
    mock_store.llm.next_response = None
    res3_retry = mock_store.analyze_target(path='doc3.md')
    assert len(res3_retry) == 1
    assert res3_retry[0]['hash'] == 'hash3'