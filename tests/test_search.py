import pytest
import struct
from qmd.store import Store, Result
from qmd.config import Config

def test_rrf_logic_manual(db_conn):
    """Verify RRF bonus math and ranking."""
    store = Store(Config(db_path=":memory:"), connection=db_conn)
    
    # Mock result sets
    list_a = [
        Result(path="doc1", title="T1", text="text1", score=1.0, source="fts"),
        Result(path="doc2", title="T2", text="text2", score=0.8, source="fts"),
    ]
    list_b = [
        Result(path="doc2", title="T2", text="text2", score=0.9, source="vec"),
        Result(path="doc3", title="T3", text="text3", score=0.7, source="vec"),
    ]
    
    # We manually simulate what hybrid_search does
    rrf_scores = {}
    k = 60
    
    def score_list(l):
        for rank, res in enumerate(l):
            key = res.path
            if key not in rrf_scores: rrf_scores[key] = 0.0
            s = 1.0 / (k + rank)
            if rank == 0: s += 0.05
            elif rank < 3: s += 0.02
            rrf_scores[key] += s

    score_list(list_a)
    score_list(list_b)
    
    # Doc 2 appeared in both: 
    # List A rank 1 (idx 1): 1/61 + 0.02
    # List B rank 0 (idx 0): 1/60 + 0.05
    expected_doc2 = (1/61 + 0.02) + (1/60 + 0.05)
    assert pytest.approx(rrf_scores["doc2"], 0.0001) == expected_doc2

def test_doc_view_candidate_limit_consistency():
    from argparse import Namespace
    from unittest.mock import MagicMock
    from qmd.main import handle_search

    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = []

    args = Namespace(
        query=["python"],
        limit=10,
        doc=True,
        flat=False,
        chunks=False,
        verbose=False,
        rerank=False,
        rerank_only=False,
        collection=None,
        lex=None,
        title=None,
        path=None,
        json=False,
        w2n=False,
    )

    handle_search(args, mock_store)

    # Verify hybrid_search was called with limit=10, not limit=30
    mock_store.hybrid_search.assert_called_once()
    assert mock_store.hybrid_search.call_args.kwargs["limit"] == 10

def test_discover_single_hit_per_doc(db_conn, monkeypatch):
    """Verify that discover returns at most 1 chunk per document with aggregated match counts."""
    from qmd.utils import compress_text
    from qmd.store import encode_vector

    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)

    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'doc 1 text with multiple python sections', 'now')")
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h2', 'doc 2 text with single python section', 'now')")

    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('coll', 'doc1.md', 'Doc 1', 'h1', 'now')")
    id1 = cursor.lastrowid
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('coll', 'doc2.md', 'Doc 2', 'h2', 'now')")
    id2 = cursor.lastrowid

    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'coll', 'doc1.md', 'Doc 1', 'doc 1 python')", (id1,))
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'coll', 'doc2.md', 'Doc 2', 'doc 2 python')", (id2,))

    dummy_vec = encode_vector([0.0] * 768)
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (1, ?)", (dummy_vec,))
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (2, ?)", (dummy_vec,))
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (3, ?)", (dummy_vec,))

    # Doc 1 has 2 matching chunks (seq 0 and seq 1)
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (1, 'h1', 0, ?, 'Intro')", (compress_text("doc 1 python chunk 0"),))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (2, 'h1', 1, ?, 'Advanced')", (compress_text("doc 1 python chunk 1"),))
    # Doc 2 has 1 matching chunk
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (3, 'h2', 0, ?, 'Guide')", (compress_text("doc 2 python chunk 0"),))

    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (1, 'coll', 'doc1.md', 'Doc 1', 'doc 1 python chunk 0', 'Intro')")
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (2, 'coll', 'doc1.md', 'Doc 1', 'doc 1 python chunk 1', 'Advanced')")
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (3, 'coll', 'doc2.md', 'Doc 2', 'doc 2 python chunk 0', 'Guide')")

    db_conn.commit()

    results = store.discover("python", limit=10)
    assert len(results) == 2
    paths = [r.path for r in results]
    assert len(set(paths)) == 2  # Exactly 1 result per document

    doc1_res = next(r for r in results if r.path == "doc1.md")
    assert doc1_res.match_count == 2
    assert doc1_res.title == "Doc 1"

    doc2_res = next(r for r in results if r.path == "doc2.md")
    assert doc2_res.match_count == 1

def test_group_results_by_doc_fts_normalization():
    from qmd.main import group_results_by_doc
    from qmd.store import Result

    results = [
        Result(path="doc1.md", title="Doc 1", text="Chunk 0 text", score=0.9, source="vec", seq_id=0),
        Result(path="doc1.md", title="Doc 1", text="Chunk 1 text", score=0.8, source="vec", seq_id=1),
        Result(path="doc1.md", title="Doc 1", text="FTS preview text", score=0.95, source="fts", seq_id=-1),
    ]

    grouped = group_results_by_doc(results)
    assert len(grouped) == 1
    assert grouped[0]["path"] == "doc1.md"
    # Chunk 0 text should come before Chunk 1 text in reading order
    assert "Chunk 0 text" in grouped[0]["snippets"][0]

def test_wide_to_narrow_search(db_conn):
    from qmd.utils import compress_text
    from qmd.store import encode_vector
    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)
    
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'python programming language', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('code', 'docs/python/guide.md', 'Python Guide', 'h1', 'now')")
    doc_id = cursor.lastrowid
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'code', 'docs/python/guide.md', 'Python Guide', 'python programming language')", (doc_id,))
    dummy_vec = encode_vector([0.0] * 768)
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (doc_id, dummy_vec))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'h1', 0, ?, '')", (doc_id, compress_text("python programming language")))
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'code', 'docs/python/guide.md', 'Python Guide', 'python programming language', '')", (doc_id,))
    db_conn.commit()

    results = store.wide_to_narrow_search("python programming", limit=5)
    assert len(results) > 0
    assert results[0].path == "docs/python/guide.md"

def test_spacy_fts_expansion():
    from qmd.utils import extract_tiered_fts_terms, build_spacy_fts_queries

    query = "How does Elon Musk feel about NASA given his daughter's recent lawsuit?"
    terms = extract_tiered_fts_terms(query)
    assert isinstance(terms, dict)
    assert "primary" in terms
    assert "secondary" in terms

    queries = build_spacy_fts_queries(query)
    assert isinstance(queries, list)
    assert len(queries) > 0

def test_fts_retrieval(db_conn):
    from qmd.utils import compress_text
    from qmd.store import encode_vector
    store = Store(Config(db_path=":memory:"), connection=db_conn)
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h_test', 'The quick brown fox jumps over the lazy dog', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('', 'test.md', 'Test Title', 'h_test', 'now')")
    doc_id = cursor.lastrowid
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)",
                   (doc_id, "", "test.md", "Test Title", "The quick brown fox jumps over the lazy dog"))
    dummy_vec = encode_vector([0.0] * 768)
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (doc_id, dummy_vec))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'h_test', 0, ?, '')", (doc_id, compress_text("The quick brown fox jumps over the lazy dog")))
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, '', 'test.md', 'Test Title', 'The quick brown fox jumps over the lazy dog', '')", (doc_id,))
    db_conn.commit()
    
    # Test basic retrieval
    results = store.search_fts("fox")
    assert len(results) == 1
    assert results[0].path == "test.md"
    
    # Test stop word filtering ("the" and "is" should be ignored, searching only for "fox")
    results_stopwords = store.search_fts("the fox is")
    assert len(results_stopwords) == 1
    assert results_stopwords[0].path == "test.md"

def test_vector_cosine_calc(db_conn, monkeypatch):
    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)
    
    # Mock LLM to return specific embedding
    class MockLLM:
        def format_query_for_embedding(self, q): return q
        def embed_batch(self, texts):
            # Return a simple 2D vector [1, 0]
            return [[1.0, 0.0]] * len(texts)
    
    monkeypatch.setattr(store, "llm", MockLLM())
    
    # Setup DB with a matching and a non-matching vector
    # Matching: [1, 0]
    # Non-matching: [0, 1]
    vec_match = struct.pack('2f', 1.0, 0.0)
    vec_no = struct.pack('2f', 0.0, 1.0)
    
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'match', 'now')")
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h2', 'no', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'm.md', 'M', 'h1', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'n.md', 'N', 'h2', 'now')")
    
    from qmd.utils import compress_text
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_match,))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (1, 'h1', 0, ?)", (compress_text("match text"),))
    
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_no,))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (2, 'h2', 0, ?)", (compress_text("no text"),))
    db_conn.commit()
    
    results = store.search_vec("query")
    assert results[0].path == "m.md"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].path == "n.md"
    assert results[1].score == pytest.approx(0.0)

def test_search_collection_filter(db_conn, monkeypatch):
    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)
    
    # Setup DB with two identical entries but different collections
    cursor = db_conn.cursor()
    
    # Insert into content first to satisfy foreign key constraints
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'python test', 'now')")
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h2', 'python test', 'now')")
    
    # Insert into documents
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('coll_a', 'doc_a.md', 'A', 'h1', 'now')")
    id_a = cursor.lastrowid
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('coll_b', 'doc_b.md', 'B', 'h2', 'now')")
    id_b = cursor.lastrowid

    # Insert into documents_fts
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)", (id_a, "coll_a", "doc_a.md", "A", "python test"))
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)", (id_b, "coll_b", "doc_b.md", "B", "python test"))
    
    from qmd.utils import compress_text
    from qmd.store import encode_vector
    dummy_vec = encode_vector([0.0] * 768)
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (id_a, dummy_vec))
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (id_b, dummy_vec))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'h1', 0, ?, '')", (id_a, compress_text("python test")))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'h2', 0, ?, '')", (id_b, compress_text("python test")))
    
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'coll_a', 'doc_a.md', 'A', 'python test', '')", (id_a,))
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'coll_b', 'doc_b.md', 'B', 'python test', '')", (id_b,))

    db_conn.commit()
    
    # Without filter, should return both
    res_all = store.search_fts("python")
    assert len(res_all) == 2
    
    # With filter, should return only coll_a
    res_a = store.search_fts("python", collection="coll_a")
    assert len(res_a) == 1
    assert res_a[0].collection == "coll_a"
    assert res_a[0].path == "doc_a.md"

def test_search_metadata_filters(db_conn, monkeypatch):
    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)
    
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h3', 'metadata test', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('docs', 'src/api/auth.md', 'Authentication API', 'h3', 'now')")
    doc_id = cursor.lastrowid
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)", (doc_id, "docs", "src/api/auth.md", "Authentication API", "metadata test"))
    from qmd.utils import compress_text
    from qmd.store import encode_vector
    dummy_vec = encode_vector([0.0] * 768)
    cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (doc_id, dummy_vec))
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'h3', 0, ?, '')", (doc_id, compress_text("metadata test")))
    cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'docs', 'src/api/auth.md', 'Authentication API', 'metadata test', '')", (doc_id,))
    db_conn.commit()
    
    # Substring match on title
    res_title = store.search_fts("metadata", title="Auth")
    assert len(res_title) == 1
    assert res_title[0].title == "Authentication API"
    
    # Substring match on path
    res_path = store.search_fts("metadata", path="api/auth")
    assert len(res_path) == 1
    assert res_path[0].path == "src/api/auth.md"
    
    # Failed match
    res_fail = store.search_fts("metadata", title="Unknown")
    assert len(res_fail) == 0


def test_search_multi_path_filters(db_conn, monkeypatch):
    config = Config(db_path=":memory:")
    store = Store(config, connection=db_conn)

    cursor = db_conn.cursor()
    from qmd.utils import compress_text
    from qmd.store import encode_vector

    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h_a', 'common text in doc a', 'now')")
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h_b', 'common text in doc b', 'now')")
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h_c', 'common text in doc c', 'now')")

    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'src/core/engine.md', 'Engine', 'h_a', 'now')")
    id_a = cursor.lastrowid
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'src/utils/helpers.md', 'Helpers', 'h_b', 'now')")
    id_b = cursor.lastrowid
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'docs/guide.md', 'Guide', 'h_c', 'now')")
    id_c = cursor.lastrowid

    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'c', 'src/core/engine.md', 'Engine', 'common text in doc a')", (id_a,))
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'c', 'src/utils/helpers.md', 'Helpers', 'common text in doc b')", (id_b,))
    cursor.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'c', 'docs/guide.md', 'Guide', 'common text in doc c')", (id_c,))

    dummy_vec = encode_vector([0.0] * 768)
    for doc_id, h, p, t in [(id_a, 'h_a', 'src/core/engine.md', 'Engine'), (id_b, 'h_b', 'src/utils/helpers.md', 'Helpers'), (id_c, 'h_c', 'docs/guide.md', 'Guide')]:
        cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (doc_id, dummy_vec))
        cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, ?, 0, ?, '')", (doc_id, h, compress_text(f"common text in {t}")))
        cursor.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'c', ?, ?, ?, '')", (doc_id, p, t, f"common text in {t}"))

    db_conn.commit()

    # Search with multiple path filters
    res_multi = store.search_fts("common", path=["src/core", "docs/"])
    found_paths = {r.path for r in res_multi}
    assert "src/core/engine.md" in found_paths
    assert "docs/guide.md" in found_paths
    assert "src/utils/helpers.md" not in found_paths


def test_search_vec_quantization(db_conn, monkeypatch):
    """Verify search_vec behavior under int8 quantization."""
    config = Config(db_path=":memory:", vector_quantization="int8")
    store = Store(config, connection=db_conn)

    class MockLLM:
        def format_query_for_embedding(self, q): return q
        def embed_batch(self, texts):
            return [[1.0, 0.0]] * len(texts)

    monkeypatch.setattr(store, "llm", MockLLM())

    from qmd.db import set_db_meta, ensure_vector_table
    from qmd.store import encode_vector

    set_db_meta(db_conn, "vector_quantization", "int8")
    ensure_vector_table(db_conn, dim=2, quant_type="int8")

    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('h1', 'match', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'm.md', 'M', 'h1', 'now')")

    vec_blob = encode_vector([1.0, 0.0], quant_type="int8")
    from qmd.utils import compress_text
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (vec_blob,))
    rowid = cursor.lastrowid
    cursor.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text) VALUES (?, 'h1', 0, ?)", (rowid, compress_text("match text")))
    db_conn.commit()

    results = store.search_vec("query")
    assert len(results) == 1
    assert results[0].path == "m.md"
    assert results[0].score > 0.8


def test_federated_multi_db_search(tmp_path):
    from qmd.db import get_connection, init_schema
    from qmd.utils import compress_text
    from qmd.store import encode_vector, Store
    from qmd.config import load_config

    master_db_path = tmp_path / "master.db"
    child_db_path = tmp_path / "child.db"

    # Setup master database
    conn_m = get_connection(master_db_path)
    init_schema(conn_m)
    cur_m = conn_m.cursor()
    cur_m.execute("INSERT INTO content (hash, body, created_at) VALUES ('hm', 'master document content about databases', 'now')")
    cur_m.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('master_coll', 'doc_m.md', 'Master Doc', 'hm', 'now')")
    id_m = cur_m.lastrowid
    cur_m.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'master_coll', 'doc_m.md', 'Master Doc', 'master document content about databases')", (id_m,))
    dummy_vec = encode_vector([0.0] * 768)
    cur_m.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (id_m, dummy_vec))
    cur_m.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'hm', 0, ?, 'MasterHeader')", (id_m, compress_text("master document content about databases")))
    cur_m.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'master_coll', 'doc_m.md', 'Master Doc', 'master document content about databases', 'MasterHeader')", (id_m,))
    conn_m.commit()
    conn_m.close()

    # Setup child database
    conn_c = get_connection(child_db_path)
    init_schema(conn_c)
    cur_c = conn_c.cursor()
    cur_c.execute("INSERT INTO content (hash, body, created_at) VALUES ('hc', 'child document content about databases', 'now')")
    cur_c.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('child_coll', 'doc_c.md', 'Child Doc', 'hc', 'now')")
    id_c = cur_c.lastrowid
    cur_c.execute("INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, 'child_coll', 'doc_c.md', 'Child Doc', 'child document content about databases')", (id_c,))
    cur_c.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (id_c, dummy_vec))
    cur_c.execute("INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers) VALUES (?, 'hc', 0, ?, 'ChildHeader')", (id_c, compress_text("child document content about databases")))
    cur_c.execute("INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers) VALUES (?, 'child_coll', 'doc_c.md', 'Child Doc', 'child document content about databases', 'ChildHeader')", (id_c,))
    conn_c.commit()
    conn_c.close()

    # Create master and included YAML configs
    child_yaml = tmp_path / "child.yml"
    child_yaml.write_text(f"""
db_path: "{child_db_path}"
embed_model: "EmbeddingGemma 300m"
collections:
  child_coll:
    path: "{tmp_path}"
    glob: "*.md"
""")
    master_yaml = tmp_path / "master.yml"
    master_yaml.write_text(f"""
db_path: "{master_db_path}"
embed_model: "EmbeddingGemma 300m"
include:
  - "{child_yaml.name}"
collections:
  master_coll:
    path: "{tmp_path}"
    glob: "*.md"
""")

    cfg = load_config(master_yaml)
    store = Store(cfg, read_only=True)

    # 1. Federated FTS search across all databases
    res_fts = store.search_fts("databases")
    assert len(res_fts) == 2
    colls = {r.collection for r in res_fts}
    assert colls == {"master_coll", "child_coll"}

    # 2. Filtered search targeting child collection
    res_child = store.search_fts("databases", collection="child_coll")
    assert len(res_child) == 1
    assert res_child[0].collection == "child_coll"
    assert res_child[0].path == "doc_c.md"

    # 3. Federated Discover search
    res_disc = store.discover("databases", limit=10)
    assert len(res_disc) == 2
    assert {r.collection for r in res_disc} == {"master_coll", "child_coll"}

    # 4. Target chunk, outline, and grep collection routing
    child_chunk = store.get_chunk_by_seq("child_coll", "doc_c.md", seq_id=0)
    assert len(child_chunk) == 1
    assert "child document content" in child_chunk[0].text

    outline = store.get_document_outline("child_coll", "doc_c.md")
    assert outline is not None
    assert outline["collection"] == "child_coll"

    grep_res = store.grep_search("content", collection="child_coll")
    assert len(grep_res) == 1
    assert grep_res[0]["collection"] == "child_coll"

    # 5. Verify mutation prohibition in federated mode
    with pytest.raises(RuntimeError, match="federated include mode"):
        store.index_collection("master_coll", cfg.collections["master_coll"])
