import pytest
import sqlite3
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from qmd.store import Store
from qmd.config import Config, CollectionConfig
from qmd.llm import LLMClient
from qmd.utils import decompress_text

# --- Mocks ---

@pytest.fixture
def mock_llm_client():
    with patch("qmd.store.LLMClient") as MockClass:
        client_instance = MockClass.return_value
        
        # Mock embed_batch to return dummy vectors of size 768
        def side_effect_embed(texts, *args, **kwargs):
            return [[0.1] * 768 for _ in texts]
            
        client_instance.embed_batch.side_effect = side_effect_embed
        client_instance.format_doc_for_embedding.side_effect = lambda t, c: f"title: {t} | text: {c}"
        
        yield client_instance

# --- Tests ---

def test_llm_client_mock():
    """Verify independent mock behavior before integration."""
    with patch("qmd.llm.httpx.Client") as mock_httpx:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}]
        }
        mock_httpx.return_value.post.return_value = mock_response
        
        client = LLMClient(embed_model="test-model")
        vecs = client.embed_batch(["hello"])
        
        assert len(vecs) == 1
        assert vecs[0] == [0.1, 0.2]
        
        # Verify payload structure
        args, kwargs = mock_httpx.return_value.post.call_args
        assert kwargs["json"]["input"] == ["hello"]
        assert kwargs["json"]["model"] == "test-model"

def test_index_file(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Create a temporary Markdown file, index it, and assert DB rows."""
    
    # 1. Setup Config & Store
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    
    # PASS db_conn HERE to avoid malformed image errors
    store = Store(config, connection=db_conn)
    
    # 2. Create File
    file1 = notes_dir / "doc1.md"
    file1.write_text("This is some test content for the vector database.")
    
    # 3. Index
    store.index_collection("test", config.collections["test"])
    
    # 4. Verify DB
    cursor = db_conn.cursor()
    
    # Check Content
    cursor.execute("SELECT count(*) FROM content")
    assert cursor.fetchone()[0] == 1
    
    # Check Documents
    cursor.execute("SELECT path, title FROM documents")
    row = cursor.fetchone()
    assert row[0] == "doc1.md"
    assert row[1] == "Doc1"
    
    # Check FTS
    cursor.execute("SELECT count(*) FROM documents_fts")
    assert cursor.fetchone()[0] == 1
    
    # Check Vectors
    cursor.execute("SELECT count(*) FROM vectors")
    assert cursor.fetchone()[0] == 1
    
    # Check LLM call
    mock_llm_client.embed_batch.assert_called_once()

def test_deduplication(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Index the same content in two different files. Verify content/vectors not duplicated."""
    
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    # PASS db_conn HERE
    store = Store(config, connection=db_conn)
    
    content = "Identical content appearing in two places."
    
    # File 1
    (notes_dir / "file_a.md").write_text(content)
    store.index_collection("test", config.collections["test"])
    
    # Reset mock to verify it's NOT called second time
    mock_llm_client.embed_batch.reset_mock()
    
    # File 2
    (notes_dir / "file_b.md").write_text(content)
    store.index_collection("test", config.collections["test"])
    
    cursor = db_conn.cursor()
    
    # Check Documents (Should be 2)
    cursor.execute("SELECT count(*) FROM documents")
    assert cursor.fetchone()[0] == 2
    
    # Check Content (Should be 1)
    cursor.execute("SELECT count(*) FROM content")
    assert cursor.fetchone()[0] == 1
    
    # Check Vectors (Should be 1 chunk set, since content is same)
    cursor.execute("SELECT count(*) FROM vectors")
    assert cursor.fetchone()[0] == 1
    
    # Ensure embed_batch was NOT called for the second file
    mock_llm_client.embed_batch.assert_not_called()

def test_file_update(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Modify a file and ensure DB updates."""
    
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    # PASS db_conn HERE
    store = Store(config, connection=db_conn)
    
    f = notes_dir / "updateme.md"
    f.write_text("Version 1")
    
    store.index_collection("test", config.collections["test"])
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT hash FROM documents WHERE path='updateme.md'")
    hash1 = cursor.fetchone()[0]
    
    # Update file
    f.write_text("Version 2 changed")
    store.index_collection("test", config.collections["test"])
    
    cursor.execute("SELECT hash FROM documents WHERE path='updateme.md'")
    hash2 = cursor.fetchone()[0]
    
    assert hash1 != hash2
    
    # Check FTS updated
    cursor.execute("SELECT body FROM documents_fts WHERE filepath='updateme.md'")
    assert cursor.fetchone()[0] == "Version 2 changed"


def test_file_move_rename(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test that moving/renaming a file updates paths without re-embedding."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    store = Store(config, connection=db_conn)
    
    # 1. Create and index file
    file1 = notes_dir / "old_name.md"
    file1.write_text("This is content that will be moved.")
    store.index_collection("test", config.collections["test"])
    
    # Verify initial state
    cursor = db_conn.cursor()
    cursor.execute("SELECT path, title FROM documents")
    row = cursor.fetchone()
    assert row[0] == "old_name.md"
    
    # 2. Rename file
    file2 = notes_dir / "new_name.md"
    file1.rename(file2)
    
    # Reset mock to ensure no LLM calls are made
    mock_llm_client.embed_batch.reset_mock()
    
    # 3. Re-index
    store.index_collection("test", config.collections["test"])
    
    # 4. Verify DB updated seamlessly
    cursor.execute("SELECT path, title FROM documents")
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "new_name.md"
    assert rows[0][1] == "New Name"
    
    cursor.execute("SELECT filepath, title FROM documents_fts")
    fts_row = cursor.fetchone()
    assert fts_row[0] == "new_name.md"
    assert fts_row[1] == "New Name"
    
    cursor.execute("SELECT filepath, title FROM chunks_fts")
    chunk_row = cursor.fetchone()
    assert chunk_row[0] == "new_name.md"
    assert chunk_row[1] == "New Name"
    
    # No new embeddings should have been generated
    mock_llm_client.embed_batch.assert_not_called()

def test_file_copy(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test that copying an existing file to a new path creates a new doc but doesn't re-embed."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    store = Store(config, connection=db_conn)
    
    # 1. Create and index file
    file1 = notes_dir / "original.md"
    file1.write_text("This content will be duplicated.")
    store.index_collection("test", config.collections["test"])
    
    # Reset mock
    mock_llm_client.embed_batch.reset_mock()
    
    # 2. Copy file
    file2 = notes_dir / "copy.md"
    file2.write_text("This content will be duplicated.")
    
    # 3. Re-index
    store.index_collection("test", config.collections["test"])
    
    # 4. Verify
    cursor = db_conn.cursor()
    cursor.execute("SELECT path FROM documents ORDER BY path")
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "copy.md"
    assert rows[1][0] == "original.md"
    
    # Content should be deduplicated
    cursor.execute("SELECT count(*) FROM content")
    assert cursor.fetchone()[0] == 1
    
    # No new embeddings should have been generated
    mock_llm_client.embed_batch.assert_not_called()

def test_indexing_with_quantization(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test indexing with int8 vector quantization configured."""
    notes_dir = tmp_path / "quant_notes"
    notes_dir.mkdir()

    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path),
        vector_quantization="int8"
    )
    store = Store(config, connection=db_conn)

    file1 = notes_dir / "quant_doc.md"
    file1.write_text("Testing int8 quantized vector storage.")

    store.index_collection("test", config.collections["test"])

    cursor = db_conn.cursor()
    cursor.execute("SELECT value FROM db_meta WHERE key='vector_quantization'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "int8"

    cursor.execute("SELECT count(*) FROM vectors")
    assert cursor.fetchone()[0] == 1

def test_indexing_error_tracking_and_retry(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test that conversion/API degradation records an indexing error and retries on subsequent index."""
    notes_dir = tmp_path / "err_notes"
    notes_dir.mkdir()

    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    store = Store(config, connection=db_conn)

    file1 = notes_dir / "doc_with_image.md"
    file1.write_text("Document containing an image that fails multimodal API.")

    # 1. First run: simulate auxiliary API error during conversion
    with patch("qmd.store.convert_to_markdown") as mock_convert:
        def convert_with_error(path, config=None, errors_out=None):
            if errors_out is not None:
                errors_out.append({"error_type": "multimodal_image_error", "message": "API timeout (504)"})
            return "Degraded text fallback"
        mock_convert.side_effect = convert_with_error

        store.index_collection("test", config.collections["test"])

    # Verify error was tracked
    errors = store.get_indexing_errors(collection="test")
    assert len(errors) == 1
    assert errors[0]["path"] == "doc_with_image.md"
    assert errors[0]["error_type"] == "multimodal_image_error"
    assert "504" in errors[0]["error_message"]

    # 2. Second run: file content hash on disk has NOT changed, but prior error triggers re-index
    mock_llm_client.embed_batch.reset_mock()
    with patch("qmd.store.convert_to_markdown") as mock_convert:
        # Second run succeeds cleanly
        mock_convert.side_effect = lambda path, config=None, errors_out=None: "Full rich markdown with image descriptions"

        store.index_collection("test", config.collections["test"])

    # Verify error entry was cleared on clean index
    errors_after = store.get_indexing_errors(collection="test")
    assert len(errors_after) == 0

    # Verify updated content was embedded
    mock_llm_client.embed_batch.assert_called()

def test_hard_indexing_failure_and_retry(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test that a hard exception during file processing records error and retries on next run."""
    notes_dir = tmp_path / "hard_err_notes"
    notes_dir.mkdir()

    config = Config(
        collections={"test": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    store = Store(config, connection=db_conn)

    file1 = notes_dir / "broken.md"
    file1.write_text("Content that will fail on initial pass.")

    # 1. Simulate hard exception during conversion
    with patch("qmd.store.convert_to_markdown", side_effect=RuntimeError("Parsing crash")):
        store.index_collection("test", config.collections["test"])

    errors = store.get_indexing_errors(collection="test")
    assert len(errors) == 1
    assert errors[0]["path"] == "broken.md"
    assert errors[0]["error_type"] == "indexing_error"
    assert "Parsing crash" in errors[0]["error_message"]

    # 2. Fix the crash on the next run
    store.index_collection("test", config.collections["test"])

    errors_fixed = store.get_indexing_errors(collection="test")
    assert len(errors_fixed) == 0

    cursor = db_conn.cursor()
    cursor.execute("SELECT path FROM documents WHERE collection='test'")
    assert cursor.fetchone()[0] == "broken.md"

def test_prune_orphaned_collections_cleans_errors(db_conn, temp_db_path, tmp_path, mock_llm_client):
    """Test that pruning removed collections also purges indexing error records."""
    notes_dir = tmp_path / "prune_err_notes"
    notes_dir.mkdir()

    config = Config(
        collections={"old_coll": CollectionConfig(path=str(notes_dir))},
        db_path=str(temp_db_path)
    )
    store = Store(config, connection=db_conn)

    file1 = notes_dir / "doc.md"
    file1.write_text("Test doc")

    with patch("qmd.store.convert_to_markdown") as mock_convert:
        def convert_with_error(path, config=None, errors_out=None):
            if errors_out is not None:
                errors_out.append({"error_type": "table_error", "message": "corrupted table"})
            return "doc content"
        mock_convert.side_effect = convert_with_error
        store.index_collection("old_coll", config.collections["old_coll"])

    assert len(store.get_indexing_errors(collection="old_coll")) == 1

    # Prune old_coll
    store.prune_orphaned_collections(active_collections=[])
    assert len(store.get_indexing_errors(collection="old_coll")) == 0
