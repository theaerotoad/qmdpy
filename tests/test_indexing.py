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
