import pytest
import sqlite3
import struct
from pathlib import Path
from qmd.utils import compute_hash, handelize, chunk_text
from qmd.config import Config, load_config, CollectionConfig

# --- Utils Tests ---

def test_hashing():
    content = "Hello World"
    # echo -n "Hello World" | shasum -a 256
    expected = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
    assert compute_hash(content) == expected

def test_handelize():
    assert handelize("Hello World!") == "hello-world"
    assert handelize("path/to/File.md") == "path-to-file-md"
    assert handelize("  Weird... chars$$  ") == "weird-chars"

def test_text_compression():
    from qmd.utils import compress_text, decompress_text

    original = "This is a test markdown string for compression." * 10
    compressed = compress_text(original)
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(original.encode('utf-8'))
    assert decompress_text(compressed) == original

    # Test backward compatibility with plain uncompressed string
    assert decompress_text("uncompressed string") == "uncompressed string"

def test_chunking():
    text = "abcdefghijklmnopqrstuvwxyz"
    # Window 10, overlap 5
    chunks = chunk_text(text, window_size=10, overlap=5)
    
    assert len(chunks) > 1
    assert chunks[0] == "abcdefghij" # 0-10
    assert chunks[1] == "fghijklmno" # 5-15
    assert chunks[-1].endswith("z")

# --- Config Tests ---

def test_config_load_defaults(tmp_path):
    # Point to non-existent file
    cfg = load_config(tmp_path / "missing.yml")
    assert isinstance(cfg, Config)
    assert cfg.collections == {}

def test_config_load_file(tmp_path):
    config_file = tmp_path / "config.yml"
    config_content = """
collections:
  my_notes:
    path: /tmp/notes
    glob: "*.txt"
"""
    config_file.write_text(config_content)
    
    cfg = load_config(config_file)
    assert "my_notes" in cfg.collections
    assert cfg.collections["my_notes"].path == "/tmp/notes"
    assert cfg.collections["my_notes"].glob == "*.txt"

# --- DB Tests ---

def test_db_init(db_conn):
    """Verify tables are created."""
    cursor = db_conn.cursor()
    tables = [
        "db_meta",
        "content", 
        "documents", 
        "documents_fts", 
        "vectors", 
        "chunk_metadata"
    ]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='virtual table';")
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    for t in tables:
        assert any(t in existing for existing in existing_tables), f"Table {t} missing"

def test_vector_storage(db_conn):
    """Verify we can store and retrieve binary blobs in the vectors table."""
    cursor = db_conn.cursor()
    
    # Create dummy float vector (768 dims)
    vec = [0.1] * 768
    # Pack to binary (float is 4 bytes, so 768*4 = 3072 bytes)
    blob = struct.pack(f'{len(vec)}f', *vec)
    
    try:
        # Insert
        cursor.execute("INSERT INTO vectors(embedding) VALUES (?)", (blob,))
        
        # In WAL/Autocommit mode (isolation_level=None), this should be visible immediately
        cursor.execute("SELECT rowid, embedding FROM vectors LIMIT 1")
        row = cursor.fetchone()
        
        assert row is not None
        rowid, stored_blob = row
        
        # Verify integrity
        assert len(stored_blob) == 3072
        
        # Unpack and check value
        unpacked = struct.unpack(f'{768}f', stored_blob)
        assert len(unpacked) == 768
        # Float precision check
        assert abs(unpacked[0] - 0.1) < 1e-6
        
    except sqlite3.OperationalError as e:
        pytest.fail(f"Vector storage failed: {e}")


def test_vector_quantization_encoding():
    from qmd.store import encode_vector, decode_vector

    vec = [0.5, -0.25, 1.0, -1.0]

    # Test "none" (float32)
    encoded_none = encode_vector(vec, "none")
    decoded_none = decode_vector(encoded_none, 4, "none")
    assert len(decoded_none) == 4
    assert pytest.approx(decoded_none) == vec

    # Test "int8"
    encoded_int8 = encode_vector(vec, "int8")
    decoded_int8 = decode_vector(encoded_int8, 4, "int8")
    assert len(decoded_int8) == 4
    assert abs(decoded_int8[0] - 0.5) < 0.02

    # Test "bit" / "binary"
    encoded_bit = encode_vector(vec, "bit")
    decoded_bit = decode_vector(encoded_bit, 4, "bit")
    assert len(decoded_bit) == 4
    assert decoded_bit[0] == 1.0   # 0.5 > 0 -> 1.0
    assert decoded_bit[1] == -1.0  # -0.25 <= 0 -> -1.0