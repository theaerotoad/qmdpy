import pytest
import sqlite3
from unittest.mock import MagicMock
from qmd.db import (
    init_schema, get_db_meta, set_db_meta, check_db_compatibility,
    update_db_last_updated, CURRENT_SCHEMA_VERSION
)
from qmd.store import Store
from qmd.config import Config, CollectionConfig

def test_schema_version_written(db_conn):
    init_schema(db_conn)
    version = get_db_meta(db_conn, "schema_version")
    assert version == str(CURRENT_SCHEMA_VERSION)

def test_last_updated_timestamp_initialized(db_conn):
    init_schema(db_conn)
    last_updated = get_db_meta(db_conn, "last_updated")
    assert last_updated is not None
    assert "T" in last_updated

def test_compatibility_rejection(db_conn):
    init_schema(db_conn)
    set_db_meta(db_conn, "schema_version", str(CURRENT_SCHEMA_VERSION + 1))
    
    with pytest.raises(ValueError, match="Unsupported database schema version"):
        check_db_compatibility(db_conn)

def test_last_updated_timestamp_updates_on_index(db_conn, tmp_path):
    init_schema(db_conn)
    initial_timestamp = "2020-01-01T00:00:00Z"
    set_db_meta(db_conn, "last_updated", initial_timestamp)
    
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    test_file = doc_dir / "test.md"
    test_file.write_text("# Test Document\n\nSome content here.")
    
    config = Config(db_path=str(tmp_path / "test.db"))
    store = Store(config=config, connection=db_conn)
    
    store.llm.embed_batch = MagicMock(return_value=[[0.1] * 768])
    
    col_cfg = CollectionConfig(path=str(doc_dir))
    store.index_collection("test_col", col_cfg)
    
    updated_timestamp = get_db_meta(db_conn, "last_updated")
    assert updated_timestamp != initial_timestamp
