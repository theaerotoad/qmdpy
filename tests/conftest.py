import pytest
import sqlite3
from pathlib import Path
from qmd.db import get_connection, init_schema

@pytest.fixture
def temp_db_path(tmp_path):
    """
    Provides a path to a database file within a temporary directory.
    Using tmp_path ensures .wal and .shm files can be created safely.
    """
    db_file = tmp_path / "test_qmd.db"
    return db_file

@pytest.fixture
def db_conn(temp_db_path):
    """Provides a configured database connection with schema initialized."""
    conn = get_connection(temp_db_path)
    init_schema(conn)
    yield conn
    conn.close()