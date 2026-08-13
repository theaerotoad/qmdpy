import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from qmd.utils import decompress_text

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

CURRENT_SCHEMA_VERSION = 2

def check_db_compatibility(conn: sqlite3.Connection):
    version_str = get_db_meta(conn, "schema_version")
    if version_str is not None:
        try:
            version = int(version_str)
            if version > CURRENT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported database schema version {version}. "
                    f"Current supported version is {CURRENT_SCHEMA_VERSION}."
                )
        except ValueError as e:
            if "Unsupported database schema version" in str(e):
                raise

def update_db_last_updated(conn: sqlite3.Connection):
    now = datetime.utcnow().isoformat() + "Z"
    set_db_meta(conn, "last_updated", now)

def register_functions(conn: sqlite3.Connection):
    try:
        conn.create_function("decompress_text", 1, decompress_text, deterministic=True)
    except Exception:
        pass

def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    if not HAS_SQLITE_VEC:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as e:
        print(f"Warning: Failed to load sqlite-vec extension: {e}")
        return False

def is_sqlite_vec_active(conn: sqlite3.Connection) -> bool:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT vec_version()")
        return cursor.fetchone() is not None
    except Exception:
        return False

def get_db_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM db_meta WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception:
        return None

def set_db_meta(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("""
        INSERT INTO db_meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, str(value)))

def ensure_vector_table(conn: sqlite3.Connection, dim: int, quant_type: str = "none"):
    quant_type = (quant_type or "none").lower()
    set_db_meta(conn, "vector_quantization", quant_type)
    set_db_meta(conn, "vector_dim", str(dim))

    if is_sqlite_vec_active(conn):
        if quant_type in ("bit", "binary"):
            col_type = f"bit[{dim}] distance_metric=hamming"
        elif quant_type in ("int8",):
            col_type = f"int8[{dim}] distance_metric=cosine"
        else:
            col_type = f"float[{dim}] distance_metric=cosine"

        conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
            rowid INTEGER PRIMARY KEY,
            embedding {col_type}
        );
        """)
    else:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL
        );
        """)

def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Connects to SQLite database.
    Sets isolation_level=None to disable Python's implicit transaction management.
    Enables WAL mode for performance.
    """
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    
    # Standard performance pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    
    register_functions(conn)
    load_sqlite_vec(conn)
    return conn

def init_history_schema(conn: sqlite3.Connection):
    cursor = conn.cursor()
    
    # 1. Global Query Vector Cache
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_cache (
        query_text TEXT NOT NULL,
        embed_model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(query_text, embed_model)
    );
    """)

    # 2. Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        db_path TEXT NOT NULL,
        db_last_updated TEXT,
        created_at TEXT NOT NULL,
        last_active_at TEXT NOT NULL
    );
    """)

    # 3. Session Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS session_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        query_text TEXT,
        lexical_query TEXT,
        db_path TEXT NOT NULL,
        db_last_updated TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """)

    # 4. Session Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS session_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        event_id INTEGER NOT NULL,
        collection TEXT NOT NULL,
        doc_path TEXT NOT NULL,
        seq_id INTEGER NOT NULL,
        rank INTEGER NOT NULL,
        score REAL NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
        FOREIGN KEY(event_id) REFERENCES session_events(id) ON DELETE CASCADE
    );
    """)

    # 5. Lookup index for excluded results
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_session_results_lookup 
    ON session_results(session_id, collection, doc_path, seq_id);
    """)

def get_history_connection(history_db_path: Path) -> sqlite3.Connection:
    if history_db_path.parent and not history_db_path.parent.exists():
        history_db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(history_db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")

    init_history_schema(conn)
    return conn

def get_cached_query_embedding(conn: sqlite3.Connection, query_text: str, embed_model: str) -> Optional[List[float]]:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM query_cache WHERE query_text = ? AND embed_model = ?", (query_text, embed_model))
        row = cursor.fetchone()
        if not row:
            try:
                cursor.execute("SELECT embedding FROM query_history WHERE query_text = ? AND embed_model = ?", (query_text, embed_model))
                row = cursor.fetchone()
            except Exception:
                pass
        if row and row[0]:
            now = datetime.now().isoformat()
            try:
                conn.execute("UPDATE query_cache SET updated_at = ? WHERE query_text = ? AND embed_model = ?", (now, query_text, embed_model))
            except Exception:
                pass
            blob = row[0]
            dim = len(blob) // 4
            return list(struct.unpack(f'{dim}f', blob))
    except Exception as e:
        print(f"Warning: Failed to fetch cached query embedding: {e}")
    return None

def save_query_embedding(conn: sqlite3.Connection, query_text: str, embed_model: str, vec: List[float]):
    try:
        now = datetime.now().isoformat()
        blob = struct.pack(f'{len(vec)}f', *vec)
        conn.execute("""
            INSERT INTO query_cache (query_text, embed_model, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(query_text, embed_model) DO UPDATE SET
                embedding = excluded.embedding,
                updated_at = excluded.updated_at
        """, (query_text, embed_model, blob, now, now))
    except Exception as e:
        print(f"Warning: Failed to save query embedding: {e}")

def init_schema(conn: sqlite3.Connection):
    """
    Idempotent creation of tables.
    """
    register_functions(conn)
    cursor = conn.cursor()

    # 0. Database Metadata Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS db_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)

    check_db_compatibility(conn)

    if get_db_meta(conn, "schema_version") is None:
        set_db_meta(conn, "schema_version", str(CURRENT_SCHEMA_VERSION))

    if get_db_meta(conn, "last_updated") is None:
        update_db_last_updated(conn)

    # 1. CAS: Content Addressable Storage
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS content (
        hash TEXT PRIMARY KEY,
        body TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    # 2. Metadata: File locations and status
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection TEXT NOT NULL,
        path TEXT NOT NULL,
        title TEXT NOT NULL,
        hash TEXT NOT NULL REFERENCES content(hash),
        modified_at TEXT NOT NULL,
        active INTEGER DEFAULT 1,
        UNIQUE(collection, path)
    );
    """)

    # 3. View for FTS5 External Content
    cursor.execute("DROP VIEW IF EXISTS document_search_view;")
    cursor.execute("""
    CREATE VIEW document_search_view AS
    SELECT 
        d.id AS id,
        d.collection AS collection,
        d.path AS filepath,
        d.title AS title,
        decompress_text(c.body) AS body
    FROM documents d
    JOIN content c ON d.hash = c.hash;
    """)

    # 4. FTS Index (BM25 with External Content Table)
    cursor.execute("SELECT sql FROM sqlite_master WHERE name='documents_fts';")
    sql_row = cursor.fetchone()

    # Drop table if schema does not use document_search_view
    if sql_row and sql_row[0] and "document_search_view" not in sql_row[0]:
        cursor.execute("DROP TABLE documents_fts;")

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        collection,
        filepath, 
        title, 
        body, 
        content='document_search_view',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    """)

    # 4. Vector Storage Initialization
    stored_dim = get_db_meta(conn, "vector_dim")
    stored_quant = get_db_meta(conn, "vector_quantization") or "none"
    if stored_dim:
        ensure_vector_table(conn, int(stored_dim), stored_quant)
    else:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL
        );
        """)

    # 5. Chunk Metadata
    cursor.execute("SELECT sql FROM sqlite_master WHERE name='chunk_metadata';")
    cm_sql = cursor.fetchone()
    if cm_sql and cm_sql[0] and "start_offset" in cm_sql[0]:
        cursor.execute("DROP TABLE chunk_metadata;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunk_metadata (
        rowid INTEGER PRIMARY KEY, -- FK to vectors.rowid
        doc_hash TEXT NOT NULL,
        seq_id INTEGER NOT NULL,
        chunk_text BLOB NOT NULL,
        FOREIGN KEY (rowid) REFERENCES vectors(rowid) ON DELETE CASCADE
    );
    """)

    # 6. Chunk-Level FTS5 Virtual Table
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        collection,
        filepath,
        title,
        body,
        headers,
        tokenize='porter unicode61'
    );
    """)

    # Populate chunks_fts automatically if empty but chunk_metadata has entries
    try:
        cursor.execute("SELECT COUNT(*) FROM chunks_fts;")
        fts_count = cursor.fetchone()[0]
        if fts_count == 0:
            cursor.execute("""
            INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers)
            SELECT 
                m.rowid, 
                d.collection, 
                d.path, 
                d.title, 
                decompress_text(m.chunk_text), 
                COALESCE(m.headers, '')
            FROM chunk_metadata m
            JOIN (
                SELECT hash, collection, path, title, MIN(id) 
                FROM documents 
                GROUP BY hash
            ) d ON m.doc_hash = d.hash;
            """)
    except Exception:
        pass