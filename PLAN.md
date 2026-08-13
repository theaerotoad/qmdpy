# QMD Session Memory & Database Versioning Architecture Plan

This document outlines the architectural specification, database schema designs, and stage-by-stage implementation plan for adding session memory, query vector caching, and database state versioning to the QMD (Quick Markdown) search tool.

---

## 1. Executive Summary & Goals

### Objectives
1. **Database Quality-of-Life (QOL) Updates (`qmd.db`)**:
   - Track database schema version (`schema_version`) to ensure forward/backward compatibility before executing queries.
   - Maintain an automated `last_updated` timestamp tracking when documents or vectors were last modified, indexed, or pruned.

2. **Session Memory & Query Cache Database (`qmd-history.db`)**:
   - Cache query vector embeddings in a dedicated table to avoid re-calling LLM embedding endpoints for duplicate/repeated queries.
   - Maintain a persistent log of search sessions, session events (`search`, `doc_view`, etc.), and returned search results down to the individual chunk level (`seq_id`).
   - Implement an `--exclude-seen` flag during search to omit previously returned chunks from the active session, allowing unseen results to shift up into view.
   - Report a banner at the top of search outputs detailing the active Session ID and the count/documents of omitted chunks.

---

## 2. Database Schemas

### 2.1 Main Search Database (`qmd.db`) Changes

Update the `db_meta` key-value table:
- **`schema_version`**: Integer string representing DB schema version (Current Target: `"2"`).
- **`last_updated`**: ISO-8601 timestamp string (e.g. `2026-08-13T13:21:00Z`). Updated automatically whenever `index_collection` or `prune_orphaned_collections` modifies data.

```sql
-- Existing table in qmd.db
CREATE TABLE IF NOT EXISTS db_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Meta keys stored:
-- 'schema_version'   -> '2'
-- 'last_updated'     -> '2026-08-13T13:21:00Z'
-- 'vector_dim'       -> '768'
-- 'vector_quantization' -> 'int8'

```

---

### 2.2 History Database (`qmd-history.db`) Schema

The history database is completely isolated from the main document store to allow independent backup, clear-history commands, or shared history across multiple collections.

```sql
-- 1. Global Query Vector Cache (decoupled from sessions)
CREATE TABLE IF NOT EXISTS query_cache (
    query_text TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(query_text, embed_model)
);

-- 2. Sessions Table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,         -- Random 8-char hex (e.g., '8f3a1b9c')
    db_path TEXT NOT NULL,               -- Path to qmd.db used
    db_last_updated TEXT,                -- Snapshot of qmd.db's last_updated
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

-- 3. Session Events Table (broadened beyond just search)
CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,            -- 'search', 'doc_view', 'chunk_fetch', etc.
    query_text TEXT,
    lexical_query TEXT,
    db_path TEXT NOT NULL,
    db_last_updated TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 4. Session Results Table (records exact chunks shown)
CREATE TABLE IF NOT EXISTS session_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    collection TEXT NOT NULL,
    doc_path TEXT NOT NULL,
    seq_id INTEGER NOT NULL,            -- Chunk index (0..N) or -1 for document view
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY(event_id) REFERENCES session_events(id) ON DELETE CASCADE
);

-- Indexes for fast lookup on --exclude-seen queries
CREATE INDEX IF NOT EXISTS idx_session_results_lookup 
ON session_results(session_id, collection, doc_path, seq_id);

```

---

## 3. Session & Exclusion Logic Workflow

1. **Session Initialization**:
* If user passes `--session <id>`, use `<id>`.
* If `--session` is omitted, automatically generate a random 8-character hex string using `secrets.token_hex(4)` (e.g., `8f3a1b9c`).
* Create or touch the record in `sessions`.


2. **Executing Search with `--exclude-seen**`:
* Query `session_results` for all `(collection, doc_path, seq_id)` tuples associated with `session_id`.
* During `search_fts` and `search_vec`, filter out candidate chunks matching the seen set **before** RRF fusion / LLM reranking.
* Calculate excluded chunk count and affected document count.


3. **Document View (`--doc`) Granular Exclusion**:
* In document-grouped view, exclusion operates at the chunk level.
* Chunks already seen in previous queries are excluded from the document's snippet list.
* If all chunks of a document have been seen, the document is omitted entirely from top results.


4. **Output Banner**:
* Include a notice in CLI and JSON output:
`[Session: 8f3a1b9c | Excluded 4 previously seen chunk(s) across 2 document(s)]`



---

## 4. Stage-by-Stage Implementation Roadmap

Each stage is designed to touch 3-4 files max and be independently testable.

```
                  +-----------------------------------+
                  | Stage 1: Config, Main DB QOL      |
                  | & Compatibility Checks            |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 2: History DB Schema &      |
                  | Query Embedding Caching           |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 3: Session Tracking,        |
                  | Exclude-Seen & CLI Output         |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 4: Web UI / API Integration |
                  | & End-to-End Test Suite           |
                  +-----------------------------------+

```

---

### Stage 1: Config, Main DB QOL & Compatibility Checks

**Files Modified (4)**:

1. `example.yml`
2. `src/qmd/config.py`
3. `src/qmd/db.py`
4. `src/qmd/store.py`

**Tasks**:

* Add `history_db_path` option to `Config` and `example.yml` (default: `./qmd-history.db` or `~/.config/qmd/qmd-history.db`).
* Define `CURRENT_SCHEMA_VERSION = 2` constant in `db.py`.
* In `init_schema()`: set `schema_version` to `2` in `db_meta` if not present.
* Add `check_db_compatibility()` helper in `db.py` to raise `ValueError` if DB `schema_version` is newer than supported.
* Update `last_updated` timestamp in `db_meta` whenever indexing or pruning runs in `store.py`.

**Tests Added**:

* `tests/test_db_qol.py`:
* Test schema version written to `db_meta`.
* Test compatibility rejection on unsupported DB version.
* Test `last_updated` timestamp updates when indexing a file.



---

### Stage 2: History DB Schema & Query Embedding Caching

**Files Modified (3)**:

1. `src/qmd/db.py`
2. `src/qmd/store.py`
3. `src/qmd/utils.py`

**Tasks**:

* Add `init_history_schema()` and `get_history_connection()` in `db.py`.
* Implement `get_cached_query_embedding()` and `save_query_embedding()` in `db.py`.
* Initialize `self.history_conn` inside `Store.__init__()`.
* Wrap vector query embedding generation in `Store.get_query_embedding()` to check/populate `query_cache`.

**Tests Added**:

* `tests/test_query_cache.py`:
* Test first search calls embedder and saves vector to `query_cache`.
* Test subsequent search for same query text retrieves vector from cache without calling LLM embedder.



---

### Stage 3: Session Tracking, Exclude-Seen & CLI Output

**Files Modified (4)**:

1. `src/qmd/store.py`
2. `src/qmd/main.py`
3. `src/qmd/formatting.py`
4. `src/qmd/db.py`

**Tasks**:

* Add helper functions in `db.py` to log session events (`record_session_event`) and save returned results (`record_session_results`).
* Add `--session` (default random 8-char hex) and `--exclude-seen` flags to search subparser in `main.py`.
* Update `Store.hybrid_search` and `Store.wide_to_narrow_search` to accept `exclude_seen_set: Optional[Set[Tuple[str, str, int]]]`.
* Filter candidates matching `exclude_seen_set` prior to score fusion / reranking.
* Record session event and results in `handle_search()`.
* Update `format_results_cli`, `format_doc_results_cli`, and JSON formatters in `formatting.py` to display the active Session ID and exclusion stats.

**Tests Added**:

* `tests/test_exclude_seen.py`:
* Test session creation and 8-char random session ID generation.
* Test initial search records results in `session_results`.
* Test second search with `--exclude-seen` filters out previously returned chunks and returns new candidates.
* Test `--doc` mode chunk-level exclusion.



---

### Stage 4: Web UI / API Integration & End-to-End Validation

**Files Modified (3)**:

1. `src/qmd/web.py`
2. `src/qmd/templates/index.html`
3. `tests/test_live_connectivity.py` (or new `tests/test_web_session.py`)

**Tasks**:

* Update `/api/search` in `web.py` to accept `session_id` and `exclude_seen` query parameters.
* Return `session_id` and `excluded_count` in API JSON responses.
* Update `templates/index.html` UI to show active Session ID badge and an "Exclude Seen" toggle checkbox.

**Tests Added**:

* `tests/test_web_session.py`:
* Test API `/api/search` session parameter handling and JSON responses.



---

## 5. Verification Matrix

| Stage | Feature | Test File | Verification Command |
| --- | --- | --- | --- |
| **1** | DB Version & `last_updated` | `tests/test_db_qol.py` | `pytest tests/test_db_qol.py` |
| **2** | Query Vector Caching | `tests/test_query_cache.py` | `pytest tests/test_query_cache.py` |
| **3** | Session Memory & `--exclude-seen` | `tests/test_exclude_seen.py` | `pytest tests/test_exclude_seen.py` |
| **4** | Web UI & Full Pipeline | `tests/test_web_session.py` | `pytest tests/test_web_session.py` |

```
