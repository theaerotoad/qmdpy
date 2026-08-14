# QMD Python Adaptation - Technical Architecture & Usage Guide

## Installation & Quick Start

### 1. Installation
You can install `qmd` as an editable local package. This allows you to modify the source code and have changes take effect immediately.

```bash
# From the project root
pip install -e .


```

**Alternative (No Install):**
You can run the module directly without installing by setting the python path:

```bash
PYTHONPATH=src python3 -m qmd.main [command]


```

### 2. Configuration

Create a configuration file at `~/.config/qmd/index.yml`. You can now define global settings (LLM URL, chunking preferences) alongside your collections.

**Priority Order:** Environment Variables > YAML Config > Defaults.

```yaml
# Global Settings (Optional)
llm_url: "http://localhost:8080" # Default: [http://127.0.0.1:8888](http://127.0.0.1:8888)
strip_links: true                # Simplify [text](url) to just "text"
target_chunk_size: 1024          # Optimal character count per chunk
max_chunk_size: 2048             # Hard limit before forcing a split

# Collections
collections:
  my_notes:
    path: "/home/user/obsidian_vault"
    glob: "**/*.md"
  work:
    path: "/home/user/work_notes"
    glob: "*.md"


```

### 3. Environment Variables

QMD offloads heavy lifting to an OpenAI-compatible server (like `llama.cpp` or `Ollama`).

```bash
export QMD_LLM_URL="http://localhost:8080"  # Overrides YAML llm_url
export EMBED_MODEL="nomic-embed-text"       # Must match your server's model alias
export RERANK_MODEL="mxbai-rerank-xsmall"   # Optional
export GENERATE_MODEL="llama-3.2-3b"        # Optional


```

### 4. Usage Commands

**Indexing:**

```bash
# Standard Indexing (Incremental)
qmd update

# Pull git changes before indexing
qmd update --pull

# Force Re-index (Ignore content hashes, re-process everything)
qmd update --force


```

**Searching & Web UI:**

```bash
# Fast Local Search (Hybrid, FTS + Semantic Vector, No LLM)
qmd search "self improvement"

# Deep Search (Uses LLM to Rerank Results)
qmd search "architecture patterns" -r

# Session Search with Exclude Seen Chunks
qmd search "distributed systems" --session 8f3a1b9c --exclude-seen

# Filtered Search (By Collection, Title, or Path)
qmd search "docker" -c work -t "networking" -p "src/"

# Document View with Output Piping
qmd search "python async" --doc --json | jq '.'

# Lexical Override (Ignore vector nuance, force BM25 match)
qmd search "error codes" --lex "ERR_CONNECTION_REFUSED"

# Start Web UI
qmd serve --port 5000


```

---

## Motivation & Overview

We are re-implementing **QMD (Quick Markdown Search)**, originally a TypeScript/Node.js application, into a lean, modular **Python** CLI tool. The primary motivation is to create a maintainable, high-performance local search engine for personal knowledge bases (Markdown notes) that leverages modern AI capabilities without the heavy weight of local model inference.

By offloading the "heavy lifting" (Embedding generation, Chat completion, and Reranking) to an external, OpenAI-compatible API server (specifically one running `llama.cpp`/`llamaswap`), we can strip away complex native dependencies like `node-llama-cpp`. The result will be a lightweight client that focuses purely on file indexing, SQLite management, and providing a fast, intuitive CLI experience.

## STEP 1: ANALYSIS & STACK SELECTION

### 1. Scope Summary

* **Core Function:** CLI tool to index and search local Markdown files.
* **Smart Chunking:** Implements "Structure-Aware Chunking" (via internal `docparse` module) to respect Markdown headers, tables, and code blocks, ensuring context is preserved in every vector.
* **Search Methods:**
* **FTS:** Full-Text Search using SQLite FTS5 (BM25). Includes built-in stop word filtering to improve baseline relevance.
* **Vector:** Semantic search using cosine similarity on stored binary BLOBs.
* **Hybrid:** Reciprocal Rank Fusion (RRF) combining FTS and Vector results.
* **Rerank:** LLM-based re-ranking of top candidates via a custom `/v1/rerank` endpoint.
* **Constraints:** Minimal dependencies. No local inference.
* **Configuration:** YAML-based (`~/.config/qmd/index.yml`).
* **Configuration:** YAML-based (`~/.config/qmd/index.yml`) with support for stripping links and tuning chunk sizes.

### 2. Tech Stack Proposal

* **Language:** Python 3.10+
* **Database:** `sqlite3` (StdLib).
* **HTTP Client:** `httpx` (async support, minimal footprint).
* **Configuration:** `PyYAML`.
* **CLI Argument Parsing:** `argparse` (StdLib).
* **Progress Bars:** `tqdm` (for indexing feedback).
* **Output Formatting:** Standard ANSI escape codes.

### 3. Clarifications (Resolved)

* **Vector Storage:** Standard SQLite Table (BLOBs) containing packed floats.
* **Reranker:** Using a custom `/v1/rerank` endpoint on the LLM server.
* **Config Format:** Using YAML (requires `PyYAML`).

## STEP 2: ARCHITECTURAL BLUEPRINT

### A. Directory Structure

```
qmd_python/
├── pyproject.toml        # Build System & Dependencies
├── README.md
├── src/
│   └── qmd/
│       ├── __init__.py
│       ├── main.py       # CLI Entry Point & Command Dispatcher
│       ├── config.py     # YAML Config Loader
│       ├── db.py         # SQLite Connection & Schema Migrations
│       ├── store.py      # Business Logic: Indexing, Search, RRF
│       ├── llm.py        # API Client (Embed/Chat/Rerank)
│       ├── utils.py      # Hashing, Path Helpers
│       ├── formatting.py # CLI Output (Colors, Snippets)
│       ├── converters.py # Document Converters
│       ├── epub.py       # EPUB Converter
│       ├── web.py        # Web Server & Search API
│       ├── templates/
│       │   └── index.html # Web Search UI
│       └── docparse/     # Structure-Aware Markdown Parsing
│           ├── parser.py
│           ├── grouper.py
│           └── models.py
└── tests/
    ├── ...


```

### B. Component Specifications

**1. File:** `src/qmd/config.py`

* **Purpose:** Handles loading, validating, and saving the user configuration.
* **Key Dependencies:** `yaml` (PyYAML), `pathlib`, `os`.
* **Core Symbols:**
* `class Config`: Global settings (`llm_url`, `strip_links`, `max_chunk_size`) and collections.
* `def load_config() -> Config`: Reads `~/.config/qmd/index.yml`, applying logic (Env > YAML > Default).

**2. File:** `src/qmd/db.py`

* **Purpose:** Manages the SQLite database connection and schema.
* **Key Dependencies:** `sqlite3`.
* **Core Symbols:**
* `def get_connection(path: Path) -> sqlite3.Connection`: Connects with WAL mode enabled.
* `def init_schema(conn: sqlite3.Connection)`: Idempotent creation of tables.

**3. File:** `src/qmd/docparse/` (Module)

* **Purpose:** Replaces naive text splitting. Parses Markdown into semantic blocks (headers, content) and groups them into optimal chunks using dynamic programming.
* **Core Symbols:**
* `parser.parse_markdown_to_blocks()`: Extracts structure and strips links/images if configured.
* `grouper.group_blocks_into_chunks()`: Assembles chunks while preserving header context.

**4. File:** `src/qmd/llm.py`

* **Purpose:** Abstraction layer for the external AI server.
* **Key Dependencies:** `httpx`, `os`.
* **Core Symbols:**
* `class LLMClient`:
* `embed_batch()`: Batch calls to `/v1/embeddings`.
* `rerank()`: Calls custom `/v1/rerank` endpoint.

**5. File:** `src/qmd/store.py`

* **Purpose:** The central controller. Orchestrates file indexing and search algorithms.
* **Key Dependencies:** `db.py`, `llm.py`, `docparse`, `tqdm`.
* **Core Symbols:**
* `class Store`:
* `index_collection(..., force=False)`: Scans files, calls `docparse`, embeds, and updates DB.
* `prune_orphaned_collections()`: Removes data for collections removed from config.
* `hybrid_search(..., rerank_only=False)`: Runs FTS + Vec + RRF + Rerank. Handles deduplication and scoring.

**6. File:** `src/qmd/formatting.py`

* **Purpose:** Formats search results for the terminal.
* **Key Dependencies:** `sys` (for ANSI codes).
* **Core Symbols:**
* `def format_results_cli(results)`: Prints colorized output with snippets.
* `def format_doc_results_cli(grouped_results)`: Prints Document View with merged snippets.

**7. File:** `src/qmd/main.py`

* **Purpose:** Parses command line arguments and invokes `Store` methods.
* **Key Dependencies:** `argparse`, `store.py`.
* **Core Symbols:**
* `def main()`: Entry point.
* `def handle_search(args)`: Logic for `qmd search` (supports `-v`, `-d`, `-r`).
* `def handle_update(args)`: Logic for `qmd update` (supports `--pull`, `--force`).

## STEP 4: PROMPTS & TEMPLATES

The `llm.py` component implements specific formatting rules and prompts extracted from the original `qmd` source.

### 1. Embedding Formats (Nomic/Gemma Style)

When sending text to the `/v1/embeddings` endpoint, you MUST prepend these prefixes:

* **For Queries:** `task: search result | query: {query_text}`
* **For Documents:** `title: {title_or_none} | text: {chunk_text}`

### 2. Query Expansion Prompt

Used in `llm.py` -> `expand_query`. This leverages the Chat API to generate "lex" (keyword), "vec" (semantic), and "hyde" (hypothetical document) variations.

```
You are a search query optimization expert. Your task is to improve retrieval by rewriting queries and generating hypothetical documents.

Original Query: {original_query}

{optional_context_block}

## Step 1: Query Analysis
Identify entities, search intent, and missing context.

## Step 2: Generate Hypothetical Document
Write a focused sentence passage that would answer the query. Include specific terminology and domain vocabulary.

## Step 3: Query Rewrites
Generate 2-3 alternative search queries that resolve ambiguities. Use terminology from the hypothetical document.

## Step 4: Final Retrieval Text
Output exactly 1-3 'lex' lines, 1-3 'vec' lines, and MAX ONE 'hyde' line.

<format>
lex: {single search term}
vec: {single vector query}
hyde: {complete hypothetical document passage from Step 2 on a SINGLE LINE}
</format>


```

## Ideas for making this more useful down the line

* [x] Persistent search sessions (both for CLI, web, and an MCP version)
* [x] Chunk-level seen-state in requests (e.g. --exclude seen)
* [x] Novelty-aware follow-up retrieval (option to not share content that's been shared in earlier related searches)
* [ ] Explicit chunk or chunk-fetch by ID
* [ ] Cursor based paging (not sure yet how to best handle this)
* [ ] Return document structure (e.g. markdown headings and sizes)
* [ ] Return collection tree or subtree (for search narrowing)
* [ ] Straight-up keyword level searching or filtering for documents (grep style?)
* [ ] Allow (if available) additional document level or directory level summary / classification and metadata that could be used in searching or revealed when sharing results and documents ("File is a review document for XXXX..." "Directory contains PDF assembly drawings..." "Directory is listing of non-fiction books concerning epidemiological issues
* [ ] Order multi-file listings by original directory order
