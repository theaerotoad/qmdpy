import time
import json
import sqlite3
import struct
import math
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from tqdm import tqdm

from qmd.db import (
    get_connection, init_schema, is_sqlite_vec_active, get_db_meta, ensure_vector_table,
    register_functions, get_history_connection, get_cached_query_embedding, save_query_embedding,
    update_db_last_updated, check_db_compatibility, CURRENT_SCHEMA_VERSION,
    get_cached_search_results, save_cached_search_results,
    record_indexing_error, clear_indexing_errors, get_indexing_errors
)
from qmd.config import Config, CollectionConfig
from qmd.llm import LLMClient
from qmd.utils import compute_hash, build_spacy_fts_queries, compress_text, decompress_text
from qmd.formatting import DIM, YELLOW, CYAN, RESET, MAGENTA
from qmd.converters import convert_to_markdown, SUPPORTED_EXTENSIONS

# Docparse integration
from qmd.docparse.parser import parse_markdown_to_blocks, extract_outline
from qmd.docparse.grouper import group_blocks_into_chunks
from qmd.docparse.models import Chunk

try:
    import dplib
except ImportError:
    dplib = None


def extract_document_date(file_path: Union[str, Path], markdown_body: str = "") -> Optional[str]:
    """
    Extracts or infers a document date using dplib from file path/filename and front matter/early content.
    """
    if dplib is None:
        return None

    path_str = str(file_path)
    sample_content = markdown_body[:4000] if markdown_body else ""

    try:
        if hasattr(dplib, "extract_date"):
            try:
                res = dplib.extract_date(path=path_str, content=sample_content)
            except TypeError:
                res = dplib.extract_date(path_str, sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(dplib, "parse_document_date"):
            try:
                res = dplib.parse_document_date(path=path_str, content=sample_content)
            except TypeError:
                res = dplib.parse_document_date(path_str, sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(dplib, "parse_date"):
            try:
                res = dplib.parse_date(path=path_str, text=sample_content)
            except TypeError:
                try:
                    res = dplib.parse_date(path_str, sample_content)
                except TypeError:
                    res = dplib.parse_date(path_str) or dplib.parse_date(sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(dplib, "parse_frontmatter") and hasattr(dplib, "parse_path"):
            res = dplib.parse_frontmatter(sample_content) or dplib.parse_path(path_str)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(dplib, "parse"):
            try:
                res = dplib.parse(path_str, sample_content)
            except TypeError:
                res = dplib.parse(path_str) or dplib.parse(sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
    except Exception:
        pass

    return None


def encode_vector(vec: List[float], quant_type: str = "none") -> bytes:
    quant_type = (quant_type or "none").lower()
    if quant_type in ("int8",):
        int8_vals = [max(-128, min(127, int(round(x * 127.0)))) for x in vec]
        return struct.pack(f'{len(vec)}b', *int8_vals)
    elif quant_type in ("bit", "binary"):
        num_bytes = (len(vec) + 7) // 8
        raw_bytes = bytearray(num_bytes)
        for i, val in enumerate(vec):
            if val > 0:
                raw_bytes[i // 8] |= (1 << (i % 8))
        return bytes(raw_bytes)
    else:
        return struct.pack(f'{len(vec)}f', *vec)


def _results_to_json(results: List['Result']) -> str:
    data = []
    for r in results:
        data.append({
            "path": r.path,
            "title": r.title,
            "text": r.text,
            "score": r.score,
            "source": r.source,
            "rank": r.rank,
            "collection": r.collection,
            "seq_id": r.seq_id,
            "headers": getattr(r, "headers", ""),
            "doc_date": getattr(r, "doc_date", None),
            "fts_score": getattr(r, "fts_score", None),
            "fts_rank": getattr(r, "fts_rank", None),
            "vec_score": getattr(r, "vec_score", None),
            "vec_rank": getattr(r, "vec_rank", None),
            "rrf_score": getattr(r, "rrf_score", None),
            "rrf_rank": getattr(r, "rrf_rank", None),
            "match_count": getattr(r, "match_count", 1),
        })
    return json.dumps(data)


def _json_to_results(json_str: str) -> List['Result']:
    data = json.loads(json_str)
    results = []
    for item in data:
        results.append(Result(
            path=item["path"],
            title=item["title"],
            text=item["text"],
            score=item["score"],
            source=item["source"],
            rank=item.get("rank"),
            collection=item.get("collection", ""),
            seq_id=item.get("seq_id", 0),
            headers=item.get("headers", ""),
            doc_date=item.get("doc_date"),
            fts_score=item.get("fts_score"),
            fts_rank=item.get("fts_rank"),
            vec_score=item.get("vec_score"),
            vec_rank=item.get("vec_rank"),
            rrf_score=item.get("rrf_score"),
            rrf_rank=item.get("rrf_rank"),
            match_count=item.get("match_count", 1),
        ))
    return results


def decode_vector(blob: bytes, dim: int, quant_type: str = "none") -> List[float]:
    quant_type = (quant_type or "none").lower()
    if quant_type in ("int8",):
        int8_vals = struct.unpack(f'{dim}b', blob)
        return [val / 127.0 for val in int8_vals]
    elif quant_type in ("bit", "binary"):
        floats = []
        for i in range(dim):
            byte_val = blob[i // 8]
            bit_set = (byte_val & (1 << (i % 8))) != 0
            floats.append(1.0 if bit_set else -1.0)
        return floats
    else:
        return list(struct.unpack(f'{dim}f', blob))

@dataclass
class Result:
    path: str
    title: str
    text: str
    score: float
    source: str  # 'fts', 'vec', 'hybrid'
    rank: Optional[int] = None
    collection: str = ""
    seq_id: int = 0  # 0 for FTS/whole doc, specific index for chunks
    headers: str = ""
    doc_date: Optional[str] = None
    fts_score: Optional[float] = None
    fts_rank: Optional[int] = None
    vec_score: Optional[float] = None
    vec_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    match_count: int = 1

class Store:
    def __init__(self, config: Config, connection: Optional[sqlite3.Connection] = None):
        self.config = config
        
        if connection:
            self.conn = connection
            check_db_compatibility(self.conn)
        else:
            db_path = Path(config.db_path) if config.db_path else Path.home() / ".config" / "qmd" / "qmd.db"
            self.conn = get_connection(db_path)
            init_schema(self.conn)
        
        history_db_path = Path(config.history_db_path) if config.history_db_path else Path.home() / ".config" / "qmd" / "qmd-history.db"
        self.history_conn = get_history_connection(history_db_path)
        self.last_exclusion_stats: Dict[str, int] = {"excluded_chunks": 0, "excluded_docs": 0}

        register_functions(self.conn)

        try:
            self.conn.execute("ALTER TABLE chunk_metadata ADD COLUMN headers TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

        self.llm = LLMClient(
            base_url=config.llm_url,
            api_key=getattr(config, "api_key", None),
            embed_url=getattr(config, "embed_url", None),
            rerank_url=getattr(config, "rerank_url", None),
            embed_api_key=getattr(config, "embed_api_key", None),
            rerank_api_key=getattr(config, "rerank_api_key", None),
            embed_model=config.embed_model,
            rerank_model=config.rerank_model,
            generate_model=config.generate_model,
            timeout=getattr(config, "request_timeout", 120.0)
        )

    def _build_search_cache_key(
        self,
        search_type: str,
        query: str,
        limit: int,
        rerank: bool,
        reranker_only: bool,
        collection: Optional[str],
        lexical_query: Optional[str],
        title: Optional[str],
        path: Optional[Union[str, List[str]]],
        fts_limit: Optional[int],
        vec_limit: Optional[int],
        rerank_candidates: Optional[int]
    ) -> str:
        db_last_updated = get_db_meta(self.conn, "last_updated") or ""
        embed_model = getattr(self.config, "embed_model", "")
        rerank_model = getattr(self.config, "rerank_model", "")

        if isinstance(path, (list, tuple, set)):
            path_val: Union[str, List[str]] = sorted([str(p).strip() for p in path if str(p).strip()])
        else:
            path_val = path or ""

        key_data = {
            "search_type": search_type,
            "query": query,
            "limit": limit,
            "rerank": rerank,
            "reranker_only": reranker_only,
            "collection": collection or "",
            "lexical_query": lexical_query or "",
            "title": title or "",
            "path": path_val,
            "fts_limit": fts_limit,
            "vec_limit": vec_limit,
            "rerank_candidates": rerank_candidates,
            "db_last_updated": db_last_updated,
            "embed_model": embed_model,
            "rerank_model": rerank_model
        }
        return compute_hash(json.dumps(key_data, sort_keys=True))

    def _get_collection_files(self, base_path: Path, collection_cfg: CollectionConfig) -> List[Path]:
        if collection_cfg.file_extensions:
            exts = [
                e.lower() if e.startswith('.') else f".{e.lower()}"
                for e in collection_cfg.file_extensions
            ]
            files = []
            for ext in exts:
                files.extend(base_path.glob(f"**/*{ext}"))
            seen = set()
            unique_files = []
            for f in files:
                if f.is_file() and f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            return unique_files
        else:
            candidates = list(base_path.glob(collection_cfg.glob))
            unique_files = []
            for f in candidates:
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if not collection_cfg.convert_non_md and ext not in {".md", ".markdown", ".txt"}:
                    continue
                if ext in SUPPORTED_EXTENSIONS or ext in {".md", ".markdown", ".txt"}:
                    unique_files.append(f)
            return unique_files

    def index_collection(self, name: str, collection_cfg: CollectionConfig, force: bool = False):
        """Scans files, detects changes, chunks, embeds, and updates DB."""
        base_path = Path(collection_cfg.path).expanduser().resolve()
        if not base_path.exists():
            print(f"Skipping {name}: Path not found {base_path}")
            return

        print(f"Indexing collection: {name} (Force={force})...")
        files = self._get_collection_files(base_path, collection_cfg)
        count_processed = 0
        count_skipped = 0
        found_rel_paths = {str(f.relative_to(base_path)) for f in files}

        # Wrap the file iterator with tqdm for a progress bar
        file_pbar = tqdm(files, desc=f"Processing {name}", unit="file")
        for file_path in file_pbar:
            rel_path = str(file_path.relative_to(base_path))
            disp_path = rel_path if len(rel_path) <= 30 else "..." + rel_path[-27:]
            file_pbar.set_postfix_str(disp_path)
            try:
                if self._process_file(name, base_path, file_path, found_rel_paths, force=force):
                    count_processed += 1
                else:
                    count_skipped += 1
            except Exception as e:
                # Use tqdm.write to print errors without breaking the progress bar layout
                tqdm.write(f"Error processing {file_path}: {e}")

        # Clean up files removed from disk within this collection
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM documents WHERE collection = ?", (name,))
        db_paths = {row[0] for row in cursor.fetchall()}
        stale_paths = db_paths - found_rel_paths
        if stale_paths:
            print(f"Cleaning up {len(stale_paths)} deleted file(s) from collection '{name}'...")
            for sp in stale_paths:
                cursor.execute("SELECT id FROM documents WHERE collection = ? AND path = ?", (name, sp))
                row = cursor.fetchone()
                if row:
                    doc_id = row[0]
                    cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
                    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.commit()
            self._cleanup_orphaned_data()

        if count_processed > 0 or stale_paths:
            update_db_last_updated(self.conn)

        print(f"Done. Processed: {count_processed}, Skipped/Unchanged: {count_skipped}")

    def prune_orphaned_collections(self, active_collections: List[str]):
        """Removes documents, FTS entries, and orphaned vectors/content for collections no longer in config."""
        cursor = self.conn.cursor()
        
        if not active_collections:
            # If no collections in config, everything in DB is an orphan
            cursor.execute("SELECT DISTINCT collection FROM documents")
        else:
            placeholders = ','.join(['?'] * len(active_collections))
            cursor.execute(
                f"SELECT DISTINCT collection FROM documents WHERE collection NOT IN ({placeholders})", 
                active_collections
            )
            
        orphans = [row[0] for row in cursor.fetchall()]
        
        if not orphans:
            self._cleanup_orphaned_data()
            return

        print(f"{YELLOW}Pruning removed collections: {orphans}{RESET}")
        
        for orphan in orphans:
            cursor.execute("SELECT id FROM documents WHERE collection = ?", (orphan,))
            orphan_ids = [row[0] for row in cursor.fetchall()]
            for oid in orphan_ids:
                cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (oid,))
                cursor.execute("DELETE FROM documents WHERE id = ?", (oid,))
            cursor.execute("DELETE FROM indexing_errors WHERE collection = ?", (orphan,))
            
        self.conn.commit()
        self._cleanup_orphaned_data()
        update_db_last_updated(self.conn)
        print(f"Pruned {len(orphans)} collection(s).")

    def _cleanup_orphaned_data(self):
        """Removes content, chunk metadata, vectors, and errors no longer referenced by any active document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM chunks_fts WHERE rowid IN (
                SELECT rowid FROM chunk_metadata WHERE doc_hash NOT IN (SELECT DISTINCT hash FROM documents)
            )
        """)
        cursor.execute("""
            DELETE FROM vectors WHERE rowid IN (
                SELECT rowid FROM chunk_metadata WHERE doc_hash NOT IN (SELECT DISTINCT hash FROM documents)
            )
        """)
        cursor.execute("""
            DELETE FROM chunk_metadata WHERE doc_hash NOT IN (SELECT DISTINCT hash FROM documents)
        """)
        cursor.execute("""
            DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)
        """)
        cursor.execute("""
            DELETE FROM indexing_errors WHERE collection NOT IN (SELECT DISTINCT collection FROM documents)
        """)
        self.conn.commit()

    def get_indexing_errors(self, collection: Optional[str] = None, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves list of documents that encountered errors or partial failures during indexing."""
        return get_indexing_errors(self.conn, collection=collection, path=path)

    def _process_file(self, collection_name: str, base_path: Path, file_path: Path, current_paths: set, force: bool = False) -> bool:
        rel_path = str(file_path.relative_to(base_path))
        try:
            raw_bytes = file_path.read_bytes()
        except Exception as e:
            record_indexing_error(
                self.conn,
                collection=collection_name,
                path=rel_path,
                doc_hash=None,
                error_type="file_read_error",
                error_message=str(e)
            )
            return False

        file_hash = compute_hash(raw_bytes)
        # Fix: Only replace underscores, preserve dashes for dates (e.g., 2019-12-20)
        title = file_path.stem.replace('_', ' ').title()
        
        has_prior_error = False
        if not force:
            check_cursor = self.conn.cursor()
            check_cursor.execute(
                "SELECT hash FROM documents WHERE collection = ? AND path = ?", 
                (collection_name, rel_path)
            )
            row = check_cursor.fetchone()
            if row:
                check_cursor.execute(
                    "SELECT count(*) FROM indexing_errors WHERE collection = ? AND path = ?",
                    (collection_name, rel_path)
                )
                err_count = check_cursor.fetchone()[0]
                has_prior_error = (err_count > 0)
            check_cursor.close()

            if row and row[0] == file_hash and not has_prior_error:
                return False

        conversion_errors: List[dict] = []
        self.conn.execute("BEGIN")
        cursor = self.conn.cursor()
        try:
            now = datetime.now().isoformat()
            
            # Check for move/rename: same hash, but existing path is no longer in current_paths
            if not force and not has_prior_error:
                cursor.execute(
                    "SELECT id, path FROM documents WHERE collection = ? AND hash = ?",
                    (collection_name, file_hash)
                )
                existing_docs = cursor.fetchall()
                for doc_id, old_path in existing_docs:
                    if old_path not in current_paths:
                        # This is a move/rename. Update metadata to reflect new path.
                        cursor.execute("SELECT body FROM content WHERE hash = ?", (file_hash,))
                        c_row = cursor.fetchone()
                        markdown_body = decompress_text(c_row[0]) if c_row else ""
                        doc_date = extract_document_date(file_path, markdown_body)

                        cursor.execute("""
                            UPDATE documents SET path = ?, title = ?, modified_at = ?, doc_date = ?
                            WHERE id = ?
                        """, (rel_path, title, now, doc_date, doc_id))
                        
                        cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
                        
                        cursor.execute(
                            "INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)",
                            (doc_id, collection_name, rel_path, title, markdown_body)
                        )
                        
                        # Update chunks_fts to point to the new path/title
                        cursor.execute("""
                            UPDATE chunks_fts SET filepath = ?, title = ?
                            WHERE rowid IN (SELECT rowid FROM chunk_metadata WHERE doc_hash = ?)
                              AND collection = ?
                        """, (rel_path, title, file_hash, collection_name))
                        
                        self.conn.commit()
                        clear_indexing_errors(self.conn, collection_name, rel_path)
                        return True

            cursor.execute("SELECT id FROM documents WHERE collection = ? AND path = ?", (collection_name, rel_path))
            existing_doc = cursor.fetchone()

            # Delete from documents_fts before updating content so FTS5 external view unindexes clean tokens
            if existing_doc:
                doc_id = existing_doc[0]
                cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))

            cursor.execute("SELECT body FROM content WHERE hash = ?", (file_hash,))
            content_row = cursor.fetchone()

            if content_row and not has_prior_error:
                content_exists = True
                markdown_body = decompress_text(content_row[0])
            else:
                content_exists = False
                markdown_body = convert_to_markdown(file_path, config=self.config, errors_out=conversion_errors)
                cursor.execute("""
                    INSERT INTO content (hash, body, created_at) VALUES (?, ?, ?)
                    ON CONFLICT(hash) DO UPDATE SET body = excluded.body, created_at = excluded.created_at
                """, (file_hash, compress_text(markdown_body), now))

            doc_date = extract_document_date(file_path, markdown_body)

            if existing_doc:
                cursor.execute("""
                    UPDATE documents SET hash = ?, modified_at = ?, title = ?, doc_date = ?
                    WHERE id = ?
                """, (file_hash, now, title, doc_date, doc_id))
            else:
                cursor.execute("""
                    INSERT INTO documents (collection, path, title, hash, modified_at, doc_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (collection_name, rel_path, title, file_hash, now, doc_date))
                doc_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)",
                (doc_id, collection_name, rel_path, title, markdown_body)
            )

            if not content_exists:
                self._generate_and_store_embeddings(cursor, file_hash, title, markdown_body, collection_name, rel_path)

            self.conn.commit()

            # Record or clear indexing errors
            clear_indexing_errors(self.conn, collection_name, rel_path)
            if conversion_errors:
                for err in conversion_errors:
                    record_indexing_error(
                        self.conn,
                        collection=collection_name,
                        path=rel_path,
                        doc_hash=file_hash,
                        error_type=err.get("error_type", "conversion_error"),
                        error_message=err.get("message", "")
                    )
            return True
        except Exception as e:
            self.conn.rollback()
            record_indexing_error(
                self.conn,
                collection=collection_name,
                path=rel_path,
                doc_hash=file_hash,
                error_type="indexing_error",
                error_message=str(e)
            )
            raise e
        finally:
            cursor.close()

    def _generate_and_store_embeddings(self, cursor: sqlite3.Cursor, doc_hash: str, title: str, markdown_body: str, collection_name: str = "", rel_path: str = ""):
        # 1. Parse markdown using docparse, passing the strip_links config
        blocks, _ = parse_markdown_to_blocks(content=markdown_body, strip_links=self.config.strip_links)
        if not blocks:
            return

        # 2. Group into chunks
        chunks = group_blocks_into_chunks(
            blocks, 
            max_chunk_size=self.config.max_chunk_size, 
            target_chunk_size=self.config.target_chunk_size
        )

        if not chunks:
            return

        # 3. Format chunks with structural context
        embedding_texts = []
        final_chunk_texts = []
        chunk_headers = []

        for chunk in chunks:
            # Construct breadcrumbs from parent headers, cleaning any leading '#' symbols
            clean_parents = [re.sub(r'^\s*#+\s*', '', h).strip() for h in chunk.parent_headers.values() if h and h.strip()]
            context_str = " > ".join(clean_parents)
            
            # Combine context with content for the embedding model
            if context_str:
                text_to_embed = f"Context: {context_str}\n\n{chunk.content}"
            else:
                text_to_embed = chunk.content
            
            formatted = self.llm.format_doc_for_embedding(title, text_to_embed)
            
            embedding_texts.append(formatted)
            final_chunk_texts.append(chunk.content)
            chunk_headers.append(context_str)

        # 4. Generate Embeddings
        batch_size = getattr(self.config, "embed_batch_size", 16)
        disp_name = rel_path if rel_path else title
        desc_str = f"  Chunking {disp_name[:25]}" if len(disp_name) > 25 else f"  Chunking {disp_name}"
        embeddings = self.llm.embed_batch(
            embedding_texts, 
            batch_size=batch_size, 
            show_progress=True, 
            desc=desc_str
        )
        if not embeddings:
            return

        dim = len(embeddings[0])
        stored_quant = get_db_meta(self.conn, "vector_quantization")
        quant_type = stored_quant or getattr(self.config, "vector_quantization", "none") or "none"
        ensure_vector_table(self.conn, dim=dim, quant_type=quant_type)
        
        # 5. Store in Vector Table and Chunk-Level FTS
        cursor.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunk_metadata WHERE doc_hash = ?)", (doc_hash,))
        cursor.execute("DELETE FROM chunk_metadata WHERE doc_hash = ?", (doc_hash,))
        
        for i, (chunk_text, embedding, context_str) in enumerate(zip(final_chunk_texts, embeddings, chunk_headers)):
            emb_blob = encode_vector(embedding, quant_type=quant_type)
            cursor.execute("INSERT INTO vectors(embedding) VALUES (?)", (emb_blob,))
            vector_rowid = cursor.lastrowid
            cursor.execute("""
                INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers)
                VALUES (?, ?, ?, ?, ?)
            """, (vector_rowid, doc_hash, i, compress_text(chunk_text), context_str))
            cursor.execute("""
                INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (vector_rowid, collection_name, rel_path, title, chunk_text, context_str))

    def search_fts(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None) -> List[Result]:
        """Lexical search directly on chunks using chunks_fts."""
        limit = limit if limit is not None else getattr(self.config, 'fts_limit', 50)
        # Allow pre-formatted FTS query expressions (quotes, AND, OR, NOT) to pass through directly
        if '"' in query or ' AND ' in query or ' OR ' in query or ' NOT ' in query:
            fts_query = query
        else:
            # Sanitize inputs by removing double quotes that could break syntax
            sanitized = query.replace('"', '')
            
            # Split into individual terms to create an intersection query (AND logic)
            raw_terms = sanitized.split()
            
            # Filter out common stop words to prevent skewed FTS rankings
            stop_words = {
                "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", 
                "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", 
                "that", "the", "their", "then", "there", "these", "they", "this", 
                "to", "was", "will", "with"
            }
            terms = [t for t in raw_terms if t.lower() not in stop_words]
            
            # Fallback to raw terms if all words were filtered out 
            if not terms:
                terms = raw_terms
                
            if not terms:
                return []
                
            # Wrap terms in quotes to handle special chars, join with AND
            fts_query = " AND ".join([f'"{term}"' for term in terms])
        
        cursor = self.conn.cursor()

        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]
        
        base_sql = """
            SELECT f.filepath, f.title, f.body, f.rank, f.collection, f.rowid, m.seq_id, COALESCE(f.headers, '')
            FROM chunks_fts f
            JOIN chunk_metadata m ON f.rowid = m.rowid
            WHERE chunks_fts MATCH ?
        """
        filters_sql = ""
        if collection: filters_sql += " AND f.collection = ?"
        if title: filters_sql += " AND f.title LIKE ?"
        if paths:
            path_clauses = " OR ".join(["f.filepath LIKE ?" for _ in paths])
            filters_sql += f" AND ({path_clauses})"
        
        order_sql = " ORDER BY f.rank LIMIT ?"
        
        # Scale up candidate over-fetch limit so seen chunks don't starve candidate quorum
        sql_limit = max(limit * 5, limit + (len(exclude_seen_set) * 2 if exclude_seen_set else 0)) if exclude_seen_set else limit

        def build_params(q_val):
            p = [q_val]
            if collection: p.append(collection)
            if title: p.append(f"%{title}%")
            for p_val in paths:
                p.append(f"%{p_val}%")
            p.append(sql_limit)
            return p

        try:
            cursor.execute(base_sql + filters_sql + order_sql, tuple(build_params(fts_query)))
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Fallback: if complex syntax fails, try raw phrase matching once more
            try:
                cursor.execute(base_sql + filters_sql + order_sql, tuple(build_params(f'"{sanitized}"')))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                print(f"{YELLOW}FTS Warning (Skipping term '{query}'): {e}{RESET}")
                return []

        results = []
        for row in rows:
            doc_path, doc_title, chunk_text, rank, coll, rowid, seq_id, headers = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
            
            key = (coll or "", doc_path, seq_id)
            if exclude_seen_set and key in exclude_seen_set:
                if excluded_chunks_tracker is not None:
                    excluded_chunks_tracker.add(key)
                continue

            # Convert SQLite FTS5 BM25 negative rank into a positive score
            raw_bm25 = -float(rank) if float(rank) < 0 else float(rank)
            calculated_fts_score = max(0.0001, raw_bm25)
            calculated_fts_rank = len(results) + 1

            results.append(Result(
                path=doc_path,
                title=doc_title,
                text=chunk_text,
                score=calculated_fts_score,
                source="fts",
                collection=coll or "",
                seq_id=seq_id,
                headers=headers,
                fts_score=calculated_fts_score,
                fts_rank=calculated_fts_rank
            ))
            if len(results) >= limit:
                break

        return results

    def get_query_embedding(self, formatted_query: str) -> List[float]:
        embed_model = getattr(self.config, "embed_model", "EmbeddingGemma 300m")
        cached_vec = get_cached_query_embedding(self.history_conn, formatted_query, embed_model)
        if cached_vec is not None:
            return cached_vec

        vecs = self.llm.embed_batch([formatted_query])
        if vecs:
            save_query_embedding(self.history_conn, formatted_query, embed_model, vecs[0])
            return vecs[0]
        return []

    def search_vec(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None) -> List[Result]:
        limit = limit if limit is not None else getattr(self.config, 'vec_limit', 50)
        query_text = self.llm.format_query_for_embedding(query)
        query_vec = self.get_query_embedding(query_text)
        if not query_vec:
            return []
        
        cursor = self.conn.cursor()
        stored_quant = get_db_meta(self.conn, "vector_quantization")
        quant_type = (stored_quant or getattr(self.config, "vector_quantization", "none") or "none").lower()

        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]

        # Check if sqlite-vec is active and virtual table is set up
        if is_sqlite_vec_active(self.conn):
            try:
                query_blob = encode_vector(query_vec, quant_type)
                extra_seen = (len(exclude_seen_set) * 2) if exclude_seen_set else 0
                k_val = (limit * 5 + extra_seen) if (collection or title or paths or exclude_seen_set) else limit

                query_sql = """
                    SELECT v.rowid, v.distance, m.chunk_text, d.path, d.title, d.collection, m.seq_id, COALESCE(m.headers, '')
                    FROM vectors v
                    JOIN chunk_metadata m ON v.rowid = m.rowid
                    JOIN documents d ON m.doc_hash = d.hash
                    WHERE v.embedding MATCH ? AND v.k = ?
                """
                params = [query_blob, k_val]

                if collection:
                    query_sql += " AND d.collection = ?"
                    params.append(collection)
                if title:
                    query_sql += " AND d.title LIKE ?"
                    params.append(f"%{title}%")
                if paths:
                    path_clauses = " OR ".join(["d.path LIKE ?" for _ in paths])
                    query_sql += f" AND ({path_clauses})"
                    for p_val in paths:
                        params.append(f"%{p_val}%")

                cursor.execute(query_sql, tuple(params))
                candidates = []
                for rowid, dist, text_blob, doc_path, doc_title, coll, seq_id, hdrs in cursor.fetchall():
                    key = (coll or "", doc_path, seq_id)
                    if exclude_seen_set and key in exclude_seen_set:
                        if excluded_chunks_tracker is not None:
                            excluded_chunks_tracker.add(key)
                        continue
                    if quant_type in ("bit", "binary"):
                        score = 1.0 / (1.0 + float(dist))
                    else:
                        score = max(0.0, 1.0 - float(dist))

                    candidates.append(Result(
                        path=doc_path, title=doc_title, text=decompress_text(text_blob), score=score,
                        source="vec", collection=coll, seq_id=seq_id, headers=hdrs
                    ))

                candidates.sort(key=lambda x: x.score, reverse=True)
                for vec_idx, c in enumerate(candidates):
                    c.vec_score = c.score
                    c.vec_rank = vec_idx + 1
                return candidates[:limit]
            except sqlite3.OperationalError:
                pass

        # Fallback: Python vector comparison
        query_sql = """
            SELECT v.embedding, m.chunk_text, d.path, d.title, d.collection, m.seq_id, COALESCE(m.headers, '')
            FROM vectors v
            JOIN chunk_metadata m ON v.rowid = m.rowid
            JOIN documents d ON m.doc_hash = d.hash
        """
        where_clauses = []
        params = []
        if collection:
            where_clauses.append("d.collection = ?")
            params.append(collection)
        if title:
            where_clauses.append("d.title LIKE ?")
            params.append(f"%{title}%")
        if paths:
            where_clauses.append("(" + " OR ".join(["d.path LIKE ?" for _ in paths]) + ")")
            for p_val in paths:
                params.append(f"%{p_val}%")
            
        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)
            
        cursor.execute(query_sql, tuple(params))
        
        dim = len(query_vec)
        candidates = []
        for emb_blob, text_blob, doc_path, doc_title, coll, seq_id, hdrs in cursor.fetchall():
            key = (coll or "", doc_path, seq_id)
            if exclude_seen_set and key in exclude_seen_set:
                if excluded_chunks_tracker is not None:
                    excluded_chunks_tracker.add(key)
                continue
            vec = decode_vector(emb_blob, dim, quant_type)
            
            dot_prod = sum(a * b for a, b in zip(query_vec, vec))
            mag_q = math.sqrt(sum(a * a for a in query_vec))
            mag_v = math.sqrt(sum(a * a for a in vec))
            sim = dot_prod / (mag_q * mag_v) if mag_q and mag_v else 0
            
            candidates.append(Result(
                path=doc_path, title=doc_title, text=decompress_text(text_blob), score=sim, 
                source="vec", collection=coll, seq_id=seq_id, headers=hdrs
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        for vec_idx, c in enumerate(candidates):
            c.vec_score = c.score
            c.vec_rank = vec_idx + 1
        return candidates[:limit]

    def hybrid_search(
        self, 
        query: str, 
        limit: int = 10, 
        verbose: bool = False, 
        rerank: bool = False, 
        reranker_only: bool = False, 
        collection: Optional[str] = None, 
        lexical_query: Optional[str] = None, 
        title: Optional[str] = None, 
        path: Optional[Union[str, List[str]]] = None,
        fts_limit: Optional[int] = None,
        vec_limit: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
        exclude_seen_set: Optional[set] = None,
        use_cache: Optional[bool] = None
    ) -> List[Result]:
        t_total_start = time.perf_counter()
        excluded_chunks_tracker: set = set()
        should_cache = use_cache if use_cache is not None else getattr(self.config, 'cache_search_results', True)
        
        if should_cache and not exclude_seen_set:
            t_cache_start = time.perf_counter()
            cache_key = self._build_search_cache_key(
                "hybrid", query, limit, rerank, reranker_only, collection, lexical_query, title, path, fts_limit, vec_limit, rerank_candidates
            )
            cached_json = get_cached_search_results(self.history_conn, cache_key)
            t_cache = (time.perf_counter() - t_cache_start) * 1000
            if cached_json:
                self.last_exclusion_stats = {"excluded_chunks": 0, "excluded_docs": 0}
                if verbose:
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
                    print(f"{DIM}Original Query:{RESET} {query}")
                    print(f"{GREEN}[Cache Hit]{RESET} Loaded results from search cache in {t_cache:.2f}ms")
                    print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                    print(f"  • Cache Lookup:            {t_cache:>7.2f} ms")
                    print(f"  • Total Hybrid Search:     {t_total:>7.2f} ms")
                    print(f"{CYAN}------------------------{RESET}\n")
                return _json_to_results(cached_json)

        fts_lim = fts_limit if fts_limit is not None else getattr(self.config, 'fts_limit', 50)
        vec_lim = vec_limit if vec_limit is not None else getattr(self.config, 'vec_limit', 50)
        rr_cand = rerank_candidates if rerank_candidates is not None else getattr(self.config, 'rerank_candidates', 20)

        t_expand_start = time.perf_counter()
        # Generate default spaCy FTS expanded lexical queries
        if lexical_query:
            lex_queries = [lexical_query]
        else:
            lex_queries = build_spacy_fts_queries(query, model_name=self.config.spacy_model)

        vec_queries = [query]
        t_expand = (time.perf_counter() - t_expand_start) * 1000

        fts_results = []
        seen_fts_keys = set()
        min_fts_quorum = 3

        t_fts_start = time.perf_counter()
        # Gated FTS Execution: Run strict conjunctions first; stop if quorum reached
        for q_idx, lq in enumerate(lex_queries):
            tier_results = self.search_fts(lq, limit=fts_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker)
            
            for r in tier_results:
                key = (r.collection, r.path, r.seq_id)
                if key not in seen_fts_keys:
                    seen_fts_keys.add(key)
                    # Assign global sequential FTS rank to maintain strict ordering across tiers
                    r.fts_rank = len(fts_results) + 1
                    if q_idx > 0:
                        tier_penalty = 0.85 ** q_idx
                        r.score *= tier_penalty
                        if r.fts_score is not None:
                            r.fts_score *= tier_penalty
                    fts_results.append(r)

            if len(fts_results) >= min_fts_quorum:
                break

        # Relaxation Fallback: If strict queries yielded fewer than min_fts_quorum hits, try OR fallback
        if len(fts_results) < min_fts_quorum:
            clean_q = re.sub(r'[^\w\s]', ' ', query.lower())
            stop_words = {
                "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", 
                "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", 
                "that", "the", "their", "then", "there", "these", "they", "this", 
                "to", "was", "will", "with"
            }
            terms = [t for t in clean_q.split() if t not in stop_words and len(t) > 1]
            if len(terms) > 1:
                or_query = " OR ".join([f'"{t}"' for t in terms])
                or_results = self.search_fts(or_query, limit=fts_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker)
                for r in or_results:
                    key = (r.collection, r.path, r.seq_id)
                    if key not in seen_fts_keys:
                        seen_fts_keys.add(key)
                        r.fts_rank = len(fts_results) + 1
                        r.score *= 0.5  # Heavy penalty for OR fallback matches
                        if r.fts_score is not None:
                            r.fts_score *= 0.5
                        fts_results.append(r)
        t_fts = (time.perf_counter() - t_fts_start) * 1000

        t_vec_start = time.perf_counter()
        vec_results = []
        for vq in vec_queries:
            vec_results.extend(self.search_vec(vq, limit=vec_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker))
        t_vec = (time.perf_counter() - t_vec_start) * 1000

        self.last_exclusion_stats = {
            "excluded_chunks": len(excluded_chunks_tracker),
            "excluded_docs": len({(coll, p) for (coll, p, seq) in excluded_chunks_tracker})
        }

        t_rrf_start = time.perf_counter()
        rrf_scores: Dict[Tuple[str, str, int], float] = {}
        metadata: Dict[Tuple[str, str, int], Result] = {}
        sources_found: Dict[Tuple[str, str, int], set] = {}
        
        def apply_rrf(results: List[Result], source_type: str, k: int = 60):
            for rank, res in enumerate(results):
                key = (res.collection, res.path, res.seq_id)
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                    metadata[key] = res
                    sources_found[key] = set()
                else:
                    existing = metadata[key]
                    if res.fts_score is not None and existing.fts_score is None:
                        existing.fts_score = res.fts_score
                        existing.fts_rank = res.fts_rank
                    if res.vec_score is not None and existing.vec_score is None:
                        existing.vec_score = res.vec_score
                        existing.vec_rank = res.vec_rank

                sources_found[key].add(source_type)
                
                score = 1.0 / (k + rank)
                if rank == 0: score += 0.05
                elif rank < 3: score += 0.02
                rrf_scores[key] += score

        apply_rrf(fts_results, "fts")
        apply_rrf(vec_results, "vec")

        # Dual-Match Quorum Boost: Boost candidates present in BOTH FTS and Vector sets
        for key, srcs in sources_found.items():
            if "fts" in srcs and "vec" in srcs:
                rrf_scores[key] *= 1.35

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for rrf_idx, (key, rrf_score) in enumerate(fused):
            res = metadata[key]
            res.rrf_score = rrf_score
            res.rrf_rank = rrf_idx + 1

        top_candidates = [metadata[key] for key, score in fused[:rr_cand]]
        t_rrf = (time.perf_counter() - t_rrf_start) * 1000

        if not top_candidates:
            if verbose:
                t_total = (time.perf_counter() - t_total_start) * 1000
                print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
                print(f"{DIM}Query:{RESET} {query}")
                print(f"{YELLOW}Lexical (FTS):{RESET} {lex_queries}")
                print(f"{MAGENTA}Vector (Semantic):{RESET} {vec_queries}")
                print(f"{DIM}Candidates: FTS={len(fts_results)} | Vector={len(vec_results)} | RRF=0{RESET}")
                print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                print(f"  • Query Expansion:         {t_expand:>7.2f} ms")
                print(f"  • Lexical (FTS) Search:    {t_fts:>7.2f} ms ({len(fts_results)} hits)")
                print(f"  • Vector (Semantic) Search:{t_vec:>7.2f} ms ({len(vec_results)} hits)")
                print(f"  • RRF Fusion:              {t_rrf:>7.2f} ms")
                print(f"  • Total Search:            {t_total:>7.2f} ms")
                print(f"{CYAN}------------------------{RESET}\n")
            return []

        final_results = []
        t_rerank = 0.0

        if not rerank and not reranker_only:
            # Skip reranking entirely
            for res in top_candidates:
                res.score = rrf_scores[(res.collection, res.path, res.seq_id)]
                res.source = "hybrid"
                final_results.append(res)
        else:
            t_rerank_start = time.perf_counter()
            rerank_docs = [f"File: {c.path}\nContent: {c.text}" for c in top_candidates]
            rerank_results = self.llm.rerank(query, rerank_docs)
            
            raw_rrf_list = [rrf_scores[(c.collection, c.path, c.seq_id)] for c in top_candidates]
            raw_rerank_list = []
            for i, res in enumerate(top_candidates):
                r_score = 0.0
                for r in rerank_results:
                    if r.get('index') == i:
                        r_score = float(r.get('score', 0.0))
                        break
                raw_rerank_list.append(r_score)

            # Min-Max Normalize RRF and Reranker scores across the candidate batch
            min_rrf, max_rrf = min(raw_rrf_list), max(raw_rrf_list)
            norm_rrf = [(s - min_rrf) / (max_rrf - min_rrf) if max_rrf > min_rrf else 1.0 for s in raw_rrf_list]

            min_rerank, max_rerank = min(raw_rerank_list), max(raw_rerank_list)
            norm_rerank = [(s - min_rerank) / (max_rerank - min_rerank) if max_rerank > min_rerank else 0.5 for s in raw_rerank_list]

            for i, res in enumerate(top_candidates):
                key = (res.collection, res.path, res.seq_id)
                n_rrf = norm_rrf[i]
                n_rerank = norm_rerank[i]

                if reranker_only:
                    final_score = n_rerank
                else:
                    final_score = (0.50 * n_rrf) + (0.50 * n_rerank)
                    
                    # Dual-Match Immunity / Floor:
                    # If candidate was retrieved by BOTH FTS and Vector search, preserve its top status
                    if key in sources_found and "fts" in sources_found[key] and "vec" in sources_found[key]:
                        dual_floor = 0.80 * n_rrf
                        final_score = max(final_score, dual_floor) + 0.15

                res.score = final_score
                res.source = "hybrid"
                final_results.append(res)
            t_rerank = (time.perf_counter() - t_rerank_start) * 1000

        t_dedup_start = time.perf_counter()
        # Sort by the final calculated score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Deduplicate results based on content, keeping the highest scored version
        unique_results = []
        seen_content = set()
        
        for res in final_results:
            # Create a signature for the content to detect exact duplicates
            content_sig = res.text.strip()
            if content_sig not in seen_content:
                unique_results.append(res)
                seen_content.add(content_sig)

        # Assign rank based on the final, unique order
        for i, res in enumerate(unique_results):
            res.rank = i + 1

        ret_results = unique_results[:limit]
        if should_cache and not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
        t_dedup = (time.perf_counter() - t_dedup_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
            print(f"{DIM}Query:{RESET} {query}")
            print(f"{YELLOW}Lexical (FTS):{RESET} {lex_queries}")
            print(f"{MAGENTA}Vector (Semantic):{RESET} {vec_queries}")
            print(f"{DIM}Candidates: FTS={len(fts_results)} | Vector={len(vec_results)} | RRF={len(top_candidates)}{RESET}")
            print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
            print(f"  • Query Expansion:         {t_expand:>7.2f} ms")
            print(f"  • Lexical (FTS) Search:    {t_fts:>7.2f} ms ({len(fts_results)} hits)")
            print(f"  • Vector (Semantic) Search:{t_vec:>7.2f} ms ({len(vec_results)} hits)")
            print(f"  • RRF Fusion:              {t_rrf:>7.2f} ms ({len(top_candidates)} candidates)")
            if rerank or reranker_only:
                print(f"  • Cross-Encoder Rerank:    {t_rerank:>7.2f} ms ({len(top_candidates)} scored)")
            print(f"  • Dedup & Slicing:         {t_dedup:>7.2f} ms ({len(ret_results)} returned)")
            print(f"  • Total Search:            {t_total:>7.2f} ms")
            print(f"{CYAN}------------------------{RESET}\n")

        return ret_results

    def wide_to_narrow_search(
        self, 
        query: str, 
        limit: int = 10, 
        verbose: bool = False, 
        rerank: bool = False, 
        reranker_only: bool = False, 
        collection: Optional[str] = None, 
        lexical_query: Optional[str] = None, 
        title: Optional[str] = None, 
        path: Optional[Union[str, List[str]]] = None,
        top_containers: int = 3,
        fts_limit: Optional[int] = None,
        vec_limit: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
        exclude_seen_set: Optional[set] = None,
        use_cache: Optional[bool] = None
    ) -> List[Result]:
        """
        Hierarchical Wide-to-Narrow (W2N) Search:
        1. Performs a wide hybrid retrieval to gather candidate chunks.
        2. Scores documents and parent directories based on chunk density and relevance.
        3. Boosts chunks residing within the top matching documents and directories.
        """
        t_total_start = time.perf_counter()
        should_cache = use_cache if use_cache is not None else getattr(self.config, 'cache_search_results', True)
        if should_cache and not exclude_seen_set:
            t_cache_start = time.perf_counter()
            cache_key = self._build_search_cache_key(
                "w2n", query, limit, rerank, reranker_only, collection, lexical_query, title, path, fts_limit, vec_limit, rerank_candidates
            )
            cached_json = get_cached_search_results(self.history_conn, cache_key)
            t_cache = (time.perf_counter() - t_cache_start) * 1000
            if cached_json:
                self.last_exclusion_stats = {"excluded_chunks": 0, "excluded_docs": 0}
                if verbose:
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Wide-to-Narrow Diagnostics ---{RESET}")
                    print(f"{DIM}Original Query:{RESET} {query}")
                    print(f"{GREEN}[Cache Hit]{RESET} Loaded results from search cache in {t_cache:.2f}ms")
                    print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                    print(f"  • Cache Lookup:            {t_cache:>7.2f} ms")
                    print(f"  • Total Wide-to-Narrow:    {t_total:>7.2f} ms")
                    print(f"{CYAN}------------------------{RESET}\n")
                return _json_to_results(cached_json)

        # Step 1: Wide Phase - Fetch a broader pool of candidates based on configured limits
        cfg_max = max(getattr(self.config, 'fts_limit', 50), getattr(self.config, 'vec_limit', 50))
        wide_limit = max(cfg_max, limit * 5)
        t_wide_start = time.perf_counter()
        wide_results = self.hybrid_search(
            query,
            limit=wide_limit,
            verbose=False,  # Sub-call timing suppressed in favor of unified W2N timing summary
            rerank=False,   # Defer reranking until after container filtering
            reranker_only=False,
            collection=collection,
            lexical_query=lexical_query,
            title=title,
            path=path,
            fts_limit=fts_limit,
            vec_limit=vec_limit,
            rerank_candidates=rerank_candidates,
            exclude_seen_set=exclude_seen_set,
            use_cache=False
        )
        t_wide = (time.perf_counter() - t_wide_start) * 1000

        if not wide_results:
            return []

        # Step 2: Aggregate relevance at Document and Directory levels
        t_agg_start = time.perf_counter()
        doc_scores: Dict[str, float] = {}
        dir_scores: Dict[str, float] = {}

        for res in wide_results:
            # Document-level aggregation
            doc_scores[res.path] = doc_scores.get(res.path, 0.0) + res.score
            
            # Directory-level aggregation (handling subfolders)
            parent_dir = str(Path(res.path).parent)
            if parent_dir and parent_dir != ".":
                dir_scores[parent_dir] = dir_scores.get(parent_dir, 0.0) + (res.score * 0.75)

        top_docs = set(sorted(doc_scores, key=doc_scores.get, reverse=True)[:top_containers])
        top_dirs = set(sorted(dir_scores, key=dir_scores.get, reverse=True)[:top_containers])
        t_agg = (time.perf_counter() - t_agg_start) * 1000

        # Step 3: Narrow Phase - Apply container density boosts
        t_boost_start = time.perf_counter()
        boosted_results = []
        for res in wide_results:
            parent_dir = str(Path(res.path).parent)
            is_top_doc = res.path in top_docs
            is_top_dir = parent_dir in top_dirs

            if is_top_doc or is_top_dir:
                # Apply boost based on container hierarchy match
                multiplier = 1.35 if is_top_doc else 1.15
                res.score *= multiplier
                boosted_results.append(res)

        # Fall back to wide results if container filtering was overly restrictive
        if len(boosted_results) < limit:
            seen_signatures = {r.path + r.text[:50] for r in boosted_results}
            for res in wide_results:
                sig = res.path + res.text[:50]
                if sig not in seen_signatures:
                    boosted_results.append(res)
                    seen_signatures.add(sig)

        boosted_results.sort(key=lambda x: x.score, reverse=True)
        candidate_pool = boosted_results[:limit * 2]
        t_boost = (time.perf_counter() - t_boost_start) * 1000

        # Step 4: Optional LLM Reranking on the top W2N candidates
        t_rerank = 0.0
        if rerank or reranker_only:
            t_rerank_start = time.perf_counter()
            rerank_docs = [f"File: {c.path}\nContent: {c.text}" for c in candidate_pool]
            rerank_results = self.llm.rerank(query, rerank_docs)
            
            raw_pool_scores = [c.score for c in candidate_pool]
            raw_rerank_list = []
            for i, res in enumerate(candidate_pool):
                r_score = 0.0
                for r in rerank_results:
                    if r.get('index') == i:
                        r_score = float(r.get('score', 0.0))
                        break
                raw_rerank_list.append(r_score)

            min_pool, max_pool = min(raw_pool_scores), max(raw_pool_scores)
            norm_pool = [(s - min_pool) / (max_pool - min_pool) if max_pool > min_pool else 1.0 for s in raw_pool_scores]

            min_rerank, max_rerank = min(raw_rerank_list), max(raw_rerank_list)
            norm_rerank = [(s - min_rerank) / (max_rerank - min_rerank) if max_rerank > min_rerank else 0.5 for s in raw_rerank_list]

            for i, res in enumerate(candidate_pool):
                if reranker_only:
                    res.score = norm_rerank[i]
                else:
                    res.score = (0.50 * norm_pool[i]) + (0.50 * norm_rerank[i])

            candidate_pool.sort(key=lambda x: x.score, reverse=True)
            t_rerank = (time.perf_counter() - t_rerank_start) * 1000

        t_dedup_start = time.perf_counter()
        for i, res in enumerate(candidate_pool):
            res.rank = i + 1

        ret_results = candidate_pool[:limit]
        if should_cache and not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
        t_dedup = (time.perf_counter() - t_dedup_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            print(f"\n{CYAN}--- Search Diagnostics (Wide-to-Narrow) ---{RESET}")
            print(f"{DIM}Query:{RESET} {query}")
            print(f"{YELLOW}Top Documents:{RESET} {list(top_docs)}")
            print(f"{MAGENTA}Top Directories:{RESET} {list(top_dirs)}")
            print(f"{DIM}Candidates: Wide={len(wide_results)} | Boosted={len(boosted_results)} | Returned={len(ret_results)}{RESET}")
            print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
            print(f"  • Wide Retrieval:          {t_wide:>7.2f} ms ({len(wide_results)} chunks)")
            print(f"  • Container Aggregation:   {t_agg:>7.2f} ms ({len(top_docs)} docs, {len(top_dirs)} dirs)")
            print(f"  • Narrow Density Boosting: {t_boost:>7.2f} ms ({len(boosted_results)} boosted)")
            if rerank or reranker_only:
                print(f"  • Cross-Encoder Rerank:    {t_rerank:>7.2f} ms ({len(candidate_pool)} scored)")
            print(f"  • Dedup & Slicing:         {t_dedup:>7.2f} ms ({len(ret_results)} returned)")
            print(f"  • Total Search:            {t_total:>7.2f} ms")
            print(f"{CYAN}---------------------------------------{RESET}\n")

        return ret_results

    def discover(
        self,
        query: str,
        limit: int = 10,
        verbose: bool = False,
        rerank: bool = False,
        reranker_only: bool = False,
        collection: Optional[str] = None,
        lexical_query: Optional[str] = None,
        title: Optional[str] = None,
        path: Optional[Union[str, List[str]]] = None,
        fts_limit: Optional[int] = None,
        vec_limit: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
        exclude_seen_set: Optional[set] = None,
        w2n: bool = False,
        use_cache: Optional[bool] = None
    ) -> List[Result]:
        """
        Discover Search: Retrieves top matching documents, returning ONLY the single
        highest-scoring chunk for each document, along with total match count per document.
        """
        t_total_start = time.perf_counter()
        should_cache = use_cache if use_cache is not None else getattr(self.config, 'cache_search_results', True)
        if should_cache and not exclude_seen_set:
            t_cache_start = time.perf_counter()
            cache_key = self._build_search_cache_key(
                "discover", query, limit, rerank, reranker_only, collection, lexical_query, title, path, fts_limit, vec_limit, rerank_candidates
            )
            cached_json = get_cached_search_results(self.history_conn, cache_key)
            t_cache = (time.perf_counter() - t_cache_start) * 1000
            if cached_json:
                self.last_exclusion_stats = {"excluded_chunks": 0, "excluded_docs": 0}
                if verbose:
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Discover Diagnostics ---{RESET}")
                    print(f"{DIM}Original Query:{RESET} {query}")
                    print(f"{GREEN}[Cache Hit]{RESET} Loaded results from search cache in {t_cache:.2f}ms")
                    print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                    print(f"  • Cache Lookup:            {t_cache:>7.2f} ms")
                    print(f"  • Total Discover Search:   {t_total:>7.2f} ms")
                    print(f"{CYAN}------------------------{RESET}\n")
                return _json_to_results(cached_json)

        fetch_limit = max(limit * 6, getattr(self.config, 'fts_limit', 50), getattr(self.config, 'vec_limit', 50))

        t_cand_start = time.perf_counter()
        if w2n:
            candidates = self.wide_to_narrow_search(
                query=query,
                limit=fetch_limit,
                verbose=False,  # Sub-call timing suppressed in favor of unified discover timing summary
                rerank=rerank,
                reranker_only=reranker_only,
                collection=collection,
                lexical_query=lexical_query,
                title=title,
                path=path,
                fts_limit=fts_limit,
                vec_limit=vec_limit,
                rerank_candidates=rerank_candidates,
                exclude_seen_set=exclude_seen_set,
                use_cache=False
            )
        else:
            candidates = self.hybrid_search(
                query=query,
                limit=fetch_limit,
                verbose=False,  # Sub-call timing suppressed in favor of unified discover timing summary
                rerank=rerank,
                reranker_only=reranker_only,
                collection=collection,
                lexical_query=lexical_query,
                title=title,
                path=path,
                fts_limit=fts_limit,
                vec_limit=vec_limit,
                rerank_candidates=rerank_candidates,
                exclude_seen_set=exclude_seen_set,
                use_cache=False
            )
        t_cand = (time.perf_counter() - t_cand_start) * 1000

        if not candidates:
            return []

        t_group_start = time.perf_counter()
        doc_map: Dict[Tuple[str, str], List[Result]] = {}
        for c in candidates:
            key = (c.collection, c.path)
            if key not in doc_map:
                doc_map[key] = []
            doc_map[key].append(c)

        discover_results: List[Result] = []
        for (coll_name, doc_path), doc_chunks in doc_map.items():
            best_chunk = max(doc_chunks, key=lambda x: x.score)

            doc_res = Result(
                path=best_chunk.path,
                title=best_chunk.title,
                text=best_chunk.text,
                score=best_chunk.score,
                source=best_chunk.source,
                collection=best_chunk.collection,
                seq_id=best_chunk.seq_id,
                headers=getattr(best_chunk, "headers", ""),
                fts_score=getattr(best_chunk, "fts_score", None),
                fts_rank=getattr(best_chunk, "fts_rank", None),
                vec_score=getattr(best_chunk, "vec_score", None),
                vec_rank=getattr(best_chunk, "vec_rank", None),
                rrf_score=getattr(best_chunk, "rrf_score", None),
                rrf_rank=getattr(best_chunk, "rrf_rank", None),
                match_count=len(doc_chunks)
            )
            discover_results.append(doc_res)

        discover_results.sort(key=lambda x: x.score, reverse=True)

        for i, res in enumerate(discover_results):
            res.rank = i + 1

        final_results = discover_results[:limit]

        if should_cache and not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(final_results))
        t_group = (time.perf_counter() - t_group_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            print(f"\n{CYAN}--- Search Diagnostics (Discover) ---{RESET}")
            print(f"{DIM}Query:{RESET} {query}")
            print(f"{DIM}Mode:{RESET} {'Wide-to-Narrow' if w2n else 'Hybrid'} | {DIM}Rerank:{RESET} {rerank or reranker_only}")
            print(f"{DIM}Candidates: {len(candidates)} chunks -> {len(final_results)} top documents{RESET}")
            print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
            print(f"  • Candidate Retrieval:     {t_cand:>7.2f} ms ({len(candidates)} chunks)")
            print(f"  • Top-Hit Doc Grouping:    {t_group:>7.2f} ms ({len(final_results)} documents)")
            print(f"  • Total Search:            {t_total:>7.2f} ms")
            print(f"{CYAN}---------------------------------{RESET}\n")

        return final_results

    def get_document_outline(self, collection: Optional[str], path: str) -> Optional[Dict[str, Any]]:
        """Extracts document structure heading hierarchy correlated with chunk sequence ranges."""
        cursor = self.conn.cursor()
        if collection:
            cursor.execute("SELECT id, collection, path, title, hash FROM documents WHERE collection = ? AND path = ?", (collection, path))
        else:
            cursor.execute("SELECT id, collection, path, title, hash FROM documents WHERE path = ?", (path,))

        doc_row = cursor.fetchone()
        if not doc_row:
            # Fallback to substring matching on path
            if collection:
                cursor.execute("SELECT id, collection, path, title, hash FROM documents WHERE collection = ? AND path LIKE ?", (collection, f"%{path}%"))
            else:
                cursor.execute("SELECT id, collection, path, title, hash FROM documents WHERE path LIKE ?", (f"%{path}%",))
            doc_row = cursor.fetchone()

        if not doc_row:
            return None

        doc_id, coll_name, doc_path, title, doc_hash = doc_row

        cursor.execute("SELECT body FROM content WHERE hash = ?", (doc_hash,))
        content_row = cursor.fetchone()
        raw_md = decompress_text(content_row[0]) if content_row else ""

        cursor.execute("SELECT rowid, seq_id, chunk_text, COALESCE(headers, '') FROM chunk_metadata WHERE doc_hash = ? ORDER BY seq_id", (doc_hash,))
        chunk_rows = cursor.fetchall()

        chunks_data = []
        for r in chunk_rows:
            rowid, seq_id, compressed_text, headers_str = r
            text = decompress_text(compressed_text)
            chunks_data.append({
                "rowid": rowid,
                "seq_id": seq_id,
                "text": text,
                "headers": headers_str
            })

        total_chunks = len(chunks_data)
        total_chars = sum(len(c["text"]) for c in chunks_data) if chunks_data else len(raw_md)

        raw_headings = extract_outline(content=raw_md)

        # Build monotonic chunk character offsets within raw_md
        chunk_offsets = []
        curr_pos = 0
        for c in chunks_data:
            prefix = c["text"].strip()[:50]
            if prefix:
                idx = raw_md.find(prefix, curr_pos)
                if idx != -1:
                    curr_pos = idx
                    chunk_offsets.append(idx)
                else:
                    chunk_offsets.append(curr_pos)
            else:
                chunk_offsets.append(curr_pos)

        def _get_seq_for_offset(offset: int) -> int:
            if not chunk_offsets:
                return 0
            seq = 0
            for seq_i, c_off in enumerate(chunk_offsets):
                if c_off <= offset:
                    seq = seq_i
                else:
                    break
            return seq

        # Calculate start_seq for each heading based on character offset
        start_seqs = [_get_seq_for_offset(h["char_offset"]) for h in raw_headings]

        structured_headings = []
        num_headings = len(raw_headings)
        last_chunk_idx = (len(chunks_data) - 1) if chunks_data else 0

        for i, h in enumerate(raw_headings):
            h_text = h["text"]
            h_level = h["level"]
            start_seq = start_seqs[i]

            # Find next heading at same or higher (<= h_level) hierarchy level
            next_same_or_higher_seq = None
            for j in range(i + 1, num_headings):
                if raw_headings[j]["level"] <= h_level:
                    next_same_or_higher_seq = start_seqs[j]
                    break

            if next_same_or_higher_seq is not None:
                if next_same_or_higher_seq > start_seq:
                    end_seq = next_same_or_higher_seq - 1
                else:
                    end_seq = start_seq
            else:
                end_seq = last_chunk_idx

            if end_seq < start_seq:
                end_seq = start_seq

            section_chars = sum(
                len(c["text"]) for c in chunks_data if start_seq <= c["seq_id"] <= end_seq
            ) if chunks_data else 0

            structured_headings.append({
                "level": h_level,
                "text": h_text,
                "start_seq": start_seq,
                "end_seq": end_seq,
                "char_count": section_chars
            })

        return {
            "collection": coll_name,
            "path": doc_path,
            "title": title,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "headings": structured_headings
        }

    def _get_chunks_in_window(self, doc_hash: str, target_seq_id: int, window: int) -> List[Result]:
        min_seq = max(0, target_seq_id - window)
        max_seq = target_seq_id + window

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.rowid, m.seq_id, m.chunk_text, COALESCE(m.headers, ''), d.path, d.title, d.collection
            FROM chunk_metadata m
            JOIN documents d ON m.doc_hash = d.hash
            WHERE m.doc_hash = ? AND m.seq_id >= ? AND m.seq_id <= ?
            ORDER BY m.seq_id ASC
        """, (doc_hash, min_seq, max_seq))

        results = []
        for row in cursor.fetchall():
            rowid, seq_id, compressed_text, headers, doc_path, title, collection = row
            chunk_text = decompress_text(compressed_text)
            results.append(Result(
                path=doc_path,
                title=title,
                text=chunk_text,
                score=1.0,
                source="chunk",
                collection=collection or "",
                seq_id=seq_id,
                headers=headers
            ))

        return results

    def get_chunk_by_id(self, rowid: Union[int, List[int]], window: int = 0) -> List[Result]:
        """Retrieves target chunk(s) by vector rowid(s) along with ±window surrounding chunks."""
        rowids = [rowid] if isinstance(rowid, int) else list(rowid)
        if not rowids:
            return []

        cursor = self.conn.cursor()
        if window == 0:
            placeholders = ','.join(['?'] * len(rowids))
            cursor.execute(f"""
                SELECT m.rowid, m.seq_id, m.chunk_text, COALESCE(m.headers, ''), d.path, d.title, d.collection
                FROM chunk_metadata m
                JOIN documents d ON m.doc_hash = d.hash
                WHERE m.rowid IN ({placeholders})
                ORDER BY d.collection, d.path, m.seq_id ASC
            """, tuple(rowids))
            results = []
            for row in cursor.fetchall():
                r_id, seq_id, compressed_text, headers, doc_path, title, collection = row
                results.append(Result(
                    path=doc_path,
                    title=title,
                    text=decompress_text(compressed_text),
                    score=1.0,
                    source="chunk",
                    collection=collection or "",
                    seq_id=seq_id,
                    headers=headers
                ))
            return results

        placeholders = ','.join(['?'] * len(rowids))
        cursor.execute(f"""
            SELECT m.doc_hash, m.seq_id
            FROM chunk_metadata m
            WHERE m.rowid IN ({placeholders})
        """, tuple(rowids))
        rows = cursor.fetchall()
        if not rows:
            return []

        doc_seq_map: Dict[str, set] = {}
        for doc_hash, seq_id in rows:
            if doc_hash not in doc_seq_map:
                doc_seq_map[doc_hash] = set()
            for s in range(max(0, seq_id - window), seq_id + window + 1):
                doc_seq_map[doc_hash].add(s)

        results = []
        for doc_hash, seq_set in doc_seq_map.items():
            seq_list = sorted(seq_set)
            seq_placeholders = ','.join(['?'] * len(seq_list))
            cursor.execute(f"""
                SELECT m.rowid, m.seq_id, m.chunk_text, COALESCE(m.headers, ''), d.path, d.title, d.collection
                FROM chunk_metadata m
                JOIN documents d ON m.doc_hash = d.hash
                WHERE m.doc_hash = ? AND m.seq_id IN ({seq_placeholders})
                ORDER BY m.seq_id ASC
            """, (doc_hash, *seq_list))
            for row in cursor.fetchall():
                r_id, seq_id, compressed_text, headers, doc_path, title, collection = row
                results.append(Result(
                    path=doc_path,
                    title=title,
                    text=decompress_text(compressed_text),
                    score=1.0,
                    source="chunk",
                    collection=collection or "",
                    seq_id=seq_id,
                    headers=headers
                ))
        return results

    def get_chunk_by_seq(self, collection: Optional[str], path: str, seq_id: Union[int, List[int]] = 0, window: int = 0) -> List[Result]:
        """Retrieves target chunk(s) by collection, path, and seq_id(s) along with ±window surrounding chunks."""
        cursor = self.conn.cursor()
        if collection:
            cursor.execute("SELECT hash, collection, path, title FROM documents WHERE collection = ? AND path = ?", (collection, path))
        else:
            cursor.execute("SELECT hash, collection, path, title FROM documents WHERE path = ?", (path,))

        row = cursor.fetchone()
        if not row:
            if collection:
                cursor.execute("SELECT hash, collection, path, title FROM documents WHERE collection = ? AND path LIKE ?", (collection, f"%{path}%"))
            else:
                cursor.execute("SELECT hash, collection, path, title FROM documents WHERE path LIKE ?", (f"%{path}%",))
            row = cursor.fetchone()

        if not row:
            return []

        doc_hash, coll_name, doc_path, title = row
        seq_ids = [seq_id] if isinstance(seq_id, int) else list(seq_id)
        if not seq_ids:
            return []

        target_seqs = set()
        for sid in seq_ids:
            for s in range(max(0, sid - window), sid + window + 1):
                target_seqs.add(s)

        sorted_seqs = sorted(target_seqs)
        placeholders = ','.join(['?'] * len(sorted_seqs))
        cursor.execute(f"""
            SELECT m.rowid, m.seq_id, m.chunk_text, COALESCE(m.headers, ''), d.path, d.title, d.collection
            FROM chunk_metadata m
            JOIN documents d ON m.doc_hash = d.hash
            WHERE m.doc_hash = ? AND m.seq_id IN ({placeholders})
            ORDER BY m.seq_id ASC
        """, (doc_hash, *sorted_seqs))

        results = []
        for r in cursor.fetchall():
            rowid, s_id, compressed_text, headers, d_path, d_title, collection_name = r
            results.append(Result(
                path=d_path,
                title=d_title,
                text=decompress_text(compressed_text),
                score=1.0,
                source="chunk",
                collection=collection_name or "",
                seq_id=s_id,
                headers=headers
            ))
        return results

    def get_collection_tree(
        self,
        collection: Optional[str] = None,
        max_depth: Optional[int] = None,
        pattern: Optional[str] = None,
        is_regex: bool = False,
        case_sensitive: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Builds a nested folder directory tree structure of indexed documents, optionally filtered by pattern/regex."""
        cursor = self.conn.cursor()

        if collection:
            cursor.execute("SELECT DISTINCT collection FROM documents WHERE collection = ?", (collection,))
            if not cursor.fetchone():
                return None
            collections = [collection]
        else:
            cursor.execute("SELECT DISTINCT collection FROM documents ORDER BY collection ASC")
            collections = [row[0] for row in cursor.fetchall()]

        if not collections and collection:
            return None

        matcher = None
        if pattern:
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                try:
                    matcher = re.compile(pattern, flags)
                except re.error as e:
                    raise ValueError(f"Invalid regular expression: {e}")
            else:
                matcher = re.compile(re.escape(pattern), flags)

        results = []

        for coll_name in collections:
            cursor.execute(
                "SELECT id, path, title FROM documents WHERE collection = ? ORDER BY path ASC",
                (coll_name,)
            )
            doc_rows = cursor.fetchall()

            dir_tree: Dict[str, Any] = {"dirs": {}, "files": []}

            for doc_id, doc_path, doc_title in doc_rows:
                if matcher:
                    if not (matcher.search(doc_path) or (doc_title and matcher.search(doc_title))):
                        continue

                norm_path = doc_path.replace("\\", "/").strip("/")
                parts = norm_path.split("/") if norm_path else []
                if not parts:
                    continue

                dir_parts = parts[:-1]
                file_name = parts[-1]

                curr = dir_tree
                for d in dir_parts:
                    if d not in curr["dirs"]:
                        curr["dirs"][d] = {"dirs": {}, "files": []}
                    curr = curr["dirs"][d]

                curr["files"].append({
                    "name": file_name,
                    "type": "file",
                    "title": doc_title or file_name,
                    "path": doc_path,
                    "doc_id": doc_id
                })

            def _convert_node(name: str, node_data: Dict[str, Any], current_depth: int) -> Dict[str, Any]:
                children = []
                if max_depth is None or current_depth < max_depth:
                    for dir_name in sorted(node_data["dirs"].keys(), key=str.lower):
                        sub_dict = node_data["dirs"][dir_name]
                        sub_node = _convert_node(dir_name, sub_dict, current_depth + 1)
                        children.append(sub_node)

                    sorted_files = sorted(node_data["files"], key=lambda f: f["name"].lower())
                    children.extend(sorted_files)

                return {
                    "name": name,
                    "type": "directory",
                    "children": children
                }

            root_node = _convert_node(coll_name, dir_tree, current_depth=0)
            results.append({
                "collection": coll_name,
                "tree": root_node
            })

        if collection:
            return results[0] if results else None
        return results

    def grep_search(
        self,
        pattern: str,
        is_regex: bool = False,
        case_sensitive: bool = False,
        collection: Optional[str] = None,
        path: Optional[Union[str, List[str]]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Performs direct string or regular expression pattern search across decompressed document bodies."""
        if not pattern:
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"Invalid regular expression: {e}")
        else:
            regex = re.compile(re.escape(pattern), flags)

        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]

        cursor = self.conn.cursor()
        query_sql = """
            SELECT d.collection, d.path, d.title, c.body
            FROM documents d
            JOIN content c ON d.hash = c.hash
        """
        where_clauses = []
        params = []
        if collection:
            where_clauses.append("d.collection = ?")
            params.append(collection)
        if paths:
            where_clauses.append("(" + " OR ".join(["d.path LIKE ?" for _ in paths]) + ")")
            for p_val in paths:
                params.append(f"%{p_val}%")

        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)
        query_sql += " ORDER BY d.collection, d.path ASC"

        cursor.execute(query_sql, tuple(params))
        rows = cursor.fetchall()

        matches = []
        for coll, doc_path, doc_title, body_blob in rows:
            text = decompress_text(body_blob) if body_blob is not None else ""
            lines = text.splitlines()
            for line_idx, line in enumerate(lines, start=1):
                match = regex.search(line)
                if match:
                    matches.append({
                        "collection": coll or "",
                        "path": doc_path,
                        "title": doc_title or doc_path,
                        "line_number": line_idx,
                        "line_text": line,
                        "match_text": match.group(0)
                    })
                    if len(matches) >= limit:
                        return matches

        return matches