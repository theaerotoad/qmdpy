import time
import json
import sqlite3
import struct
import math
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from tqdm import tqdm

from qmd.db import (
    get_connection, init_schema, is_sqlite_vec_active, get_db_meta, ensure_vector_table,
    register_functions, get_history_connection, get_cached_query_embedding, save_query_embedding,
    update_db_last_updated, check_db_compatibility, CURRENT_SCHEMA_VERSION,
    get_cached_search_results, save_cached_search_results
)
from qmd.config import Config, CollectionConfig
from qmd.llm import LLMClient
from qmd.utils import compute_hash, build_spacy_fts_queries, compress_text, decompress_text
from qmd.formatting import DIM, YELLOW, CYAN, RESET, MAGENTA
from qmd.converters import convert_to_markdown, SUPPORTED_EXTENSIONS

# Docparse integration
from qmd.docparse.parser import parse_markdown_to_blocks
from qmd.docparse.grouper import group_blocks_into_chunks
from qmd.docparse.models import Chunk


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


def _results_to_json(results: List[Result]) -> str:
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
            "fts_score": getattr(r, "fts_score", None),
            "fts_rank": getattr(r, "fts_rank", None),
            "vec_score": getattr(r, "vec_score", None),
            "vec_rank": getattr(r, "vec_rank", None),
            "rrf_score": getattr(r, "rrf_score", None),
            "rrf_rank": getattr(r, "rrf_rank", None),
        })
    return json.dumps(data)


def _json_to_results(json_str: str) -> List[Result]:
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
            fts_score=item.get("fts_score"),
            fts_rank=item.get("fts_rank"),
            vec_score=item.get("vec_score"),
            vec_rank=item.get("vec_rank"),
            rrf_score=item.get("rrf_score"),
            rrf_rank=item.get("rrf_rank"),
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
    fts_score: Optional[float] = None
    fts_rank: Optional[int] = None
    vec_score: Optional[float] = None
    vec_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None

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
        path: Optional[str],
        fts_limit: Optional[int],
        vec_limit: Optional[int],
        rerank_candidates: Optional[int]
    ) -> str:
        db_last_updated = get_db_meta(self.conn, "last_updated") or ""
        embed_model = getattr(self.config, "embed_model", "")
        rerank_model = getattr(self.config, "rerank_model", "")

        key_data = {
            "search_type": search_type,
            "query": query,
            "limit": limit,
            "rerank": rerank,
            "reranker_only": reranker_only,
            "collection": collection or "",
            "lexical_query": lexical_query or "",
            "title": title or "",
            "path": path or "",
            "fts_limit": fts_limit,
            "vec_limit": vec_limit,
            "rerank_candidates": rerank_candidates,
            "db_last_updated": db_last_updated,
            "embed_model": embed_model,
            "rerank_model": rerank_model
        }
        return compute_hash(json.dumps(key_data, sort_keys=True))

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
        found_rel_paths = set()

        # Wrap the file iterator with tqdm for a progress bar
        file_pbar = tqdm(files, desc=f"Processing {name}", unit="file")
        for file_path in file_pbar:
            rel_path = str(file_path.relative_to(base_path))
            found_rel_paths.add(rel_path)
            disp_path = rel_path if len(rel_path) <= 30 else "..." + rel_path[-27:]
            file_pbar.set_postfix_str(disp_path)
            try:
                if self._process_file(name, base_path, file_path, force=force):
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
            
        self.conn.commit()
        self._cleanup_orphaned_data()
        update_db_last_updated(self.conn)
        print(f"Pruned {len(orphans)} collection(s).")

    def _cleanup_orphaned_data(self):
        """Removes content, chunk metadata, and vectors no longer referenced by any active document."""
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
        self.conn.commit()

    def _process_file(self, collection_name: str, base_path: Path, file_path: Path, force: bool = False) -> bool:
        rel_path = str(file_path.relative_to(base_path))
        try:
            raw_bytes = file_path.read_bytes()
        except Exception:
            return False

        file_hash = compute_hash(raw_bytes)
        # Fix: Only replace underscores, preserve dashes for dates (e.g., 2019-12-20)
        title = file_path.stem.replace('_', ' ').title()
        
        if not force:
            check_cursor = self.conn.cursor()
            check_cursor.execute(
                "SELECT hash FROM documents WHERE collection = ? AND path = ?", 
                (collection_name, rel_path)
            )
            row = check_cursor.fetchone()
            check_cursor.close()

            if row and row[0] == file_hash:
                return False

        self.conn.execute("BEGIN")
        cursor = self.conn.cursor()
        try:
            now = datetime.now().isoformat()
            cursor.execute("SELECT body FROM content WHERE hash = ?", (file_hash,))
            content_row = cursor.fetchone()

            if content_row:
                content_exists = True
                markdown_body = decompress_text(content_row[0])
            else:
                content_exists = False
                markdown_body = convert_to_markdown(file_path)
                cursor.execute(
                    "INSERT INTO content (hash, body, created_at) VALUES (?, ?, ?)",
                    (file_hash, compress_text(markdown_body), now)
                )

            cursor.execute("SELECT id FROM documents WHERE collection = ? AND path = ?", (collection_name, rel_path))
            existing_doc = cursor.fetchone()

            if existing_doc:
                doc_id = existing_doc[0]
                cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
                cursor.execute("""
                    UPDATE documents SET hash = ?, modified_at = ?, title = ?
                    WHERE id = ?
                """, (file_hash, now, title, doc_id))
            else:
                cursor.execute("""
                    INSERT INTO documents (collection, path, title, hash, modified_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (collection_name, rel_path, title, file_hash, now))
                doc_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)",
                (doc_id, collection_name, rel_path, title, markdown_body)
            )

            if not content_exists:
                self._generate_and_store_embeddings(cursor, file_hash, title, markdown_body, collection_name, rel_path)

            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
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

    def search_fts(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[str] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None) -> List[Result]:
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
        
        base_sql = """
            SELECT f.filepath, f.title, f.body, f.rank, f.collection, f.rowid, m.seq_id, COALESCE(f.headers, '')
            FROM chunks_fts f
            JOIN chunk_metadata m ON f.rowid = m.rowid
            WHERE chunks_fts MATCH ?
        """
        filters_sql = ""
        if collection: filters_sql += " AND f.collection = ?"
        if title: filters_sql += " AND f.title LIKE ?"
        if path: filters_sql += " AND f.filepath LIKE ?"
        
        order_sql = " ORDER BY f.rank LIMIT ?"
        
        # Scale up candidate over-fetch limit so seen chunks don't starve candidate quorum
        sql_limit = max(limit * 5, limit + (len(exclude_seen_set) * 2 if exclude_seen_set else 0)) if exclude_seen_set else limit

        def build_params(q_val):
            p = [q_val]
            if collection: p.append(collection)
            if title: p.append(f"%{title}%")
            if path: p.append(f"%{path}%")
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

    def search_vec(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[str] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None) -> List[Result]:
        limit = limit if limit is not None else getattr(self.config, 'vec_limit', 50)
        query_text = self.llm.format_query_for_embedding(query)
        query_vec = self.get_query_embedding(query_text)
        if not query_vec:
            return []
        
        cursor = self.conn.cursor()
        stored_quant = get_db_meta(self.conn, "vector_quantization")
        quant_type = (stored_quant or getattr(self.config, "vector_quantization", "none") or "none").lower()

        # Check if sqlite-vec is active and virtual table is set up
        if is_sqlite_vec_active(self.conn):
            try:
                query_blob = encode_vector(query_vec, quant_type)
                extra_seen = (len(exclude_seen_set) * 2) if exclude_seen_set else 0
                k_val = (limit * 5 + extra_seen) if (collection or title or path or exclude_seen_set) else limit

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
                if path:
                    query_sql += " AND d.path LIKE ?"
                    params.append(f"%{path}%")

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
        if path:
            where_clauses.append("d.path LIKE ?")
            params.append(f"%{path}%")
            
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
        path: Optional[str] = None,
        fts_limit: Optional[int] = None,
        vec_limit: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
        exclude_seen_set: Optional[set] = None
    ) -> List[Result]:
        excluded_chunks_tracker: set = set()
        
        if not exclude_seen_set:
            cache_key = self._build_search_cache_key(
                "hybrid", query, limit, rerank, reranker_only, collection, lexical_query, title, path, fts_limit, vec_limit, rerank_candidates
            )
            cached_json = get_cached_search_results(self.history_conn, cache_key)
            if cached_json:
                self.last_exclusion_stats = {"excluded_chunks": 0, "excluded_docs": 0}
                return _json_to_results(cached_json)

        fts_lim = fts_limit if fts_limit is not None else getattr(self.config, 'fts_limit', 50)
        vec_lim = vec_limit if vec_limit is not None else getattr(self.config, 'vec_limit', 50)
        rr_cand = rerank_candidates if rerank_candidates is not None else getattr(self.config, 'rerank_candidates', 20)

        # Generate default spaCy FTS expanded lexical queries
        if lexical_query:
            lex_queries = [lexical_query]
        else:
            lex_queries = build_spacy_fts_queries(query, model_name=self.config.spacy_model)

        vec_queries = [query]

        if verbose:
            print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
            print(f"{DIM}Original Query:{RESET} {query}")
            print(f"{YELLOW}Lexical (FTS):{RESET} {lex_queries}")
            print(f"{MAGENTA}Vector (Semantic):{RESET} {vec_queries}")
            print(f"{CYAN}--------------------------{RESET}\n")

        fts_results = []
        seen_fts_keys = set()
        min_fts_quorum = 3

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

        vec_results = []
        for vq in vec_queries:
            vec_results.extend(self.search_vec(vq, limit=vec_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker))

        self.last_exclusion_stats = {
            "excluded_chunks": len(excluded_chunks_tracker),
            "excluded_docs": len({(coll, p) for (coll, p, seq) in excluded_chunks_tracker})
        }

        if verbose:
            print(f"{DIM}Candidates found -> FTS: {len(fts_results)} | Vector: {len(vec_results)}{RESET}")

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

        if not top_candidates:
            if verbose: print(f"{YELLOW}No candidates remained after RRF fusion.{RESET}")
            return []

        final_results = []

        if not rerank and not reranker_only:
            # Skip reranking entirely
            for res in top_candidates:
                res.score = rrf_scores[(res.collection, res.path, res.seq_id)]
                res.source = "hybrid"
                final_results.append(res)
        else:
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
        if not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
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
        path: Optional[str] = None,
        top_containers: int = 3,
        fts_limit: Optional[int] = None,
        vec_limit: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
        exclude_seen_set: Optional[set] = None
    ) -> List[Result]:
        """
        Hierarchical Wide-to-Narrow (W2N) Search:
        1. Performs a wide hybrid retrieval to gather candidate chunks.
        2. Scores documents and parent directories based on chunk density and relevance.
        3. Boosts chunks residing within the top matching documents and directories.
        """
        if not exclude_seen_set:
            cache_key = self._build_search_cache_key(
                "w2n", query, limit, rerank, reranker_only, collection, lexical_query, title, path, fts_limit, vec_limit, rerank_candidates
            )
            cached_json = get_cached_search_results(self.history_conn, cache_key)
            if cached_json:
                self.last_exclusion_stats = {"excluded_chunks": 0, "excluded_docs": 0}
                return _json_to_results(cached_json)

        # Step 1: Wide Phase - Fetch a broader pool of candidates based on configured limits
        cfg_max = max(getattr(self.config, 'fts_limit', 50), getattr(self.config, 'vec_limit', 50))
        wide_limit = max(cfg_max, limit * 5)
        wide_results = self.hybrid_search(
            query,
            limit=wide_limit,
            verbose=verbose,
            rerank=False,  # Defer reranking until after container filtering
            reranker_only=False,
            collection=collection,
            lexical_query=lexical_query,
            title=title,
            path=path,
            fts_limit=fts_limit,
            vec_limit=vec_limit,
            rerank_candidates=rerank_candidates,
            exclude_seen_set=exclude_seen_set
        )

        if not wide_results:
            return []

        # Step 2: Aggregate relevance at Document and Directory levels
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

        if verbose:
            print(f"{CYAN}--- Wide-to-Narrow Container Diagnostics ---{RESET}")
            print(f"{YELLOW}Top Matching Documents:{RESET} {list(top_docs)}")
            print(f"{MAGENTA}Top Matching Directories:{RESET} {list(top_dirs)}")
            print(f"{CYAN}---------------------------------------------{RESET}\n")

        # Step 3: Narrow Phase - Apply container density boosts
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

        # Step 4: Optional LLM Reranking on the top W2N candidates
        if rerank or reranker_only:
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

        for i, res in enumerate(candidate_pool):
            res.rank = i + 1

        ret_results = candidate_pool[:limit]
        if not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
        return ret_results