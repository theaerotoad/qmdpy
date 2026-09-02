import sys
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from tqdm import tqdm

from qmd.config import CollectionConfig
from qmd.formatting import YELLOW, RESET
from qmd.converters import convert_to_markdown, SUPPORTED_EXTENSIONS
from qmd.utils import compute_hash, compress_text, decompress_text
from qmd.docparse.parser import parse_markdown_to_blocks
from qmd.docparse.grouper import group_blocks_into_chunks
from qmd.db import (
    update_db_last_updated, record_indexing_error, clear_indexing_errors,
    get_indexing_errors as db_get_indexing_errors, get_db_meta, ensure_vector_table
)
from .models import extract_document_date, encode_vector


class IndexingMixin:
    """Handles collection scanning, file conversion, embedding generation, indexing, and orphan pruning."""

    def _get_collection_files(self, base_path: Path, collection_cfg: CollectionConfig) -> List[Path]:
        if collection_cfg.file_extensions is not None:
            exts = [
                e.lower() if e.startswith('.') else f".{e.lower()}"
                for e in collection_cfg.file_extensions
            ]
            files = []
            for ext in exts:
                for f in base_path.glob(f"**/*{ext}"):
                    if f.is_file() and f.suffix.lower() in exts:
                        files.append(f)
            seen = set()
            unique_files = []
            for f in files:
                if f not in seen:
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
        if self.read_only or getattr(self.config, "is_federated", False):
            raise RuntimeError("Cannot index collection in read-only or federated include mode.")
        base_path = Path(collection_cfg.path).expanduser().resolve()
        if not base_path.exists():
            print(f"Skipping {name}: Path not found {base_path}")
            return

        print(f"Indexing collection: {name} (Force={force})...")
        files = self._get_collection_files(base_path, collection_cfg)
        count_processed = 0
        count_skipped = 0
        found_rel_paths = {str(f.relative_to(base_path)) for f in files}

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
                tqdm.write(f"Error processing {file_path}: {e}")

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
                clear_indexing_errors(self.conn, name, sp)
            self.conn.commit()
            self._cleanup_orphaned_data()

        # Clean up stale indexing errors for files no longer in collection scope
        cursor.execute("SELECT DISTINCT path FROM indexing_errors WHERE collection = ?", (name,))
        recorded_error_paths = {row[0] for row in cursor.fetchall()}
        stale_error_paths = recorded_error_paths - found_rel_paths
        if stale_error_paths:
            for sep in stale_error_paths:
                clear_indexing_errors(self.conn, name, sep)
            self.conn.commit()

        if count_processed > 0 or stale_paths or stale_error_paths:
            update_db_last_updated(self.conn)
            if getattr(self, 'usearch_index', None) is not None and not self.read_only:
                self.usearch_index.save(self.usearch_path)

        print(f"Done. Processed: {count_processed}, Skipped/Unchanged: {count_skipped}")

    def prune_orphaned_collections(self, active_collections: List[str]):
        """Removes documents, FTS entries, and orphaned vectors/content for collections no longer in config."""
        if self.read_only or getattr(self.config, "is_federated", False):
            raise RuntimeError("Cannot prune collections in read-only or federated include mode.")
        cursor = self.conn.cursor()

        if not active_collections:
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
        if getattr(self, 'usearch_index', None) is not None and not self.read_only:
            self.usearch_index.save(self.usearch_path)
        print(f"Pruned {len(orphans)} collection(s).")

    def _cleanup_orphaned_data(self):
        """Removes content, chunk metadata, vectors, and errors no longer referenced by any active document."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT rowid FROM chunk_metadata WHERE doc_hash NOT IN (SELECT DISTINCT hash FROM documents)")
        orphan_rowids = [row[0] for row in cursor.fetchall()]
        if orphan_rowids and getattr(self, 'usearch_index', None) is not None and not self.read_only:
            for rid in orphan_rowids:
                try:
                    self.usearch_index.remove(rid)
                except Exception:
                    pass

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

    def get_indexing_errors(self, collection: Optional[Union[str, List[str]]] = None, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves list of documents that encountered errors or partial failures during indexing."""
        target_stores = self._get_target_stores_for_collection(collection)
        all_errors = []
        for store in target_stores:
            if store is self:
                local_errors = db_get_indexing_errors(self.conn, collection=collection, path=path)
                filtered_errors = []
                for err in local_errors:
                    coll_name = err.get("collection")
                    coll_cfg = getattr(self.config, "collections", {}).get(coll_name)
                    if coll_cfg:
                        err_path = err.get("path", "")
                        ext = Path(err_path).suffix.lower()
                        if coll_cfg.file_extensions is not None:
                            allowed = {
                                e.lower() if e.startswith('.') else f".{e.lower()}"
                                for e in coll_cfg.file_extensions
                            }
                            if ext not in allowed:
                                continue
                        elif not coll_cfg.convert_non_md and ext not in {".md", ".markdown", ".txt"}:
                            continue
                    filtered_errors.append(err)
                all_errors.extend(filtered_errors)
            else:
                all_errors.extend(store.get_indexing_errors(collection=collection, path=path))
        return all_errors

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
            conv_fn = getattr(sys.modules.get("qmd.store"), "convert_to_markdown", convert_to_markdown)
            date_fn = getattr(sys.modules.get("qmd.store"), "extract_document_date", extract_document_date)

            if not force and not has_prior_error:
                cursor.execute(
                    "SELECT id, path FROM documents WHERE collection = ? AND hash = ?",
                    (collection_name, file_hash)
                )
                existing_docs = cursor.fetchall()
                for doc_id, old_path in existing_docs:
                    if old_path not in current_paths:
                        cursor.execute("SELECT body FROM content WHERE hash = ?", (file_hash,))
                        c_row = cursor.fetchone()
                        markdown_body = decompress_text(c_row[0]) if c_row else ""
                        doc_date = date_fn(file_path, markdown_body)

                        cursor.execute("""
                            UPDATE documents SET path = ?, title = ?, modified_at = ?, doc_date = ?
                            WHERE id = ?
                        """, (rel_path, title, now, doc_date, doc_id))

                        cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))

                        cursor.execute(
                            "INSERT INTO documents_fts (rowid, collection, filepath, title, body) VALUES (?, ?, ?, ?, ?)",
                            (doc_id, collection_name, rel_path, title, markdown_body)
                        )

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
                markdown_body = conv_fn(file_path, config=self.config, errors_out=conversion_errors)
                cursor.execute("""
                    INSERT INTO content (hash, body, created_at) VALUES (?, ?, ?)
                    ON CONFLICT(hash) DO UPDATE SET body = excluded.body, created_at = excluded.created_at
                """, (file_hash, compress_text(markdown_body), now))

            doc_date = date_fn(file_path, markdown_body)

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
        blocks, _ = parse_markdown_to_blocks(content=markdown_body, strip_links=self.config.strip_links)
        if not blocks:
            return

        chunks = group_blocks_into_chunks(
            blocks,
            max_chunk_size=self.config.max_chunk_size,
            target_chunk_size=self.config.target_chunk_size
        )

        if not chunks:
            return

        embedding_texts = []
        final_chunk_texts = []
        chunk_headers = []

        for chunk in chunks:
            clean_parents = [re.sub(r'^\s*#+\s*', '', h).strip() for h in chunk.parent_headers.values() if h and h.strip()]
            context_str = " > ".join(clean_parents)

            if context_str:
                text_to_embed = f"Context: {context_str}\n\n{chunk.content}"
            else:
                text_to_embed = chunk.content

            formatted = self.llm.format_doc_for_embedding(title, text_to_embed)

            embedding_texts.append(formatted)
            final_chunk_texts.append(chunk.content)
            chunk_headers.append(context_str)

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

        cursor.execute("SELECT rowid FROM chunk_metadata WHERE doc_hash = ?", (doc_hash,))
        old_rowids = [r[0] for r in cursor.fetchall()]

        cursor.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunk_metadata WHERE doc_hash = ?)", (doc_hash,))
        cursor.execute("DELETE FROM chunk_metadata WHERE doc_hash = ?", (doc_hash,))
        if old_rowids:
            placeholders = ','.join(['?'] * len(old_rowids))
            cursor.execute(f"DELETE FROM vectors WHERE rowid IN ({placeholders})", tuple(old_rowids))
            if getattr(self, 'usearch_index', None) is not None and not self.read_only:
                for rid in old_rowids:
                    try:
                        self.usearch_index.remove(rid)
                    except Exception:
                        pass

        new_rowids = []
        new_vectors = []
        for i, (chunk_text, embedding, context_str) in enumerate(zip(final_chunk_texts, embeddings, chunk_headers)):
            emb_blob = encode_vector(embedding, quant_type=quant_type)
            cursor.execute("INSERT INTO vectors(embedding) VALUES (?)", (emb_blob,))
            vector_rowid = cursor.lastrowid
            new_rowids.append(vector_rowid)
            new_vectors.append(embedding)

            cursor.execute("""
                INSERT INTO chunk_metadata (rowid, doc_hash, seq_id, chunk_text, headers)
                VALUES (?, ?, ?, ?, ?)
            """, (vector_rowid, doc_hash, i, compress_text(chunk_text), context_str))
            cursor.execute("""
                INSERT INTO chunks_fts (rowid, collection, filepath, title, body, headers)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (vector_rowid, collection_name, rel_path, title, chunk_text, context_str))

        if getattr(self, 'usearch_index', None) is not None and not self.read_only:
            import numpy as np
            valid_rids = []
            valid_vecs = []
            for rid, vec in zip(new_rowids, new_vectors):
                if rid is not None:
                    valid_rids.append(rid)
                    valid_vecs.append(vec)
            if valid_rids:
                keys = np.array(valid_rids, dtype=np.uint64)
                vectors_arr = np.array(valid_vecs, dtype=np.float32)
                self.usearch_index.add(keys, vectors_arr)