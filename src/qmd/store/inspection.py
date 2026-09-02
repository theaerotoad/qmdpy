import re
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from qmd.db import get_db_meta
from qmd.utils import decompress_text
from qmd.docparse.parser import extract_outline
from .models import Result, _format_size, _build_collection_sql_filter


class InspectionMixin:
    """Handles stats, document outlines, chunk viewing, collection tree, and grep search."""

    def _get_local_stats(self, collection: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        last_updated = get_db_meta(self.conn, "last_updated") or ""
        cache_key = (last_updated, str(collection) if collection else "")
        if hasattr(self, "_local_stats_cache") and self._local_stats_cache.get("key") == cache_key:
            return self._local_stats_cache["data"]

        cursor = self.conn.cursor()
        coll_sql, coll_params = _build_collection_sql_filter("collection", collection)

        if collection and coll_sql:
            cursor.execute(f"SELECT COUNT(*), COUNT(DISTINCT collection) FROM documents WHERE 1=1 {coll_sql}", tuple(coll_params))
            row = cursor.fetchone()
            docs_count = row[0] or 0
            colls_count = row[1] or 0

            cursor.execute(f"SELECT path, collection FROM documents WHERE 1=1 {coll_sql}", tuple(coll_params))
            doc_rows = cursor.fetchall()
            dirs = {str(Path(p).parent) for p, _ in doc_rows if Path(p).parent != Path(".")}
            colls = {c for _, c in doc_rows if c}

            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM chunk_metadata m
                JOIN documents d ON m.doc_hash = d.hash
                WHERE 1=1 {coll_sql}
            """, tuple(coll_params))
            chunks_count = cursor.fetchone()[0] or 0
        else:
            cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
            chunks_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT collection) FROM documents")
            row = cursor.fetchone()
            docs_count = row[0] or 0
            colls_count = row[1] or 0

            cursor.execute("SELECT path, collection FROM documents")
            doc_rows = cursor.fetchall()
            dirs = {str(Path(p).parent) for p, _ in doc_rows if Path(p).parent != Path(".")}
            colls = {c for _, c in doc_rows if c}

        quant_type = get_db_meta(self.conn, "vector_quantization") or getattr(self.config, "vector_quantization", "none") or "none"
        dim = get_db_meta(self.conn, "vector_dim")

        file_size = 0
        if getattr(self.config, "db_path", None):
            try:
                p = Path(self.config.db_path)
                if p.exists():
                    file_size = p.stat().st_size
            except Exception:
                pass

        stats_data = {
            "chunks": chunks_count,
            "docs": docs_count,
            "dirs": dirs,
            "collections": colls,
            "quant_type": quant_type,
            "dim": dim,
            "file_size_bytes": file_size
        }
        self._local_stats_cache = {"key": cache_key, "data": stats_data}
        return stats_data

    def get_stats(self, collection: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        target_stores = self._get_target_stores_for_collection(collection)

        total_chunks = 0
        total_docs = 0
        total_size_bytes = 0
        all_dirs = set()
        all_colls = set()
        quant_types = set()
        dims = set()

        for store in target_stores:
            s_stats = store._get_local_stats(collection=collection)
            total_chunks += s_stats["chunks"]
            total_docs += s_stats["docs"]
            total_size_bytes += s_stats["file_size_bytes"]
            all_dirs.update(s_stats["dirs"])
            all_colls.update(s_stats["collections"])
            if s_stats["quant_type"]:
                quant_types.add(s_stats["quant_type"])
            if s_stats["dim"]:
                dims.add(s_stats["dim"])

        return {
            "chunks": total_chunks,
            "docs": total_docs,
            "dirs_count": len(all_dirs),
            "colls_count": len(all_colls),
            "collections": sorted(list(all_colls)),
            "size_bytes": total_size_bytes,
            "quant_type": "/".join(sorted(quant_types)) if quant_types else "none",
            "dim": "/".join(sorted(str(d) for d in dims)) if dims else None,
        }

    def _format_stats_line(self, stats: Dict[str, Any]) -> str:
        chunks = f"{stats['chunks']:,}"
        docs = f"{stats['docs']:,}"
        dirs = f"{stats['dirs_count']:,}"
        colls = f"{stats['colls_count']:,}"
        size_str = _format_size(stats['size_bytes']) if stats['size_bytes'] > 0 else ""

        vec_spec = []
        if stats.get('quant_type'):
            vec_spec.append(stats['quant_type'])
        if stats.get('dim'):
            vec_spec.append(f"{stats['dim']}d")
        vec_str = f", {', '.join(vec_spec)}" if vec_spec else ""

        size_part = f" ({size_str}{vec_str})" if (size_str or vec_str) else ""

        coll_word = "collection" if stats['colls_count'] == 1 else "collections"
        file_word = "file" if stats['docs'] == 1 else "files"
        chunk_word = "chunk" if stats['chunks'] == 1 else "chunks"
        dir_word = "directory" if stats['dirs_count'] == 1 else "directories"

        return f"{chunks} {chunk_word} across {docs} {file_word} across {dirs} {dir_word} in {colls} {coll_word}{size_part}"

    def _get_target_stores_for_collection(self, collection: Optional[Union[str, List[str]]]) -> List[Any]:
        if not self.child_stores:
            return [self]

        if not collection:
            return [self] + self.child_stores

        coll_tokens = []
        if isinstance(collection, str):
            if collection.strip():
                coll_tokens = [c.strip().lower() for c in collection.split(',') if c.strip()]
        elif isinstance(collection, (list, tuple, set)):
            coll_tokens = [str(c).strip().lower() for c in collection if str(c).strip()]

        if not coll_tokens:
            return [self] + self.child_stores

        matched_stores = set()
        for coll_name, store in self.collection_store_map.items():
            coll_lower = coll_name.lower()
            for tok in coll_tokens:
                if '*' in tok or '?' in tok:
                    if fnmatch.fnmatch(coll_lower, tok):
                        matched_stores.add(store)
                        break
                else:
                    if tok in coll_lower:
                        matched_stores.add(store)
                        break

        if matched_stores:
            ordered = []
            if self in matched_stores:
                ordered.append(self)
            for cs in self.child_stores:
                if cs in matched_stores and cs not in ordered:
                    ordered.append(cs)
            return ordered

        return [self] + self.child_stores

    def get_document_outline(
        self,
        collection: Optional[str],
        path: str,
        max_depth: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Extracts document structure heading hierarchy correlated with chunk sequence ranges."""
        target_stores = self._get_target_stores_for_collection(collection)
        for store in target_stores:
            if store is self:
                res = self._get_document_outline_local(collection, path, max_depth, pattern)
            else:
                res = store.get_document_outline(collection, path, max_depth, pattern)
            if res is not None:
                return res
        return None

    def _get_document_outline_local(
        self,
        collection: Optional[str],
        path: str,
        max_depth: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        coll_sql, coll_params = _build_collection_sql_filter("collection", collection)

        cursor.execute(f"SELECT id, collection, path, title, hash FROM documents WHERE path = ?{coll_sql}", (path, *coll_params))
        doc_row = cursor.fetchone()
        if not doc_row:
            cursor.execute(f"SELECT id, collection, path, title, hash FROM documents WHERE path LIKE ?{coll_sql}", (f"%{path}%", *coll_params))
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

        start_seqs = [_get_seq_for_offset(h["char_offset"]) for h in raw_headings]

        structured_headings = []
        num_headings = len(raw_headings)
        last_chunk_idx = (len(chunks_data) - 1) if chunks_data else 0

        for i, h in enumerate(raw_headings):
            h_text = h["text"]
            h_level = h["level"]
            start_seq = start_seqs[i]

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

        max_level_present = max((h["level"] for h in structured_headings), default=1) if structured_headings else 1

        effective_depth = None
        if isinstance(max_depth, int) and max_depth > 0:
            effective_depth = max_depth
            structured_headings = [h for h in structured_headings if h["level"] <= max_depth]
        elif max_depth is None:
            if len(structured_headings) <= 500:
                effective_depth = max_level_present
            else:
                chosen_depth = 1
                for d in range(1, max_level_present + 1):
                    count_at_d = sum(1 for h in structured_headings if h["level"] <= d)
                    if count_at_d <= 500:
                        chosen_depth = d
                    else:
                        break
                effective_depth = chosen_depth
                structured_headings = [h for h in structured_headings if h["level"] <= effective_depth]

        if isinstance(pattern, str) and pattern:
            p_lower = pattern.lower()
            structured_headings = [h for h in structured_headings if p_lower in h["text"].lower()]

        has_more_depth = (effective_depth is not None and effective_depth < max_level_present)

        return {
            "collection": coll_name,
            "path": doc_path,
            "title": title,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "depth": effective_depth,
            "max_depth": max_level_present,
            "has_more_depth": has_more_depth,
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

        res = self._get_chunk_by_id_local(rowids, window=window)
        if res:
            return res

        if self.child_stores:
            for child in self.child_stores:
                child_res = child.get_chunk_by_id(rowids, window=window)
                if child_res:
                    return child_res
        return []

    def _get_chunk_by_id_local(self, rowids: List[int], window: int = 0) -> List[Result]:
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
        target_stores = self._get_target_stores_for_collection(collection)
        for store in target_stores:
            if store is self:
                res = self._get_chunk_by_seq_local(collection, path, seq_id=seq_id, window=window)
            else:
                res = store.get_chunk_by_seq(collection, path, seq_id=seq_id, window=window)
            if res:
                return res
        return []

    def _get_chunk_by_seq_local(self, collection: Optional[str], path: str, seq_id: Union[int, List[int]] = 0, window: int = 0) -> List[Result]:
        cursor = self.conn.cursor()
        coll_sql, coll_params = _build_collection_sql_filter("collection", collection)

        cursor.execute(f"SELECT hash, collection, path, title FROM documents WHERE path = ?{coll_sql}", (path, *coll_params))
        row = cursor.fetchone()
        if not row:
            cursor.execute(f"SELECT hash, collection, path, title FROM documents WHERE path LIKE ?{coll_sql}", (f"%{path}%", *coll_params))
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
        target_stores = self._get_target_stores_for_collection(collection)
        all_trees: List[Dict[str, Any]] = []
        for store in target_stores:
            if store is self:
                tree = self._get_collection_tree_local(collection, max_depth, pattern, is_regex, case_sensitive)
            else:
                tree = store.get_collection_tree(collection, max_depth, pattern, is_regex, case_sensitive)
            if isinstance(tree, list):
                all_trees.extend(tree)
            elif isinstance(tree, dict):
                all_trees.append(tree)

        if collection and len(all_trees) == 1:
            return all_trees[0]
        if not all_trees and collection:
            return None
        return all_trees

    def _get_collection_tree_local(
        self,
        collection: Optional[str] = None,
        max_depth: Optional[int] = None,
        pattern: Optional[str] = None,
        is_regex: bool = False,
        case_sensitive: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        cursor = self.conn.cursor()

        if collection:
            coll_sql, coll_params = _build_collection_sql_filter("collection", collection)
            where_clause = " WHERE " + coll_sql[5:] if coll_sql else ""
            cursor.execute(f"SELECT DISTINCT collection FROM documents{where_clause} ORDER BY collection ASC", tuple(coll_params))
            collections = [row[0] for row in cursor.fetchall()]
            if not collections:
                return None
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
        collection: Optional[Union[str, List[str]]] = None,
        path: Optional[Union[str, List[str]]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Performs direct string or regular expression pattern search across decompressed document bodies."""
        if not pattern:
            return []

        target_stores = self._get_target_stores_for_collection(collection)
        all_matches = []
        for store in target_stores:
            if len(all_matches) >= limit:
                break
            rem = limit - len(all_matches)
            if store is self:
                matches = self._grep_search_local(pattern, is_regex=is_regex, case_sensitive=case_sensitive, collection=collection, path=path, limit=rem)
            else:
                matches = store.grep_search(pattern, is_regex=is_regex, case_sensitive=case_sensitive, collection=collection, path=path, limit=rem)
            all_matches.extend(matches)

        return all_matches[:limit]

    def _grep_search_local(
        self,
        pattern: str,
        is_regex: bool = False,
        case_sensitive: bool = False,
        collection: Optional[Union[str, List[str]]] = None,
        path: Optional[Union[str, List[str]]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:

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
        coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
        if coll_sql:
            where_clauses.append(coll_sql[5:])
            params.extend(coll_params)
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