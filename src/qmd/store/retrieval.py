import json
import sqlite3
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union

from qmd.db import (
    is_sqlite_vec_active, get_db_meta, get_cached_query_embedding, save_query_embedding
)
from qmd.utils import compute_hash, decompress_text
from qmd.formatting import YELLOW, RESET
from .models import Result, encode_vector, decode_vector, _build_collection_sql_filter


class RetrievalMixin:
    """Handles basic FTS lexical search, vector KNN, text hydration, and query embeddings."""

    def _hydrate_results_text_local(self, results: List[Result]):
        needed = [r for r in results if not r.text]
        if not needed:
            return

        cursor = self.conn.cursor()
        with_rowids = [r for r in needed if r.rowid is not None]
        without_rowids = [r for r in needed if r.rowid is None]

        if with_rowids:
            unique_rowids = list({r.rowid for r in with_rowids})
            placeholders = ','.join(['?'] * len(unique_rowids))
            cursor.execute(
                f"SELECT rowid, chunk_text FROM chunk_metadata WHERE rowid IN ({placeholders})",
                tuple(unique_rowids)
            )
            text_map = {row[0]: decompress_text(row[1]) for row in cursor.fetchall()}
            for r in with_rowids:
                if r.rowid in text_map:
                    r.text = text_map[r.rowid]

        if without_rowids:
            for r in without_rowids:
                cursor.execute("""
                    SELECT m.chunk_text, m.rowid
                    FROM chunk_metadata m
                    JOIN documents d ON m.doc_hash = d.hash
                    WHERE d.collection = ? AND d.path = ? AND m.seq_id = ?
                """, (r.collection, r.path, r.seq_id))
                row = cursor.fetchone()
                if row:
                    r.text = decompress_text(row[0])
                    if r.rowid is None and row[1] is not None:
                        r.rowid = row[1]

    def _hydrate_results_text(self, results: List[Result]):
        """Hydrates missing or deferred text in Result objects across local and federated stores."""
        needed = [r for r in results if not r.text]
        if not needed:
            return

        if self.child_stores:
            store_batches: Dict[Any, List[Result]] = {}
            for r in needed:
                target_store = self.collection_store_map.get(r.collection, self)
                store_batches.setdefault(target_store, []).append(r)

            if len(store_batches) > 1:
                def _hydrate_batch(entry):
                    st, batch = entry
                    if st is self:
                        self._hydrate_results_text_local(batch)
                    else:
                        st._hydrate_results_text(batch)

                with ThreadPoolExecutor(max_workers=min(len(store_batches), 8)) as executor:
                    list(executor.map(_hydrate_batch, store_batches.items()))
            else:
                for target_store, batch in store_batches.items():
                    if target_store is self:
                        self._hydrate_results_text_local(batch)
                    else:
                        target_store._hydrate_results_text(batch)
        else:
            self._hydrate_results_text_local(needed)

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
        all_stores = [self] + getattr(self, "child_stores", [])
        db_last_updated = ",".join([get_db_meta(s.conn, "last_updated") or "" for s in all_stores])
        embed_model = getattr(self.config, "embed_model", "")
        rerank_model = getattr(self.config, "rerank_model", "")

        if isinstance(path, (list, tuple, set)):
            path_val: Union[str, List[str]] = sorted([str(p).strip() for p in path if str(p).strip()])
        else:
            path_val = path or ""

        if isinstance(collection, (list, tuple, set)):
            coll_val: Union[str, List[str]] = sorted([str(c).strip() for c in collection if str(c).strip()])
        else:
            coll_val = collection or ""

        key_data = {
            "search_type": search_type,
            "query": query,
            "limit": limit,
            "rerank": rerank,
            "reranker_only": reranker_only,
            "collection": coll_val,
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

    def _search_fts_local(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None, defer_text: bool = False) -> List[Result]:
        limit = limit if limit is not None else getattr(self.config, 'fts_limit', 50)
        if '"' in query or ' AND ' in query or ' OR ' in query or ' NOT ' in query:
            fts_query = query
        else:
            sanitized = query.replace('"', '')
            raw_terms = sanitized.split()
            stop_words = {
                "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", 
                "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", 
                "that", "the", "their", "then", "there", "these", "they", "this", 
                "to", "was", "will", "with"
            }
            terms = [t for t in raw_terms if t.lower() not in stop_words]
            if not terms:
                terms = raw_terms
            if not terms:
                return []
            fts_query = " AND ".join([f'"{term}"' for term in terms])

        cursor = self.conn.cursor()
        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]

        body_col = "'' AS body" if defer_text else "f.body"
        base_sql = f"""
            SELECT f.filepath, f.title, {body_col}, f.rank, f.collection, f.rowid, m.seq_id, COALESCE(f.headers, '')
            FROM chunks_fts f
            JOIN chunk_metadata m ON f.rowid = m.rowid
            WHERE chunks_fts MATCH ?
        """
        coll_sql, coll_params = _build_collection_sql_filter("f.collection", collection)
        filters_sql = coll_sql
        if title: filters_sql += " AND f.title LIKE ?"
        if paths:
            path_clauses = " OR ".join(["f.filepath LIKE ?" for _ in paths])
            filters_sql += f" AND ({path_clauses})"

        order_sql = " ORDER BY f.rank LIMIT ?"
        sql_limit = max(limit * 5, limit + (len(exclude_seen_set) * 2 if exclude_seen_set else 0)) if exclude_seen_set else limit

        def build_params(q_val):
            p = [q_val]
            p.extend(coll_params)
            if title: p.append(f"%{title}%")
            for p_val in paths:
                p.append(f"%{p_val}%")
            p.append(sql_limit)
            return p

        try:
            cursor.execute(base_sql + filters_sql + order_sql, tuple(build_params(fts_query)))
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
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

            raw_bm25 = -float(rank) if float(rank) < 0 else float(rank)
            calculated_fts_score = max(0.0001, raw_bm25)
            calculated_fts_rank = len(results) + 1

            results.append(Result(
                path=doc_path,
                title=doc_title,
                text=chunk_text if not defer_text else "",
                score=calculated_fts_score,
                source="fts",
                collection=coll or "",
                seq_id=seq_id,
                headers=headers,
                fts_score=calculated_fts_score,
                fts_rank=calculated_fts_rank,
                rowid=rowid
            ))
            if len(results) >= limit:
                break

        return results

    def search_fts(self, query: str, limit: Optional[int] = None, collection: Optional[Union[str, List[str]]] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None, defer_text: bool = False) -> List[Result]:
        """Lexical search directly on chunks across local and federated stores."""
        limit = limit if limit is not None else getattr(self.config, 'fts_limit', 50)
        target_stores = self._get_target_stores_for_collection(collection)
        if len(target_stores) == 1 and target_stores[0] is self:
            return self._search_fts_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, defer_text=defer_text)

        all_results = []
        if len(target_stores) > 1:
            lock = threading.Lock()

            def _query_store_fts(s: Any) -> List[Result]:
                local_tracker = set() if excluded_chunks_tracker is not None else None
                if s is self:
                    res = self._search_fts_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=local_tracker, defer_text=defer_text)
                else:
                    res = s.search_fts(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=local_tracker, defer_text=defer_text)
                if local_tracker and excluded_chunks_tracker is not None:
                    with lock:
                        excluded_chunks_tracker.update(local_tracker)
                return res

            with ThreadPoolExecutor(max_workers=min(len(target_stores), 8)) as executor:
                for res_list in executor.map(_query_store_fts, target_stores):
                    all_results.extend(res_list)
        else:
            for store in target_stores:
                if store is self:
                    res = self._search_fts_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, defer_text=defer_text)
                else:
                    res = store.search_fts(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, defer_text=defer_text)
                all_results.extend(res)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]

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

    def _search_vec_local(self, query: str, limit: Optional[int] = None, collection: Optional[str] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None, query_vec: Optional[List[float]] = None, defer_text: bool = False) -> List[Result]:
        limit = limit if limit is not None else getattr(self.config, 'vec_limit', 50)
        if query_vec is None:
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

        if getattr(self, 'usearch_index', None) is not None:
            try:
                import numpy as np
                extra_seen = (len(exclude_seen_set) * 2) if exclude_seen_set else 0
                k_val = max(limit * 8 + extra_seen, 500) if (collection or title or paths or exclude_seen_set) else limit

                query_arr = np.array(query_vec, dtype=np.float32)
                matches = self.usearch_index.search(query_arr, k_val)

                if len(matches) > 0:
                    match_keys = matches.keys.tolist() if hasattr(matches.keys, 'tolist') else list(matches.keys)
                    match_distances = matches.distances.tolist() if hasattr(matches.distances, 'tolist') else list(matches.distances)
                    dist_map = dict(zip(match_keys, match_distances))

                    placeholders = ','.join(['?'] * len(match_keys))
                    chunk_col = "NULL AS chunk_text" if defer_text else "m.chunk_text"

                    query_sql = f"""
                        SELECT m.rowid, {chunk_col}, d.path, d.title, d.collection, m.seq_id, COALESCE(m.headers, '')
                        FROM chunk_metadata m
                        JOIN documents d ON m.doc_hash = d.hash
                        WHERE m.rowid IN ({placeholders})
                    """
                    params = list(match_keys)

                    coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
                    query_sql += coll_sql
                    params.extend(coll_params)

                    if title:
                        query_sql += " AND d.title LIKE ?"
                        params.append(f"%{title}%")
                    if paths:
                        path_clauses = " OR ".join(["d.path LIKE ?" for _ in paths])
                        query_sql += f" AND ({path_clauses})"
                        for p_val in paths:
                            params.append(f"%{p_val}%")

                    cursor.execute(query_sql, tuple(params))
                    raw_candidates = []
                    for r_id, text_blob, doc_path, doc_title, coll, seq_id, hdrs in cursor.fetchall():
                        key = (coll or "", doc_path, seq_id)
                        if exclude_seen_set and key in exclude_seen_set:
                            if excluded_chunks_tracker is not None:
                                excluded_chunks_tracker.add(key)
                            continue

                        dist = dist_map.get(r_id, 1.0)
                        if quant_type in ("bit", "binary"):
                            score = 1.0 / (1.0 + float(dist))
                        else:
                            score = max(0.0, 1.0 - float(dist))

                        raw_candidates.append((score, doc_path, doc_title, text_blob, coll, seq_id, hdrs, r_id))

                    raw_candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates = []
                    for vec_idx, (score, doc_path, doc_title, text_blob, coll, seq_id, hdrs, r_id) in enumerate(raw_candidates[:limit]):
                        chunk_str = decompress_text(text_blob) if (not defer_text and text_blob is not None) else ""
                        candidates.append(Result(
                            path=doc_path, title=doc_title, text=chunk_str, score=score,
                            source="vec", collection=coll, seq_id=seq_id, headers=hdrs,
                            vec_score=score, vec_rank=vec_idx + 1,
                            rowid=r_id
                        ))
                    return candidates
            except Exception as e:
                print(f"{YELLOW}Warning: usearch ANN query failed ({e}), using fallback scanner.{RESET}")

        if is_sqlite_vec_active(self.conn):
            try:
                query_blob = encode_vector(query_vec, quant_type)
                extra_seen = (len(exclude_seen_set) * 2) if exclude_seen_set else 0
                k_val = max(limit * 8 + extra_seen, 500) if (collection or title or paths or exclude_seen_set) else limit

                chunk_col = "NULL AS chunk_text" if defer_text else "m.chunk_text"
                query_sql = f"""
                    SELECT v.rowid, v.distance, {chunk_col}, d.path, d.title, d.collection, m.seq_id, COALESCE(m.headers, '')
                    FROM (
                        SELECT rowid, distance
                        FROM vectors
                        WHERE embedding MATCH ? AND k = ?
                    ) v
                    JOIN chunk_metadata m ON v.rowid = m.rowid
                    JOIN documents d ON m.doc_hash = d.hash
                    WHERE 1=1
                """
                params = [query_blob, k_val]

                coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
                query_sql += coll_sql
                params.extend(coll_params)
                if title:
                    query_sql += " AND d.title LIKE ?"
                    params.append(f"%{title}%")
                if paths:
                    path_clauses = " OR ".join(["d.path LIKE ?" for _ in paths])
                    query_sql += f" AND ({path_clauses})"
                    for p_val in paths:
                        params.append(f"%{p_val}%")

                cursor.execute(query_sql, tuple(params))
                raw_candidates = []
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

                    raw_candidates.append((score, doc_path, doc_title, text_blob, coll, seq_id, hdrs, rowid))

                raw_candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = []
                for vec_idx, (score, doc_path, doc_title, text_blob, coll, seq_id, hdrs, r_id) in enumerate(raw_candidates[:limit]):
                    chunk_str = decompress_text(text_blob) if (not defer_text and text_blob is not None) else ""
                    candidates.append(Result(
                        path=doc_path, title=doc_title, text=chunk_str, score=score,
                        source="vec", collection=coll, seq_id=seq_id, headers=hdrs,
                        vec_score=score, vec_rank=vec_idx + 1,
                        rowid=r_id
                    ))
                return candidates
            except sqlite3.OperationalError as e:
                print(f"{YELLOW}Warning: sqlite-vec accelerated query failed ({e}), using fallback scanner.{RESET}")

        chunk_col = "NULL AS chunk_text" if defer_text else "m.chunk_text"
        query_sql = f"""
            SELECT v.embedding, {chunk_col}, d.path, d.title, d.collection, m.seq_id, COALESCE(m.headers, ''), v.rowid
            FROM vectors v
            JOIN chunk_metadata m ON v.rowid = m.rowid
            JOIN documents d ON m.doc_hash = d.hash
        """
        where_clauses = []
        params = []
        coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
        if coll_sql:
            where_clauses.append(coll_sql[5:])
            params.extend(coll_params)
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
        mag_q = math.sqrt(sum(a * a for a in query_vec))
        if mag_q == 0:
            return []

        raw_candidates = []
        for emb_blob, text_blob, doc_path, doc_title, coll, seq_id, hdrs, r_id in cursor.fetchall():
            key = (coll or "", doc_path, seq_id)
            if exclude_seen_set and key in exclude_seen_set:
                if excluded_chunks_tracker is not None:
                    excluded_chunks_tracker.add(key)
                continue
            vec = decode_vector(emb_blob, dim, quant_type)

            dot_prod = sum(a * b for a, b in zip(query_vec, vec))
            mag_v = math.sqrt(sum(a * a for a in vec))
            sim = dot_prod / (mag_q * mag_v) if mag_v else 0

            raw_candidates.append((sim, doc_path, doc_title, text_blob, coll, seq_id, hdrs, r_id))

        raw_candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for vec_idx, (score, doc_path, doc_title, text_blob, coll, seq_id, hdrs, r_id) in enumerate(raw_candidates[:limit]):
            chunk_str = decompress_text(text_blob) if (not defer_text and text_blob is not None) else ""
            candidates.append(Result(
                path=doc_path, title=doc_title, text=chunk_str, score=score, 
                source="vec", collection=coll, seq_id=seq_id, headers=hdrs,
                vec_score=score, vec_rank=vec_idx + 1,
                rowid=r_id
            ))
        return candidates

    def search_vec(self, query: str, limit: Optional[int] = None, collection: Optional[Union[str, List[str]]] = None, title: Optional[str] = None, path: Optional[Union[str, List[str]]] = None, exclude_seen_set: Optional[set] = None, excluded_chunks_tracker: Optional[set] = None, query_vec: Optional[List[float]] = None, defer_text: bool = False) -> List[Result]:
        limit = limit if limit is not None else getattr(self.config, 'vec_limit', 50)
        if query_vec is None:
            query_text = self.llm.format_query_for_embedding(query)
            query_vec = self.get_query_embedding(query_text)
        if not query_vec:
            return []

        target_stores = self._get_target_stores_for_collection(collection)
        if len(target_stores) == 1 and target_stores[0] is self:
            return self._search_vec_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, query_vec=query_vec, defer_text=defer_text)

        all_results = []
        if len(target_stores) > 1:
            lock = threading.Lock()

            def _query_store_vec(s: Any) -> List[Result]:
                local_tracker = set() if excluded_chunks_tracker is not None else None
                if s is self:
                    res = self._search_vec_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=local_tracker, query_vec=query_vec, defer_text=defer_text)
                else:
                    res = s.search_vec(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=local_tracker, query_vec=query_vec, defer_text=defer_text)
                if local_tracker and excluded_chunks_tracker is not None:
                    with lock:
                        excluded_chunks_tracker.update(local_tracker)
                return res

            with ThreadPoolExecutor(max_workers=min(len(target_stores), 8)) as executor:
                for res_list in executor.map(_query_store_vec, target_stores):
                    all_results.extend(res_list)
        else:
            for store in target_stores:
                if store is self:
                    res = self._search_vec_local(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, query_vec=query_vec, defer_text=defer_text)
                else:
                    res = store.search_vec(query, limit=limit, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, query_vec=query_vec, defer_text=defer_text)
                all_results.extend(res)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]