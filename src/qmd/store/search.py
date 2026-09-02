import time
import json
import sqlite3
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from qmd.db import (
    is_sqlite_vec_active, get_db_meta, get_cached_query_embedding, save_query_embedding,
    get_cached_search_results, save_cached_search_results
)
from qmd.utils import compute_hash, build_spacy_fts_queries, decompress_text
from qmd.formatting import DIM, YELLOW, CYAN, RESET, MAGENTA, GREEN
from .models import Result, encode_vector, decode_vector, _build_collection_sql_filter, _results_to_json, _json_to_results


class SearchMixin:
    """Handles query caching, text hydration, FTS lexical search, vector KNN, hybrid search, W2N, and discovery."""

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
        use_cache: Optional[bool] = None,
        defer_text: bool = False
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
                    corpus_line = self._format_stats_line(self.get_stats(collection=collection))
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
                    if corpus_line:
                        print(f"{DIM}Corpus:{RESET} {corpus_line}")
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
        for q_idx, lq in enumerate(lex_queries):
            tier_results = self.search_fts(lq, limit=fts_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, defer_text=True)

            for r in tier_results:
                key = (r.collection, r.path, r.seq_id)
                if key not in seen_fts_keys:
                    seen_fts_keys.add(key)
                    r.fts_rank = len(fts_results) + 1
                    if q_idx > 0:
                        tier_penalty = 0.85 ** q_idx
                        r.score *= tier_penalty
                        if r.fts_score is not None:
                            r.fts_score *= tier_penalty
                    fts_results.append(r)

            if len(fts_results) >= min_fts_quorum:
                break

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
                or_results = self.search_fts(or_query, limit=fts_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, defer_text=True)
                for r in or_results:
                    key = (r.collection, r.path, r.seq_id)
                    if key not in seen_fts_keys:
                        seen_fts_keys.add(key)
                        r.fts_rank = len(fts_results) + 1
                        r.score *= 0.5
                        if r.fts_score is not None:
                            r.fts_score *= 0.5
                        fts_results.append(r)
        t_fts = (time.perf_counter() - t_fts_start) * 1000

        t_embed_start = time.perf_counter()
        vec_query_embeddings = {}
        for vq in vec_queries:
            query_text = self.llm.format_query_for_embedding(vq)
            q_vec = self.get_query_embedding(query_text)
            if q_vec:
                vec_query_embeddings[vq] = q_vec
        t_embed = (time.perf_counter() - t_embed_start) * 1000

        t_vec_start = time.perf_counter()
        vec_results = []
        for vq in vec_queries:
            q_vec = vec_query_embeddings.get(vq)
            if q_vec:
                vec_results.extend(self.search_vec(vq, limit=vec_lim, collection=collection, title=title, path=path, exclude_seen_set=exclude_seen_set, excluded_chunks_tracker=excluded_chunks_tracker, query_vec=q_vec, defer_text=True))
        t_vec_knn = (time.perf_counter() - t_vec_start) * 1000

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
                corpus_line = self._format_stats_line(self.get_stats(collection=collection))
                t_total = (time.perf_counter() - t_total_start) * 1000
                print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
                if corpus_line:
                    print(f"{DIM}Corpus:{RESET} {corpus_line}")
                print(f"{DIM}Query:{RESET} {query}")
                print(f"{YELLOW}Lexical (FTS):{RESET} {lex_queries}")
                print(f"{MAGENTA}Vector (Semantic):{RESET} {vec_queries}")
                print(f"{DIM}Candidates: FTS={len(fts_results)} | Vector={len(vec_results)} | RRF=0{RESET}")
                print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                print(f"  • Query Expansion:         {t_expand:>7.2f} ms")
                print(f"  • Lexical (FTS) Search:    {t_fts:>7.2f} ms ({len(fts_results)} hits)")
                print(f"  • Query Embedding (LLM):   {t_embed:>7.2f} ms")
                print(f"  • Vector (KNN) Search:     {t_vec_knn:>7.2f} ms ({len(vec_results)} hits)")
                print(f"  • RRF Fusion:              {t_rrf:>7.2f} ms")
                print(f"  • Total Search:            {t_total:>7.2f} ms")
                print(f"{CYAN}------------------------{RESET}\n")
            return []

        final_results = []
        t_rerank = 0.0

        if not rerank and not reranker_only:
            for res in top_candidates:
                res.score = rrf_scores[(res.collection, res.path, res.seq_id)]
                res.source = "hybrid"
                final_results.append(res)
        else:
            t_rerank_start = time.perf_counter()
            self._hydrate_results_text(top_candidates)
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
                    if key in sources_found and "fts" in sources_found[key] and "vec" in sources_found[key]:
                        dual_floor = 0.80 * n_rrf
                        final_score = max(final_score, dual_floor) + 0.15

                res.score = final_score
                res.source = "hybrid"
                final_results.append(res)
            t_rerank = (time.perf_counter() - t_rerank_start) * 1000

        t_dedup_start = time.perf_counter()
        final_results.sort(key=lambda x: x.score, reverse=True)

        unique_results = []
        seen_content = set()

        if not defer_text and not (rerank or reranker_only):
            self._hydrate_results_text(final_results)

        for res in final_results:
            if res.text:
                content_sig = res.text.strip()
            else:
                content_sig = f"{res.collection}:{res.path}:{res.seq_id}"
            if content_sig not in seen_content:
                unique_results.append(res)
                seen_content.add(content_sig)

        for i, res in enumerate(unique_results):
            res.rank = i + 1

        ret_results = unique_results[:limit]
        if not defer_text:
            self._hydrate_results_text(ret_results)
        if should_cache and not exclude_seen_set:
            self._hydrate_results_text(ret_results)
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
        t_dedup = (time.perf_counter() - t_dedup_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            corpus_line = self._format_stats_line(self.get_stats(collection=collection))
            print(f"\n{CYAN}--- Search Diagnostics ---{RESET}")
            if corpus_line:
                print(f"{DIM}Corpus:{RESET} {corpus_line}")
            print(f"{DIM}Query:{RESET} {query}")
            print(f"{YELLOW}Lexical (FTS):{RESET} {lex_queries}")
            print(f"{MAGENTA}Vector (Semantic):{RESET} {vec_queries}")
            print(f"{DIM}Candidates: FTS={len(fts_results)} | Vector={len(vec_results)} | RRF={len(top_candidates)}{RESET}")
            print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
            print(f"  • Query Expansion:         {t_expand:>7.2f} ms")
            print(f"  • Lexical (FTS) Search:    {t_fts:>7.2f} ms ({len(fts_results)} hits)")
            print(f"  • Query Embedding (LLM):   {t_embed:>7.2f} ms")
            print(f"  • Vector (KNN) Search:     {t_vec_knn:>7.2f} ms ({len(vec_results)} hits)")
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
        use_cache: Optional[bool] = None,
        defer_text: bool = False
    ) -> List[Result]:
        """Hierarchical Wide-to-Narrow (W2N) Search."""
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
                    corpus_line = self._format_stats_line(self.get_stats(collection=collection))
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Wide-to-Narrow Diagnostics ---{RESET}")
                    if corpus_line:
                        print(f"{DIM}Corpus:{RESET} {corpus_line}")
                    print(f"{DIM}Original Query:{RESET} {query}")
                    print(f"{GREEN}[Cache Hit]{RESET} Loaded results from search cache in {t_cache:.2f}ms")
                    print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
                    print(f"  • Cache Lookup:            {t_cache:>7.2f} ms")
                    print(f"  • Total Wide-to-Narrow:    {t_total:>7.2f} ms")
                    print(f"{CYAN}------------------------{RESET}\n")
                return _json_to_results(cached_json)

        cfg_max = max(getattr(self.config, 'fts_limit', 50), getattr(self.config, 'vec_limit', 50))
        wide_limit = max(cfg_max, limit * 5)
        t_wide_start = time.perf_counter()
        wide_results = self.hybrid_search(
            query,
            limit=wide_limit,
            verbose=False,
            rerank=False,
            reranker_only=False,
            collection=collection,
            lexical_query=lexical_query,
            title=title,
            path=path,
            fts_limit=fts_limit,
            vec_limit=vec_limit,
            rerank_candidates=rerank_candidates,
            exclude_seen_set=exclude_seen_set,
            use_cache=False,
            defer_text=True
        )
        t_wide = (time.perf_counter() - t_wide_start) * 1000

        if not wide_results:
            return []

        t_agg_start = time.perf_counter()
        doc_scores: Dict[str, float] = {}
        dir_scores: Dict[str, float] = {}

        for res in wide_results:
            doc_scores[res.path] = doc_scores.get(res.path, 0.0) + res.score
            parent_dir = str(Path(res.path).parent)
            if parent_dir and parent_dir != ".":
                dir_scores[parent_dir] = dir_scores.get(parent_dir, 0.0) + (res.score * 0.75)

        top_docs = set(sorted(doc_scores, key=doc_scores.get, reverse=True)[:top_containers])
        top_dirs = set(sorted(dir_scores, key=dir_scores.get, reverse=True)[:top_containers])
        t_agg = (time.perf_counter() - t_agg_start) * 1000

        t_boost_start = time.perf_counter()
        boosted_results = []
        for res in wide_results:
            parent_dir = str(Path(res.path).parent)
            is_top_doc = res.path in top_docs
            is_top_dir = parent_dir in top_dirs

            if is_top_doc or is_top_dir:
                multiplier = 1.35 if is_top_doc else 1.15
                res.score *= multiplier
                boosted_results.append(res)

        if len(boosted_results) < limit:
            seen_signatures = {(r.collection, r.path, r.seq_id) for r in boosted_results}
            for res in wide_results:
                sig = (res.collection, res.path, res.seq_id)
                if sig not in seen_signatures:
                    boosted_results.append(res)
                    seen_signatures.add(sig)

        boosted_results.sort(key=lambda x: x.score, reverse=True)
        candidate_pool = boosted_results[:limit * 2]
        t_boost = (time.perf_counter() - t_boost_start) * 1000

        t_rerank = 0.0
        if rerank or reranker_only:
            t_rerank_start = time.perf_counter()
            self._hydrate_results_text(candidate_pool)
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
        if not defer_text:
            self._hydrate_results_text(ret_results)
        if should_cache and not exclude_seen_set:
            self._hydrate_results_text(ret_results)
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(ret_results))
        t_dedup = (time.perf_counter() - t_dedup_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            corpus_line = self._format_stats_line(self.get_stats(collection=collection))
            print(f"\n{CYAN}--- Search Diagnostics (Wide-to-Narrow) ---{RESET}")
            if corpus_line:
                print(f"{DIM}Corpus:{RESET} {corpus_line}")
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
        """Discover Search: Single top hit per document with aggregated match counts."""
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
                    corpus_line = self._format_stats_line(self.get_stats(collection=collection))
                    t_total = (time.perf_counter() - t_total_start) * 1000
                    print(f"\n{CYAN}--- Discover Diagnostics ---{RESET}")
                    if corpus_line:
                        print(f"{DIM}Corpus:{RESET} {corpus_line}")
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
                verbose=False,
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
                use_cache=False,
                defer_text=True
            )
        else:
            candidates = self.hybrid_search(
                query=query,
                limit=fetch_limit,
                verbose=False,
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
                use_cache=False,
                defer_text=True
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
        self._hydrate_results_text(final_results)

        if should_cache and not exclude_seen_set:
            save_cached_search_results(self.history_conn, cache_key, query, _results_to_json(final_results))
        t_group = (time.perf_counter() - t_group_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        if verbose:
            corpus_line = self._format_stats_line(self.get_stats(collection=collection))
            print(f"\n{CYAN}--- Search Diagnostics (Discover) ---{RESET}")
            if corpus_line:
                print(f"{DIM}Corpus:{RESET} {corpus_line}")
            print(f"{DIM}Query:{RESET} {query}")
            print(f"{DIM}Mode:{RESET} {'Wide-to-Narrow' if w2n else 'Hybrid'} | {DIM}Rerank:{RESET} {rerank or reranker_only}")
            print(f"{DIM}Candidates: {len(candidates)} chunks -> {len(final_results)} top documents{RESET}")
            print(f"\n{CYAN}--- Timing Breakdown ---{RESET}")
            print(f"  • Candidate Retrieval:     {t_cand:>7.2f} ms ({len(candidates)} chunks)")
            print(f"  • Top-Hit Doc Grouping:    {t_group:>7.2f} ms ({len(final_results)} documents)")
            print(f"  • Total Search:            {t_total:>7.2f} ms")
            print(f"{CYAN}---------------------------------{RESET}\n")

        return final_results