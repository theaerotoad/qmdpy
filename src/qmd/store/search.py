import time
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

from qmd.db import get_cached_search_results, save_cached_search_results
from qmd.utils import build_spacy_fts_queries
from qmd.formatting import DIM, YELLOW, CYAN, RESET, MAGENTA, GREEN
from .models import Result, _results_to_json, _json_to_results


class SearchMixin:
    """Handles hybrid search, wide-to-narrow search, and document discovery."""

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