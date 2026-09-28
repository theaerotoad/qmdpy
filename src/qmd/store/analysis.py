import time
import json
from datetime import datetime
from typing import List, Optional, Union, Dict, Any, Tuple

from tqdm import tqdm

from qmd.utils import decompress_text
from .models import _build_collection_sql_filter

class AnalysisMixin:
    """Handles LLM-based metadata extraction and summarization on document subsets."""

    def get_analysis_report(
        self,
        collection: Optional[str] = None,
        path: Optional[Union[str, List[str]]] = None,
        title: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetches existing analysis reports from the database without invoking the LLM."""
        cursor = self.conn.cursor()

        query_sql = """
            SELECT d.hash, d.path, d.title, d.collection,
                   da.summary, da.authors, da.tags, da.dates, da.questions, da.doc_type, da.alt_title
            FROM documents d
            JOIN document_analysis da ON d.hash = da.doc_hash
        """
        where_clauses = []
        params = []

        coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
        if coll_sql:
            where_clauses.append(coll_sql[5:])  # Remove leading " AND "
            params.extend(coll_params)

        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]

        if paths:
            where_clauses.append("(" + " OR ".join(["d.path LIKE ?" for _ in paths]) + ")")
            for p_val in paths:
                params.append(f"%{p_val}%")

        if title:
            where_clauses.append("d.title LIKE ?")
            params.append(f"%{title}%")

        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)

        query_sql += " ORDER BY da.analyzed_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query_sql, tuple(params))
        rows = cursor.fetchall()

        results = []
        for r in rows:
            doc_hash, doc_path, doc_title, doc_coll, summary, authors_raw, tags_raw, dates_raw, questions_raw, doc_type, alt_title = r

            def safe_json(val, default):
                if not val: return default
                try: return json.loads(val)
                except: return default

            analysis_res = {
                "summary": summary or "",
                "authors": safe_json(authors_raw, []),
                "tags": safe_json(tags_raw, []),
                "dates": safe_json(dates_raw, []),
                "questions": safe_json(questions_raw, []),
                "doc_type": doc_type or "",
                "altTitle": alt_title or ""
            }

            results.append({
                "path": doc_path,
                "collection": doc_coll,
                "hash": doc_hash,
                "title": doc_title,
                **analysis_res
            })

        return results

    def build_custom_report(self, report_cfg: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Builds a highly customizable dataset based on YAML or CLI defined fields and scoped searches."""
        target_cfg = report_cfg.get("target", {})
        req_fields = report_cfg.get("fields", [])
        dynamic_queries = report_cfg.get("dynamic_queries", [])
        sort_rules = report_cfg.get("sort", [])

        # 1. Extract base documents, joining document_analysis
        base_results = self.get_analysis_report(
            collection=target_cfg.get("collection"),
            path=target_cfg.get("path"),
            title=target_cfg.get("title"),
            limit=10000  # Pull wide pool first before memory-sort and limit
        )

        cursor = self.conn.cursor()

        # 2. Process dynamic fields and queries per document
        enriched_results = []
        for doc in base_results:
            # Augment with explicit DB fields (like chunk_count) if needed
            if "chunk_count" in req_fields or any(s.get("field") == "chunk_count" for s in sort_rules):
                cursor.execute("SELECT count(*) FROM chunk_metadata WHERE doc_hash = ?", (doc["hash"],))
                doc["chunk_count"] = cursor.fetchone()[0]

            # Execute Dynamic Scoped Queries
            for dq in dynamic_queries:
                q_id = dq.get("id")
                if not q_id or dq.get("type") != "search_snippets":
                    continue
                
                q_limit = dq.get("limit", 3)
                snippets = self.hybrid_search(
                    query=dq.get("query", ""),
                    collection=doc["collection"],
                    path=doc["path"],
                    limit=q_limit,
                    verbose=False
                )
                
                fmt = dq.get("format", "list")
                if fmt == "bulleted":
                    doc[q_id] = "\n".join([f"- {s.text.strip()}" for s in snippets])
                else:
                    doc[q_id] = [s.text.strip() for s in snippets]

            enriched_results.append(doc)

        # 3. Apply Multi-key Sorting
        if sort_rules:
            def sort_key(item: Dict) -> Tuple:
                keys = []
                for rule in sort_rules:
                    val = item.get(rule["field"])
                    # Convert lists to strings for comparison safety
                    if isinstance(val, list):
                        val = str(val)
                    # Handle missing numeric/string sorting natively
                    if val is None:
                        keys.append("")
                    else:
                        keys.append(val)
                return tuple(keys)
            
            # Python sorting is stable, so we apply reverse sorts sequentially
            # from least important to most important (backwards through the sort rules list)
            for rule in reversed(sort_rules):
                is_reverse = str(rule.get("order", "asc")).lower() in ("desc", "descending", "-", "reverse")
                enriched_results.sort(
                    key=lambda x: str(x.get(rule["field"], "")) if isinstance(x.get(rule["field"]), list) else (x.get(rule["field"]) or ""), 
                    reverse=is_reverse
                )

        # 4. Apply Final Limit
        enriched_results = enriched_results[:limit]

        # 5. Filter Dictionary Keys (Field Selection)
        if req_fields:
            final_results = []
            for doc in enriched_results:
                final_doc = {}
                for f in req_fields:
                    if f in doc:
                        final_doc[f] = doc[f]
                final_results.append(final_doc)
            return final_results

        return enriched_results

    def get_random_analysis_questions(self, limit: int = 3) -> List[Dict[str, str]]:
        """Fetches random questions generated from document analysis to display as suggestions."""
        import random
        questions = []
        
        # Check all target stores in case of federation
        target_stores = [self]
        if hasattr(self, "collection_store_map") and self.collection_store_map:
            target_stores = list(self.collection_store_map.values())
            random.shuffle(target_stores)
        
        for ts in target_stores:
            cursor = ts.conn.cursor()
            try:
                cursor.execute("""
                    SELECT da.questions, d.title
                    FROM document_analysis da
                    JOIN documents d ON d.hash = da.doc_hash
                    WHERE da.questions IS NOT NULL AND da.questions != '[]' AND da.questions != ''
                    ORDER BY RANDOM()
                    LIMIT 10
                """)
                rows = cursor.fetchall()
                for q_raw, title in rows:
                    try:
                        q_list = json.loads(q_raw)
                        if q_list:
                            q = random.choice(q_list)
                            questions.append({"question": q, "title": title})
                            if len(questions) >= limit:
                                return questions
                    except Exception:
                        continue
            except Exception:
                pass
                
        return questions

    def analyze_target(
        self,
        collection: Optional[str] = None,
        path: Optional[Union[str, List[str]]] = None,
        title: Optional[str] = None,
        limit: int = 100,
        time_limit: Optional[float] = None,
        force: bool = False,
        outdated: bool = False,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        if getattr(self, "read_only", False):
            raise RuntimeError("Cannot analyze documents in read-only mode.")

        cursor = self.conn.cursor()

        query_sql = """
            SELECT d.hash, d.path, d.title, d.collection
            FROM documents d
        """
        where_clauses = []
        params = []

        if not force:
            query_sql += " LEFT JOIN document_analysis da ON d.hash = da.doc_hash"
            if outdated:
                prompt_ver = getattr(self.llm, "ANALYSIS_PROMPT_VERSION", "1.0")
                where_clauses.append("(da.doc_hash IS NULL OR da.prompt_version IS NULL OR da.prompt_version != ?)")
                params.append(prompt_ver)
            else:
                where_clauses.append("da.doc_hash IS NULL")

        coll_sql, coll_params = _build_collection_sql_filter("d.collection", collection)
        if coll_sql:
            where_clauses.append(coll_sql[5:])  # Remove leading " AND "
            params.extend(coll_params)

        paths = []
        if isinstance(path, str):
            if path.strip():
                paths = [p.strip() for p in path.split(',') if p.strip()]
        elif isinstance(path, (list, tuple, set)):
            paths = [str(p).strip() for p in path if str(p).strip()]

        if paths:
            where_clauses.append("(" + " OR ".join(["d.path LIKE ?" for _ in paths]) + ")")
            for p_val in paths:
                params.append(f"%{p_val}%")

        if title:
            where_clauses.append("d.title LIKE ?")
            params.append(f"%{title}%")

        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)

        query_sql += " LIMIT ?"
        params.append(limit)

        cursor.execute(query_sql, tuple(params))
        docs_to_analyze = cursor.fetchall()

        results = []
        if not docs_to_analyze:
            return results

        start_time = time.time()
        pbar = tqdm(docs_to_analyze, desc="Analyzing documents", unit="doc")
        for doc_hash, doc_path, doc_title, doc_coll in pbar:
            if time_limit is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= time_limit:
                    tqdm.write(f"\nTime limit of {time_limit} hours reached. Stopping analysis.")
                    break

            disp_path = doc_path if len(doc_path) <= 35 else "..." + doc_path[-32:]
            pbar.set_postfix_str(disp_path)

            if verbose:
                tqdm.write(f"Analyzing {doc_path}...")

            # Fetch only the first 5 chunks chronologically
            cursor.execute("""
                SELECT chunk_text FROM chunk_metadata
                WHERE doc_hash = ?
                ORDER BY seq_id ASC
                LIMIT 5
            """, (doc_hash,))
            chunks = cursor.fetchall()
            
            if not chunks:
                continue

            text_content = "\n\n".join([decompress_text(c[0]) for c in chunks])

            # Call LLM
            try:
                analysis_res = self.llm.analyze_document(doc_title, text_content, path=doc_path)
            except Exception as e:
                if verbose:
                    tqdm.write(f"Error analyzing {doc_path}: {e}")
                continue

            # Store in DB
            prompt_version = getattr(self.llm, "ANALYSIS_PROMPT_VERSION", "1.0")
            now = datetime.utcnow().isoformat() + "Z"
            cursor.execute("""
                INSERT INTO document_analysis (doc_hash, summary, authors, tags, dates, questions, doc_type, alt_title, prompt_version, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    summary = excluded.summary,
                    authors = excluded.authors,
                    tags = excluded.tags,
                    dates = excluded.dates,
                    questions = excluded.questions,
                    doc_type = excluded.doc_type,
                    alt_title = excluded.alt_title,
                    prompt_version = excluded.prompt_version,
                    analyzed_at = excluded.analyzed_at
            """, (
                doc_hash,
                analysis_res.get("summary", ""),
                json.dumps(analysis_res.get("authors", [])),
                json.dumps(analysis_res.get("tags", [])),
                json.dumps(analysis_res.get("dates", [])),
                json.dumps(analysis_res.get("questions", [])),
                analysis_res.get("doc_type", ""),
                analysis_res.get("altTitle", ""),
                prompt_version,
                now
            ))

            results.append({
                "path": doc_path,
                "collection": doc_coll,
                "hash": doc_hash,
                "analysis": analysis_res
            })
            
            # Commit after each document to save progress in case of interruption
            self.conn.commit()

        return results