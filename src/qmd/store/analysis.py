import time
import json
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

from tqdm import tqdm

from qmd.utils import decompress_text
from .models import _build_collection_sql_filter

class AnalysisMixin:
    """Handles LLM-based metadata extraction and summarization on document subsets."""

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