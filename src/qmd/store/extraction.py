from typing import List, Optional, Dict, Any
from .models import Result

class ExtractionMixin:
    """Provides smart document extraction via representative sampling."""

    def extract_document(
        self,
        path: str,
        collection: Optional[str] = None,
        queries: Optional[List[str]] = None,
        max_chunks: int = 30,
        head_chunks: int = 3,
        tail_chunks: int = 3,
        top_k_per_query: int = 3,
        rerank: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts a representative sample of a document based on boundaries, headers, and semantic queries.
        """
        if queries is None:
            queries = []

        # 1. Fetch document outline to get bounds and headers
        outline = self.get_document_outline(collection, path)
        if not outline:
            return None

        total_chunks = outline.get("total_chunks", 0)
        resolved_coll = outline.get("collection", collection)
        resolved_path = outline.get("path", path)
        max_seq = total_chunks - 1

        if max_seq < 0:
            return None

        # Fast path: if the document is smaller than max_chunks, return it all
        if total_chunks <= max_chunks:
            chunks = self.get_chunks_by_seq_ids(resolved_coll, resolved_path, list(range(total_chunks)))
            
            # Optionally grab analysis metadata if present
            analysis_reports = self.get_analysis_report(collection=resolved_coll, path=resolved_path, limit=1)
            metadata = analysis_reports[0].get("analysis", {}) if analysis_reports else {}

            return {
                "metadata": metadata,
                "outline": outline,
                "chunks": chunks,
                "total_chunks": total_chunks,
                "selected_chunks": len(chunks)
            }

        # 2. Build priorities
        p1_seqs = set()
        p2_seqs = set()
        p3_seqs = set()

        # Priority 1: Head and Tail
        for i in range(min(head_chunks, total_chunks)):
            p1_seqs.add(i)
        
        tail_start = max(0, total_chunks - tail_chunks)
        for i in range(tail_start, total_chunks):
            p1_seqs.add(i)

        # Priority 2: Headers (levels 1, 2, 3)
        for h in outline.get("headings", []):
            if h.get("level", 1) <= 3:
                start_seq = h.get("start_seq", 0)
                p2_seqs.add(start_seq)
                if start_seq > 0:
                    p2_seqs.add(start_seq - 1)

        # Priority 3: Semantic Queries
        for query in queries:
            if not query.strip():
                continue
            # Use hybrid_search to support optional LLM reranking
            search_results = self.hybrid_search(
                query,
                limit=top_k_per_query,
                collection=resolved_coll,
                path=resolved_path,
                rerank=rerank
            )
            for res in search_results:
                if res.seq_id is not None:
                    p3_seqs.add(res.seq_id)

        # 3. Budgeting / Trimming
        selected_seqs = set()

        # Add P1 first
        for seq in p1_seqs:
            if len(selected_seqs) < max_chunks:
                selected_seqs.add(seq)

        # Add P2
        # Sort P2 to preserve chronological order if we have to truncate
        for seq in sorted(p2_seqs):
            if seq not in selected_seqs and len(selected_seqs) < max_chunks:
                selected_seqs.add(seq)

        # Add P3
        # Sort P3 to preserve chronological order
        for seq in sorted(p3_seqs):
            if seq not in selected_seqs and len(selected_seqs) < max_chunks:
                selected_seqs.add(seq)

        # 4. Fetch the chunks
        final_seqs = sorted(selected_seqs)
        chunks = self.get_chunks_by_seq_ids(resolved_coll, resolved_path, final_seqs)

        # 5. Get Metadata
        analysis_reports = self.get_analysis_report(collection=resolved_coll, path=resolved_path, limit=1)
        metadata = analysis_reports[0].get("analysis", {}) if analysis_reports else {}

        return {
            "metadata": metadata,
            "outline": outline,
            "chunks": chunks,
            "total_chunks": total_chunks,
            "selected_chunks": len(chunks)
        }