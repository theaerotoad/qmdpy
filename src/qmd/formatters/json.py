import json
import csv
import io
from typing import List, Dict, Optional, Tuple, Any, Union

def format_results_json(results: List, verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Outputs results as JSON for piping."""
    data = []
    for res in results:
        item = {
            "path": res.path,
            "title": res.title,
            "text": res.text,
            "score": res.score,
            "source": res.source,
            "collection": res.collection,
            "seq_id": res.seq_id,
            "headers": getattr(res, "headers", ""),
            "alt_title": getattr(res, "alt_title", ""),
            "doc_type": getattr(res, "doc_type", ""),
            "doc_date": getattr(res, "doc_date", ""),
            "authors": getattr(res, "authors", [])
        }
        if session_id:
            item["session_id"] = session_id
        if isinstance(exclusion_stats, dict):
            item["excluded_count"] = exclusion_stats.get("excluded_chunks", 0)
        if verbose or getattr(res, "fts_rank", None) is not None or getattr(res, "vec_rank", None) is not None:
            item["fts_score"] = getattr(res, "fts_score", None)
            item["fts_rank"] = getattr(res, "fts_rank", None)
            item["vec_score"] = getattr(res, "vec_score", None)
            item["vec_rank"] = getattr(res, "vec_rank", None)
            item["rrf_score"] = getattr(res, "rrf_score", None)
            item["rrf_rank"] = getattr(res, "rrf_rank", None)
        data.append(item)
    print(json.dumps(data, indent=2))

def format_doc_results_json(grouped_results: List[Dict], session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Outputs document-grouped results as JSON for piping."""
    if session_id or isinstance(exclusion_stats, dict):
        for doc in grouped_results:
            if session_id:
                doc["session_id"] = session_id
            if isinstance(exclusion_stats, dict):
                doc["excluded_count"] = exclusion_stats.get("excluded_chunks", 0)
    print(json.dumps(grouped_results, indent=2))

def format_discover_json(results: List, verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Outputs discovered document results as JSON for piping."""
    data = []
    for res in results:
        item = {
            "path": res.path,
            "title": res.title,
            "text": res.text,
            "score": res.score,
            "source": res.source,
            "collection": res.collection,
            "seq_id": res.seq_id,
            "headers": getattr(res, "headers", ""),
            "alt_title": getattr(res, "alt_title", ""),
            "doc_type": getattr(res, "doc_type", ""),
            "doc_date": getattr(res, "doc_date", ""),
            "authors": getattr(res, "authors", []),
            "match_count": getattr(res, "match_count", 1)
        }
        if session_id:
            item["session_id"] = session_id
        if isinstance(exclusion_stats, dict):
            item["excluded_count"] = exclusion_stats.get("excluded_chunks", 0)
        if verbose or getattr(res, "fts_rank", None) is not None or getattr(res, "vec_rank", None) is not None:
            item["fts_score"] = getattr(res, "fts_score", None)
            item["fts_rank"] = getattr(res, "fts_rank", None)
            item["vec_score"] = getattr(res, "vec_score", None)
            item["vec_rank"] = getattr(res, "vec_rank", None)
            item["rrf_score"] = getattr(res, "rrf_score", None)
            item["rrf_rank"] = getattr(res, "rrf_rank", None)
        data.append(item)
    print(json.dumps(data, indent=2))

def format_report_json(results: List[Dict[str, Any]]):
    """Outputs document analysis reports as JSON."""
    print(json.dumps(results, indent=2))

def format_report_csv(results: List[Dict[str, Any]], print_output: bool = True) -> str:
    """Outputs flattened reports to standard CSV, flattening lists with semicolons."""
    if not results:
        return ""
    
    flat_results = []
    all_keys = set()
    
    for res in results:
        flat = {}
        # Merge core fields and analysis subset
        for k, v in res.items():
            if k == "analysis" and isinstance(v, dict):
                for ak, av in v.items():
                    if isinstance(av, list):
                        flat[ak] = "; ".join(str(i) for i in av)
                    else:
                        flat[ak] = str(av)
                continue
            
            if isinstance(v, list):
                flat[k] = "; ".join(str(i) for i in v)
            else:
                flat[k] = str(v) if v is not None else ""
        
        flat_results.append(flat)
        all_keys.update(flat.keys())
    
    headers = sorted(list(all_keys))
    # Prioritize standard identifier columns
    for priority_col in reversed(["collection", "path", "title", "hash"]):
        if priority_col in headers:
            headers.insert(0, headers.pop(headers.index(priority_col)))
            
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for r in flat_results:
        writer.writerow(r)
        
    csv_str = output.getvalue()
    if print_output:
        print(csv_str, end="")
    return csv_str

def format_collection_tree_json(tree_data: Union[Dict, List[Dict]]):
    """Outputs collection folder directory tree as JSON."""
    print(json.dumps(tree_data, indent=2))