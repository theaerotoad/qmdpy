import sys
import json
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

try:
    import dplib
except ImportError:
    dplib = None


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def extract_document_date(file_path: Union[str, Path], markdown_body: str = "") -> Optional[str]:
    """
    Extracts or infers a document date using dplib from file path/filename and front matter/early content.
    """
    store_module = sys.modules.get("qmd.store")
    if store_module and getattr(store_module, "extract_document_date", None) is not extract_document_date:
        return store_module.extract_document_date(file_path, markdown_body)

    active_dplib = getattr(sys.modules.get("qmd.store"), "dplib", dplib)
    if active_dplib is None:
        return None

    path_str = str(file_path)
    sample_content = markdown_body[:4000] if markdown_body else ""

    try:
        if hasattr(active_dplib, "extract_date"):
            try:
                res = active_dplib.extract_date(path=path_str, content=sample_content)
            except TypeError:
                res = active_dplib.extract_date(path_str, sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(active_dplib, "parse_document_date"):
            try:
                res = active_dplib.parse_document_date(path=path_str, content=sample_content)
            except TypeError:
                res = active_dplib.parse_document_date(path_str, sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(active_dplib, "parse_date"):
            try:
                res = active_dplib.parse_date(path=path_str, text=sample_content)
            except TypeError:
                try:
                    res = active_dplib.parse_date(path_str, sample_content)
                except TypeError:
                    res = active_dplib.parse_date(path_str) or active_dplib.parse_date(sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(active_dplib, "parse_frontmatter") and hasattr(active_dplib, "parse_path"):
            res = active_dplib.parse_frontmatter(sample_content) or active_dplib.parse_path(path_str)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
        elif hasattr(active_dplib, "parse"):
            try:
                res = active_dplib.parse(path_str, sample_content)
            except TypeError:
                res = active_dplib.parse(path_str) or active_dplib.parse(sample_content)
            if res:
                return res.isoformat() if hasattr(res, "isoformat") else str(res)
    except Exception:
        pass

    return None


def _build_collection_sql_filter(column_name: str, collection: Optional[Union[str, List[str]]]) -> Tuple[str, List[str]]:
    """
    Constructs SQL WHERE clause and parameter list for collection filtering.
    Supports exact match, substring match, wildcards (*, ?), and comma-separated lists.
    """
    if not collection:
        return "", []

    tokens = []
    if isinstance(collection, str):
        if collection.strip():
            tokens = [c.strip() for c in collection.split(',') if c.strip()]
    elif isinstance(collection, (list, tuple, set)):
        tokens = [str(c).strip() for c in collection if str(c).strip()]

    if not tokens:
        return "", []

    clauses = []
    params = []
    for tok in tokens:
        if '*' in tok or '?' in tok:
            clauses.append(f"{column_name} LIKE ?")
            params.append(tok.replace('*', '%').replace('?', '_'))
        else:
            clauses.append(f"{column_name} LIKE ?")
            params.append(f"%{tok}%")

    sql = f" AND ({' OR '.join(clauses)})" if clauses else ""
    return sql, params


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
    doc_date: Optional[str] = None
    fts_score: Optional[float] = None
    fts_rank: Optional[int] = None
    vec_score: Optional[float] = None
    vec_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    match_count: int = 1
    rowid: Optional[int] = None


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
            "doc_date": getattr(r, "doc_date", None),
            "fts_score": getattr(r, "fts_score", None),
            "fts_rank": getattr(r, "fts_rank", None),
            "vec_score": getattr(r, "vec_score", None),
            "vec_rank": getattr(r, "vec_rank", None),
            "rrf_score": getattr(r, "rrf_score", None),
            "rrf_rank": getattr(r, "rrf_rank", None),
            "match_count": getattr(r, "match_count", 1),
            "rowid": getattr(r, "rowid", None),
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
            doc_date=item.get("doc_date"),
            fts_score=item.get("fts_score"),
            fts_rank=item.get("fts_rank"),
            vec_score=item.get("vec_score"),
            vec_rank=item.get("vec_rank"),
            rrf_score=item.get("rrf_score"),
            rrf_rank=item.get("rrf_rank"),
            match_count=item.get("match_count", 1),
            rowid=item.get("rowid"),
        ))
    return results