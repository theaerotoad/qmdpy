import sys
import subprocess
import os
import secrets
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

from qmd.config import load_config
from qmd.store import Store, Result
from qmd.db import get_seen_chunks_for_session, record_session_event, record_session_results, get_db_meta
from qmd.formatters.cli import (
    format_results_cli, format_doc_results_cli, format_discover_cli,
    format_outline_cli, format_chunks_cli, format_collection_tree_cli,
    format_map_cli, format_report_cli
)
from qmd.formatters.json import (
    format_results_json, format_doc_results_json, format_discover_json,
    format_report_json, format_report_csv
)
from qmd.formatters.xml import (
    format_discover_xml, format_results_xml, format_doc_results_xml,
    format_chunks_xml, format_outline_xml, format_collection_tree_xml,
    format_map_xml, format_report_xml
)
from qmd.formatters.core import set_plain_mode, strip_ansi
from qmd.formatters.colors import Colors
from qmd.utils import redact_pii, parse_target_spec, parse_int_ranges

def _clean_header_part(part: str) -> str:
    """Strips leading/trailing markdown hashes, whitespace, and formatting."""
    if not part:
        return ""
    return re.sub(r'^\s*#+\s*', '', part).strip()

def _normalize_str(s: str) -> str:
    """Normalizes string for fuzzy comparison (lowercase, alphanumeric only)."""
    return re.sub(r'[^\w]', '', s.lower())

def _is_xml_output(args) -> bool:
    """Determines whether XML output is requested via flags or QMD_XML environment variable."""
    env_xml = os.environ.get("QMD_XML", "").strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(args, "xml", False) or getattr(args, "llm", False) or env_xml)

def merge_overlapping_snippets(snippets: List[Tuple], doc_title: str = "") -> List[str]:
    """
    Takes a list of (seq_id, text) or (seq_id, text, headers). 
    Sorts non-negative seq_ids chronologically; treats -1 (FTS fallback) gracefully.
    Merges texts if they have significant overlap or containment.
    Inserts markdown section headings when a snippet introduces a new section header,
    suppressing redundant or duplicate headers already shown in document view.
    Includes explicit break markers indicating skipped chunk counts between non-contiguous blocks.
    """
    if not snippets:
        return []

    normalized = []
    for item in snippets:
        if len(item) == 3:
            seq_id, text, headers = item
        elif len(item) == 2:
            seq_id, text = item
            headers = ""
        else:
            continue
        if text and text.strip():
            normalized.append((seq_id, text.strip(), headers or ""))

    if not normalized:
        return []

    chunk_snips = sorted([s for s in normalized if s[0] >= 0], key=lambda x: int(x[0]))
    fts_snips = [s for s in normalized if s[0] < 0]

    # Deduplicate chunk_snips with identical seq_id
    unique_chunk_snips = []
    seen_seqs = set()
    for seq_id, text, headers in chunk_snips:
        if seq_id not in seen_seqs:
            unique_chunk_snips.append((seq_id, text, headers))
            seen_seqs.add(seq_id)

    sorted_snips = unique_chunk_snips if unique_chunk_snips else fts_snips

    doc_title_norm = _normalize_str(doc_title)
    
    merged_blocks: List[Tuple[int, int, str, str]] = []  # List of (start_seq, end_seq, headers, text)

    for seq_id, next_text, next_headers in sorted_snips:
        if not merged_blocks:
            merged_blocks.append((seq_id, seq_id, next_headers, next_text))
            continue

        if any(next_text in text for _, _, _, text in merged_blocks):
            continue

        contained_indices = [idx for idx, (_, _, _, text) in enumerate(merged_blocks) if text in next_text]
        if contained_indices:
            first_idx = contained_indices[0]
            curr_start = min([merged_blocks[i][0] for i in contained_indices] + [seq_id])
            curr_end = max([merged_blocks[i][1] for i in contained_indices] + [seq_id])
            curr_hdr = merged_blocks[first_idx][2] or next_headers
            merged_blocks[first_idx] = (curr_start, curr_end, curr_hdr, next_text)
            for idx in reversed(contained_indices[1:]):
                merged_blocks.pop(idx)
            continue

        last_start, last_end, last_hdr, last_text = merged_blocks[-1]
        curr_strip = last_text.rstrip()
        next_strip = next_text.lstrip()

        overlap_found = False
        max_k = min(len(curr_strip), len(next_strip))
        for k in range(min(max_k, 250), 19, -1):
            if curr_strip.endswith(next_strip[:k]):
                new_end = max(last_end, seq_id)
                merged_blocks[-1] = (last_start, new_end, last_hdr, curr_strip[:-k] + next_strip)
                overlap_found = True
                break

        if not overlap_found:
            merged_blocks.append((seq_id, seq_id, next_headers, next_text))

    if unique_chunk_snips and fts_snips:
        for fts_seq, fts_text, fts_headers in fts_snips:
            if not any(fts_text in text or text in fts_text for _, _, _, text in merged_blocks):
                merged_blocks.append((fts_seq, fts_seq, fts_headers, fts_text))

    # Format merged blocks into strings with explicit chunk break indicators
    formatted_snippets: List[str] = []
    shown_header_parts: set = set()

    if doc_title_norm:
        shown_header_parts.add(doc_title_norm)

    # Indicate skipped chunks if the first block starts after seq_id 0
    if unique_chunk_snips and merged_blocks and merged_blocks[0][0] > 0:
        first_start = merged_blocks[0][0]
        skip_msg = f"(... {first_start} chunk{'s' if first_start > 1 else ''} skipped ...)"
        formatted_snippets.append(skip_msg)

    for idx, (start_seq, end_seq, raw_headers, text) in enumerate(merged_blocks):
        if idx > 0:
            prev_end_seq = merged_blocks[idx - 1][1]
            if start_seq >= 0 and prev_end_seq >= 0:
                gap = start_seq - prev_end_seq - 1
                if gap > 0:
                    skip_msg = f"(... {gap} chunk{'s' if gap > 1 else ''} skipped ...)"
                    formatted_snippets.append(skip_msg)
            else:
                skip_msg = "(... chunks skipped ...)"
                formatted_snippets.append(skip_msg)

        # Split headers by ' > ' and clean each level
        raw_parts = [p.strip() for p in raw_headers.split('>') if p.strip()] if raw_headers else []
        clean_parts = [_clean_header_part(p) for p in raw_parts if _clean_header_part(p)]

        # Filter out parts matching document title or already shown in document view
        new_parts = []
        for part in clean_parts:
            part_norm = _normalize_str(part)
            if not part_norm or part_norm == doc_title_norm:
                continue
            if part_norm not in shown_header_parts:
                new_parts.append(part)

        # Check if text already starts with a header matching any element of new_parts
        first_line = text.lstrip().split('\n', 1)[0].strip()
        first_line_clean = _clean_header_part(first_line)
        first_line_norm = _normalize_str(first_line_clean)

        if new_parts and _normalize_str(new_parts[-1]) == first_line_norm:
            shown_header_parts.add(_normalize_str(new_parts[-1]))
            new_parts.pop()

        for p in new_parts:
            shown_header_parts.add(_normalize_str(p))

        if new_parts:
            header_line = " > ".join(new_parts)
            formatted = f"## {header_line}\n\n{text}"
        else:
            formatted = text

        formatted_snippets.append(formatted)

    return formatted_snippets

def group_results_by_doc(results: List[Result]) -> List[Dict]:
    """
    Groups chunks by title (to merge duplicate files), keeps max score, and merges text into reading order.
    """
    docs = {}
    for r in results:
        # Group by title to seamlessly merge identical books across different collections/paths
        key = r.title.strip().lower() if r.title else (r.collection, r.path)
        if key not in docs:
            docs[key] = {
                "title": r.title,
                "collection": r.collection,
                "path": r.path,
                "score": r.score,
                "raw_snippets": [],
                "chunks": []
            }
        
        if r.score > docs[key]["score"]:
            docs[key]["score"] = r.score
            
        if not any(x[1] == r.text for x in docs[key]["raw_snippets"]):
            docs[key]["raw_snippets"].append((r.seq_id, r.text, getattr(r, 'headers', '')))
            docs[key]["chunks"].append({
                "seq_id": r.seq_id,
                "score": r.score,
                "rank": r.rank,
                "fts_score": getattr(r, "fts_score", None),
                "fts_rank": getattr(r, "fts_rank", None),
                "vec_score": getattr(r, "vec_score", None),
                "vec_rank": getattr(r, "vec_rank", None),
                "rrf_score": getattr(r, "rrf_score", None),
                "rrf_rank": getattr(r, "rrf_rank", None),
                "source": r.source,
                "headers": getattr(r, "headers", ""),
                "doc_date": getattr(r, "doc_date", None),
                "alt_title": getattr(r, "alt_title", None),
                "doc_type": getattr(r, "doc_type", None),
                "authors": getattr(r, "authors", None),
                "text": r.text
            })
            
    output_list = []
    for d in docs.values():
        d["snippets"] = merge_overlapping_snippets(d["raw_snippets"], doc_title=d["title"])
        del d["raw_snippets"]
        output_list.append(d)
        
    return sorted(output_list, key=lambda x: x['score'], reverse=True)

def handle_discover(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    env_deep = os.environ.get("QMD_DEEP", "").strip().lower() in ("1", "true", "yes", "on")
    is_deep = getattr(args, "deep", False) or env_deep
    rerank = getattr(args, "rerank", False) or is_deep
    is_w2n = getattr(args, "w2n", False) or getattr(args, "broad", False)

    query = " ".join(args.query)
    from qmd.utils import parse_query_directives
    query, directives = parse_query_directives(query)

    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 10)
    if "limit" in directives:
        limit = directives["limit"]

    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)

    session_id = getattr(args, "session", None)
    has_explicit_session = session_id is not None
    if not session_id:
        session_id = secrets.token_hex(4)

    include_seen = getattr(args, "include_seen", False)
    if "exclude_seen" in directives:
        exclude_seen = directives["exclude_seen"]
    elif has_explicit_session:
        exclude_seen = not include_seen
    else:
        exclude_seen = getattr(args, "exclude_seen", False) and not include_seen

    if "rerank" in directives:
        rerank = directives["rerank"]

    exclude_seen_set = set()
    seen_chunks_count = 0
    if store.history_conn:
        all_seen = get_seen_chunks_for_session(store.history_conn, session_id)
        seen_chunks_count = len(all_seen)
        if exclude_seen:
            exclude_seen_set = all_seen

    search_kwargs = {
        "query": query,
        "limit": limit,
        "verbose": args.verbose,
        "rerank": rerank,
        "reranker_only": getattr(args, "rerank_only", False),
        "collection": directives.get("collection", args.collection),
        "lexical_query": directives.get("lex", args.lex),
        "title": directives.get("title", args.title),
        "path": directives.get("path", args.path),
        "fts_limit": fts_limit,
        "vec_limit": vec_limit,
        "rerank_candidates": rerank_candidates,
        "exclude_seen_set": exclude_seen_set,
        "w2n": is_w2n,
        "dirlist": directives.get("dirlist", getattr(args, "dirlist", False)),
    }
    if getattr(args, "no_cache", False):
        search_kwargs["use_cache"] = False

    results = store.discover(**search_kwargs)

    redact_pii_flag = directives.get("redact_pii", getattr(args, "redact_pii", False))
    if redact_pii_flag:
        for r in results:
            r.text = redact_pii(r.text)
            if hasattr(r, "title") and r.title:
                r.title = redact_pii(r.title)

    event_type = "discover"
    db_last_updated = get_db_meta(store.conn, "last_updated")
    event_id = record_session_event(
        store.history_conn,
        session_id,
        event_type,
        query,
        directives.get("lex", getattr(args, "lex", None)),
        str(store.config.db_path),
        db_last_updated
    )

    max_chunks = getattr(args, "max_chunks", None)
    if max_chunks is None:
        cfg_val = getattr(getattr(store, "config", None), "max_chunks_per_response", 30)
        max_chunks = cfg_val if isinstance(cfg_val, int) else 30

    truncation_info = None
    if isinstance(max_chunks, int) and max_chunks > 0 and len(results) > max_chunks:
        omitted = len(results) - max_chunks
        truncation_info = {
            "omitted_remaining": omitted,
            "limit": max_chunks
        }
        results = results[:max_chunks]

    record_session_results(store.history_conn, session_id, event_id, results)

    if args.json:
        format_discover_json(results, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)
    elif is_xml:
        format_discover_xml(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, seen_chunks=seen_chunks_count, truncation_info=truncation_info)
    else:
        format_discover_cli(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)

def handle_map(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    env_deep = os.environ.get("QMD_DEEP", "").strip().lower() in ("1", "true", "yes", "on")
    is_deep = getattr(args, "deep", False) or env_deep
    rerank = getattr(args, "rerank", False) or is_deep

    query = " ".join(args.query)
    from qmd.utils import parse_query_directives
    query, directives = parse_query_directives(query)

    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 100)
    if "limit" in directives:
        limit = directives["limit"]

    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)

    search_kwargs = {
        "query": query,
        "limit": limit,
        "verbose": args.verbose,
        "rerank": rerank,
        "reranker_only": getattr(args, "rerank_only", False),
        "collection": directives.get("collection", args.collection),
        "lexical_query": directives.get("lex", args.lex),
        "title": directives.get("title", args.title),
        "path": directives.get("path", args.path),
        "fts_limit": fts_limit,
        "vec_limit": vec_limit,
        "rerank_candidates": rerank_candidates,
        "exclude_seen_set": set(),
        "dirlist": directives.get("dirlist", getattr(args, "dirlist", False)),
    }
    if getattr(args, "no_cache", False):
        search_kwargs["use_cache"] = False

    tree_results = store.map_search(**search_kwargs)

    if args.json:
        import json
        print(json.dumps(tree_results, indent=2))
    elif is_xml:
        format_map_xml(tree_results, query=query)
    else:
        format_map_cli(tree_results, query=query)

def handle_search(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    env_deep = os.environ.get("QMD_DEEP", "").strip().lower() in ("1", "true", "yes", "on")
    is_deep = getattr(args, "deep", False) or env_deep
    rerank = getattr(args, "rerank", False) or is_deep
    is_flat = getattr(args, "flat", False) or getattr(args, "chunks", False)
    is_doc = not is_flat
    is_w2n = getattr(args, "w2n", False) or getattr(args, "broad", False)

    query = " ".join(args.query)
    from qmd.utils import parse_query_directives
    query, directives = parse_query_directives(query)

    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 10)
    if "limit" in directives:
        limit = directives["limit"]

    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)
    
    session_id = getattr(args, "session", None)
    has_explicit_session = session_id is not None
    if not session_id:
        session_id = secrets.token_hex(4)

    include_seen = getattr(args, "include_seen", False)
    if "exclude_seen" in directives:
        exclude_seen = directives["exclude_seen"]
    elif has_explicit_session:
        exclude_seen = not include_seen
    else:
        exclude_seen = getattr(args, "exclude_seen", False) and not include_seen

    if "rerank" in directives:
        rerank = directives["rerank"]

    exclude_seen_set = set()
    seen_chunks_count = 0
    if store.history_conn:
        all_seen = get_seen_chunks_for_session(store.history_conn, session_id)
        seen_chunks_count = len(all_seen)
        if exclude_seen:
            exclude_seen_set = all_seen

    search_kwargs = {
        "limit": limit,
        "verbose": args.verbose,
        "rerank": rerank,
        "reranker_only": getattr(args, "rerank_only", False),
        "collection": directives.get("collection", args.collection),
        "lexical_query": directives.get("lex", args.lex),
        "title": directives.get("title", args.title),
        "path": directives.get("path", args.path),
        "fts_limit": fts_limit,
        "vec_limit": vec_limit,
        "rerank_candidates": rerank_candidates,
        "exclude_seen_set": exclude_seen_set,
        "dirlist": directives.get("dirlist", getattr(args, "dirlist", False)),
    }
    if getattr(args, "no_cache", False):
        search_kwargs["use_cache"] = False

    if is_w2n:
        results = store.wide_to_narrow_search(
            query,
            **search_kwargs
        )
    else:
        results = store.hybrid_search(
            query, 
            **search_kwargs
        )
    
    redact_pii_flag = directives.get("redact_pii", getattr(args, "redact_pii", False))
    if redact_pii_flag:
        for r in results:
            r.text = redact_pii(r.text)
            if hasattr(r, "title") and r.title:
                r.title = redact_pii(r.title)

    event_type = "doc_view" if is_doc else "search"
    db_last_updated = get_db_meta(store.conn, "last_updated")
    event_id = record_session_event(
        store.history_conn,
        session_id,
        event_type,
        query,
        directives.get("lex", getattr(args, "lex", None)),
        str(store.config.db_path),
        db_last_updated
    )

    # Determine max_chunks cap
    max_chunks = getattr(args, "max_chunks", None)
    if max_chunks is None:
        cfg_val = getattr(getattr(store, "config", None), "max_chunks_per_response", 30)
        max_chunks = cfg_val if isinstance(cfg_val, int) else 30

    # Truncate flat results globally to strictly preserve ranking order
    truncation_info = None
    if isinstance(max_chunks, int) and max_chunks > 0 and len(results) > max_chunks:
        omitted = len(results) - max_chunks
        truncation_info = {
            "omitted_remaining": omitted,
            "limit": max_chunks
        }
        results = results[:max_chunks]

    if is_doc:
        grouped = group_results_by_doc(results)
        grouped = grouped[:limit]

        # Record session results at chunk level
        shown_chunks = []
        for doc in grouped:
            for c in doc.get("chunks", []):
                shown_chunks.append({
                    "collection": doc.get("collection", ""),
                    "path": doc.get("path", ""),
                    "seq_id": c.get("seq_id", 0),
                    "rank": c.get("rank", 0),
                    "score": c.get("score", 0.0)
                })
        record_session_results(store.history_conn, session_id, event_id, shown_chunks)

        if args.json:
            format_doc_results_json(grouped, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)
        elif is_xml:
            format_doc_results_xml(grouped, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, seen_chunks=seen_chunks_count, truncation_info=truncation_info)
        else:
            format_doc_results_cli(grouped, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)
    else:
        record_session_results(store.history_conn, session_id, event_id, results)
        if args.json:
            format_results_json(results, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)
        elif is_xml:
            format_results_xml(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, seen_chunks=seen_chunks_count, truncation_info=truncation_info)
        else:
            format_results_cli(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)

def handle_extract(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    env_deep = os.environ.get("QMD_DEEP", "").strip().lower() in ("1", "true", "yes", "on")
    is_deep = getattr(args, "deep", False) or env_deep
    rerank = getattr(args, "rerank", False) or is_deep

    spec = parse_target_spec(args.path, default_collection=getattr(args, "collection", None))
    coll = spec["collection"] or getattr(args, "collection", None)
    target_path = spec["path"] if spec["path"] is not None else args.path

    try:
        data = store.extract_document(
            path=target_path,
            collection=coll,
            queries=args.queries,
            max_chunks=args.max_chunks,
            head_chunks=args.head,
            tail_chunks=args.tail,
            top_k_per_query=args.top_k,
            rerank=rerank
        )
    except ValueError as e:
        if args.json:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

    if not data:
        if args.json:
            import json
            print(json.dumps({"error": "Document not found"}, indent=2))
        else:
            print(f"{Colors.RED}Error: Document not found matching path '{target_path}'{Colors.RESET}")
        sys.exit(1)

    if args.json:
        import json
        
        # Convert Result objects in chunks to dicts for JSON
        if "chunks" in data:
            from qmd.store.models import _results_to_json
            data["chunks"] = json.loads(_results_to_json(data["chunks"]))
        
        print(json.dumps(data, indent=2))
    elif is_xml:
        from qmd.formatters.xml import format_extraction_xml
        format_extraction_xml(data)
    else:
        from qmd.formatters.cli import format_extraction_cli
        format_extraction_cli(data)

def handle_outline(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    spec = parse_target_spec(args.path, default_collection=getattr(args, "collection", None))
    coll = spec["collection"] or getattr(args, "collection", None)
    target_path = spec["path"] if spec["path"] is not None else args.path

    depth = getattr(args, "depth", None)
    max_depth = depth if isinstance(depth, int) else None

    pattern = getattr(args, "pattern", None)
    pattern_val = pattern if isinstance(pattern, str) else None

    outline = store.get_document_outline(
        collection=coll,
        path=target_path,
        max_depth=max_depth,
        pattern=pattern_val
    )
    if not outline:
        if args.json:
            import json
            print(json.dumps({"error": "Document not found"}, indent=2))
        else:
            print(f"{Colors.RED}Error: Document not found matching path '{target_path}'{Colors.RESET}")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(outline, indent=2))
    elif is_xml:
        format_outline_xml(outline)
    else:
        format_outline_cli(outline)

def handle_chunk(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    target = args.target
    results = []

    spec = parse_target_spec(target, default_collection=getattr(args, "collection", None))
    coll = spec["collection"] or getattr(args, "collection", None)

    cli_seq = getattr(args, "seq", None)
    if cli_seq is not None:
        seq_ids = parse_int_ranges(cli_seq)
        if seq_ids is None:
            if args.json:
                import json
                print(json.dumps({"error": f"Invalid sequence range specification: '{args.seq}'"}, indent=2))
            else:
                print(f"{Colors.RED}Error: Invalid sequence range specification: '{args.seq}'{Colors.RESET}")
            sys.exit(1)
    else:
        seq_ids = spec["seq"]

    if spec["row_ids"] is not None and cli_seq is None and spec["path"] is None:
        results = store.get_chunk_by_id(spec["row_ids"], window=args.window)
    elif spec["path"] is not None:
        target_seq = seq_ids if seq_ids is not None else 0
        results = store.get_chunk_by_seq(coll, spec["path"], seq_id=target_seq, window=args.window)
    elif spec["row_ids"] is not None:
        results = store.get_chunk_by_id(spec["row_ids"], window=args.window)
    else:
        results = store.get_chunk_by_seq(coll, target, seq_id=0, window=args.window)

    if not results:
        if args.json:
            import json
            print(json.dumps({"error": "Chunk not found"}, indent=2))
        else:
            print(f"{Colors.RED}Error: Chunk not found.{Colors.RESET}")
        sys.exit(1)

    # Apply max_chunks_per_response cap to prevent context explosion
    max_chunks = getattr(args, "max_chunks", None)
    if max_chunks is None:
        cfg_val = getattr(getattr(store, "config", None), "max_chunks_per_response", 30)
        max_chunks = cfg_val if isinstance(cfg_val, int) else 30

    truncation_info = None
    if isinstance(max_chunks, int) and max_chunks > 0 and len(results) > max_chunks:
        omitted_remaining = len(results) - max_chunks
        last_rendered = results[max_chunks - 1]
        next_chunk = results[max_chunks]
        next_start_seq = getattr(next_chunk, "seq_id", 0)
        
        # Calculate next page range
        next_slice_len = min(omitted_remaining, max_chunks)
        next_end_seq = next_start_seq + next_slice_len - 1
        seq_range = f"{next_start_seq}-{next_end_seq}" if next_start_seq != next_end_seq else f"{next_start_seq}"
        
        target_path = spec["path"] or getattr(next_chunk, "path", "")
        coll_name = coll or getattr(next_chunk, "collection", "")
        resume_ref = f"{coll_name}:{target_path}:{seq_range}" if coll_name else f"{target_path}:{seq_range}"
        resume_cmd = f"qmd read '{resume_ref}'"

        truncation_info = {
            "omitted_remaining": omitted_remaining,
            "limit": max_chunks,
            "resume_cmd": resume_cmd
        }
        results = results[:max_chunks]

    if args.json:
        format_results_json(results)
    elif is_xml:
        format_chunks_xml(results, window=args.window, truncation_info=truncation_info)
    else:
        format_chunks_cli(results, window=args.window, truncation_info=truncation_info)

def handle_collection_tree(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    coll_name = getattr(args, "name", None) or getattr(args, "collection", None)
    pattern = getattr(args, "pattern", None)
    is_regex = getattr(args, "regex", False)
    case_sensitive = getattr(args, "case_sensitive", False) and not getattr(args, "ignore_case", False)

    try:
        tree_data = store.get_collection_tree(
            collection=coll_name,
            max_depth=getattr(args, "depth", None),
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive
        )
    except ValueError as e:
        if args.json:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

    if coll_name and tree_data is None:
        if args.json:
            import json
            print(json.dumps({"error": f"Collection '{coll_name}' not found"}, indent=2))
        else:
            print(f"{Colors.RED}Error: Collection '{coll_name}' not found.{Colors.RESET}")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(tree_data, indent=2))
    elif is_xml:
        format_collection_tree_xml(tree_data)
    else:
        format_collection_tree_cli(tree_data)

def handle_guide(args, store: Optional[Store] = None):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    if is_xml:
        guide_xml = """<qmd_guide>
  <overview>QMD is a local search, retrieval, and document inspection engine designed for LLM agents. Formulate queries as natural language questions rather than keyword searches.</overview>
  <batch_execution_format>
When requested to perform research or retrieve data using QMD, emit 1 to 5 commands wrapped in a <qmd_commands> XML container. Frame queries as clear natural language questions:

<qmd_commands>
  qmd discover "what are the main principles of orbital mechanics?"
  qmd search "how do orbital transfer maneuvers work?"
  qmd outline "coll:path.md"
  qmd read "coll:path.md:10-15"
</qmd_commands>
  </batch_execution_format>
  <workflow>
    <step num="1" name="Discovery">Use `qmd discover "natural language question?"` for top-level SERP results (1 hit per document), or `qmd map "question?"` to view a semantic density tree of results across folders.</step>
    <step num="2" name="Search">Use `qmd search "natural language question?"` for comprehensive document-grouped results.</step>
    <step num="3" name="Orient">Use `qmd outline "<target>"` to inspect heading hierarchies and chunk sequence spans.</step>
    <step num="4" name="Read">Use `qmd read "<target>"` to fetch chunks/ranges, or execute exact `read="..."` / `expand="..."` attributes from search results.</step>
  </workflow>
  <shorthand_targets>
    <target syntax="coll:path:seq_range">e.g. `Books:history.epub:10-15` or `qmd://Books/history.epub:10-15`</target>
    <target syntax="coll:path">e.g. `Books:history.epub` or `qmd://Books/history.epub`</target>
    <target syntax="path:seq_range">e.g. `notes.md:0-3`</target>
    <target syntax="row_ids">e.g. `10-15` or `22,40,25-27`</target>
  </shorthand_targets>
  <agent_tips>
    <tip>Natural language questions: formulate search queries as natural language questions rather than keyword lists for optimal semantic retrieval.</tip>
    <tip>Triage first: `qmd discover "question?"` gives a fast 1-hit-per-document overview before deep reading.</tip>
    <tip>Batch execution: emit 1 to 5 sequential commands in a <qmd_commands> block for unified batch processing.</tip>
    <tip>Agent hypermedia: copy-paste the `read="..."`, `outline="..."`, and `search="..."` attributes directly into CLI calls.</tip>
    <tip>Safety cap: large reads truncate at max_chunks (default 30) and provide a `resume="..."` command for the next slice.</tip>
  </agent_tips>
</qmd_guide>"""
        print(guide_xml)
        return guide_xml
    else:
        guide_md = f"""{Colors.BOLD}QMD LLM Agent Research & Inspection Guide{Colors.RESET}

{Colors.CYAN}## Workflow & Decision Matrix{Colors.RESET}
1. {Colors.BOLD}Triage & Discovery:{Colors.RESET}
   - `qmd discover "question?"` - Single top hit per document SERP view (use natural language questions)
   - `qmd map "question?"` - Semantic directory tree mapping (density clustering)
   - `qmd collections` - List indexed collections and paths
   - `qmd tree [collection] [-p pattern]` - Explore document directory trees

2. {Colors.BOLD}Retrieval & Search:{Colors.RESET}
   - `qmd search "question?"` - Document-ordered search results (natural language question)
   - `qmd search "question?" --deep` - Deep search: doc-grouped + LLM reranked
   - `qmd search "question?" --session <id>` - Session search (auto-deduplicates seen chunks)
   - `qmd search "question?" --broad` - Broad hierarchical search (wide-to-narrow)

3. {Colors.BOLD}Document Structure & Orientation:{Colors.RESET}
   - `qmd outline "<target>"` - Table of contents with chunk sequence mappings

4. {Colors.BOLD}Targeted Reading:{Colors.RESET}
   - `qmd read "<target>"` - Read specific chunks, ranges, or surrounding context
   - Examples: `qmd read 'Books:doc.epub:10-15'`, `qmd read 'qmd://Books/doc.epub:3'`

{Colors.CYAN}## Target Shorthand Syntax{Colors.RESET}
- Full URI: `qmd://<collection>/<path>[:<seq>]`
- Shorthand: `<collection>:<path>[:<seq>]`
- Relative: `<path>[:<seq>]`
- Chunk Row IDs: `10-15` or `22,40,25-27`

{Colors.CYAN}## Query Formulation & Agent Best Practices{Colors.RESET}
- Frame queries as clear natural language questions rather than keyword lists for optimal semantic retrieval.
- Use `qmd discover` to scan across multiple documents without flooding context.
- Pass `--session <id>` when conducting multi-turn research to avoid ingesting duplicate context.
- Follow actionable XML attributes (`read="..."`, `outline="..."`, `search="..."`, `resume="..."`).
- Large reads truncate at 30 chunks with a resume hint to protect context windows."""
        if getattr(args, "plain", False):
            guide_md = strip_ansi(guide_md)
        res = guide_md.strip()
        print(res)
        return res

def handle_collections_list(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)
    if getattr(args, "json", False):
        import json
        data = {name: {"path": cfg.path, "glob": cfg.glob} for name, cfg in store.config.collections.items()}
        print(json.dumps(data, indent=2))
    else:
        for name, cfg in store.config.collections.items():
            print(f"{Colors.GREEN}{name}{Colors.RESET}: {cfg.path} ({cfg.glob})")

def handle_update(args, store: Store):
    if getattr(args, "verbose", False):
        os.environ["QMD_VERBOSE"] = "1"

    config = store.config
    if getattr(config, "is_federated", False):
        print(f"{Colors.RED}Error: Updating/indexing is disabled in federated include mode. Update individual collection configurations directly.{Colors.RESET}")
        sys.exit(1)
        
    if getattr(args, "build_ann", False):
        store.build_usearch_index()
        return

    # 1. Update existing collections
    for name, coll_cfg in config.collections.items():
        if args.collection:
            c_filter = args.collection.lower()
            if name.lower() != c_filter and c_filter not in name.lower():
                continue

        if args.pull:
            repo_path = Path(coll_cfg.path).expanduser().resolve()
            if (repo_path / ".git").exists():
                print(f"Pulling latest changes for {name}...")
                try:
                    subprocess.run(
                        ["git", "pull"], 
                        cwd=repo_path, 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    print(f"{Colors.GREEN}✓ Git pull successful.{Colors.RESET}")
                except subprocess.CalledProcessError as e:
                    print(f"{Colors.RED}✗ Git pull failed for {name}: {e.stderr.strip()}{Colors.RESET}")
            else:
                print(f"Skipping git pull: {name} is not a git repository.")
        
        store.index_collection(name, coll_cfg, force=args.force, verbose=getattr(args, "verbose", False), quick=getattr(args, "quick", False))

    # 2. Prune removed collections
    active_collections = list(config.collections.keys())
    store.prune_orphaned_collections(active_collections)

    # 3. Automatically build / update the HNSW ANN index
    if not getattr(args, "no_ann", False):
        store.build_usearch_index()

    # 4. Report indexing errors if any remain
    errors = store.get_indexing_errors(collection=args.collection)
    if errors:
        print(f"\n{Colors.YELLOW}Notice: {len(errors)} file(s) have unresolved indexing errors/degradations.{Colors.RESET}")
        for err in errors[:5]:
            print(f"  • [{err['collection']}] {err['path']} - {err['error_type']}: {err['error_message']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more.")
    else:
        print(f"{Colors.GREEN}✓ All files indexed cleanly with zero errors.{Colors.RESET}")

def handle_report(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    # 1. Build Configuration Map from YAML or CLI arguments
    report_cfg = {}
    if getattr(args, "yaml", None):
        import yaml
        try:
            with open(args.yaml, 'r', encoding='utf-8') as f:
                report_cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"{Colors.RED}Error loading YAML file: {e}{Colors.RESET}")
            sys.exit(1)
    else:
        fields = []
        sort_rules = []
        if getattr(args, "fields", None):
            raw_fields = [f.strip() for f in args.fields.split(',') if f.strip()]
            for rf in raw_fields:
                if rf.endswith('+'):
                    sort_rules.append({"field": rf[:-1], "order": "asc"})
                    fields.append(rf[:-1])
                elif rf.endswith('-'):
                    sort_rules.append({"field": rf[:-1], "order": "desc"})
                    fields.append(rf[:-1])
                else:
                    fields.append(rf)
        if fields:
            report_cfg["fields"] = fields
        if sort_rules:
            report_cfg["sort"] = sort_rules

    # Merge CLI limits and targets into config so YAML can be overridden or augmented by CLI
    if "target" not in report_cfg:
        report_cfg["target"] = {}
    
    collection = getattr(args, "collection", None)
    if collection: report_cfg["target"]["collection"] = collection
    
    path = getattr(args, "path", None)
    if path: report_cfg["target"]["path"] = path
    
    title = getattr(args, "title", None)
    if title: report_cfg["target"]["title"] = title

    limit = getattr(args, "limit", 100)

    # Execute custom report compilation
    results = store.build_custom_report(report_cfg=report_cfg, limit=limit)

    if getattr(args, "json", False):
        format_report_json(results)
    elif getattr(args, "csv", False):
        format_report_csv(results)
    elif is_xml:
        format_report_xml(results)
    else:
        format_report_cli(results)

def handle_analyze(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)

    limit = getattr(args, "limit", 100)
    time_limit = getattr(args, "time_limit", None)
    force = getattr(args, "force", False)
    outdated = getattr(args, "outdated", False)
    verbose = getattr(args, "verbose", False)
    
    collection = getattr(args, "collection", None)
    path = getattr(args, "path", None)
    title = getattr(args, "title", None)

    results = store.analyze_target(
        collection=collection,
        path=path,
        title=title,
        limit=limit,
        time_limit=time_limit,
        force=force,
        outdated=outdated,
        verbose=verbose
    )

    if getattr(args, "json", False):
        import json
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print(f"{Colors.YELLOW}No documents found to analyze or all matched documents are already analyzed.{Colors.RESET}")
            return
        print(f"\n{Colors.CYAN}--- Document Analysis Results ---{Colors.RESET}")
        format_report_cli(results)