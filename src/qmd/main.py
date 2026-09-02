import argparse
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
from qmd.formatting import (
    format_results_cli, format_doc_results_cli, format_results_json, format_doc_results_json,
    format_discover_cli, format_discover_json, format_discover_xml,
    format_outline_cli, format_chunks_cli, format_results_xml, format_doc_results_xml,
    format_chunks_xml, format_outline_xml, format_collection_tree_cli, format_collection_tree_xml,
    format_grep_cli, format_grep_json, format_grep_xml,
    set_plain_mode, strip_ansi, BOLD, CYAN, GREEN, RED, RESET, YELLOW
)
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

    chunk_snips = sorted([s for s in normalized if s[0] >= 0], key=lambda x: x[0])
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
    Groups chunks by file path, keeps max score, and merges text into reading order.
    """
    docs = {}
    for r in results:
        key = (r.collection, r.path)
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
    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 10)
    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)

    session_id = getattr(args, "session", None)
    has_explicit_session = session_id is not None
    if not session_id:
        session_id = secrets.token_hex(4)

    include_seen = getattr(args, "include_seen", False)
    if has_explicit_session:
        exclude_seen = not include_seen
    else:
        exclude_seen = getattr(args, "exclude_seen", False) and not include_seen

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
        "collection": args.collection,
        "lexical_query": args.lex,
        "title": args.title,
        "path": args.path,
        "fts_limit": fts_limit,
        "vec_limit": vec_limit,
        "rerank_candidates": rerank_candidates,
        "exclude_seen_set": exclude_seen_set,
        "w2n": is_w2n,
    }
    if getattr(args, "no_cache", False):
        search_kwargs["use_cache"] = False

    results = store.discover(**search_kwargs)

    if getattr(args, "redact_pii", False):
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
        getattr(args, "lex", None),
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
    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 10)
    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)
    
    session_id = getattr(args, "session", None)
    has_explicit_session = session_id is not None
    if not session_id:
        session_id = secrets.token_hex(4)

    include_seen = getattr(args, "include_seen", False)
    if has_explicit_session:
        exclude_seen = not include_seen
    else:
        exclude_seen = getattr(args, "exclude_seen", False) and not include_seen

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
        "reranker_only": args.rerank_only,
        "collection": args.collection,
        "lexical_query": args.lex,
        "title": args.title,
        "path": args.path,
        "fts_limit": fts_limit,
        "vec_limit": vec_limit,
        "rerank_candidates": rerank_candidates,
        "exclude_seen_set": exclude_seen_set,
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
    
    if getattr(args, "redact_pii", False):
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
        getattr(args, "lex", None),
        str(store.config.db_path),
        db_last_updated
    )

    # Determine max_chunks cap
    max_chunks = getattr(args, "max_chunks", None)
    if max_chunks is None:
        cfg_val = getattr(getattr(store, "config", None), "max_chunks_per_response", 30)
        max_chunks = cfg_val if isinstance(cfg_val, int) else 30

    if is_doc:
        grouped = group_results_by_doc(results)
        grouped = grouped[:limit]

        # Enforce max_chunks cap across all grouped documents
        truncation_info = None
        if isinstance(max_chunks, int) and max_chunks > 0:
            total_doc_chunks = sum(len(d.get("chunks", [])) for d in grouped)
            if total_doc_chunks > max_chunks:
                omitted = total_doc_chunks - max_chunks
                truncation_info = {
                    "omitted_remaining": omitted,
                    "limit": max_chunks
                }
                curr_count = 0
                new_grouped = []
                for d in grouped:
                    doc_chunks = d.get("chunks", [])
                    if curr_count + len(doc_chunks) <= max_chunks:
                        new_grouped.append(d)
                        curr_count += len(doc_chunks)
                    else:
                        remaining_slots = max_chunks - curr_count
                        if remaining_slots > 0:
                            d["chunks"] = doc_chunks[:remaining_slots]
                            new_grouped.append(d)
                            curr_count += remaining_slots
                        break
                grouped = new_grouped
        
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
            format_results_json(results, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)
        elif is_xml:
            format_results_xml(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, seen_chunks=seen_chunks_count, truncation_info=truncation_info)
        else:
            format_results_cli(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats, truncation_info=truncation_info)

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
            print(f"{RED}Error: Document not found matching path '{target_path}'{RESET}")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(outline, indent=2))
    elif is_xml:
        format_outline_xml(outline)
    else:
        format_outline_cli(outline)

def handle_grep(args, store: Store):
    is_xml = _is_xml_output(args)
    if getattr(args, "plain", False) or is_xml:
        set_plain_mode(True)

    pattern = args.pattern
    is_regex = getattr(args, "regex", False)
    case_sensitive = getattr(args, "case_sensitive", False) and not getattr(args, "ignore_case", False)
    limit = getattr(args, "limit", 50) or 50

    coll = getattr(args, "collection", None)
    path = getattr(args, "path", None)
    if path:
        spec = parse_target_spec(path, default_collection=coll)
        if spec["collection"]:
            coll = spec["collection"]
        if spec["path"]:
            path = spec["path"]

    try:
        results = store.grep_search(
            pattern=pattern,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            collection=coll,
            path=path,
            limit=limit
        )
    except ValueError as e:
        if args.json:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

    if args.json:
        format_grep_json(results)
    elif is_xml:
        format_grep_xml(results, pattern=pattern, is_regex=is_regex, case_sensitive=case_sensitive)
    else:
        format_grep_cli(results, pattern=pattern)

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
                print(f"{RED}Error: Invalid sequence range specification: '{args.seq}'{RESET}")
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
            print(f"{RED}Error: Chunk not found.{RESET}")
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
            print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

    if coll_name and tree_data is None:
        if args.json:
            import json
            print(json.dumps({"error": f"Collection '{coll_name}' not found"}, indent=2))
        else:
            print(f"{RED}Error: Collection '{coll_name}' not found.{RESET}")
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
    <step num="1" name="Discovery">Use `qmd discover "natural language question?"` for top-level SERP results (1 hit per document), or `qmd collections` / `qmd tree` to explore paths.</step>
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
        guide_md = f"""{BOLD}QMD LLM Agent Research & Inspection Guide{RESET}

{CYAN}## Workflow & Decision Matrix{RESET}
1. {BOLD}Triage & Discovery:{RESET}
   - `qmd discover "question?"` - Single top hit per document SERP view (use natural language questions)
   - `qmd collections` - List indexed collections and paths
   - `qmd tree [collection] [-p pattern]` - Explore document directory trees

2. {BOLD}Retrieval & Search:{RESET}
   - `qmd search "question?"` - Document-ordered search results (natural language question)
   - `qmd search "question?" --deep` - Deep search: doc-grouped + LLM reranked
   - `qmd search "question?" --session <id>` - Session search (auto-deduplicates seen chunks)
   - `qmd search "question?" --broad` - Broad hierarchical search (wide-to-narrow)

3. {BOLD}Document Structure & Orientation:{RESET}
   - `qmd outline "<target>"` - Table of contents with chunk sequence mappings

4. {BOLD}Targeted Reading:{RESET}
   - `qmd read "<target>"` - Read specific chunks, ranges, or surrounding context
   - Examples: `qmd read 'Books:doc.epub:10-15'`, `qmd read 'qmd://Books/doc.epub:3'`

{CYAN}## Target Shorthand Syntax{RESET}
- Full URI: `qmd://<collection>/<path>[:<seq>]`
- Shorthand: `<collection>:<path>[:<seq>]`
- Relative: `<path>[:<seq>]`
- Chunk Row IDs: `10-15` or `22,40,25-27`

{CYAN}## Query Formulation & Agent Best Practices{RESET}
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
            print(f"{GREEN}{name}{RESET}: {cfg.path} ({cfg.glob})")

def handle_update(args, store: Store):
    if getattr(args, "verbose", False):
        os.environ["QMD_VERBOSE"] = "1"

    config = store.config
    if getattr(config, "is_federated", False):
        print(f"{RED}Error: Updating/indexing is disabled in federated include mode. Update individual collection configurations directly.{RESET}")
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
                    print(f"{GREEN}✓ Git pull successful.{RESET}")
                except subprocess.CalledProcessError as e:
                    print(f"{RED}✗ Git pull failed for {name}: {e.stderr.strip()}{RESET}")
            else:
                print(f"Skipping git pull: {name} is not a git repository.")
        
        store.index_collection(name, coll_cfg, force=args.force, verbose=getattr(args, "verbose", False))

    # 2. Prune removed collections
    active_collections = list(config.collections.keys())
    store.prune_orphaned_collections(active_collections)

    # 3. Automatically build / update the HNSW ANN index
    if not getattr(args, "no_ann", False):
        store.build_usearch_index()

    # 4. Report indexing errors if any remain
    errors = store.get_indexing_errors(collection=args.collection)
    if errors:
        print(f"\n{YELLOW}Notice: {len(errors)} file(s) have unresolved indexing errors/degradations.{RESET}")
        for err in errors[:5]:
            print(f"  • [{err['collection']}] {err['path']} - {err['error_type']}: {err['error_message']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more.")
    else:
        print(f"{GREEN}✓ All files indexed cleanly with zero errors.{RESET}")

class HelpAllAction(argparse.Action):
    root_parser: Optional[argparse.ArgumentParser] = None

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help="Show help for all commands and subcommands and exit"):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        target_parser = HelpAllAction.root_parser or parser
        print(format_help_all(target_parser))
        parser.exit(0)

def format_help_all(parser: argparse.ArgumentParser) -> str:
    """
    Recursively collects and formats help text for the root parser and all subcommands.
    """
    sections = [parser.format_help().rstrip()]

    def _collect_subparsers(p: argparse.ArgumentParser, seen: Set[int]):
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, subp in action.choices.items():
                    if id(subp) in seen:
                        continue
                    seen.add(id(subp))
                    sections.append(subp.format_help().rstrip())
                    _collect_subparsers(subp, seen)

    seen = {id(parser)}
    _collect_subparsers(parser, seen)
    separator = "\n\n" + "=" * 80 + "\n\n"
    return separator.join(sections) + "\n"

def build_parser():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-C", "--config", type=str, default=argparse.SUPPRESS, help="Path to custom YAML configuration file")
    parent_parser.add_argument("--helpall", action=HelpAllAction, help="Show help for all commands and subcommands and exit")

    parser = argparse.ArgumentParser(prog="qmd", description="Query Multiple Documents", parents=[parent_parser])
    HelpAllAction.root_parser = parser
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", aliases=["find", "disc"], help="Discover top document matches (1 top hit per document)", parents=[parent_parser])
    discover_parser.add_argument("query", nargs="+", help="The natural language question or search terms")

    d_filter_group = discover_parser.add_argument_group("Target & Filters")
    d_filter_group.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    d_filter_group.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    d_filter_group.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    d_filter_group.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")

    d_mode_group = discover_parser.add_argument_group("Search Mode & Quality")
    d_mode_group.add_argument("--deep", action="store_true", help="Deep discovery: enable LLM reranking")
    d_mode_group.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    d_mode_group.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    d_mode_group.add_argument("-w", "--w2n", action="store_true", help="Use Wide-to-Narrow hierarchical search")
    d_mode_group.add_argument("--broad", action="store_true", help="Broad search: alias for Wide-to-Narrow hierarchical search (--w2n)")
    d_mode_group.add_argument("--limit", type=int, default=None, help="Number of final documents to show")
    d_mode_group.add_argument("--max-chunks", type=int, default=None, help="Max documents to return (defaults to config max_chunks_per_response or 30)")
    d_mode_group.add_argument("--no-cache", action="store_true", help="Bypass and do not write to search result cache")
    d_mode_group.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    d_mode_group.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    d_mode_group.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")

    d_session_group = discover_parser.add_argument_group("Session & History")
    d_session_group.add_argument("--session", type=str, help="Session ID for tracking history and deduplication")
    d_session_group.add_argument("--exclude-seen", action="store_true", help="Exclude previously seen chunks from the active session")
    d_session_group.add_argument("--include-seen", action="store_true", help="Include previously seen chunks (overrides default session deduplication)")

    d_output_group = discover_parser.add_argument_group("Output & Formatting")
    d_output_group.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    d_output_group.add_argument("--llm", action="store_true", help="Alias for --xml (optimizes output for LLM agent context)")
    d_output_group.add_argument("--json", action="store_true", help="Output results in JSON format")
    d_output_group.add_argument("--redact-pii", "--redact", action="store_true", help="Redact email addresses and phone numbers from search results")
    d_output_group.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    d_output_group.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")

    search_parser = subparsers.add_parser("search", aliases=["query", "q"], help="Hybrid vector + lexical search (document-ordered by default)", parents=[parent_parser])
    search_parser.add_argument("query", nargs="+", help="The natural language question or search terms")

    filter_group = search_parser.add_argument_group("Target & Filters")
    filter_group.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    filter_group.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    filter_group.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    filter_group.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")

    mode_group = search_parser.add_argument_group("Search Mode & Quality")
    mode_group.add_argument("--deep", action="store_true", help="Deep search: document grouping with LLM reranking (implies -r)")
    mode_group.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    mode_group.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    mode_group.add_argument("-w", "--w2n", action="store_true", help="Use Wide-to-Narrow hierarchical search")
    mode_group.add_argument("--broad", action="store_true", help="Broad search: alias for Wide-to-Narrow hierarchical search (--w2n)")
    mode_group.add_argument("--limit", type=int, default=None, help="Number of final results to show")
    mode_group.add_argument("--max-chunks", type=int, default=None, help="Max chunks to return (defaults to config max_chunks_per_response or 30)")
    mode_group.add_argument("--no-cache", action="store_true", help="Bypass and do not write to search result cache")
    mode_group.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    mode_group.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    mode_group.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")

    session_group = search_parser.add_argument_group("Session & History")
    session_group.add_argument("--session", type=str, help="Session ID for tracking history and deduplication")
    session_group.add_argument("--exclude-seen", action="store_true", help="Exclude previously seen chunks from the active session")
    session_group.add_argument("--include-seen", action="store_true", help="Include previously seen chunks (overrides default session deduplication)")

    output_group = search_parser.add_argument_group("Output & Formatting")
    output_group.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    output_group.add_argument("--llm", action="store_true", help="Alias for --xml (optimizes output for LLM agent context)")
    output_group.add_argument("--json", action="store_true", help="Output results in JSON format")
    output_group.add_argument("-d", "--doc", action="store_true", default=True, help="Group results by document (default)")
    output_group.add_argument("--flat", "--chunks", action="store_true", help="Output flat individual chunks instead of document-grouped view")
    output_group.add_argument("--redact-pii", "--redact", action="store_true", help="Redact email addresses and phone numbers from search results")
    output_group.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")

    outline_parser = subparsers.add_parser("outline", help="Show heading outline and chunk mapping for a document", parents=[parent_parser])
    outline_parser.add_argument("path", help="Path or relative path to the document")
    outline_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    outline_parser.add_argument("-d", "--depth", type=int, default=None, help="Maximum heading depth to display (e.g. 1 for H1 only, 2 for H1-H2)")
    outline_parser.add_argument("-p", "--pattern", type=str, default=None, help="Filter headings by substring pattern")
    outline_parser.add_argument("--json", action="store_true", help="Output outline as JSON")
    outline_parser.add_argument("--xml", action="store_true", help="Output outline as XML for LLM context")
    outline_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    outline_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    grep_parser = subparsers.add_parser("grep", help="[Deprecated: use discover or search] Direct pattern search across raw document bodies", parents=[parent_parser])
    grep_parser.add_argument("pattern", help="Search pattern or regular expression")
    grep_parser.add_argument("-r", "--regex", action="store_true", help="Treat pattern as a regular expression")
    grep_parser.add_argument("-s", "--case-sensitive", action="store_true", help="Perform case-sensitive matching")
    grep_parser.add_argument("-i", "--ignore-case", action="store_true", help="Perform case-insensitive matching (default)")
    grep_parser.add_argument("-c", "--collection", type=str, default=None, help="Filter by collection name")
    grep_parser.add_argument("-p", "--path", type=str, default=None, help="Filter by file path substring")
    grep_parser.add_argument("--limit", type=int, default=50, help="Maximum number of matching lines to return")
    grep_parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    grep_parser.add_argument("--xml", action="store_true", help="Output results in XML format for LLM context")
    grep_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    grep_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    read_parser = subparsers.add_parser("read", aliases=["chunk", "get", "view"], help="Fetch specific chunk by rowid, path, or URI with surrounding context window", parents=[parent_parser])
    read_parser.add_argument("target", help="Document path, URI (qmd://...), shorthand (coll:path:seq), OR chunk rowid/ranges")
    read_parser.add_argument("--seq", type=str, default=None, help="Sequence ID or range of chunk(s) (when target is a document path, e.g. 0, 1-5, 2,4,6-8)")
    read_parser.add_argument("-w", "--window", type=int, default=0, help="Number of surrounding chunks to fetch on each side")
    read_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    read_parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to return per read call (defaults to config max_chunks_per_response or 30)")
    read_parser.add_argument("--json", action="store_true", help="Output chunks as JSON")
    read_parser.add_argument("--xml", action="store_true", help="Output chunks as XML for LLM context")
    read_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    read_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    tree_parser = subparsers.add_parser("tree", help="Display hierarchical directory tree of indexed files", parents=[parent_parser])
    tree_parser.add_argument("name", nargs="?", default=None, help="Optional collection name")
    tree_parser.add_argument("-c", "--collection", type=str, default=None, help="Filter by collection name")
    tree_parser.add_argument("-p", "--pattern", type=str, default=None, help="Pattern or substring to filter file paths and titles")
    tree_parser.add_argument("-r", "--regex", action="store_true", help="Treat pattern as a regular expression")
    tree_parser.add_argument("-s", "--case-sensitive", action="store_true", help="Perform case-sensitive matching")
    tree_parser.add_argument("-i", "--ignore-case", action="store_true", help="Perform case-insensitive matching (default)")
    tree_parser.add_argument("--depth", type=int, default=None, help="Maximum directory depth to display")
    tree_parser.add_argument("--json", action="store_true", help="Output directory tree as JSON")
    tree_parser.add_argument("--xml", action="store_true", help="Output directory tree as XML for LLM context")
    tree_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    tree_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    colls_parser = subparsers.add_parser("collections", aliases=["colls"], help="List configured collections and paths", parents=[parent_parser])
    colls_parser.add_argument("--json", action="store_true", help="Output collections list as JSON")
    colls_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    guide_parser = subparsers.add_parser("guide", help="Emit LLM agent research guide and command decision matrix", parents=[parent_parser])
    guide_parser.add_argument("--xml", action="store_true", help="Output guide in XML format")
    guide_parser.add_argument("--llm", action="store_true", help="Alias for --xml")
    guide_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    update_parser = subparsers.add_parser("update", help="Update the index", parents=[parent_parser])
    update_parser.add_argument("--pull", action="store_true", help="Run 'git pull' before indexing")
    update_parser.add_argument("-f", "--force", action="store_true", help="Force re-indexing of all files, ignoring hash checks")
    update_parser.add_argument("-c", "--collection", type=str, help="Only update a specific collection")
    update_parser.add_argument("--build-ann", action="store_true", help="Build a usearch HNSW approximate nearest neighbor index from the existing vector table")
    update_parser.add_argument("--no-ann", action="store_true", help="Skip automatic HNSW ANN index build/update")
    update_parser.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info during update")

    coll_parser = subparsers.add_parser("collection", help="Manage collections", parents=[parent_parser])
    coll_sub = coll_parser.add_subparsers(dest="subcommand", required=True)
    coll_list = coll_sub.add_parser("list", help="List all configured collections", parents=[parent_parser])

    coll_tree = coll_sub.add_parser("tree", help="Display directory tree of indexed files", parents=[parent_parser])
    coll_tree.add_argument("name", nargs="?", default=None, help="Optional collection name")
    coll_tree.add_argument("-c", "--collection", type=str, default=None, help="Filter by collection name")
    coll_tree.add_argument("-p", "--pattern", type=str, default=None, help="Pattern or substring to filter file paths and titles")
    coll_tree.add_argument("-r", "--regex", action="store_true", help="Treat pattern as a regular expression")
    coll_tree.add_argument("-s", "--case-sensitive", action="store_true", help="Perform case-sensitive matching")
    coll_tree.add_argument("-i", "--ignore-case", action="store_true", help="Perform case-insensitive matching (default)")
    coll_tree.add_argument("--depth", type=int, default=None, help="Maximum directory depth to display")
    coll_tree.add_argument("--json", action="store_true", help="Output directory tree as JSON")
    coll_tree.add_argument("--xml", action="store_true", help="Output directory tree as XML for LLM context")
    coll_tree.add_argument("--llm", action="store_true", help="Alias for --xml")
    coll_tree.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    serve_parser = subparsers.add_parser("serve", help="Start the web UI", parents=[parent_parser])
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")

    mcp_parser = subparsers.add_parser("mcp", help="Start the stdio MCP server", parents=[parent_parser])

    return parser

def execute_command(args, store):
    if args.command in ["discover", "find", "disc"]:
        handle_discover(args, store)
    elif args.command in ["search", "query", "q"]:
        handle_search(args, store)
    elif args.command == "outline":
        handle_outline(args, store)
    elif args.command == "grep":
        handle_grep(args, store)
    elif args.command in ["read", "chunk", "get", "view"]:
        handle_chunk(args, store)
    elif args.command == "tree":
        handle_collection_tree(args, store)
    elif args.command in ["collections", "colls"]:
        handle_collections_list(args, store)
    elif args.command == "guide":
        handle_guide(args, store)
    elif args.command == "update":
        handle_update(args, store)
    elif args.command == "collection":
        if args.subcommand == "list":
            handle_collections_list(args, store)
        elif args.subcommand == "tree":
            handle_collection_tree(args, store)
    elif args.command == "serve":
        from qmd.web import start_server
        start_server(port=args.port, config_path=getattr(args, "config", None))
    elif args.command == "mcp":
        from qmd.mcp_server import run_mcp_server
        run_mcp_server(getattr(args, "config", None))

def main():
    parser = build_parser()
    args = parser.parse_args()
    config_path = getattr(args, "config", None)
    
    if args.command == "mcp":
        from qmd.mcp_server import run_mcp_server
        run_mcp_server(config_path)
        return

    config = load_config(config_path)
    is_write = args.command in ["update"]
    if is_write and getattr(config, "is_federated", False):
        print(f"{RED}Error: Updating/indexing is disabled in federated include mode. Update individual collection configurations directly.{RESET}")
        sys.exit(1)
    store = Store(config, read_only=not is_write)

    try:
        execute_command(args, store)
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()