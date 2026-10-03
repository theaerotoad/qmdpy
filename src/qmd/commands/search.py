import os
import secrets
from typing import Optional, Set

from qmd.store import Store
from qmd.db import get_seen_chunks_for_session, record_session_event, record_session_results, get_db_meta
from qmd.utils import redact_pii
from qmd.formatters.cli import format_discover_cli, format_map_cli, format_results_cli, format_doc_results_cli
from qmd.formatters.json import format_discover_json, format_results_json, format_doc_results_json
from qmd.formatters.xml import format_discover_xml, format_map_xml, format_results_xml, format_doc_results_xml
from qmd.formatters.core import set_plain_mode

from .helpers import _is_xml_output, group_results_by_doc

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