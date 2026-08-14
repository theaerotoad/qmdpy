import argparse
import sys
import subprocess
import os
import secrets
import re
from pathlib import Path
from typing import List, Dict, Tuple

from qmd.config import load_config
from qmd.store import Store, Result
from qmd.db import get_seen_chunks_for_session, record_session_event, record_session_results, get_db_meta
from qmd.formatting import format_results_cli, format_doc_results_cli, format_results_json, format_doc_results_json, format_outline_cli, format_chunks_cli, set_plain_mode, RED, GREEN, RESET
from qmd.utils import redact_pii

def _clean_header_part(part: str) -> str:
    """Strips leading/trailing markdown hashes, whitespace, and formatting."""
    if not part:
        return ""
    return re.sub(r'^\s*#+\s*', '', part).strip()

def _normalize_str(s: str) -> str:
    """Normalizes string for fuzzy comparison (lowercase, alphanumeric only)."""
    return re.sub(r'[^\w]', '', s.lower())

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
                "headers": getattr(r, "headers", "")
            })
            
    output_list = []
    for d in docs.values():
        d["snippets"] = merge_overlapping_snippets(d["raw_snippets"], doc_title=d["title"])
        del d["raw_snippets"]
        output_list.append(d)
        
    return sorted(output_list, key=lambda x: x['score'], reverse=True)

def handle_search(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)

    query = " ".join(args.query)
    limit = args.limit if getattr(args, "limit", None) is not None else getattr(store.config, "default_limit", 10)
    fts_limit = getattr(args, "fts_limit", None)
    vec_limit = getattr(args, "vec_limit", None)
    rerank_candidates = getattr(args, "rerank_candidates", None)
    
    session_id = getattr(args, "session", None)
    if not session_id:
        session_id = secrets.token_hex(4)

    exclude_seen = getattr(args, "exclude_seen", False)
    exclude_seen_set = set()
    if exclude_seen:
        exclude_seen_set = get_seen_chunks_for_session(store.history_conn, session_id)

    if getattr(args, "w2n", False):
        results = store.wide_to_narrow_search(
            query,
            limit=limit,
            verbose=args.verbose,
            rerank=args.rerank,
            reranker_only=args.rerank_only,
            collection=args.collection,
            lexical_query=args.lex,
            title=args.title,
            path=args.path,
            fts_limit=fts_limit,
            vec_limit=vec_limit,
            rerank_candidates=rerank_candidates,
            exclude_seen_set=exclude_seen_set
        )
    else:
        results = store.hybrid_search(
            query, 
            limit=limit, 
            verbose=args.verbose,
            rerank=args.rerank,
            reranker_only=args.rerank_only,
            collection=args.collection,
            lexical_query=args.lex,
            title=args.title,
            path=args.path,
            fts_limit=fts_limit,
            vec_limit=vec_limit,
            rerank_candidates=rerank_candidates,
            exclude_seen_set=exclude_seen_set
        )
    
    if getattr(args, "redact_pii", False):
        for r in results:
            r.text = redact_pii(r.text)
            if hasattr(r, "title") and r.title:
                r.title = redact_pii(r.title)

    event_type = "doc_view" if args.doc else "search"
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

    if args.doc:
        grouped = group_results_by_doc(results)
        grouped = grouped[:args.limit]
        
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
            format_doc_results_json(grouped, session_id=session_id, exclusion_stats=store.last_exclusion_stats)
        else:
            format_doc_results_cli(grouped, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats)
    else:
        record_session_results(store.history_conn, session_id, event_id, results)
        if args.json:
            format_results_json(results, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats)
        else:
            format_results_cli(results, query=query, verbose=args.verbose, session_id=session_id, exclusion_stats=store.last_exclusion_stats)

def handle_outline(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)

    outline = store.get_document_outline(collection=args.collection, path=args.path)
    if not outline:
        if args.json:
            import json
            print(json.dumps({"error": "Document not found"}, indent=2))
        else:
            print(f"{RED}Error: Document not found matching path '{args.path}'{RESET}")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(outline, indent=2))
    else:
        format_outline_cli(outline)

def handle_chunk(args, store: Store):
    if getattr(args, "plain", False):
        set_plain_mode(True)

    target = args.target
    results = []

    if target.isdigit() and args.seq is None:
        rowid = int(target)
        results = store.get_chunk_by_id(rowid, window=args.window)
    else:
        seq = args.seq if args.seq is not None else 0
        results = store.get_chunk_by_seq(args.collection, target, seq_id=seq, window=args.window)

    if not results:
        if args.json:
            import json
            print(json.dumps({"error": "Chunk not found"}, indent=2))
        else:
            print(f"{RED}Error: Chunk not found.{RESET}")
        sys.exit(1)

    if args.json:
        format_results_json(results)
    else:
        format_chunks_cli(results, window=args.window)

def handle_update(args, store: Store):
    config = store.config
    
    # 1. Update existing collections
    for name, coll_cfg in config.collections.items():
        if args.collection and name != args.collection:
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
        
        store.index_collection(name, coll_cfg, force=args.force)

    # 2. Prune removed collections
    active_collections = list(config.collections.keys())
    store.prune_orphaned_collections(active_collections)

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-C", "--config", type=str, default=argparse.SUPPRESS, help="Path to custom YAML configuration file")

    parser = argparse.ArgumentParser(prog="qmd", description="Quick Markdown Search", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", aliases=["query", "q"], help="Search the index", parents=[parent_parser])
    search_parser.add_argument("query", nargs="+", help="The search terms")
    search_parser.add_argument("--limit", type=int, default=None, help="Number of final results to show")
    search_parser.add_argument("--fts-limit", type=int, default=None, help="Max number of FTS (lexical) matches to retrieve")
    search_parser.add_argument("--vec-limit", type=int, default=None, help="Max number of Vector (semantic) matches to retrieve")
    search_parser.add_argument("--rerank-candidates", type=int, default=None, help="Number of combined RRF candidates to send to reranker")
    search_parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show diagnostic info")
    search_parser.add_argument("-d", "--doc", action="store_true", help="Group results by document (Document View)")
    search_parser.add_argument("-r", "--rerank", action="store_true", help="Use LLM to rerank results")
    search_parser.add_argument("--rerank-only", action="store_true", help="Sort results purely by reranker score (implies --rerank)")
    search_parser.add_argument("-c", "--collection", type=str, help="Filter results by a specific collection")
    search_parser.add_argument("-t", "--title", type=str, help="Filter results by a specific title (substring match)")
    search_parser.add_argument("-p", "--path", type=str, help="Filter results by a specific path (substring match)")
    search_parser.add_argument("--lex", type=str, help="Override the lexical (FTS) search terms")
    search_parser.add_argument("-w", "--w2n", action="store_true", help="Use Wide-to-Narrow hierarchical search")
    search_parser.add_argument("--redact-pii", "--redact", action="store_true", help="Redact email addresses and phone numbers from search results")
    search_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting in search output")
    search_parser.add_argument("--session", type=str, help="Session ID for tracking history and excluding seen results")
    search_parser.add_argument("--exclude-seen", action="store_true", help="Exclude previously seen chunks from the active session")

    outline_parser = subparsers.add_parser("outline", help="Show heading outline and chunk mapping for a document", parents=[parent_parser])
    outline_parser.add_argument("path", help="Path or relative path to the document")
    outline_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    outline_parser.add_argument("--json", action="store_true", help="Output outline as JSON")
    outline_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    chunk_parser = subparsers.add_parser("chunk", help="Fetch specific chunk by rowid or path with surrounding context window", parents=[parent_parser])
    chunk_parser.add_argument("target", help="Document path OR chunk rowid")
    chunk_parser.add_argument("--seq", type=int, default=None, help="Sequence ID of chunk (when target is a document path)")
    chunk_parser.add_argument("-w", "--window", type=int, default=0, help="Number of surrounding chunks to fetch on each side")
    chunk_parser.add_argument("-c", "--collection", type=str, help="Filter by collection name")
    chunk_parser.add_argument("--json", action="store_true", help="Output chunks as JSON")
    chunk_parser.add_argument("--plain", action="store_true", help="Disable ASCII color formatting")

    update_parser = subparsers.add_parser("update", help="Update the index", parents=[parent_parser])
    update_parser.add_argument("--pull", action="store_true", help="Run 'git pull' before indexing")
    update_parser.add_argument("-f", "--force", action="store_true", help="Force re-indexing of all files, ignoring hash checks")
    update_parser.add_argument("-c", "--collection", type=str, help="Only update a specific collection")

    coll_parser = subparsers.add_parser("collection", help="Manage collections", parents=[parent_parser])
    coll_sub = coll_parser.add_subparsers(dest="subcommand", required=True)
    coll_list = coll_sub.add_parser("list", help="List all configured collections", parents=[parent_parser])

    serve_parser = subparsers.add_parser("serve", help="Start the web UI", parents=[parent_parser])
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")

    args = parser.parse_args()
    config_path = getattr(args, "config", None)
    config = load_config(config_path)
    store = Store(config)

    try:
        if args.command in ["search", "query", "q"]:
            handle_search(args, store)
        elif args.command == "outline":
            handle_outline(args, store)
        elif args.command == "chunk":
            handle_chunk(args, store)
        elif args.command == "update":
            handle_update(args, store)
        elif args.command == "collection":
            if args.subcommand == "list":
                for name, cfg in config.collections.items():
                    print(f"{GREEN}{name}{RESET}: {cfg.path} ({cfg.glob})")
        elif args.command == "serve":
            from qmd.web import start_server
            start_server(port=args.port, config_path=config_path)
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
