import sys
from qmd.store import Store
from qmd.utils import parse_target_spec, parse_int_ranges
from qmd.formatters.cli import format_extraction_cli, format_outline_cli, format_chunks_cli
from qmd.formatters.xml import format_extraction_xml, format_outline_xml, format_chunks_xml
from qmd.formatters.json import format_results_json
from qmd.formatters.core import set_plain_mode
from qmd.formatters.colors import Colors

from .helpers import _is_xml_output

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
        format_extraction_xml(data)
    else:
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