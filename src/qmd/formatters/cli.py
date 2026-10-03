from typing import List, Dict, Optional, Tuple, Any, Union
from .colors import c
from .core import highlight_keywords

def format_results_cli(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Prints standard search results (snippets)."""
    if session_id:
        stats_str = ""
        if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
            c_count = exclusion_stats["excluded_chunks"]
            d_count = exclusion_stats.get("excluded_docs", 0)
            stats_str = f" | Excluded {c_count} previously seen chunk(s) across {d_count} document(s)"
        print(f"{c.DIM}[Session: {session_id}{stats_str}]{c.RESET}")

    if not results:
        print(f"\n{c.RED}No results found.{c.RESET}")
        return

    print(f"\n{c.DIM}Found {len(results)} chunks:{c.RESET}\n")
    
    for i, res in enumerate(results):
        rank_str = f"{i+1}."
        path_str = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        header_str = f" {c.CYAN}[{res.headers}]{c.RESET}" if getattr(res, 'headers', None) else ""

        print(f"{c.GREEN}{rank_str}{c.RESET} {c.BOLD}{res.title}{c.RESET}{header_str} {c.DIM}({path_str}){c.RESET}")

        meta_parts = []
        if getattr(res, 'alt_title', None): meta_parts.append(f"Alt: {res.alt_title}")
        if getattr(res, 'doc_type', None): meta_parts.append(f"Type: {res.doc_type}")
        if getattr(res, 'doc_date', None): meta_parts.append(f"Date: {res.doc_date}")
        if getattr(res, 'authors', None): meta_parts.append(f"Authors: {', '.join(res.authors)}")
        if meta_parts:
            print(f"   {c.CYAN}{' | '.join(meta_parts)}{c.RESET}")

        snippet = res.text[:2000].replace("\n", " ") + "..."
        highlighted = highlight_keywords(snippet, query)
        print(f"   {highlighted}")
        
        print(f"   {c.DIM}Score: {res.score:.4f} | Source: {res.source} | Rank: {res.rank}{c.RESET}")
        if verbose:
            fts_rank = getattr(res, 'fts_rank', None)
            fts_score = getattr(res, 'fts_score', None)
            vec_rank = getattr(res, 'vec_rank', None)
            vec_score = getattr(res, 'vec_score', None)
            rrf_rank = getattr(res, 'rrf_rank', None)
            rrf_score = getattr(res, 'rrf_score', None)

            fts_str = f"rank {fts_rank} (score {fts_score:.4f})" if fts_rank is not None and fts_score is not None else "N/A"
            vec_str = f"rank {vec_rank} (score {vec_score:.4f})" if vec_rank is not None and vec_score is not None else "N/A"
            rrf_str = f"rank {rrf_rank} (score {rrf_score:.4f})" if rrf_rank is not None and rrf_score is not None else "N/A"

            print(f"   {c.CYAN}↳ Ranking Details -> FTS: {fts_str} | Vector: {vec_str} | RRF: {rrf_str}{c.RESET}")
        print()

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        print(f"{c.YELLOW}[... Truncated {omitted} remaining result chunk(s) to protect context ...]{c.RESET}\n")


def format_doc_results_cli(grouped_results: List[Dict], query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Prints results grouped by document with combined snippets."""
    if session_id:
        stats_str = ""
        if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
            c_count = exclusion_stats["excluded_chunks"]
            d_count = exclusion_stats.get("excluded_docs", 0)
            stats_str = f" | Excluded {c_count} previously seen chunk(s) across {d_count} document(s)"
        print(f"{c.DIM}[Session: {session_id}{stats_str}]{c.RESET}")

    if not grouped_results:
        print(f"\n{c.RED}No documents found.{c.RESET}")
        return

    print(f"\n{c.DIM}Found {len(grouped_results)} documents:{c.RESET}\n")

    for i, doc in enumerate(grouped_results):
        rank_str = f"{i+1}."
        path_str = f"qmd://{doc['collection']}/{doc['path']}" if doc['collection'] else doc['path']
        
        print(f"{c.GREEN}{rank_str}{c.RESET} {c.BOLD}{doc['title']}{c.RESET} {c.DIM}({path_str}){c.RESET}")

        meta_parts = []
        if doc.get("chunks"):
            first_chunk = doc["chunks"][0]
            if first_chunk.get("alt_title"): meta_parts.append(f"Alt: {first_chunk.get('alt_title')}")
            if first_chunk.get("doc_type"): meta_parts.append(f"Type: {first_chunk.get('doc_type')}")
            if first_chunk.get("doc_date"): meta_parts.append(f"Date: {first_chunk.get('doc_date')}")
            if first_chunk.get("authors"): meta_parts.append(f"Authors: {', '.join(first_chunk.get('authors'))}")
        
        if meta_parts:
            print(f"   {c.CYAN}{' | '.join(meta_parts)}{c.RESET}")

        print(f"   {c.DIM}Max Score: {doc['score']:.4f}{c.RESET}")

        if verbose and doc.get("chunks"):
            for chunk_data in doc["chunks"]:
                fts_rank = chunk_data.get('fts_rank')
                fts_score = chunk_data.get('fts_score')
                vec_rank = chunk_data.get('vec_rank')
                vec_score = chunk_data.get('vec_score')
                rrf_rank = chunk_data.get('rrf_rank')
                rrf_score = chunk_data.get('rrf_score')

                fts_str = f"rank {fts_rank} (score {fts_score:.4f})" if fts_rank is not None and fts_score is not None else "N/A"
                vec_str = f"rank {vec_rank} (score {vec_score:.4f})" if vec_rank is not None and vec_score is not None else "N/A"
                rrf_str = f"rank {rrf_rank} (score {rrf_score:.4f})" if rrf_rank is not None and rrf_score is not None else "N/A"

                print(f"   {c.CYAN}↳ Chunk seq {chunk_data['seq_id']} -> Score: {chunk_data['score']:.4f} | FTS: {fts_str} | Vector: {vec_str} | RRF: {rrf_str}{c.RESET}")

        # Separator line
        print(f"   {c.DIM}--- Matches ---{c.RESET}")
        
        rendered_blocks = []
        for snip in doc['snippets']:
            if snip.startswith("(...") and snip.endswith("...)"):
                rendered_blocks.append(f"   {c.DIM}{snip}{c.RESET}")
            else:
                indented = "\n".join("   " + line for line in snip.split("\n"))
                rendered_blocks.append(highlight_keywords(indented, query))
        
        print("\n\n".join(rendered_blocks) + "\n")

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        print(f"{c.YELLOW}[... Truncated {omitted} remaining result chunk(s) to protect context ...]{c.RESET}\n")


def format_discover_cli(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Prints discovered top-hit document results (1 per document, SERP style)."""
    if session_id:
        stats_str = ""
        if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
            c_count = exclusion_stats["excluded_chunks"]
            d_count = exclusion_stats.get("excluded_docs", 0)
            stats_str = f" | Excluded {c_count} previously seen chunk(s) across {d_count} document(s)"
        print(f"{c.DIM}[Session: {session_id}{stats_str}]{c.RESET}")

    if not results:
        print(f"\n{c.RED}No documents discovered.{c.RESET}")
        return

    print(f"\n{c.DIM}Discovered {len(results)} document{'s' if len(results) != 1 else ''}:{c.RESET}\n")

    for i, res in enumerate(results):
        rank_str = f"{i+1}."
        path_str = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        header_str = f" {c.CYAN}[{res.headers}]{c.RESET}" if getattr(res, 'headers', None) else ""
        match_count = getattr(res, "match_count", 1)
        matches_badge = f" {c.MAGENTA}({match_count} match{'es' if match_count != 1 else ''} in doc){c.RESET}" if match_count > 1 else ""

        print(f"{c.GREEN}{rank_str}{c.RESET} {c.BOLD}{res.title}{c.RESET}{header_str}{matches_badge} {c.DIM}({path_str}){c.RESET}")

        meta_parts = []
        if getattr(res, 'alt_title', None): meta_parts.append(f"Alt: {res.alt_title}")
        if getattr(res, 'doc_type', None): meta_parts.append(f"Type: {res.doc_type}")
        if getattr(res, 'doc_date', None): meta_parts.append(f"Date: {res.doc_date}")
        if getattr(res, 'authors', None): meta_parts.append(f"Authors: {', '.join(res.authors)}")
        if meta_parts:
            print(f"   {c.CYAN}{' | '.join(meta_parts)}{c.RESET}")

        snippet = res.text[:2000].replace("\n", " ").strip()
        if len(res.text) > 2000:
            snippet += "..."
        highlighted = highlight_keywords(snippet, query)
        print(f"   {highlighted}")

        print(f"   {c.DIM}Score: {res.score:.4f} | Source: {res.source} | Top Chunk Seq: {res.seq_id} | Matches: {match_count}{c.RESET}")
        if verbose:
            fts_rank = getattr(res, 'fts_rank', None)
            fts_score = getattr(res, 'fts_score', None)
            vec_rank = getattr(res, 'vec_rank', None)
            vec_score = getattr(res, 'vec_score', None)
            rrf_rank = getattr(res, 'rrf_rank', None)
            rrf_score = getattr(res, 'rrf_score', None)

            fts_str = f"rank {fts_rank} (score {fts_score:.4f})" if fts_rank is not None and fts_score is not None else "N/A"
            vec_str = f"rank {vec_rank} (score {vec_score:.4f})" if vec_rank is not None and vec_score is not None else "N/A"
            rrf_str = f"rank {rrf_rank} (score {rrf_score:.4f})" if rrf_rank is not None and rrf_score is not None else "N/A"

            print(f"   {c.CYAN}↳ Ranking Details -> FTS: {fts_str} | Vector: {vec_str} | RRF: {rrf_str}{c.RESET}")
        print()

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        print(f"{c.YELLOW}[... Truncated {omitted} remaining document(s) to protect context ...]{c.RESET}\n")


def format_outline_cli(outline: Dict):
    """Prints document heading outline and chunk mapping."""
    if not outline:
        print(f"\n{c.RED}No outline available.{c.RESET}")
        return

    path_str = f"qmd://{outline['collection']}/{outline['path']}" if outline.get('collection') else outline['path']
    print(f"\n{c.BOLD}{outline['title']}{c.RESET} {c.DIM}({path_str}){c.RESET}")
    
    depth_info = ""
    if outline.get('depth') is not None and outline.get('max_depth') is not None:
        depth = outline['depth']
        max_d = outline['max_depth']
        if outline.get('has_more_depth'):
            depth_info = f" | Depth: {depth}/{max_d} (use -d to expand)"
        else:
            depth_info = f" | Depth: {depth}"

    print(f"{c.DIM}Total Chunks: {outline['total_chunks']} | Total Chars: {outline['total_chars']}{depth_info}{c.RESET}\n")

    for h in outline.get('headings', []):
        indent = "  " * (h['level'] - 1)
        level_hashes = "#" * h['level']
        seq_str = f"[seq: {h['start_seq']}-{h['end_seq']}]" if h['start_seq'] != h['end_seq'] else f"[seq: {h['start_seq']}]"
        print(f"{indent}{c.CYAN}{level_hashes}{c.RESET} {c.BOLD}{h['text']}{c.RESET} {c.YELLOW}{seq_str}{c.RESET} {c.DIM}({h['char_count']} chars){c.RESET}")

    if outline.get('has_more_depth'):
        next_d = (outline.get('depth') or 1) + 1
        print(f"\n{c.DIM}[... Deeper headings omitted (showing depth {outline.get('depth')} of {outline.get('max_depth')}). Use -d {next_d} or -d {outline.get('max_depth')} to expand ...]{c.RESET}")

    print()


def format_chunks_cli(results: List, window: int = 0, truncation_info: Optional[Dict] = None):
    """Prints retrieved chunk(s) with context window headers."""
    if not results:
        print(f"\n{c.RED}No chunks found.{c.RESET}")
        return

    distinct_docs = {(r.collection, r.path) for r in results}
    if len(distinct_docs) == 1:
        res0 = results[0]
        path_str = f"qmd://{res0.collection}/{res0.path}" if res0.collection else res0.path
        print(f"\n{c.BOLD}{res0.title}{c.RESET} {c.DIM}({path_str}){c.RESET}")
        print(f"{c.DIM}Retrieved {len(results)} chunk(s) (window: ±{window}){c.RESET}\n")

        for res in results:
            hdr_str = f" {c.CYAN}[{res.headers}]{c.RESET}" if getattr(res, 'headers', None) else ""
            print(f"{c.GREEN}Chunk {res.seq_id}{c.RESET}{hdr_str}")
            print(f"{res.text}\n")

        if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
            omitted = truncation_info["omitted_remaining"]
            resume_cmd = truncation_info.get("resume_cmd", "")
            print(f"{c.YELLOW}[... Truncated {omitted} remaining chunk(s) to protect context. Next page: {resume_cmd} ...]{c.RESET}\n")
    else:
        print(f"\n{c.DIM}Retrieved {len(results)} chunk(s) across {len(distinct_docs)} documents (window: ±{window}){c.RESET}\n")
        current_doc = None
        for res in results:
            doc_key = (res.collection, res.path)
            if doc_key != current_doc:
                current_doc = doc_key
                path_str = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
                print(f"{c.BOLD}{res.title}{c.RESET} {c.DIM}({path_str}){c.RESET}")

            hdr_str = f" {c.CYAN}[{res.headers}]{c.RESET}" if getattr(res, 'headers', None) else ""
            print(f"{c.GREEN}Chunk {res.seq_id}{c.RESET}{hdr_str}")
            print(f"{res.text}\n")

        if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
            omitted = truncation_info["omitted_remaining"]
            resume_cmd = truncation_info.get("resume_cmd", "")
            print(f"{c.YELLOW}[... Truncated {omitted} remaining chunk(s) to protect context. Next page: {resume_cmd} ...]{c.RESET}\n")


def format_report_cli(results: List[Dict[str, Any]]):
    """Prints document analysis reports (handles both nested 'analysis' format and flattened custom reports)."""
    if not results:
        print(f"{c.YELLOW}No analysis reports found for the specified documents.{c.RESET}")
        return

    print(f"\n{c.CYAN}--- Document Analysis Reports ---{c.RESET}")
    for res in results:
        path_str = res.get('path', 'Unknown')
        coll_str = res.get('collection', 'None')
        print(f"\n{c.GREEN}File: {path_str}{c.RESET} (Collection: {coll_str})")
        
        # Support backward compatibility with nested 'analysis' objects or flattened dynamic fields
        data = res.get('analysis', res)
        
        for k, v in data.items():
            if k in ('path', 'collection', 'hash', 'analysis'):
                continue
            
            if isinstance(v, list) and v:
                print(f"  {c.BOLD}{k.title()}:{c.RESET}")
                for item in v:
                    # Truncate long list items slightly so CLI isn't destroyed
                    snippet = str(item).replace("\n", " ").strip()
                    if len(snippet) > 120: snippet = snippet[:117] + "..."
                    print(f"    - {snippet}")
            elif v and isinstance(v, str):
                # Handle multi-line strings gracefully
                if "\n" in v:
                    print(f"  {c.BOLD}{k.title()}:{c.RESET}")
                    for line in v.split("\n"):
                        print(f"    {line.strip()}")
                else:
                    print(f"  {c.BOLD}{k.title()}:{c.RESET} {v}")
            elif v:
                print(f"  {c.BOLD}{k.title()}:{c.RESET} {v}")
                
    print(f"\n{c.CYAN}---------------------------------{c.RESET}")


def format_map_cli(tree_data: List[Dict], query: str = ""):
    """Prints the semantic relevance map tree in ASCII format, scaled to a Density Score."""
    if not tree_data:
        print(f"\n{c.RED}No matching regions found.{c.RESET}")
        return

    print(f"\n{c.DIM}Semantic density map for:{c.RESET} {query}\n")

    # 1. Collect and rank directories
    all_dirs = []
    def _collect_dirs(node, coll_name):
        path_val = node.get('path', '')
        path_str = f"qmd://{coll_name}/{path_val}".rstrip('/')
        
        # Count files directly inside this directory
        file_count = sum(1 for child in node.get("children", []) if child.get("type") == "file")
        
        all_dirs.append({
            "path": path_str,
            "score": node.get("score", 0.0) * 100.0,
            "file_count": file_count
        })
        
        for child in node.get("children", []):
            if child.get("type") == "directory":
                _collect_dirs(child, coll_name)

    for item in tree_data:
        _collect_dirs(item.get("tree", {}), item.get("collection", ""))

    # Deduplicate and sort
    unique_dirs = list({d["path"]: d for d in all_dirs}.values())
    top_dirs = sorted(unique_dirs, key=lambda x: x["score"], reverse=True)
    top_dirs = [d for d in top_dirs if d["score"] > 0]

    if top_dirs:
        print(f"{c.BOLD}{c.YELLOW}Top 10 Directories by Relevance Density:{c.RESET}")
        for i, d in enumerate(top_dirs[:10]):
            rank_str = f"{i+1}."
            score_str = f"[{d['score']:.1f}]"
            file_str = f"({d['file_count']} matching files)" if d['file_count'] > 0 else ""
            print(f"  {c.GREEN}{rank_str:<3}{c.RESET} {c.YELLOW}{score_str:>6}{c.RESET} {c.CYAN}{d['path']}{c.RESET} {c.DIM}{file_str}{c.RESET}")
        print("\n" + c.DIM + "-" * 60 + c.RESET + "\n")

    # 2. Render Tree (Directories only)
    for item_idx, item in enumerate(tree_data):
        if item_idx > 0:
            print()
        coll_name = item.get("collection", "")
        root_node = item.get("tree", {})
        coll_score = item.get("score", 0.0) * 100.0
        if not root_node:
            continue

        print(f"{c.BOLD}{c.CYAN}{coll_name}/{c.RESET} {c.YELLOW}[Density: {coll_score:.1f}]{c.RESET}")

        def _render_node(node: Dict, prefix: str = ""):
            # Filter to only directories to make the map scannable
            children = [child for child in node.get("children", []) if child.get("type") == "directory"]
            count = len(children)
            
            for i, child in enumerate(children):
                is_last = (i == count - 1)
                connector = "└── " if is_last else "├── "
                sub_prefix = "    " if is_last else "│   "

                c_name = child.get("name", "")
                score_val = child.get("score", 0.0) * 100.0
                score_str = f" {c.YELLOW}[{score_val:.1f}]{c.RESET}"
                
                child_direct_files = sum(1 for c_ in child.get("children", []) if c_.get("type") == "file")
                file_str = f" {c.DIM}({child_direct_files} files){c.RESET}" if child_direct_files > 0 else ""
                
                print(f"{prefix}{c.DIM}{connector}{c.RESET}{c.BOLD}{c.CYAN}{c_name}/{c.RESET}{score_str}{file_str}")
                _render_node(child, prefix + sub_prefix)

        _render_node(root_node, "")
    print()


def format_collection_tree_cli(tree_data: Union[Dict, List[Dict]]):
    """Prints collection folder directory tree in ASCII format."""
    if not tree_data:
        print(f"\n{c.RED}No collections or documents found.{c.RESET}")
        return

    items = tree_data if isinstance(tree_data, list) else [tree_data]

    for item_idx, item in enumerate(items):
        if item_idx > 0:
            print()
        coll_name = item.get("collection", "")
        root_node = item.get("tree", {})
        if not root_node:
            continue

        print(f"{c.BOLD}{c.CYAN}{coll_name}/{c.RESET}")

        def _render_node(node: Dict, prefix: str = ""):
            children = node.get("children", [])
            count = len(children)
            for i, child in enumerate(children):
                is_last = (i == count - 1)
                connector = "└── " if is_last else "├── "
                sub_prefix = "    " if is_last else "│   "

                c_type = child.get("type", "file")
                c_name = child.get("name", "")
                if c_type == "directory":
                    print(f"{prefix}{c.DIM}{connector}{c.RESET}{c.BOLD}{c.CYAN}{c_name}/{c.RESET}")
                    _render_node(child, prefix + sub_prefix)
                else:
                    title = child.get("title")
                    title_str = f" {c.DIM}({title}){c.RESET}" if title and title != c_name else ""
                    print(f"{prefix}{c.DIM}{connector}{c.RESET}{c.GREEN}{c_name}{c.RESET}{title_str}")

        _render_node(root_node, "")
    print()


def format_extraction_cli(data: Dict):
    """Prints smart extraction results."""
    outline = data.get("outline", {})
    metadata = data.get("metadata", {})
    chunks = data.get("chunks", [])
    
    path_str = f"qmd://{outline.get('collection')}/{outline.get('path')}" if outline.get('collection') else outline.get('path', '')
    title = metadata.get("title") or outline.get("title") or path_str
    
    print(f"\n{c.BOLD}Extraction: {title}{c.RESET} {c.DIM}({path_str}){c.RESET}")
    print(f"{c.DIM}Selected {len(chunks)} of {data.get('total_chunks')} chunks to meet budget constraints.{c.RESET}\n")
    
    if metadata:
        print(f"{c.CYAN}--- Document Metadata ---{c.RESET}")
        if metadata.get("altTitle"): print(f"  {c.BOLD}Alt Title:{c.RESET} {metadata['altTitle']}")
        if metadata.get("doc_type"): print(f"  {c.BOLD}Type:{c.RESET} {metadata['doc_type']}")
        if metadata.get("authors"): print(f"  {c.BOLD}Authors:{c.RESET} {', '.join(metadata['authors'])}")
        if metadata.get("summary"): print(f"  {c.BOLD}Summary:{c.RESET} {metadata['summary']}")
        print(f"{c.CYAN}-------------------------{c.RESET}\n")

    if outline.get("headings"):
        print(f"{c.BOLD}Document Outline:{c.RESET}")
        for h in outline["headings"]:
            indent = "  " * (h['level'] - 1)
            seq_str = f"[seq: {h['start_seq']}-{h['end_seq']}]" if h['start_seq'] != h['end_seq'] else f"[seq: {h['start_seq']}]"
            print(f"{indent}{c.CYAN}{'#' * h['level']}{c.RESET} {h['text']} {c.YELLOW}{seq_str}{c.RESET}")
        print()

    print(f"{c.BOLD}Extracted Content:{c.RESET}\n")
    
    blocks = []
    current_block = []
    for res in chunks:
        if not current_block:
            current_block.append(res)
        elif res.seq_id == current_block[-1].seq_id + 1:
            current_block.append(res)
        else:
            blocks.append(current_block)
            current_block = [res]
    if current_block:
        blocks.append(current_block)
        
    prev_end_seq = None
    for block in blocks:
        start_seq = block[0].seq_id
        end_seq = block[-1].seq_id
        
        if prev_end_seq is not None:
            gap = start_seq - prev_end_seq - 1
            if gap > 0:
                print(f"{c.YELLOW}[... {gap} chunks omitted ...]{c.RESET}\n")
                
        first_header = next((getattr(ck, 'headers', '') for ck in block if getattr(ck, 'headers', '')), "")
                
        hdr_str = f" {c.CYAN}[{first_header}]{c.RESET}" if first_header else ""
        seq_label = f"Chunks {start_seq}-{end_seq}" if start_seq != end_seq else f"Chunk {start_seq}"
        
        print(f"{c.GREEN}{seq_label}{c.RESET}{hdr_str}")
        print("\n\n".join(ck.text for ck in block) + "\n")
        
        prev_end_seq = end_seq