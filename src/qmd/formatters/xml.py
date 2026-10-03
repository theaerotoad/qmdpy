from typing import List, Dict, Optional, Tuple, Any, Union
from .core import escape_xml_attr, escape_xml_text, strip_ansi

MAX_GAP_EXPAND_CHUNKS = 3

def format_results_xml(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, seen_chunks: Optional[int] = None, truncation_info: Optional[Dict] = None, print_output: bool = True):
    """Outputs flat search results as XML for LLM context."""
    query_attr = escape_xml_attr(query)
    total_matches = len(results)
    session_attr = f' session_id="{escape_xml_attr(session_id)}"' if session_id else ""
    seen_attr = f' seen_chunks="{seen_chunks}"' if seen_chunks is not None and session_id else ""
    hint_attr = f' next_query_hint="--session {escape_xml_attr(session_id)}"' if session_id else ""
    excl_attr = ""
    if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
        excl_attr = f' excluded_chunks="{exclusion_stats["excluded_chunks"]}" excluded_docs="{exclusion_stats.get("excluded_docs", 0)}"'

    lines = [f'<search_results query="{query_attr}" total_matches="{total_matches}"{session_attr}{seen_attr}{hint_attr}{excl_attr}>']

    for i, res in enumerate(results):
        rank = res.rank if res.rank is not None else (i + 1)
        score_str = f"{res.score:.4f}"
        seq = res.seq_id
        prev_seq = max(0, seq - 1) if seq > 0 else 0
        next_seq = seq + 1
        doc_uri = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        section = getattr(res, 'headers', '') or ""
        coll_attr = f' collection="{escape_xml_attr(res.collection)}"' if res.collection else ""
        path_attr = f' path="{escape_xml_attr(res.path)}"' if res.path else ""

        target_ref = f"{res.collection}:{res.path}:{seq}" if res.collection else f"{res.path}:{seq}"
        outline_ref = f"{res.collection}:{res.path}" if res.collection else f"{res.path}"
        read_cmd = f"qmd read '{target_ref}'"
        outline_cmd = f"qmd outline '{outline_ref}'"

        clean_text = strip_ansi(res.text).strip()
        chars = len(clean_text)

        res_tag = (
            f'  <result\n'
            f'    rank="{rank}"\n'
            f'    score="{score_str}"\n'
            f'    document="{escape_xml_attr(doc_uri)}"{coll_attr}{path_attr}\n'
        )

        if getattr(res, 'alt_title', None): res_tag += f'    alt_title="{escape_xml_attr(res.alt_title)}"\n'
        if getattr(res, 'doc_type', None): res_tag += f'    doc_type="{escape_xml_attr(res.doc_type)}"\n'
        if getattr(res, 'doc_date', None): res_tag += f'    doc_date="{escape_xml_attr(res.doc_date)}"\n'
        if getattr(res, 'authors', None): res_tag += f'    authors="{escape_xml_attr(", ".join(res.authors))}"\n'

        res_tag += (
            f'    seq="{seq}"\n'
            f'    chars="{chars}"\n'
            f'    title="{escape_xml_attr(res.title)}"\n'
            f'    section="{escape_xml_attr(section)}"\n'
            f'    prev_seq="{prev_seq}"\n'
            f'    next_seq="{next_seq}"\n'
            f'    read="{escape_xml_attr(read_cmd)}"\n'
            f'    outline="{escape_xml_attr(outline_cmd)}"'
        )
        if verbose:
            if getattr(res, 'fts_score', None) is not None:
                res_tag += f'\n    fts_score="{res.fts_score:.4f}" fts_rank="{res.fts_rank}"'
            if getattr(res, 'vec_score', None) is not None:
                res_tag += f'\n    vec_score="{res.vec_score:.4f}" vec_rank="{res.vec_rank}"'
            if getattr(res, 'rrf_score', None) is not None:
                res_tag += f'\n    rrf_score="{res.rrf_score:.4f}" rrf_rank="{res.rrf_rank}"'
        res_tag += '>'

        lines.append(res_tag)
        lines.append(escape_xml_text(clean_text))
        lines.append('  </result>')

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        limit_val = truncation_info.get("limit", len(results))
        lines.append(f'  <truncation omitted_chunks="{omitted}" reason="max_chunks_per_response limit ({limit_val}) reached" />')

    lines.append('</search_results>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_doc_results_xml(grouped_results: List[Dict], query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, seen_chunks: Optional[int] = None, truncation_info: Optional[Dict] = None, print_output: bool = True):
    """Outputs document-grouped results as structured XML in document-sequential order."""
    if not grouped_results:
        output = '<search_results query="" total_matches="0" total_documents="0">\n</search_results>'
        if print_output:
            print(output)
        return output

    query_attr = escape_xml_attr(query)
    total_matches = sum(len(d.get("chunks", [])) for d in grouped_results)
    total_docs = len(grouped_results)
    session_attr = f' session_id="{escape_xml_attr(session_id)}"' if session_id else ""
    seen_attr = f' seen_chunks="{seen_chunks}"' if seen_chunks is not None and session_id else ""
    hint_attr = f' next_query_hint="--session {escape_xml_attr(session_id)}"' if session_id else ""
    excl_attr = ""
    if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
        excl_attr = f' excluded_chunks="{exclusion_stats["excluded_chunks"]}" excluded_docs="{exclusion_stats.get("excluded_docs", 0)}"'

    lines = [f'<search_results query="{query_attr}" total_matches="{total_matches}" total_documents="{total_docs}"{session_attr}{seen_attr}{hint_attr}{excl_attr}>']

    for doc in grouped_results:
        coll = doc.get('collection', '') or ""
        path = doc.get('path', '') or ""
        doc_uri = f"qmd://{coll}/{path}" if coll else path
        title_attr = escape_xml_attr(doc.get('title', ''))
        coll_attr = f' collection="{escape_xml_attr(coll)}"' if coll else ""
        path_attr = f' path="{escape_xml_attr(path)}"' if path else ""
        
        meta_attrs = ""
        if doc.get("chunks"):
            first_chunk = doc["chunks"][0]
            if first_chunk.get("alt_title"): meta_attrs += f' alt_title="{escape_xml_attr(first_chunk["alt_title"])}"'
            if first_chunk.get("doc_type"): meta_attrs += f' doc_type="{escape_xml_attr(first_chunk["doc_type"])}"'
            if first_chunk.get("doc_date"): meta_attrs += f' doc_date="{escape_xml_attr(first_chunk["doc_date"])}"'
            if first_chunk.get("authors"): meta_attrs += f' authors="{escape_xml_attr(", ".join(first_chunk["authors"]))}"'

        lines.append(f'  <document uri="{escape_xml_attr(doc_uri)}"{coll_attr}{path_attr} title="{title_attr}"{meta_attrs}>')

        outline_ref = f"{coll}:{path}" if coll else path
        doc_outline_cmd = f"qmd outline '{outline_ref}'"

        raw_chunks = doc.get("chunks", [])
        sorted_chunks = sorted(raw_chunks, key=lambda x: int(x.get("seq_id", 0)))

        unique_chunks = []
        seen_seq = set()
        for c in sorted_chunks:
            sid = c.get("seq_id", 0)
            if sid not in seen_seq:
                seen_seq.add(sid)
                unique_chunks.append(c)

        prev_end_seq = None
        for c in unique_chunks:
            seq = c.get("seq_id", 0)
            if prev_end_seq is None:
                if seq > 0:
                    gap_len = seq
                    gap_to = seq - 1
                    gap_range = f"0-{gap_to}" if gap_to > 0 else "0"
                    gap_ref = f"{coll}:{path}:{gap_range}" if coll else f"{path}:{gap_range}"
                    expand_attr = f' expand="qmd read \'{escape_xml_attr(gap_ref)}\'"' if gap_len <= MAX_GAP_EXPAND_CHUNKS else ""
                    lines.append(f'    <gap omitted_chunks="{gap_len}" from_seq="0" to_seq="{gap_to}"{expand_attr} />')
            else:
                gap = seq - prev_end_seq - 1
                if gap > 0:
                    gap_from = prev_end_seq + 1
                    gap_to = seq - 1
                    gap_range = f"{gap_from}-{gap_to}" if gap_from != gap_to else f"{gap_from}"
                    gap_ref = f"{coll}:{path}:{gap_range}" if coll else f"{path}:{gap_range}"
                    expand_attr = f' expand="qmd read \'{escape_xml_attr(gap_ref)}\'"' if gap <= MAX_GAP_EXPAND_CHUNKS else ""
                    lines.append(f'    <gap omitted_chunks="{gap}" from_seq="{gap_from}" to_seq="{gap_to}"{expand_attr} />')

            rank = c.get("rank")
            rank_attr = f' rank="{rank}"' if rank is not None else ""
            score_attr = f' score="{c["score"]:.4f}"' if "score" in c else ""
            section_attr = f' section="{escape_xml_attr(c.get("headers", ""))}"' if c.get("headers") else ""

            chunk_text = strip_ansi(c.get("text", "")).strip()
            chars = len(chunk_text)
            chars_attr = f' chars="{chars}"'

            chunk_ref = f"{coll}:{path}:{seq}" if coll else f"{path}:{seq}"
            read_attr = f' read="qmd read \'{escape_xml_attr(chunk_ref)}\'"'
            outline_attr = f' outline="{escape_xml_attr(doc_outline_cmd)}"'

            lines.append(f'    <chunk seq="{seq}"{rank_attr}{score_attr}{chars_attr}{section_attr}{read_attr}{outline_attr}>')
            lines.append(escape_xml_text(chunk_text))
            lines.append('    </chunk>')
            prev_end_seq = seq

        lines.append('  </document>')

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        limit_val = truncation_info.get("limit", sum(len(d.get("chunks", [])) for d in grouped_results))
        lines.append(f'  <truncation omitted_chunks="{omitted}" reason="max_chunks_per_response limit ({limit_val}) reached" />')

    lines.append('</search_results>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_discover_xml(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, seen_chunks: Optional[int] = None, truncation_info: Optional[Dict] = None, print_output: bool = True) -> str:
    """Outputs discovered top-hit document results as XML for LLM context."""
    query_attr = escape_xml_attr(query)
    total_docs = len(results)
    session_attr = f' session_id="{escape_xml_attr(session_id)}"' if session_id else ""
    seen_attr = f' seen_chunks="{seen_chunks}"' if seen_chunks is not None and session_id else ""
    hint_attr = f' next_query_hint="--session {escape_xml_attr(session_id)}"' if session_id else ""
    excl_attr = ""
    if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
        excl_attr = f' excluded_chunks="{exclusion_stats["excluded_chunks"]}" excluded_docs="{exclusion_stats.get("excluded_docs", 0)}"'

    lines = [f'<discover_results query="{query_attr}" total_documents="{total_docs}"{session_attr}{seen_attr}{hint_attr}{excl_attr}>']

    for i, res in enumerate(results):
        rank = res.rank if res.rank is not None else (i + 1)
        score_str = f"{res.score:.4f}"
        seq = res.seq_id
        doc_uri = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        section = getattr(res, 'headers', '') or ""
        coll_attr = f' collection="{escape_xml_attr(res.collection)}"' if res.collection else ""
        path_attr = f' path="{escape_xml_attr(res.path)}"' if res.path else ""
        match_count = getattr(res, "match_count", 1)

        target_ref = f"{res.collection}:{res.path}:{seq}" if res.collection else f"{res.path}:{seq}"
        outline_ref = f"{res.collection}:{res.path}" if res.collection else f"{res.path}"
        read_cmd = f"qmd read '{target_ref}'"
        outline_cmd = f"qmd outline '{outline_ref}'"
        
        search_filter_path = f" -c '{escape_xml_attr(res.collection)}'" if res.collection else ""
        search_filter_path += f" -p '{escape_xml_attr(res.path)}'"
        search_cmd = f"qmd search \"{escape_xml_attr(query)}\"{search_filter_path}"

        clean_text = strip_ansi(res.text).strip()
        chars = len(clean_text)

        res_tag = (
            f'  <document\n'
            f'    rank="{rank}"\n'
            f'    score="{score_str}"\n'
            f'    uri="{escape_xml_attr(doc_uri)}"{coll_attr}{path_attr}\n'
            f'    title="{escape_xml_attr(res.title)}"\n'
        )

        if getattr(res, 'alt_title', None): res_tag += f'    alt_title="{escape_xml_attr(res.alt_title)}"\n'
        if getattr(res, 'doc_type', None): res_tag += f'    doc_type="{escape_xml_attr(res.doc_type)}"\n'
        if getattr(res, 'doc_date', None): res_tag += f'    doc_date="{escape_xml_attr(res.doc_date)}"\n'
        if getattr(res, 'authors', None): res_tag += f'    authors="{escape_xml_attr(", ".join(res.authors))}"\n'

        res_tag += (
            f'    top_chunk_seq="{seq}"\n'
            f'    match_count="{match_count}"\n'
            f'    chars="{chars}"\n'
            f'    section="{escape_xml_attr(section)}"\n'
            f'    read="{escape_xml_attr(read_cmd)}"\n'
            f'    outline="{escape_xml_attr(outline_cmd)}"\n'
            f'    search="{escape_xml_attr(search_cmd)}"'
        )
        if verbose:
            if getattr(res, 'fts_score', None) is not None:
                res_tag += f'\n    fts_score="{res.fts_score:.4f}" fts_rank="{res.fts_rank}"'
            if getattr(res, 'vec_score', None) is not None:
                res_tag += f'\n    vec_score="{res.vec_score:.4f}" vec_rank="{res.vec_rank}"'
            if getattr(res, 'rrf_score', None) is not None:
                res_tag += f'\n    rrf_score="{res.rrf_score:.4f}" rrf_rank="{res.rrf_rank}"'
        res_tag += '>'

        lines.append(res_tag)
        lines.append(escape_xml_text(clean_text))
        lines.append('  </document>')

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        limit_val = truncation_info.get("limit", len(results))
        lines.append(f'  <truncation omitted_documents="{omitted}" reason="max_chunks_per_response limit ({limit_val}) reached" />')

    lines.append('</discover_results>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_chunks_xml(results: List, window: int = 0, truncation_info: Optional[Dict] = None, print_output: bool = True):
    """Outputs retrieved chunks in XML format."""
    if not results:
        output = '<document>\n</document>'
        if print_output:
            print(output)
        return output

    docs: Dict[Tuple[str, str], List] = {}
    doc_titles: Dict[Tuple[str, str], str] = {}
    for res in results:
        key = (res.collection or "", res.path)
        if key not in docs:
            docs[key] = []
            doc_titles[key] = res.title
        docs[key].append(res)

    lines = []
    for key, chunk_list in docs.items():
        coll, path = key
        uri = f"qmd://{coll}/{path}" if coll else path
        title = doc_titles[key]
        coll_attr = f' collection="{escape_xml_attr(coll)}"' if coll else ""
        path_attr = f' path="{escape_xml_attr(path)}"' if path else ""
        lines.append(f'<document uri="{escape_xml_attr(uri)}"{coll_attr}{path_attr} title="{escape_xml_attr(title)}">')

        sorted_chunks = sorted(chunk_list, key=lambda x: int(x.seq_id))
        prev_seq = None
        for res in sorted_chunks:
            seq = res.seq_id
            if prev_seq is not None:
                gap = seq - prev_seq - 1
                if gap > 0:
                    gap_from = prev_seq + 1
                    gap_to = seq - 1
                    gap_range = f"{gap_from}-{gap_to}" if gap_from != gap_to else f"{gap_from}"
                    gap_ref = f"{coll}:{path}:{gap_range}" if coll else f"{path}:{gap_range}"
                    expand_attr = f' expand="qmd read \'{escape_xml_attr(gap_ref)}\'"' if gap <= MAX_GAP_EXPAND_CHUNKS else ""
                    lines.append(f'  <gap omitted_chunks="{gap}" from_seq="{gap_from}" to_seq="{gap_to}"{expand_attr} />')

            clean_text = strip_ansi(res.text).strip()
            chars = len(clean_text)
            chars_attr = f' chars="{chars}"'
            sec_attr = f' section="{escape_xml_attr(res.headers)}"' if getattr(res, 'headers', None) else ""
            lines.append(f'  <chunk seq="{seq}"{chars_attr}{sec_attr}>')
            lines.append(escape_xml_text(clean_text))
            lines.append('  </chunk>')
            prev_seq = seq

        if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
            omitted = truncation_info["omitted_remaining"]
            limit = truncation_info.get("limit", len(results))
            resume_cmd = escape_xml_attr(truncation_info.get("resume_cmd", ""))
            lines.append(f'  <truncation omitted_chunks="{omitted}" reason="max_chunks_per_response limit ({limit}) reached" resume="{resume_cmd}" />')

        lines.append('</document>')

    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def _build_heading_tree(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds a tree of heading nodes from a flat list of headings with hierarchy levels."""
    root_nodes: List[Dict[str, Any]] = []
    stack: List[Tuple[int, Dict[str, Any]]] = []

    for h in headings:
        node = {
            "level": h.get("level", 1),
            "text": h.get("text", ""),
            "start_seq": h.get("start_seq", 0),
            "end_seq": h.get("end_seq", 0),
            "char_count": h.get("char_count", 0),
            "children": []
        }
        while stack and stack[-1][0] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root_nodes.append(node)

        stack.append((node["level"], node))

    return root_nodes

def _render_heading_node_xml(node: Dict[str, Any], indent_level: int, lines: List[str]):
    """Recursively formats heading tree nodes into compact, token-efficient XML."""
    indent = "  " * indent_level
    level = node.get("level", 1)
    start_seq = node.get("start_seq", 0)
    end_seq = node.get("end_seq", 0)
    chars = node.get("char_count", 0)
    text = escape_xml_attr(node.get("text", ""))
    seq_str = f"{start_seq}-{end_seq}" if start_seq != end_seq else f"{start_seq}"
    children = node.get("children", [])

    if children:
        lines.append(f'{indent}<section level="{level}" title="{text}" seq="{seq_str}" chars="{chars}">')
        for child in children:
            _render_heading_node_xml(child, indent_level + 1, lines)
        lines.append(f'{indent}</section>')
    else:
        lines.append(f'{indent}<section level="{level}" title="{text}" seq="{seq_str}" chars="{chars}" />')

def format_outline_xml(outline: Dict, print_output: bool = True) -> str:
    """Outputs document heading outline as a token-efficient XML tree for LLM context."""
    if not outline:
        output = '<outline>\n</outline>'
        if print_output:
            print(output)
        return output

    coll = outline.get('collection', '') or ""
    path = outline.get('path', '') or ""
    uri = f"qmd://{coll}/{path}" if coll else path
    title = escape_xml_attr(outline.get('title', ''))
    total_chunks = outline.get('total_chunks', 0)
    total_chars = outline.get('total_chars', 0)
    coll_attr = f' collection="{escape_xml_attr(coll)}"' if coll else ""
    path_attr = f' path="{escape_xml_attr(path)}"' if path else ""

    depth = outline.get('depth')
    max_depth = outline.get('max_depth')
    has_more_depth = outline.get('has_more_depth', False)
    if depth is not None and max_depth is not None and depth < max_depth:
        has_more_depth = True

    depth_attr = f' depth="{depth}"' if depth is not None else ""
    max_depth_attr = f' max_depth="{max_depth}"' if max_depth is not None else ""
    more_attr = ' more_levels_available="true"' if has_more_depth else ""

    lines = [
        f'<outline uri="{escape_xml_attr(uri)}"{coll_attr}{path_attr} title="{title}" total_chunks="{total_chunks}" total_chars="{total_chars}"{depth_attr}{max_depth_attr}{more_attr}>'
    ]

    headings = outline.get('headings', [])
    tree_nodes = _build_heading_tree(headings)
    for node in tree_nodes:
        _render_heading_node_xml(node, 1, lines)

    if has_more_depth:
        next_depth = (depth + 1) if isinstance(depth, int) else 2
        lines.append(f'  <more_levels current_depth="{depth}" max_depth="{max_depth}" hint="Use --depth {next_depth} or higher to view deeper headings" />')

    lines.append('</outline>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_report_xml(results: List[Dict[str, Any]], print_output: bool = True) -> str:
    """Outputs document analysis reports as XML for LLM context."""
    if not results:
        output = '<analysis_reports>\n</analysis_reports>'
        if print_output: print(output)
        return output

    lines = ['<analysis_reports>']
    for res in results:
        coll = res.get('collection', '') or ""
        path = res.get('path', '') or ""
        uri = f"qmd://{coll}/{path}" if coll else path
        
        coll_attr = f' collection="{escape_xml_attr(coll)}"' if coll else ""
        path_attr = f' path="{escape_xml_attr(path)}"' if path else ""
        lines.append(f'  <report uri="{escape_xml_attr(uri)}"{coll_attr}{path_attr}>')
        
        # Support backward compatibility with nested 'analysis' objects or flattened dynamic fields
        data = res.get('analysis', res)
        
        for k, v in data.items():
            if k in ('path', 'collection', 'hash', 'analysis'):
                continue
                
            # Replace spaces with underscores for valid XML tags
            safe_tag = k.replace(" ", "_").lower()
            
            if isinstance(v, list) and v:
                lines.append(f'    <{safe_tag}>')
                for item in v:
                    lines.append(f'      <item>{escape_xml_text(str(item))}</item>')
                lines.append(f'    </{safe_tag}>')
            elif v:
                lines.append(f'    <{safe_tag}>{escape_xml_text(str(v))}</{safe_tag}>')
                
        lines.append('  </report>')
        
    lines.append('</analysis_reports>')
    output = "\n".join(lines)
    if print_output: print(output)
    return output

def format_map_xml(tree_data: List[Dict], query: str = "", print_output: bool = True):
    """Outputs semantic map tree as XML for LLM context, focusing on density scores."""
    if not tree_data:
        output = '<map_results>\n</map_results>'
        if print_output:
            print(output)
        return output

    query_attr = escape_xml_attr(query)
    lines = [f'<map_results query="{query_attr}">']

    all_dirs = []
    def _collect_dirs(node, coll_name):
        path_val = node.get('path', '')
        path_str = f"qmd://{coll_name}/{path_val}".rstrip('/')
        file_count = sum(1 for c in node.get("children", []) if c.get("type") == "file")
        
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

    unique_dirs = list({d["path"]: d for d in all_dirs}.values())
    top_dirs = sorted(unique_dirs, key=lambda x: x["score"], reverse=True)
    top_dirs = [d for d in top_dirs if d["score"] > 0]
    
    if top_dirs:
        lines.append('  <top_directories>')
        for d in top_dirs[:10]:
            lines.append(f'    <directory path="{escape_xml_attr(d["path"])}" density_score="{d["score"]:.1f}" matching_files="{d["file_count"]}" />')
        lines.append('  </top_directories>')

    def _node_to_xml(node: Dict, indent_level: int):
        indent = "  " * indent_level
        children = node.get("children", [])
        for child in children:
            c_type = child.get("type", "file")
            c_name = escape_xml_attr(child.get("name", ""))
            score = child.get("score", 0.0) * 100.0
            
            if c_type == "directory":
                c_children = child.get("children", [])
                if c_children:
                    lines.append(f'{indent}<directory name="{c_name}" density_score="{score:.1f}">')
                    _node_to_xml(child, indent_level + 1)
                    lines.append(f'{indent}</directory>')
                else:
                    lines.append(f'{indent}<directory name="{c_name}" density_score="{score:.1f}" />')
            else:
                title_attr = f' title="{escape_xml_attr(child.get("title", ""))}"' if child.get("title") else ""
                path_attr = f' path="{escape_xml_attr(child.get("path", ""))}"' if child.get("path") else ""
                lines.append(f'{indent}<file name="{c_name}"{title_attr}{path_attr} density_score="{score:.1f}" />')

    for item in tree_data:
        coll_name = escape_xml_attr(item.get("collection", ""))
        coll_score = item.get("score", 0.0) * 100.0
        indent_base = 1
        indent = "  " * indent_base
        root_node = item.get("tree", {})
        lines.append(f'{indent}<collection_tree collection="{coll_name}" density_score="{coll_score:.1f}">')
        _node_to_xml(root_node, indent_base + 1)
        lines.append(f'{indent}</collection_tree>')

    lines.append('</map_results>')

    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_collection_tree_xml(tree_data: Union[Dict, List[Dict]], print_output: bool = True):
    """Outputs collection folder tree as XML for LLM context."""
    if not tree_data:
        output = '<collections>\n</collections>'
        if print_output:
            print(output)
        return output

    items = tree_data if isinstance(tree_data, list) else [tree_data]
    lines = []
    
    wrap_all = len(items) > 1
    if wrap_all:
        lines.append('<collections>')

    def _node_to_xml(node: Dict, indent_level: int):
        indent = "  " * indent_level
        children = node.get("children", [])
        for child in children:
            c_type = child.get("type", "file")
            c_name = escape_xml_attr(child.get("name", ""))
            if c_type == "directory":
                c_children = child.get("children", [])
                if c_children:
                    lines.append(f'{indent}<directory name="{c_name}">')
                    _node_to_xml(child, indent_level + 1)
                    lines.append(f'{indent}</directory>')
                else:
                    lines.append(f'{indent}<directory name="{c_name}" />')
            else:
                title_attr = f' title="{escape_xml_attr(child.get("title", ""))}"' if child.get("title") else ""
                path_attr = f' path="{escape_xml_attr(child.get("path", ""))}"' if child.get("path") else ""
                doc_id = child.get("doc_id")
                id_attr = f' doc_id="{doc_id}"' if doc_id is not None else ""
                lines.append(f'{indent}<file name="{c_name}"{title_attr}{path_attr}{id_attr} />')

    for item in items:
        coll_name = escape_xml_attr(item.get("collection", ""))
        indent_base = 1 if wrap_all else 0
        indent = "  " * indent_base
        root_node = item.get("tree", {})
        lines.append(f'{indent}<collection_tree collection="{coll_name}">')
        _node_to_xml(root_node, indent_base + 1)
        lines.append(f'{indent}</collection_tree>')

    if wrap_all:
        lines.append('</collections>')

    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_extraction_xml(data: Dict, print_output: bool = True) -> str:
    """Outputs smart extraction results as XML."""
    outline = data.get("outline", {})
    metadata = data.get("metadata", {})
    chunks = data.get("chunks", [])
    total_chunks = data.get("total_chunks", 0)
    
    coll = outline.get("collection", "")
    path = outline.get("path", "")
    uri = f"qmd://{coll}/{path}" if coll else path
    
    lines = []
    lines.append(f'<extracted_document uri="{escape_xml_attr(uri)}" total_chunks="{total_chunks}" selected_chunks="{len(chunks)}">')
    
    if metadata:
        lines.append('  <metadata>')
        for k, v in metadata.items():
            if v and isinstance(v, list):
                v = ", ".join(v)
            if v and isinstance(v, str):
                lines.append(f'    <{k}>{escape_xml_text(v)}</{k}>')
        lines.append('  </metadata>')
        
    if outline:
        lines.append('  <outline>')
        for h in outline.get("headings", []):
            level = h.get("level", 1)
            text = escape_xml_attr(h.get("text", ""))
            start_seq = h.get("start_seq", 0)
            end_seq = h.get("end_seq", 0)
            seq_str = f"{start_seq}-{end_seq}" if start_seq != end_seq else f"{start_seq}"
            lines.append(f'    <section level="{level}" title="{text}" seq="{seq_str}" />')
        lines.append('  </outline>')
        
    lines.append('  <content>')
    
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
                gap_from = prev_end_seq + 1
                gap_to = start_seq - 1
                gap_range = f"{gap_from}-{gap_to}" if gap_from != gap_to else f"{gap_from}"
                gap_ref = f"{coll}:{path}:{gap_range}" if coll else f"{path}:{gap_range}"
                expand_attr = f' expand="qmd read \'{escape_xml_attr(gap_ref)}\'"' if gap <= MAX_GAP_EXPAND_CHUNKS else ""
                lines.append(f'    <gap omitted_chunks="{gap}" from_seq="{gap_from}" to_seq="{gap_to}"{expand_attr} />')
                
        clean_texts = [strip_ansi(c.text).strip() for c in block]
        combined_text = "\n\n".join(clean_texts)
        chars = len(combined_text)
        
        first_header = next((getattr(ck, 'headers', '') for ck in block if getattr(ck, 'headers', '')), "")
        sec_attr = f' section="{escape_xml_attr(first_header)}"' if first_header else ""
        
        seq_attr = f' seq="{start_seq}-{end_seq}"' if start_seq != end_seq else f' seq="{start_seq}"'
        
        lines.append(f'    <chunk{seq_attr} chars="{chars}"{sec_attr}>')
        lines.append(escape_xml_text(combined_text))
        lines.append('    </chunk>')
        
        prev_end_seq = end_seq
        
    lines.append('  </content>')
    lines.append('</extracted_document>')
    
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output