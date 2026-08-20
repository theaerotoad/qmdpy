import sys
import re
import html
import json
from typing import List, Dict, Optional, Tuple, Any, Union

PLAIN_MODE = False
MAX_GAP_EXPAND_CHUNKS = 3

# ANSI Escape Codes
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# Foreground Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
DIM = "\033[90m"

def set_plain_mode(enabled: bool = True):
    global PLAIN_MODE, RESET, BOLD, UNDERLINE, CYAN, GREEN, YELLOW, RED, MAGENTA, DIM
    PLAIN_MODE = enabled
    if enabled:
        RESET = ""
        BOLD = ""
        UNDERLINE = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
        MAGENTA = ""
        DIM = ""

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def escape_xml_attr(val: Any) -> str:
    """Escapes strings for safe inclusion in XML attribute values and strips ANSI."""
    if val is None:
        return ""
    clean = strip_ansi(str(val))
    return clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def extract_lexical_terms(query: str) -> List[str]:
    """Extracts lexical (FTS) search terms from query or FTS expression."""
    if not query or not query.strip():
        return []

    collected_terms = set()

    def _add_term(t: str):
        if not t:
            return
        cleaned = t.strip('"\'()[]{}').strip()
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", 
            "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", 
            "that", "the", "their", "then", "there", "these", "they", "this", 
            "to", "was", "will", "with", "near"
        }
        if cleaned and cleaned.lower() not in stop_words and len(cleaned) > 1:
            collected_terms.add(cleaned)

    def _flatten(obj):
        if isinstance(obj, str):
            s = obj.strip('"\'()[]{}').strip()
            if not s:
                return
            if ' ' in s:
                _add_term(s)
            quoted = re.findall(r'"([^"]+)"', s)
            for q in quoted:
                _add_term(q)
            unquoted = re.sub(r'"[^"]+"', ' ', s)
            for word in re.findall(r'\b\w+\b', unquoted):
                _add_term(word)
        elif isinstance(obj, dict):
            for v in obj.values():
                _flatten(v)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _flatten(item)

    _flatten(query)

    try:
        from qmd.utils import extract_tiered_fts_terms
        tiered = extract_tiered_fts_terms(query)
        _flatten(tiered)
    except Exception:
        pass

    return list(collected_terms)

def highlight_keywords(text: str, query: str) -> str:
    """Highlights Lexical (FTS) terms in text using ANSI bold/yellow."""
    if not query or not text or PLAIN_MODE:
        return text

    terms = extract_lexical_terms(query)
    if not terms:
        return text

    # Sort longer terms first so multi-word phrases match before individual sub-words
    keywords = sorted(set(terms), key=len, reverse=True)
    pattern = re.compile(f"({'|'.join(re.escape(k) for k in keywords)})", re.IGNORECASE)
    return pattern.sub(f"{YELLOW}{BOLD}\\1{RESET}", text)

def format_results_cli(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Prints standard search results (snippets)."""
    if session_id:
        stats_str = ""
        if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
            c_count = exclusion_stats["excluded_chunks"]
            d_count = exclusion_stats.get("excluded_docs", 0)
            stats_str = f" | Excluded {c_count} previously seen chunk(s) across {d_count} document(s)"
        print(f"{DIM}[Session: {session_id}{stats_str}]{RESET}")

    if not results:
        print(f"\n{RED}No results found.{RESET}")
        return

    print(f"\n{DIM}Found {len(results)} chunks:{RESET}\n")
    
    for i, res in enumerate(results):
        rank_str = f"{i+1}."
        path_str = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        header_str = f" {CYAN}[{res.headers}]{RESET}" if getattr(res, 'headers', None) else ""
        
        print(f"{GREEN}{rank_str}{RESET} {BOLD}{res.title}{RESET}{header_str} {DIM}({path_str}){RESET}")
        
        snippet = res.text[:2000].replace("\n", " ") + "..."
        highlighted = highlight_keywords(snippet, query)
        print(f"   {highlighted}")
        
        print(f"   {DIM}Score: {res.score:.4f} | Source: {res.source} | Rank: {res.rank}{RESET}")
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

            print(f"   {CYAN}↳ Ranking Details -> FTS: {fts_str} | Vector: {vec_str} | RRF: {rrf_str}{RESET}")
        print()

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        print(f"{YELLOW}[... Truncated {omitted} remaining result chunk(s) to protect context ...]{RESET}\n")

def format_doc_results_cli(grouped_results: List[Dict], query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None, truncation_info: Optional[Dict] = None):
    """Prints results grouped by document with combined snippets."""
    if session_id:
        stats_str = ""
        if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
            c_count = exclusion_stats["excluded_chunks"]
            d_count = exclusion_stats.get("excluded_docs", 0)
            stats_str = f" | Excluded {c_count} previously seen chunk(s) across {d_count} document(s)"
        print(f"{DIM}[Session: {session_id}{stats_str}]{RESET}")

    if not grouped_results:
        print(f"\n{RED}No documents found.{RESET}")
        return

    print(f"\n{DIM}Found {len(grouped_results)} documents:{RESET}\n")

    for i, doc in enumerate(grouped_results):
        rank_str = f"{i+1}."
        path_str = f"qmd://{doc['collection']}/{doc['path']}" if doc['collection'] else doc['path']
        
        print(f"{GREEN}{rank_str}{RESET} {BOLD}{doc['title']}{RESET} {DIM}({path_str}){RESET}")
        print(f"   {DIM}Max Score: {doc['score']:.4f}{RESET}")

        if verbose and doc.get("chunks"):
            for c in doc["chunks"]:
                fts_rank = c.get('fts_rank')
                fts_score = c.get('fts_score')
                vec_rank = c.get('vec_rank')
                vec_score = c.get('vec_score')
                rrf_rank = c.get('rrf_rank')
                rrf_score = c.get('rrf_score')

                fts_str = f"rank {fts_rank} (score {fts_score:.4f})" if fts_rank is not None and fts_score is not None else "N/A"
                vec_str = f"rank {vec_rank} (score {vec_score:.4f})" if vec_rank is not None and vec_score is not None else "N/A"
                rrf_str = f"rank {rrf_rank} (score {rrf_score:.4f})" if rrf_rank is not None and rrf_score is not None else "N/A"

                print(f"   {CYAN}↳ Chunk seq {c['seq_id']} -> Score: {c['score']:.4f} | FTS: {fts_str} | Vector: {vec_str} | RRF: {rrf_str}{RESET}")

        # Separator line
        print(f"   {DIM}--- Matches ---{RESET}")
        
        rendered_blocks = []
        for snip in doc['snippets']:
            if snip.startswith("(...") and snip.endswith("...)"):
                rendered_blocks.append(f"   {DIM}{snip}{RESET}")
            else:
                indented = "\n".join("   " + line for line in snip.split("\n"))
                rendered_blocks.append(highlight_keywords(indented, query))
        
        print("\n\n".join(rendered_blocks) + "\n")

    if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
        omitted = truncation_info["omitted_remaining"]
        print(f"{YELLOW}[... Truncated {omitted} remaining result chunk(s) to protect context ...]{RESET}\n")

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
            "headers": getattr(res, "headers", "")
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

def format_outline_cli(outline: Dict):
    """Prints document heading outline and chunk mapping."""
    if not outline:
        print(f"\n{RED}No outline available.{RESET}")
        return

    path_str = f"qmd://{outline['collection']}/{outline['path']}" if outline.get('collection') else outline['path']
    print(f"\n{BOLD}{outline['title']}{RESET} {DIM}({path_str}){RESET}")
    print(f"{DIM}Total Chunks: {outline['total_chunks']} | Total Chars: {outline['total_chars']}{RESET}\n")

    for h in outline.get('headings', []):
        indent = "  " * (h['level'] - 1)
        level_hashes = "#" * h['level']
        seq_str = f"[seq: {h['start_seq']}-{h['end_seq']}]" if h['start_seq'] != h['end_seq'] else f"[seq: {h['start_seq']}]"
        print(f"{indent}{CYAN}{level_hashes}{RESET} {BOLD}{h['text']}{RESET} {YELLOW}{seq_str}{RESET} {DIM}({h['char_count']} chars){RESET}")
    print()

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
        lines.append(clean_text)
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
        lines.append(f'  <document uri="{escape_xml_attr(doc_uri)}"{coll_attr}{path_attr} title="{title_attr}">')

        outline_ref = f"{coll}:{path}" if coll else path
        doc_outline_cmd = f"qmd outline '{outline_ref}'"

        raw_chunks = doc.get("chunks", [])
        sorted_chunks = sorted(raw_chunks, key=lambda x: x.get("seq_id", 0))

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
            lines.append(chunk_text)
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

        sorted_chunks = sorted(chunk_list, key=lambda x: x.seq_id)
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
            lines.append(clean_text)
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

def format_outline_xml(outline: Dict, print_output: bool = True):
    """Outputs document heading outline as XML."""
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

    lines = [
        f'<outline uri="{escape_xml_attr(uri)}"{coll_attr}{path_attr} title="{title}" total_chunks="{total_chunks}" total_chars="{total_chars}">'
    ]
    for h in outline.get('headings', []):
        level = h.get('level', 1)
        start_seq = h.get('start_seq', 0)
        end_seq = h.get('end_seq', 0)
        char_count = h.get('char_count', 0)
        text = escape_xml_attr(h.get('text', ''))
        seq_range = f"{start_seq}-{end_seq}" if start_seq != end_seq else f"{start_seq}"
        read_ref = f"{coll}:{path}:{seq_range}" if coll else f"{path}:{seq_range}"
        read_attr = f' read="qmd read \'{escape_xml_attr(read_ref)}\'"'
        lines.append(
            f'  <heading level="{level}" start_seq="{start_seq}" end_seq="{end_seq}" char_count="{char_count}"{read_attr}>{text}</heading>'
        )
    lines.append('</outline>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output

def format_chunks_cli(results: List, window: int = 0, truncation_info: Optional[Dict] = None):
    """Prints retrieved chunk(s) with context window headers."""
    if not results:
        print(f"\n{RED}No chunks found.{RESET}")
        return

    distinct_docs = {(r.collection, r.path) for r in results}
    if len(distinct_docs) == 1:
        res0 = results[0]
        path_str = f"qmd://{res0.collection}/{res0.path}" if res0.collection else res0.path
        print(f"\n{BOLD}{res0.title}{RESET} {DIM}({path_str}){RESET}")
        print(f"{DIM}Retrieved {len(results)} chunk(s) (window: ±{window}){RESET}\n")

        for res in results:
            hdr_str = f" {CYAN}[{res.headers}]{RESET}" if getattr(res, 'headers', None) else ""
            print(f"{GREEN}Chunk {res.seq_id}{RESET}{hdr_str}")
            print(f"{res.text}\n")

        if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
            omitted = truncation_info["omitted_remaining"]
            resume_cmd = truncation_info.get("resume_cmd", "")
            print(f"{YELLOW}[... Truncated {omitted} remaining chunk(s) to protect context. Next page: {resume_cmd} ...]{RESET}\n")
    else:
        print(f"\n{DIM}Retrieved {len(results)} chunk(s) across {len(distinct_docs)} documents (window: ±{window}){RESET}\n")
        current_doc = None
        for res in results:
            doc_key = (res.collection, res.path)
            if doc_key != current_doc:
                current_doc = doc_key
                path_str = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
                print(f"{BOLD}{res.title}{RESET} {DIM}({path_str}){RESET}")

            hdr_str = f" {CYAN}[{res.headers}]{RESET}" if getattr(res, 'headers', None) else ""
            print(f"{GREEN}Chunk {res.seq_id}{RESET}{hdr_str}")
            print(f"{res.text}\n")

        if truncation_info and truncation_info.get("omitted_remaining", 0) > 0:
            omitted = truncation_info["omitted_remaining"]
            resume_cmd = truncation_info.get("resume_cmd", "")
            print(f"{YELLOW}[... Truncated {omitted} remaining chunk(s) to protect context. Next page: {resume_cmd} ...]{RESET}\n")

def format_collection_tree_cli(tree_data: Union[Dict, List[Dict]]):
    """Prints collection folder directory tree in ASCII format."""
    if not tree_data:
        print(f"\n{RED}No collections or documents found.{RESET}")
        return

    items = tree_data if isinstance(tree_data, list) else [tree_data]

    for item_idx, item in enumerate(items):
        if item_idx > 0:
            print()
        coll_name = item.get("collection", "")
        root_node = item.get("tree", {})
        if not root_node:
            continue

        print(f"{BOLD}{CYAN}{coll_name}/{RESET}")

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
                    print(f"{prefix}{DIM}{connector}{RESET}{BOLD}{CYAN}{c_name}/{RESET}")
                    _render_node(child, prefix + sub_prefix)
                else:
                    title = child.get("title")
                    title_str = f" {DIM}({title}){RESET}" if title and title != c_name else ""
                    print(f"{prefix}{DIM}{connector}{RESET}{GREEN}{c_name}{RESET}{title_str}")

        _render_node(root_node, "")
    print()

def format_collection_tree_json(tree_data: Union[Dict, List[Dict]]):
    """Outputs collection folder directory tree as JSON."""
    print(json.dumps(tree_data, indent=2))

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

def format_grep_cli(results: List[Dict], pattern: str = ""):
    """Prints grep pattern search results with line numbers and matches."""
    if not results:
        print(f"\n{RED}No matching patterns found.{RESET}")
        return

    print(f"\n{DIM}Found {len(results)} match{'es' if len(results) != 1 else ''}:{RESET}\n")

    current_doc = None
    for res in results:
        doc_key = (res.get("collection", ""), res.get("path", ""))
        if doc_key != current_doc:
            current_doc = doc_key
            coll = res.get("collection", "")
            path = res.get("path", "")
            title = res.get("title", "")
            uri = f"qmd://{coll}/{path}" if coll else path
            title_str = f" {BOLD}{title}{RESET}" if title and title != path else ""
            print(f"{GREEN}{uri}{RESET}{title_str}")

        line_no = res.get("line_number", 0)
        line_text = res.get("line_text", "")
        highlighted_line = line_text

        if not PLAIN_MODE and res.get("match_text"):
            m_text = re.escape(res["match_text"])
            highlighted_line = re.sub(f"({m_text})", f"{YELLOW}{BOLD}\\1{RESET}", line_text)

        print(f"  {CYAN}{line_no:4d}:{RESET} {highlighted_line}")
    print()

def format_grep_json(results: List[Dict]):
    """Outputs grep pattern search results as JSON."""
    print(json.dumps(results, indent=2))

def format_grep_xml(results: List[Dict], pattern: str = "", is_regex: bool = False, case_sensitive: bool = False, print_output: bool = True):
    """Outputs grep pattern search results as structured XML for LLM context."""
    pattern_attr = escape_xml_attr(pattern)
    regex_attr = 'true' if is_regex else 'false'
    case_attr = 'true' if case_sensitive else 'false'
    total_matches = len(results)

    if not results:
        output = f'<grep_results pattern="{pattern_attr}" is_regex="{regex_attr}" case_sensitive="{case_attr}" total_matches="0">\n</grep_results>'
        if print_output:
            print(output)
        return output

    docs: Dict[Tuple[str, str], List[Dict]] = {}
    doc_titles: Dict[Tuple[str, str], str] = {}
    for res in results:
        key = (res.get("collection", ""), res.get("path", ""))
        if key not in docs:
            docs[key] = []
            doc_titles[key] = res.get("title", "")
        docs[key].append(res)

    lines = [
        f'<grep_results pattern="{pattern_attr}" is_regex="{regex_attr}" case_sensitive="{case_attr}" total_matches="{total_matches}" total_documents="{len(docs)}">'
    ]

    for key, match_list in docs.items():
        coll, path = key
        uri = f"qmd://{coll}/{path}" if coll else path
        title = doc_titles[key]
        lines.append(f'  <document uri="{escape_xml_attr(uri)}" title="{escape_xml_attr(title)}">')

        for m in match_list:
            line_no = m.get("line_number", 0)
            line_text = html.escape(strip_ansi(m.get("line_text", "")))
            lines.append(f'    <match line="{line_no}">{line_text}</match>')

        lines.append('  </document>')

    lines.append('</grep_results>')
    output = "\n".join(lines)
    if print_output:
        print(output)
    return output