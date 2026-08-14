import sys
import re
import html
from typing import List, Dict, Optional, Tuple, Any

PLAIN_MODE = False

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
    return html.escape(clean, quote=True)

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

def format_results_cli(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
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

def format_doc_results_cli(grouped_results: List[Dict], query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
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

def format_results_json(results: List, verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
    """Outputs results as JSON for piping."""
    import json
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

def format_doc_results_json(grouped_results: List[Dict], session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
    """Outputs document-grouped results as JSON for piping."""
    import json
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

def format_results_xml(results: List, query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
    """Outputs flat search results as XML for LLM context."""
    query_attr = escape_xml_attr(query)
    total_matches = len(results)
    session_attr = f' session_id="{escape_xml_attr(session_id)}"' if session_id else ""
    excl_attr = ""
    if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
        excl_attr = f' excluded_chunks="{exclusion_stats["excluded_chunks"]}" excluded_docs="{exclusion_stats.get("excluded_docs", 0)}"'

    lines = [f'<search_results query="{query_attr}" total_matches="{total_matches}"{session_attr}{excl_attr}>']

    for i, res in enumerate(results):
        rank = res.rank if res.rank is not None else (i + 1)
        score_str = f"{res.score:.4f}"
        seq = res.seq_id
        prev_seq = max(0, seq - 1) if seq > 0 else 0
        next_seq = seq + 1
        doc_uri = f"qmd://{res.collection}/{res.path}" if res.collection else res.path
        section = getattr(res, 'headers', '') or ""

        res_tag = (
            f'  <result\n'
            f'    rank="{rank}"\n'
            f'    score="{score_str}"\n'
            f'    document="{escape_xml_attr(doc_uri)}"\n'
            f'    seq="{seq}"\n'
            f'    title="{escape_xml_attr(res.title)}"\n'
            f'    section="{escape_xml_attr(section)}"\n'
            f'    prev_seq="{prev_seq}"\n'
            f'    next_seq="{next_seq}"'
        )
        if verbose:
            if getattr(res, 'fts_score', None) is not None:
                res_tag += f'\n    fts_score="{res.fts_score:.4f}" fts_rank="{res.fts_rank}"'
            if getattr(res, 'vec_score', None) is not None:
                res_tag += f'\n    vec_score="{res.vec_score:.4f}" vec_rank="{res.vec_rank}"'
            if getattr(res, 'rrf_score', None) is not None:
                res_tag += f'\n    rrf_score="{res.rrf_score:.4f}" rrf_rank="{res.rrf_rank}"'
        res_tag += '>'

        clean_text = strip_ansi(res.text).strip()
        lines.append(res_tag)
        lines.append(clean_text)
        lines.append('  </result>')

    lines.append('</search_results>')
    print("\n".join(lines))

def format_doc_results_xml(grouped_results: List[Dict], query: str = "", verbose: bool = False, session_id: Optional[str] = None, exclusion_stats: Optional[Dict] = None):
    """Outputs document-grouped results as structured XML in document-sequential order."""
    if not grouped_results:
        print('<search_results query="" total_matches="0" total_documents="0">\n</search_results>')
        return

    query_attr = escape_xml_attr(query)
    total_matches = sum(len(d.get("chunks", [])) for d in grouped_results)
    total_docs = len(grouped_results)
    session_attr = f' session_id="{escape_xml_attr(session_id)}"' if session_id else ""
    excl_attr = ""
    if isinstance(exclusion_stats, dict) and exclusion_stats.get("excluded_chunks", 0) > 0:
        excl_attr = f' excluded_chunks="{exclusion_stats["excluded_chunks"]}" excluded_docs="{exclusion_stats.get("excluded_docs", 0)}"'

    lines = [f'<search_results query="{query_attr}" total_matches="{total_matches}" total_documents="{total_docs}"{session_attr}{excl_attr}>']

    for doc in grouped_results:
        doc_uri = f"qmd://{doc['collection']}/{doc['path']}" if doc.get('collection') else doc.get('path', '')
        title_attr = escape_xml_attr(doc.get('title', ''))
        lines.append(f'  <document uri="{escape_xml_attr(doc_uri)}" title="{title_attr}">')

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
                    lines.append(f'    <gap omitted_chunks="{seq}" from_seq="0" to_seq="{seq - 1}" />')
            else:
                gap = seq - prev_end_seq - 1
                if gap > 0:
                    lines.append(f'    <gap omitted_chunks="{gap}" from_seq="{prev_end_seq + 1}" to_seq="{seq - 1}" />')

            rank = c.get("rank")
            rank_attr = f' rank="{rank}"' if rank is not None else ""
            score_attr = f' score="{c["score"]:.4f}"' if "score" in c else ""
            section_attr = f' section="{escape_xml_attr(c.get("headers", ""))}"' if c.get("headers") else ""

            lines.append(f'    <chunk seq="{seq}"{rank_attr}{score_attr}{section_attr}>')
            lines.append(strip_ansi(c.get("text", "")).strip())
            lines.append('    </chunk>')
            prev_end_seq = seq

        lines.append('  </document>')

    lines.append('</search_results>')
    print("\n".join(lines))

def format_chunks_xml(results: List, window: int = 0):
    """Outputs retrieved chunks in XML format."""
    if not results:
        print('<document>\n</document>')
        return

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
        lines.append(f'<document uri="{escape_xml_attr(uri)}" title="{escape_xml_attr(title)}">')

        sorted_chunks = sorted(chunk_list, key=lambda x: x.seq_id)
        prev_seq = None
        for res in sorted_chunks:
            seq = res.seq_id
            if prev_seq is not None:
                gap = seq - prev_seq - 1
                if gap > 0:
                    lines.append(f'  <gap omitted_chunks="{gap}" from_seq="{prev_seq + 1}" to_seq="{seq - 1}" />')

            sec_attr = f' section="{escape_xml_attr(res.headers)}"' if getattr(res, 'headers', None) else ""
            lines.append(f'  <chunk seq="{seq}"{sec_attr}>')
            lines.append(strip_ansi(res.text).strip())
            lines.append('  </chunk>')
            prev_seq = seq

        lines.append('</document>')

    print("\n".join(lines))

def format_outline_xml(outline: Dict):
    """Outputs document heading outline as XML."""
    if not outline:
        print('<outline>\n</outline>')
        return

    uri = f"qmd://{outline['collection']}/{outline['path']}" if outline.get('collection') else outline.get('path', '')
    title = escape_xml_attr(outline.get('title', ''))
    total_chunks = outline.get('total_chunks', 0)
    total_chars = outline.get('total_chars', 0)

    lines = [
        f'<outline uri="{escape_xml_attr(uri)}" title="{title}" total_chunks="{total_chunks}" total_chars="{total_chars}">'
    ]
    for h in outline.get('headings', []):
        level = h.get('level', 1)
        start_seq = h.get('start_seq', 0)
        end_seq = h.get('end_seq', 0)
        char_count = h.get('char_count', 0)
        text = escape_xml_attr(h.get('text', ''))
        lines.append(
            f'  <heading level="{level}" start_seq="{start_seq}" end_seq="{end_seq}" char_count="{char_count}">{text}</heading>'
        )
    lines.append('</outline>')
    print("\n".join(lines))

def format_chunks_cli(results: List, window: int = 0):
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
