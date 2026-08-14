import sys
import re

from typing import List, Dict, Optional

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
