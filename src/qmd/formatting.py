import sys
import re
from typing import List, Dict

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

def highlight_keywords(text: str, query: str) -> str:
    """Highlights query terms in text using ANSI bold/yellow."""
    if not query:
        return text
        
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", 
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", 
        "that", "the", "their", "then", "there", "these", "they", "this", 
        "to", "was", "will", "with"
    }
    
    keywords = [re.escape(k) for k in query.split() if len(k) > 2 and k.lower() not in stop_words]
    if not keywords:
        keywords = [re.escape(query)]
        
    pattern = re.compile(f"({'|'.join(keywords)})", re.IGNORECASE)
    return pattern.sub(f"{YELLOW}{BOLD}\\1{RESET}", text)

def format_results_cli(results: List, query: str = "", verbose: bool = False):
    """Prints standard search results (snippets)."""
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

def format_doc_results_cli(grouped_results: List[Dict], query: str = "", verbose: bool = False):
    """Prints results grouped by document with combined snippets."""
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

def format_results_json(results: List, verbose: bool = False):
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
        if verbose or getattr(res, "fts_rank", None) is not None or getattr(res, "vec_rank", None) is not None:
            item["fts_score"] = getattr(res, "fts_score", None)
            item["fts_rank"] = getattr(res, "fts_rank", None)
            item["vec_score"] = getattr(res, "vec_score", None)
            item["vec_rank"] = getattr(res, "vec_rank", None)
            item["rrf_score"] = getattr(res, "rrf_score", None)
            item["rrf_rank"] = getattr(res, "rrf_rank", None)
        data.append(item)
    print(json.dumps(data, indent=2))

def format_doc_results_json(grouped_results: List[Dict]):
    """Outputs document-grouped results as JSON for piping."""
    import json
    print(json.dumps(grouped_results, indent=2))
