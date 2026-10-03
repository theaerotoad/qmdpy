import os
import re
from typing import List, Dict, Tuple
from qmd.store import Result

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

    chunk_snips = sorted([s for s in normalized if s[0] >= 0], key=lambda x: int(x[0]))
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
    Groups chunks by title (to merge duplicate files), keeps max score, and merges text into reading order.
    """
    docs = {}
    for r in results:
        # Group by title to seamlessly merge identical books across different collections/paths
        key = r.title.strip().lower() if r.title else (r.collection, r.path)
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
                "doc_date": getattr(r, "doc_date", None),
                "alt_title": getattr(r, "alt_title", None),
                "doc_type": getattr(r, "doc_type", None),
                "authors": getattr(r, "authors", None),
                "text": r.text
            })
            
    output_list = []
    for d in docs.values():
        d["snippets"] = merge_overlapping_snippets(d["raw_snippets"], doc_title=d["title"])
        del d["raw_snippets"]
        output_list.append(d)
        
    return sorted(output_list, key=lambda x: x['score'], reverse=True)