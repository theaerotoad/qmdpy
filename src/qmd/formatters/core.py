import re
from typing import List, Any
from .colors import c

def set_plain_mode(enabled: bool = True):
    c.set_plain_mode(enabled)

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def escape_xml_attr(val: Any) -> str:
    """Escapes strings for safe inclusion in XML attribute values and strips ANSI."""
    if val is None:
        return ""
    clean = strip_ansi(str(val))
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
    return clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def escape_xml_text(val: Any) -> str:
    """Escapes strings for safe inclusion in XML text nodes, strips ANSI and invalid XML chars."""
    if val is None:
        return ""
    clean = strip_ansi(str(val))
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
    return clean.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

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
    if not query or not text or c.PLAIN_MODE:
        return text

    terms = extract_lexical_terms(query)
    if not terms:
        return text

    # Sort longer terms first so multi-word phrases match before individual sub-words
    keywords = sorted(set(terms), key=len, reverse=True)
    pattern = re.compile(f"({'|'.join(re.escape(k) for k in keywords)})", re.IGNORECASE)
    return pattern.sub(f"{c.YELLOW}{c.BOLD}\\1{c.RESET}", text)