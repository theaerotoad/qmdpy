import hashlib
import re
from typing import List, Dict, Optional, Any, Union, Tuple

_SPACY_MODELS: Dict[str, Any] = {}

def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """
    Lazy loads and caches the spaCy model.
    Returns None if spaCy or the specified model is unavailable.
    """
    global _SPACY_MODELS
    if model_name in _SPACY_MODELS:
        return _SPACY_MODELS[model_name]

    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError:
            try:
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
            except Exception:
                nlp = None
        _SPACY_MODELS[model_name] = nlp
        return nlp
    except Exception:
        _SPACY_MODELS[model_name] = None
        return None

GENERIC_META_NOUNS = {
    "feeling", "feelings", "feel", "feels", "felt",
    "thing", "things",
    "way", "ways",
    "reason", "reasons",
    "idea", "ideas",
    "problem", "problems",
    "type", "types",
    "kind", "kinds",
    "stuff",
    "part", "parts",
    "something", "anything", "nothing", "everything",
    "example", "examples",
    "fact", "facts",
    "info", "information",
    "question", "questions",
    "answer", "answers",
    "detail", "details",
    "issue", "issues",
    "item", "items",
    "result", "results",
    "case", "cases",
    "time", "times"
}

def extract_tiered_fts_terms(query: str, model_name: str = "en_core_web_sm") -> Dict[str, List[str]]:
    """
    Uses spaCy to extract tiered FTS terms from a query:
      1. Primary Terms: Named Entities & Acronyms
      2. Secondary Terms: High-IDF Nouns, Adjectives, Verbs (filtering out generic meta-nouns).
    """
    nlp = get_spacy_nlp(model_name)
    if nlp is None:
        return {"primary": [], "secondary": []}

    doc = nlp(query)

    # 1. Primary Terms: Named Entities & Acronyms
    primary_entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]

    # Track token indices used by entities
    used_token_indices = {i for ent in doc.ents for i in range(ent.start, ent.end)}

    valid_pos = {"NOUN", "PROPN", "ADJ", "VERB", "NUM"}

    def _is_meaningful_term(tok) -> bool:
        text_lower = tok.text.lower().strip()
        lemma_lower = tok.lemma_.lower().strip()
        if tok.is_stop or tok.is_punct or len(text_lower) <= 1:
            return False
        if text_lower in GENERIC_META_NOUNS or lemma_lower in GENERIC_META_NOUNS:
            return False
        return True

    # 2. Secondary Terms: Noun Chunks (excluding stopwords, entity tokens, and generic meta-nouns)
    secondary_chunks = []
    for chunk in doc.noun_chunks:
        chunk_indices = set(range(chunk.start, chunk.end))

        if chunk_indices.intersection(used_token_indices):
            continue

        clean_tokens = [tok.text for tok in chunk if _is_meaningful_term(tok) and tok.pos_ in valid_pos]
        if clean_tokens:
            phrase = " ".join(clean_tokens).strip()
            if phrase:
                secondary_chunks.append(phrase)
                used_token_indices.update(chunk_indices)

    # 3. Secondary Terms: Standalone Content Words (Nouns, Adjectives, Verbs, Numbers)
    secondary_words = []
    for tok in doc:
        if tok.i not in used_token_indices:
            if tok.pos_ in valid_pos and _is_meaningful_term(tok):
                secondary_words.append(tok.text.strip())

    # Fallback: If filtering meta-nouns removed all secondary words, preserve non-stopwords
    if not secondary_chunks and not secondary_words:
        for tok in doc:
            if tok.i not in used_token_indices and not tok.is_stop and not tok.is_punct and len(tok.text.strip()) > 1:
                if tok.pos_ in valid_pos:
                    secondary_words.append(tok.text.strip())

    # Deduplicate while preserving order
    primary = list(dict.fromkeys(primary_entities))
    secondary = list(dict.fromkeys(secondary_chunks + secondary_words))

    return {
        "primary": primary,
        "secondary": secondary
    }

def build_spacy_fts_queries(query: str, model_name: str = "en_core_web_sm") -> List[str]:
    """
    Constructs SQLite FTS query variations using intent gating:
      - Entity Queries (Entities present): Strict conjunctions on entities + secondary terms.
      - Descriptive Queries (No entities): Strict AND pass followed by disjunctive OR pass.
    """
    terms = extract_tiered_fts_terms(query, model_name=model_name)
    primary = terms.get("primary", [])
    secondary = terms.get("secondary", [])

    if not primary and not secondary:
        return [query]

    def _sanitize_term(t: str) -> str:
        return re.sub(r'["\':?*^~]', '', t).strip()

    primary_terms = [_sanitize_term(t) for t in primary if _sanitize_term(t)]

    secondary_words = []
    for s in secondary:
        clean_s = _sanitize_term(s)
        if clean_s:
            for word in clean_s.split():
                if word and word not in secondary_words:
                    secondary_words.append(word)

    primary_fts = [f'"{p}"' for p in primary_terms]
    secondary_fts = [f'"{w}"' for w in secondary_words]

    queries = []

    if primary_fts:
        # --- Intent: Entity Query ---
        # Tier 1: Entities + Secondary terms
        if secondary_fts:
            queries.append(" AND ".join(primary_fts + secondary_fts))
        # Tier 2: Entities only (Strict Anchor)
        queries.append(" AND ".join(primary_fts))
    else:
        # --- Intent: Descriptive / Symptom / Concept Query ---
        if secondary_fts:
            # Tier 1: Strict AND pass on all content words
            queries.append(" AND ".join(secondary_fts))
            # Tier 2: Disjunctive OR pass if more than 1 term (allows BM25 coverage ranking across adjectives/nouns)
            if len(secondary_fts) > 1:
                queries.append(" OR ".join(secondary_fts))

    unique_queries = list(dict.fromkeys(queries))
    return unique_queries if unique_queries else [query]

import zlib

def compress_text(text: str) -> bytes:
    """Compresses text using zlib with maximum compression level (level 9)."""
    if not text:
        return b""
    return zlib.compress(text.encode('utf-8'), level=9)

def decompress_text(data: Union[str, bytes, None]) -> str:
    """Decompresses text from zlib bytes, falling back gracefully for uncompressed strings."""
    if not data:
        return ""
    if isinstance(data, str):
        return data
    try:
        return zlib.decompress(data).decode('utf-8')
    except Exception:
        try:
            return data.decode('utf-8')
        except Exception:
            return str(data)

def compute_hash(content: Union[str, bytes]) -> str:
    """Computes SHA-256 hex digest of the content."""
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content
    return hashlib.sha256(content_bytes).hexdigest()

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_REGEX = re.compile(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b')

def redact_pii(text: str) -> str:
    """Replaces email addresses and phone numbers in text with <redacted>."""
    if not text:
        return text
    text = EMAIL_REGEX.sub('<redacted>', text)
    text = PHONE_REGEX.sub('<redacted>', text)
    return text

def handelize(path: str) -> str:
    """
    Normalizes paths for indexing (lowercase, safe chars).
    Replaces non-alphanumeric characters with hyphens.
    """
    # Lowercase
    s = path.lower()
    # Replace non-alphanumeric with hyphen
    s = re.sub(r'[^a-z0-9]+', '-', s)
    # Strip leading/trailing hyphens
    s = s.strip('-')
    return s

def chunk_text(text: str, window_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping character-based windows.
    
    Args:
        text: The content to chunk.
        window_size: The number of characters per chunk.
        overlap: The number of overlapping characters between chunks.
    
    Returns:
        List of chunk strings.
    """
    if not text:
        return []
    
    if len(text) <= window_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Stop if we've reached the end
        if end >= text_len:
            break
            
        start += (window_size - overlap)
        
    return chunks

def parse_int_ranges(spec: Union[str, int]) -> Optional[List[int]]:
    """
    Parses a string or integer containing integers, comma-separated values, and ranges
    (e.g., '234-299', '23,24,29', '22,40, 25-37', 5) into a sorted list of unique integers.
    Returns None if the spec is invalid or does not match an integer range pattern.
    """
    if isinstance(spec, int):
        return [spec]
    if not isinstance(spec, str):
        return None

    spec = spec.strip()
    if not spec:
        return None

    parts = [p.strip() for p in spec.split(',') if p.strip()]
    if not parts:
        return None

    result_set = set()
    for part in parts:
        if '-' in part:
            subparts = [sp.strip() for sp in part.split('-')]
            if len(subparts) != 2:
                return None
            if not (subparts[0].isdigit() and subparts[1].isdigit()):
                return None
            start, end = int(subparts[0]), int(subparts[1])
            if start <= end:
                result_set.update(range(start, end + 1))
            else:
                result_set.update(range(end, start + 1))
        else:
            if not part.isdigit():
                return None
            result_set.add(int(part))

    return sorted(result_set)

def parse_target_spec(target: Union[str, int, None], default_collection: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a target string or int into structured components:
      - URIs: qmd://<collection>/<path>[:<seq_spec>]
      - Shorthands: <collection>:<path>[:<seq_spec>]
      - Relative targets: <path>:<seq_spec> or <path>
      - Chunk row ID / ID ranges: e.g. 5, "10-15", "22,40,25-27"

    Returns a dict:
      {
        "collection": Optional[str],
        "path": Optional[str],
        "seq": Optional[List[int]],
        "seq_ids": Optional[List[int]],  # alias for seq
        "row_ids": Optional[List[int]]
      }
    """
    if target is None:
        return {
            "collection": default_collection,
            "path": None,
            "seq": None,
            "seq_ids": None,
            "row_ids": None
        }

    if isinstance(target, int):
        return {
            "collection": default_collection,
            "path": None,
            "seq": None,
            "seq_ids": None,
            "row_ids": [target]
        }

    if not isinstance(target, str):
        return {
            "collection": default_collection,
            "path": None,
            "seq": None,
            "seq_ids": None,
            "row_ids": None
        }

    s = target.strip()
    if not s:
        return {
            "collection": default_collection,
            "path": None,
            "seq": None,
            "seq_ids": None,
            "row_ids": None
        }

    # 1. Pure row ID or integer ranges (e.g., "5", "10-15", "22,40,25-27")
    if re.match(r'^[0-9,\-\s]+$', s):
        parsed_ids = parse_int_ranges(s)
        if parsed_ids is not None:
            return {
                "collection": default_collection,
                "path": None,
                "seq": None,
                "seq_ids": None,
                "row_ids": parsed_ids
            }

    # 2. URI scheme: qmd://<collection>/<path>[:<seq_spec>]
    if s.startswith("qmd://"):
        uri_body = s[6:]
        seq = None
        if ":" in uri_body:
            body_part, seq_part = uri_body.rsplit(":", 1)
            parsed_seq = parse_int_ranges(seq_part)
            if parsed_seq is not None:
                uri_body = body_part
                seq = parsed_seq

        coll = default_collection
        if "/" in uri_body:
            coll_part, path_part = uri_body.split("/", 1)
            coll = coll_part if coll_part else default_collection
            path = path_part
        else:
            path = uri_body

        return {
            "collection": coll,
            "path": path,
            "seq": seq,
            "seq_ids": seq,
            "row_ids": None
        }

    # 3. Check for Windows absolute path (e.g. C:\path or C:/path)
    has_drive_letter = bool(re.match(r'^[a-zA-Z]:[\\/]', s))

    coll = default_collection
    path = s
    seq = None

    if not has_drive_letter and ":" in s:
        parts = s.split(":")
        last_seq = parse_int_ranges(parts[-1])
        if last_seq is not None and len(parts) >= 2:
            seq = last_seq
            remaining = ":".join(parts[:-1])
            if ":" in remaining:
                coll_part, path_part = remaining.split(":", 1)
                coll = coll_part if coll_part else default_collection
                path = path_part
            else:
                path = remaining
        else:
            if len(parts) == 2:
                coll = parts[0] if parts[0] else default_collection
                path = parts[1]
            else:
                coll = parts[0] if parts[0] else default_collection
                path = ":".join(parts[1:])

    return {
        "collection": coll,
        "path": path,
        "seq": seq,
        "seq_ids": seq,
        "row_ids": None
    }

def parse_query_directives(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parses inline cheat codes and directives from a query string.
    
    Supported tokens:
      - title:"..." / t:"..." / title:...
      - path:"..." / file:"..." / p:...
      - col:... / in:... / c:...
      - lex:"..." / fts:"..." / l:...
      - limit:N / n:N
      - pii:on / pii:off / pii:true / pii:false / --redact-pii / --no-pii
      - seen:exclude / seen:on / seen:off / seen:true / seen:false / --exclude-seen
      - rerank:on / rerank:off / rr:on / rr:off / rerank:true / rerank:false
      - regex:on / regex:off / regex:true / regex:false
      - case:on / case:off / case:true / case:false
    
    Returns:
        (clean_query, directives_dict)
    """
    if not query:
        return "", {}

    text = query
    directives: Dict[str, Any] = {}

    def _extract_str(pattern: str) -> Optional[str]:
        nonlocal text
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        val = match.group(1) or match.group(2) or match.group(3)
        text = text[:match.start()] + " " + text[match.end():]
        return val

    # Title filter
    t = _extract_str(r'(?:title|t):(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
    if t is not None:
        directives["title"] = t

    # Path / File filter
    p = _extract_str(r'(?:path|file|p):(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
    if p is not None:
        directives["path"] = p

    # Collection filter
    c = _extract_str(r'(?:col|in|c):(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
    if c is not None:
        directives["collection"] = c

    # Lexical / FTS override
    l = _extract_str(r'(?:lex|fts|l):(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
    if l is not None:
        directives["lex"] = l

    # Numeric limit
    limit_match = re.search(r'(?:limit|n):(\d+)', text, re.IGNORECASE)
    if limit_match:
        directives["limit"] = int(limit_match.group(1))
        text = text[:limit_match.start()] + " " + text[limit_match.end():]

    # Redact PII flag
    pii_match = re.search(r'(?:pii):(?:"(on|off|true|false)"|\'(on|off|true|false)\'|(on|off|true|false))|(--redact-pii|--no-pii)', text, re.IGNORECASE)
    if pii_match:
        raw_val = (pii_match.group(1) or pii_match.group(2) or pii_match.group(3) or pii_match.group(4) or "").lower()
        if raw_val in ("on", "true", "--redact-pii"):
            directives["redact_pii"] = True
        elif raw_val in ("off", "false", "--no-pii"):
            directives["redact_pii"] = False
        text = text[:pii_match.start()] + " " + text[pii_match.end():]

    # Exclude Seen flag
    seen_match = re.search(r'(?:seen):(?:"(exclude|off|on|true|false)"|\'(exclude|off|on|true|false)\'|(exclude|off|on|true|false))|(--exclude-seen)', text, re.IGNORECASE)
    if seen_match:
        raw_val = (seen_match.group(1) or seen_match.group(2) or seen_match.group(3) or seen_match.group(4) or "").lower()
        if raw_val in ("exclude", "on", "true", "--exclude-seen"):
            directives["exclude_seen"] = True
        elif raw_val in ("off", "false"):
            directives["exclude_seen"] = False
        text = text[:seen_match.start()] + " " + text[seen_match.end():]

    # Rerank flag
    rr_match = re.search(r'(?:rerank|rr):(?:"(on|off|true|false)"|\'(on|off|true|false)\'|(on|off|true|false))', text, re.IGNORECASE)
    if rr_match:
        raw_val = (rr_match.group(1) or rr_match.group(2) or rr_match.group(3) or "").lower()
        directives["rerank"] = raw_val in ("on", "true")
        text = text[:rr_match.start()] + " " + text[rr_match.end():]

    # Regex flag (for grep)
    regex_match = re.search(r'(?:regex):(?:"(on|off|true|false)"|\'(on|off|true|false)\'|(on|off|true|false))', text, re.IGNORECASE)
    if regex_match:
        raw_val = (regex_match.group(1) or regex_match.group(2) or regex_match.group(3) or "").lower()
        directives["regex"] = raw_val in ("on", "true")
        text = text[:regex_match.start()] + " " + text[regex_match.end():]

    # Case sensitivity flag (for grep)
    case_match = re.search(r'(?:case):(?:"(on|off|true|false)"|\'(on|off|true|false)\'|(on|off|true|false))', text, re.IGNORECASE)
    if case_match:
        raw_val = (case_match.group(1) or case_match.group(2) or case_match.group(3) or "").lower()
        directives["case_sensitive"] = raw_val in ("on", "true")
        text = text[:case_match.start()] + " " + text[case_match.end():]

    clean_query = re.sub(r'\s+', ' ', text).strip()
    return clean_query, directives