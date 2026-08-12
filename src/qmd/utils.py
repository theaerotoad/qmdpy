import hashlib
import re
from typing import List, Dict, Optional, Any, Union

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