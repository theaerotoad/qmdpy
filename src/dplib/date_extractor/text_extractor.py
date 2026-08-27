from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional
import dateparser
from dateparser.search import search_dates
import spacy
from spacy.language import Language

from date_extractor.models import DateCandidate, ExtractionSource

# Reusable month pattern supporting full names, 3/4-letter abbreviations, and optional trailing periods (e.g. Dec., Sept., Feb.)
MONTH_NAME_PATTERN = (
    r"(?:Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|"
    r"Jun(?:e|\.)?|Jul(?:y|\.)?|Aug(?:ust|\.)?|Sep(?:t(?:ember)?|\.)?|Sept\.?|"
    r"Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?)"
)

YEAR_4DIGIT_PATTERN = r"(?:1[0-9]|20)\d{2}"
YEAR_2OR4DIGIT_PATTERN = r"(?:(?:1[0-9]|20)\d{2}|(?:\'?[0-9]{2}))"

# Common header label prefixes in document bodies (including YAML/TOML/JSON metadata and prose headers)
HEADER_PREFIX_PATTERN = re.compile(
    r"(?:[\"\']?(?:date|created|published|created_at|updated_at|publish_date|publish_at|modified|as_of|as\s+of|dated|generated(?:\s+on)?|recorded(?:\s+on)?)[\"\']?\s*[:=]\s*[\"\']?(?P<date>[^\r\n;\"]{3,60})[\"\']?)",
    re.IGNORECASE,
)

# Text date patterns supporting standards, ISO 8601, RFC 5322/1123, ANSI C, ordinals, 2-digit years, and historical dates
TEXT_DATE_PATTERNS = [
    # RFC 5322 / RFC 1123 / HTTP Header (e.g. Wed, 26 Aug 2026 19:02:19 -0700 or Wed, 26 Aug 2026 19:02:19 GMT)
    re.compile(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(?:1[0-9]|20)\d{2}\s+\d{2}:\d{2}(?::\d{2})?(?:\s+[+-]\d{4}|\s+GMT|\s+UTC)?\b",
        re.IGNORECASE,
    ),
    # ANSI C asctime() (e.g. Wed Aug 26 19:02:19 2026)
    re.compile(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+(?:1[0-9]|20)\d{2}\b",
        re.IGNORECASE,
    ),
    # ISO 8601 Datetime with Timezone / Offset / UTC (e.g. 2026-08-26T19:02:19Z, 2026-08-26T15:02:19-04:00, 2026-08-26 19:02:19)
    re.compile(
        r"(?<!\d)(?:1[0-9]|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[T\s](?:[01]\d|2[0-3]):(?:[0-5]\d)(?::(?:[0-5]\d))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    ),
    # Full Written Date with Weekday (e.g. Wednesday, August 26, 2026, Sat, Dec. 19, 1942, Saturday, February 13th 1841)
    re.compile(
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|[-_/.\s]+)"
        + YEAR_2OR4DIGIT_PATTERN
        + r"\b",
        re.IGNORECASE,
    ),
    # Month DD, YYYY or Month DD YYYY or Month DD, YY (e.g. August 26, 2026, Dec. 19, 1942, February 13th 1841, Aug.26.2026, Dec 19, 92, Dec. 19, '92)
    re.compile(
        r"\b"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|[-_/.\s]+)"
        + YEAR_2OR4DIGIT_PATTERN
        + r"\b",
        re.IGNORECASE,
    ),
    # DD Month YYYY or DD Month YY or DD of Month YYYY (e.g. 26 August 2026, 13th of February 1841, 13th day of February, 1841, 19 Dec. 1942, 26th Aug 2026, 26.Aug.2026, 26-Aug-2026, 19 Dec 92)
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?(?:\s+day)?(?:\s+of)?[-_/.\s]+"
        + MONTH_NAME_PATTERN
        + r"(?:,\s*|[-_/.\s]+)"
        + YEAR_2OR4DIGIT_PATTERN
        + r"\b",
        re.IGNORECASE,
    ),
    # ISO 8601 Ordinal (e.g. 2026-238)
    re.compile(
        r"(?<!\d)(?:1[0-9]|20)\d{2}-(?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])(?!\d)"
    ),
    # ISO 8601 Week Date (e.g. 2026-W35-3 or 2026-W35)
    re.compile(
        r"(?<!\d)(?:1[0-9]|20)\d{2}-W(?:0[1-9]|[1-4]\d|5[0-3])(?:-[1-7])?(?!\d)",
        re.IGNORECASE,
    ),
    # Fiscal Quarter notation (e.g. FY2026-Q3, 2026-Q3)
    re.compile(r"\b(?:FY)?(?:1[0-9]|20)\d{2}[-_ ]?Q[1-4]\b", re.IGNORECASE),
    # ISO date across delimiters: 2024-08-15, 2024/08/15, 2024.08.15, 2024_08_15, 2024 08 15, 1942-12-19
    re.compile(
        r"(?<!\d)(?:1[0-9]|20)\d{2}[-_/.\s](?:0?[1-9]|1[0-2])[-_/.\s](?:0?[1-9]|[12]\d|3[01])(?!\d)"
    ),
    # US / UK date across delimiters: MM/DD/YYYY, DD.MM.YYYY, MM_DD_YYYY, DD-MM-YYYY, MM DD YYYY, 7/2/1992
    re.compile(
        r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-_/.\s](?:0?[1-9]|1[0-2])[-_/.\s](?:1[0-9]|20)\d{2}(?!\d)"
    ),
    # Numeric date with 2-digit year across delimiters: M/D/YY, MM/DD/YY, DD/MM/YY, DD.MM.YY (e.g. 7/2/92, 07/02/92, 7-2-92, 7.2.92, 19/12/92)
    re.compile(
        r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:[0-9]{2})(?!\d)"
    ),
    # Month YYYY or Month YY (e.g. January 2024, Jan.2024, Dec. 1942, February 1841, Jan_2024, Jan-2024, Jan/2024)
    re.compile(
        r"\b"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]+(?:(?:1[0-9]|20)\d{2}|(?:\'?[0-9]{2}))\b",
        re.IGNORECASE,
    ),
]


def _is_text_full_date(raw: str) -> bool:
    if re.search(r"Q[1-4]", raw, re.IGNORECASE):
        return False
    if re.search(r"(?:1[0-9]|20)\d{2}-(?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])\b", raw):
        return True
    if re.search(r"W\d{2}-[1-7]\b", raw, re.IGNORECASE):
        return True
    if re.search(r"W\d{2}\b", raw, re.IGNORECASE):
        return False
    has_month = bool(re.search(MONTH_NAME_PATTERN, raw, re.IGNORECASE))
    has_4digit_year = bool(re.search(r"\b(?:1[0-9]|20)\d{2}\b", raw))
    without_4digit_yr = re.sub(r"\b(?:1[0-9]|20)\d{2}\b", "", raw)
    digits = [int(d) for d in re.findall(r"\d+", without_4digit_yr)]
    if has_month:
        if has_4digit_year:
            return any(1 <= d <= 31 for d in digits)
        if len(digits) >= 2:
            return any(1 <= d <= 31 for d in digits)
        return False
    all_digits = re.findall(r"\d+", raw)
    return len(all_digits) >= 3


class TextDateExtractor:
    """Extracts date entities from the initial portion of file text using spaCy and dateparser."""

    def __init__(
        self,
        nlp: Optional[Language] = None,
        dateparser_settings: Optional[dict] = None,
    ):
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")

        self.dateparser_settings = dateparser_settings or {
            "PREFER_DATES_FROM": "past",
            "REQUIRE_PARTS": ["year"],
        }

    def extract(self, text: str, max_chars: int = 500) -> List[DateCandidate]:
        snippet = text[:max_chars]
        candidates: List[DateCandidate] = []
        seen_spans: set[tuple[int, int]] = set()

        def _is_overlapping(start: int, end: int) -> bool:
            return any(not (end <= s[0] or start >= s[1]) for s in seen_spans)

        # Step 1: Check Structured Front Matter (Split fields, date arrays, and numeric epoch timestamps)
        structured_candidate = self._parse_structured_frontmatter(snippet)
        if structured_candidate:
            candidates.append(structured_candidate)

        # Step 2: Check explicit labeled header patterns (e.g. "Date: 2024-03-12", "published: '2026-08-26 19:02:19'", "Date: Dec. 19, 1942.")
        for match in HEADER_PREFIX_PATTERN.finditer(snippet):
            raw_val = match.group("date").strip().strip("\"'").rstrip(".,;\"'").strip()
            parsed = self._parse_date_string(raw_val)
            if parsed and 1000 <= parsed.year <= 2100:
                span = (match.start(), match.end())
                if not _is_overlapping(*span):
                    seen_spans.add(span)
                    candidates.append(
                        DateCandidate(
                            raw_text=raw_val,
                            parsed_date=parsed,
                            source=ExtractionSource.TEXT_HEADER,
                            confidence=0.98,
                            start_char=match.start(),
                            end_char=match.end(),
                            is_full_date=_is_text_full_date(raw_val),
                        )
                    )

        # Step 3: Regex date patterns across the header snippet
        for pattern in TEXT_DATE_PATTERNS:
            for match in pattern.finditer(snippet):
                span = (match.start(), match.end())
                if _is_overlapping(*span):
                    continue
                raw_text = match.group().strip().rstrip(".,;\"'")
                parsed = self._parse_date_string(raw_text)
                if parsed and 1000 <= parsed.year <= 2100:
                    seen_spans.add(span)
                    early_boost = 0.05 if match.start() < 150 else 0.0
                    confidence = min(0.90 + early_boost, 0.95)
                    candidates.append(
                        DateCandidate(
                            raw_text=raw_text,
                            parsed_date=parsed,
                            source=ExtractionSource.TEXT_HEADER,
                            confidence=confidence,
                            start_char=match.start(),
                            end_char=match.end(),
                            is_full_date=_is_text_full_date(raw_text),
                        )
                    )

        # Step 4: Use spaCy NER DATE extraction
        doc = self.nlp(snippet)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                span = (ent.start_char, ent.end_char)
                if _is_overlapping(*span):
                    continue

                cleaned_ent = ent.text.strip().rstrip(".,;\"'")
                parsed = self._parse_date_string(cleaned_ent)
                if parsed and 1000 <= parsed.year <= 2100:
                    seen_spans.add(span)
                    early_boost = 0.10 if ent.start_char < 150 else 0.0
                    confidence = min(0.85 + early_boost, 0.95)
                    candidates.append(
                        DateCandidate(
                            raw_text=cleaned_ent,
                            parsed_date=parsed,
                            source=ExtractionSource.TEXT_HEADER,
                            confidence=confidence,
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                            is_full_date=_is_text_full_date(cleaned_ent),
                        )
                    )

        # Step 5: Fallback search_dates if nothing found yet
        if not candidates:
            try:
                found_dates = search_dates(snippet, settings=self.dateparser_settings)
                if found_dates:
                    for raw_str, dt in found_dates:
                        if not re.search(r"\d", raw_str):
                            continue
                        if dt and 1000 <= dt.year <= 2100:
                            cleaned_raw = raw_str.strip().rstrip(".,;\"'")
                            candidates.append(
                                DateCandidate(
                                    raw_text=cleaned_raw,
                                    parsed_date=dt,
                                    source=ExtractionSource.TEXT_HEADER,
                                    confidence=0.80,
                                    is_full_date=_is_text_full_date(cleaned_raw),
                                )
                            )
            except Exception:
                pass

        # Priority ordering for document text candidates:
        # 1. Full dates (is_full_date=True) before partial dates
        # 2. Confidence (explicit headers / frontmatter >= 0.95 first)
        # 3. First in document (lowest start_char)
        candidates.sort(
            key=lambda c: (
                1 if c.is_full_date else 0,
                c.confidence,
                -(c.start_char if c.start_char is not None else 999999),
                len(c.raw_text),
            ),
            reverse=True,
        )
        return candidates

    def _parse_date_string(self, text: str) -> Optional[datetime]:
        """Parses standard, specialized, and variant date strings into datetime objects."""
        cleaned = text.strip().rstrip(".,;\"'")
        if not cleaned:
            return None

        specialized = self._parse_specialized_date(cleaned)
        if specialized:
            return specialized

        parsed = dateparser.parse(cleaned, settings=self.dateparser_settings)
        if parsed:
            return parsed

        # Normalize apostrophe years e.g. "Dec. 19th, '92" -> "Dec. 19th, 92"
        normalized = cleaned.replace("'", "").replace("`", "")
        parsed = dateparser.parse(normalized, settings=self.dateparser_settings)
        if parsed:
            return parsed

        # Normalize legal/prose phrasing e.g. "13th day of February, 1841" -> "13th February, 1841"
        normalized = re.sub(r"\b(?:day\s+of|of)\b", " ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        parsed = dateparser.parse(normalized, settings=self.dateparser_settings)
        if parsed:
            return parsed

        # Normalize punctuation delimiters
        normalized = re.sub(r"[-_/.]+", " ", normalized).strip()
        parsed = dateparser.parse(normalized, settings=self.dateparser_settings)
        if parsed:
            return parsed

        return None

    def _parse_specialized_date(self, text: str) -> Optional[datetime]:
        """Handles specialized formats like Ordinals, ISO Weeks, Quarters, and Epoch timestamps."""
        text = text.strip()

        # Unix timestamp seconds (10 digits) or milliseconds (13 digits)
        if re.fullmatch(r"\d{10}(?:\d{3})?", text):
            try:
                ts = int(text)
                if len(text) == 13:
                    ts = ts / 1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
                if 1000 <= dt.year <= 2100:
                    return dt
            except (ValueError, OverflowError, OSError):
                pass

        # ISO Ordinal: YYYY-DDD
        if re.fullmatch(r"(?:1[0-9]|20)\d{2}-(?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])", text):
            try:
                return datetime.strptime(text, "%Y-%j")
            except ValueError:
                pass

        # ISO Week Date: YYYY-Www-d or YYYY-Www
        w_match = re.fullmatch(r"((?:1[0-9]|20)\d{2})-W(0[1-9]|[1-4]\d|5[0-3])(?:-([1-7]))?", text, re.IGNORECASE)
        if w_match:
            yr = int(w_match.group(1))
            wk = int(w_match.group(2))
            day = int(w_match.group(3)) if w_match.group(3) else 1
            try:
                return datetime.fromisocalendar(yr, wk, day)
            except ValueError:
                pass

        # Fiscal / Calendar Quarter: FY2026-Q3, 2026-Q3
        q_match = re.fullmatch(r"(?:FY)?((?:1[0-9]|20)\d{2})[-_ ]?Q([1-4])", text, re.IGNORECASE)
        if q_match:
            yr = int(q_match.group(1))
            q = int(q_match.group(2))
            quarter_month_map = {1: 1, 2: 4, 3: 7, 4: 10}
            return datetime(yr, quarter_month_map[q], 1)

        return None

    def _parse_structured_frontmatter(self, snippet: str) -> Optional[DateCandidate]:
        """Extracts date from split YAML/TOML front matter fields or date arrays."""
        # Split fields: year: 2026, month: 8, day: 26
        year_match = re.search(r"(?:^|\n)\s*year\s*[:=]\s*[\"']?((?:1[0-9]|20)\d{2})[\"']?", snippet, re.IGNORECASE)
        month_match = re.search(r"(?:^|\n)\s*month\s*[:=]\s*[\"']?([0-1]?\d)[\"']?", snippet, re.IGNORECASE)
        day_match = re.search(r"(?:^|\n)\s*day\s*[:=]\s*[\"']?([0-3]?\d)[\"']?", snippet, re.IGNORECASE)

        if year_match and month_match:
            try:
                yr = int(year_match.group(1))
                mo = int(month_match.group(1))
                dy = int(day_match.group(1)) if day_match else 1
                dt = datetime(yr, mo, dy)
                return DateCandidate(
                    raw_text=f"year: {yr}, month: {mo}, day: {dy}",
                    parsed_date=dt,
                    source=ExtractionSource.TEXT_HEADER,
                    confidence=0.98,
                    is_full_date=True,
                )
            except ValueError:
                pass

        # Front Matter Date Array: date_parts: [2026, 8, 26] or [2026, 8]
        array_match = re.search(r"(?:date_parts|date|published)\s*[:=]\s*\[\s*((?:1[0-9]|20)\d{2})\s*,\s*(\d{1,2})(?:\s*,\s*(\d{1,2}))?\s*\]", snippet, re.IGNORECASE)
        if array_match:
            try:
                yr = int(array_match.group(1))
                mo = int(array_match.group(2))
                dy = int(array_match.group(3)) if array_match.group(3) else 1
                dt = datetime(yr, mo, dy)
                return DateCandidate(
                    raw_text=array_match.group(0),
                    parsed_date=dt,
                    source=ExtractionSource.TEXT_HEADER,
                    confidence=0.98,
                    is_full_date=True,
                )
            except ValueError:
                pass

        # Timestamp in JSON / Front Matter: "timestamp": 1787796139 or 1787796139000
        ts_match = re.search(r"(?:[\"\']?timestamp[\"\']?|epoch)\s*[:=]\s*[\"\']?(\d{10}(?:\d{3})?)[\"\']?", snippet, re.IGNORECASE)
        if ts_match:
            raw_ts = ts_match.group(1)
            ts = int(raw_ts)
            if len(raw_ts) == 13:
                ts = ts / 1000.0
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
                if 1990 <= dt.year <= 2100:
                    return DateCandidate(
                        raw_text=ts_match.group(0),
                        parsed_date=dt,
                        source=ExtractionSource.TEXT_HEADER,
                        confidence=0.95,
                        is_full_date=True,
                    )
            except (ValueError, OverflowError, OSError):
                pass

        return None