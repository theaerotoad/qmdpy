from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import dateparser

from .models import DateCandidate, ExtractionSource

MONTH_NAME_PATTERN = (
    r"(?:Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|"
    r"Jun(?:e|\.)?|Jul(?:y|\.)?|Aug(?:ust|\.)?|Sep(?:t(?:ember)?|\.)?|Sept\.?|"
    r"Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?)"
)

# Patterns for filenames, paths, and directory hierarchies supporting spaces, slashes, dashes, periods, underscores, and standards
FILENAME_PATTERNS = [
    # Full ISO Date-Time: 2026-08-26T19:02:19Z, 2026-08-26T15:02:19-04:00, 20260826T190219Z
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_/.](?:0[1-9]|1[0-2])[-_/.](?:0[1-9]|[12]\d|3[01])[T\s](?:[01]\d|2[0-3])[-_:.](?:[0-5]\d)(?:[-_:.](?:[0-5]\d))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    ),
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])T(?:[01]\d|2[0-3])(?:[0-5]\d)(?:[0-5]\d)?Z?)(?!\d)"
    ),
    # Compact 14-digit timestamp YYYYMMDDHHMMSS (e.g. audit_log_20260826190219.csv)
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:[01]\d|2[0-3])(?:[0-5]\d)(?:[0-5]\d))(?!\d)"
    ),
    # Full ISO Calendar Date across delimiters (e.g. 2026-08-26, 2026/08/26, 2026_08_26, 2026.08.26)
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_/.\s](?:0[1-9]|1[0-2])[-_/.\s](?:0[1-9]|[12]\d|3[01]))(?!\d)"
    ),
    # ISO 8601 Ordinal / Day-of-Year Date (e.g. 2026-238, 2026_238)
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_](?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6]))(?!\d)"
    ),
    # ISO 8601 Week Date (e.g. 2026-W35-3, 2026-W35)
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_]W(?:0[1-9]|[1-4]\d|5[0-3])(?:[-_][1-7])?)(?!\d)",
        re.IGNORECASE,
    ),
    # Fiscal / Calendar Quarter Notation in paths/names (e.g. reports/2026/Q3, FY2026-Q3, FY2026-Q3_summary, 2026-Q3, 2026_Q3)
    re.compile(
        r"(?<!\d)(?P<date>(?:FY[-_ ]?)?(?:1[0-9]|20)\d{2}[-_/. ]*Q[1-4])(?![a-zA-Z0-9])",
        re.IGNORECASE,
    ),
    # Year + Month name + Day or Year + MonthDay (e.g. 2026/June14, 2026.June.14, 2026_June_14, 1841/Feb/13)
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_/.\s]*"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]*(?:0?[1-9]|[12]\d|3[01]))(?!\d)",
        re.IGNORECASE,
    ),
    # Day + Month name + Year (e.g. 14_June_2026, 14.June.2026, 14 June 2026, 14-Jun-2026, 14/June/2026, 13-Feb-1841, 19_Dec_1942)
    re.compile(
        r"(?<!\d)(?P<date>(?:0?[1-9]|[12]\d|3[01])[-_/.\s]*"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]+(?:1[0-9]|20)\d{2})(?!\d)",
        re.IGNORECASE,
    ),
    # Month name + Day + Year (e.g. June_14_2026, June.14.2026, June 14 2026, Jun-14-2026, Dec_19_1942, February_13th_1841)
    re.compile(
        r"(?<!\d)(?P<date>"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]*(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[-_/.\s]+(?:1[0-9]|20)\d{2})(?!\d)",
        re.IGNORECASE,
    ),
    # Month name + Year (e.g. Oct_2024, changelog-august-2026, October 2024, 2024_October, 2024/October, Dec_1942, Feb_1841)
    re.compile(
        r"(?P<date>"
        + MONTH_NAME_PATTERN
        + r"[-_/.\s]+(?:1[0-9]|20)\d{2})(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_/.\s]+"
        + MONTH_NAME_PATTERN
        + r")(?![a-zA-Z0-9])",
        re.IGNORECASE,
    ),
    # Compact 8-digit YYYYMMDD with 4-digit century
    re.compile(
        r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?!\d)"
    ),
    # Day-Month-Year or Month-Day-Year (e.g. 20-05-2024, 05.20.2024, 20_05_2024, 26-08-2026)
    re.compile(
        r"(?<!\d)(?P<date>(?:0[1-9]|[12]\d|3[01])[-_/.\s](?:0[1-9]|1[0-2])[-_/.\s](?:1[0-9]|20)\d{2})(?!\d)"
    ),
    # Year-Month directory pattern (e.g. 2026/09, backups/2026-08, 2026_09, 2026.09)
    re.compile(r"(?<!\d)(?P<date>(?:1[0-9]|20)\d{2}[-_/.\s](?:0[1-9]|1[0-2]))(?![0-9])"),
]


def _is_full_date(matched_str: str) -> bool:
    """Determine if a matched string contains a full date (year + month + day / ordinal / week date)."""
    if re.search(r"Q[1-4]", matched_str, re.IGNORECASE):
        return False
    if re.search(r"(?:1[0-9]|20)\d{2}[-_](?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])\b", matched_str):
        return True
    if re.search(r"W\d{2}[-_][1-7]\b", matched_str, re.IGNORECASE):
        return True
    if re.search(r"W\d{2}\b", matched_str, re.IGNORECASE):
        return False
    if re.search(r"(?:1[0-9]|20)\d{6}", matched_str):
        return True

    has_month_name = bool(re.search(MONTH_NAME_PATTERN, matched_str, re.IGNORECASE))
    without_year = re.sub(r"\b(?:1[0-9]|20)\d{2}\b", "", matched_str)
    digits = re.findall(r"\d+", without_year)
    if has_month_name:
        return len(digits) >= 1
    return len(digits) >= 2


class FilenameDateExtractor:
    """Extracts date information from file names, paths, and directory hierarchies."""

    def __init__(
        self,
        dateparser_settings: Optional[dict] = None,
        include_parent_directories: bool = True,
        skip_ambiguous_six_digit: bool = True,
        require_four_digit_year: bool = True,
    ):
        self.dateparser_settings = dateparser_settings or {
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_DATES_FROM": "past",
            "REQUIRE_PARTS": ["year"],
        }
        self.include_parent_directories = include_parent_directories
        self.skip_ambiguous_six_digit = skip_ambiguous_six_digit
        self.require_four_digit_year = require_four_digit_year

    def extract(self, file_path_or_name: str | Path) -> List[DateCandidate]:
        path_str = str(file_path_or_name).replace("\\", "/")
        path = Path(file_path_or_name)

        # Skip ambiguous 6-digit dates like 240312 that lack a distinct 4-digit century
        if self.skip_ambiguous_six_digit:
            if re.search(r"(?<!\d)\d{6}(?!\d)", path_str) and not re.search(
                r"\b(?:1[0-9]|20)\d{2}\b", path_str
            ):
                return []

        candidates: List[DateCandidate] = []
        found_spans: set[tuple[int, int]] = set()

        def _is_overlapping(start: int, end: int) -> bool:
            return any(not (end <= s[0] or start >= s[1]) for s in found_spans)

        for pattern in FILENAME_PATTERNS:
            for match in pattern.finditer(path_str):
                span = (match.start(), match.end())
                if _is_overlapping(*span):
                    continue

                matched_str = match.group("date")
                parsed: Optional[datetime] = None
                cleaned_str = matched_str

                # 1. Handle compact 14-digit timestamp YYYYMMDDHHMMSS
                if re.fullmatch(r"(?:19|20)\d{12}", matched_str):
                    try:
                        parsed = datetime.strptime(matched_str, "%Y%m%d%H%M%S")
                    except ValueError:
                        parsed = None

                # 2. Handle standard ISO calendar date YYYY-MM-DD across delimiters
                elif re.fullmatch(r"(?:19|20)\d{2}[-_/.](?:0[1-9]|1[0-2])[-_/.](?:0[1-9]|[12]\d|3[01])", matched_str):
                    norm_iso = re.sub(r"[-_/.]", "-", matched_str)
                    try:
                        parsed = datetime.strptime(norm_iso, "%Y-%m-%d")
                    except ValueError:
                        parsed = None

                # 3. Handle compact 8-digit numeric format YYYYMMDD
                elif re.fullmatch(r"(?:19|20)\d{6}", matched_str):
                    try:
                        parsed = datetime.strptime(matched_str, "%Y%m%d")
                    except ValueError:
                        parsed = None

                # 3. Handle ISO Ordinal dates (YYYY-DDD or YYYY_DDD)
                elif re.fullmatch(r"(?:19|20)\d{2}[-_](?:00[1-9]|0[1-9]\d|[12]\d{2}|3[0-5]\d|36[0-6])", matched_str):
                    normalized_ord = matched_str.replace("_", "-")
                    try:
                        parsed = datetime.strptime(normalized_ord, "%Y-%j")
                    except ValueError:
                        parsed = None

                # 4. Handle ISO Week dates (e.g. 2026-W35-3 or 2026-W35)
                elif re.search(r"W\d{2}", matched_str, re.IGNORECASE):
                    w_match = re.search(r"((?:19|20)\d{2})[-_]W(\d{2})(?:[-_]([1-7]))?", matched_str, re.IGNORECASE)
                    if w_match:
                        yr = int(w_match.group(1))
                        wk = int(w_match.group(2))
                        day = int(w_match.group(3)) if w_match.group(3) else 1
                        try:
                            parsed = datetime.fromisocalendar(yr, wk, day)
                        except ValueError:
                            parsed = None

                # 5. Handle Quarter notations (e.g. 2026/Q3, FY2026-Q3, 2026-Q3)
                elif re.search(r"Q[1-4]", matched_str, re.IGNORECASE):
                    q_match = re.search(r"((?:19|20)\d{2})[-_/. ]*Q([1-4])", matched_str, re.IGNORECASE)
                    if q_match:
                        yr = int(q_match.group(1))
                        q = int(q_match.group(2))
                        quarter_month_map = {1: 1, 2: 4, 3: 7, 4: 10}
                        parsed = datetime(yr, quarter_month_map[q], 1)

                if parsed is None:
                    # Normalize separators (spaces, slashes, dashes, dots, underscores) and split merged month/day
                    cleaned_str = re.sub(r"[-_/.\s]+", " ", matched_str).strip()
                    cleaned_str = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", cleaned_str)
                    cleaned_str = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", cleaned_str)

                    parsed = dateparser.parse(
                        cleaned_str, settings=self.dateparser_settings
                    ) or dateparser.parse(
                        matched_str, settings=self.dateparser_settings
                    )

                if parsed:
                    if self.require_four_digit_year and not (
                        1000 <= parsed.year <= 2100
                    ):
                        continue

                    full_date = _is_full_date(matched_str)
                    confidence = 0.95 if full_date else 0.80

                    found_spans.add(span)
                    has_4digit = bool(re.search(r"\b(?:1[0-9]|20)\d{2}\b", matched_str))
                    candidates.append(
                        DateCandidate(
                            raw_text=matched_str,
                            parsed_date=parsed,
                            source=ExtractionSource.FILENAME,
                            confidence=confidence,
                            start_char=match.start(),
                            end_char=match.end(),
                            is_full_date=full_date,
                            has_4digit_year=has_4digit,
                        )
                    )

        # Fallback: tokenized parsing across path stem
        if not candidates:
            token_candidate = self._parse_tokens(path.stem)
            if token_candidate:
                candidates.append(token_candidate)

        # Priority ordering for path candidates:
        # 1. Full dates (is_full_date=True) before partial dates
        # 2. 4-digit explicit year before 2-digit year
        # 3. Last in path (highest start_char in path string)
        # 4. Confidence and text length descending
        candidates.sort(
            key=lambda c: (
                1 if c.is_full_date else 0,
                1 if c.has_4digit_year else 0,
                c.start_char if c.start_char is not None else -1,
                c.confidence,
                len(c.raw_text),
            ),
            reverse=True,
        )
        return candidates

    def _parse_tokens(self, stem: str) -> Optional[DateCandidate]:
        if self.skip_ambiguous_six_digit and re.search(r"(?<!\d)\d{6}(?!\d)", stem):
            return None

        if self.require_four_digit_year and not re.search(r"\b(?:1[0-9]|20)\d{2}\b", stem):
            return None

        normalized = re.sub(r"[_\-.]+", " ", stem)
        parsed = dateparser.parse(normalized, settings=self.dateparser_settings)
        if parsed and (not self.require_four_digit_year or 1000 <= parsed.year <= 2100):
            return DateCandidate(
                raw_text=stem,
                parsed_date=parsed,
                source=ExtractionSource.FILENAME,
                confidence=0.60,
                is_full_date=_is_full_date(stem),
                has_4digit_year=bool(re.search(r"\b(?:1[0-9]|20)\d{2}\b", stem)),
            )
        return None