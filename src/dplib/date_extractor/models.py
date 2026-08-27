from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ExtractionSource(str, Enum):
    FILENAME = "filename"
    TEXT_HEADER = "text_header"


@dataclass(frozen=True)
class DateCandidate:
    """Represents a potential date match found in filename or content text."""

    raw_text: str
    parsed_date: datetime
    source: ExtractionSource
    confidence: float
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    is_full_date: bool = True


@dataclass
class DateExtractionReport:
    """Consolidated outcome of date extraction from a file."""

    filename: str
    resolved_date: Optional[datetime] = None
    primary_source: Optional[ExtractionSource] = None
    confidence: float = 0.0
    candidates: list[DateCandidate] = field(default_factory=list)
    inspected_text_preview: str = ""