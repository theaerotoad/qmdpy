from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .filename_extractor import FilenameDateExtractor
from .models import DateCandidate, DateExtractionReport, ExtractionSource
from .text_extractor import TextDateExtractor


class DateResolver:
    """Unified engine to infer and resolve document dates from filename and text content."""

    def __init__(
        self,
        filename_extractor: Optional[FilenameDateExtractor] = None,
        text_extractor: Optional[TextDateExtractor] = None,
        header_char_limit: int = 2000,
    ):
        self.filename_extractor = filename_extractor or FilenameDateExtractor()
        self.text_extractor = text_extractor or TextDateExtractor()
        self.header_char_limit = header_char_limit

    def resolve_from_file(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> DateExtractionReport:
        path = Path(file_path)
        content_preview = ""

        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding=encoding, errors="ignore") as f:
                    content_preview = f.read(self.header_char_limit)
            except Exception:
                content_preview = ""

        return self.resolve(
            filename=str(path).replace("\\", "/"),
            text_content=content_preview,
        )

    def resolve(
        self,
        filename: str,
        text_content: str = "",
    ) -> DateExtractionReport:
        filename_candidates: list[DateCandidate] = []
        text_candidates: list[DateCandidate] = []

        # 1. Extract from filename / path
        if filename:
            filename_candidates = self.filename_extractor.extract(filename)

        # 2. Extract from content header
        preview = text_content[: self.header_char_limit] if text_content else ""
        if preview.strip():
            text_candidates = self.text_extractor.extract(preview, max_chars=self.header_char_limit)

        all_candidates = filename_candidates + text_candidates

        if not all_candidates:
            return DateExtractionReport(
                filename=filename,
                resolved_date=None,
                primary_source=None,
                confidence=0.0,
                candidates=[],
                inspected_text_preview=preview,
            )

        # 3. Select winning candidate based on priority hierarchy:
        # Priority 1: Filename / Path candidate (last in path > earlier full date in path > partial date in path)
        # Priority 2: Document text candidate (first in document > later in document)
        if filename_candidates:
            top = filename_candidates[0]
            # If filename candidate is only a partial date, but document text has a full date with 4-digit year,
            # prefer the more specific full document date.
            if not top.is_full_date and text_candidates and text_candidates[0].is_full_date and text_candidates[0].has_4digit_year:
                top = text_candidates[0]
        else:
            top = text_candidates[0]

        # Agreement boost: if filename candidate and text candidate agree on the date, boost confidence to 0.99
        if filename_candidates and text_candidates:
            if any(fc.parsed_date.date() == tc.parsed_date.date() for fc in filename_candidates for tc in text_candidates):
                top = DateCandidate(
                    raw_text=top.raw_text,
                    parsed_date=top.parsed_date,
                    source=top.source,
                    confidence=0.99,
                    start_char=top.start_char,
                    end_char=top.end_char,
                    is_full_date=top.is_full_date,
                    has_4digit_year=top.has_4digit_year,
                )

        return DateExtractionReport(
            filename=filename,
            resolved_date=top.parsed_date,
            primary_source=top.source,
            confidence=top.confidence,
            candidates=all_candidates,
            inspected_text_preview=preview,
        )