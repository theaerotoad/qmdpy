"""dplib: Date and metadata parsing library for document paths, filenames, and text content."""

from typing import Optional, Union
from pathlib import Path
from datetime import datetime

from dplib.date_extractor.models import DateCandidate, DateExtractionReport, ExtractionSource
from dplib.date_extractor.filename_extractor import FilenameDateExtractor
from dplib.date_extractor.text_extractor import TextDateExtractor
from dplib.date_extractor.resolver import DateResolver

__all__ = [
    "DateCandidate",
    "DateExtractionReport",
    "ExtractionSource",
    "FilenameDateExtractor",
    "TextDateExtractor",
    "DateResolver",
    "extract_date",
]

def extract_date(path: Union[str, Path] = "", content: str = "") -> Optional[datetime]:
    """Convenience function to resolve document date from path and/or content preview."""
    resolver = DateResolver()
    report = resolver.resolve(filename=str(path).replace("\\", "/"), text_content=content)
    return report.resolved_date