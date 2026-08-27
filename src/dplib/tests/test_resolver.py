from datetime import datetime
from pathlib import Path
import pytest

from date_extractor.models import ExtractionSource
from date_extractor.resolver import DateResolver


@pytest.fixture
def resolver():
    return DateResolver()


def test_resolve_from_filename_only(resolver):
    report = resolver.resolve(filename="invoice_2023-12-01.txt", text_content="No dates here.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2023
    assert report.resolved_date.month == 12
    assert report.primary_source == ExtractionSource.FILENAME


def test_resolve_from_text_header_only(resolver):
    header = "CONFIDENTIAL MEMO\nDate: 2024-04-10\nRecipient: All Staff"
    report = resolver.resolve(filename="memo_untitled.txt", text_content=header)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 4
    assert report.resolved_date.day == 10
    assert report.primary_source == ExtractionSource.TEXT_HEADER


def test_resolve_agreement_boost(resolver):
    header = "Report generated on 2024-08-15 by automated pipeline."
    report = resolver.resolve(filename="report_2024-08-15.log", text_content=header)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.confidence >= 0.95


def test_resolve_from_hierarchical_path(resolver):
    report = resolver.resolve(filename="2026/06/10/notes.md", text_content="General meeting notes.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2026
    assert report.resolved_date.month == 6
    assert report.resolved_date.day == 10


def test_resolve_ambiguous_six_digit_skipped(resolver):
    report = resolver.resolve(filename="240312/notes.md", text_content="No dates present here.")
    assert report.resolved_date is None
    assert report.confidence == 0.0


def test_resolve_from_file_path(resolver, tmp_path: Path):
    file_path = tmp_path / "summary_2024_01_10.txt"
    file_path.write_text("Created: 2024-01-10\nInitial project kickoff notes.", encoding="utf-8")

    report = resolver.resolve_from_file(file_path)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 1
    assert report.resolved_date.day == 10
    assert "summary_2024_01_10.txt" in report.filename