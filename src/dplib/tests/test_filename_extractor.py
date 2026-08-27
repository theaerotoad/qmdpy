from datetime import datetime
import pytest

from date_extractor.filename_extractor import FilenameDateExtractor
from date_extractor.models import ExtractionSource


@pytest.fixture
def extractor():
    return FilenameDateExtractor()


def test_extract_iso_format(extractor):
    results = extractor.extract("quarterly_report_2024-03-15.txt")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2024
    assert results[0].parsed_date.month == 3
    assert results[0].parsed_date.day == 15
    assert results[0].source == ExtractionSource.FILENAME


def test_extract_compact_numeric_format(extractor):
    results = extractor.extract("backup_20231105.log")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2023
    assert results[0].parsed_date.month == 11
    assert results[0].parsed_date.day == 5


def test_extract_month_name_format(extractor):
    results = extractor.extract("statement_October_2024.pdf")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2024
    assert results[0].parsed_date.month == 10


def test_no_date_in_filename(extractor):
    results = extractor.extract("random_document_untitled.docx")
    assert len(results) == 0


def test_extract_hierarchical_path_ymd(extractor):
    results = extractor.extract("2026/06/10/notes.md")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 6
    assert results[0].parsed_date.day == 10


def test_extract_nested_redundant_hierarchy(extractor):
    results = extractor.extract("2026/2026-09/2026-09-12/hello.txt")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 9
    assert results[0].parsed_date.day == 12


def test_extract_year_monthday_path(extractor):
    results = extractor.extract("2026/JUne14")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 6
    assert results[0].parsed_date.day == 14


def test_extract_iso_datetime_and_compact_timestamp(extractor):
    results = extractor.extract("audit_log_20260826190219.csv")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26
    assert results[0].parsed_date.hour == 19
    assert results[0].parsed_date.minute == 2
    assert results[0].parsed_date.second == 19


def test_extract_static_site_generator_slug_and_routes(extractor):
    results = extractor.extract("blog/2026-08-26-product-launch/index.md")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26

    res2 = extractor.extract("2026-08-26-welcome.markdown")
    assert len(res2) >= 1
    assert res2[0].parsed_date.year == 2026
    assert res2[0].parsed_date.month == 8
    assert res2[0].parsed_date.day == 26


def test_extract_iso_ordinal_and_week_date(extractor):
    # 2026-238 is Aug 26, 2026
    results = extractor.extract("data_export_2026-238.parquet")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26

    # 2026-W35-3 is Wednesday, Aug 26, 2026
    res_week = extractor.extract("sprint-2026-W35-3.docx")
    assert len(res_week) >= 1
    assert res_week[0].parsed_date.year == 2026
    assert res_week[0].parsed_date.month == 8
    assert res_week[0].parsed_date.day == 26


def test_extract_quarter_paths(extractor):
    results = extractor.extract("reports/2026/Q3/financials.xlsx")
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 7
    assert results[0].parsed_date.day == 1

    res_fy = extractor.extract("FY2026-Q3_summary.pdf")
    assert len(res_fy) >= 1
    assert res_fy[0].parsed_date.year == 2026
    assert res_fy[0].parsed_date.month == 7


@pytest.mark.parametrize(
    "filename,expected_year,expected_month,expected_day",
    [
        ("report.2024.05.20.txt", 2024, 5, 20),
        ("report_2024_05_20.txt", 2024, 5, 20),
        ("report 2024 05 20.txt", 2024, 5, 20),
        ("finance_2024-08.15 notes.txt", 2024, 8, 15),
        ("2026.June.14.pdf", 2026, 6, 14),
        ("2026_June_14.docx", 2026, 6, 14),
        ("2026 June 14 notes.txt", 2026, 6, 14),
        ("14.June.2026.pdf", 2026, 6, 14),
        ("14_June_2026.docx", 2026, 6, 14),
        ("14 June 2026.log", 2026, 6, 14),
        ("June_14_2026.md", 2026, 6, 14),
        ("June.14.2026.md", 2026, 6, 14),
    ],
)
def test_extract_various_delimiters(extractor, filename, expected_year, expected_month, expected_day):
    results = extractor.extract(filename)
    assert len(results) >= 1
    assert results[0].parsed_date.year == expected_year
    assert results[0].parsed_date.month == expected_month
    assert results[0].parsed_date.day == expected_day


def test_skip_ambiguous_six_digit(extractor):
    # Unclear 6-digit representations should be skipped
    assert len(extractor.extract("240312")) == 0
    assert len(extractor.extract("backup_240312.txt")) == 0
    assert len(extractor.extract("240312/notes.md")) == 0