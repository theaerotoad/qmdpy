from datetime import datetime
import pytest

from date_extractor.models import ExtractionSource
from date_extractor.text_extractor import TextDateExtractor


@pytest.fixture
def extractor():
    return TextDateExtractor()


def test_extract_labeled_header(extractor):
    text = (
        "Project Specification\n"
        "Date: 2024-06-18\n"
        "Author: Engineering Team\n\n"
        "This project defines the system architecture..."
    )
    results = extractor.extract(text, max_chars=500)
    assert len(results) >= 1
    first = results[0]
    assert first.parsed_date.year == 2024
    assert first.parsed_date.month == 6
    assert first.parsed_date.day == 18
    assert first.source == ExtractionSource.TEXT_HEADER


def test_extract_prose_ner_date(extractor):
    text = (
        "Meeting Minutes\n"
        "Recorded on January 15, 2024 in San Francisco.\n"
        "Attendees reviewed Q4 financial performance and roadmap."
    )
    results = extractor.extract(text, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2024
    assert results[0].parsed_date.month == 1
    assert results[0].parsed_date.day == 15


def test_respects_max_chars(extractor):
    # Place date beyond max_chars limit
    text = ("A" * 600) + "\nDate: 2024-01-01\n"
    results = extractor.extract(text, max_chars=500)
    assert len(results) == 0


@pytest.mark.parametrize(
    "header_text,expected_year,expected_month,expected_day",
    [
        ("Date: 2024.06.18\nAuthor: Engineering", 2024, 6, 18),
        ("Created: 2024_06_18\nAuthor: Engineering", 2024, 6, 18),
        ("Dated: 2024 06 18\nAuthor: Engineering", 2024, 6, 18),
        ("Published: 2024/06/18\nAuthor: Engineering", 2024, 6, 18),
        ("Meeting held on 15.January.2024 in SF.", 2024, 1, 15),
        ("Meeting held on 15_January_2024 in SF.", 2024, 1, 15),
        ("Meeting held on January.15.2024 in SF.", 2024, 1, 15),
        ("Meeting held on January_15_2024 in SF.", 2024, 1, 15),
    ],
)
def test_extract_text_with_various_delimiters(extractor, header_text, expected_year, expected_month, expected_day):
    results = extractor.extract(header_text, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == expected_year
    assert results[0].parsed_date.month == expected_month
    assert results[0].parsed_date.day == expected_day


def test_extract_frontmatter_split_fields(extractor):
    yaml_header = "---\ntitle: Document Title\nyear: 2026\nmonth: 8\nday: 26\n---"
    results = extractor.extract(yaml_header, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26


def test_extract_frontmatter_array_and_toml(extractor):
    toml_header = 'title = "Release"\ndate_parts = [2026, 8, 26]\nauthor = "Team"'
    results = extractor.extract(toml_header, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26


def test_extract_json_metadata_and_epoch_timestamp(extractor):
    json_header = '{\n  "title": "Metrics",\n  "timestamp": 1787796139000\n}'
    results = extractor.extract(json_header, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 27 or results[0].parsed_date.day == 26


def test_extract_rfc_and_asctime_headers(extractor):
    rfc_header = "Received: by mail.example.com; Wed, 26 Aug 2026 19:02:19 -0700\nFrom: sys@example.com"
    results = extractor.extract(rfc_header, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 2026
    assert results[0].parsed_date.month == 8
    assert results[0].parsed_date.day == 26

    asctime_header = "Log started on Wed Aug 26 19:02:19 2026 by daemon"
    res_asc = extractor.extract(asctime_header, max_chars=500)
    assert len(res_asc) >= 1
    assert res_asc[0].parsed_date.year == 2026
    assert res_asc[0].parsed_date.month == 8
    assert res_asc[0].parsed_date.day == 26


def test_extract_period_abbreviated_month(extractor):
    text = (
        "CONFIDENTIAL DISPATCH\n"
        "Date: Dec. 19, 1942.\n"
        "Headquarters, Pacific Fleet\n\n"
        "Operations summary report..."
    )
    results = extractor.extract(text, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 1942
    assert results[0].parsed_date.month == 12
    assert results[0].parsed_date.day == 19
    assert results[0].is_full_date is True


def test_extract_ordinal_historical_date(extractor):
    text = (
        "HISTORICAL RECORD\n"
        "Entered into registry on February 13th 1841. Signed by clerk.\n"
        "All property deeds verified."
    )
    results = extractor.extract(text, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 1841
    assert results[0].parsed_date.month == 2
    assert results[0].parsed_date.day == 13
    assert results[0].is_full_date is True


def test_extract_two_digit_year_numeric(extractor):
    text = (
        "MEMORANDUM\n"
        "Date: 7/2/92.\n"
        "From: Regional Office\n"
        "Subject: Budget Review"
    )
    results = extractor.extract(text, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == 1992
    assert results[0].parsed_date.month == 7
    assert results[0].parsed_date.day == 2
    assert results[0].is_full_date is True


@pytest.mark.parametrize(
    "snippet,expected_year,expected_month,expected_day",
    [
        ("Signed on Dec. 19, 1942. Transmitted via telegraph.", 1942, 12, 19),
        ("Recorded on February 13th 1841. Archived in library.", 1841, 2, 13),
        ("Meeting held on 7/2/92. Minutes attached below.", 1992, 7, 2),
        ("Dated this 13th day of February, 1841.", 1841, 2, 13),
        ("Filed on 19th of Dec., 1942.", 1942, 12, 19),
        ("Published on Sept. 3rd, 2021 by Editor.", 2021, 9, 3),
        ("Revision released on Dec. 19, '92.", 1992, 12, 19),
        ("Drafted on 07/02/92 in Chicago.", 1992, 7, 2),
        ("Created: 7-2-92\nAuthor: Finance", 1992, 7, 2),
        ("as of Dec. 19, 1942.", 1942, 12, 19),
        ("Dispatch sent on 19 June 1842 by courier.", 1842, 6, 19),
        ("Recorded on 15 Aug 2000 during session.", 2000, 8, 15),
        ("Event occurred 15 Aug. 2000 in London.", 2000, 8, 15),
        ("Notice issued on 19 June, 1842.", 1842, 6, 19),
    ],
)
def test_extract_common_document_header_formats(extractor, snippet, expected_year, expected_month, expected_day):
    results = extractor.extract(snippet, max_chars=500)
    assert len(results) >= 1
    assert results[0].parsed_date.year == expected_year
    assert results[0].parsed_date.month == expected_month
    assert results[0].parsed_date.day == expected_day
    assert results[0].is_full_date is True