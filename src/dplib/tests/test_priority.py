from datetime import datetime
import pytest

from date_extractor.models import ExtractionSource
from date_extractor.resolver import DateResolver


@pytest.fixture
def resolver():
    return DateResolver()


def test_priority_last_date_in_path_wins(resolver):
    # Path has multiple full dates: 2023-01-01 earlier and 2024-05-15 later in path
    path = "archive/2023-01-01/reports/2024-05-15/summary.txt"
    report = resolver.resolve(filename=path, text_content="No content dates.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 5
    assert report.resolved_date.day == 15
    assert report.primary_source == ExtractionSource.FILENAME


def test_priority_filename_date_over_parent_dir_date(resolver):
    # Directory date: 2023-11-20, Filename date: 2024-01-10 -> Filename date is last in path
    path = "2023-11-20/invoice_2024-01-10.pdf"
    report = resolver.resolve(filename=path, text_content="No dates.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 1
    assert report.resolved_date.day == 10


def test_priority_full_date_in_path_over_later_partial_date(resolver):
    # Path has full date 2024-08-15 and partial quarter date 2024-Q3 later
    path = "reports/2024-08-15/2024-Q3/summary.xlsx"
    report = resolver.resolve(filename=path, text_content="Quarterly overview.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 8
    assert report.resolved_date.day == 15


def test_priority_last_partial_date_in_path_when_no_full_dates(resolver):
    # Path has only partial dates: 2023-01 earlier vs 2024-08 later
    path = "backups/2023-01/2024-08/notes.txt"
    report = resolver.resolve(filename=path, text_content="No dates.")
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 8


def test_priority_path_date_over_document_date(resolver):
    # Filename specifies 2024-06-01, document text header says 2023-01-01
    path = "reports/2024-06-01/summary.txt"
    text = "Date: 2023-01-01\nOld template header."
    report = resolver.resolve(filename=path, text_content=text)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 6
    assert report.resolved_date.day == 1
    assert report.primary_source == ExtractionSource.FILENAME


def test_priority_first_date_in_document_when_no_path_date(resolver):
    # No date in path; document contains multiple sequential dates
    text = (
        "Project kickoff occurred on 2024-02-01 with stakeholders.\n"
        "Follow-up milestones scheduled for 2024-03-15 and launch on 2024-04-01."
    )
    report = resolver.resolve(filename="project_overview.txt", text_content=text)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 2
    assert report.resolved_date.day == 1
    assert report.primary_source == ExtractionSource.TEXT_HEADER


def test_priority_first_prose_date_wins(resolver):
    text = (
        "Recorded on January 10, 2024 during the executive sync.\n"
        "Next meeting planned for March 25, 2024."
    )
    report = resolver.resolve(filename="minutes.docx", text_content=text)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 1
    assert report.resolved_date.day == 10


def test_priority_agreement_boost_between_path_and_document(resolver):
    # Matching path date and document date gives agreement boost
    path = "report_2024-07-20.txt"
    text = "Published on 2024-07-20 by operations."
    report = resolver.resolve(filename=path, text_content=text)
    assert report.resolved_date is not None
    assert report.resolved_date.year == 2024
    assert report.resolved_date.month == 7
    assert report.resolved_date.day == 20
    assert report.confidence >= 0.98