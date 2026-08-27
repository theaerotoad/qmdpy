# Date Extractor

A robust Python library designed to detect, extract, and normalize dates from file paths, filenames, directory structures, and document header text.

The library combines regular expressions, spaCy Named Entity Recognition (NER), and dateparser to handle a wide variety of standard, human-written, and historical date formats commonly found in document headers and archival systems.

---

## Features

- Extracts dates from both filenames/directory paths and document header snippets.
- Supports historical dates spanning from 1000 to 2100.
- Handles standard ISO formats, RFC/HTTP headers, ANSI C asctime, fiscal quarters, ordinal dates, and ISO week dates.
- Understands human-written dates with period-abbreviated months (e.g., `Dec.`), ordinal suffixes (`1st`, `2nd`, `3rd`, `13th`), legal phrases (`13th day of February, 1841`), and 2-digit years (`7/2/92`, `'92`).
- Parses structured front matter including YAML/TOML split fields (`year`, `month`, `day`), date arrays (`[2026, 8, 26]`), and Unix epoch timestamps (seconds or milliseconds).
- Intelligent resolution engine with confidence scoring, full vs. partial date prioritization, and agreement boosting between filename and text content.
- English language support for all month names, day names, and relative phrases.

---

## Dependencies

The library requires Python 3.9+ and the following packages:

- `spacy >= 3.7.0`: Used for linguistic Named Entity Recognition (`DATE` entities).
- `dateparser >= 1.2.0`: Parses arbitrary natural language date strings.
- `python-dateutil >= 2.8.2`: Provides standard date arithmetic and parsing support.
- `pytest >= 8.0.0` (development/testing): Test runner for unit and integration suites.

Optional spaCy model:
- `en_core_web_sm`: If installed, `TextDateExtractor` will utilize it for NER. If not installed, it gracefully falls back to a blank English pipeline with a sentencizer.

Install dependencies using:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

---

## Architecture and Core Modules

```
date_extractor/
├── __init__.py
├── models.py              # Data structures: DateCandidate, DateExtractionReport, ExtractionSource
├── filename_extractor.py  # Extracts dates from filenames, paths, and parent directories
├── text_extractor.py      # Extracts dates from document headers, front matter, and prose
└── resolver.py            # Orchestrates extraction and resolves the primary date

```

### 1. Data Models (`date_extractor.models`)

* `ExtractionSource`: Enum indicating where a candidate was located (`FILENAME` or `TEXT_HEADER`).
* `DateCandidate`: Holds individual candidate metadata:
* `raw_text`: The matched substring.
* `parsed_date`: The normalized `datetime` instance.
* `source`: `ExtractionSource` enum value.
* `confidence`: Score between `0.0` and `1.0`.
* `start_char` / `end_char`: Character offsets within the input text.
* `is_full_date`: Boolean flag indicating whether year, month, and day are all present.


* `DateExtractionReport`: Final consolidated result containing:
* `filename`: Target file name or path.
* `resolved_date`: The primary chosen `datetime` object (or `None`).
* `primary_source`: Source of the winning candidate.
* `confidence`: Final confidence score (including agreement boosts).
* `candidates`: List of all discovered `DateCandidate` instances.
* `inspected_text_preview`: Truncated snippet of the inspected document header.



### 2. Date Resolver (`date_extractor.resolver.DateResolver`)

The resolver acts as the primary public API. It accepts a filename/path and optional text content (or reads directly from disk via `resolve_from_file`), runs both extractors, and applies the following resolution rules:

1. Prefers full dates (year + month + day) over partial dates (year + month or quarter).
2. Applies an agreement boost (+0.15 confidence) when the filename date and document body date point to the same calendar day.
3. Weighs source confidence and candidate positions (earlier in document text or deeper in filename path).

### 3. Text Extractor (`date_extractor.text_extractor.TextDateExtractor`)

Inspects up to a configurable maximum character limit (default 500 characters) at the start of a document. It executes an extraction pipeline:

1. Structured front matter (YAML/TOML/JSON fields, date arrays, numeric timestamps).
2. Explicit header labels (`Date:`, `Created:`, `Published:`, `Dated:`, `Recorded on:`, `as of:`).
3. Pre-compiled regex patterns for standard and human date formats.
4. spaCy NER entity detection.
5. Fallback date search via `dateparser.search.search_dates`.

### 4. Filename Extractor (`date_extractor.filename_extractor.FilenameDateExtractor`)

Inspects the entire file path, leaf filename, or directory hierarchy.

* Matches nested patterns like `archive/2026/08/26/file.pdf` or `2024/Q3/summary.docx`.
* Distinguishes compact numeric timestamps (`YYYYMMDDHHMMSS` and `YYYYMMDD`).
* Filters out ambiguous 6-digit sequences (e.g. `240312`) unless explicit 4-digit years are present.

---

## Supported Date Formats (English)

### Standard & Technical Formats

* ISO 8601 Calendar: `2024-08-15`, `2024/08/15`, `2024.08.15`, `2024_08_15`, `2024 08 15`
* ISO 8601 Datetime with Timezone: `2026-08-26T19:02:19Z`, `2026-08-26 15:02:19-04:00`, `2026-08-26 19:02:19`
* ISO 8601 Ordinal (Day of Year): `2026-238`, `2026_238`
* ISO 8601 Week Date: `2026-W35-3`, `2026-W35`
* Compact Timestamps: `20260826190219`, `20260826`
* RFC 5322 / RFC 1123 / HTTP: `Wed, 26 Aug 2026 19:02:19 -0700`, `Wed, 26 Aug 2026 19:02:19 GMT`
* ANSI C `asctime()`: `Wed Aug 26 19:02:19 2026`
* Quarters: `2026-Q3`, `FY2026-Q3`, `2026/Q3`, `2026_Q3`

### Human-Written & Historical Dates

* Month DD, YYYY: `August 26, 2026`, `Dec. 19, 1942`, `February 13th 1841`, `Sept. 3rd, 2021`, `Aug.26.2026`
* DD Month YYYY: `26 August 2026`, `19 June 1842`, `15 Aug 2000`, `15 Aug. 2000`, `26th Aug 2026`
* Legal & Formal Phrasing: `Dated this 13th day of February, 1841`, `Filed on 19th of Dec., 1942`
* Dates with Weekdays: `Wednesday, August 26, 2026`, `Sat, Dec. 19, 1942`
* Month Abbreviations: Full names (`January`..`December`), 3-letter abbreviations (`Jan`..`Dec`), period-suffixed (`Jan.`..`Dec.`), and 4-letter forms (`Sept`, `Sept.`)

### Two-Digit Years & Common Numeric Delimiters

* Numeric 2-Digit Years: `7/2/92`, `07/02/92`, `7-2-92`, `7.2.92`, `19/12/92`
* Written 2-Digit Years: `Dec. 19, '92`, `Dec 19, 92`
* Delimiters: Hyphens (`-`), slashes (`/`), dots (`.`), underscores (`_`), and whitespace (` `)

### Metadata & Front Matter

* Split YAML/TOML fields:
```yaml
---
year: 1942
month: 12
day: 19
---

```


* Date array:
```toml
date_parts = [1841, 2, 13]

```


* JSON metadata with Epoch timestamps (seconds or milliseconds):
```json
{
  "title": "Dispatch",
  "timestamp": 1787796139000
}

```



---

## Usage Examples

### 1. High-Level Resolution (`DateResolver`)

```python
from date_extractor.resolver import DateResolver

resolver = DateResolver()

# Resolve from filename and document text content
report = resolver.resolve(
    filename="dispatch_1942_12.txt",
    text_content="CONFIDENTIAL\nDate: Dec. 19, 1942.\nOperations report follows...",
)

print(f"Resolved Date: {report.resolved_date}")
# Output: Resolved Date: 1942-12-19 00:00:00

print(f"Primary Source: {report.primary_source}")
# Output: Primary Source: ExtractionSource.TEXT_HEADER

print(f"Confidence: {report.confidence}")

```

### 2. Resolving Directly from a File

```python
from pathlib import Path
from date_extractor.resolver import DateResolver

resolver = DateResolver()
report = resolver.resolve_from_file(Path("archive/1841/deeds/contract_feb13.txt"))

if report.resolved_date:
    print(f"Found date {report.resolved_date} with confidence {report.confidence:.2f}")

```

### 3. Extracting from Text Headers (`TextDateExtractor`)

```python
from date_extractor.text_extractor import TextDateExtractor

extractor = TextDateExtractor()
header_text = """
MEMORANDUM
Date: 7/2/92.
From: Regional Office
Subject: Quarterly Review
"""

candidates = extractor.extract(header_text, max_chars=500)
for c in candidates:
    print(f"Matched: '{c.raw_text}' -> {c.parsed_date.date()} (Full: {c.is_full_date}, Confidence: {c.confidence})")

```

### 4. Extracting from Filenames and Paths (`FilenameDateExtractor`)

```python
from date_extractor.filename_extractor import FilenameDateExtractor

extractor = FilenameDateExtractor()

paths = [
    "backups/2026/08/26/database.dump",
    "reports/FY2026-Q3_summary.pdf",
    "scans/14_June_2026_receipt.pdf",
    "letters/1841-02-13-official-statement.docx",
]

for p in paths:
    candidates = extractor.extract(p)
    if candidates:
        best = candidates[0]
        print(f"Path: {p} -> {best.parsed_date.date()} (Confidence: {best.confidence})")

```

---

## Running Tests

Execute the full test suite using `pytest`:

```bash
pytest

```

