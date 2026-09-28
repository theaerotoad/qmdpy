# QMD Custom Report Builder Specification

This document outlines the architecture and syntax for the next phase of the `qmd report` command, expanding it from a simple analysis viewer into a powerful data extraction and reporting pipeline. 

The goal is to allow users (and LLM agents) to construct highly specific datasets across federated databases, outputting them to CLI, XML, JSON, or CSV.

## 1. Quick CLI Mode (Simple Reports)

For fast, ad-hoc reports, users can specify fields and sorting directly via CLI flags.

**Syntax:**
`qmd report [target] --fields="field1[+|-],field2[+|-]" --[xml|csv|json]`

**Modifiers:**
* `+` : Sort Ascending
* `-` : Sort Descending

**Examples:**
* `qmd report -c "Docs" --fields="path,altTitle,dates-,authors"` (Sorts by date descending)
* `qmd report -p "Papers" --fields="type+,chunk_count-" --csv > report.csv` (Primary sort type ascending, secondary sort chunk count descending)

---

## 2. YAML Configuration Mode (Detailed Reports)

For deep data extraction, LLM-based targeted queries, and complex reporting pipelines, users can pass a YAML configuration file.

**Syntax:**
`qmd report -y my_report.yml --csv > output.csv`

### Example `report.yml`

```yaml
name: "New Deal Perspectives Review"
description: "Extracts author opinions and disagreements across historical papers."

# 1. Target Scope
target:
  collection: "Papers"
  path: "1930s/"  # Optional substring filter

# 2. Sorting Rules
sort:
  - field: "dates"
    order: "asc"
  - field: "altTitle"
    order: "asc"

# 3. Base Fields (Standard DB / Analysis columns)
fields:
  - collection
  - path
  - dates
  - altTitle
  - authors
  - summary

# 4. Dynamic Queries (Scoped to EACH document)
dynamic_queries:
  # Type A: LLM Answer (Uses quick_answer or analyze prompt pattern scoped to the doc)
  - id: "author_opinion"
    type: "llm_answer"
    prompt: "What does the author think of the new deal?"
    fallback: "Not mentioned"
  
  # Type B: Top Extracted Snippets (Uses standard hybrid search scoped to the doc)
  - id: "disagreements"
    type: "search_snippets"
    query: "Examples of major disagreement"
    limit: 3               # Return top 3 chunks
    format: "bulleted"     # Combine into a single string for the CSV cell

```

---

## 3. Available Base Fields

These fields require no additional LLM calls; they are pulled directly from the `documents`, `content`, and `document_analysis` SQL tables.

### System Fields

* `collection` : The source collection name.
* `path` : The relative path to the file.
* `hash` : Document content hash.
* `chunk_count` : Total number of chunks in the document.
* `last_indexed` : Timestamp of last index update.

### Analysis Fields (Requires prior `qmd analyze`)

* `altTitle` : LLM inferred descriptive title.
* `doc_type` : Inferred document type (e.g., transcript, letter, research paper).
* `summary` : Brief executive summary.
* `authors` : Array of extracted author names.
* `tags` : Array of semantic tags.
* `dates` : Array of inferred contextual dates.
* `questions` : Array of generated follow-up questions.

---

## 4. Output Formats

1. **CLI Pretty Print (Default):** A clean, readable terminal output detailing the target file, base fields, and any dynamic query results underneath.
2. **XML (`--xml`):** Highly structured output optimized for an LLM agent to ingest.
3. **JSON (`--json`):** Standard list of dictionaries for piping to other tools or web UIs.
4. **CSV (`--csv`):** Flattened table format.
* *Note on arrays/lists (like `authors` or `dynamic_queries` results):* In CSV mode, these will be automatically joined by semicolons (e.g., `Author A; Author B`) to ensure standard CSV compliance without breaking columns.

---

## 5. Next Implementation Steps

1. **CLI Argument Parser Update:** Add `--fields`, `--csv`, and `-y/--yaml` arguments to `report_parser`.
2. **Data Fetcher Refactoring:** Update `get_analysis_report` (or create a new `build_custom_report` method in the `AnalysisMixin`) to dynamically select, filter, and sort standard database columns.
3. **Dynamic Query Execution:** Add logic to iterate over documents and run scoped searches (`search_snippets`) or targeted extractions (`llm_answer`) per document.
4. **CSV Formatter:** Add `format_report_csv` to `qmd.formatting`.
