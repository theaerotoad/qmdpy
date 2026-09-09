# Vision & Multimodal Junk Filter Specification

This document details the filtering heuristics applied by `_clean_vision_markdown` (in `src/qmd/converters/images.py`) to prune spurious lines, OCR artifacts, model decoding loops, and hallucinated noise from image processing output.

---

## 1. Overview & Purpose

When processing document images, charts, and slide screenshots, vision models and OCR endpoints often output decoding artifacts when encountering noisy scans, dense hatching, watermarks, decorative rules, or corrupt bounding boxes.

The junk filter evaluates text line-by-line, applying deterministic quality gates to remove non-informational noise while preserving legitimate Markdown syntax, mathematical notation, tables, and document outlines.

---

## 2. Protected Constructs (Exempt from Filtering)

The following structural blocks bypass line-level quality checks entirely:

* **Fenced Code Blocks:** Any content enclosed within triple backticks (` ``` `).
* **Valid Markdown Tables:** Rows delimited by pipes (`| ... |`), including table headers and separator rows (`|---|---|`), provided the row contains non-pipe text. Pipe spam (e.g. `|||||||`) is dropped.
* **Math Expressions:** Standalone display equations (`$$...$$`) and inline math blocks (`$...$`).
* **Standalone Links & Images:** Markdown image tags (`![alt](url)`) and links (`[label](url)`).
* **Thematic Breaks:** Markdown horizontal rules consisting of 3 or more hyphens, asterisks, or underscores (`---`, `***`, `___`).
* **Blank Lines:** Preserved to maintain document structure and paragraph separation.

---

## 3. Preprocessing & Normalization

Before evaluating rejection rules, lines undergo non-destructive preprocessing:

1. **Prefix Stripping for Analysis:** Common Markdown prefixes (`#`, `>`, `-`, `*`, `+`, `1.`) are stripped before evaluating text quality. Lines containing only a prefix with no content are discarded.
2. **Dot-Leader Normalization:** Lines containing four or more consecutive dots (common in Tables of Contents like `Chapter 1 .......... 15`) are condensed into standard ellipses (`Chapter 1 ... 15`) so dot leaders are not falsely flagged as character runs or symbol soup.

---

## 4. Rejection Criteria

Standard text lines are evaluated against the following heuristics and discarded if any rule matches:

| Check | Name | Description | Example Matches |
|---|---|---|---|
| **A** | **Zero Alphanumeric Characters** | The line contains no letters or numbers (`[a-zA-Z0-9]`) after removing valid Markdown prefixes. | `~_!@#$%>?^`, `.....::::::`, `(((((((((` |
| **B** | **Glitch & Non-Printable ASCII** | Contains 2+ Unicode replacement characters (`\ufffd`), replacement characters exceeding 10% of line length, or raw non-printable ASCII control characters (byte value < 32, excluding `\t` and `\n`). | `Line with \ufffd\ufffd artifacts`, embedded NULL bytes |
| **C** | **Excessive Symbol Density** | For lines with 8 or more non-whitespace characters, less than 25% of characters are alphanumeric (> 75% punctuation and symbols). | `!@#$%^&*()_+{}`, `- ????????`, `+ - / * = % & ~` |
| **D** | **Runaway Character Repetition** | 6 or more consecutive identical characters (`([^\s])\1{5,}`), indicating a model token stutter loop. | `((((((((((((((`, `===========`, `????????????` |
| **E** | **Consonant Gibberish** | Individual words/tokens of 12 or more characters that contain zero vowels (`aeiou`) or have an extreme vowel deficit (< 10% vowels). | `bcdfghjklmnpqrstvwxyz`, `strngthwthtflx` |
| **F** | **Repetition Hallucination Loops** | Exact consecutive duplicate lines (case-insensitive). Up to 2 consecutive occurrences are retained; 3rd and subsequent repetitions are discarded. | Repeated lines of `None`, `N/A`, or looped captions |

---

## 5. Verbose Diagnostics

When verbose mode is active (`--verbose`, `verbose: true` in config, or `QMD_VERBOSE=1`), filtered lines are logged in the inline debug block:

```markdown
> **[DEBUG: Multimodal LLM for `chart.png`]**
> **Model Output (512 chars):**
> Extracted chart text and notes...
> **[Junk Filter: Omitted 3 spurious line(s)]**
>   - `~_!@#$%>?^`
>   - `None`
>   - `None`

```

If all extracted content from an image fails validation, the cleaner outputs an empty string (`""`) so no spurious empty containers or broken formatting pollute the final indexed Markdown.
