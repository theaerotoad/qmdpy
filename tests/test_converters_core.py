import sys
from unittest.mock import MagicMock

# Preemptively stub onnxruntime before pymupdf4llm import to prevent native C-extension segfault
for _mod in ("onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from pathlib import Path
from qmd.converters import convert_to_markdown, _format_matrix_to_md_table, is_supported_file

def test_supported_extensions():
    assert is_supported_file("document.docx")
    assert is_supported_file("presentation.pptx")
    assert is_supported_file("sheet.xlsx")
    assert is_supported_file("data.csv")
    assert is_supported_file("page.html")
    assert is_supported_file("notes.md")
    assert is_supported_file("document.pdf")
    assert is_supported_file("book.epub")
    assert is_supported_file("book.mobi")

def test_condense_repeating_lines():
    from qmd.converters import _condense_repeating_lines
    
    # Test 1: Single line repeating
    text = "Hello\n" * 10
    condensed = _condense_repeating_lines(text, threshold=5)
    assert "Hello" in condensed
    assert "repeated 9 more times" in condensed
    assert len(condensed.splitlines()) < 10

    # Test 2: Multi-line block repeating
    block = "Line 1\nLine 2\n\n"
    text = block * 6
    condensed = _condense_repeating_lines(text, threshold=5)
    assert "Line 1" in condensed
    assert "Line 2" in condensed
    assert "repeated 5 more times" in condensed

    # Test 3: Below threshold, no change
    text = "Hello\n" * 4
    condensed = _condense_repeating_lines(text, threshold=5)
    assert condensed.strip() == "Hello\nHello\nHello\nHello"

    # Test 4: Does not mess up normal things
    text = "Line 1\nLine 2\nLine 3\nLine 1\nLine 4\n"
    condensed = _condense_repeating_lines(text, threshold=5)
    assert condensed.strip() == text.strip()

    # Test 5: Ignores blank lines repetition
    text = "A\n\n\n\n\n\n\n\nB"
    condensed = _condense_repeating_lines(text, threshold=5)
    # Should not condense purely empty blocks
    assert "repeated" not in condensed

    # Test 6: Multipass nested repetitions (macro blocks)
    inner = "Task A\n"
    macro1 = inner * 11 + "Task B\n"
    macro2 = inner * 15 + "Task B\n"
    # macro1 appears once, macro2 appears 6 times
    text = macro1 + macro2 * 6
    condensed = _condense_repeating_lines(text, threshold=5, max_block_size=200)
    
    # We should have one inner skip for macro1, one inner skip for the remaining macro2s, 
    # and one outer skip for the 6 repetitions of macro2.
    assert condensed.count("skipped") == 3

def test_format_matrix_to_md_table():
    matrix = [
        ["Header 1", "Header 2"],
        ["Value 1", "Value 2"]
    ]
    expected = "| Header 1 | Header 2 |\n| --- | --- |\n| Value 1 | Value 2 |"
    assert _format_matrix_to_md_table(matrix) == expected

def test_format_matrix_to_md_table_compression():
    matrix = [
        ["", "", "", "Col4", ""],
        ["", "", "", "", ""],
        ["", "", "", "", ""],
        ["", "", "", "", ""],
        ["", "", "", "Data", ""]
    ]
    md = _format_matrix_to_md_table(matrix)
    assert "<3 empty cols>" in md
    assert "<3 empty rows skipped>" in md
    assert "| <3 empty cols> | Col4 |  |" in md
    assert "| ... | <3 empty rows skipped> |  |" in md
    
    # Check that completely empty matrices return empty string
    empty_matrix = [["", ""], ["", ""]]
    assert _format_matrix_to_md_table(empty_matrix) == ""

def test_convert_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("Name,Age\nAlice,30\nBob,25\n", encoding="utf-8")

    md = convert_to_markdown(csv_file)
    assert "| Name | Age |" in md
    assert "| Alice | 30 |" in md
    assert "| Bob | 25 |" in md

def test_convert_html(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><h1>Title</h1><p>Hello world</p><ul><li>Item 1</li></ul></body></html>", encoding="utf-8")

    md = convert_to_markdown(html_file)
    assert "# Title" in md
    assert "Hello world" in md
    assert "- Item 1" in md

def test_convert_text_file(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Simple text file line 1\nLine 2\n", encoding="utf-8")

    md = convert_to_markdown(txt_file)
    assert "Simple text file line 1" in md

def test_pdf_fallback_extraction(tmp_path, monkeypatch):
    import pymupdf
    import pymupdf4llm
    from qmd.converters import convert_to_markdown

    pdf_path = tmp_path / "fallback_test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Fallback text output")
    doc.save(str(pdf_path))
    doc.close()

    def mock_to_markdown(*args, **kwargs):
        raise ValueError("Simulated Colorspace Error")

    monkeypatch.setattr(pymupdf4llm, "to_markdown", mock_to_markdown)

    errors = []
    md = convert_to_markdown(pdf_path, errors_out=errors)

    assert "Fallback text output" in md
    assert len(errors) == 1
    assert errors[0]["error_type"] == "pdf_fallback_used"
    assert "Simulated Colorspace Error" in errors[0]["message"]


def test_binary_file_rejection(tmp_path):
    bin_file = tmp_path / "test.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\x04\x05PDF-binary-junk")

    with pytest.raises(ValueError, match="appears to be a binary file"):
        convert_to_markdown(bin_file)

def test_sanitize_surrogates():
    from qmd.converters import _sanitize_text
    bad_str = "Hello \ud800 World"
    sanitized = _sanitize_text(bad_str)
    assert "Hello" in sanitized
    assert "World" in sanitized
    # Verify it encodes to UTF-8 without raising UnicodeEncodeError
    sanitized.encode("utf-8")

def test_math_formula_extraction():
    from lxml import etree
    from qmd.converters import _extract_text_and_math

    omml_xml = """
    <w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
         xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <w:r><w:t>Equation: </w:t></w:r>
      <m:oMath>
        <m:f>
          <m:num><m:r><m:t>1</m:t></m:r></m:num>
          <m:den><m:r><m:t>2</m:t></m:r></m:den>
        </m:f>
      </m:oMath>
      <w:r><w:t> is a half.</w:t></w:r>
    </w:p>
    """
    node = etree.fromstring(omml_xml)
    result = _extract_text_and_math(node)
    assert "Equation: " in result
    assert "$\\frac{1}{2}$" in result
    assert " is a half." in result

    mml_xml = """
    <w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
         xmlns:m="http://www.w3.org/1998/Math/MathML">
      <m:math>
        <m:mi>x</m:mi>
      </m:math>
    </w:p>
    """
    node_mml = etree.fromstring(mml_xml)
    result_mml = _extract_text_and_math(node_mml)
    assert "x" in result_mml or "$" in result_mml


def test_guess_document_date(tmp_path):
    from qmd.converters import guess_document_date

    # Path date
    f1 = tmp_path / "2024-08-20_report.txt"
    f1.write_text("Regular content without internal date.", encoding="utf-8")
    d1 = guess_document_date(f1, f1.read_text(encoding="utf-8"))
    assert d1 is not None
    assert "2024-08-20" in d1

    # Content date
    f2 = tmp_path / "meeting_notes.md"
    content2 = "---\ndate: 2025-01-15\n---\nDiscussion points."
    f2.write_text(content2, encoding="utf-8")
    d2 = guess_document_date(f2, content2)
    assert d2 is not None
    assert "2025-01-15" in d2

    # No date
    f3 = tmp_path / "random_document.txt"
    content3 = "Just some text without dates."
    f3.write_text(content3, encoding="utf-8")
    d3 = guess_document_date(f3, content3)
    assert d3 is None