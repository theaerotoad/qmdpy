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

def test_format_matrix_to_md_table():
    matrix = [
        ["Header 1", "Header 2"],
        ["Value 1", "Value 2"]
    ]
    expected = "| Header 1 | Header 2 |\n| --- | --- |\n| Value 1 | Value 2 |"
    assert _format_matrix_to_md_table(matrix) == expected

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
