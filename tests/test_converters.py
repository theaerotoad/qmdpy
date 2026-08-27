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

def test_convert_epub(tmp_path):
    import zipfile
    epub_file = tmp_path / "test.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book Title</dc:title>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
  </spine>
</package>"""

    ch1_xhtml = """<!DOCTYPE html>
<html>
<head><title>Chapter 1</title></head>
<body>
  <h1>Chapter 1</h1>
  <p>This is a test paragraph in the EPUB.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)

    md = convert_to_markdown(epub_file)
    assert "Test Book Title" in md or "Chapter 1" in md
    assert "This is a test paragraph in the EPUB." in md

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

def test_process_image_routing_and_concurrency(monkeypatch):
    from qmd.converters import _process_image, _process_images_concurrently
    from qmd.config import Config

    calls = []

    def mock_multimodal(image_bytes, filename, config):
        calls.append(("multimodal", filename))
        return f"# MD for {filename}"

    def mock_vision(image_bytes, filename, config):
        calls.append(("vision", filename))
        return f"# Vision for {filename}"

    monkeypatch.setattr("qmd.converters._process_image_multimodal_llm", mock_multimodal)
    monkeypatch.setattr("qmd.converters._process_image_vision_api", mock_vision)

    cfg_multi = Config.from_dict({
        "multimodal_model": "gpt-4o-mini",
        "multimodal_url": "http://127.0.0.1:8888",
        "max_image_concurrency": 2
    })
    res_multi = _process_image(b"pngbytes", "img1.png", cfg_multi)
    assert res_multi == "# MD for img1.png"
    assert calls[-1] == ("multimodal", "img1.png")

    cfg_vision = Config.from_dict({
        "vision_url": "http://127.0.0.1:8891/detect"
    })
    res_vision = _process_image(b"pngbytes", "img2.png", cfg_vision)
    assert res_vision == "# Vision for img2.png"
    assert calls[-1] == ("vision", "img2.png")

    items = [(b"b1", "a.png"), (b"b2", "b.png"), (b"b3", "c.png")]
    concurrent_results = _process_images_concurrently(items, cfg_multi, max_workers=2)
    assert len(concurrent_results) == 3
    assert concurrent_results[0] == "# MD for a.png"
    assert concurrent_results[1] == "# MD for b.png"
    assert concurrent_results[2] == "# MD for c.png"


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


def test_converter_main_inferred_date_output(tmp_path, capsys, monkeypatch):
    import sys
    from qmd.converters import main

    f = tmp_path / "2026-03-30_summary.md"
    f.write_text("# Project Summary\nAll done.", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["converters.py", str(f)])
    main()

    captured = capsys.readouterr()
    assert "Inferred Date:" in captured.out
    assert "2026-03-30" in captured.out
