import sys
from unittest.mock import MagicMock

# Preemptively stub onnxruntime before pymupdf4llm import to prevent native C-extension segfault
for _mod in ("onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from pathlib import Path
from qmd.converters import convert_to_markdown

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

def test_convert_epub_with_ncx_toc(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_ncx.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample NCX Book</dc:title>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>"""

    toc_ncx = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="np-1" playOrder="1">
      <navLabel><text>Chapter 1: The Beginning</text></navLabel>
      <content src="ch1.xhtml"/>
      <navPoint id="np-2" playOrder="2">
        <navLabel><text>Section 1.1: Foundations</text></navLabel>
        <content src="ch1.xhtml#sec1_1"/>
      </navPoint>
    </navPoint>
    <navPoint id="np-3" playOrder="3">
      <navLabel><text>Chapter 2: The Next Step</text></navLabel>
      <content src="ch2.xhtml"/>
    </navPoint>
  </navMap>
</ncx>"""

    ch1_xhtml = """<!DOCTYPE html>
<html>
<body>
  <p class="title">Chapter 1: The Beginning</p>
  <p>Introductory text for chapter one.</p>
  <div id="sec1_1">
    <p class="subtitle">Section 1.1: Foundations</p>
    <p>Foundation details.</p>
  </div>
</body>
</html>"""

    ch2_xhtml = """<!DOCTYPE html>
<html>
<body>
  <p>Chapter 2: The Next Step</p>
  <p>Next step details.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/toc.ncx", toc_ncx)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)
        z.writestr("OEBPS/ch2.xhtml", ch2_xhtml)

    md = convert_to_markdown(epub_file)
    assert "# Sample NCX Book" in md
    assert "## Chapter 1: The Beginning" in md
    assert "### Section 1.1: Foundations" in md
    assert "## Chapter 2: The Next Step" in md
    assert "Introductory text for chapter one." in md
    assert "Foundation details." in md

def test_convert_epub_with_nav_xhtml_toc(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_nav.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>EPUB 3 Guide</dc:title>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="content.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
  </spine>
</package>"""

    nav_xhtml = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
      <li>
        <a href="content.xhtml#p1">Part 1: Overview</a>
        <ol>
          <li><a href="content.xhtml#sec1">Chapter 1: Getting Started</a></li>
        </ol>
      </li>
    </ol>
  </nav>
</body>
</html>"""

    content_xhtml = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
  <section id="p1">
    <h1>Part 1: Overview</h1>
    <p>Part 1 text.</p>
    <section id="sec1">
      <h1>Chapter 1: Getting Started</h1>
      <p>Getting started body text.</p>
    </section>
  </section>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/package.opf", package_opf)
        z.writestr("OEBPS/nav.xhtml", nav_xhtml)
        z.writestr("OEBPS/content.xhtml", content_xhtml)

    md = convert_to_markdown(epub_file)
    assert "# EPUB 3 Guide" in md
    assert "## Part 1: Overview" in md
    assert "### Chapter 1: Getting Started" in md
    assert "Getting started body text." in md

def test_convert_epub_deduplicates_repeated_headings_and_titles(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_dedup.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample Engineering Guide</dc:title>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="p1" href="part1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="p1"/>
    <itemref idref="c1"/>
  </spine>
</package>"""

    toc_ncx = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="np-1" playOrder="1">
      <navLabel><text>PART I: SYSTEM DESIGN</text></navLabel>
      <content src="part1.xhtml"/>
    </navPoint>
    <navPoint id="np-2" playOrder="2">
      <navLabel><text>CHAPTER 1: Foundational Architecture</text></navLabel>
      <content src="ch1.xhtml"/>
    </navPoint>
  </navMap>
</ncx>"""

    part1_xhtml = """<!DOCTYPE html>
<html>
<body>
  <p class="part-title">PART I: SYSTEM DESIGN</p>
</body>
</html>"""

    ch1_xhtml = """<!DOCTYPE html>
<html>
<body>
  <p class="chapter-title">CHAPTER 1: Foundational Architecture</p>
  <p>CHAPTER 1: Foundational Architecture</p>
  <p>System components are structured to ensure high availability and maintainability.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/toc.ncx", toc_ncx)
        z.writestr("OEBPS/part1.xhtml", part1_xhtml)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)

    md = convert_to_markdown(epub_file)
    assert md.count("PART I: SYSTEM DESIGN") == 1
    assert md.count("CHAPTER 1: Foundational Architecture") == 1
    assert "## PART I: SYSTEM DESIGN" in md
    assert "## CHAPTER 1: Foundational Architecture" in md
    assert "System components are structured to ensure high availability and maintainability." in md

def test_convert_epub_promotes_all_caps_subheadings_and_preserves_quote_attributions(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_caps_headings.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Modular Software Patterns</dc:title>
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
<body>
  <h2>Chapter 1: Principles</h2>
  <p>Understanding modular patterns helps organize large applications.</p>
  <p>“Simplicity is prerequisite for reliability.”</p>
  <p>EDSGER W. DIJKSTRA</p>
  <p>As software systems scale, clarity becomes more critical than cleverness.</p>
  <p>CORE SYSTEM CONSTRAINTS</p>
  <p>The primary constraint in distributed design is network latency and partition tolerance.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)

    md = convert_to_markdown(epub_file)
    # Quote attribution should remain plain body text
    assert "EDSGER W. DIJKSTRA" in md
    assert "# EDSGER W. DIJKSTRA" not in md
    assert "## EDSGER W. DIJKSTRA" not in md
    assert "### EDSGER W. DIJKSTRA" not in md

    # Standalone uppercase section line should be promoted to a heading
    assert "### CORE SYSTEM CONSTRAINTS" in md or "## CORE SYSTEM CONSTRAINTS" in md
    assert "The primary constraint in distributed design" in md

def test_convert_epub_heading_gap_compression(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_gap_compression.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Negotiation Strategy Guide</dc:title>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>"""

    ch1_xhtml = """<!DOCTYPE html>
<html>
<body>
  <h2>1. Thinking Differently</h2>
  <h4>HOW THIS BOOK IS DIFFERENT</h4>
  <p>First approach details.</p>
  <h4>INVISIBILITY</h4>
  <p>Second approach details.</p>
</body>
</html>"""

    ch2_xhtml = """<!DOCTYPE html>
<html>
<body>
  <h2>2. People Are Everything</h2>
  <p>Focus on relationships.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)
        z.writestr("OEBPS/ch2.xhtml", ch2_xhtml)

    md = convert_to_markdown(epub_file)
    assert "# Negotiation Strategy Guide" in md
    assert "## 1. Thinking Differently" in md
    assert "### HOW THIS BOOK IS DIFFERENT" in md
    assert "### INVISIBILITY" in md
    assert "#### HOW THIS BOOK IS DIFFERENT" not in md
    assert "## 2. People Are Everything" in md

def test_convert_epub_merges_broken_title_lines_and_connector_headings(tmp_path):
    import zipfile
    epub_file = tmp_path / "test_merged_headings.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Principles of Modern Systems</dc:title>
  </metadata>
  <manifest>
    <item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="title"/>
    <itemref idref="ch1"/>
  </spine>
</package>"""

    title_xhtml = """<!DOCTYPE html>
<html>
<body>
  <p>Growth for the</p>
  <p>Nine System Patterns</p>
  <h2>ALICE W. THURSTON</h2>
  <h2>and</h2>
  <h2>BOB M. CARPENTER</h2>
</body>
</html>"""

    ch1_xhtml = """<!DOCTYPE html>
<html>
<body>
  <h4>EACH PATTERN HAS UNIQUE ADVANTAGES—</h4>
  <h4>AND PREDICTABLE TRADEOFFS</h4>
  <p>System components should be decomposed according to domain boundaries.</p>
</body>
</html>"""

    with zipfile.ZipFile(epub_file, 'w') as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/title.xhtml", title_xhtml)
        z.writestr("OEBPS/ch1.xhtml", ch1_xhtml)

    md = convert_to_markdown(epub_file)
    assert "Growth for the Nine System Patterns" in md
    assert "## ALICE W. THURSTON and BOB M. CARPENTER" in md
    assert "## and\n" not in md
    assert "EACH PATTERN HAS UNIQUE ADVANTAGES— AND PREDICTABLE TRADEOFFS" in md

def test_convert_mobi(tmp_path, monkeypatch):
    class MockMobi:
        def extract(self, path):
            tempdir = tmp_path / "mobi_temp"
            tempdir.mkdir(exist_ok=True)
            
            # Simulate a MOBI7 extraction returning an HTML file
            html_file = tempdir / "extracted.html"
            html_content = '''<!DOCTYPE html>
            <html>
            <body>
              <p><b><i>The Awakening</i></b></p>
              <p>Body text to separate the headings.</p>
              <p><b>Self-Assessment</b></p>
              <p>More body text to separate the headings.</p>
              <p>MOBI INTRODUCTION</p>
              <p>This is a test paragraph from a mobi conversion.</p>
            </body>
            </html>'''
            html_file.write_text(html_content, encoding="utf-8")

            return str(tempdir), str(html_file)

    import sys
    monkeypatch.setitem(sys.modules, 'mobi', MockMobi())
    
    mobi_file = tmp_path / "test.mobi"
    mobi_file.write_bytes(b"dummy mobi content")
    
    from qmd.converters import convert_to_markdown
    md = convert_to_markdown(mobi_file)
    
    # Ensure EPUB parsing logic ran and MOBI faked headings were promoted
    assert "The Awakening" in md
    assert "Self-Assessment" in md
    assert "MOBI INTRODUCTION" in md
    assert "This is a test paragraph from a mobi conversion." in md
    # Check that they were converted to structural headings
    assert "# The Awakening" in md
    assert "## Self-Assessment" in md