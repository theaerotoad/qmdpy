"""
Document and Text Converters for QMD.

Converts non-markdown formats (.docx, .pptx, .xlsx, .csv, .html) into Markdown text.
"""
import csv
import io
import re
from pathlib import Path
from typing import Union, List

SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".txt",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html", ".htm",
    ".epub"
}


def is_supported_file(file_path: Union[str, Path]) -> bool:
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def _sanitize_text(text: str) -> str:
    """
    Sanitizes string by replacing invalid unicode surrogates (U+D800 - U+DFFF)
    with replacement characters so it can be encoded to UTF-8 without errors.
    """
    if not text:
        return ""
    try:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')
    except Exception:
        return text


def convert_to_markdown(file_path: Union[str, Path]) -> str:
    """
    Converts a supported document or text file to Markdown.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in {".md", ".markdown", ".txt"}:
        raw_md = _convert_text(path)
    elif ext == ".pdf":
        raw_md = _convert_pdf(path)
    elif ext == ".docx":
        raw_md = _convert_docx(path)
    elif ext == ".pptx":
        raw_md = _convert_pptx(path)
    elif ext == ".xlsx":
        raw_md = _convert_xlsx(path)
    elif ext == ".csv":
        raw_md = _convert_csv(path)
    elif ext in {".html", ".htm"}:
        raw_md = _convert_html(path)
    elif ext == ".epub":
        raw_md = _convert_epub(path)
    else:
        raw_md = _convert_text(path)

    return _sanitize_text(raw_md)


def _convert_text(path: Path) -> str:
    raw_bytes = path.read_bytes()

    # Reject binary files containing null bytes in header/first chunk
    if b'\x00' in raw_bytes[:8192]:
        raise ValueError(f"File '{path.name}' appears to be a binary file and cannot be converted as plain text.")

    for enc in ("utf-8", "utf-8-sig"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue

    try:
        return raw_bytes.decode("latin-1")
    except Exception as e:
        raise ValueError(f"Could not decode text file '{path.name}': {e}")


def _convert_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf
        except ImportError:
            raise ImportError("pypdf is required for converting .pdf files. Install with `pip install pypdf`.")

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file '{path.name}': {e}")

    md_lines: List[str] = []

    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        if text and text.strip():
            md_lines.append(f"## Page {i}\n")
            md_lines.append(text.strip() + "\n")

    return "\n".join(md_lines).strip()


def _convert_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for converting .docx files. Install with `pip install python-docx`.")

    doc = docx.Document(str(path))
    md_lines: List[str] = []

    for elem in doc.element.body:
        tag = elem.tag.split('}')[-1]
        if tag == 'p':
            p = docx.text.paragraph.Paragraph(elem, doc)
            text = p.text.strip()
            if not text:
                continue
            style_name = p.style.name.lower() if p.style else ""
            if "heading 1" in style_name:
                md_lines.append(f"# {text}\n")
            elif "heading 2" in style_name:
                md_lines.append(f"## {text}\n")
            elif "heading 3" in style_name:
                md_lines.append(f"### {text}\n")
            elif "heading 4" in style_name:
                md_lines.append(f"#### {text}\n")
            elif "heading" in style_name:
                md_lines.append(f"##### {text}\n")
            elif "list" in style_name or "bullet" in style_name:
                md_lines.append(f"- {text}")
            else:
                md_lines.append(f"{text}\n")

        elif tag == 'tbl':
            tbl = docx.table.Table(elem, doc)
            table_md = _format_matrix_to_md_table([
                [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                for row in tbl.rows
            ])
            if table_md:
                md_lines.append(table_md + "\n")

    return "\n".join(md_lines).strip()


def _convert_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required for converting .pptx files. Install with `pip install python-pptx`.")

    prs = Presentation(str(path))
    md_lines: List[str] = []

    for i, slide in enumerate(prs.slides, 1):
        slide_title = ""
        slide_texts = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if not text:
                    continue
                if shape == slide.shapes.title:
                    slide_title = text.replace('\n', ' ')
                else:
                    for paragraph in shape.text_frame.paragraphs:
                        p_text = paragraph.text.strip()
                        if p_text:
                            level = paragraph.level
                            indent = "  " * level
                            slide_texts.append(f"{indent}- {p_text}")
            elif shape.has_table:
                table_matrix = []
                for row in shape.table.rows:
                    table_matrix.append([cell.text.strip().replace('\n', ' ') for cell in row.cells])
                table_md = _format_matrix_to_md_table(table_matrix)
                if table_md:
                    slide_texts.append(table_md)

        header = f"## Slide {i}"
        if slide_title:
            header += f": {slide_title}"
        md_lines.append(header + "\n")

        if slide_texts:
            md_lines.extend(slide_texts)
            md_lines.append("")

    return "\n".join(md_lines).strip()


def _convert_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for converting .xlsx files. Install with `pip install openpyxl`.")

    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    md_lines: List[str] = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        md_lines.append(f"## Sheet: {sheet_name}\n")

        matrix = []
        for row in sheet.iter_rows(values_only=True):
            if not row or all(v is None for v in row):
                continue
            row_vals = [str(v).strip().replace('\n', ' ') if v is not None else "" for v in row]
            matrix.append(row_vals)

        table_md = _format_matrix_to_md_table(matrix)
        if table_md:
            md_lines.append(table_md + "\n")

    wb.close()
    return "\n".join(md_lines).strip()


def _convert_csv(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="replace")

    reader = csv.reader(io.StringIO(content))
    matrix = []
    for row in reader:
        if row and any(cell.strip() for cell in row):
            matrix.append([cell.strip().replace('\n', ' ') for cell in row])

    return _format_matrix_to_md_table(matrix)


def _convert_epub(path: Path) -> str:
    from qmd.epub import convert_epub_to_markdown
    try:
        return convert_epub_to_markdown(path)
    except Exception as e:
        raise ValueError(f"Failed to parse EPUB file '{path.name}': {e}")


def _convert_html(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="replace")

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required for converting .html files. Install with `pip install beautifulsoup4`.")

    soup = BeautifulSoup(content, 'html.parser')

    for element in soup(["script", "style", "head", "title", "meta", "noscript", "svg", "iframe"]):
        element.decompose()

    md_lines: List[str] = []

    def process_node(node):
        if isinstance(node, str):
            text = node.strip()
            if text:
                md_lines.append(text)
            return

        name = node.name.lower() if node.name else ""

        if name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(name[1])
            prefix = "#" * level
            md_lines.append(f"\n{prefix} {node.get_text(strip=True)}\n")
        elif name == "p":
            text = node.get_text(strip=True)
            if text:
                md_lines.append(f"\n{text}\n")
        elif name in ["ul", "ol"]:
            for li in node.find_all("li", recursive=False):
                li_text = li.get_text(strip=True)
                if li_text:
                    md_lines.append(f"- {li_text}")
            md_lines.append("")
        elif name == "table":
            matrix = []
            for tr in node.find_all("tr"):
                row = [td.get_text(strip=True).replace('\n', ' ') for td in tr.find_all(["th", "td"])]
                if row:
                    matrix.append(row)
            table_md = _format_matrix_to_md_table(matrix)
            if table_md:
                md_lines.append(f"\n{table_md}\n")
        elif name in ["pre", "code"]:
            code_text = node.get_text()
            md_lines.append(f"\n```\n{code_text}\n```\n")
        else:
            for child in node.children:
                process_node(child)

    process_node(soup.body or soup)

    result = "\n".join(md_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _format_matrix_to_md_table(matrix: List[List[str]]) -> str:
    if not matrix:
        return ""

    max_cols = max(len(row) for row in matrix)
    if max_cols == 0:
        return ""

    norm_matrix = []
    for row in matrix:
        norm_row = row + [""] * (max_cols - len(row))
        norm_row = [cell.replace("|", "\\|") for cell in norm_row]
        norm_matrix.append(norm_row)

    header = norm_matrix[0]
    separator = ["---"] * max_cols
    rows = norm_matrix[1:] if len(norm_matrix) > 1 else []

    header_line = "| " + " | ".join(header) + " |"
    sep_line = "| " + " | ".join(separator) + " |"

    body_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, sep_line] + body_lines)


def main():
    import argparse
    import sys

    # Ensure parent package directory (src) is in sys.path if running standalone script
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    parser = argparse.ArgumentParser(
        description="Convert a document (.docx, .pptx, .xlsx, .csv, .html, .epub, .md) to Markdown and inspect parsed blocks/chunks."
    )
    parser.add_argument("file_path", type=str, help="Path to the document to convert")
    parser.add_argument("-o", "--output", type=str, help="Optional output path to save the Markdown content")
    parser.add_argument("--show-blocks", action="store_true", help="Display parsed semantic blocks from docparse")
    parser.add_argument("--show-chunks", action="store_true", help="Display chunked content ready for embedding")

    args = parser.parse_args()

    file_path = Path(args.file_path).expanduser().resolve()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    if not is_supported_file(file_path):
        print(f"Warning: Extension '{file_path.suffix}' is not explicitly supported. Attempting plain text conversion...", file=sys.stderr)

    try:
        md_content = convert_to_markdown(file_path)
    except Exception as e:
        print(f"Error converting {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"CONVERTED MARKDOWN ({file_path.name})")
    print("=" * 80)
    print(md_content)
    print("=" * 80)
    print(f"Stats: {len(md_content)} characters, {len(md_content.splitlines())} lines")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(md_content, encoding="utf-8")
        print(f"\nSaved output to: {out_path}")

    if args.show_blocks or args.show_chunks:
        try:
            from qmd.docparse.parser import parse_markdown_to_blocks
            from qmd.docparse.grouper import group_blocks_into_chunks

            blocks, has_tables = parse_markdown_to_blocks(content=md_content)
            print("\n" + "=" * 80)
            print(f"PARSED SEMANTIC BLOCKS ({len(blocks)} blocks found, has_tables={has_tables})")
            print("=" * 80)
            for i, block in enumerate(blocks, 1):
                print(f" Block #{i} [H{block.header_level}: {block.header_text}] (line {block.start_line}):")
                preview = block.content[:150].replace('\n', ' ')
                print(f"   {preview}..." if len(block.content) > 150 else f"   {preview}")

            if args.show_chunks:
                chunks = group_blocks_into_chunks(blocks, max_chunk_size=2048, target_chunk_size=1024)
                print("\n" + "=" * 80)
                print(f"GENERATED CHUNKS ({len(chunks)} chunks generated)")
                print("=" * 80)
                for i, chunk in enumerate(chunks, 1):
                    headers = " > ".join(chunk.parent_headers.values()) if chunk.parent_headers else "Root"
                    print(f"\n--- Chunk #{i} (Context: {headers}) ---")
                    print(chunk.content)
        except Exception as e:
            print(f"\nError running docparse analysis: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
