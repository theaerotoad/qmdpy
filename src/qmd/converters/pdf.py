"""
PDF Document Converter and Heading Normalizer for QMD.
Integrates PyMuPDF and pymupdf4llm with automatic layout-aware fallback extraction.
"""
import contextlib
import io
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from .images import _is_image_processing_enabled, _process_images_concurrently

try:
    import pymupdf
    import pymupdf4llm
    HAS_PYMUPDF = True
    
    # Silence PyMuPDF C-level warnings globally
    if hasattr(pymupdf, "TOOLS"):
        pymupdf.TOOLS.mupdf_display_errors(False)
except ImportError:
    HAS_PYMUPDF = False
    pymupdf = None
    pymupdf4llm = None


def _get_converter_fn(name: str, fallback):
    mod = sys.modules.get("qmd.converters")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


def clean_heading_text(raw_heading: str) -> str | None:
    text = re.sub(r'<[^>]+>', '', raw_heading)
    text = re.sub(r'[*_~`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip("•·-–— \t")
    if not re.search(r'[a-zA-Z0-9]', text):
        return None
    return text


def normalize_markdown_headings(markdown_text: str, book_title: str | None = None) -> str:
    text = re.sub(r'^(?:#{1,6})\s+(\*\*(?:FIGURE|TABLE|EXHIBIT|CHART|STEP\s+\d+).*?\*\*)\s*$', r'\1', markdown_text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^#{1,6}\s+([A-Z])\s*$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^(?:#{1,6}\s+)?Chapter\s+(?:<u>)?([0-9]+|[IVXLCDM]+)(?:</u>)?\s*\n+(?:#{1,6}\s+)?([^\n]+)', r'## Chapter \1: \2', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^#{1,6}\s+([0-9Xx-]{10,17}|TERMS OF USE)\s*$', r'\1', text, flags=re.MULTILINE)

    cleaned_lines = []
    for line in text.splitlines():
        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if match:
            hashes, raw_heading = match.groups()
            clean_text = clean_heading_text(raw_heading)
            if clean_text is not None:
                cleaned_lines.append(f"{hashes} {clean_text}")
        else:
            cleaned_lines.append(line)

    lines = cleaned_lines
    first_heading_idx, first_heading_level, first_heading_text = None, None, None
    for idx, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if match:
            first_heading_idx = idx
            first_heading_level = len(match.group(1))
            first_heading_text = match.group(2).strip()
            break

    has_top_l1 = False
    if first_heading_idx is not None and first_heading_level == 1:
        if book_title:
            t_norm = book_title.lower().strip()
            h_norm = first_heading_text.lower().strip()
            if h_norm == t_norm or h_norm in t_norm or t_norm in h_norm:
                has_top_l1 = True
        else:
            if first_heading_idx == 0 or all(not l.strip() for l in lines[:first_heading_idx]):
                has_top_l1 = True

    stack = [(1, 1)] if (has_top_l1 or book_title) else [(0, 0)]
    final_lines = []
    if not has_top_l1 and book_title:
        final_lines.append(f"# {book_title}\n")

    for idx, line in enumerate(lines):
        if has_top_l1 and idx == first_heading_idx:
            final_lines.append(f"# {first_heading_text}")
            continue

        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if not match:
            final_lines.append(line)
            continue

        hashes, heading_content = match.groups()
        raw_lvl = len(hashes)
        if re.match(r'^Chapter\s+([0-9]+|[IVXLCDM]+):', heading_content, re.IGNORECASE):
            raw_lvl = 2

        if raw_lvl > stack[-1][0]:
            new_assigned = min(6, stack[-1][1] + 1)
            stack.append((raw_lvl, new_assigned))
        elif raw_lvl == stack[-1][0]:
            new_assigned = stack[-1][1]
        else:
            while len(stack) > 1 and stack[-1][0] > raw_lvl:
                stack.pop()
            if stack[-1][0] == raw_lvl:
                new_assigned = stack[-1][1]
            else:
                new_assigned = min(6, stack[-1][1] + 1)
                stack.append((raw_lvl, new_assigned))

        final_lines.append(f"{'#' * new_assigned} {heading_content.strip()}")

    result = "\n".join(final_lines).strip()
    return re.sub(r'\n{3,}', '\n\n', result)


def build_header_detector(doc, max_levels=5):
    toc = doc.get_toc()
    if toc:
        toc_by_page = {}
        for lvl, title, page_num in toc:
            p_idx = page_num - 1
            cleaned_title = title.strip().lower()
            if cleaned_title:
                toc_by_page.setdefault(p_idx, []).append((min(lvl, max_levels), cleaned_title))

        def toc_header_fn(span, page=None):
            if page is None: return ""
            page_num = page.number
            if page_num not in toc_by_page: return ""
            text = span.get("text", "").strip().lower()
            if len(text) <= 2: return ""
            for lvl, title in toc_by_page[page_num]:
                if text == title or text.startswith(title) or title.startswith(text):
                    return "#" * lvl + " "
            return ""
        return toc_header_fn

    font_sizes = Counter()
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if len(text) > 2:
                            size = round(span.get("size", 0))
                            font_sizes[size] += len(text)
    if not font_sizes:
        return None

    body_size = font_sizes.most_common(1)[0][0]
    larger_sizes = sorted([s for s in font_sizes.keys() if s > body_size], reverse=True)
    header_mapping = {size: min(idx + 1, max_levels) for idx, size in enumerate(larger_sizes[:max_levels])}

    def font_size_header_fn(span, page=None):
        text = span.get("text", "").strip()
        if len(text) <= 2: return ""
        size = round(span.get("size", 0))
        if size in header_mapping:
            return "#" * header_mapping[size] + " "
        return ""

    return font_size_header_fn


def _convert_pdf(path: Path, config=None, errors_out: Optional[List[dict]] = None) -> str:
    verbose = os.environ.get("QMD_VERBOSE") == "1"
    
    if not HAS_PYMUPDF:
        raise ImportError("pymupdf and pymupdf4llm are required for converting .pdf files. Install with `pip install pymupdf pymupdf4llm`.")

    if verbose:
        print(f"[Verbose PDF] Attempting to open {path} with pymupdf...", flush=True)

    doc = pymupdf.open(str(path))
    
    # 0. Sanitize the PDF to fix corrupted xrefs/colorspaces before pymupdf4llm chokes
    try:
        sanitized_bytes = doc.tobytes(garbage=4, clean=True, deflate=True)
        doc.close()
        doc = pymupdf.open(stream=sanitized_bytes, filetype="pdf")
        if verbose:
            print(f"[Verbose PDF] Successfully sanitized {path} in memory.", flush=True)
    except Exception as e:
        if verbose:
            print(f"[Verbose PDF] Pre-sanitization failed ({e}), proceeding with original.", flush=True)
        doc = pymupdf.open(str(path))

    if verbose:
        print(f"[Verbose PDF] Successfully opened {path}. Parsing headers...", flush=True)
        
    hdr_fn = build_header_detector(doc, max_levels=5)

    # 2. Extract markdown using pymupdf4llm (suppressing noisy OCR stdout)
    if verbose:
        print(f"[Verbose PDF] Executing pymupdf4llm.to_markdown on {path}...", flush=True)

    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                page_chunks = pymupdf4llm.to_markdown(
                    doc,
                    hdr_info=hdr_fn,
                    header=False,
                    footer=False,
                    write_images=False,
                    page_chunks=True,
                    show_progress=False
                )
            except TypeError:
                page_chunks = pymupdf4llm.to_markdown(
                    doc,
                    hdr_info=hdr_fn,
                    header=False,
                    footer=False,
                    write_images=False,
                    page_chunks=True
                )
        if verbose:
            print(f"[Verbose PDF] Successfully parsed PDF {path} to markdown.", flush=True)
    except Exception as e:
        if verbose:
            print(f"[Verbose PDF] pymupdf4llm failed ({e}). Falling back to layout-aware text extraction for {path}...", flush=True)
        if errors_out is not None:
            errors_out.append({"error_type": "pdf_fallback_used", "message": f"{path.name}: {e}"})
        
        page_chunks = []
        for i in range(len(doc)):
            page = doc[i]
            page_md = []
            try:
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") == 0:  # text block
                        block_lines = []
                        header_prefix = ""
                        for line in block.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                text = span.get("text", "")
                                if text.strip():
                                    prefix = hdr_fn(span, page) if hdr_fn else ""
                                    if prefix and not header_prefix:
                                        header_prefix = prefix
                                    line_text += text
                            if line_text.strip():
                                block_lines.append(line_text.strip())
                        
                        if block_lines:
                            merged = " ".join(block_lines)
                            if header_prefix:
                                page_md.append(f"\n{header_prefix}{merged}\n")
                            else:
                                page_md.append(f"{merged}\n")
                
                final_page_text = "\n".join(page_md) if page_md else page.get_text("text")
            except Exception:
                final_page_text = page.get_text("text")
                
            page_chunks.append({"text": final_page_text})

        if errors_out is not None and not any(chunk.get("text", "").strip() for chunk in page_chunks):
            errors_out.append({
                "error_type": "pdf_extraction_failed",
                "message": f"{path.name}: pymupdf4llm failed ({e}) and layout fallback extraction yielded no text"
            })
    
    seen_xrefs = set()
    
    if isinstance(page_chunks, str):
        page_chunks = [{"text": page_chunks}]

    md_pages = []
    for i, chunk in enumerate(page_chunks):
        page_text = chunk.get("text", "")
        
        def _format_pic_text(match):
            text = match.group(1).strip()
            text = re.sub(r'\s+', ' ', text)
            alt = f"Image with text: {text}" if text else "Image"
            return f"\n![{alt}](pdf_image.png)\n"
            
        page_text = re.sub(r'<!--\s*Start of picture text\s*-->(.*?)<!--\s*End of picture text\s*-->\n*', _format_pic_text, page_text, flags=re.DOTALL | re.IGNORECASE)
        page_text = re.sub(r'<!--.*?-->\n*', '', page_text, flags=re.DOTALL)
        md_pages.append(page_text)

    img_enabled_fn = _get_converter_fn("_is_image_processing_enabled", _is_image_processing_enabled)
    proc_imgs_fn = _get_converter_fn("_process_images_concurrently", _process_images_concurrently)

    if config and img_enabled_fn(config):
        try:
            pdf_images = []
            for i in range(len(doc)):
                page = doc[i]
                for img in page.get_images():
                    xref = img[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception:
                        continue
                    if not base_image:
                        continue
                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width > 0 and height > 0 and (width <= 2 or height <= 2):
                        continue
                    ext = base_image.get("ext", "png")
                    filename = f"page_{i+1}_img_{xref}.{ext}"
                    pdf_images.append((i, image_bytes, filename))

            if pdf_images:
                image_inputs = [(img_bytes, fn) for _, img_bytes, fn in pdf_images]
                results = proc_imgs_fn(image_inputs, config, errors_out=errors_out)
                for (page_idx, _, _), img_md in zip(pdf_images, results):
                    if img_md and page_idx < len(md_pages):
                        md_pages[page_idx] += f"\n\n{img_md}\n"
        except Exception:
            pass
        
    raw_md = "\n\n".join(md_pages)
    doc.close()

    stem = re.sub(r'^[a-z0-9.]+[_-]', '', path.stem, flags=re.IGNORECASE)
    book_title = stem.replace("-", " ").replace("_", " ").title()

    return normalize_markdown_headings(raw_md, book_title=book_title)