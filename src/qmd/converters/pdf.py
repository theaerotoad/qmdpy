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

from io import BytesIO
import base64
import httpx
from PIL import Image

try:
    from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

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


def pil_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_images_from_pdf(doc, dpi: int = 150) -> List[Image.Image]:
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def crop_element(img: Image.Image, bbox) -> Image.Image:
    """Safely crop an element using its Docling BoundingBox, accounting for coordinate origin."""
    origin_str = str(getattr(bbox, "coord_origin", "TOPLEFT")).upper()
    
    if "BOTTOMLEFT" in origin_str:
        t = img.height - bbox.t
        b = img.height - bbox.b
        return img.crop((int(bbox.l), int(min(t, b)), int(bbox.r), int(max(t, b))))
    else:
        return img.crop((int(bbox.l), int(bbox.t), int(bbox.r), int(bbox.b)))

def adjust_markdown_headers_docling(markdown_text: str, doc) -> str:
    header_heights = {}
    
    for item, _ in doc.iterate_items():
        if getattr(item, "label", "") == "section_header" and hasattr(item, "text"):
            prov = item.prov[0] if item.prov else None
            if prov and hasattr(prov, "bbox"):
                height = abs(prov.bbox.b - prov.bbox.t)
                header_heights[item.text.strip()] = height
                
    if not header_heights:
        return markdown_text
        
    tolerance = 8.0
    sorted_heights = sorted(list(header_heights.values()), reverse=True)
    clusters = []
    current_cluster = [sorted_heights[0]]
    for h in sorted_heights[1:]:
        if current_cluster[0] - h < tolerance:
            current_cluster.append(h)
        else:
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [h]
    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))
        
    def get_level(h):
        closest_i = 0
        min_diff = float('inf')
        for i, cluster_h in enumerate(clusters):
            diff = abs(cluster_h - h)
            if diff < min_diff:
                min_diff = diff
                closest_i = i
                
        if min_diff <= tolerance:
            return min(closest_i + 1, 6)
        return 2 
        
    lines = markdown_text.split('\n')
    for i, line in enumerate(lines):
        match = re.match(r'^(#+)\s+(.*)', line)
        if match:
            text = match.group(2).strip()
            if text in header_heights:
                lvl = get_level(header_heights[text])
                lines[i] = ('#' * lvl) + ' ' + text
    return '\n'.join(lines)

def _call_docling_model(img: Image.Image, api_url: str, api_key: str, model_name: str, timeout: float = 120.0) -> str:
    base64_img = pil_to_base64(img)
    
    img_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
    }
    text_content = {"type": "text", "text": "Convert this page to docling."}
    
    content_array = [text_content, img_content]
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content_array
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }
    
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "sk-no-key-required":
        headers["Authorization"] = f"Bearer {api_key}"
        
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        if not api_url.endswith("/v1/chat/completions"):
            post_url = api_url.rstrip("/") + "/v1/chat/completions"
        else:
            post_url = api_url
            
        response = client.post(post_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


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

    pdf_engine = getattr(config, "pdf_engine", "pymupdf") if config else "pymupdf"

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

    if pdf_engine.lower() == "docling":
        if not HAS_DOCLING:
            raise ImportError("docling-core is required for the docling pdf_engine. Install with `pip install docling-core`.")
        
        docling_url = getattr(config, "docling_url", None)
        if not docling_url:
            raise ValueError("docling_url must be provided in config when using pdf_engine: docling")
            
        docling_api_key = getattr(config, "docling_api_key", "sk-no-key-required")
        docling_model = getattr(config, "docling_model", "Docling")
        timeout = getattr(config, "request_timeout", 120.0)

        if verbose:
            print(f"[Verbose PDF] Extracting images from {path} for Docling...", flush=True)
            
        images = extract_images_from_pdf(doc)
        all_doctags = []
        
        for idx, img in enumerate(images):
            if verbose:
                print(f"[Verbose PDF] Requesting DocTags for page {idx + 1} from {docling_model} at {docling_url}...", flush=True)
            try:
                doctags = _call_docling_model(img, docling_url, docling_api_key, docling_model, timeout=timeout)
                all_doctags.append(doctags)
            except Exception as e:
                if verbose:
                    print(f"[Verbose PDF] Docling extraction failed for page {idx + 1}: {e}", flush=True)
                all_doctags.append("")
            
        if verbose:
            print(f"[Verbose PDF] Assembling Markdown with docling-core...", flush=True)
            
        try:
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_doctags, images)
            doc_obj = DoclingDocument.load_from_doctags(doctag_document=doctags_doc, document_name=path.stem)
            
            raw_md = doc_obj.export_to_markdown()
            raw_md = adjust_markdown_headers_docling(raw_md, doc_obj)

            img_enabled_fn = _get_converter_fn("_is_image_processing_enabled", _is_image_processing_enabled)
            proc_imgs_fn = _get_converter_fn("_process_images_concurrently", _process_images_concurrently)

            if config and img_enabled_fn(config):
                try:
                    pdf_images = []
                    items_with_images = []
                    visual_count = 0
                    for item, _ in doc_obj.iterate_items():
                        if item.label in ["picture", "chart"]:
                            prov = item.prov[0] if item.prov else None
                            if not prov:
                                continue
                            
                            visual_count += 1
                            page_no = prov.page_no
                            if page_no < 1 or page_no > len(images):
                                continue
                                
                            source_img = images[page_no - 1]
                            cropped_img = crop_element(source_img, prov.bbox)
                            
                            buffered = BytesIO()
                            cropped_img.save(buffered, format="PNG")
                            image_bytes = buffered.getvalue()
                            
                            filename = f"page_{page_no}_{item.label}_{visual_count}.png"
                            pdf_images.append((image_bytes, filename))
                            items_with_images.append(item.label)

                    if pdf_images:
                        results = proc_imgs_fn(pdf_images, config, errors_out=errors_out)
                        for item_label, vision_md in zip(items_with_images, results):
                            if vision_md:
                                replacement = f"\n\n{vision_md}\n\n"
                            else:
                                replacement = ""
                            
                            # Replace the first occurrence of the image/chart placeholder
                            raw_md, count = re.subn(r'<!-*\s*(image|chart)\s*-*>', replacement, raw_md, count=1)
                            if count == 0 and vision_md:
                                # Fallback if placeholder not found
                                raw_md += replacement
                except Exception as e:
                    if verbose:
                        print(f"[Verbose PDF] Docling image extraction/processing failed: {e}", flush=True)

            # Remove any remaining unhandled docling image placeholders
            raw_md = re.sub(r'<!-*\s*(image|chart)\s*-*>', '', raw_md)

            # Process full page embeddings if enabled
            if config and getattr(config, "embed_pages", False):
                try:
                    import hashlib
                    page_images = []
                    for i, img in enumerate(images):
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        image_bytes = buffered.getvalue()
                        filename = f"full_page_{i+1}.png"
                        page_images.append((i, image_bytes, filename))
                    
                    if page_images:
                        image_inputs = [(b, fn) for _, b, fn in page_images]
                        results = proc_imgs_fn(image_inputs, config, errors_out=errors_out)
                        
                        # We need to inject the extracted hash into the markdown for the chunker to pick up
                        # We use an ephemeral HTML comment that the parser will strip out later
                        page_markers = []
                        for (_, _, _), img_tag in zip(page_images, results):
                            if img_tag:
                                # Extract the hash from the <img src="qmd-img://HASH" /> tag
                                match = re.search(r'qmd-img://([a-f0-9]{64})', img_tag)
                                if match:
                                    h = match.group(1)
                                    page_markers.append(f"\n\n<!-- qmd-page-img:{h} -->\n\n")
                                else:
                                    page_markers.append("")
                            else:
                                page_markers.append("")

                        # Split Docling markdown by its page breaks if they exist, or just append
                        # Docling often uses horizontal rules or specific headers, but it varies. 
                        # Since Docling's raw_md doesn't always have explicit page break markers, 
                        # injecting accurately per-page into the flat markdown is tough.
                        # For now, we prepend them or append them.
                        # Actually, a better approach is to append the page marker at the end of the markdown 
                        # for the whole doc if we can't reliably find page boundaries in Docling markdown.
                        # But wait, we CAN do this perfectly for pymupdf fallback!
                        if any(page_markers):
                            raw_md += "".join(page_markers)
                except Exception as e:
                    if verbose:
                        print(f"[Verbose PDF] Docling page embedding failed: {e}", flush=True)

        except Exception as e:
            if verbose:
                print(f"[Verbose PDF] Docling document assembly failed ({e}). Falling back to layout-aware extraction.", flush=True)
            raw_md = ""
            if errors_out is not None:
                errors_out.append({"error_type": "pdf_docling_failed", "message": f"{path.name}: {e}"})
                
        if raw_md.strip():
            doc.close()
            stem = re.sub(r'^[a-z0-9.]+[_-]', '', path.stem, flags=re.IGNORECASE)
            book_title = stem.replace("-", " ").replace("_", " ").title()
            return normalize_markdown_headings(raw_md, book_title=book_title)
        
        if verbose:
            print(f"[Verbose PDF] Docling returned empty. Falling back to pymupdf...", flush=True)

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
        
        # Strip pymupdf4llm picture text blocks rather than inserting OCR text or fake image links
        page_text = re.sub(r'<!--\s*Start of picture text\s*-->(.*?)<!--\s*End of picture text\s*-->\n*', '', page_text, flags=re.DOTALL | re.IGNORECASE)
        page_text = re.sub(r'<!--.*?-->\n*', '', page_text, flags=re.DOTALL)
        md_pages.append(page_text)

    img_enabled_fn = _get_converter_fn("_is_image_processing_enabled", _is_image_processing_enabled)
    proc_imgs_fn = _get_converter_fn("_process_images_concurrently", _process_images_concurrently)

    if config and getattr(config, "embed_pages", False):
        try:
            page_images = []
            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                filename = f"page_{i+1}_full.png"
                page_images.append((i, image_bytes, filename))
            
            if page_images:
                image_inputs = [(b, fn) for _, b, fn in page_images]
                results = proc_imgs_fn(image_inputs, config, errors_out=errors_out)
                for (page_idx, _, _), img_tag in zip(page_images, results):
                    if img_tag and page_idx < len(md_pages):
                        match = re.search(r'qmd-img://([a-f0-9]{64})', img_tag)
                        if match:
                            h = match.group(1)
                            md_pages[page_idx] = f"<!-- qmd-page-img:{h} -->\n\n" + md_pages[page_idx]
        except Exception as e:
            if verbose:
                print(f"[Verbose PDF] Page embedding failed: {e}", flush=True)

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