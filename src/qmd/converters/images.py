"""
Vision and Multimodal LLM Image Processing for QMD.
Handles OCR, diagram transcription, and layout analysis for embedded document images.
"""
import os
import re
import sys
from typing import List, Optional, Union


def _get_converter_fn(name: str, fallback):
    mod = sys.modules.get("qmd.converters")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


def _clean_vision_markdown(text: str) -> tuple[str, list[str]]:
    """
    Cleans vision and multimodal OCR output by removing spurious junk lines,
    runaway character loops, symbol soup, and repetitive hallucination loops.
    
    Returns:
        (cleaned_text, omitted_lines)
    """
    if not text:
        return "", []

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    omitted_lines: list[str] = []
    
    in_code_block = False
    last_norm_line: Optional[str] = None
    consecutive_repeats = 0

    for line in lines:
        stripped = line.strip()
        
        # 1. Blank lines: preserve structure, reset repeat tracking
        if not stripped:
            cleaned_lines.append("")
            last_norm_line = None
            consecutive_repeats = 0
            continue

        # 2. Fenced code blocks: preserve verbatim
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            last_norm_line = None
            consecutive_repeats = 0
            continue

        if in_code_block:
            cleaned_lines.append(line)
            continue

        # 3. Markdown tables: preserve valid rows & separators
        if stripped.startswith("|") and stripped.endswith("|"):
            # Check for non-empty table row (not just pipe spam like |||||||)
            pipe_content = stripped.replace("|", "").strip()
            if pipe_content:
                cleaned_lines.append(line)
                last_norm_line = stripped.lower()
                continue
            else:
                omitted_lines.append(line)
                continue

        # 4. Display or inline math
        if stripped.startswith("$$") or (stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2):
            cleaned_lines.append(line)
            last_norm_line = None
            consecutive_repeats = 0
            continue

        # 5. Standalone Markdown images or links
        if re.match(r'^[ \t]*!\[.*?\]\(.*?\)[ \t]*$', line) or re.match(r'^[ \t]*\[.*?\]\(.*?\)[ \t]*$', line):
            cleaned_lines.append(line)
            last_norm_line = None
            consecutive_repeats = 0
            continue

        # 6. Legitimate thematic breaks / horizontal rules (---, ***, ___)
        if re.match(r'^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$', line):
            cleaned_lines.append(line)
            last_norm_line = None
            consecutive_repeats = 0
            continue

        # Extract content after common markdown prefixes (headings, blockquotes, bullets)
        content = re.sub(r'^(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+\.\s+)', '', stripped).strip()

        # If line had a prefix but empty content, omit
        if not content:
            omitted_lines.append(line)
            continue

        # Normalize dot leaders (e.g. "Chapter 1 .......... 15" -> "Chapter 1 ... 15")
        normalized_content = re.sub(r'\s*\.{4,}\s*', ' ... ', content)

        # Quality Check A: Zero alphanumeric characters (pure symbol soup)
        if not re.search(r'[a-zA-Z0-9]', normalized_content):
            omitted_lines.append(line)
            continue

        # Quality Check B: Corrupt Unicode replacement characters or non-printable ASCII
        if normalized_content.count('\ufffd') >= 2 or (normalized_content.count('\ufffd') / len(normalized_content)) > 0.1:
            omitted_lines.append(line)
            continue
        if any(ord(c) < 32 and c not in ('\t', '\n') for c in normalized_content):
            omitted_lines.append(line)
            continue

        # Quality Check C: Extreme symbol/punctuation density (> 75% symbols for 8+ char lines)
        non_ws = [c for c in normalized_content if not c.isspace()]
        if len(non_ws) >= 8:
            alnum_count = sum(1 for c in non_ws if c.isalnum())
            if (alnum_count / len(non_ws)) < 0.25:
                omitted_lines.append(line)
                continue

        # Quality Check D: Runaway character repetition loop (6+ identical consecutive characters)
        if re.search(r'([^\s])\1{5,}', normalized_content):
            omitted_lines.append(line)
            continue

        # Quality Check E: Non-word vowel-less or vowel-depleted consonant gibberish tokens (12+ chars)
        tokens = [re.sub(r'^\W+|\W+$', '', tok) for tok in normalized_content.split()]
        if any(len(tok) >= 12 and tok.isalpha() and (not re.search(r'[aeiouAEIOU]', tok) or len(re.findall(r'[aeiouyAEIOUY]', tok)) / len(tok) < 0.1) for tok in tokens):
            omitted_lines.append(line)
            continue

        # Quality Check F: Repetitive line hallucination loop (drop 3rd and subsequent identical lines)
        norm_line = stripped.lower()
        if norm_line == last_norm_line:
            consecutive_repeats += 1
            if consecutive_repeats >= 2:
                omitted_lines.append(line)
                continue
        else:
            last_norm_line = norm_line
            consecutive_repeats = 0

        # Line passed all checks: use dot-leader cleaned line if modified
        if normalized_content != content:
            prefix_match = re.match(r'^(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+\.\s+)', stripped)
            prefix = prefix_match.group(0) if prefix_match else ""
            cleaned_lines.append(f"{prefix}{normalized_content}")
        else:
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result, omitted_lines


def clean_vision_text(text: str) -> str:
    """Convenience helper returning cleaned vision text without metadata."""
    cleaned, _ = _clean_vision_markdown(text)
    return cleaned


def _is_image_processing_enabled(config) -> bool:
    if not config:
        return False
    return bool(
        getattr(config, "vision_url", None)
        or getattr(config, "multimodal_url", None)
        or getattr(config, "multimodal_model", None)
    )


def _is_verbose(config=None) -> bool:
    if os.environ.get("QMD_VERBOSE") == "1":
        return True
    if config is not None:
        return bool(getattr(config, "verbose", False) or getattr(config, "verbose_images", False))
    return False


def _process_image_multimodal_llm(image_bytes: bytes, filename: str, config, errors_out: Optional[List[dict]] = None) -> str:
    if not config or not image_bytes:
        return ""
    try:
        from qmd.llm import LLMClient
        client = LLMClient(
            base_url=getattr(config, "llm_url", None),
            api_key=getattr(config, "api_key", None),
            multimodal_url=getattr(config, "multimodal_url", None),
            multimodal_api_key=getattr(config, "multimodal_api_key", None),
            multimodal_model=getattr(config, "multimodal_model", None),
            multimodal_prompt=getattr(config, "multimodal_prompt", None),
            timeout=getattr(config, "request_timeout", 120.0),
        )
        res = client.process_image(image_bytes, filename=filename)
        cleaned_res, omitted_lines = _clean_vision_markdown(res)

        if _is_verbose(config):
            debug_lines = [
                f"> **[DEBUG: Multimodal LLM for `{filename}`]**"
            ]
            if res:
                output_preview = cleaned_res if cleaned_res else "*(All output filtered as spurious junk)*"
                debug_lines.extend([
                    f"> **Model Output ({len(res)} chars):**",
                    output_preview
                ])
                if omitted_lines:
                    debug_lines.append(f"> **[Junk Filter: Omitted {len(omitted_lines)} spurious line(s)]**")
                    for o in omitted_lines[:10]:
                        debug_lines.append(f">   - `{o}`")
                    if len(omitted_lines) > 10:
                        debug_lines.append(f">   - *... and {len(omitted_lines) - 10} more line(s)*")
            else:
                debug_lines.append("> *(Model returned empty response)*")
            return "\n\n" + "\n".join(debug_lines) + "\n\n"
        return cleaned_res
    except Exception as e:
        print(f"Warning: Multimodal LLM error for {filename}: {e}")
        if errors_out is not None:
            errors_out.append({"error_type": "multimodal_image_error", "message": f"{filename}: {e}"})
        if _is_verbose(config):
            return f"\n\n> **[DEBUG: Multimodal LLM Error for `{filename}`: {e}]**\n\n"
        return ""


def _process_image(image_bytes: bytes, filename: str, config, errors_out: Optional[List[dict]] = None) -> str:
    if not config or not image_bytes:
        return ""
    if getattr(config, "multimodal_url", None) or getattr(config, "multimodal_model", None):
        multi_fn = _get_converter_fn("_process_image_multimodal_llm", _process_image_multimodal_llm)
        try:
            return multi_fn(image_bytes, filename, config, errors_out=errors_out)
        except TypeError:
            return multi_fn(image_bytes, filename, config)
    elif getattr(config, "vision_url", None):
        vision_fn = _get_converter_fn("_process_image_vision_api", _process_image_vision_api)
        try:
            return vision_fn(image_bytes, filename, config, errors_out=errors_out)
        except TypeError:
            return vision_fn(image_bytes, filename, config)
    return ""


def _process_images_concurrently(
    items: List[tuple],
    config,
    max_workers: Union[int, None] = None,
    errors_out: Optional[List[dict]] = None
) -> List[str]:
    if not items:
        return []
    if max_workers is None:
        max_workers = (
            getattr(config, "max_image_concurrency", None)
            or getattr(config, "max_simultaneous_images", None)
            or 4
        )
    max_workers = max(1, int(max_workers))

    proc_image_fn = _get_converter_fn("_process_image", _process_image)

    if len(items) == 1 or max_workers == 1:
        return [proc_image_fn(b, fn, config, errors_out=errors_out) for b, fn in items]

    import concurrent.futures
    workers = min(len(items), max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(proc_image_fn, b, fn, config, errors_out) for b, fn in items]
        results = []
        for (b, fn), f in zip(items, futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Warning: Concurrent image processing error: {e}")
                if errors_out is not None:
                    errors_out.append({"error_type": "image_processing_error", "message": str(e)})
                if _is_verbose(config):
                    results.append(f"\n\n> **[DEBUG: Image Processing Worker Error for `{fn}`: {e}]**\n\n")
                else:
                    results.append("")
        return results


def _process_image_vision_api(image_bytes: bytes, filename: str, config, errors_out: Optional[List[dict]] = None) -> str:
    if not config or not getattr(config, "vision_url", None):
        return ""
    
    try:
        import httpx
        import base64
    except ImportError:
        return ""
    
    vision_url = config.vision_url
    headers = {}
    if getattr(config, "vision_api_key", None):
        headers["Authorization"] = f"Bearer {config.vision_api_key}"
        
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/jpeg"
    if filename.lower().endswith(".png"): mime_type = "image/png"
    
    payload = {
        "image": f"data:{mime_type};base64,{b64_image}",
        "extract_tables": True,
        "extract_text": True
    }
    
    try:
        resp = httpx.post(vision_url, json=payload, headers=headers, timeout=getattr(config, "request_timeout", 120.0))
        resp.raise_for_status()
        data = resp.json()
        
        detections = data.get("detections", [])
        if not detections and "results" in data and len(data["results"]) > 0:
            detections = data["results"][0].get("detections", [])
            
        md_lines = []
        text_labels = {
            "caption", "footnote", "formula", "list-item", 
            "page-footer", "page-header", "section-header", 
            "text", "title"
        }
        
        for d in detections:
            lbl = d.get("label", "").lower()
            if lbl == "picture":
                text = d.get("text", "").strip()
                alt = f"Image with text: {text}" if text else "Image"
                md_lines.append(f"![{alt}]({filename})")
            elif lbl == "table":
                if d.get("markdown"):
                    md_lines.append(d["markdown"])
                elif d.get("html"):
                    md_lines.append(d["html"])
            elif lbl in text_labels:
                if d.get("text"):
                    md_lines.append(d["text"].strip())
                    
        parsed_result = "\n\n".join(md_lines)
        cleaned_result, omitted_lines = _clean_vision_markdown(parsed_result)

        if _is_verbose(config):
            import json
            try:
                raw_json = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                raw_json = str(data)

            debug_lines = [
                f"> **[DEBUG: Vision API / Image Layout for `{filename}`]**",
                "> **Raw Endpoint Response:**",
                "```json",
                raw_json,
                "```"
            ]
            if cleaned_result:
                debug_lines.extend([
                    f"> **Parsed Content ({len(md_lines)} item(s)):**",
                    cleaned_result
                ])
                if omitted_lines:
                    debug_lines.append(f"> **[Junk Filter: Omitted {len(omitted_lines)} spurious line(s)]**")
                    for o in omitted_lines[:10]:
                        debug_lines.append(f">   - `{o}`")
                    if len(omitted_lines) > 10:
                        debug_lines.append(f">   - *... and {len(omitted_lines) - 10} more line(s)*")
            else:
                debug_lines.append("> *(No content extracted by built-in parser from above detections)*")

            return "\n\n" + "\n".join(debug_lines) + "\n\n"

        return cleaned_result
    except Exception as e:
        resp_details = ""
        if 'resp' in locals() and hasattr(resp, 'text') and resp.text:
            resp_details = f"\n> **Response Body:**\n```\n{resp.text[:2000]}\n```"
        print(f"Warning: Vision API error for {filename}: {e}")
        if errors_out is not None:
            errors_out.append({"error_type": "vision_api_error", "message": f"{filename}: {e}"})
        if _is_verbose(config):
            return f"\n\n> **[DEBUG: Vision API Error for `{filename}`: {e}]**{resp_details}\n\n"
        return ""