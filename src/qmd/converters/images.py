"""
Vision and Multimodal LLM Image Processing for QMD.
Handles OCR, diagram transcription, and layout analysis for embedded document images.
"""
import os
import sys
from typing import List, Optional, Union


def _get_converter_fn(name: str, fallback):
    mod = sys.modules.get("qmd.converters")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


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
        if _is_verbose(config):
            debug_lines = [
                f"> **[DEBUG: Multimodal LLM for `{filename}`]**"
            ]
            if res:
                debug_lines.extend([
                    f"> **Model Output ({len(res)} chars):**",
                    res
                ])
            else:
                debug_lines.append("> *(Model returned empty response)*")
            return "\n\n" + "\n".join(debug_lines) + "\n\n"
        return res
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
            if parsed_result:
                debug_lines.extend([
                    f"> **Parsed Content ({len(md_lines)} item(s)):**",
                    parsed_result
                ])
            else:
                debug_lines.append("> *(No content extracted by built-in parser from above detections)*")

            return "\n\n" + "\n".join(debug_lines) + "\n\n"

        return parsed_result
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