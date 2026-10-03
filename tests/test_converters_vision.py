import sys
from unittest.mock import MagicMock

# Preemptively stub onnxruntime before pymupdf4llm import to prevent native C-extension segfault
for _mod in ("onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest

def test_process_image_routing_and_concurrency(monkeypatch):
    from qmd.converters.images import _process_image, _process_images_concurrently
    from qmd.config import Config

    calls = []

    def mock_multimodal(image_bytes, filename, config, errors_out=None):
        calls.append(("multimodal", filename))
        return f"# MD for {filename}"

    def mock_vision(image_bytes, filename, config, errors_out=None):
        calls.append(("vision", filename))
        return f"# Vision for {filename}"

    monkeypatch.setattr("qmd.converters.images._process_image_multimodal_llm", mock_multimodal)
    monkeypatch.setattr("qmd.converters.images._process_image_vision_api", mock_vision)

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

def test_vision_api_verbose_output(monkeypatch):
    import httpx
    from qmd.converters.images import _process_image_vision_api
    from qmd.config import Config

    class MockResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "detections": [
                    {"label": "unknown_layout_box", "confidence": 0.95, "content": "mystery layout text"}
                ]
            }

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: MockResponse())

    # Quiet mode: empty because label is unrecognized
    cfg_quiet = Config.from_dict({"vision_url": "http://127.0.0.1:9000"})
    res_quiet = _process_image_vision_api(b"fakebytes", "chart.png", cfg_quiet)
    assert res_quiet == ""

    # Verbose mode: renders inline debug block with full raw JSON
    cfg_verbose = Config.from_dict({"vision_url": "http://127.0.0.1:9000"})
    cfg_verbose.verbose = True
    res_verbose = _process_image_vision_api(b"fakebytes", "chart.png", cfg_verbose)
    assert "[DEBUG: Vision API / Image Layout for `chart.png`]" in res_verbose
    assert "unknown_layout_box" in res_verbose
    assert "mystery layout text" in res_verbose
    assert "No content extracted by built-in parser" in res_verbose

def test_vision_api_verbose_error(monkeypatch):
    import httpx
    from qmd.converters.images import _process_image_vision_api
    from qmd.config import Config

    def mock_post_fail(*args, **kwargs):
        raise httpx.ConnectError("Connection refused by layout service")

    monkeypatch.setattr(httpx, "post", mock_post_fail)

    cfg = Config.from_dict({"vision_url": "http://127.0.0.1:9000"})
    cfg.verbose = True
    errors = []
    res = _process_image_vision_api(b"fakebytes", "chart.png", cfg, errors_out=errors)
    assert "[DEBUG: Vision API Error for `chart.png`" in res
    assert "Connection refused" in res
    assert len(errors) == 1

def test_clean_vision_markdown_filters_junk():
    from qmd.converters.images import _clean_vision_markdown

    raw_junk = (
        "Valid header line\n"
        "~_!@#$%>?^\n"
        ".....::::::\n"
        "(((((((((((((((((((((\n"
        "None\n"
        "None\n"
        "None\n"
        "None\n"
        "Valid middle line\n"
        "bcdfghjklmnpqrstvwxyz\n"
        "- ????????\n"
        "Another valid line"
    )

    cleaned, omitted = _clean_vision_markdown(raw_junk)

    assert "Valid header line" in cleaned
    assert "Valid middle line" in cleaned
    assert "Another valid line" in cleaned
    assert "~_!@#$%>?^" not in cleaned
    assert ".....::::::" not in cleaned
    assert "(((((((((((((((((((((" not in cleaned
    assert "bcdfghjklmnpqrstvwxyz" not in cleaned
    assert "- ????" not in cleaned

    # Repetition loop: at most 2 "None" kept, subsequent omitted
    assert cleaned.count("None") == 2
    assert len(omitted) >= 5

def test_clean_vision_markdown_preserves_valid_markdown():
    from qmd.converters.images import _clean_vision_markdown

    valid_md = (
        "# Title\n\n"
        "Introduction .................... 5\n\n"
        "```python\n"
        "# Comment inside code: %%%%%\n"
        "x = [1, 2, 3]\n"
        "```\n\n"
        "| Name | Score |\n"
        "| --- | --- |\n"
        "| Alice | 100% |\n\n"
        "$$\\sum_{i=1}^n x_i$$\n\n"
        "![Alt text](image.png)\n"
        "> Quote block with punctuation: 'Hello!'"
    )

    cleaned, omitted = _clean_vision_markdown(valid_md)

    assert "# Title" in cleaned
    # Dot leader in TOC line normalized to ellipses without dropping line
    assert "Introduction ... 5" in cleaned
    assert "# Comment inside code: %%%%%" in cleaned
    assert "| Alice | 100% |" in cleaned
    assert "$$\\sum_{i=1}^n x_i$$" in cleaned
    assert "![Alt text](image.png)" in cleaned
    assert "> Quote block with punctuation: 'Hello!'" in cleaned
    assert len(omitted) == 0

def test_wrap_vision_xml_behavior():
    from qmd.converters.images import _wrap_vision_xml

    # 1. Plain image embed should NOT be wrapped
    img_only = "![A photograph of a server rack](rack.png)"
    assert _wrap_vision_xml(img_only) == img_only

    # Multiple image embeds should NOT be wrapped
    multi_img = "![Image 1](a.png)\n![Image 2](b.png)"
    assert _wrap_vision_xml(multi_img) == multi_img

    # 2. Text, tables, or diagrams SHOULD be wrapped in <vision>
    text_content = "Architecture diagram shows 3 backend microservices."
    wrapped_text = _wrap_vision_xml(text_content)
    assert wrapped_text == f"<vision>\n{text_content}\n</vision>"

    table_content = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    wrapped_table = _wrap_vision_xml(table_content)
    assert wrapped_table == f"<vision>\n{table_content}\n</vision>"

    # 3. Empty input stays empty
    assert _wrap_vision_xml("") == ""
    assert _wrap_vision_xml("   ") == ""

    # 4. Do not double-wrap
    already_wrapped = "<vision>\nSome text\n</vision>"
    assert _wrap_vision_xml(already_wrapped) == already_wrapped

def test_multimodal_verbose_output_with_junk_filter(monkeypatch):
    from qmd.converters.images import _process_image_multimodal_llm
    from qmd.config import Config

    class MockLLMWithJunk:
        def __init__(self, *args, **kwargs):
            pass
        def process_image(self, image_bytes, filename):
            return "Valid diagram label\n~~~~~!!!!!!@#$%\nNone\nNone\nNone\nNone"

    monkeypatch.setattr("qmd.llm.LLMClient", MockLLMWithJunk)

    cfg = Config.from_dict({
        "multimodal_url": "http://127.0.0.1:8888",
        "multimodal_model": "gpt-4o"
    })
    cfg.verbose = True
    res = _process_image_multimodal_llm(b"fakebytes", "noisy_diagram.png", cfg)

    assert "[DEBUG: Multimodal LLM for `noisy_diagram.png`]" in res
    assert "[Junk Filter: Omitted" in res
    assert "Valid diagram label" in res

def test_multimodal_verbose_output(monkeypatch):
    from qmd.converters.images import _process_image_multimodal_llm
    from qmd.config import Config

    class MockLLMClient:
        def __init__(self, *args, **kwargs):
            pass
        def process_image(self, image_bytes, filename):
            return "Detailed image description from multimodal model"

    monkeypatch.setattr("qmd.llm.LLMClient", MockLLMClient)

    cfg = Config.from_dict({
        "multimodal_url": "http://127.0.0.1:8888",
        "multimodal_model": "gpt-4o"
    })
    cfg.verbose = True
    res = _process_image_multimodal_llm(b"fakebytes", "diagram.png", cfg)
    assert "[DEBUG: Multimodal LLM for `diagram.png`]" in res
    assert "Detailed image description from multimodal model" in res