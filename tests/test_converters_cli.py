import sys
from unittest.mock import MagicMock

# Preemptively stub onnxruntime before pymupdf4llm import to prevent native C-extension segfault
for _mod in ("onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest

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

def test_converter_main_clean_output(tmp_path, capsys, monkeypatch):
    import sys
    from qmd.converters import main

    f = tmp_path / "2026-03-30_summary.md"
    f.write_text("# Project Summary\nAll done.", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["converters.py", str(f), "--clean"])
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "# Project Summary\nAll done."
    assert "Inferred Date:" not in captured.out
    assert "CONVERTED MARKDOWN" not in captured.out
    assert "Stats:" not in captured.out
    assert "=" * 10 not in captured.out

def test_converter_main_clean_output_with_file_output(tmp_path, capsys, monkeypatch):
    import sys
    from qmd.converters import main

    src = tmp_path / "input.md"
    src.write_text("# Direct Output\nJust the markdown.", encoding="utf-8")
    out_file = tmp_path / "output.md"

    monkeypatch.setattr(sys, "argv", ["converters.py", str(src), "--clean", "-o", str(out_file)])
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "# Direct Output\nJust the markdown."
    assert "Saved output to:" not in captured.out
    assert out_file.read_text(encoding="utf-8") == "# Direct Output\nJust the markdown."