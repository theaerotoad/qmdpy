"""
CLI entry point for running converters directly via:
    python src/qmd/converters <file_path>
or:
    python -m qmd.converters <file_path>
"""
import sys
from pathlib import Path

# Ensure root / src is in sys.path when invoked directly as a directory or script
cur = Path(__file__).resolve().parent
while cur.name != "src" and cur.parent != cur:
    cur = cur.parent
src_dir = str(cur) if cur.name == "src" else str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from qmd.converters import main

if __name__ == "__main__":
    main()