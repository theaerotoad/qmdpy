import pytest
from unittest.mock import MagicMock
from qmd.config import Config, CollectionConfig
from qmd.store import Store
from qmd.main import handle_outline, handle_chunk

SAMPLE_MD = """# CLI Test Document

Section body text.

## Subsection

More text here.
"""


def test_cli_outline_handler(db_conn, temp_db_path, tmp_path, capsys):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "cli_test.md"
    doc_path.write_text(SAMPLE_MD, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("cli_coll", coll_cfg, force=True)

    # Test outline handler with JSON output
    args = MagicMock()
    args.path = "cli_test.md"
    args.collection = "cli_coll"
    args.json = True
    args.plain = True

    handle_outline(args, store)
    captured = capsys.readouterr()
    assert '"title": "Cli Test"' in captured.out
    assert '"CLI Test Document"' in captured.out


def test_cli_chunk_handler(db_conn, temp_db_path, tmp_path, capsys):
    config = Config()
    config.db_path = temp_db_path
    store = Store(config, connection=db_conn)

    doc_path = tmp_path / "cli_test.md"
    doc_path.write_text(SAMPLE_MD, encoding="utf-8")

    mock_llm = MagicMock()
    mock_llm.format_doc_for_embedding.side_effect = lambda t, c: c
    mock_llm.embed_batch.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    store.llm = mock_llm

    coll_cfg = CollectionConfig(path=str(tmp_path), glob="*.md")
    store.index_collection("cli_coll", coll_cfg, force=True)

    # Test chunk handler with path & seq_id
    args = MagicMock()
    args.target = "cli_test.md"
    args.seq = 0
    args.window = 1
    args.collection = "cli_coll"
    args.json = True
    args.plain = True

    handle_chunk(args, store)
    captured = capsys.readouterr()
    assert '"path": "cli_test.md"' in captured.out
    assert '"seq_id": 0' in captured.out

