import pytest
from unittest.mock import MagicMock, patch
from qmd.main import main
import sys
from pathlib import Path

def test_arg_parsing_search(monkeypatch):
    """Test that search arguments are correctly routed."""
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "test", "query", "--limit", "5"])
    
    with patch("qmd.main.Store") as MockStore, \
         patch("qmd.main.load_config") as MockConfig:
        
        mock_store_inst = MockStore.return_value
        mock_store_inst.hybrid_search.return_value = []
        
        main()
        
        # Verify call includes defaults for verbose, rerank, reranker_only, collection, lex, title, and path
        mock_store_inst.hybrid_search.assert_called_once_with(
            "test query",
            limit=5,
            verbose=False,
            rerank=False,
            reranker_only=False,
            collection=None,
            lexical_query=None,
            title=None,
            path=None,
            fts_limit=None,
            vec_limit=None,
            rerank_candidates=None,
            exclude_seen_set=set()
        )

def test_git_pull_trigger(monkeypatch, tmp_path):
    """Test that --pull attempts a git command."""
    monkeypatch.setattr(sys, "argv", ["qmd", "update", "--pull"])
    
    fake_repo = tmp_path / "my_repo"
    fake_repo.mkdir()
    (fake_repo / ".git").mkdir()
    
    from qmd.config import Config, CollectionConfig
    mock_cfg = Config(collections={"test": CollectionConfig(path=str(fake_repo))})
    
    with patch("qmd.main.Store") as MockStore, \
         patch("qmd.main.load_config", return_value=mock_cfg), \
         patch("subprocess.run") as mock_run:
        
        MockStore.return_value.config = mock_cfg
        
        main()
        
        assert mock_run.called
        args, kwargs = mock_run.call_args
        assert args[0] == ["git", "pull"]
        assert kwargs["cwd"] == fake_repo

def test_parse_int_ranges():
    from qmd.utils import parse_int_ranges
    assert parse_int_ranges("234-237") == [234, 235, 236, 237]
    assert parse_int_ranges("23,24,29") == [23, 24, 29]
    assert parse_int_ranges("22,40, 25-27") == [22, 25, 26, 27, 40]
    assert parse_int_ranges(42) == [42]
    assert parse_int_ranges("5") == [5]
    assert parse_int_ranges("path/to/doc.md") is None
    assert parse_int_ranges("2020-01-01") is None

def test_chunk_arg_parsing_ranges(monkeypatch):
    """Test that chunk command parses rowid ranges and lists."""
    monkeypatch.setattr(sys, "argv", ["qmd", "chunk", "22,40,25-27", "-w", "1"])
    with patch("qmd.main.Store") as MockStore, \
         patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.get_chunk_by_id.return_value = [MagicMock(collection="", path="test.md", title="Test", headers="", seq_id=22, text="Sample")]
        main()
        mock_store.get_chunk_by_id.assert_called_once_with([22, 25, 26, 27, 40], window=1)

def test_chunk_arg_parsing_seq_ranges(monkeypatch):
    """Test that chunk command parses --seq ranges for document paths."""
    monkeypatch.setattr(sys, "argv", ["qmd", "chunk", "notes.md", "--seq", "1-3,5", "-c", "wiki"])
    with patch("qmd.main.Store") as MockStore, \
         patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.get_chunk_by_seq.return_value = [MagicMock(collection="wiki", path="notes.md", title="Notes", headers="", seq_id=1, text="Sample")]
        main()
        mock_store.get_chunk_by_seq.assert_called_once_with("wiki", "notes.md", seq_id=[1, 2, 3, 5], window=0)
