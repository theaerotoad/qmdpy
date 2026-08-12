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
            path=None
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
