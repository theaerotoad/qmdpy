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

def test_xml_formatting_flat_and_doc(capsys):
    from qmd.formatting import format_results_xml, format_doc_results_xml, format_chunks_xml, format_outline_xml
    from qmd.store import Result

    r1 = Result(path="doc.md", title='Special "Doc" & Co', text="First snippet with **markdown**", score=0.95, source="hybrid", rank=1, collection="main", seq_id=10, headers="Intro > Overview")
    r2 = Result(path="doc.md", title='Special "Doc" & Co', text="Second snippet", score=0.80, source="hybrid", rank=2, collection="main", seq_id=15, headers="Details")

    # Flat XML
    format_results_xml([r1, r2], query="test query")
    out = capsys.readouterr().out
    assert '<search_results query="test query" total_matches="2">' in out
    assert 'rank="1"' in out
    assert 'prev_seq="9"' in out
    assert 'next_seq="11"' in out
    assert 'document="qmd://main/doc.md"' in out
    assert 'title="Special &quot;Doc&quot; &amp; Co"' in out
    assert 'First snippet with **markdown**' in out

    # Doc-grouped sequential XML with gaps (gap > 3 has no expand; gap <= 3 has expand)
    grouped = [{
        "title": 'Special "Doc" & Co',
        "collection": "main",
        "path": "doc.md",
        "score": 0.95,
        "chunks": [
            {"seq_id": 10, "score": 0.95, "rank": 1, "headers": "Intro", "text": "Chunk 10 text"},
            {"seq_id": 13, "score": 0.85, "rank": 2, "headers": "Middle", "text": "Chunk 13 text"},
            {"seq_id": 20, "score": 0.75, "rank": 3, "headers": "Details", "text": "Chunk 20 text"}
        ]
    }]
    format_doc_results_xml(grouped, query="test query")
    doc_out = capsys.readouterr().out
    # Gap of 10 chunks (> 3): no expand
    assert '<gap omitted_chunks="10" from_seq="0" to_seq="9" />' in doc_out
    # Gap of 2 chunks (<= 3): includes expand
    assert '<gap omitted_chunks="2" from_seq="11" to_seq="12" expand="qmd read \'main:doc.md:11-12\'" />' in doc_out
    # Gap of 6 chunks (> 3): no expand
    assert '<gap omitted_chunks="6" from_seq="14" to_seq="19" />' in doc_out
    assert '<chunk seq="10" rank="1" score="0.9500" chars="13" section="Intro"' in doc_out
    assert 'read="qmd read \'main:doc.md:10\'"' in doc_out
    assert 'outline="qmd outline \'main:doc.md\'"' in doc_out

    # Chunks XML
    format_chunks_xml([r1, r2])
    chunk_out = capsys.readouterr().out
    assert '<document uri="qmd://main/doc.md"' in chunk_out
    assert 'collection="main"' in chunk_out
    assert 'path="doc.md"' in chunk_out
    # Gap of 4 chunks (> 3): no expand attribute
    assert '<gap omitted_chunks="4" from_seq="11" to_seq="14" />' in chunk_out
    assert '<chunk seq="10" chars="31" section="Intro &gt; Overview">' in chunk_out
    assert 'read="qmd read \'main:doc.md:10\'' not in chunk_out

    # Outline XML
    outline = {
        "collection": "main",
        "path": "doc.md",
        "title": "Doc",
        "total_chunks": 20,
        "total_chars": 5000,
        "headings": [{"level": 1, "start_seq": 0, "end_seq": 9, "char_count": 2500, "text": "Intro & Setup"}]
    }
    format_outline_xml(outline)
    outline_out = capsys.readouterr().out
    assert '<outline uri="qmd://main/doc.md"' in outline_out
    assert 'collection="main"' in outline_out
    assert 'path="doc.md"' in outline_out
    assert 'title="Doc"' in outline_out
    assert '<heading level="1" start_seq="0" end_seq="9" char_count="2500" read="qmd read \'main:doc.md:0-9\'">Intro &amp; Setup</heading>' in outline_out

def test_cli_xml_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "helios", "--xml", "--flat"])
    with patch("qmd.main.Store") as MockStore, \
         patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = [
            MagicMock(collection="space", path="helios.md", title="Helios", score=0.9, rank=1, seq_id=147, headers="Mission", text="Helios 1 deep space mission")
        ]
        main()
        out = capsys.readouterr().out
        assert '<search_results query="helios"' in out
        assert '<result' in out
        assert 'Helios 1 deep space mission' in out

def test_cli_discover_command(monkeypatch, capsys):
    """Test qmd discover CLI command routing and output."""
    from qmd.store import Result
    monkeypatch.setattr(sys, "argv", ["qmd", "discover", "helios", "--limit", "5"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.discover.return_value = [
            Result(collection="space", path="helios.md", title="Helios", score=0.92, source="hybrid", rank=1, seq_id=10, headers="Overview", text="Helios solar probe mission overview", match_count=3)
        ]
        mock_store.last_exclusion_stats = {}
        main()
        out = capsys.readouterr().out
        assert "Discovered 1 document" in out
        assert "Helios" in out
        assert "3 matches in doc" in out
        assert "qmd://space/helios.md" in out

def test_discover_xml_and_json_formatting(capsys):
    """Test format_discover_xml and format_discover_json."""
    from qmd.formatting import format_discover_xml, format_discover_json
    from qmd.store import Result

    r = Result(collection="space", path="helios.md", title="Helios Mission", score=0.92, source="hybrid", rank=1, seq_id=10, headers="Overview", text="Helios solar probe mission", match_count=4)

    # XML format
    format_discover_xml([r], query="helios", session_id="sess_123")
    xml_out = capsys.readouterr().out
    assert '<discover_results query="helios" total_documents="1" session_id="sess_123"' in xml_out
    assert '<document' in xml_out
    assert 'uri="qmd://space/helios.md"' in xml_out
    assert 'title="Helios Mission"' in xml_out
    assert 'top_chunk_seq="10"' in xml_out
    assert 'match_count="4"' in xml_out
    assert 'search="qmd search &quot;helios&quot; -c &apos;space&apos; -p &apos;helios.md&apos;"' in xml_out

    # JSON format
    format_discover_json([r])
    json_out = capsys.readouterr().out
    assert '"match_count": 4' in json_out
    assert '"title": "Helios Mission"' in json_out

def test_helpall_flag(monkeypatch, capsys):
    """Test that --helpall outputs help for all subcommands and exits with 0."""
    monkeypatch.setattr(sys, "argv", ["qmd", "--helpall"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Query Multiple Documents" in out
    assert "discover" in out
    assert "search" in out
    assert "outline" in out
    assert "grep" in out
    assert "chunk" in out
    assert "update" in out
    assert "collection" in out
    assert "serve" in out
    assert "mcp" in out
    assert "=" * 80 in out

def test_helpall_subcommand_flag(monkeypatch, capsys):
    """Test that --helpall works when passed after a subcommand."""
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "--helpall"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Query Multiple Documents" in out
    assert "outline" in out
    assert "grep" in out

def test_format_help_all_direct():
    """Test format_help_all helper function directly."""
    from qmd.main import build_parser, format_help_all
    parser = build_parser()
    help_text = format_help_all(parser)
    assert "Query Multiple Documents" in help_text
    assert "usage: qmd discover" in help_text
    assert "usage: qmd search" in help_text
    assert "usage: qmd grep" in help_text
    assert "Target & Filters:" in help_text
    assert "Search Mode & Quality:" in help_text
    assert "Session & History:" in help_text

def test_search_presets_deep_and_broad(monkeypatch):
    """Test that --deep and --broad presets activate the expected flags."""
    # Test --deep (implies rerank and doc)
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "apollo", "--deep"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = []
        main()
        mock_store.hybrid_search.assert_called_once()
        assert mock_store.hybrid_search.call_args.kwargs["rerank"] is True

    # Test --broad (implies w2n)
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "apollo", "--broad"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.wide_to_narrow_search.return_value = []
        main()
        mock_store.wide_to_narrow_search.assert_called_once()

def test_session_smart_defaults(monkeypatch):
    """Test that --session implies exclude_seen unless --include-seen is provided."""
    # With --session: exclude_seen should default to True
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "query", "--session", "sess1"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"), \
         patch("qmd.main.get_seen_chunks_for_session", return_value={("c", "p", 1)}):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = []
        main()
        assert mock_store.hybrid_search.call_args.kwargs["exclude_seen_set"] == {("c", "p", 1)}

    # With --session and --include-seen: exclude_seen_set should be empty set
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "query", "--session", "sess1", "--include-seen"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"), \
         patch("qmd.main.get_seen_chunks_for_session", return_value={("c", "p", 1)}):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = []
        main()
        assert mock_store.hybrid_search.call_args.kwargs["exclude_seen_set"] == set()

def test_actionable_xml_attributes(capsys):
    """Test that Phase 5 actionable XML attributes are rendered."""
    from qmd.formatting import format_results_xml, format_doc_results_xml, format_outline_xml
    from qmd.store import Result

    r = Result(path="space/apollo.epub", title="Apollo", text="Mission details here", score=0.88, source="hybrid", rank=1, collection="Books", seq_id=70, headers="Budget > FY67")

    format_results_xml([r], query="apollo", session_id="test_sess", seen_chunks=5)
    out = capsys.readouterr().out
    assert 'session_id="test_sess"' in out
    assert 'seen_chunks="5"' in out
    assert 'next_query_hint="--session test_sess"' in out
    assert 'read="qmd read \'Books:space/apollo.epub:70\'"' in out
    assert 'outline="qmd outline \'Books:space/apollo.epub\'"' in out
    assert 'chars=' in out

    # Test doc results XML gap and chunk
    grouped = [{
        "title": "Apollo",
        "collection": "Books",
        "path": "space/apollo.epub",
        "score": 0.88,
        "chunks": [
            {"seq_id": 2, "score": 0.90, "rank": 1, "headers": "Intro", "text": "Launch"},
            {"seq_id": 4, "score": 0.88, "rank": 2, "headers": "Budget", "text": "Mission details"}
        ]
    }]
    format_doc_results_xml(grouped, query="apollo")
    doc_out = capsys.readouterr().out
    # Gap of 2 (0-1) is <= 3: has expand
    assert '<gap omitted_chunks="2" from_seq="0" to_seq="1" expand="qmd read \'Books:space/apollo.epub:0-1\'" />' in doc_out
    # Gap of 1 (seq 3) is <= 3: has expand
    assert '<gap omitted_chunks="1" from_seq="3" to_seq="3" expand="qmd read \'Books:space/apollo.epub:3\'" />' in doc_out
    assert 'read="qmd read \'Books:space/apollo.epub:4\'"' in doc_out
    assert 'outline="qmd outline \'Books:space/apollo.epub\'"' in doc_out

def test_search_truncation_safety_cap(monkeypatch, capsys):
    """Test that search results exceeding max_chunks are truncated and emit truncation XML."""
    from qmd.store import Result

    mock_results = [
        Result(collection="Books", path="apollo.epub", title="Apollo", text=f"Match {i}", score=1.0 - (i * 0.01), source="vec", rank=i+1, seq_id=i)
        for i in range(40)
    ]
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "apollo", "--limit", "40", "--max-chunks", "15", "--xml", "--flat"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = mock_results
        mock_store.last_exclusion_stats = {}
        main()
        out = capsys.readouterr().out
        assert 'total_matches="15"' in out
        assert '<truncation omitted_chunks="25" reason="max_chunks_per_response limit (15) reached" />' in out

def test_read_truncation_safety_cap(monkeypatch, capsys):
    """Test that reading a huge range of chunks truncates and emits a resumption hint."""
    from unittest.mock import MagicMock, patch
    from qmd.store import Result

    # Mock returning 50 chunks
    mock_results = [
        Result(collection="Books", path="apollo.epub", title="Apollo", text=f"Text {i}", score=1.0, source="vec", rank=i+1, seq_id=i)
        for i in range(50)
    ]
    monkeypatch.setattr(sys, "argv", ["qmd", "read", "Books:apollo.epub:0-49", "--max-chunks", "10", "--xml"])
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.get_chunk_by_seq.return_value = mock_results
        main()
        out = capsys.readouterr().out
        assert '<chunk seq="0"' in out
        assert '<chunk seq="9"' in out
        assert '<chunk seq="10"' not in out
        assert '<truncation omitted_chunks="40" reason="max_chunks_per_response limit (10) reached" resume="qmd read \'Books:apollo.epub:10-19\'"' in out

def test_env_var_qmd_config(monkeypatch, tmp_path):
    """Test that QMD_CONFIG sets the default configuration path when -C is not passed."""
    custom_cfg = tmp_path / "custom_config.yml"
    custom_cfg.write_text("db_path: /tmp/custom.db\n")
    
    monkeypatch.setenv("QMD_CONFIG", str(custom_cfg))
    monkeypatch.setattr(sys, "argv", ["qmd", "collections", "--plain"])
    
    with patch("qmd.main.Store") as MockStore:
        main()
        MockStore.assert_called_once()
        # Verify load_config picked up the custom path
        loaded_cfg = MockStore.call_args[0][0]
        assert loaded_cfg.config_path == str(custom_cfg.resolve())

def test_env_var_qmd_xml(monkeypatch, capsys):
    """Test that QMD_XML='1' defaults command outputs to XML without --xml flag."""
    monkeypatch.setenv("QMD_XML", "1")
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "apollo", "--flat"])
    
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = [
            MagicMock(collection="space", path="apollo.md", title="Apollo", score=0.95, rank=1, seq_id=1, headers="Intro", text="Apollo mission details")
        ]
        mock_store.last_exclusion_stats = {}
        main()
        out = capsys.readouterr().out
        assert '<search_results query="apollo"' in out
        assert '<result' in out
        assert 'Apollo mission details' in out

def test_env_var_qmd_deep(monkeypatch):
    """Test that QMD_DEEP='1' automatically triggers deep search (reranking + doc grouping)."""
    monkeypatch.setenv("QMD_DEEP", "1")
    monkeypatch.setattr(sys, "argv", ["qmd", "search", "apollo"])
    
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        mock_store.hybrid_search.return_value = []
        mock_store.last_exclusion_stats = {}
        main()
        mock_store.hybrid_search.assert_called_once()
        assert mock_store.hybrid_search.call_args.kwargs["rerank"] is True

def test_guide_command(monkeypatch, capsys):
    """Test that qmd guide outputs the agent workflow matrix and XML."""
    # Plain text guide
    monkeypatch.setattr(sys, "argv", ["qmd", "guide", "--plain"])
    with patch("qmd.main.Store"), patch("qmd.main.load_config"):
        main()
        out = capsys.readouterr().out
        assert "QMD LLM Agent Research & Inspection Guide" in out
        assert "qmd discover" in out
        assert "qmd search" in out
        assert "qmd read" in out
        assert "qmd outline" in out
        assert "qmd tree" in out

    # XML guide
    monkeypatch.setattr(sys, "argv", ["qmd", "guide", "--xml"])
    with patch("qmd.main.Store"), patch("qmd.main.load_config"):
        main()
        out = capsys.readouterr().out
        assert "<qmd_guide>" in out
        assert "<workflow>" in out
        assert "<shorthand_targets>" in out
        assert "</qmd_guide>" in out
