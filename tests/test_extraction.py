import json
import pytest
from unittest.mock import MagicMock, patch
import sys

from qmd.config import Config
from qmd.store import Store
from qmd.store.models import Result
from qmd.main import main
from qmd.formatting import format_extraction_cli, format_extraction_xml, set_plain_mode


def test_extract_document_fast_path(db_conn, temp_db_path):
    """Test that documents smaller than max_chunks bypass sampling and return entirely."""
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    with patch.object(store, 'get_document_outline') as mock_outline, \
         patch.object(store, 'get_chunks_by_seq_ids') as mock_get_chunks, \
         patch.object(store, 'get_analysis_report') as mock_analysis:
         
        mock_outline.return_value = {
            "total_chunks": 5,
            "collection": "docs",
            "path": "short.md",
            "headings": []
        }
        
        mock_get_chunks.return_value = [Result(path="short.md", title="", text=f"Chunk {i}", score=1.0, source="chunk", seq_id=i) for i in range(5)]
        mock_analysis.return_value = [{"analysis": {"title": "Short Doc"}}]

        # Call with max_chunks=10 (larger than total_chunks=5)
        data = store.extract_document("short.md", collection="docs", max_chunks=10)

        assert data is not None
        assert data["total_chunks"] == 5
        assert data["selected_chunks"] == 5
        assert len(data["chunks"]) == 5
        
        # It should have requested all chunks exactly 0 through 4
        mock_get_chunks.assert_called_once_with("docs", "short.md", [0, 1, 2, 3, 4])


def test_extract_document_budget_trimming(db_conn, temp_db_path):
    """Test greedy-grab sampling respects priority and strict budget constraints."""
    cfg = Config(db_path=str(temp_db_path))
    store = Store(cfg, connection=db_conn)

    with patch.object(store, 'get_document_outline') as mock_outline, \
         patch.object(store, 'get_chunks_by_seq_ids') as mock_get_chunks, \
         patch.object(store, 'hybrid_search') as mock_search, \
         patch.object(store, 'get_analysis_report') as mock_analysis:
         
        mock_outline.return_value = {
            "total_chunks": 50,
            "collection": "docs",
            "path": "long.md",
            "headings": [
                {"level": 1, "start_seq": 10, "text": "H1"},
                {"level": 2, "start_seq": 20, "text": "H2"}
            ]
        }
        
        mock_search.return_value = [
            Result(path="long.md", title="", text="Semantic", score=0.9, source="vec", seq_id=30)
        ]
        
        mock_get_chunks.side_effect = lambda coll, path, seqs: [
            Result(path=path, title="", text=f"C{s}", score=1.0, source="chunk", seq_id=s) for s in seqs
        ]

        # max_chunks=6.
        # P1: Head(0, 1) and Tail(48, 49) = 4 chunks
        # P2: Headers(9, 10) and (19, 20) = 4 chunks. But budget only has room for 2!
        # P3: Semantic(30) = 1 chunk. Should be dropped entirely.
        data = store.extract_document(
            path="long.md",
            collection="docs",
            queries=["test query"],
            max_chunks=6,
            head_chunks=2,
            tail_chunks=2
        )

        assert data is not None
        assert data["total_chunks"] == 50
        assert data["selected_chunks"] == 6
        
        # Verify the selected sequence IDs
        seqs = [c.seq_id for c in data["chunks"]]
        assert seqs == [0, 1, 9, 10, 48, 49]
        
        # Semantic search should still have been called because it happens before trimming
        mock_search.assert_called_once()


def test_extract_formatting_contiguous_blocks_cli(capsys):
    """Test that contiguous chunks are grouped together cleanly in CLI output."""
    set_plain_mode(True)
    
    chunks = [
        Result(path="doc.md", title="", text="Paragraph 1", score=1.0, source="chunk", seq_id=0, headers="Intro"),
        Result(path="doc.md", title="", text="Paragraph 2", score=1.0, source="chunk", seq_id=1, headers="Intro"),
        Result(path="doc.md", title="", text="Paragraph 3", score=1.0, source="chunk", seq_id=2, headers="Intro"),
        # Gap of 2 chunks (3, 4)
        Result(path="doc.md", title="", text="Paragraph 6", score=1.0, source="chunk", seq_id=5, headers="Details"),
    ]
    
    data = {
        "metadata": {"title": "Test Doc", "authors": ["Alice"]},
        "outline": {"collection": "docs", "path": "doc.md", "total_chunks": 10},
        "chunks": chunks,
        "total_chunks": 10
    }
    
    format_extraction_cli(data)
    out = capsys.readouterr().out
    
    # Check headers and chunk groupings
    assert "Extraction: Test Doc" in out
    assert "Chunks 0-2 [Intro]" in out
    assert "Paragraph 1\n\nParagraph 2\n\nParagraph 3" in out
    assert "[... 2 chunks omitted ...]" in out
    assert "Chunk 5 [Details]" in out
    assert "Paragraph 6" in out


def test_extract_formatting_contiguous_blocks_xml(capsys):
    """Test that contiguous chunks are grouped into single <chunk> tags in XML output."""
    chunks = [
        Result(path="doc.md", title="", text="Line 1", score=1.0, source="chunk", seq_id=0, headers="Part 1"),
        Result(path="doc.md", title="", text="Line 2", score=1.0, source="chunk", seq_id=1, headers="Part 1"),
        # Gap
        Result(path="doc.md", title="", text="Line 6", score=1.0, source="chunk", seq_id=5, headers="Part 2"),
    ]
    
    data = {
        "metadata": {"title": "XML Doc"},
        "outline": {"collection": "docs", "path": "doc.md", "total_chunks": 10},
        "chunks": chunks,
        "total_chunks": 10
    }
    
    format_extraction_xml(data)
    out = capsys.readouterr().out
    
    # Ensure chunk tags combine seq_ids properly
    assert '<chunk seq="0-1" chars="14" section="Part 1">' in out
    assert 'Line 1\n\nLine 2\n    </chunk>' in out
    
    # Gap output
    assert '<gap omitted_chunks="3" from_seq="2" to_seq="4" expand="qmd read \'docs:doc.md:2-4\'" />' in out
    
    # Single chunk
    assert '<chunk seq="5" chars="6" section="Part 2">' in out


def test_cli_extract_command(monkeypatch, capsys):
    """Test the CLI routing and JSON rendering for the extract command."""
    monkeypatch.setattr(sys, "argv", [
        "qmd", "extract", "doc.pdf", "-c", "papers", "-q", "innovations", "flaws", 
        "--max-chunks", "15", "--json"
    ])
    
    with patch("qmd.main.Store") as MockStore, patch("qmd.main.load_config"):
        mock_store = MockStore.return_value
        
        # Return a simple dict structure representing a valid extraction
        mock_store.extract_document.return_value = {
            "metadata": {"title": "Mock PDF"},
            "outline": {"total_chunks": 100},
            "chunks": [Result(path="doc.pdf", title="", text="test", score=1.0, source="chunk", seq_id=0)],
            "total_chunks": 100,
            "selected_chunks": 1
        }
        
        main()
        
        # Verify kwargs were parsed from CLI args properly
        mock_store.extract_document.assert_called_once_with(
            path="doc.pdf",
            collection="papers",
            queries=["innovations", "flaws"],
            max_chunks=15,
            head_chunks=3,
            tail_chunks=3,
            top_k_per_query=3,
            rerank=False
        )
        
        # Verify JSON output structure
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["metadata"]["title"] == "Mock PDF"
        assert len(parsed["chunks"]) == 1
        assert parsed["chunks"][0]["seq_id"] == 0