import sys
from pathlib import Path
import pytest

import qmd.store
from qmd.store import (
    Store,
    Result,
    encode_vector,
    decode_vector,
    extract_document_date,
    _results_to_json,
    _json_to_results,
    _format_size,
    _build_collection_sql_filter,
)
from qmd.store.models import Result as ModelResult
from qmd.store.vector_index import VectorIndexMixin
from qmd.store.inspection import InspectionMixin
from qmd.store.indexing import IndexingMixin
from qmd.store.search import SearchMixin
from qmd.store.retrieval import RetrievalMixin
from qmd.config import Config


def test_store_is_package():
    """Verify qmd.store is loaded as a package directory with __path__."""
    assert hasattr(qmd.store, "__path__"), "qmd.store should be a package directory, not a single module file."
    package_dir = Path(qmd.store.__file__).parent
    assert package_dir.is_dir()
    assert (package_dir / "__init__.py").exists()


def test_submodule_line_count_ceiling():
    """Enforce architectural constraint: no submodule in qmd.store may exceed 800 lines."""
    package_dir = Path(qmd.store.__file__).parent
    py_files = list(package_dir.glob("*.py"))
    assert len(py_files) >= 5, f"Expected modular submodules in {package_dir}, found {len(py_files)}"

    for py_file in py_files:
        lines = py_file.read_text(encoding="utf-8").splitlines()
        line_count = len(lines)
        assert line_count <= 800, (
            f"Module '{py_file.name}' exceeds the 800-line architectural ceiling: {line_count} lines"
        )


def test_store_inherits_all_mixins():
    """Ensure Store class properly composes all domain mixins."""
    assert issubclass(Store, VectorIndexMixin)
    assert issubclass(Store, InspectionMixin)
    assert issubclass(Store, IndexingMixin)
    assert issubclass(Store, RetrievalMixin)
    assert issubclass(Store, SearchMixin)


def test_store_public_methods_present():
    """Verify all critical public API methods remain available on Store instances."""
    expected_methods = [
        # Search
        "search_fts",
        "search_vec",
        "hybrid_search",
        "wide_to_narrow_search",
        "discover",
        "get_query_embedding",
        # Indexing & Pruning
        "index_collection",
        "prune_orphaned_collections",
        "get_indexing_errors",
        # Inspection & Outline
        "get_stats",
        "get_document_outline",
        "get_chunk_by_id",
        "get_chunk_by_seq",
        "get_collection_tree",
        "grep_search",
        # Vector Index
        "build_usearch_index",
    ]

    for method_name in expected_methods:
        assert hasattr(Store, method_name), f"Store missing required method '{method_name}'"
        assert callable(getattr(Store, method_name)), f"Store.{method_name} must be callable"


def test_module_reexports_and_submodule_parity():
    """Verify top-level package exports match submodule definitions."""
    assert Result is ModelResult
    assert callable(encode_vector)
    assert callable(decode_vector)
    assert callable(extract_document_date)
    assert callable(_results_to_json)
    assert callable(_json_to_results)
    assert callable(_format_size)
    assert callable(_build_collection_sql_filter)


def test_dynamic_patching_compatibility(monkeypatch):
    """
    Ensure test patches against qmd.store.<name> dynamically resolve
    correctly inside submodules without breaking mixin behaviors.
    """
    custom_date = "2026-09-02"
    monkeypatch.setattr(qmd.store, "extract_document_date", lambda path, body="": custom_date)

    from qmd.store.models import extract_document_date as models_extract
    # Dynamic sys.modules lookup ensures compatibility even if called via submodule
    result = models_extract("dummy.md", "some body")
    assert result == custom_date