import sys
import os
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from qmd.db import (
    get_connection, init_schema, register_functions,
    get_history_connection, check_db_compatibility
)
from qmd.config import Config
from qmd.llm import LLMClient
from qmd.converters import convert_to_markdown

try:
    import dplib
except ImportError:
    dplib = None

from .models import (
    Result,
    _format_size,
    extract_document_date,
    _build_collection_sql_filter,
    encode_vector,
    decode_vector,
    _results_to_json,
    _json_to_results,
)
from .vector_index import VectorIndexMixin
from .inspection import InspectionMixin
from .indexing import IndexingMixin
from .search import SearchMixin
from .retrieval import RetrievalMixin
from .analysis import AnalysisMixin
from .extraction import ExtractionMixin


class Store(VectorIndexMixin, InspectionMixin, IndexingMixin, RetrievalMixin, SearchMixin, AnalysisMixin, ExtractionMixin):
    def __init__(self, config: Config, connection: Optional[sqlite3.Connection] = None, read_only: bool = False):
        self.config = config
        self.read_only = read_only
        self.child_stores: List['Store'] = []
        self.collection_store_map: Dict[str, 'Store'] = {}

        if connection:
            self.conn = connection
            check_db_compatibility(self.conn)
        elif config.db_path:
            db_path = Path(config.db_path)
            if read_only:
                if not db_path.exists():
                    temp_conn = get_connection(db_path, read_only=False)
                    init_schema(temp_conn)
                    temp_conn.close()
                self.conn = get_connection(db_path, read_only=True)
                check_db_compatibility(self.conn)
            else:
                self.conn = get_connection(db_path, read_only=False)
                init_schema(self.conn)
        elif getattr(config, "is_federated", False) or getattr(config, "include", None):
            self.conn = None
        else:
            db_path = Path.home() / ".config" / "qmd" / "qmd.db"
            if read_only:
                if not db_path.exists():
                    temp_conn = get_connection(db_path, read_only=False)
                    init_schema(temp_conn)
                    temp_conn.close()
                self.conn = get_connection(db_path, read_only=True)
                check_db_compatibility(self.conn)
            else:
                self.conn = get_connection(db_path, read_only=False)
                init_schema(self.conn)

        history_db_path = Path(config.history_db_path) if config.history_db_path else Path.home() / ".config" / "qmd" / "qmd-history.db"
        self.history_conn = get_history_connection(history_db_path)
        self.last_exclusion_stats: Dict[str, int] = {"excluded_chunks": 0, "excluded_docs": 0}

        if self.conn is not None:
            register_functions(self.conn)
            self._init_usearch()
            self._ensure_query_indexes()

            if not self.read_only:
                try:
                    self.conn.execute("ALTER TABLE chunk_metadata ADD COLUMN headers TEXT")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass
        else:
            self.usearch_index = None
            self.usearch_path = None

        llm_cls = getattr(sys.modules.get("qmd.store"), "LLMClient", LLMClient)
        self.llm = llm_cls(
            base_url=config.llm_url,
            api_key=getattr(config, "api_key", None),
            embed_url=getattr(config, "embed_url", None),
            rerank_url=getattr(config, "rerank_url", None),
            embed_api_key=getattr(config, "embed_api_key", None),
            rerank_api_key=getattr(config, "rerank_api_key", None),
            embed_model=config.embed_model,
            rerank_model=config.rerank_model,
            generate_model=config.generate_model,
            timeout=getattr(config, "request_timeout", 120.0)
        )

        if getattr(config, "included_configs", None):
            child_read_only = False if (not self.read_only and getattr(config, "includes_update", False)) else True
            for child_cfg in config.included_configs:
                child_store = Store(child_cfg, read_only=child_read_only)
                child_store.llm = self.llm
                child_store.history_conn = self.history_conn
                self.child_stores.append(child_store)
                for coll_name in child_cfg.collections.keys():
                    self.collection_store_map[coll_name] = child_store

        for coll_name in config.collections.keys():
            if coll_name not in self.collection_store_map:
                self.collection_store_map[coll_name] = self

    def _get_target_stores_for_collection(self, collection: Optional[Union[str, List[str]]] = None) -> List['Store']:
        target_stores = super()._get_target_stores_for_collection(collection)
        if self.conn is None:
            return [s for s in target_stores if s is not self]
        return target_stores

    def build_usearch_index(self):
        if getattr(self.config, "is_federated", False):
            for child in self.child_stores:
                child.build_usearch_index()
            if self.conn is not None:
                super().build_usearch_index()
        else:
            if self.conn is not None:
                super().build_usearch_index()


__all__ = [
    "Store",
    "Result",
    "encode_vector",
    "decode_vector",
    "extract_document_date",
    "_results_to_json",
    "_json_to_results",
    "_format_size",
    "_build_collection_sql_filter",
    "LLMClient",
    "convert_to_markdown",
    "dplib",
]