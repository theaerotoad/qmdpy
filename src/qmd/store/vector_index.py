import os
from pathlib import Path

from qmd.formatting import YELLOW, GREEN, RED, RESET
from qmd.db import get_db_meta, ensure_vector_table, get_connection
from .models import decode_vector


class VectorIndexMixin:
    """Handles vector ANN indexing using usearch and database index management."""

    def _init_usearch(self):
        self.usearch_index = None
        if getattr(self.config, "db_path", None):
            self.usearch_path = f"{self.config.db_path}.usearch"
        else:
            self.usearch_path = str(Path.home() / ".config" / "qmd" / "qmd.db.usearch")

        if os.path.exists(self.usearch_path):
            try:
                import usearch.index
                dim = get_db_meta(self.conn, "vector_dim")
                quant_type = get_db_meta(self.conn, "vector_quantization") or getattr(self.config, "vector_quantization", "none") or "none"

                if not dim:
                    cursor = self.conn.cursor()
                    try:
                        cursor.execute("SELECT embedding FROM vectors LIMIT 1")
                        row = cursor.fetchone()
                        if row and row[0]:
                            blob_len = len(row[0])
                            if quant_type == "int8":
                                dim = blob_len
                            elif quant_type in ("bit", "binary"):
                                dim = blob_len * 8
                            else:
                                dim = blob_len // 4
                    except Exception:
                        pass

                if dim:
                    dtype = "f32"
                    if quant_type == "int8":
                        dtype = "i8"
                    elif quant_type in ("bit", "binary"):
                        dtype = "b1"

                    self.usearch_index = usearch.index.Index(ndim=int(dim), metric="cos", dtype=dtype)
                    if self.read_only:
                        self.usearch_index.view(self.usearch_path)
                    else:
                        self.usearch_index.load(self.usearch_path)
            except ImportError:
                print(f"{YELLOW}Warning: .usearch index found but 'usearch' package is not installed. Ignoring.{RESET}")
            except Exception as e:
                print(f"{YELLOW}Warning: Failed to load .usearch index: {e}{RESET}")

    def build_usearch_index(self):
        """Builds a HNSW ANN index using usearch from the existing vectors table."""
        try:
            import usearch.index
            import numpy as np
        except ImportError:
            print(f"{RED}Error: 'usearch' and 'numpy' packages are required to build the ANN index.{RESET}")
            return False

        cursor = self.conn.cursor()
        dim = get_db_meta(self.conn, "vector_dim")
        quant_type = get_db_meta(self.conn, "vector_quantization") or getattr(self.config, "vector_quantization", "none") or "none"

        if not dim:
            cursor.execute("SELECT embedding FROM vectors LIMIT 1")
            row = cursor.fetchone()
            if not row or not row[0]:
                print(f"{YELLOW}No vectors found in database. Skipping usearch index build.{RESET}")
                return True

            blob_len = len(row[0])
            if quant_type == "int8":
                dim = blob_len
            elif quant_type in ("bit", "binary"):
                dim = blob_len * 8
            else:
                dim = blob_len // 4

            ensure_vector_table(self.conn, dim=dim, quant_type=quant_type)
            print(f"{YELLOW}Inferred vector dimension {dim} from existing database payload.{RESET}")

        dim = int(dim)
        dtype = "f32"
        if quant_type == "int8":
            dtype = "i8"
        elif quant_type in ("bit", "binary"):
            dtype = "b1"

        print(f"Building usearch ANN index (dim={dim}, dtype={dtype})...")

        index = usearch.index.Index(ndim=dim, metric="cos", dtype=dtype)

        cursor = self.conn.cursor()
        cursor.execute("SELECT rowid, embedding FROM vectors")

        rowids = []
        vectors = []
        batch_size = 10000
        count = 0

        for row in cursor.fetchall():
            r_id, emb_blob = row
            vec = decode_vector(emb_blob, dim, quant_type)
            rowids.append(r_id)
            vectors.append(vec)

            if len(rowids) >= batch_size:
                index.add(np.array(rowids, dtype=np.uint64), np.array(vectors, dtype=np.float32))
                count += len(rowids)
                rowids = []
                vectors = []

        if rowids:
            index.add(np.array(rowids, dtype=np.uint64), np.array(vectors, dtype=np.float32))
            count += len(rowids)

        parent_dir = Path(self.usearch_path).parent
        if parent_dir and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        index.save(self.usearch_path)
        print(f"{GREEN}✓ Successfully built usearch index with {count} vectors at {self.usearch_path}{RESET}")
        self.usearch_index = index
        return True

    def _ensure_query_indexes(self):
        """Ensures critical performance indexes exist on documents and chunk_metadata even for legacy databases."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_documents_hash'")
            if cursor.fetchone():
                return

            if not self.read_only:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_metadata_doc_hash ON chunk_metadata(doc_hash);")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);")
            elif getattr(self.config, "db_path", None):
                db_p = Path(self.config.db_path)
                if db_p.exists() and os.access(db_p, os.W_OK):
                    try:
                        rw_conn = get_connection(db_p, read_only=False)
                        rw_conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);")
                        rw_conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_metadata_doc_hash ON chunk_metadata(doc_hash);")
                        rw_conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);")
                        rw_conn.close()
                    except Exception:
                        pass
        except Exception:
            pass