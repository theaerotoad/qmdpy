#!/usr/bin/env python3
"""
Document Manifold Learning & Soft Clustering Tool for QMD Databases.

Extracts chunk embeddings per document from a QMD SQLite database (read-only),
computes mean-pooled document vectors (L2-normalized), performs UMAP manifold
projection, and fits a Gaussian Mixture Model (GMM) to obtain continuous
soft cluster memberships and ambiguity/entropy metrics.

Outputs:
  1. A structured JSON data artifact containing coordinates, soft memberships,
     cluster parameters, covariance ellipses, and bridge documents.
  2. A standalone interactive HTML/JS visualization to interrogate the manifold.
"""

import os
import sys
import json
import math
import struct
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Ensure optional scientific packages are clearly reported if missing
try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: 'numpy'. Please install via: pip install numpy")

try:
    import umap
except ImportError:
    sys.exit("Missing dependency: 'umap-learn'. Please install via: pip install umap-learn")

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    sys.exit("Missing dependency: 'scikit-learn'. Please install via: pip install scikit-learn")

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: 'pyyaml'. Please install via: pip install pyyaml")


def decompress_chunk_text(blob: Any) -> str:
    """Decompresses zlib-compressed chunk text or returns raw string."""
    if not blob:
        return ""
    if isinstance(blob, str):
        return blob
    try:
        import zlib
        return zlib.decompress(blob).decode("utf-8", errors="replace")
    except Exception:
        try:
            return blob.decode("utf-8", errors="replace")
        except Exception:
            return ""


def decode_vector_np(blob: bytes, dim: Optional[int] = None, quant_type: str = "none") -> Optional[np.ndarray]:
    """Decodes vector blob into a float32 numpy array matching QMD quantization formats."""
    if not blob:
        return None
    quant = (quant_type or "none").lower()
    try:
        if quant in ("int8",):
            arr = np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0
            if dim and len(arr) != dim:
                arr = arr[:dim]
            return arr
        elif quant in ("bit", "binary"):
            if dim is None:
                dim = len(blob) * 8
            floats = []
            for i in range(dim):
                byte_val = blob[i // 8]
                bit_set = (byte_val & (1 << (i % 8))) != 0
                floats.append(1.0 if bit_set else -1.0)
            return np.array(floats, dtype=np.float32)
        else:
            arr = np.frombuffer(blob, dtype=np.float32)
            if dim and len(arr) != dim:
                arr = arr[:dim]
            return arr
    except Exception as e:
        print(f"Warning: Failed to decode vector: {e}", file=sys.stderr)
        return None


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Connects to SQLite database in strictly read-only mode."""
    resolved_path = db_path.resolve().as_posix()
    uri = f"file:{resolved_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)

    # Attempt to load sqlite-vec if present, silently ignore if not installed
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass

    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA cache_size = -64000;")
    return conn


def extract_document_vectors(
    conn: sqlite3.Connection,
    collection_filter: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray, int, str]:
    """
    Queries documents and their chunk vectors, computing the mean-pooled,
    L2-normalized vector for each document.
    """
    cursor = conn.cursor()

    # Read vector configuration metadata
    meta = {}
    try:
        cursor.execute("SELECT key, value FROM db_meta")
        meta = dict(cursor.fetchall())
    except Exception:
        pass

    vector_dim = int(meta.get("vector_dim", 0)) if meta.get("vector_dim") else None
    quant_type = meta.get("vector_quantization", "none").lower()

    # Build document query
    sql = """
    SELECT 
        d.id AS doc_id,
        d.collection,
        d.path,
        d.title,
        d.doc_date,
        d.hash AS doc_hash,
        cm.seq_id,
        cm.chunk_text,
        cm.headers,
        v.embedding
    FROM documents d
    JOIN chunk_metadata cm ON d.hash = cm.doc_hash
    JOIN vectors v ON cm.rowid = v.rowid
    WHERE d.active = 1
    """
    params = []
    if collection_filter:
        sql += " AND d.collection = ?"
        params.append(collection_filter)

    sql += " ORDER BY d.id, cm.seq_id"

    cursor.execute(sql, tuple(params))

    doc_records: List[Dict[str, Any]] = []
    doc_vectors_list: List[np.ndarray] = []

    current_doc_id = None
    current_meta: Optional[Dict[str, Any]] = None
    current_chunk_vecs: List[np.ndarray] = []
    first_chunk_snippet: str = ""

    def finalize_current():
        nonlocal current_meta, current_chunk_vecs, first_chunk_snippet
        if not current_meta or not current_chunk_vecs:
            return
        # Mean pool across chunks
        matrix = np.vstack(current_chunk_vecs)
        mean_vec = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-12:
            mean_vec = mean_vec / norm

        current_meta["chunk_count"] = len(current_chunk_vecs)
        current_meta["snippet"] = first_chunk_snippet[:350].strip()
        doc_records.append(current_meta)
        doc_vectors_list.append(mean_vec)

    for row in cursor:
        doc_id, coll, path_str, title_str, doc_date, doc_hash, seq_id, chunk_blob, headers, emb_blob = row

        if current_doc_id is not None and doc_id != current_doc_id:
            finalize_current()
            current_chunk_vecs = []
            first_chunk_snippet = ""

        current_doc_id = doc_id
        if not current_chunk_vecs:
            title = title_str if title_str and title_str.strip() else Path(path_str).name
            current_meta = {
                "id": doc_id,
                "collection": coll,
                "path": path_str,
                "title": title,
                "doc_date": doc_date,
                "doc_hash": doc_hash
            }
            raw_text = decompress_chunk_text(chunk_blob)
            prefix = f"[{headers}] " if headers else ""
            first_chunk_snippet = prefix + raw_text.replace("\n", " ")

        vec = decode_vector_np(emb_blob, dim=vector_dim, quant_type=quant_type)
        if vec is not None:
            current_chunk_vecs.append(vec)

    finalize_current()

    if not doc_vectors_list:
        return [], np.empty((0, 0)), 0, quant_type

    X = np.vstack(doc_vectors_list)
    actual_dim = X.shape[1]
    return doc_records, X, actual_dim, quant_type


def compute_covariance_ellipses(
    means: np.ndarray,
    covariances: np.ndarray,
    cov_type: str
) -> List[Dict[str, Any]]:
    """Calculates 2-sigma (~95% confidence) contour ellipses for 2D Gaussians."""
    ellipses = []
    k_components = means.shape[0]

    for k in range(k_components):
        mean_k = means[k]
        if cov_type == "full":
            cov_k = covariances[k]
        elif cov_type == "diag":
            cov_k = np.diag(covariances[k])
        elif cov_type == "spherical":
            cov_k = np.eye(2) * covariances[k]
        elif cov_type == "tied":
            cov_k = covariances
        else:
            cov_k = np.eye(2)

        vals, vecs = np.linalg.eigh(cov_k)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # 2 standard deviations for 95% ellipse
        rx = 2.0 * float(np.sqrt(max(1e-5, vals[0])))
        ry = 2.0 * float(np.sqrt(max(1e-5, vals[1])))
        angle_deg = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))

        ellipses.append({
            "cx": round(float(mean_k[0]), 4),
            "cy": round(float(mean_k[1]), 4),
            "rx": round(rx, 4),
            "ry": round(ry, 4),
            "angle": round(angle_deg, 2)
        })
    return ellipses


def extract_cluster_topics(
    doc_records: List[Dict[str, Any]],
    soft_probs: np.ndarray,
    n_clusters: int
) -> List[List[str]]:
    """Extracts distinctive topical terms for each cluster using weighted TF-IDF."""
    corpus = [f"{d['title']} {d['title']} {d['snippet']}" for d in doc_records]
    try:
        tfidf = TfidfVectorizer(
            max_features=600,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        X_tfidf = tfidf.fit_transform(corpus)
        vocab = np.array(tfidf.get_feature_names_out())

        cluster_keywords: List[List[str]] = []
        for k in range(n_clusters):
            weights = soft_probs[:, k]
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                profile = np.asarray(X_tfidf.T.dot(weights)).ravel() / weight_sum
                top_indices = profile.argsort()[::-1][:7]
                terms = [vocab[i] for i in top_indices if profile[i] > 0]
                cluster_keywords.append(terms)
            else:
                cluster_keywords.append([])
        return cluster_keywords
    except Exception:
        return [[] for _ in range(n_clusters)]


def generate_embedded_html(artifact_json_str: str, template_path: Optional[Path] = None) -> str:
    """Generates a self-contained HTML explorer with the data artifact pre-embedded."""
    if template_path and template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    else:
        # Check standard adjacent directory
        adjacent = Path(__file__).parent / "visualizer.html"
        if adjacent.exists():
            with open(adjacent, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            raise FileNotFoundError("Could not locate 'visualizer.html' template.")

    placeholder = '<script id="manifold-data" type="application/json">{}</script>'
    replacement = f'<script id="manifold-data" type="application/json">\n{artifact_json_str}\n</script>'

    if placeholder in template:
        return template.replace(placeholder, replacement)
    elif 'id="manifold-data"' in template:
        import re
        return re.sub(
            r'<script id="manifold-data" type="application/json">.*?</script>',
            replacement,
            template,
            flags=re.DOTALL
        )
    else:
        # Insert before </head>
        return template.replace("</head>", f"{replacement}\n</head>")


def main():
    parser = argparse.ArgumentParser(
        description="Extract QMD document embeddings, compute UMAP manifold, and fit GMM soft clustering."
    )
    parser.add_argument("--config", "-c", type=str, default="example.yml", help="Path to QMD config.yml")
    parser.add_argument("--db", type=str, default=None, help="Direct path to SQLite database (overrides config)")
    parser.add_argument("--collection", type=str, default=None, help="Filter to a specific collection name")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", type=str, default="cosine", help="UMAP distance metric (default: cosine)")
    parser.add_argument("--n-clusters", "-k", type=int, default=None, help="Fixed cluster count (default: auto via BIC)")
    parser.add_argument("--min-clusters", type=int, default=2, help="Min clusters for auto BIC selection (default: 2)")
    parser.add_argument("--max-clusters", type=int, default=16, help="Max clusters for auto BIC selection (default: 16)")
    parser.add_argument("--covariance-type", type=str, default="full", choices=["full", "tied", "diag", "spherical"], help="GMM covariance type")
    parser.add_argument("--output-json", "-o", type=str, default="manifold_data.json", help="Output path for JSON artifact")
    parser.add_argument("--output-html", type=str, default="manifold_explorer.html", help="Output path for standalone HTML visualizer")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Determine database path
    db_path_str = args.db
    if not db_path_str and args.config:
        cfg_file = Path(args.config)
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
                raw_db = cfg_data.get("db_path", "./example.db")
                db_path_str = str((cfg_file.parent / raw_db).resolve())
            except Exception as e:
                print(f"Warning: Failed to parse config file: {e}", file=sys.stderr)

    if not db_path_str:
        db_path_str = "./example.db"

    db_path = Path(db_path_str)
    if not db_path.exists():
        sys.exit(f"Error: SQLite database not found at '{db_path}'. Specify with --db or check your config.")

    print(f"Connecting to database (read-only): {db_path}")
    conn = get_readonly_connection(db_path)

    print("Extracting document chunks and mean-pooling embeddings...")
    doc_records, X, dim, quant_type = extract_document_vectors(conn, collection_filter=args.collection)
    conn.close()

    n_docs = len(doc_records)
    print(f"Loaded {n_docs} documents with {dim}-dimensional vectors (Quantization: {quant_type}).")

    if n_docs < 2:
        sys.exit("Error: At least 2 documents with embeddings are required for manifold learning.")

    # Parameter adjustments for small datasets
    effective_neighbors = min(args.n_neighbors, n_docs - 1)
    effective_neighbors = max(2, effective_neighbors)

    print(f"Running UMAP projection (n_neighbors={effective_neighbors}, min_dist={args.min_dist}, metric='{args.metric}')...")
    reducer = umap.UMAP(
        n_neighbors=effective_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_components=2,
        random_state=args.random_state
    )
    coords_2d = reducer.fit_transform(X)

    # GMM Model Selection
    bic_records = []
    if args.n_clusters is not None:
        best_k = min(args.n_clusters, n_docs)
        print(f"Fitting GMM with specified cluster count k={best_k}...")
        gmm = GaussianMixture(
            n_components=best_k,
            covariance_type=args.covariance_type,
            random_state=args.random_state,
            max_iter=300
        )
        gmm.fit(coords_2d)
    else:
        k_min = max(2, args.min_clusters)
        k_max = min(args.max_clusters, n_docs - 1)
        if k_max < k_min:
            k_max = k_min

        print(f"Evaluating optimal cluster count k in range [{k_min}, {k_max}] using BIC...")
        best_bic = float("inf")
        best_k = k_min
        best_gmm = None

        for k in range(k_min, k_max + 1):
            cand_gmm = GaussianMixture(
                n_components=k,
                covariance_type=args.covariance_type,
                random_state=args.random_state,
                max_iter=200
            )
            cand_gmm.fit(coords_2d)
            bic = cand_gmm.bic(coords_2d)
            aic = cand_gmm.aic(coords_2d)
            bic_records.append({"k": k, "bic": round(float(bic), 2), "aic": round(float(aic), 2)})
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = cand_gmm

        print(f"Optimal cluster count selected: k={best_k} (BIC: {best_bic:.2f})")
        gmm = best_gmm

    # Soft membership inference
    soft_probs = gmm.predict_proba(coords_2d)  # (N, K)
    primary_clusters = np.argmax(soft_probs, axis=1)
    confidences = np.max(soft_probs, axis=1)

    # Shannon Entropy for ambiguity / bridge measurement
    eps = 1e-12
    clipped = np.clip(soft_probs, eps, 1.0)
    entropies = -np.sum(clipped * np.log2(clipped), axis=1)
    max_h = math.log2(best_k) if best_k > 1 else 1.0
    norm_entropies = np.clip(entropies / max_h, 0.0, 1.0)

    # Covariance ellipses for visualizer
    ellipses = compute_covariance_ellipses(gmm.means_, gmm.covariances_, args.covariance_type)

    # Distinctive keywords per cluster
    cluster_topics = extract_cluster_topics(doc_records, soft_probs, best_k)

    # Format document objects
    formatted_docs = []
    for i, d in enumerate(doc_records):
        probs_i = soft_probs[i]
        memberships = [
            {"cluster": int(c), "prob": round(float(probs_i[c]), 4)}
            for c in range(best_k)
            if probs_i[c] >= 0.005
        ]
        memberships.sort(key=lambda x: x["prob"], reverse=True)

        formatted_docs.append({
            "id": d["id"],
            "title": d["title"],
            "path": d["path"],
            "collection": d["collection"],
            "doc_date": d["doc_date"],
            "chunks_count": d["chunk_count"],
            "snippet": d["snippet"],
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4),
            "primary_cluster": int(primary_clusters[i]),
            "confidence": round(float(confidences[i]), 4),
            "entropy": round(float(norm_entropies[i]), 4),
            "memberships": memberships
        })

    # Cluster summaries
    cluster_summaries = []
    for k in range(best_k):
        hard_count = int(np.sum(primary_clusters == k))
        soft_weight = round(float(np.sum(soft_probs[:, k])), 2)
        top_exemplar_indices = np.argsort(soft_probs[:, k])[::-1][:5]
        exemplars = [
            {
                "id": doc_records[idx]["id"],
                "title": doc_records[idx]["title"],
                "path": doc_records[idx]["path"],
                "prob": round(float(soft_probs[idx, k]), 4)
            }
            for idx in top_exemplar_indices
            if soft_probs[idx, k] >= 0.05
        ]

        cluster_summaries.append({
            "id": k,
            "hard_count": hard_count,
            "soft_weight": soft_weight,
            "terms": cluster_topics[k],
            "ellipse": ellipses[k],
            "exemplars": exemplars
        })

    # Top bridge documents (documents with high ambiguity across clusters)
    bridge_indices = np.argsort(norm_entropies)[::-1][:15]
    bridge_documents = [
        {
            "id": doc_records[idx]["id"],
            "title": doc_records[idx]["title"],
            "path": doc_records[idx]["path"],
            "collection": doc_records[idx]["collection"],
            "entropy": round(float(norm_entropies[idx]), 4),
            "top_clusters": sorted(
                [{"cluster": c, "prob": round(float(soft_probs[idx, c]), 4)} for c in range(best_k)],
                key=lambda x: x["prob"],
                reverse=True
            )[:3]
        }
        for idx in bridge_indices
        if norm_entropies[idx] > 0.35
    ]

    artifact = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "db_path": str(db_path),
            "total_documents": n_docs,
            "embedding_dim": dim,
            "quantization": quant_type,
            "umap_params": {
                "n_neighbors": effective_neighbors,
                "min_dist": args.min_dist,
                "metric": args.metric
            },
            "gmm_params": {
                "n_clusters": best_k,
                "covariance_type": args.covariance_type,
                "auto_selected": args.n_clusters is None
            },
            "bic_scores": bic_records
        },
        "clusters": cluster_summaries,
        "documents": formatted_docs,
        "bridge_documents": bridge_documents
    }

    # Save JSON artifact
    out_json = Path(args.output_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved manifold JSON artifact to: {out_json}")

    # Generate standalone HTML
    if args.output_html:
        out_html = Path(args.output_html)
        artifact_json_str = json.dumps(artifact)
        try:
            rendered_html = generate_embedded_html(artifact_json_str)
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(rendered_html)
            print(f"Saved standalone interactive HTML viewer to: {out_html}")
            print(f"Open in your browser: file://{out_html.resolve()}")
        except Exception as e:
            print(f"Warning: Could not build standalone HTML: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()