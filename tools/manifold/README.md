# QMD Document Manifold Learning & Soft Clustering Tool

A standalone research tool for exploring document distributions, UMAP manifold embeddings, and Gaussian Mixture Model (GMM) soft clustering using vectors from an existing QMD SQLite database.

## Key Architecture & Safety Principles

1. **Strictly Read-Only**: Opens your database using SQLite's `?mode=ro` URI parameter. It will **never** alter, write to, or migrate your primary indexing database.
2. **Mean-Pooled Document Vectors**: Pools chunk embeddings per document (`np.mean`) and applies L2 normalization to form clean document-level representations.
3. **Continuous Soft Clustering (GMM)**: Calculates posterior probabilities $P(\text{cluster}_k \mid \text{doc}_i)$ to reveal multi-cluster affinities rather than forcing binary partition boundaries.
4. **Ambiguity / Entropy Measurement**: Computes normalized Shannon entropy for every document to expose "bridge" or "hybrid" documents spanning multiple topics.
5. **Zero-Dependency Interactive UI**: Generates a self-contained HTML5 Canvas visualizer with the data artifact pre-embedded, running completely offline with smooth 60fps pan/zoom.

---

## Installation & Requirements

Ensure you have installed the required scientific libraries:

```bash
pip install umap-learn scikit-learn numpy pyyaml

```

*(If your QMD database uses `sqlite-vec`, ensure `sqlite-vec` is installed in your python environment).*

---

## Quick Start

Run the tool referencing your `example.yml` configuration:

```bash
python tools/manifold/run_manifold.py --config example.yml

```

Or target an explicit database directly:

```bash
python tools/manifold/run_manifold.py --db /path/to/my-database.db

```

This generates:

* `manifold_data.json`: The complete structured JSON artifact with coordinates, soft membership vectors, covariance ellipses, and bridge documents.
* `manifold_explorer.html`: The standalone interactive visualizer with the data embedded.

To explore your results, open `manifold_explorer.html` in any web browser:

```bash
xdg-open manifold_explorer.html  # Linux
open manifold_explorer.html      # macOS

```

---

## CLI Options & Customization

| Option | Default | Description |
| --- | --- | --- |
| `--config`, `-c` | `example.yml` | Path to QMD YAML config file |
| `--db` | `None` | Direct path to SQLite DB (overrides config) |
| `--collection` | `None` | Filter to a specific collection |
| `--n-neighbors` | `15` | UMAP local neighborhood size |
| `--min-dist` | `0.1` | UMAP minimum point distance |
| `--metric` | `cosine` | UMAP distance metric |
| `--n-clusters`, `-k` | `None` (auto) | Fixed cluster count. If omitted, finds best $K$ via BIC |
| `--min-clusters` | `2` | Minimum $K$ to test during BIC auto-selection |
| `--max-clusters` | `16` | Maximum $K$ to test during BIC auto-selection |
| `--covariance-type` | `full` | GMM covariance: `full`, `tied`, `diag`, `spherical` |
| `--output-json`, `-o` | `manifold_data.json` | Path to save the JSON data artifact |
| `--output-html` | `manifold_explorer.html` | Path to save the standalone HTML viewer |
| `--random-state` | `42` | Random seed for reproducibility |

---

## Visualizer Capabilities

* **Pan & Zoom**: Click and drag to pan; mouse wheel to zoom around cursor.
* **Coloring Modes**:
* **Primary Cluster**: Discrete categorical palette with alpha proportional to assignment confidence.
* **Soft Membership Heatmap**: Focus on any individual cluster to view continuous membership decay across all documents.
* **Ambiguity / Entropy**: Blue indicates unambiguous single-cluster documents; orange/red highlights hybrid cross-boundary documents.
* **Collection**: Color by QMD collection origin.


* **Contour Ellipses**: 2-sigma (~95% confidence) Gaussian contour ellipses for every cluster component.
* **Document Detail Drawer**: Click any point to view its soft membership breakdown, metadata, and preview snippet.
* **Bridge Documents Explorer**: Immediate access to the documents exhibiting the highest multi-cluster ambiguity.
* **Portability**: The top bar includes a **"Load JSON"** button to load any other generated `manifold_data.json` run into the same viewer.
