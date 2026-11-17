import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _run_clean_python(script: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """
    Executes Python code in a fresh subprocess so real onnxruntime is imported
    without the global stub injected in tests/conftest.py.
    """
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    pp_parts = [str(SRC_DIR), str(REPO_ROOT)]
    if existing_pp:
        pp_parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _get_env_dir(var_name: str) -> str | None:
    val = os.environ.get(var_name, "").strip()
    if not val:
        return None
    expanded = str(Path(val).expanduser().resolve())
    return expanded if os.path.isdir(expanded) else None


@pytest.mark.skipif(
    _get_env_dir("QMD_TEST_EMBED_MODEL_DIR") is None,
    reason="Set QMD_TEST_EMBED_MODEL_DIR to a downloaded EmbeddingGemma ONNX model directory to run live embedder tests.",
)
def test_live_onnx_embedder_smoke():
    embed_dir = _get_env_dir("QMD_TEST_EMBED_MODEL_DIR")
    assert embed_dir is not None

    proc = _run_clean_python(
        """
        import json
        import math
        import os
        import numpy as np
        from qmd.onnx.embedding import EmbeddingGemmaONNX

        embedder = EmbeddingGemmaONNX(model_dir=os.environ["QMD_TEST_EMBED_MODEL_DIR"], use_gpu=False)

        empty_vecs, empty_toks = embedder.embed([])
        assert empty_vecs == [] and empty_toks == 0

        texts = [
            "task: search result | query: sweet orchard fruit and apples",
            "title: Orchard Guide | text: Crisp red apples and pears ripen in autumn orchards.",
            "title: Freight Logistics | text: Heavy diesel semi trucks haul cargo across interstate highways.",
        ]
        vecs, token_count = embedder.embed(texts, batch_size=2)
        assert token_count > 0
        assert len(vecs) == 3

        dim = len(vecs[0])
        assert dim > 0
        for v in vecs:
            assert len(v) == dim
            assert all(isinstance(x, float) and math.isfinite(x) for x in v)
            norm = float(np.linalg.norm(np.array(v, dtype=np.float32)))
            assert abs(norm - 1.0) < 1e-3, f"Expected unit-normalized vector, got norm={norm}"

        q_vec = np.array(vecs[0], dtype=np.float32)
        fruit_vec = np.array(vecs[1], dtype=np.float32)
        truck_vec = np.array(vecs[2], dtype=np.float32)

        sim_fruit = float(np.dot(q_vec, fruit_vec))
        sim_truck = float(np.dot(q_vec, truck_vec))
        assert sim_fruit > sim_truck, f"Expected sim_fruit ({sim_fruit}) > sim_truck ({sim_truck})"
        print(json.dumps({"dim": dim, "sim_fruit": sim_fruit, "sim_truck": sim_truck}))
        """,
        extra_env={"QMD_TEST_EMBED_MODEL_DIR": embed_dir},
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


@pytest.mark.skipif(
    _get_env_dir("QMD_TEST_RERANK_MODEL_DIR") is None,
    reason="Set QMD_TEST_RERANK_MODEL_DIR to a downloaded Ettin ONNX reranker directory to run live reranker tests.",
)
def test_live_onnx_reranker_smoke():
    rerank_dir = _get_env_dir("QMD_TEST_RERANK_MODEL_DIR")
    assert rerank_dir is not None

    proc = _run_clean_python(
        """
        import json
        import math
        import os
        from qmd.llm import LLMClient

        client = LLMClient(
            rerank_provider="onnx",
            rerank_model_dir=os.environ["QMD_TEST_RERANK_MODEL_DIR"],
            onnx_use_gpu=False,
        )

        assert client.rerank("anything", []) == []

        query = "What citrus fruit is high in vitamin C?"
        docs = [
            "A manual transmission uses a clutch pedal and mechanical gear selector.",
            "Oranges, lemons, and grapefruits are citrus fruits rich in vitamin C.",
        ]
        results = client.rerank(query, docs)
        assert len(results) == 2

        for item in results:
            assert "index" in item and "score" in item
            assert math.isfinite(item["score"])
            assert 0.0 <= item["score"] <= 1.0, f"Score out of [0, 1]: {item['score']}"

        assert results[0]["index"] == 1, f"Expected relevant doc index 1 first, got {results}"
        assert results[1]["index"] == 0
        assert results[0]["score"] > results[1]["score"]
        print(json.dumps(results))
        """,
        extra_env={"QMD_TEST_RERANK_MODEL_DIR": rerank_dir},
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


@pytest.mark.skipif(
    _get_env_dir("QMD_TEST_EMBED_MODEL_DIR") is None or _get_env_dir("QMD_TEST_RERANK_MODEL_DIR") is None,
    reason="Set both QMD_TEST_EMBED_MODEL_DIR and QMD_TEST_RERANK_MODEL_DIR to run end-to-end local ONNX search tests.",
)
def test_live_onnx_end_to_end_qmd_pipeline(tmp_path):
    embed_dir = _get_env_dir("QMD_TEST_EMBED_MODEL_DIR")
    rerank_dir = _get_env_dir("QMD_TEST_RERANK_MODEL_DIR")
    assert embed_dir is not None and rerank_dir is not None

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "citrus_tart.md").write_text(
        "# Citrus Tart Recipe\n\nWhisk fresh Meyer lemon juice, blood orange zest, eggs, and sugar into a smooth citrus curd.\n",
        encoding="utf-8",
    )
    (docs_dir / "diesel_engine.md").write_text(
        "# Diesel Engine Maintenance\n\nInspect glow plugs, drain water from the fuel separator, and test turbocharger boost pressure.\n",
        encoding="utf-8",
    )
    (docs_dir / "orbital_mechanics.md").write_text(
        "# Orbital Transfer Maneuvers\n\nA Hohmann transfer uses two impulsive engine burns to move between coplanar circular orbits.\n",
        encoding="utf-8",
    )

    yaml_path = tmp_path / "qmd_live_onnx.yml"
    db_path = tmp_path / "live_qmd.db"
    history_db_path = tmp_path / "live_history.db"

    yaml_path.write_text(
        textwrap.dedent(
            f"""
            db_path: "{db_path}"
            history_db_path: "{history_db_path}"
            cache_search_results: false

            embed_provider: onnx
            rerank_provider: onnx

            embed_model: "google/embeddinggemma-300m"
            embed_model_dir: "{embed_dir}"
            embed_max_length: 2048
            embed_batch_size: 8

            rerank_model: "cross-encoder/ettin-reranker-live"
            rerank_model_dir: "{rerank_dir}"
            rerank_max_length: 8192
            rerank_batch_size: 8

            onnx_use_gpu: false

            collections:
              notes:
                path: "{docs_dir}"
                glob: "*.md"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    proc = _run_clean_python(
        """
        import os
        from qmd.config import load_config
        from qmd.store import Store

        cfg = load_config(os.environ["QMD_LIVE_TEST_YAML"])
        assert cfg.embed_provider == "onnx"
        assert cfg.rerank_provider == "onnx"
        assert cfg.llm_url is None

        store = Store(cfg)
        store.index_collection("notes", cfg.collections["notes"])
        store.build_usearch_index()

        results = store.hybrid_search(
            "how do I make a lemon and orange dessert curd?",
            limit=3,
            rerank=True,
            use_cache=False,
        )
        assert len(results) >= 1, "Expected at least one search result"
        assert results[0].path == "citrus_tart.md", f"Expected citrus_tart.md first, got {[r.path for r in results]}"
        """,
        extra_env={"QMD_LIVE_TEST_YAML": str(yaml_path)},
    )
    assert proc.returncode == 0, f"End-to-end subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"