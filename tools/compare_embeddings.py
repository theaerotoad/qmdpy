#!/usr/bin/env python3
"""
Diagnostic CLI utility to compare embeddings produced by a remote OpenAI-compatible
`/v1/embeddings` endpoint (e.g. llama.cpp) against local ONNX inference (`EmbeddingGemmaONNX`).

Usage example:
    python3 tools/compare_embeddings.py \
        --url http://127.0.0.1:8080 \
        --model embeddinggemma-300m \
        --onnx-path models/embeddinggemma-300m/model_quantized.onnx \
        "how do orbital transfer maneuvers work?"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qmd.llm import LLMClient  # noqa: E402
from qmd.onnx.embedding import EmbeddingGemmaONNX  # noqa: E402


def _normalize_embeddings_endpoint(raw_url: str) -> str:
    """Normalizes a base server URL or full endpoint URL to .../v1/embeddings."""
    cleaned = raw_url.strip().rstrip("/")
    if cleaned.endswith("/v1/embeddings"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/embeddings"
    return f"{cleaned}/v1/embeddings"


def _resolve_model_paths(
    onnx_path: Optional[str],
    model_dir: Optional[str],
) -> Tuple[str, Optional[str]]:
    """
    Resolves (model_dir, explicit_onnx_path) so tokenizer.json and the ONNX graph
    can both be located whether the user passes --onnx-path, --model-dir, or both.
    """
    resolved_onnx: Optional[Path] = Path(onnx_path).expanduser().resolve() if onnx_path else None
    resolved_dir: Optional[Path] = Path(model_dir).expanduser().resolve() if model_dir else None

    if resolved_onnx is not None:
        if not resolved_onnx.exists():
            raise FileNotFoundError(f"ONNX file not found: {resolved_onnx}")
        if resolved_onnx.is_dir():
            resolved_dir = resolved_onnx
            resolved_onnx = None

    if resolved_dir is None and resolved_onnx is not None:
        parent = resolved_onnx.parent
        if (parent / "tokenizer.json").is_file():
            resolved_dir = parent
        elif (parent.parent / "tokenizer.json").is_file():
            resolved_dir = parent.parent
        else:
            resolved_dir = parent

    if resolved_dir is None:
        raise ValueError("Provide at least one of --onnx-path or --model-dir.")

    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {resolved_dir}")

    return str(resolved_dir), (str(resolved_onnx) if resolved_onnx is not None else None)


def _fetch_remote_embeddings(
    endpoint_url: str,
    model: str,
    texts: List[str],
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> List[List[float]]:
    """Calls an OpenAI-compatible /v1/embeddings endpoint and returns ordered float vectors."""
    token = (
        api_key
        or os.environ.get("QMD_EMBED_API_KEY")
        or os.environ.get("QMD_LLM_API_KEY")
        or "sk-no-key-required"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": texts,
        "model": model,
    }
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        resp = client.post(endpoint_url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    items = body.get("data", [])
    if len(items) != len(texts):
        raise RuntimeError(f"Remote endpoint returned {len(items)} embeddings for {len(texts)} inputs.")
    items.sort(key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def _inspect_all_onnx_outputs(
    embedder: EmbeddingGemmaONNX,
    texts: List[str],
) -> Dict[str, np.ndarray]:
    """
    Runs a forward pass on the underlying ONNX session and collects every named output
    reduced to 2D (batch, dim) where applicable (e.g. `sentence_embedding` vs pooled
    `last_hidden_state`), so architectural mismatches can be diagnosed immediately.
    """
    encodings = embedder.tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    onnx_inputs: Dict[str, Any] = {}
    if "input_ids" in embedder.input_names:
        onnx_inputs["input_ids"] = input_ids
    if "attention_mask" in embedder.input_names:
        onnx_inputs["attention_mask"] = attention_mask
    if "token_type_ids" in embedder.input_names:
        onnx_inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

    raw_outputs = embedder.session.run(None, onnx_inputs)
    output_metas = embedder.session.get_outputs()

    extracted: Dict[str, np.ndarray] = {}
    for idx, raw in enumerate(raw_outputs):
        name = output_metas[idx].name if idx < len(output_metas) else f"output_{idx}"
        arr = np.asarray(raw, dtype=np.float32)

        if arr.ndim == 3:
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_emb = np.sum(arr * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            vec = sum_emb / sum_mask
            norms = np.clip(np.linalg.norm(vec, ord=2, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
            extracted[f"{name} (mean_pooled+L2)"] = vec / norms
        elif arr.ndim == 2:
            norms = np.clip(np.linalg.norm(arr, ord=2, axis=-1, keepdims=True), a_min=1e-12, a_max=None)
            extracted[f"{name} (direct+L2)"] = arr / norms

    return extracted


def _compare_vectors(remote_vec: List[float], local_vec: List[float]) -> Dict[str, Any]:
    """Computes similarity and error metrics between two embedding vectors."""
    r = np.asarray(remote_vec, dtype=np.float64)
    l = np.asarray(local_vec, dtype=np.float64)

    r_norm = float(np.linalg.norm(r))
    l_norm = float(np.linalg.norm(l))

    metrics: Dict[str, Any] = {
        "remote_dim": int(r.shape[0]),
        "local_dim": int(l.shape[0]),
        "remote_l2_norm": r_norm,
        "local_l2_norm": l_norm,
        "remote_preview": [round(float(x), 6) for x in r[:6]],
        "local_preview": [round(float(x), 6) for x in l[:6]],
    }

    if r.shape != l.shape:
        metrics["dimension_mismatch"] = True
        return metrics

    denom = max(r_norm * l_norm, 1e-12)
    cosine_sim = float(np.dot(r, l) / denom)
    diff = r - l
    l2_dist = float(np.linalg.norm(diff))
    mae = float(np.mean(np.abs(diff)))
    max_abs_diff = float(np.max(np.abs(diff)))

    r_centered = r - np.mean(r)
    l_centered = l - np.mean(l)
    corr_denom = float(np.linalg.norm(r_centered) * np.linalg.norm(l_centered))
    pearson = float(np.dot(r_centered, l_centered) / corr_denom) if corr_denom > 0 else 0.0

    top_diff_indices = np.argsort(np.abs(diff))[::-1][:5].tolist()

    metrics.update(
        {
            "dimension_mismatch": False,
            "cosine_similarity": cosine_sim,
            "cosine_distance": 1.0 - cosine_sim,
            "l2_distance": l2_dist,
            "mean_abs_diff": mae,
            "max_abs_diff": max_abs_diff,
            "pearson_correlation": pearson,
            "top_5_diff_dims": [
                {
                    "dim": int(i),
                    "remote": round(float(r[i]), 6),
                    "local": round(float(l[i]), 6),
                    "abs_diff": round(float(abs(diff[i])), 6),
                }
                for i in top_diff_indices
            ],
        }
    )
    return metrics


def _format_texts(phrases: List[str], fmt: str, title: str) -> List[str]:
    if fmt == "raw":
        return phrases
    helper = LLMClient()
    if fmt == "query":
        return [helper.format_query_for_embedding(p) for p in phrases]
    if fmt == "doc":
        return [helper.format_doc_for_embedding(title, p) for p in phrases]
    raise ValueError(f"Unsupported format mode: {fmt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare embeddings from a remote HTTP endpoint against a local ONNX model."
    )
    parser.add_argument(
        "phrases",
        nargs="+",
        help="One or more text phrases to embed and compare.",
    )
    parser.add_argument(
        "--url",
        "--embed-url",
        dest="url",
        required=True,
        help="Remote server base URL or /v1/embeddings endpoint (e.g. http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--model",
        "--embed-model",
        dest="model",
        default=os.environ.get("EMBED_MODEL", "embeddinggemma-300m"),
        help="Remote embedding model identifier (default: $EMBED_MODEL or 'embeddinggemma-300m').",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Optional Bearer API key for the remote endpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        dest="onnx_path",
        default=None,
        help="Path to the local .onnx embedding model file.",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=None,
        help="Directory containing tokenizer.json and ONNX model files (inferred from --onnx-path if omitted).",
    )
    parser.add_argument(
        "--format",
        choices=["raw", "query", "doc"],
        default="raw",
        help="Optional EmbeddingGemma prompt wrapper applied before embedding (default: raw).",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Document title used when --format doc is selected.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Tokenizer max sequence length for local ONNX inference (default: 2048).",
    )
    parser.add_argument(
        "--onnx-output",
        dest="onnx_output",
        default=None,
        help="Explicit ONNX output tensor name to use (e.g. 'sentence_embedding' or 'last_hidden_state').",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable CUDAExecutionProvider for local ONNX inference if available.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full comparison diagnostics as JSON.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    endpoint_url = _normalize_embeddings_endpoint(args.url)
    model_dir, explicit_onnx_path = _resolve_model_paths(args.onnx_path, args.model_dir)
    formatted_texts = _format_texts(args.phrases, args.format, args.title)

    remote_vectors = _fetch_remote_embeddings(
        endpoint_url=endpoint_url,
        model=args.model,
        texts=formatted_texts,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    embedder = EmbeddingGemmaONNX(
        model_dir=model_dir,
        onnx_path=explicit_onnx_path,
        max_length=args.max_length,
        use_gpu=args.use_gpu,
        output_name=args.onnx_output,
    )
    local_vectors, token_count = embedder.embed(formatted_texts, batch_size=len(formatted_texts))
    all_onnx_outputs = _inspect_all_onnx_outputs(embedder, formatted_texts)

    comparisons = []
    for idx, (raw_phrase, sent_text, r_vec, l_vec) in enumerate(
        zip(args.phrases, formatted_texts, remote_vectors, local_vectors)
    ):
        comp = _compare_vectors(r_vec, l_vec)
        comp["phrase_index"] = idx
        comp["raw_phrase"] = raw_phrase
        comp["formatted_input"] = sent_text

        tensor_checks: Dict[str, Dict[str, Any]] = {}
        for out_name, out_batch in all_onnx_outputs.items():
            out_comp = _compare_vectors(r_vec, out_batch[idx].tolist())
            if not out_comp.get("dimension_mismatch"):
                tensor_checks[out_name] = {
                    "cosine_similarity": out_comp["cosine_similarity"],
                    "l2_distance": out_comp["l2_distance"],
                    "max_abs_diff": out_comp["max_abs_diff"],
                }
        comp["onnx_output_tensors_vs_remote"] = tensor_checks
        comparisons.append(comp)

    pairwise_summary = None
    if len(formatted_texts) >= 2:
        r_mat = np.asarray(remote_vectors, dtype=np.float64)
        l_mat = np.asarray(local_vectors, dtype=np.float64)
        if r_mat.shape == l_mat.shape:
            r_normed = r_mat / np.clip(np.linalg.norm(r_mat, axis=1, keepdims=True), 1e-12, None)
            l_normed = l_mat / np.clip(np.linalg.norm(l_mat, axis=1, keepdims=True), 1e-12, None)
            r_sim = r_normed @ r_normed.T
            l_sim = l_normed @ l_normed.T
            pairwise_summary = {
                "remote_pairwise_cosine": np.round(r_sim, 6).tolist(),
                "local_pairwise_cosine": np.round(l_sim, 6).tolist(),
                "max_pairwise_sim_diff": float(np.max(np.abs(r_sim - l_sim))),
            }

    report = {
        "remote_endpoint": endpoint_url,
        "remote_model": args.model,
        "local_model_dir": model_dir,
        "local_onnx_path": explicit_onnx_path,
        "local_onnx_output_names": embedder.output_names,
        "local_total_tokens": token_count,
        "format_mode": args.format,
        "comparisons": comparisons,
        "pairwise": pairwise_summary,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== Embedding Comparison: Remote HTTP vs Local ONNX ===")
    print(f"Remote Endpoint : {endpoint_url} (model={args.model})")
    print(f"Local Model Dir : {model_dir}")
    if explicit_onnx_path:
        print(f"Local ONNX File : {explicit_onnx_path}")
    print(f"ONNX Outputs    : {embedder.output_names} (tokens={token_count})")
    print(f"Format Mode     : {args.format}")

    for comp in comparisons:
        print(f"\n--- Phrase [{comp['phrase_index']}] ---")
        print(f"Input Text      : {comp['formatted_input']!r}")
        print(f"Dimensions      : remote={comp['remote_dim']}, local={comp['local_dim']}")
        print(f"L2 Norms        : remote={comp['remote_l2_norm']:.6f}, local={comp['local_l2_norm']:.6f}")
        print(f"Remote [:6]     : {comp['remote_preview']}")
        print(f"Local  [:6]     : {comp['local_preview']}")

        if comp.get("dimension_mismatch"):
            print("Result          : DIMENSION MISMATCH — vectors cannot be compared directly.")
            continue

        print(f"Cosine Sim      : {comp['cosine_similarity']:.6f}  (1.0 = identical direction)")
        print(f"L2 Distance     : {comp['l2_distance']:.6f}  (0.0 = identical)")
        print(f"Mean Abs Diff   : {comp['mean_abs_diff']:.6f}")
        print(f"Max Abs Diff    : {comp['max_abs_diff']:.6f}")
        print(f"Pearson Corr    : {comp['pearson_correlation']:.6f}")

        if comp["onnx_output_tensors_vs_remote"]:
            print("All ONNX Output Tensors vs Remote:")
            for t_name, t_stats in comp["onnx_output_tensors_vs_remote"].items():
                print(
                    f"  - {t_name}: cosine_sim={t_stats['cosine_similarity']:.6f}, "
                    f"l2_dist={t_stats['l2_distance']:.6f}, max_diff={t_stats['max_abs_diff']:.6f}"
                )

    if pairwise_summary is not None:
        print("\n--- Pairwise Similarity Matrix Comparison ---")
        print(f"Max Pairwise Cosine Diff : {pairwise_summary['max_pairwise_sim_diff']:.6f}")
        print(f"Remote Pairwise Matrix   : {pairwise_summary['remote_pairwise_cosine']}")
        print(f"Local Pairwise Matrix    : {pairwise_summary['local_pairwise_cosine']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
