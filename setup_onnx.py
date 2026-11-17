#!/usr/bin/env python3
"""
Noninteractive Model Setup and Provisioning Script for qmdpy Local ONNX Inference.

Downloads and validates local ONNX weights for:
  - EmbeddingGemma text embeddings (onnx-community/embeddinggemma-300m-ONNX)
  - Ettin cross-encoder reranking (cross-encoder/ettin-reranker-<size>-v1)

Emits a ready-to-paste qmd YAML configuration block upon completion.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parent / "models")
EMBEDDING_REPO_ID = "onnx-community/embeddinggemma-300m-ONNX"
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DEFAULT_SUBDIR = "embeddinggemma-300m"
ETTIN_SIZES = ("17m", "32m", "68m", "150m", "400m", "1b")

STANDALONE_EMBED_MIN_BYTES = 50 * 1024 * 1024  # 50 MB


def is_lfs_pointer(path: str) -> bool:
    """Returns True if path is an unresolved Git LFS text pointer."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) >= 2048:
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256)
        return (
            "git-lfs" in head
            or "oid sha256" in head
            or "version https://git-lfs.github.com/spec" in head
        )
    except OSError:
        return False


def is_valid_file(
    path: str,
    min_size_kb: int = 1,
    min_bytes: Optional[int] = None,
) -> bool:
    """Verifies that a local file exists, is not a Git LFS pointer, and meets size thresholds."""
    if not os.path.isfile(path):
        return False
    if is_lfs_pointer(path):
        return False
    required_bytes = (
        min_bytes
        if min_bytes is not None
        else (min_size_kb * 1024 if min_size_kb > 0 else 1)
    )
    try:
        return os.path.getsize(path) >= required_bytes
    except OSError:
        return False


def _cleanup_temp_files(directory: str) -> None:
    """Removes any leftover .tmp files under directory."""
    if not os.path.isdir(directory):
        return
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".tmp"):
                try:
                    os.remove(os.path.join(root, fname))
                except OSError:
                    pass


def download_file_stream(
    url: str,
    dest_path: str,
    desc: str = "",
    optional: bool = False,
    silent_404: bool = False,
    min_size_kb: int = 1,
    min_bytes: Optional[int] = None,
    quiet: bool = False,
) -> bool:
    """
    Streams a file from url to dest_path using a temporary file and atomic rename.
    Validates Content-Length completeness, minimum size, and Git LFS pointers.
    Always cleans up temporary files on failure.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = f"{dest_path}.tmp"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    headers = {"User-Agent": "Mozilla/5.0 (qmd-ONNX-Setup-Tool)"}
    req = urllib.request.Request(url, headers=headers)

    required_bytes = (
        min_bytes
        if min_bytes is not None
        else (min_size_kb * 1024 if min_size_kb > 0 else 1)
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            total_bytes = int(total_size) if total_size and str(total_size).isdigit() else 0
            downloaded = 0
            block_size = 64 * 1024

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                if not quiet:
                    if total_bytes > 0:
                        percent = downloaded / total_bytes * 100
                        mb_curr = downloaded / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded // total_bytes)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        status_line = (
                            f"\r  {CYAN}[{bar}]{RESET} {percent:5.1f}% "
                            f"({mb_curr:5.1f}/{mb_total:5.1f} MB) {desc}"
                        )
                    else:
                        mb_curr = downloaded / (1024 * 1024)
                        status_line = f"\r  {CYAN}[Downloading]{RESET} {mb_curr:5.1f} MB downloaded {desc}"

                    sys.stdout.write(status_line)
                    sys.stdout.flush()

        if not quiet:
            print()

        if total_bytes > 0 and downloaded < total_bytes:
            if not silent_404 and not optional:
                print(
                    f"  {RED}Error: Incomplete download for {url} "
                    f"({downloaded}/{total_bytes} bytes).{RESET}"
                )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

        if is_lfs_pointer(temp_path):
            if not silent_404 and not optional:
                print(
                    f"  {RED}Error: Received a Git-LFS text pointer rather than binary payload for {url}.{RESET}"
                )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

        actual_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        if actual_size < required_bytes:
            if not silent_404 and not optional:
                print(
                    f"  {RED}Error: Downloaded file too small for {url} "
                    f"({actual_size} bytes < {required_bytes} bytes).{RESET}"
                )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

        os.replace(temp_path, dest_path)
        return True

    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if not silent_404 and not optional:
            print(f"\n  {YELLOW}HTTP Error {e.code} for {url}{RESET}")
        return False
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if not silent_404 and not optional:
            print(f"\n  {RED}Download error for {url}: {e}{RESET}")
        return False


def download_file_with_candidates(
    urls: List[str],
    dest_path: str,
    desc: str = "",
    optional: bool = False,
    min_size_kb: int = 1,
    min_bytes: Optional[int] = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Tries candidate URLs in order until one succeeds, reusing valid local files unless forced."""
    if not force and is_valid_file(dest_path, min_size_kb=min_size_kb, min_bytes=min_bytes):
        if not quiet:
            filename = os.path.basename(dest_path)
            print(f"  {GREEN}Found locally:{RESET} {filename}")
        return True

    for i, url in enumerate(urls):
        is_last = i == len(urls) - 1
        silent = (not is_last) or optional
        success = download_file_stream(
            url,
            dest_path,
            desc=desc,
            optional=optional,
            silent_404=silent,
            min_size_kb=min_size_kb,
            min_bytes=min_bytes,
            quiet=quiet,
        )
        if success:
            return True

    return optional


def _inspect_onnx_graph(onnx_path: str) -> Optional[Dict[str, Any]]:
    """
    Inspects ONNX graph inputs, outputs, and external tensor data references without
    loading heavy external weight blobs. Returns None if onnx is unavailable or the
    file is not a parseable ONNX protobuf.
    """
    try:
        import onnx
    except ImportError:
        return None

    try:
        model = onnx.load_model(onnx_path, load_external_data=False)
        external_files = sorted(
            {
                entry.value
                for init in model.graph.initializer
                if init.data_location == onnx.TensorProto.EXTERNAL
                for entry in init.external_data
                if entry.key == "location" and entry.value
            }
        )
        return {
            "input_names": [inp.name for inp in model.graph.input],
            "output_names": [out.name for out in model.graph.output],
            "external_data_files": external_files,
        }
    except Exception:
        return None


def _is_valid_safetensors_file(path: str) -> bool:
    """Verifies that a safetensors file exists, is not an LFS pointer, and parses cleanly."""
    if not is_valid_file(path, min_bytes=16):
        return False
    try:
        from safetensors.numpy import load_file

        load_file(path)
        return True
    except ImportError:
        return True
    except Exception:
        return False


def validate_embedding_layout(
    dest_dir: str,
    standalone_min_bytes: int = STANDALONE_EMBED_MIN_BYTES,
) -> Tuple[bool, str]:
    """
    Validates that dest_dir holds a complete EmbeddingGemma ONNX layout:
      - valid tokenizer.json
      - valid ONNX graph (standalone or accompanied by required external tensor data)
    """
    tok_path = os.path.join(dest_dir, "tokenizer.json")
    if not is_valid_file(tok_path, min_size_kb=1):
        return False, f"Missing or invalid tokenizer.json in '{dest_dir}'."

    candidates = [
        os.path.join(dest_dir, "model_quantized.onnx"),
        os.path.join(dest_dir, "onnx", "model_quantized.onnx"),
        os.path.join(dest_dir, "model.onnx"),
        os.path.join(dest_dir, "onnx", "model.onnx"),
        os.path.join(dest_dir, "embedding_model.onnx"),
    ]
    for search_dir in [dest_dir, os.path.join(dest_dir, "onnx")]:
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.endswith(".onnx"):
                    cand = os.path.join(search_dir, fname)
                    if cand not in candidates:
                        candidates.append(cand)

    valid_graphs = [c for c in candidates if is_valid_file(c, min_size_kb=1)]
    if not valid_graphs:
        return False, f"No valid ONNX embedding graph found in '{dest_dir}'."

    last_reason = ""
    for onnx_p in valid_graphs:
        graph_info = _inspect_onnx_graph(onnx_p)
        if graph_info is not None:
            ext_files = graph_info["external_data_files"]
            if ext_files:
                missing_ext = [
                    ext
                    for ext in ext_files
                    if not is_valid_file(os.path.join(os.path.dirname(onnx_p), ext), min_bytes=1)
                ]
                if missing_ext:
                    last_reason = (
                        f"ONNX graph '{onnx_p}' requires missing external tensor data: {missing_ext}"
                    )
                    continue
                return True, f"Valid external-data ONNX embedding layout ({os.path.basename(onnx_p)})."
            return True, f"Valid standalone ONNX embedding layout ({os.path.basename(onnx_p)})."

        # Fallback validation when onnx protobuf parser is unavailable or file is synthetic
        if os.path.getsize(onnx_p) >= standalone_min_bytes:
            return True, f"Valid standalone ONNX embedding layout ({os.path.basename(onnx_p)})."

        data_candidates = [
            f"{onnx_p}_data",
            f"{onnx_p}.data",
            os.path.join(dest_dir, "model_quantized.onnx_data"),
            os.path.join(dest_dir, "model_quantized.onnx.data"),
            os.path.join(dest_dir, "model.onnx_data"),
            os.path.join(dest_dir, "model.onnx.data"),
        ]
        if any(is_valid_file(df, min_size_kb=1000) for df in data_candidates):
            return True, f"Valid external-data ONNX embedding layout ({os.path.basename(onnx_p)})."

        last_reason = (
            f"ONNX graph '{onnx_p}' is smaller than 50 MB and is missing its companion "
            f"external tensor data file (.onnx_data)."
        )

    return False, last_reason or f"Incomplete ONNX embedding layout in '{dest_dir}'."


def validate_reranker_layout(dest_dir: str) -> Tuple[bool, str]:
    """
    Validates that dest_dir holds a complete Ettin reranker ONNX layout:
      - valid tokenizer.json
      - valid ONNX graph (plus any external data referenced by the graph)
      - valid classification head safetensors (2_Dense, 3_LayerNorm, 4_Dense or unified model.safetensors)
        whenever the ONNX graph does not directly output logits/scores
    """
    tok_path = os.path.join(dest_dir, "tokenizer.json")
    if not is_valid_file(tok_path, min_size_kb=1):
        return False, f"Missing or invalid tokenizer.json in '{dest_dir}'."

    candidates = [
        os.path.join(dest_dir, "model_quantized.onnx"),
        os.path.join(dest_dir, "onnx", "model_quantized.onnx"),
        os.path.join(dest_dir, "model.onnx"),
        os.path.join(dest_dir, "onnx", "model.onnx"),
        os.path.join(dest_dir, "model_O4.onnx"),
        os.path.join(dest_dir, "onnx", "model_O4.onnx"),
        os.path.join(dest_dir, "model_O3.onnx"),
        os.path.join(dest_dir, "onnx", "model_O3.onnx"),
        os.path.join(dest_dir, "model_O2.onnx"),
        os.path.join(dest_dir, "onnx", "model_O2.onnx"),
        os.path.join(dest_dir, "model_O1.onnx"),
        os.path.join(dest_dir, "onnx", "model_O1.onnx"),
    ]
    for search_dir in [dest_dir, os.path.join(dest_dir, "onnx")]:
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.endswith(".onnx"):
                    cand = os.path.join(search_dir, fname)
                    if cand not in candidates:
                        candidates.append(cand)

    usable_graphs: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for cand in candidates:
        if not is_valid_file(cand, min_size_kb=1):
            continue
        graph_info = _inspect_onnx_graph(cand)
        if graph_info is not None:
            ext_files = graph_info["external_data_files"]
            missing_ext = [
                ext
                for ext in ext_files
                if not is_valid_file(os.path.join(os.path.dirname(cand), ext), min_bytes=1)
            ]
            if missing_ext:
                continue
            usable_graphs.append((cand, graph_info))
        elif os.path.getsize(cand) >= 500 * 1024:
            usable_graphs.append((cand, None))

    if not usable_graphs:
        return False, f"No valid ONNX reranker graph found in '{dest_dir}'."

    for onnx_p, graph_info in usable_graphs:
        if graph_info is not None:
            out_names = [o.lower() for o in graph_info["output_names"]]
            has_direct_logits = any("logits" in o or "score" in o for o in out_names)
            if has_direct_logits:
                return True, f"Valid direct-logit ONNX reranker layout ({os.path.basename(onnx_p)})."

    # Backbone-only graph (or unparsed graph) requires external classification head weights
    d2_cands = [
        os.path.join(dest_dir, "2_Dense", "model.safetensors"),
        os.path.join(dest_dir, "dense", "model.safetensors"),
    ]
    ln3_cands = [
        os.path.join(dest_dir, "3_LayerNorm", "model.safetensors"),
        os.path.join(dest_dir, "layernorm", "model.safetensors"),
    ]
    d4_cands = [
        os.path.join(dest_dir, "4_Dense", "model.safetensors"),
        os.path.join(dest_dir, "dense_1", "model.safetensors"),
    ]

    has_split_head = (
        any(_is_valid_safetensors_file(p) for p in d2_cands)
        and any(_is_valid_safetensors_file(p) for p in ln3_cands)
        and any(_is_valid_safetensors_file(p) for p in d4_cands)
    )
    if has_split_head:
        return True, "Valid backbone + classification-head ONNX reranker layout."

    root_safetensors = [
        os.path.join(dest_dir, "model.safetensors"),
        os.path.join(dest_dir, "pytorch_model.safetensors"),
    ]
    for cand in root_safetensors:
        if _is_valid_safetensors_file(cand):
            try:
                from safetensors.numpy import load_file

                tensors = load_file(cand)
                has_d2 = any(
                    any(x in k for x in ["2_Dense", "2.linear", "classifier.dense"])
                    for k in tensors
                )
                has_ln3 = any(
                    any(x in k for x in ["3_LayerNorm", "3.norm", "classifier.layer_norm", "classifier.norm"])
                    for k in tensors
                )
                has_d4 = any(
                    any(x in k for x in ["4_Dense", "4.linear", "classifier.out_proj", "classifier.linear"])
                    for k in tensors
                )
                if has_d2 and has_ln3 and has_d4:
                    return True, "Valid backbone + unified classification-head ONNX reranker layout."
            except ImportError:
                return True, "Valid backbone + unified classification-head ONNX reranker layout."
            except Exception:
                pass

    return (
        False,
        f"Reranker graph in '{dest_dir}' requires classification head safetensors "
        f"('2_Dense/model.safetensors', '3_LayerNorm/model.safetensors', '4_Dense/model.safetensors'), "
        f"which are missing or invalid.",
    )


def get_onnx_manifest(
    embed_dir: Optional[str] = None,
    rerank_dir: Optional[str] = None,
    reranker_size: str = "150m",
    include_embed: bool = True,
    include_rerank: bool = True,
) -> Dict[str, Any]:
    """Builds the download manifest containing only EmbeddingGemma and Ettin reranker assets."""
    if reranker_size not in ETTIN_SIZES:
        raise ValueError(
            f"Unsupported Ettin reranker size '{reranker_size}'. Choose from: {', '.join(ETTIN_SIZES)}"
        )

    manifest: Dict[str, Any] = {}

    if include_embed:
        e_base = f"https://huggingface.co/{EMBEDDING_REPO_ID}/resolve/main"
        manifest["embedding"] = {
            "repo_id": EMBEDDING_REPO_ID,
            "model_name": EMBEDDING_MODEL_NAME,
            "dest_dir": embed_dir,
            "files": [
                {
                    "local_path": "config.json",
                    "urls": [f"{e_base}/config.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "tokenizer.json",
                    "urls": [f"{e_base}/tokenizer.json"],
                    "optional": False,
                    "min_size_kb": 1,
                },
                {
                    "local_path": "tokenizer_config.json",
                    "urls": [f"{e_base}/tokenizer_config.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "special_tokens_map.json",
                    "urls": [f"{e_base}/special_tokens_map.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "model_quantized.onnx",
                    "urls": [
                        f"{e_base}/onnx/model_quantized.onnx",
                        f"{e_base}/model_quantized.onnx",
                    ],
                    "optional": True,
                    "min_size_kb": 50,
                },
                {
                    "local_path": "model_quantized.onnx_data",
                    "urls": [
                        f"{e_base}/onnx/model_quantized.onnx_data",
                        f"{e_base}/onnx/model_quantized.onnx.data",
                        f"{e_base}/model_quantized.onnx_data",
                        f"{e_base}/model_quantized.onnx.data",
                    ],
                    "optional": True,
                    "min_size_kb": 1000,
                },
                {
                    "local_path": "model.onnx",
                    "urls": [
                        f"{e_base}/onnx/model.onnx",
                        f"{e_base}/model.onnx",
                    ],
                    "optional": True,
                    "min_size_kb": 50,
                },
                {
                    "local_path": "model.onnx_data",
                    "urls": [
                        f"{e_base}/onnx/model.onnx_data",
                        f"{e_base}/onnx/model.onnx.data",
                        f"{e_base}/model.onnx_data",
                        f"{e_base}/model.onnx.data",
                    ],
                    "optional": True,
                    "min_size_kb": 1000,
                },
            ],
        }

    if include_rerank:
        r_repo = f"cross-encoder/ettin-reranker-{reranker_size}-v1"
        r_base = f"https://huggingface.co/{r_repo}/resolve/main"
        manifest["reranker"] = {
            "repo_id": r_repo,
            "model_name": r_repo,
            "size": reranker_size,
            "dest_dir": rerank_dir,
            "files": [
                {
                    "local_path": "config.json",
                    "urls": [f"{r_base}/config.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "tokenizer.json",
                    "urls": [f"{r_base}/tokenizer.json"],
                    "optional": False,
                    "min_size_kb": 1,
                },
                {
                    "local_path": "tokenizer_config.json",
                    "urls": [f"{r_base}/tokenizer_config.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "special_tokens_map.json",
                    "urls": [f"{r_base}/special_tokens_map.json"],
                    "optional": True,
                    "min_bytes": 1,
                },
                {
                    "local_path": "model.onnx",
                    "urls": [
                        f"{r_base}/onnx/model.onnx",
                        f"{r_base}/model.onnx",
                        f"{r_base}/onnx/model_O3.onnx",
                        f"{r_base}/onnx/model_O2.onnx",
                        f"{r_base}/onnx/model_O1.onnx",
                        f"{r_base}/onnx/model_O4.onnx",
                    ],
                    "optional": False,
                    "min_size_kb": 500,
                },
                {
                    "local_path": "model.onnx_data",
                    "urls": [
                        f"{r_base}/onnx/model_O4.onnx_data",
                        f"{r_base}/onnx/model.onnx_data",
                        f"{r_base}/model.onnx_data",
                    ],
                    "optional": True,
                    "min_size_kb": 1000,
                },
                {
                    "local_path": "2_Dense/model.safetensors",
                    "urls": [
                        f"{r_base}/2_Dense/model.safetensors",
                        f"{r_base}/dense/model.safetensors",
                    ],
                    "optional": True,
                    "min_size_kb": 1,
                },
                {
                    "local_path": "3_LayerNorm/model.safetensors",
                    "urls": [
                        f"{r_base}/3_LayerNorm/model.safetensors",
                        f"{r_base}/layernorm/model.safetensors",
                    ],
                    "optional": True,
                    "min_size_kb": 1,
                },
                {
                    "local_path": "4_Dense/model.safetensors",
                    "urls": [
                        f"{r_base}/4_Dense/model.safetensors",
                        f"{r_base}/dense_1/model.safetensors",
                    ],
                    "optional": True,
                    "min_size_kb": 1,
                },
                {
                    "local_path": "model.safetensors",
                    "urls": [f"{r_base}/model.safetensors"],
                    "optional": True,
                    "min_size_kb": 10,
                },
            ],
        }

    return manifest


def download_embedding_gemma(
    dest_dir: str,
    force: bool = False,
    quiet: bool = False,
) -> Tuple[bool, str]:
    """Downloads and validates EmbeddingGemma ONNX model files."""
    if not force:
        valid, _ = validate_embedding_layout(dest_dir)
        if valid:
            if not quiet:
                print(f"  {GREEN}Existing valid EmbeddingGemma weights found in {dest_dir}.{RESET}")
            return True, EMBEDDING_MODEL_NAME

    manifest = get_onnx_manifest(embed_dir=dest_dir, include_embed=True, include_rerank=False)["embedding"]
    os.makedirs(dest_dir, exist_ok=True)
    _cleanup_temp_files(dest_dir)

    if not quiet:
        print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
        print(f"{CYAN}Hugging Face Source:{RESET} {manifest['repo_id']}")

    file_map = {entry["local_path"]: entry for entry in manifest["files"]}

    # 1. Tokenizer and metadata files
    for key in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        spec = file_map[key]
        ok = download_file_with_candidates(
            spec["urls"],
            os.path.join(dest_dir, key),
            desc=f"({key})",
            optional=spec["optional"],
            min_size_kb=spec.get("min_size_kb", 1),
            min_bytes=spec.get("min_bytes"),
            force=force,
            quiet=quiet,
        )
        if not ok and not spec["optional"]:
            _cleanup_temp_files(dest_dir)
            return False, f"Failed to download required EmbeddingGemma file: {key}"

    # 2. ONNX graph + companion external data (prefer quantized graph, fall back to FP32 graph)
    quant_spec = file_map["model_quantized.onnx"]
    got_quant = download_file_with_candidates(
        quant_spec["urls"],
        os.path.join(dest_dir, "model_quantized.onnx"),
        desc="(model_quantized.onnx)",
        optional=True,
        min_size_kb=quant_spec.get("min_size_kb", 50),
        force=force,
        quiet=quiet,
    )
    if got_quant:
        qdata_spec = file_map["model_quantized.onnx_data"]
        download_file_with_candidates(
            qdata_spec["urls"],
            os.path.join(dest_dir, "model_quantized.onnx_data"),
            desc="(model_quantized.onnx_data)",
            optional=True,
            min_size_kb=qdata_spec.get("min_size_kb", 1000),
            force=force,
            quiet=quiet,
        )
    else:
        fp32_spec = file_map["model.onnx"]
        got_fp32 = download_file_with_candidates(
            fp32_spec["urls"],
            os.path.join(dest_dir, "model.onnx"),
            desc="(model.onnx)",
            optional=False,
            min_size_kb=fp32_spec.get("min_size_kb", 50),
            force=force,
            quiet=quiet,
        )
        if not got_fp32:
            _cleanup_temp_files(dest_dir)
            return False, "Failed to download EmbeddingGemma ONNX model graph."

        fdata_spec = file_map["model.onnx_data"]
        download_file_with_candidates(
            fdata_spec["urls"],
            os.path.join(dest_dir, "model.onnx_data"),
            desc="(model.onnx_data)",
            optional=True,
            min_size_kb=fdata_spec.get("min_size_kb", 1000),
            force=force,
            quiet=quiet,
        )

    valid, msg = validate_embedding_layout(dest_dir)
    if not valid:
        _cleanup_temp_files(dest_dir)
        return False, msg

    return True, EMBEDDING_MODEL_NAME


def download_ettin_reranker(
    dest_dir: str,
    size: str = "150m",
    force: bool = False,
    quiet: bool = False,
) -> Tuple[bool, str]:
    """Downloads and validates Ettin cross-encoder ONNX model and classification head files."""
    manifest = get_onnx_manifest(
        rerank_dir=dest_dir,
        reranker_size=size,
        include_embed=False,
        include_rerank=True,
    )["reranker"]
    repo_id = manifest["repo_id"]

    if not force:
        valid, _ = validate_reranker_layout(dest_dir)
        if valid:
            if not quiet:
                print(f"  {GREEN}Existing valid Ettin Reranker ({size}) weights found in {dest_dir}.{RESET}")
            return True, repo_id

    os.makedirs(dest_dir, exist_ok=True)
    _cleanup_temp_files(dest_dir)

    if not quiet:
        print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
        print(f"{CYAN}Hugging Face Source:{RESET} {repo_id}")

    for spec in manifest["files"]:
        local_rel = spec["local_path"]
        target_path = os.path.join(dest_dir, local_rel)
        ok = download_file_with_candidates(
            spec["urls"],
            target_path,
            desc=f"({local_rel})",
            optional=spec["optional"],
            min_size_kb=spec.get("min_size_kb", 1),
            min_bytes=spec.get("min_bytes"),
            force=force,
            quiet=quiet,
        )
        if not ok and not spec["optional"]:
            _cleanup_temp_files(dest_dir)
            return False, f"Failed to download required Ettin file: {local_rel}"

    valid, msg = validate_reranker_layout(dest_dir)
    if not valid:
        _cleanup_temp_files(dest_dir)
        return False, msg

    return True, repo_id


def quantize_onnx_model(input_onnx_path: str, output_onnx_path: str, quiet: bool = False) -> bool:
    """Quantizes an ONNX FP32 reranker graph to dynamic INT8 using onnxruntime.quantization."""
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as err:
        print(
            f"  {YELLOW}Quantization dependency missing ({err}). "
            f"Install optional ONNX packages: pip install -r requirements_onnx.txt{RESET}"
        )
        return False

    if not quiet:
        print(f"  {CYAN}Quantizing ONNX model to INT8 (dynamic)...{RESET}")
        print(f"  Input : {input_onnx_path} ({os.path.getsize(input_onnx_path) / (1024 * 1024):.1f} MB)")

    temp_output = f"{output_onnx_path}.tmp"
    try:
        quantize_dynamic(
            model_input=input_onnx_path,
            model_output=temp_output,
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
        if not is_valid_file(temp_output, min_bytes=1024):
            if os.path.exists(temp_output):
                os.remove(temp_output)
            print(f"  {RED}Quantization output invalid or missing.{RESET}")
            return False

        test_opt = ort.SessionOptions()
        test_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        _ = ort.InferenceSession(temp_output, test_opt, providers=["CPUExecutionProvider"])

        os.replace(temp_output, output_onnx_path)
        if not quiet:
            print(
                f"  {GREEN}Quantized model verified and saved:{RESET} {output_onnx_path} "
                f"({os.path.getsize(output_onnx_path) / (1024 * 1024):.1f} MB)"
            )
        return True
    except Exception as e:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                pass
        print(f"  {RED}Failed to quantize model: {e}{RESET}")
        return False


def render_qmd_yaml(
    embed_dir: Optional[str] = None,
    embed_model: str = EMBEDDING_MODEL_NAME,
    rerank_dir: Optional[str] = None,
    rerank_model: str = "cross-encoder/ettin-reranker-150m-v1",
    use_gpu: bool = False,
) -> str:
    """Renders a ready-to-paste qmd YAML snippet with resolved absolute paths."""
    embed_provider = "onnx" if embed_dir else "remote"
    rerank_provider = "onnx" if rerank_dir else "remote"

    abs_embed_dir = (
        Path(embed_dir).expanduser().resolve().as_posix().replace("\\", "/")
        if embed_dir
        else None
    )
    abs_rerank_dir = (
        Path(rerank_dir).expanduser().resolve().as_posix().replace("\\", "/")
        if rerank_dir
        else None
    )

    lines = [
        "# qmd local ONNX embedding & reranking configuration",
        f"embed_provider: {embed_provider}",
        f"rerank_provider: {rerank_provider}",
        "",
    ]

    if abs_embed_dir:
        lines.extend(
            [
                f'embed_model: "{embed_model}"',
                f'embed_model_dir: "{abs_embed_dir}"',
                "embed_onnx_path: null",
                "embed_max_length: 2048",
                "embed_batch_size: 16",
                "",
            ]
        )

    if abs_rerank_dir:
        lines.extend(
            [
                f'rerank_model: "{rerank_model}"',
                f'rerank_model_dir: "{abs_rerank_dir}"',
                "rerank_onnx_path: null",
                "rerank_max_length: 8192",
                "rerank_batch_size: 32",
                "",
            ]
        )

    lines.append(f"onnx_use_gpu: {'true' if use_gpu else 'false'}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate EmbeddingGemma & Ettin ONNX models for qmdpy."
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Base directory for downloaded models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--embed-dir",
        default=None,
        help="Explicit directory for EmbeddingGemma weights (default: <models-dir>/embeddinggemma-300m)",
    )
    parser.add_argument(
        "--rerank-dir",
        default=None,
        help="Explicit directory for Ettin reranker weights (default: <models-dir>/ettin-reranker-<size>-v1)",
    )
    parser.add_argument(
        "--reranker-size",
        choices=list(ETTIN_SIZES),
        default="150m",
        help="Ettin cross-encoder model variant (default: 150m)",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Download and validate only the EmbeddingGemma model",
    )
    parser.add_argument(
        "--rerank-only",
        action="store_true",
        help="Download and validate only the Ettin reranker model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing local model files",
    )
    parser.add_argument(
        "--quantize-reranker",
        action="store_true",
        help="Quantize the Ettin reranker ONNX model to dynamic INT8 after download",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable onnx_use_gpu: true in the emitted qmd YAML snippet",
    )
    parser.add_argument(
        "--yaml-out",
        default=None,
        help="Optional file path to save the emitted qmd YAML snippet",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress streaming download progress bars",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.embed_only and args.rerank_only:
        print(f"{RED}Error: --embed-only and --rerank-only cannot be used together.{RESET}", file=sys.stderr)
        return 1

    include_embed = not args.rerank_only
    include_rerank = not args.embed_only

    base_models_dir = Path(args.models_dir).expanduser().resolve()
    embed_dir = (
        str(Path(args.embed_dir).expanduser().resolve())
        if args.embed_dir
        else str(base_models_dir / EMBEDDING_DEFAULT_SUBDIR)
    )
    rerank_dir = (
        str(Path(args.rerank_dir).expanduser().resolve())
        if args.rerank_dir
        else str(base_models_dir / f"ettin-reranker-{args.reranker_size}-v1")
    )

    embed_model_name = EMBEDDING_MODEL_NAME
    rerank_model_name = f"cross-encoder/ettin-reranker-{args.reranker_size}-v1"

    if include_embed:
        ok, info = download_embedding_gemma(embed_dir, force=args.force, quiet=args.quiet)
        if not ok:
            print(f"{RED}EmbeddingGemma setup failed: {info}{RESET}", file=sys.stderr)
            return 1
        embed_model_name = info

    if include_rerank:
        ok, info = download_ettin_reranker(
            rerank_dir,
            size=args.reranker_size,
            force=args.force,
            quiet=args.quiet,
        )
        if not ok:
            print(f"{RED}Ettin Reranker setup failed: {info}{RESET}", file=sys.stderr)
            return 1
        rerank_model_name = info

        if args.quantize_reranker:
            quantized_path = os.path.join(rerank_dir, "model_quantized.onnx")
            if args.force or not is_valid_file(quantized_path, min_bytes=1024):
                base_onnx = None
                for cand in ["model.onnx", "onnx/model.onnx", "onnx/model_O3.onnx", "onnx/model_O4.onnx"]:
                    p = os.path.join(rerank_dir, cand)
                    if is_valid_file(p, min_bytes=1024):
                        base_onnx = p
                        break
                if not base_onnx or not quantize_onnx_model(base_onnx, quantized_path, quiet=args.quiet):
                    print(f"{RED}Ettin Reranker quantization failed.{RESET}", file=sys.stderr)
                    return 1

    yaml_block = render_qmd_yaml(
        embed_dir=embed_dir if include_embed else None,
        embed_model=embed_model_name,
        rerank_dir=rerank_dir if include_rerank else None,
        rerank_model=rerank_model_name,
        use_gpu=args.use_gpu,
    )

    if args.yaml_out:
        out_path = Path(args.yaml_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_block, encoding="utf-8")

    print(f"\n{GREEN}{BOLD}ONNX model setup complete.{RESET}")
    print(f"{BOLD}Add the following to your qmd YAML configuration:{RESET}\n")
    print(yaml_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
