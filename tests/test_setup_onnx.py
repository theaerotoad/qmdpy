import io
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx
from onnx import TensorProto, helper
import pytest
from safetensors.numpy import save_file
import numpy as np

import setup_onnx
from qmd.config import load_config


class FakeHTTPResponse:
    """In-memory streaming HTTP response compatible with urllib context manager."""

    def __init__(self, payload: bytes, content_length: int | None = None, fail_after_bytes: int | None = None):
        self._stream = io.BytesIO(payload)
        self._content_length = content_length if content_length is not None else len(payload)
        self._fail_after_bytes = fail_after_bytes
        self._read_so_far = 0

    def getheader(self, name: str, default=None):
        if name.lower() == "content-length" and self._content_length is not None:
            return str(self._content_length)
        return default

    def read(self, size: int = -1) -> bytes:
        if self._fail_after_bytes is not None and self._read_so_far >= self._fail_after_bytes:
            raise ConnectionResetError("Simulated interrupted stream")
        chunk = self._stream.read(size)
        self._read_so_far += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_onnx_bytes(output_name: str = "last_hidden_state", external_data_file: str | None = None) -> bytes:
    """Creates a minimal valid ONNX graph serialized to bytes."""
    inp = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4])
    out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 4])

    initializers = []
    if external_data_file:
        ext_tensor = helper.make_tensor(
            name="ext_weight",
            data_type=TensorProto.FLOAT,
            dims=[4],
            vals=b"",
            raw=True,
        )
        ext_tensor.data_location = TensorProto.EXTERNAL
        entry = ext_tensor.external_data.add()
        entry.key = "location"
        entry.value = external_data_file
        initializers.append(ext_tensor)

    node = helper.make_node("Identity", inputs=["input_ids"], outputs=[output_name])
    graph = helper.make_graph([node], "test_graph", [inp], [out], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    return model.SerializeToString()


def _write_reranker_head(dest_dir: Path) -> None:
    """Writes valid 2_Dense, 3_LayerNorm, and 4_Dense classification head safetensors."""
    for folder, tensors in [
        ("2_Dense", {"linear.weight": np.ones((4, 4), dtype=np.float32)}),
        ("3_LayerNorm", {"norm.weight": np.ones((4,), dtype=np.float32)}),
        ("4_Dense", {"linear.weight": np.ones((1, 4), dtype=np.float32)}),
    ]:
        sub = dest_dir / folder
        sub.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(sub / "model.safetensors"))


def test_manifest_scope_and_reranker_size():
    manifest = setup_onnx.get_onnx_manifest(
        embed_dir="/tmp/embed",
        rerank_dir="/tmp/rerank",
        reranker_size="68m",
    )
    assert set(manifest.keys()) == {"embedding", "reranker"}
    assert manifest["embedding"]["repo_id"] == "onnx-community/embeddinggemma-300m-ONNX"
    assert manifest["reranker"]["repo_id"] == "cross-encoder/ettin-reranker-68m-v1"
    assert manifest["reranker"]["size"] == "68m"

    with pytest.raises(ValueError, match="Unsupported Ettin reranker size"):
        setup_onnx.get_onnx_manifest(reranker_size="999b")


def test_no_vision_or_table_assets_in_script():
    script_text = Path(setup_onnx.__file__).read_text(encoding="utf-8").lower()
    forbidden_terms = [
        "doclaynet",
        "slanet",
        "yolov8",
        "rapidstructure",
        "rapidtable",
        "tesseract",
        "cv2",
        "flask",
    ]
    for term in forbidden_terms:
        assert term not in script_text


def test_streaming_download_and_candidate_fallback(tmp_path):
    dest = tmp_path / "model.onnx"
    payload = b"O" * 2048
    calls = []

    def fake_urlopen(req, timeout=60):
        calls.append(req.full_url)
        if req.full_url.endswith("/missing.onnx"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeHTTPResponse(payload)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        ok = setup_onnx.download_file_with_candidates(
            ["https://example.com/missing.onnx", "https://example.com/found.onnx"],
            str(dest),
            min_size_kb=1,
            quiet=True,
        )

    assert ok is True
    assert len(calls) == 2
    assert dest.read_bytes() == payload
    assert not Path(f"{dest}.tmp").exists()


def test_interrupted_and_http_failed_downloads_clean_temp_files(tmp_path):
    # 1. Interrupted stream mid-download
    dest_interrupted = tmp_path / "interrupted.bin"

    def fake_interrupted(req, timeout=60):
        return FakeHTTPResponse(b"X" * 4096, content_length=4096, fail_after_bytes=1024)

    with patch("urllib.request.urlopen", side_effect=fake_interrupted):
        ok = setup_onnx.download_file_stream(
            "https://example.com/interrupted.bin",
            str(dest_interrupted),
            min_size_kb=1,
            quiet=True,
        )

    assert ok is False
    assert not dest_interrupted.exists()
    assert not Path(f"{dest_interrupted}.tmp").exists()

    # 2. Truncated stream (EOF before Content-Length)
    dest_truncated = tmp_path / "truncated.bin"

    def fake_truncated(req, timeout=60):
        return FakeHTTPResponse(b"Y" * 512, content_length=4096)

    with patch("urllib.request.urlopen", side_effect=fake_truncated):
        ok = setup_onnx.download_file_stream(
            "https://example.com/truncated.bin",
            str(dest_truncated),
            min_bytes=100,
            quiet=True,
        )

    assert ok is False
    assert not dest_truncated.exists()
    assert not Path(f"{dest_truncated}.tmp").exists()

    # 3. HTTP 500 error
    dest_http_err = tmp_path / "err.bin"

    def fake_http_err(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", hdrs=None, fp=None)

    with patch("urllib.request.urlopen", side_effect=fake_http_err):
        ok = setup_onnx.download_file_stream(
            "https://example.com/err.bin",
            str(dest_http_err),
            min_bytes=10,
            quiet=True,
        )

    assert ok is False
    assert not dest_http_err.exists()
    assert not Path(f"{dest_http_err}.tmp").exists()


def test_lfs_pointer_and_undersized_files_rejected(tmp_path):
    lfs_payload = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        b"size 123456789\n"
    )
    dest_lfs = tmp_path / "lfs.onnx"
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(lfs_payload)):
        ok = setup_onnx.download_file_stream(
            "https://example.com/lfs.onnx",
            str(dest_lfs),
            min_bytes=10,
            quiet=True,
        )
    assert ok is False
    assert not dest_lfs.exists()
    assert not Path(f"{dest_lfs}.tmp").exists()

    dest_small = tmp_path / "small.onnx"
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(b"tiny")):
        ok = setup_onnx.download_file_stream(
            "https://example.com/small.onnx",
            str(dest_small),
            min_size_kb=1,
            quiet=True,
        )
    assert ok is False
    assert not dest_small.exists()
    assert not Path(f"{dest_small}.tmp").exists()


def test_embedding_layout_validation_standalone_and_external_data(tmp_path):
    # 1. Graph declaring external data fails when external data file is missing
    ext_dir = tmp_path / "embed_ext"
    ext_dir.mkdir()
    (ext_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (ext_dir / "model_quantized.onnx").write_bytes(
        _make_onnx_bytes(output_name="last_hidden_state", external_data_file="model_quantized.onnx_data")
    )

    valid, msg = setup_onnx.validate_embedding_layout(str(ext_dir))
    assert valid is False
    assert "external tensor data" in msg

    # Adding the companion external data file makes it valid
    (ext_dir / "model_quantized.onnx_data").write_bytes(b"\x00" * 4096)
    valid, _ = setup_onnx.validate_embedding_layout(str(ext_dir))
    assert valid is True

    # 2. Standalone ONNX graph (no external data) validates directly
    standalone_dir = tmp_path / "embed_standalone"
    standalone_dir.mkdir()
    (standalone_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (standalone_dir / "model.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))

    valid, _ = setup_onnx.validate_embedding_layout(str(standalone_dir))
    assert valid is True


def test_reranker_layout_validation_head_requirements(tmp_path):
    # 1. Backbone-only ONNX without classification head safetensors fails
    backbone_dir = tmp_path / "rerank_backbone"
    backbone_dir.mkdir()
    (backbone_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (backbone_dir / "model.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))

    valid, msg = setup_onnx.validate_reranker_layout(str(backbone_dir))
    assert valid is False
    assert "classification head" in msg

    # Adding 2_Dense, 3_LayerNorm, and 4_Dense safetensors makes it valid
    _write_reranker_head(backbone_dir)
    valid, _ = setup_onnx.validate_reranker_layout(str(backbone_dir))
    assert valid is True

    # 2. Direct-logit ONNX graph does not require external classification head safetensors
    logits_dir = tmp_path / "rerank_logits"
    logits_dir.mkdir()
    (logits_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (logits_dir / "model.onnx").write_bytes(_make_onnx_bytes(output_name="logits"))

    valid, _ = setup_onnx.validate_reranker_layout(str(logits_dir))
    assert valid is True


def test_existing_valid_files_reused_unless_forced(tmp_path):
    models_dir = tmp_path / "models"
    embed_dir = models_dir / "embeddinggemma-300m"
    rerank_dir = models_dir / "ettin-reranker-150m-v1"
    embed_dir.mkdir(parents=True)
    rerank_dir.mkdir(parents=True)

    (embed_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (embed_dir / "model_quantized.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))

    (rerank_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (rerank_dir / "model.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))
    _write_reranker_head(rerank_dir)

    # Without --force, urlopen should never be invoked
    with patch("urllib.request.urlopen", side_effect=AssertionError("Should not download when valid")):
        rc = setup_onnx.main(["--models-dir", str(models_dir), "--reranker-size", "150m", "--quiet"])
    assert rc == 0

    # With --force, urlopen IS invoked and fails if network fails
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")) as mock_open:
        rc_forced = setup_onnx.main(
            ["--models-dir", str(models_dir), "--reranker-size", "150m", "--force", "--quiet"]
        )
    assert mock_open.called
    assert rc_forced == 1


def test_generated_yaml_loads_through_qmd_config(tmp_path, capsys):
    models_dir = tmp_path / "models"
    embed_dir = models_dir / "embeddinggemma-300m"
    rerank_dir = models_dir / "ettin-reranker-150m-v1"
    embed_dir.mkdir(parents=True)
    rerank_dir.mkdir(parents=True)

    (embed_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (embed_dir / "model_quantized.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))
    (rerank_dir / "tokenizer.json").write_bytes(b"{" + b" " * 2048 + b"}")
    (rerank_dir / "model.onnx").write_bytes(_make_onnx_bytes(output_name="last_hidden_state"))
    _write_reranker_head(rerank_dir)

    yaml_out = tmp_path / "qmd_onnx.yml"
    rc = setup_onnx.main(
        [
            "--models-dir",
            str(models_dir),
            "--reranker-size",
            "150m",
            "--yaml-out",
            str(yaml_out),
            "--quiet",
        ]
    )
    assert rc == 0
    assert yaml_out.exists()

    cfg = load_config(yaml_out)
    assert cfg.embed_provider == "onnx"
    assert cfg.rerank_provider == "onnx"
    assert cfg.embed_model == "google/embeddinggemma-300m"
    assert cfg.embed_model_dir == str(embed_dir.resolve())
    assert cfg.embed_onnx_path is None
    assert cfg.embed_max_length == 2048
    assert cfg.embed_batch_size == 16
    assert cfg.rerank_model == "cross-encoder/ettin-reranker-150m-v1"
    assert cfg.rerank_model_dir == str(rerank_dir.resolve())
    assert cfg.rerank_onnx_path is None
    assert cfg.rerank_max_length == 8192
    assert cfg.rerank_batch_size == 32
    assert cfg.onnx_use_gpu is False
