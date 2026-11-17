import pytest
import os
import json
import numpy as np

# We skip these tests if the optional ONNX packages are missing.
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from qmd.onnx import EmbeddingGemmaONNX, EttinONNXReranker
from qmd.onnx.utils import _gelu_numpy, _get_tensor


# --- Fake Utilities for Test Injection ---

class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)

class FakeTokenizer:
    def token_to_id(self, token):
        return 0
    def enable_truncation(self, **kwargs):
        pass
    def enable_padding(self, **kwargs):
        pass
    def encode_batch(self, texts):
        return [FakeEncoding([1, 2, 3]) for _ in texts]

class FakeTokenizerModule:
    @staticmethod
    def from_file(path):
        return FakeTokenizer()

class FakeNodeArg:
    def __init__(self, name):
        self.name = name

class FakeSession:
    def __init__(self, output_value, input_names=["input_ids", "attention_mask"], output_names=["output"]):
        self.output_value = output_value
        self.input_names = input_names
        self.output_names = output_names
        
    def get_inputs(self):
        return [FakeNodeArg(name) for name in self.input_names]
        
    def get_outputs(self):
        return [FakeNodeArg(name) for name in self.output_names]
        
    def get_providers(self):
        return ["CPUExecutionProvider"]
        
    def run(self, output_names, input_feed, run_options=None):
        batch_size = len(input_feed["input_ids"])
        return [np.repeat(self.output_value, batch_size, axis=0)]

@pytest.fixture
def mock_tokenizer_and_files(monkeypatch, tmp_path):
    monkeypatch.setattr("qmd.onnx.embedding.Tokenizer", FakeTokenizerModule)
    monkeypatch.setattr("qmd.onnx.reranker.Tokenizer", FakeTokenizerModule)
    
    # Create fake tokenizer.json
    (tmp_path / "tokenizer.json").write_text("{}")
    
    # Fake valid ONNX file
    fake_onnx = tmp_path / "model.onnx"
    with open(fake_onnx, "wb") as f:
        f.write(b"0" * 2048) # large enough to pass size check
        
    return tmp_path, fake_onnx

# --- Embedding Engine Tests ---

def test_embedder_empty_input(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    def fake_init(*args, **kwargs): return None
    monkeypatch.setattr(ort, "InferenceSession", fake_init)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    
    # Stub constructor to set up minimal state for empty test
    embedder = EmbeddingGemmaONNX(model_dir=str(tmp_path), onnx_path=str(fake_onnx))
    embedder.session = FakeSession(np.zeros((1, 10))) 
    
    vectors, count = embedder.embed([])
    assert vectors == []
    assert count == 0

def test_embedder_2d_output(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    # Mock return shape (batch_size=1, hidden_dim=4)
    fake_session = FakeSession(np.array([[2.0, 0.0, 0.0, 0.0]]))
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_session)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    embedder = EmbeddingGemmaONNX(model_dir=str(tmp_path))
    vectors, count = embedder.embed(["hello world"])
    
    assert len(vectors) == 1
    assert len(vectors[0]) == 4
    assert count == 3 # 3 tokens from FakeEncoding
    
    # L2 normalized by default
    assert pytest.approx(vectors[0][0]) == 1.0
    assert pytest.approx(vectors[0][1]) == 0.0

def test_embedder_3d_mean_pooling(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    # Mock return shape (batch_size=1, seq_len=3, hidden_dim=4)
    # The FakeEncoding has 3 tokens and attention mask [1, 1, 1]
    # Summing seq axis and normalizing should happen
    fake_output = np.array([[[1.0, 1.0, 1.0, 1.0], 
                             [2.0, 2.0, 2.0, 2.0], 
                             [0.0, 0.0, 0.0, 0.0]]])
    
    fake_session = FakeSession(fake_output)
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_session)
    
    embedder = EmbeddingGemmaONNX(model_dir=str(tmp_path), normalize_embeddings=False)
    vectors, _ = embedder.embed(["test"])
    
    assert len(vectors) == 1
    # mean of [1,2,0] is 1
    assert pytest.approx(vectors[0]) == [1.0, 1.0, 1.0, 1.0]

def test_embedder_missing_files_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        EmbeddingGemmaONNX(model_dir=str(tmp_path))

    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="No valid ONNX embedding model found"):
        EmbeddingGemmaONNX(model_dir=str(tmp_path))


# --- Reranker Engine Tests ---

def test_reranker_empty_input(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: FakeSession(np.zeros((1, 1))))
    monkeypatch.setattr(EttinONNXReranker, "_load_classification_head", lambda self: {})
    
    reranker = EttinONNXReranker(model_dir=str(tmp_path))
    scores, count = reranker.predict([])
    assert scores == []
    assert count == 0

def test_reranker_2d_logit_output(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    # Mock return shape (batch_size=1, 1) -> typical direct logit
    fake_session = FakeSession(np.array([[2.0]]))
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_session)
    monkeypatch.setattr(EttinONNXReranker, "_load_classification_head", lambda self: {})
    
    reranker = EttinONNXReranker(model_dir=str(tmp_path), normalize_scores=True)
    scores, count = reranker.predict([("query", "doc1")])
    
    assert len(scores) == 1
    assert count == 3
    # Sigmoid(2.0) is approx 0.88
    assert 0.87 < scores[0] < 0.89

def test_reranker_backbone_only_error(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    # Output name suggests backbone only
    fake_session = FakeSession(np.zeros((1, 3, 10)), output_names=["last_hidden_state"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_session)
    monkeypatch.setattr(EttinONNXReranker, "_load_classification_head", lambda self: {})
    
    with pytest.raises(FileNotFoundError, match="requires the classification head weights"):
        EttinONNXReranker(model_dir=str(tmp_path))

def test_reranker_backbone_with_head_forward(mock_tokenizer_and_files, monkeypatch):
    tmp_path, fake_onnx = mock_tokenizer_and_files
    
    fake_session = FakeSession(np.ones((1, 3, 4)), output_names=["last_hidden_state"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: fake_session)
    
    # Mock classification head weights for a dim=4 network
    head_weights = {
        "d2": {"linear.weight": np.eye(4), "linear.bias": np.zeros(4)},
        "ln3": {"norm.weight": np.ones(4), "norm.bias": np.zeros(4)},
        "d4": {"linear.weight": np.ones((1, 4)), "linear.bias": np.array([0.5])}
    }
    monkeypatch.setattr(EttinONNXReranker, "_load_classification_head", lambda self: head_weights)
    
    reranker = EttinONNXReranker(model_dir=str(tmp_path), normalize_scores=False)
    scores, _ = reranker.predict([("query", "doc")])
    
    assert len(scores) == 1
    assert isinstance(scores[0], float)

def test_utils_gelu():
    # Test numpy gelu approximation matches basic properties
    x = np.array([-1.0, 0.0, 1.0])
    y = _gelu_numpy(x)
    assert y[0] < 0  # approx -0.158
    assert y[1] == 0.0
    assert y[2] > 0  # approx 0.841

def test_utils_get_tensor():
    d = {"layer.weight": 1, "classifier.dense.bias": 2}
    assert _get_tensor(d, "weight") == 1
    assert _get_tensor(d, "dense.bias") == 2
    assert _get_tensor(d, "missing") is None
