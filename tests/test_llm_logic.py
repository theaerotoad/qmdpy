import pytest
from unittest.mock import MagicMock, patch
from qmd.llm import LLMClient

@pytest.fixture
def mock_httpx():
    with patch("qmd.llm.httpx.Client") as mock:
        yield mock

def test_embedder_sorting_and_extraction(mock_httpx):
    """
    Verify that embeddings are sorted by index to match input order,
    even if the API returns them out of order.
    """
    # Setup mock response with mixed-up indices
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 2, "embedding": [0.3, 0.3]},
            {"index": 0, "embedding": [0.1, 0.1]},
            {"index": 1, "embedding": [0.2, 0.2]}
        ]
    }
    mock_httpx.return_value.post.return_value = mock_response

    client = LLMClient()
    # Input has 3 items
    results = client.embed_batch(["A", "B", "C"])

    # Expect 3 results sorted: 0.1, 0.2, 0.3
    assert len(results) == 3
    assert results[0] == [0.1, 0.1]
    assert results[1] == [0.2, 0.2]
    assert results[2] == [0.3, 0.3]

def test_reranker_payload_structure(mock_httpx):
    """
    Verify the reranker sends correct payload and extracts results.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "score": 0.99},
            {"index": 1, "score": 0.01}
        ]
    }
    mock_httpx.return_value.post.return_value = mock_response

    client = LLMClient(rerank_model="my-reranker")
    results = client.rerank("query", ["doc1", "doc2"])

    # Check output
    assert len(results) == 2
    assert results[0]["score"] == 0.99

    # Check input payload
    args, kwargs = mock_httpx.return_value.post.call_args
    assert kwargs["json"]["model"] == "my-reranker"
    assert kwargs["json"]["query"] == "query"
    assert kwargs["json"]["documents"] == ["doc1", "doc2"]

def test_multimodal_process_image_payload_and_strict_prompt(mock_httpx):
    """
    Verify the multimodal LLM endpoint sends OpenAI-compatible payload
    with system OCR rules, user data URI image, and strips outer fences.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```markdown\n## Chart Overview\n| Month | Sales |\n| --- | --- |\n| Jan | $100 |\n```"
                }
            }
        ]
    }
    mock_httpx.return_value.post.return_value = mock_response

    client = LLMClient(
        multimodal_url="http://127.0.0.1:9999",
        multimodal_model="gpt-4o-mini"
    )
    result = client.process_image(b"\x89PNG\r\n\x1a\nfakeimagebytes", filename="sales_chart.png")

    assert "## Chart Overview" in result
    assert "| Month | Sales |" in result
    assert not result.startswith("```")
    assert not result.endswith("```")

    args, kwargs = mock_httpx.return_value.post.call_args
    assert kwargs["json"]["model"] == "gpt-4o-mini"
    messages = kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert "OCR" in messages[0]["content"]
    assert "visual-analysis" in messages[0]["content"]
    assert "Return only the result:" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")

def test_local_embedder_logic():
    client = LLMClient(embed_provider="onnx", embed_model_dir="dummy")
    
    # Mock engine
    class FakeEngine:
        def embed(self, texts, batch_size):
            # return dim=3 vectors
            return [[0.1, 0.2, 0.3] for _ in texts], 10
            
    fake_engine = FakeEngine()
    client._local_embedder = fake_engine
    
    res = client.embed_batch(["A", "B"])
    assert len(res) == 2
    assert len(res[0]) == 3
    
    # Validation checks
    fake_engine.embed = lambda t, b: ([[0.1, 0.2, 0.3]], 10) # Wrong count
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
        client.embed_batch(["A", "B"])
        
    fake_engine.embed = lambda t, b: ([[], []], 10) # Empty dim
    with pytest.raises(RuntimeError, match="empty vector dimension"):
        client.embed_batch(["A", "B"])
        
    fake_engine.embed = lambda t, b: ([[0.1, 0.2], [0.1, 0.2, 0.3]], 10) # Inconsistent
    with pytest.raises(RuntimeError, match="inconsistent vector dimensions"):
        client.embed_batch(["A", "B"])
        
    import math
    fake_engine.embed = lambda t, b: ([[0.1, math.inf, 0.3], [0.1, 0.2, 0.3]], 10) # Inf
    with pytest.raises(RuntimeError, match="non-finite"):
        client.embed_batch(["A", "B"])

def test_local_reranker_logic():
    client = LLMClient(rerank_provider="onnx", rerank_model_dir="dummy")
    
    class FakeRerankEngine:
        def predict(self, pairs, batch_size):
            # Make the second doc score higher to test sorting
            return [0.1, 0.9], 10
            
    fake_engine = FakeRerankEngine()
    client._local_reranker = fake_engine
    
    res = client.rerank("q", ["d1", "d2"])
    assert len(res) == 2
    assert res[0]["index"] == 1
    assert res[0]["score"] == 0.9
    assert res[1]["index"] == 0
    assert res[1]["score"] == 0.1

    fake_engine.predict = lambda p, b: ([0.1], 10)
    with pytest.raises(RuntimeError, match="returned 1 scores for 2 docs"):
        client.rerank("q", ["d1", "d2"])

def test_lazy_initialization(monkeypatch):
    client = LLMClient(embed_provider="onnx", rerank_provider="onnx")
    
    # Test no onnx raises error safely
    with monkeypatch.context() as m:
        import sys
        m.setitem(sys.modules, "qmd.onnx", None)
        with pytest.raises(RuntimeError, match="ONNX packages are not installed"):
            client._get_local_embedder()
