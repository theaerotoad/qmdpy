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
