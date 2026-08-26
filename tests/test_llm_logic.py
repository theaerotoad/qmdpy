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
    assert "NO CONVERSATIONAL PREAMBLE" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")