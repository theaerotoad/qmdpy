import pytest
import httpx
from qmd.llm import LLMClient

@pytest.fixture(scope="module")
def live_client():
    """
    Creates a client using defaults (or env vars) for live testing.
    """
    return LLMClient()

def check_connection(client):
    """Helper to check connectivity before running test logic."""
    try:
        client.client.get("/v1/models")
    except httpx.ConnectError:
        pytest.skip(f"Live server not reachable at {client.base_url}")

def test_live_embedding_model(live_client):
    """
    Verifies the configured EMBED_MODEL returns vectors.
    """
    check_connection(live_client)
    
    text = "This is a live connectivity test."
    try:
        # The client wraps the formatting, so we pass raw text
        # But wait, LLMClient.embed_batch expects strings.
        # It's actually the Store that usually formats them.
        # For a raw model test, we'll pass a simple string.
        vectors = live_client.embed_batch([text])
        
        assert len(vectors) == 1
        assert len(vectors[0]) > 0
        assert isinstance(vectors[0][0], float)
        
    except httpx.HTTPStatusError as e:
        pytest.fail(f"Embed model '{live_client.embed_model}' returned {e.response.status_code}: {e.response.text}")

def test_live_generation_model(live_client):
    """
    Verifies the configured GENERATE_MODEL accepts chat prompts.
    """
    check_connection(live_client)
    
    try:
        # We use the actual expand_query method to test the full prompt flow
        # If the model is too dumb to follow the lex/vec format, we might get empty list,
        # but as long as it doesn't 500/404, the connectivity is good.
        results = live_client.expand_query("testing live connection")
        
        # We don't assert len(results) > 0 strictly, because a small model might 
        # just chat back without using the "lex:" prefix.
        # We just want to ensure the request completed successfully.
        assert isinstance(results, list)
        
    except httpx.HTTPStatusError as e:
        pytest.fail(f"Generate model '{live_client.generate_model}' returned {e.response.status_code}: {e.response.text}")

def test_live_rerank_model(live_client):
    """
    Verifies the configured RERANK_MODEL works on the /v1/rerank endpoint.
    """
    check_connection(live_client)
    
    query = "fruit"
    docs = ["apple", "car"]
    
    try:
        results = live_client.rerank(query, docs)
        
        assert len(results) > 0
        first = results[0]
        assert "score" in first
        assert "index" in first
        
        # Sanity check: apple should score higher for fruit than car
        # (Assuming a reasonably competent model)
        # We won't fail on this logic, just checking structure is enough for connectivity.
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            pytest.fail(f"Rerank endpoint /v1/rerank not found on server {live_client.base_url}")
        pytest.fail(f"Rerank model '{live_client.rerank_model}' returned {e.response.status_code}: {e.response.text}")