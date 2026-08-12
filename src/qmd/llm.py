import os
import json
import httpx
from typing import List, Dict, Optional
from tqdm import tqdm

class LLMClient:
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        api_key: Optional[str] = None,
        embed_url: Optional[str] = None,
        rerank_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        rerank_api_key: Optional[str] = None,
        embed_model: str = "EmbeddingGemma 300m",
        rerank_model: str = "Qwen Rerank 0.6B",
        generate_model: str = "Gemma4 26A4B",
        timeout: float = 120.0
    ):
        # Default to local server if not set. 
        self.base_url = base_url or os.environ.get("QMD_LLM_URL", "http://127.0.0.1:8888")
        self.api_key = api_key or os.environ.get("QMD_LLM_API_KEY", "sk-no-key-required")
        
        self.embed_url = embed_url or os.environ.get("QMD_EMBED_URL") or self.base_url
        self.embed_api_key = embed_api_key or os.environ.get("QMD_EMBED_API_KEY") or self.api_key
        
        self.rerank_url = rerank_url or os.environ.get("QMD_RERANK_URL") or self.base_url
        self.rerank_api_key = rerank_api_key or os.environ.get("QMD_RERANK_API_KEY") or self.api_key

        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.generate_model = generate_model
        self.timeout = timeout
        
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )

        if self.embed_url == self.base_url and self.embed_api_key == self.api_key:
            self.embed_client = self.client
        else:
            self.embed_client = httpx.Client(
                base_url=self.embed_url,
                headers={"Authorization": f"Bearer {self.embed_api_key}"},
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )

        if self.rerank_url == self.base_url and self.rerank_api_key == self.api_key:
            self.rerank_client = self.client
        elif self.rerank_url == self.embed_url and self.rerank_api_key == self.embed_api_key:
            self.rerank_client = self.embed_client
        else:
            self.rerank_client = httpx.Client(
                base_url=self.rerank_url,
                headers={"Authorization": f"Bearer {self.rerank_api_key}"},
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )

    def format_doc_for_embedding(self, title: str, text: str) -> str:
        """Formats document text according to Nomic/Gemma rules."""
        safe_title = title if title else ""
        return f"title: {safe_title} | text: {text}"

    def format_query_for_embedding(self, query: str) -> str:
        """Formats query text according to Nomic/Gemma rules."""
        return f"task: search result | query: {query}"

    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 16, 
        show_progress: bool = False, 
        desc: str = "Embedding chunks"
    ) -> List[List[float]]:
        """
        Calls /v1/embeddings in batches to prevent timeouts on slow hardware or large files.
        """
        if not texts:
            return []

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []

        batch_iter = tqdm(batches, desc=desc, leave=False, unit="batch") if (show_progress and len(batches) > 1) else batches

        for batch in batch_iter:
            try:
                response = self.embed_client.post("/v1/embeddings", json={
                    "input": batch,
                    "model": self.embed_model
                })
                response.raise_for_status()
                data = response.json()
                items = data.get("data", [])
                items.sort(key=lambda x: x["index"])
                all_embeddings.extend([item["embedding"] for item in items])
            except httpx.TimeoutException as e:
                raise RuntimeError(
                    f"Embedding request timed out for a batch of {len(batch)} item(s). "
                    f"Consider increasing 'request_timeout' (current: {self.timeout}s) or decreasing 'embed_batch_size'."
                ) from e

        return all_embeddings

    def rerank(self, query: str, documents: List[str]) -> List[Dict]:
        """
        Calls custom /v1/rerank endpoint.
        Normalizes output to ensure 'score' key exists.
        """
        if not documents:
            return []

        try:
            response = self.rerank_client.post("/v1/rerank", json={
                "query": query,
                "documents": documents,
                "model": self.rerank_model
            })
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Rerank request timed out for {len(documents)} document(s). "
                f"Consider increasing 'request_timeout' (current: {self.timeout}s)."
            ) from e
        
        raw_results = data.get("results", [])
        normalized = []
        
        for item in raw_results:
            # Copy to avoid mutating original if that matters, though here it doesn't.
            new_item = item.copy()
            # If 'relevance_score' exists but 'score' does not, map it.
            if "score" not in new_item and "relevance_score" in new_item:
                new_item["score"] = new_item["relevance_score"]
            normalized.append(new_item)
            
        return normalized

    def expand_query(self, query: str, context: str = "") -> List[str]:
        """
        Generates search variations (lex, vec, hyde).
        """
        prompt = f"""
You are a search query optimization expert. Your task is to improve retrieval by rewriting queries and generating hypothetical documents based on the input provided at the end.

### Step 1: Query Analysis
Identify entities, search intent, and missing context.

### Step 2: Generate Hypothetical Document
Write a focused sentence passage that would answer the query. Include specific terminology and domain vocabulary.

### Step 3: Query Rewrites
Generate 2-3 alternative search queries that resolve ambiguities. Use terminology from the hypothetical document.

### Step 4: Final Retrieval Text
Output exactly 1-3 'lex' lines, 1-3 'vec' lines, and MAX ONE 'hyde' line.

<format>
lex: {{single search term}}
vec: {{single vector query}}
hyde: {{complete hypothetical document passage from Step 2 on a SINGLE LINE}}
</format>

<rules>
- DO NOT repeat the same line.
- Each 'lex:' line MUST be a different keyword variation based on the ORIGINAL QUERY.
- Each 'vec:' line MUST be a different semantic variation based on the ORIGINAL QUERY.
- The 'hyde:' line MUST be the full sentence passage from Step 2, but all on one line.
</rules>

---
Original Query: {query}

Context: {context}

Final Output:
"""
        response = self.client.post("/v1/chat/completions", json={
            "model": self.generate_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512
        })
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # Parse lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        valid_lines = []
        for line in lines:
            if line.startswith(('lex:', 'vec:', 'hyde:')):
                # Normalize internal whitespace: replace multiple spaces with single space
                clean_line = re.sub(r'\s+', ' ', line)
                valid_lines.append(clean_line)
        
        return valid_lines
