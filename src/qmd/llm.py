import os
import re
import json
import httpx
from typing import List, Dict, Optional
from tqdm import tqdm

ORIGINAL_MULTIMODAL_PROMPT = """You are an expert Document Analysis and Optical Character Recognition (OCR) engine.
Your task is to transcribe and convert the provided image into clean, structured Markdown text.

CRITICAL INSTRUCTIONS & STRICT RULES:
1. NO CONVERSATIONAL PREAMBLE OR CHATTER: Output ONLY the transcribed content. Do NOT include phrases like "Here is the text", "This image shows", "Sure", or "In summary".
2. DO NOT WRAP THE ENTIRE RESPONSE IN A TRIPLE-BACKTICK MARKDOWN CODEBLOCK. Output raw markdown directly.
3. COMPLETE TEXT EXTRACTION (OCR): Transcribe ALL visible text, labels, headers, annotations, captions, footnotes, numbers, and bullet points verbatim and accurately. Do not skip or summarize text.
4. STRUCTURED DATA & TABLES: If the image contains a table, chart data, or matrix, convert it into a standard Markdown table using pipes (| col1 | col2 |) with proper separator lines (|---|---|).
5. DIAGRAMS, FLOWCHARTS & CHARTS: Transcribe all node labels, flow steps, axis labels, legend entries, and data points into structured Markdown (using headings, numbered steps, or bullet points).
6. MATHEMATICAL FORMULAS: Transcribe mathematical equations and symbols into LaTeX notation ($...$ for inline, $$...$$ for standalone display equations).
7. VISUAL ONLY / PHOTOS: If the image is a photograph, diagram, or illustration with little to no text, provide a concise, factual description in 1-3 sentences formatted as: ![Description of image](image.png).
8. PRESERVE HIERARCHY: Use appropriate Markdown headers (#, ##, ###), bullet lists (-), or numbered lists (1.) to preserve the visual reading order and hierarchy of the document."""

DEFAULT_MULTIMODAL_PROMPT = """You are a document OCR and visual-analysis engine. Convert the attached image into faithful, structured, inline-safe Markdown.

Return only the result:

- No introduction, conclusion, explanation, apology, or code fence.
- Do not use Markdown headings. No line may begin with `#`.
- Do not use thematic breaks such as `---`.
- Do not use pipes merely to separate text. Use pipes only inside a valid Markdown table.
- Use bold only for structural labels or source text that is visibly a heading.
- Do not bold ordinary values, dates, axis labels, tick labels, or legend entries.

Complete both tasks:

1. Transcribe all readable text.
2. Describe all meaningful non-text visual content.
Do not omit task 2 merely because text was successfully transcribed.

**Text transcription rules**
- Copy all readable text exactly as displayed.
- Include titles, headings, labels, values, dates, units, annotations, captions, legends, footnotes, and footer text.
- Preserve capitalization, spelling, punctuation, numeric precision, identifiers, and spaces around punctuation.
- Do not correct, normalize, rename, paraphrase, reorder, or deduplicate text.
- Use `[illegible]` for unreadable text instead of guessing.
- Use `[partially illegible: readable portion]` when only part is readable.

**Reading order and structure**
- Follow the natural visual reading order.
- Process separate columns, panels, figures, or regions independently.
- Preserve the relationship between labels, legends, captions, and the visual content they describe.
- Render source headings as bold text on their own lines rather than Markdown headings.
- Use lists only when the source or visual grouping supports them.
- Do not create sections for content that is absent.

**Tables**

- Use a Markdown table only for an actual table with explicit rows, columns, and cells.
- Charts, plots, timelines, diagrams, alignment, whitespace, and graphical grids are not tables.
- Do not create table cells from axis ticks or graphical positions.
- If an actual table cannot be represented reliably, use labeled rows or a structured list.

**Required visual descriptions**

- Identify every meaningful visual region, including charts, plots, timelines, diagrams, flowcharts, photographs, illustrations, maps, and interface elements.
- After transcribing the text associated with a visual region, add:

  `**Visual description:**`

- Under that label, provide one or more concise bullet points describing what is visibly represented.
- A transcription of labels, axes, or legends is not a substitute for the visual description.
- Do not describe purely decorative borders, backgrounds, or spacing.
- Do not infer purpose, intent, causality, or facts that are not visually established.

**Charts, plots, and timelines**

When a chart, plot, or timeline is present, always include:

- Its displayed title, if any.
- Axis titles and units.
- Readable tick labels.
- Category, row, or lane labels.
- Annotations.
- Legend entries in displayed order.
- A required visual description of the marks, series, and overall visible patterns.

For the visual description:

- Identify the visual type when clear, such as timeline, line chart, bar chart, scatter plot, or stacked-area chart.
- Describe each clearly distinguishable series, lane, or category when practical.
- Carefully trace rows and lanes from their labels so marks are not assigned to adjacent rows.
- Describe patterns objectively using terms such as `isolated marks`, `sparse marks`, `repeated clusters`, `dense marks`, `nearly continuous marks`, `gradual increase`, `sharp decrease`, `plateau`, `gap`, and `narrow spike`.
- Describe major relative patterns even when exact values cannot be recovered.
- Do not use subjective terms such as `significant`, `important`, or `dramatic`.
- Do not invent exact values, timestamps, durations, or relationships from graphical positions.
- If a graphical value can be estimated reliably from an axis, say `approximately`.
- Do not associate a change or event with a date unless it is annotated or clearly aligned with a labeled tick.
- Do not enumerate colors unless color is needed to connect marks with a clearly readable legend.
- If individual marks are too numerous or small to enumerate, describe their density and distribution.
- If the graphical content cannot be interpreted reliably, write:
  `- Graphical marks are present, but their values or associations cannot be determined reliably from the image.`

Use this structure for each chart when applicable:

**[Exact displayed chart title or concise chart label]**

**Axis and category text:**

- [exact transcribed text]

**Legend:**

- [exact legend entry]

**Visual description:**

- [chart type and overall organization]
- [objective pattern by series, lane, or category]
- [major increases, decreases, clusters, gaps, or spikes]
- [uncertainty statement when needed]

**Diagrams and flowcharts**

- Transcribe all readable node, box, connector, and annotation text.
- Represent clearly visible directed connections as `[A] → [B]`.
- Do not invent connections or direction.
- Under `**Visual description:**`, describe the visible organization and only the relationships that are clear.

**Forms and interfaces**

- Preserve field labels and displayed values.
- Represent clearly checked boxes as `[x]` and unchecked boxes as `[ ]`.
- Include visible buttons, menus, tabs, alerts, and status messages.
- Under `**Visual description:**`, briefly describe the layout and visible state when that information is meaningful.
- Do not infer hidden, truncated, disabled, or unselected content.

**Photographs and illustrations**

- Transcribe any visible text first.
- Under `**Visual description:**`, provide one to three concise, factual sentences.
- Do not speculate about identities, locations, purposes, or events not visually established.
- If the entire input is primarily a photograph or illustration, use:

  `![Factual visual description](image.png)`

**Mathematics**

- Use `$...$` for inline expressions and `$$...$$` for standalone equations.
- Preserve equation numbers, symbols, variables, and surrounding labels.
- Do not solve or simplify equations unless the image does so.

Before responding, verify:

1. No line begins with `#`.
2. No code fence or thematic break was added.
3. Pipes appear only in actual Markdown tables.
4. All readable text was transcribed.
5. Every meaningful non-text visual region has a `**Visual description:**`.
6. A chart’s labels and legend were kept with that chart.
7. No chart, plot, or graphical grid was treated as a table.
8. No value, timestamp, connection, or relationship was invented.
9. The response contains only the structured result."""

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
        multimodal_url: Optional[str] = None,
        multimodal_api_key: Optional[str] = None,
        multimodal_model: Optional[str] = None,
        multimodal_prompt: Optional[str] = None,
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
        self.multimodal_url = multimodal_url or os.environ.get("QMD_MULTIMODAL_URL") or self.base_url
        self.multimodal_api_key = multimodal_api_key or os.environ.get("QMD_MULTIMODAL_API_KEY") or self.api_key
        self.multimodal_model = multimodal_model or os.environ.get("QMD_MULTIMODAL_MODEL") or os.environ.get("MULTIMODAL_MODEL") or self.generate_model
        self.multimodal_prompt = multimodal_prompt or os.environ.get("QMD_MULTIMODAL_PROMPT") or DEFAULT_MULTIMODAL_PROMPT
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

        if self.multimodal_url == self.base_url and self.multimodal_api_key == self.api_key:
            self.multimodal_client = self.client
        elif self.multimodal_url == self.embed_url and self.multimodal_api_key == self.embed_api_key:
            self.multimodal_client = self.embed_client
        elif self.multimodal_url == self.rerank_url and self.multimodal_api_key == self.rerank_api_key:
            self.multimodal_client = self.rerank_client
        else:
            self.multimodal_client = httpx.Client(
                base_url=self.multimodal_url,
                headers={"Authorization": f"Bearer {self.multimodal_api_key}"},
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

    def process_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> str:
        """
        Transcribes an image into structured Markdown text using an OpenAI-compatible multimodal endpoint.
        """
        if not image_bytes:
            return ""

        import base64

        if not mime_type:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".png":
                mime_type = "image/png"
            elif ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".webp":
                mime_type = "image/webp"
            elif ext == ".gif":
                mime_type = "image/gif"
            else:
                if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    mime_type = "image/png"
                elif image_bytes.startswith(b"\xff\xd8\xff"):
                    mime_type = "image/jpeg"
                elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
                    mime_type = "image/webp"
                elif image_bytes.startswith(b"GIF8"):
                    mime_type = "image/gif"
                else:
                    mime_type = "image/jpeg"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        system_instruction = prompt or self.multimodal_prompt or DEFAULT_MULTIMODAL_PROMPT

        user_content = [
            {
                "type": "text",
                "text": "Extract and transcribe all text, tables, diagrams, and content from this image into structured Markdown format according to the rules."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}"
                }
            }
        ]

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.multimodal_client.post("/v1/chat/completions", json={
                "model": self.multimodal_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096
            })
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            raw_content = choices[0].get("message", {}).get("content", "")
            return self._clean_multimodal_output(raw_content)
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Multimodal LLM request timed out for {filename}. "
                f"Consider increasing 'request_timeout' (current: {self.timeout}s)."
            ) from e

    def _clean_multimodal_output(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        fence_match = re.match(r'^```(?:markdown)?\s*\n([\s\S]*?)\n```$', text, flags=re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        return text
