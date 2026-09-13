import json
import logging
from typing import Generator, Optional
import httpx
from flask import Blueprint, Response, request, stream_with_context

logger = logging.getLogger(__name__)

quick_answer_bp = Blueprint("quick_answer", __name__)

QUICK_ANSWER_SYSTEM_PROMPT = """You are a concise, factual document assistant. Your job is to answer the user's question directly using ONLY the provided search result XML context.

RULES:
1. Relevance Check: First assess whether the provided search result XML contains information relevant to the user's query.
2. If the results are NOT relevant or do not contain enough facts to answer, respond ONLY with the exact words:
NOT RELEVANT
3. If the results ARE relevant:
   - Provide a direct, clear answer in natural prose in LESS THAN 100 WORDS.
   - Do NOT guess or use outside knowledge.
   - Cite sources inline using numeric citations like [1] or [2] referring to the 'rank' attribute of the chunk/result in the search XML.
"""


def _get_llm_endpoint_and_model():
    """Retrieve LLM URL and generation model name from current config."""
    try:
        from qmd.web import get_config
        cfg = get_config()
        base_url = getattr(cfg, "llm_url", None) or "http://127.0.0.1:9888"
        model = getattr(cfg, "generate_model", None) or "Gemma4 26A4B"
    except Exception as e:
        logger.warning(f"Could not read config for quick answer LLM: {e}")
        base_url = "http://127.0.0.1:9888"
        model = "Gemma4 26A4B"
    return base_url.rstrip("/"), model


def generate_quick_answer_stream(query: str, xml_context: str) -> Generator[str, None, None]:
    """Streams LLM tokens for quick answer as Server-Sent Events (SSE)."""
    base_url, model = _get_llm_endpoint_and_model()

    # Determine completions endpoint
    if base_url.endswith("/v1"):
        endpoint = f"{base_url}/chat/completions"
    else:
        endpoint = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": QUICK_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User Query: {query}\n\nSearch Results XML:\n{xml_context}",
            },
        ],
        "max_tokens": 8192,
        "stream": True,
        # Explicitly disable reasoning as requested for faster inference
        "enable_thinking": False,
        "thinking_budget_tokens": 2048,
        "extra_body": {
            "enable_thinking": False,
        },
    }

    headers = {"Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    # Try fallback to /chat/completions without /v1
                    fallback_endpoint = f"{base_url}/chat/completions"
                    if endpoint != fallback_endpoint and response.status_code == 404:
                        with client.stream(
                            "POST", fallback_endpoint, json=payload, headers=headers
                        ) as fb_response:
                            if fb_response.status_code != 200:
                                err_msg = f"LLM error: HTTP {fb_response.status_code}"
                                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                                return
                            yield from _read_sse_stream(fb_response)
                            return

                    err_msg = f"LLM error: HTTP {response.status_code}"
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return

                yield from _read_sse_stream(response)
    except Exception as exc:
        logger.error(f"Quick answer streaming exception: {exc}")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


def _read_sse_stream(response: httpx.Response) -> Generator[str, None, None]:
    """Parses raw SSE lines from an OpenAI-compatible /chat/completions streaming response."""
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.strip()
        if not line_str.startswith("data:"):
            continue

        data_payload = line_str[len("data:") :].strip()
        if data_payload == "[DONE]":
            yield "data: [DONE]\n\n"
            break

        try:
            parsed = json.loads(data_payload)
            choices = parsed.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield f"data: {json.dumps({'delta': content})}\n\n"
        except json.JSONDecodeError:
            continue


@quick_answer_bp.route("/api/quick_answer", methods=["POST"])
def quick_answer_endpoint():
    """HTTP endpoint receiving {query: str, xml: str} and returning an SSE stream."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    xml_context = (data.get("xml") or "").strip()

    if not query or not xml_context:
        return Response(
            json.dumps({"error": "Both 'query' and 'xml' are required."}),
            status=400,
            mimetype="application/json",
        )

    return Response(
        stream_with_context(generate_quick_answer_stream(query, xml_context)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
