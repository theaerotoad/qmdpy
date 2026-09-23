import json
import logging
import threading
from typing import Generator, Optional, Dict
import httpx
from flask import Blueprint, Response, request, stream_with_context, jsonify

logger = logging.getLogger(__name__)

quick_answer_bp = Blueprint("quick_answer", __name__)


class ActiveStreamHandle:
    """Manages cancellation and immediate socket termination for an in-flight LLM request."""

    def __init__(self):
        self.cancel_event = threading.Event()
        self.client: Optional[httpx.Client] = None
        self.response: Optional[httpx.Response] = None
        self.lock = threading.Lock()

    def attach(self, client: httpx.Client, response: httpx.Response) -> bool:
        with self.lock:
            if self.cancel_event.is_set():
                try:
                    response.close()
                except Exception:
                    pass
                try:
                    client.close()
                except Exception:
                    pass
                return False
            self.client = client
            self.response = response
            return True

    def abort(self):
        with self.lock:
            self.cancel_event.set()
            if self.response:
                try:
                    self.response.close()
                except Exception:
                    pass
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass


_active_handles: Dict[str, ActiveStreamHandle] = {}
_handles_lock = threading.Lock()


def abort_session_stream(session_id: str) -> bool:
    """Aborts any active LLM generation for the specified session."""
    with _handles_lock:
        handle = _active_handles.pop(session_id, None)
    if handle:
        handle.abort()
        return True
    return False

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
    """Retrieve LLM URL, generation model name, and API key from current config."""
    try:
        from qmd.web import get_config
        import os
        cfg = get_config()
        base_url = getattr(cfg, "llm_url", None) or os.environ.get("QMD_LLM_URL")
        model = getattr(cfg, "generate_model", None) or os.environ.get("GENERATE_MODEL")
        api_key = getattr(cfg, "llm_api_key", None) or os.environ.get("QMD_LLM_API_KEY", "sk-no-key-required")
    except Exception as e:
        logger.warning(f"Could not read config for quick answer LLM: {e}")
        base_url = None
        model = None
        api_key = "sk-no-key-required"
    if base_url:
        base_url = base_url.rstrip("/")
    return base_url, model, api_key


def generate_quick_answer_stream(
    query: str, xml_context: str, session_id: str, handle: ActiveStreamHandle
) -> Generator[str, None, None]:
    """Streams LLM tokens for quick answer as Server-Sent Events (SSE)."""
    base_url, model, api_key = _get_llm_endpoint_and_model()
    
    if not base_url or not model:
        yield f"data: {json.dumps({'error': 'LLM URL or generation model not configured.'})}\n\n"
        return

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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        if handle.cancel_event.is_set():
            return

        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                if not handle.attach(client, response):
                    return

                if response.status_code != 200:
                    # Try fallback to /chat/completions without /v1
                    fallback_endpoint = f"{base_url}/chat/completions"
                    if endpoint != fallback_endpoint and response.status_code == 404:
                        with client.stream(
                            "POST", fallback_endpoint, json=payload, headers=headers
                        ) as fb_response:
                            if not handle.attach(client, fb_response):
                                return
                            if fb_response.status_code != 200:
                                err_msg = f"LLM error: HTTP {fb_response.status_code}"
                                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                                return
                            yield from _read_sse_stream(fb_response, handle)
                            return

                    err_msg = f"LLM error: HTTP {response.status_code}"
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return

                yield from _read_sse_stream(response, handle)
    except (httpx.StreamClosed, httpx.RequestError, GeneratorExit):
        logger.debug("Quick answer stream terminated/aborted.")
        return
    except Exception as exc:
        if handle.cancel_event.is_set():
            return
        logger.error(f"Quick answer streaming exception: {exc}")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        handle.abort()
        with _handles_lock:
            if _active_handles.get(session_id) is handle:
                _active_handles.pop(session_id, None)


def _read_sse_stream(
    response: httpx.Response, handle: ActiveStreamHandle
) -> Generator[str, None, None]:
    """Parses raw SSE lines from an OpenAI-compatible /chat/completions streaming response."""
    try:
        for line in response.iter_lines():
            if handle.cancel_event.is_set():
                break
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
    except (httpx.StreamClosed, httpx.RequestError):
        pass


@quick_answer_bp.route("/api/quick_answer/abort", methods=["POST"])
def quick_answer_abort_endpoint():
    """Immediately terminates the upstream LLM connection for a session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or "default"
    aborted = abort_session_stream(session_id)
    return jsonify({"status": "ok", "aborted": aborted})


@quick_answer_bp.route("/api/quick_answer", methods=["POST"])
def quick_answer_endpoint():
    """HTTP endpoint receiving {query: str, xml: str, session_id: str} and returning an SSE stream."""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    xml_context = (data.get("xml") or "").strip()
    session_id = data.get("session_id") or "default"

    if not query or not xml_context:
        return Response(
            json.dumps({"error": "Both 'query' and 'xml' are required."}),
            status=400,
            mimetype="application/json",
        )

    # Immediately abort any previous generator / upstream LLM request for this session
    abort_session_stream(session_id)

    handle = ActiveStreamHandle()
    with _handles_lock:
        _active_handles[session_id] = handle

    return Response(
        stream_with_context(generate_quick_answer_stream(query, xml_context, session_id, handle)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
