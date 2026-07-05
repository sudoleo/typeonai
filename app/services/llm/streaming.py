from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Generator, Iterator, Optional, Tuple
from urllib.parse import quote

import openai
import requests
from fastapi.responses import StreamingResponse

import app.core.config as cfg
from app.services.llm.citations import (
    make_llm_result,
    parse_anthropic_response,
    parse_gemini_response,
    parse_mistral_content,
    parse_openai_response,
    result_text,
    source_response,
)
from app.services.llm.engines import (
    _error,
    _gemini_search_tool_unsupported,
    _google_adc_headers,
    _log_model_selection,
    _mistral_builtin_tools_unsupported,
    _mistral_headers,
    build_provider_payload,
    query_mistral,
)

logger = logging.getLogger(__name__)

StreamEvent = Dict[str, Any]

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_pack(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def iter_sse_events(response: requests.Response) -> Iterator[Tuple[Optional[str], str]]:
    """Liest Server-Sent Events (event/data-Paare) aus einer requests-Streaming-Response."""
    event_name: Optional[str] = None
    data_lines: list = []
    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else (raw_line or "")
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _parse_json(data_str: str) -> Optional[dict]:
    try:
        parsed = json.loads(data_str)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Provider-Primitive: liefern {"type": "delta", "text": ...} Events und am Ende
# {"type": "final", "result": LLMResult} mit der nachbearbeiteten Gesamtantwort.
# ---------------------------------------------------------------------------

def _stream_openai_responses(*, api_key: str, base_url: str, payload: dict, provider: str) -> Generator[StreamEvent, None, None]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    resp = requests.post(
        f"{base_url.rstrip('/')}/responses",
        json=request_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120,
        stream=True,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")

    final_response = None
    for event_name, data_str in iter_sse_events(resp):
        data = _parse_json(data_str)
        if data is None:
            continue
        event_type = data.get("type") or event_name or ""
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if delta:
                yield {"type": "delta", "text": delta}
        elif event_type in {"response.completed", "response.incomplete"}:
            final_response = data.get("response") or {}
        elif event_type == "response.failed":
            error_info = (data.get("response") or {}).get("error") or {}
            raise RuntimeError(str(error_info.get("message") or "response.failed"))
        elif event_type == "error":
            raise RuntimeError(str(data.get("message") or data))

    if final_response is None:
        raise RuntimeError("Stream ended before a completed response was received.")
    yield {"type": "final", "result": parse_openai_response(final_response, provider=provider)}


def _stream_anthropic_messages(*, api_key: str, payload: dict) -> Generator[StreamEvent, None, None]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json=request_payload,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        timeout=120,
        stream=True,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")

    blocks: Dict[int, dict] = {}
    order: list = []
    for event_name, data_str in iter_sse_events(resp):
        data = _parse_json(data_str)
        if data is None:
            continue
        event_type = data.get("type") or event_name or ""
        if event_type == "content_block_start":
            index = data.get("index")
            content_block = data.get("content_block") or {}
            blocks[index] = {
                "type": content_block.get("type"),
                "text": content_block.get("text") or "",
                "citations": list(content_block.get("citations") or []),
            }
            order.append(index)
            if blocks[index]["type"] == "text" and blocks[index]["text"]:
                yield {"type": "delta", "text": blocks[index]["text"]}
        elif event_type == "content_block_delta":
            index = data.get("index")
            if index not in blocks:
                blocks[index] = {"type": "text", "text": "", "citations": []}
                order.append(index)
            block = blocks[index]
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text") or ""
                if text and block["type"] == "text":
                    block["text"] += text
                    yield {"type": "delta", "text": text}
            elif delta.get("type") == "citations_delta":
                citation = delta.get("citation")
                if citation:
                    block["citations"].append(citation)
        elif event_type == "error":
            error_info = data.get("error") or {}
            raise RuntimeError(str(error_info.get("message") or data))

    content = [
        {"type": "text", "text": blocks[i]["text"], "citations": blocks[i]["citations"]}
        for i in order
        if blocks[i]["type"] == "text"
    ]
    result = parse_anthropic_response({"content": content})
    if not result_text(result):
        result = make_llm_result("Error: No response found in the API response.", [])
    yield {"type": "final", "result": result}


def _stream_gemini_generate(*, model_name: str, payload: dict, api_key: Optional[str]) -> Generator[StreamEvent, None, None]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model_name, safe='')}:streamGenerateContent"

    def _request(request_payload: dict) -> requests.Response:
        request_kwargs: Dict[str, Any] = {
            "json": request_payload,
            "timeout": 120,
            "stream": True,
            "params": {"alt": "sse"},
        }
        if api_key:
            request_kwargs["params"]["key"] = api_key
        else:
            request_kwargs["headers"] = _google_adc_headers()
        return requests.post(url, **request_kwargs)

    request_payload = dict(payload)
    resp = _request(request_payload)
    if resp.status_code >= 400 and _gemini_search_tool_unsupported(resp.text):
        request_payload.pop("tools", None)
        resp = _request(request_payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")

    text_parts: list = []
    grounding_metadata = None
    finish_reason = None
    for _, data_str in iter_sse_events(resp):
        data = _parse_json(data_str)
        if data is None:
            continue
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        if candidate.get("finishReason"):
            finish_reason = candidate["finishReason"]
        metadata = candidate.get("groundingMetadata") or candidate.get("grounding_metadata")
        if metadata:
            grounding_metadata = metadata
        for part in ((candidate.get("content") or {}).get("parts")) or []:
            if part.get("thought"):
                continue
            text = part.get("text")
            if text:
                text_parts.append(text)
                yield {"type": "delta", "text": text}

    merged = {
        "candidates": [{
            "content": {"parts": [{"text": "".join(text_parts)}]},
            "groundingMetadata": grounding_metadata or {},
            "finishReason": finish_reason,
        }]
    }
    parsed = parse_gemini_response(merged)
    if result_text(parsed):
        yield {"type": "final", "result": parsed}
        return

    frs = str(finish_reason)
    if frs in ("2", "MAX_TOKENS", "FinishReason.MAX_TOKENS"):
        message = "Error with Gemini: hit max tokens before producing text. Raise max_output_tokens or trim input."
    elif frs in ("3", "SAFETY", "FinishReason.SAFETY"):
        message = "Error with Gemini: response was blocked by safety filters."
    elif frs in ("4", "RECITATION", "FinishReason.RECITATION"):
        message = "Error with Gemini: response suppressed by recitation policy."
    else:
        message = f"Error with Gemini: empty response payload (finish_reason={frs})."
    yield {"type": "final", "result": make_llm_result(message, [])}


def _extract_mistral_delta_items(content: Any) -> list:
    items: list = []
    if isinstance(content, str):
        if content:
            items.append({"type": "text", "text": content})
    elif isinstance(content, dict):
        chunk_type = content.get("type")
        if chunk_type == "text":
            if content.get("text"):
                items.append({"type": "text", "text": content["text"]})
        elif chunk_type in {"tool_reference", "reference"}:
            items.append(content)
    elif isinstance(content, list):
        for chunk in content:
            items.extend(_extract_mistral_delta_items(chunk))
    return items


def _stream_mistral_conversations(*, api_key: str, payload: dict) -> Generator[StreamEvent, None, None]:
    request_payload = dict(payload)
    request_payload["stream"] = True

    def _request() -> requests.Response:
        return requests.post(
            "https://api.mistral.ai/v1/conversations",
            json=request_payload,
            headers=_mistral_headers(api_key),
            timeout=120,
            stream=True,
        )

    resp = _request()
    if _mistral_builtin_tools_unsupported(resp):
        request_payload.pop("tools", None)
        resp = _request()
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")

    content_items: list = []
    for event_name, data_str in iter_sse_events(resp):
        data = _parse_json(data_str)
        if data is None:
            continue
        event_type = data.get("type") or event_name or ""
        if event_type == "message.output.delta":
            for item in _extract_mistral_delta_items(data.get("content")):
                content_items.append(item)
                if item.get("type") == "text" and item.get("text"):
                    yield {"type": "delta", "text": item["text"]}
        elif event_type in {"conversation.response.error", "error"}:
            raise RuntimeError(str(data.get("message") or data))

    yield {"type": "final", "result": parse_mistral_content(content_items)}


def _stream_chat_completions(*, client: openai.OpenAI, payload: dict) -> Generator[StreamEvent, None, None]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    stream = client.chat.completions.create(**request_payload)
    parts: list = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta else None
        if text:
            parts.append(text)
            yield {"type": "delta", "text": text}
    yield {"type": "final", "result": make_llm_result("".join(parts).strip(), [])}


# ---------------------------------------------------------------------------
# Engine-Wrapper: Gegenstücke zu query_openai/query_mistral/... aus engines.py.
# Fehler werden – wie in den nicht-streamenden Varianten – als Error-Result
# zurückgegeben statt als Exception nach außen zu schlagen.
# ---------------------------------------------------------------------------

def stream_openai_query(
    question: str,
    api_key: str,
    deep_search: bool = False,
    system_prompt: str = None,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "openai",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("OpenAI", request_data["api_model"], deep_search, model_override)
        yield from _stream_openai_responses(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            payload=request_data["payload"],
            provider="openai",
        )
    except Exception as e:
        yield {"type": "final", "result": _error("OpenAI", e)}


def stream_grok_query(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "grok",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("Grok", request_data["api_model"], deep_search, model_override)
        yield from _stream_openai_responses(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            payload=request_data["payload"],
            provider="grok",
        )
    except Exception as e:
        yield {"type": "final", "result": _error("Grok", str(e))}


def stream_claude_query(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "anthropic",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("Claude", request_data["api_model"], deep_search, model_override)
        yield from _stream_anthropic_messages(api_key=api_key, payload=request_data["payload"])
    except Exception as e:
        yield {"type": "final", "result": _error("Anthropic", str(e))}


def stream_gemini_query(
    question: str,
    user_api_key: Optional[str] = None,
    deep_search: bool = False,
    system_prompt: str = None,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "gemini",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("Gemini", request_data["api_model"], deep_search, model_override)
        api_key = (user_api_key or os.environ.get("DEVELOPER_GEMINI_API_KEY") or "").strip() or None
        yield from _stream_gemini_generate(
            model_name=request_data["api_model"],
            payload=request_data["payload"],
            api_key=api_key,
        )
    except Exception as e:
        yield {"type": "final", "result": _error("Gemini", e)}


def stream_deepseek_query(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "deepseek",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("DeepSeek", request_data["api_model"], deep_search, model_override)
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        yield from _stream_chat_completions(client=client, payload=request_data["payload"])
    except Exception as e:
        yield {"type": "final", "result": _error("DeepSeek", e)}


def stream_mistral_query(
    question: str,
    api_key: str,
    system_prompt: str = None,
    deep_search: bool = False,
    model_override: str = None,
    max_output_tokens: Optional[int] = None,
    attachments: Optional[list] = None,
) -> Generator[StreamEvent, None, None]:
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)
    try:
        request_data = build_provider_payload(
            "mistral",
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )
        _log_model_selection("Mistral", request_data["api_model"], deep_search, model_override)

        yielded_text = False
        final_result = None
        for item in _stream_mistral_conversations(api_key=api_key, payload=request_data["payload"]):
            if item.get("type") == "delta":
                yielded_text = True
                yield item
            else:
                final_result = item.get("result")

        if final_result is None or (not yielded_text and not result_text(final_result)):
            # Sicherheitsnetz: falls das Streaming-Event-Format keine Inhalte
            # geliefert hat, antwortet der reguläre (nicht-streamende) Pfad.
            final_result = query_mistral(
                question, api_key, system_prompt, deep_search=deep_search,
                model_override=model_override, max_output_tokens=max_tokens,
                attachments=attachments,
            )
        yield {"type": "final", "result": final_result}
    except Exception as e:
        yield {"type": "final", "result": _error("Mistral", str(e))}


# ---------------------------------------------------------------------------
# Reine Text-Streams für die Consensus-/Differences-Engines.
# ---------------------------------------------------------------------------

def stream_chat_completion_text(
    *,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int,
    base_url: Optional[str] = None,
    token_param: str = "max_tokens",
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
) -> Iterator[str]:
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    kwargs = {"model": model, "messages": messages, "stream": True, token_param: max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    for chunk in client.chat.completions.create(**kwargs):
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta else None
        if text:
            yield text


def stream_mistral_chat_text(
    *,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
) -> Iterator[str]:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    if model in cfg.MISTRAL_REASONING_MODELS:
        payload["reasoning_effort"] = "high"
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        json=payload,
        headers=_mistral_headers(api_key),
        timeout=120,
        stream=True,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} - {resp.text}")
    for _, data_str in iter_sse_events(resp):
        if data_str.strip() == "[DONE]":
            break
        data = _parse_json(data_str)
        if not data:
            continue
        delta = ((data.get("choices") or [{}])[0].get("delta") or {})
        content = delta.get("content")
        # content kann ein String oder eine Liste von Content-Chunks sein
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, str):
                    if chunk:
                        yield chunk
                elif isinstance(chunk, dict) and chunk.get("type") == "text" and chunk.get("text"):
                    yield chunk["text"]


def stream_anthropic_text(
    *,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    assistant_prefill: Optional[str] = None,
) -> Iterator[str]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if assistant_prefill:
        # Prefill erzwingt den Antwortanfang (z. B. "{" für JSON-Ausgaben);
        # der Prefill gehört zum Ergebnistext und wird daher mit ausgegeben.
        payload["messages"].append({"role": "assistant", "content": assistant_prefill})
        yield assistant_prefill
    for item in _stream_anthropic_messages(api_key=api_key, payload=payload):
        if item.get("type") == "delta":
            yield item["text"]


def stream_gemini_payload_text(*, api_model: str, payload: dict, api_key: Optional[str]) -> Iterator[str]:
    """Streamt einen fertig gebauten Gemini-Payload (Consensus-/Differences-Engine).

    Liefert nur Text-Deltas; ein leerer Stream mit Fehler-Result (Safety-Block,
    max_tokens ohne Text) schlägt als RuntimeError nach außen, damit die
    Retry-Logik der Aufrufer greift."""
    key = (api_key or "").strip() or None
    got_delta = False
    final_result = None
    for item in _stream_gemini_generate(model_name=api_model, payload=payload, api_key=key):
        if item.get("type") == "delta":
            got_delta = True
            yield item["text"]
        else:
            final_result = item.get("result")
    if not got_delta:
        message = result_text(final_result) if final_result else ""
        raise RuntimeError(message or "Gemini: empty response payload.")


# ---------------------------------------------------------------------------
# FastAPI-Helfer
# ---------------------------------------------------------------------------

def streaming_model_response(stream_gen: Generator[StreamEvent, None, None], provider_label: str, extra_fields: Optional[dict] = None) -> StreamingResponse:
    """Verpackt einen Engine-Stream als SSE-Response mit delta/final-Events.

    Das final-Event hat dieselbe Struktur wie die bisherige JSON-Antwort
    (response, sources, free_usage_remaining, ...), damit das Frontend die
    bestehende Auswertung weiterverwenden kann.
    """
    extras = dict(extra_fields or {})

    def event_source():
        try:
            for item in stream_gen:
                if item.get("type") == "delta":
                    yield sse_pack("delta", {"text": item.get("text") or ""})
                elif item.get("type") == "final":
                    yield sse_pack("final", source_response(item.get("result"), **extras))
                    return
        except Exception as exc:
            logger.exception("Streaming failed for %s", provider_label)
            payload = {
                "error": f"{provider_label} could not complete this request. Please try again later.",
                "error_code": "provider_stream_failed",
                "response": "",
                "sources": [],
            }
            payload.update(extras)
            yield sse_pack("final", payload)

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=dict(SSE_HEADERS))
