"""Unified OpenRouter streaming plus the app's stable SSE response contract."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Generator, Iterator, Optional, Tuple

import requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

import app.core.config as cfg
from app.core.observability import safe_exception
from app.services.llm.citations import (
    coerce_text,
    parse_openrouter_response,
    source_response,
)
from app.services.llm.engines import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    _error,
    _log_model_selection,
    _merge_nested_config,
    _raise_provider_http_status,
    build_provider_payload,
    openrouter_headers,
)
from app.services.llm.provider_runtime import (
    PROVIDER_HTTP_TIMEOUT,
    ProviderCancellation,
    ProviderCancelled,
    bind_provider_cancellation,
    managed_provider_resource,
    raise_if_provider_cancelled,
)

logger = logging.getLogger(__name__)

StreamEvent = Dict[str, Any]

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_pack(event: str, data: Dict[str, Any]) -> str:
    encoded = jsonable_encoder(data)
    return f"event: {event}\ndata: {json.dumps(encoded, ensure_ascii=False)}\n\n"


SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0


class ProviderStreamingResponse(StreamingResponse):
    """StreamingResponse that propagates disconnects to the provider request."""

    def __init__(self, *args, cancellation: ProviderCancellation, **kwargs):
        self.provider_cancellation = cancellation
        super().__init__(*args, **kwargs)

    async def listen_for_disconnect(self, receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                self.provider_cancellation.cancel()
                return

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            self.provider_cancellation.cancel()
            raise
        finally:
            self.provider_cancellation.cancel()


def iter_sse_with_keepalive(
    source,
    interval_seconds: float = SSE_KEEPALIVE_INTERVAL_SECONDS,
    cancellation: ProviderCancellation | None = None,
):
    events: queue.Queue = queue.Queue()
    done = object()
    cancellation = cancellation or ProviderCancellation()

    def pump():
        try:
            with bind_provider_cancellation(cancellation):
                for item in source:
                    cancellation.raise_if_cancelled()
                    events.put(item)
            events.put(done)
        except ProviderCancelled:
            events.put(done)
        except BaseException as exc:  # noqa: BLE001
            events.put(exc)

    threading.Thread(
        target=pump,
        daemon=True,
        name="sse-keepalive-pump",
    ).start()

    try:
        while True:
            try:
                item = events.get(timeout=interval_seconds)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancellation.cancel()


def keepalive_streaming_response(source) -> ProviderStreamingResponse:
    cancellation = ProviderCancellation()
    return ProviderStreamingResponse(
        iter_sse_with_keepalive(source, cancellation=cancellation),
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
        cancellation=cancellation,
    )


def iter_sse_events(response: requests.Response) -> Iterator[Tuple[Optional[str], str]]:
    """Read standard event/data pairs, ignoring OpenRouter heartbeat comments."""
    with managed_provider_resource(response):
        event_name: Optional[str] = None
        data_lines: list[str] = []
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


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "".join(
        coerce_text(item.get("text") or item.get("content"))
        if isinstance(item, dict) else coerce_text(item)
        for item in value
    )


def _iter_openrouter_chunks(*, api_key: str, payload: dict) -> Iterator[StreamEvent]:
    raise_if_provider_cancelled()
    request_payload = dict(payload)
    request_payload["stream"] = True
    response = requests.post(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=openrouter_headers(api_key),
        json=request_payload,
        timeout=PROVIDER_HTTP_TIMEOUT,
        stream=True,
    )
    if response.status_code >= 400:
        with managed_provider_resource(response):
            _raise_provider_http_status(response)

    for _, data_str in iter_sse_events(response):
        raise_if_provider_cancelled()
        if data_str.strip() == "[DONE]":
            return
        data = _parse_json(data_str)
        if not data:
            continue
        if data.get("error"):
            raise RuntimeError("OpenRouter stream returned an error event")
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            text = _content_text(delta.get("content"))
            if text:
                yield {"type": "delta", "text": text}
            if any(delta.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
                yield {"type": "reasoning"}
            annotations = delta.get("annotations") or delta.get("citations")
            if annotations:
                yield {"type": "annotations", "annotations": annotations}
            if choice.get("finish_reason"):
                yield {"type": "finish", "reason": choice["finish_reason"]}


def _stream_openrouter_chat_completion(
    *,
    api_key: str,
    payload: dict,
    provider: str,
) -> Generator[StreamEvent, None, None]:
    text_parts: list[str] = []
    annotations: list = []
    finish_reason = None
    for event in _iter_openrouter_chunks(api_key=api_key, payload=payload):
        event_type = event.get("type")
        if event_type == "delta":
            text_parts.append(event["text"])
            yield event
        elif event_type == "reasoning":
            yield event
        elif event_type == "annotations":
            value = event.get("annotations")
            annotations.extend(value if isinstance(value, list) else [value])
        elif event_type == "finish":
            finish_reason = event.get("reason")

    answer = "".join(text_parts)
    if not answer.strip():
        message = (
            "The model ran out of output tokens while reasoning and never produced an answer. "
            "Please try again or simplify the question."
            if finish_reason == "length"
            else "The model returned no answer. Please try again."
        )
        yield {"type": "final", "result": {
            "text": "",
            "sources": [],
            "error": message,
            "error_code": "empty_reasoning_response",
        }}
        return
    yield {
        "type": "final",
        "result": parse_openrouter_response(answer, annotations, provider),
    }


def stream_model_query(
    provider: str,
    question: str,
    api_key: str,
    system_prompt: str | None = None,
    deep_search: bool = False,
    model_override: str | None = None,
    max_output_tokens: int | None = None,
    attachments: list[dict] | None = None,
) -> Generator[StreamEvent, None, None]:
    label = cfg.provider_label(provider)
    try:
        request_data = build_provider_payload(
            provider,
            question=question,
            system_prompt=system_prompt,
            model_override=model_override,
            deep_search=deep_search,
            max_output_tokens=max_output_tokens,
            attachments=attachments,
        )
        _log_model_selection(label, request_data["api_model"], deep_search, model_override)
        yield from _stream_openrouter_chat_completion(
            api_key=api_key,
            payload=request_data["payload"],
            provider=provider,
        )
    except ProviderCancelled:
        raise
    except Exception as exc:
        raise_if_provider_cancelled()
        yield {"type": "final", "result": _error(label, exc)}


def stream_chat_completion_text(
    *,
    api_key: str,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    request_config: dict | None = None,
) -> Iterator[StreamEvent]:
    """Stream text for Consensus/Differences through the same transport."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "provider": {"zdr": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    _merge_nested_config(payload, request_config)
    for event in _iter_openrouter_chunks(api_key=api_key, payload=payload):
        if event.get("type") in {"delta", "reasoning"}:
            yield event


def streaming_model_response(
    stream_gen: Generator[StreamEvent, None, None],
    provider_label: str,
    extra_fields: Optional[dict] = None,
) -> StreamingResponse:
    """Wrap engine events in the app's existing delta/final SSE contract."""
    extras = dict(extra_fields or {})
    cancellation = ProviderCancellation()

    def event_source():
        last_reasoning_at = None
        try:
            with bind_provider_cancellation(cancellation):
                for item in stream_gen:
                    cancellation.raise_if_cancelled()
                    if item.get("type") == "delta":
                        text = coerce_text(item.get("text"))
                        if text:
                            yield sse_pack("delta", {"text": text})
                    elif item.get("type") == "reasoning":
                        now = time.monotonic()
                        if last_reasoning_at is None or now - last_reasoning_at >= 2.0:
                            last_reasoning_at = now
                            yield sse_pack("reasoning", {"text": ""})
                    elif item.get("type") == "final":
                        yield sse_pack("final", source_response(item.get("result"), **extras))
                        return
        except ProviderCancelled:
            return
        except Exception as exc:
            logger.error(
                "Provider stream failed provider=%s category=%s",
                provider_label,
                safe_exception(exc),
            )
            payload = {
                "error": f"{provider_label} could not complete this request. Please try again later.",
                "error_code": "provider_stream_failed",
                "response": "",
                "sources": [],
            }
            payload.update(extras)
            yield sse_pack("final", payload)
        finally:
            cancellation.cancel()

    # Der Provider-Generator laeuft bewusst im eigenen Pump-Thread von
    # iter_sse_with_keepalive. Das ist keine Kosmetik: Starlette zieht einen
    # synchronen Iterator ueber iterate_in_threadpool, und jedes next() darf auf
    # einem anderen Worker-Thread landen. Die Cancellation haengt aber an einem
    # threading.local -- ohne festen Thread liest ein laufender Stream die
    # Cancellation eines fremden, laengst beendeten Laufs, bricht als
    # ProviderCancelled ab und schickt nie ein final-Event.
    return ProviderStreamingResponse(
        iter_sse_with_keepalive(event_source(), cancellation=cancellation),
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
        cancellation=cancellation,
    )
