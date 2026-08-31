"""Benchmark transport backed by one OpenRouter Chat Completions request.

The benchmark deliberately does not duplicate the production provider
adapters. ``build_provider_payload`` already resolves a benchmark model to its
OpenRouter ``api_model`` and creates the provider-neutral Chat Completions
payload. This module only performs that request and normalises the one
OpenRouter response shape for the benchmark runner.

``execute`` remains the public runner contract and accepts an injectable
``http_post`` so tests never need live network access. ``call_provider`` is a
small compatibility alias for callers that used that name in earlier
benchmark integrations.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

import requests

from app.services.llm.citations import (
    coerce_text,
    parse_openrouter_response,
    result_sources,
    result_text,
)
from app.services.llm.credentials import openrouter_api_key
from app.services.llm.engines import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    openrouter_headers,
)
from app.services.llm.provider_runtime import managed_provider_resource


class TransportResponse(TypedDict):
    """Normalised result written by the benchmark runner."""

    text: str
    sources: list
    usage: dict[str, int]
    raw: dict[str, Any] | None
    status: int
    latency_ms: float
    error: str | None
    error_code: str | None


DEFAULT_TIMEOUT_SECONDS = 600


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def extract_usage(
    raw_or_provider: dict[str, Any] | str,
    legacy_raw: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Return OpenRouter Chat Completions usage as prompt/completion/total."""

    # ``extract_usage(provider, raw)`` was an old public helper signature.
    # Provider is intentionally ignored now: OpenRouter has one usage schema.
    raw = legacy_raw if legacy_raw is not None else raw_or_provider
    if not isinstance(raw, dict):
        raw = {}
    usage = raw.get("usage") or {}
    prompt = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _as_int(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    total = _as_int(usage.get("total_tokens")) or prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}


def parse_text_and_sources(
    raw_or_provider: dict[str, Any] | str,
    legacy_raw: dict[str, Any] | None = None,
    provider: str = "openrouter",
) -> tuple[str, list]:
    """Extract text and URL citations from one OpenRouter response.

    ``provider`` is retained only as the source label in the benchmark output;
    it never selects a transport or parser.
    """

    # Keep the former ``(provider, raw)`` call shape source-compatible while
    # removing its provider-specific parser dispatch.
    if isinstance(raw_or_provider, str) and isinstance(legacy_raw, dict):
        provider = raw_or_provider
        raw = legacy_raw
    else:
        raw = raw_or_provider if isinstance(raw_or_provider, dict) else {}

    message = (((raw.get("choices") or [{}])[0].get("message")) or {})
    parsed = parse_openrouter_response(
        coerce_text(message.get("content")),
        message.get("annotations") or raw.get("citations") or [],
        provider=provider,
    )
    return result_text(parsed), result_sources(parsed)


def _error_result(
    message: str,
    code: str,
    latency_ms: float,
    status: int = 0,
) -> TransportResponse:
    return {
        "text": "",
        "sources": [],
        "usage": {"prompt": 0, "completion": 0, "total": 0},
        "raw": None,
        "status": status,
        "latency_ms": latency_ms,
        "error": message,
        "error_code": code,
    }


def _key_value(api_key: str | dict | None) -> str:
    """Accept the runner's legacy string and the shared credential mapping."""

    if isinstance(api_key, dict):
        return openrouter_api_key(api_key) or ""
    return str(api_key or "").strip()


def execute(
    request_data: dict,
    api_key: str | dict | None,
    *,
    http_post: Callable | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> TransportResponse:
    """Call OpenRouter Chat Completions and return the benchmark result.

    ``request_data`` is normally produced by
    ``app.services.llm.engines.build_provider_payload``. Its ``provider`` value
    is metadata only; all model families use the exact same endpoint, headers,
    payload format and response parser.
    """

    payload = dict(request_data.get("payload") or {})
    provider = str(request_data.get("provider") or "openrouter")
    # The payload builder is authoritative, but keep hand-built requests safe:
    # the resolved API model is the ModelConfig.api_model carried by request_data.
    api_model = request_data.get("api_model")
    if api_model:
        payload["model"] = api_model
    payload.setdefault("provider", {"zdr": True})

    post = http_post or requests.post
    started = time.perf_counter()
    try:
        response = post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            json=payload,
            headers=openrouter_headers(_key_value(api_key)),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - transport failures are expected
        latency_ms = (time.perf_counter() - started) * 1000
        return _error_result(str(exc), "transport_request_failed", latency_ms)

    latency_ms = (time.perf_counter() - started) * 1000
    with managed_provider_resource(response):
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            return _error_result(
                f"OpenRouter HTTP {status}", "provider_http_error", latency_ms, status
            )

        try:
            raw = response.json()
            text, sources = parse_text_and_sources(raw, provider=provider)
            usage = extract_usage(raw)
        except Exception as exc:  # noqa: BLE001 - malformed upstream responses
            return _error_result(str(exc), "response_parse_failed", latency_ms, status)

    return {
        "text": text,
        "sources": sources,
        "usage": usage,
        "raw": raw,
        "status": status,
        "latency_ms": latency_ms,
        "error": None,
        "error_code": None,
    }


def call_provider(
    request_data: dict,
    api_key: str | dict | None,
    *,
    http_post: Callable | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> TransportResponse:
    """Compatibility entry point for the benchmark's unified transport."""

    return execute(request_data, api_key, http_post=http_post, timeout=timeout)
