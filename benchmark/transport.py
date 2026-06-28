"""Schlanke Transport-Schicht (E1).

Wiederverwendet:
- ``build_provider_payload(..., benchmark_mode=True)`` fuer Payload + Modellaufloesung,
- die Response-Parser aus ``app.services.llm.citations`` fuer Text + Quellen.

Uebernimmt selbst: HTTP-Call **und** Usage-Erfassung (aus demselben Roh-JSON, das
auch der Parser bekommt). Die produktiven ``query_*``-Funktionen bleiben
unangetastet.

``execute()`` macht keinen Call beim Import; der HTTP-POST ist via ``http_post``
injizierbar (Tests mocken ihn).
"""

from __future__ import annotations

import time

import requests

from app.services.llm.citations import (
    parse_anthropic_response,
    parse_gemini_response,
    parse_mistral_content,
    parse_openai_response,
    result_sources,
    result_text,
)

# Provider -> (HTTP-URL-Builder, Header-Builder). Gemini nutzt zusaetzlich
# Query-Params fuer den Key.
_OPENAI_URL = "https://api.openai.com/v1/responses"
_GROK_URL = "https://api.x.ai/v1/responses"
_MISTRAL_URL = "https://api.mistral.ai/v1/conversations"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent"


def _bearer(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _url(provider: str, request_data: dict) -> str:
    if provider == "openai":
        return _OPENAI_URL
    if provider == "grok":
        return _GROK_URL
    if provider == "mistral":
        return _MISTRAL_URL
    if provider == "anthropic":
        return _ANTHROPIC_URL
    if provider == "deepseek":
        return _DEEPSEEK_URL
    if provider == "gemini":
        return _GEMINI_URL.format(api_model=request_data["api_model"])
    raise ValueError(f"Unsupported provider for transport: {provider}")


def _gemini_adc_headers() -> dict:
    """ADC-Bearer-Header fuer Gemini (Live-Pfad). In Tests gemockt – wird nur
    aufgerufen, wenn fuer Gemini **kein** API-Key vorliegt."""
    from app.services.llm.credentials import gemini_adc_headers

    return gemini_adc_headers()


def _auth_for(provider: str, api_key: str | None) -> tuple[dict, dict | None]:
    """Liefert (headers, params) je Provider. Gemini nutzt den API-Key in den
    Query-Params; ohne Key faellt es – wie die Produktion – auf ADC-Header zurueck."""
    if provider == "anthropic":
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }, None
    if provider == "gemini":
        if api_key:
            return {"Content-Type": "application/json"}, {"key": api_key}
        return _gemini_adc_headers(), None
    return _bearer(api_key), None


def parse_text_and_sources(provider: str, raw: dict) -> tuple[str, list]:
    """Text + Quellen aus dem Roh-JSON – dieselben Parser wie die Produktion."""
    if provider in ("openai", "grok"):
        parsed = parse_openai_response(raw, provider=provider)
    elif provider == "anthropic":
        parsed = parse_anthropic_response(raw)
    elif provider == "gemini":
        parsed = parse_gemini_response(raw)
    elif provider == "mistral":
        parsed = parse_mistral_content(_mistral_content(raw))
    elif provider == "deepseek":
        text = (
            (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        ).strip()
        return text, []
    else:
        raise ValueError(f"Unsupported provider for parsing: {provider}")
    return result_text(parsed), result_sources(parsed)


def _mistral_content(raw: dict):
    """Repliziert die Content-Extraktion aus query_mistral (engines.py)."""
    content = []
    for output in raw.get("outputs", []) or []:
        if output.get("type") == "message.output":
            output_content = output.get("content", "")
            if isinstance(output_content, list):
                content.extend(output_content)
            elif output_content:
                content.append({"type": "text", "text": str(output_content)})
    if not content:
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def extract_usage(provider: str, raw: dict) -> dict:
    """Normalisiert Usage zu ``{prompt, completion, total}`` aus dem Roh-JSON.

    Die produktiven Parser verwerfen Usage (Plan §2) – hier wird sie selbst gezogen.
    """
    prompt = completion = total = 0

    if provider == "gemini":
        meta = raw.get("usageMetadata") or raw.get("usage_metadata") or {}
        prompt = _as_int(meta.get("promptTokenCount") or meta.get("prompt_token_count"))
        completion = _as_int(
            meta.get("candidatesTokenCount") or meta.get("candidates_token_count")
        )
        total = _as_int(meta.get("totalTokenCount") or meta.get("total_token_count"))
    else:
        usage = raw.get("usage") or {}
        # OpenAI/Grok Responses-API: input_tokens / output_tokens / total_tokens.
        # Anthropic: input_tokens / output_tokens (kein total).
        # DeepSeek/Mistral Chat: prompt_tokens / completion_tokens / total_tokens.
        prompt = _as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        completion = _as_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total = _as_int(usage.get("total_tokens"))

    if not total:
        total = prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}


def _as_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


# Grosszuegiges Read-Timeout: Reasoning-Modelle generieren bei hohen Output-Limits
# (24576/32768) teils mehrere Minuten. Bei 120s lief Mistral im 10er-Preview (q271)
# in einen Read-Timeout. Der Benchmark ist ein Batch-Job (nicht latenzkritisch) und
# kann fehlgeschlagene Zellen ohnehin per Resume erneut versuchen.
DEFAULT_TIMEOUT_SECONDS = 600


def execute(
    request_data: dict,
    api_key: str,
    *,
    http_post=None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Fuehrt einen einzelnen Provider-Call aus und liefert ein einheitliches
    Ergebnis-Dict: ``{text, sources, usage, raw, status, latency_ms, error,
    error_code}``."""
    provider = request_data["provider"]
    payload = request_data["payload"]
    post = http_post or requests.post

    url = _url(provider, request_data)
    headers, params = _auth_for(provider, api_key)

    started = time.perf_counter()
    try:
        response = post(url, json=payload, headers=headers, params=params, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 – Transportfehler sind erwartbar
        latency_ms = (time.perf_counter() - started) * 1000
        return _error_result(str(exc), "transport_request_failed", latency_ms)

    latency_ms = (time.perf_counter() - started) * 1000
    status = getattr(response, "status_code", 0)
    if status >= 400:
        body = getattr(response, "text", "")
        return _error_result(f"{status} - {body}", "provider_http_error", latency_ms, status)

    raw = response.json()
    text, sources = parse_text_and_sources(provider, raw)
    usage = extract_usage(provider, raw)
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


def _error_result(message: str, code: str, latency_ms: float, status: int = 0) -> dict:
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
