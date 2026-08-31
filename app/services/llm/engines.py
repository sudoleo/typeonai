"""Unified OpenRouter transport for all consens.io model families."""

from __future__ import annotations

import logging
from typing import Any

import requests

import app.core.config as cfg
from app.core.observability import safe_exception
from app.services.llm.attachments import (
    IMAGE_MIMES,
    build_attachment_question_suffix,
    native_attachments_for_provider,
)
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import coerce_text, parse_openrouter_response, result_text
from app.services.llm.provider_runtime import PROVIDER_HTTP_TIMEOUT, managed_provider_resource

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_COMPLETIONS_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_REFERER = "https://consens.io"
OPENROUTER_TITLE = "consens.io"

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": cfg.DEFAULT_OPENAI_MODEL,
    "mistral": cfg.DEFAULT_MISTRAL_MODEL,
    "anthropic": cfg.DEFAULT_ANTHROPIC_MODEL,
    "gemini": cfg.GEMINI_FLASH_MODEL,
    "deepseek": cfg.DEFAULT_DEEPSEEK_MODEL,
    "grok": cfg.DEFAULT_GROK_MODEL,
}

_DEEP_SEARCH_MODEL_BY_PROVIDER = {
    "openai": "gpt-5.5",
    "mistral": cfg.MISTRAL_PRO_MODEL,
    "anthropic": cfg.ANTHROPIC_PRO_MODEL,
    "gemini": cfg.GEMINI_PRO_MODEL,
    "deepseek": cfg.DEEPSEEK_PRO_MODEL,
    "grok": "grok-4.3",
}


class _ProviderHTTPStatusError(RuntimeError):
    """Content-free upstream status error for metrics and retry policy."""

    def __init__(self, status_code: int):
        super().__init__("upstream provider returned an HTTP error")
        self.status_code = int(status_code)


def _raise_provider_http_status(response) -> None:
    raise _ProviderHTTPStatusError(int(response.status_code))


def _error(provider: str, error: Exception | str, *, timeout: bool = False):
    category = safe_exception(error) if isinstance(error, BaseException) else "provider_error"
    error_code = (
        "provider_timeout"
        if timeout
        or (isinstance(error, BaseException) and "timeout" in category.lower())
        or category.endswith(":408")
        or category.endswith(":504")
        else "provider_request_failed"
    )
    logger.error("Provider request failed provider=%s category=%s", provider, category)
    return {
        "text": "",
        "sources": [],
        "error": f"{provider} could not complete this request. Please try again later.",
        "error_code": error_code,
    }


def _merge_nested_config(payload: dict, config: dict | None):
    if not config:
        return
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            _merge_nested_config(payload[key], value)
        else:
            payload[key] = value


def _log_model_selection(
    provider: str,
    api_model: str,
    deep_search: bool,
    model_override: str | None,
) -> None:
    logger.info(
        "Provider model selected: %s -> %s | deep_search=%s | override=%s",
        provider,
        api_model,
        deep_search,
        model_override,
    )


def openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }


def _openrouter_user_content(question: str, attachments: list[dict]) -> str | list[dict]:
    if not attachments:
        return question
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for attachment in attachments:
        mime = attachment["mime"]
        data_url = f"data:{mime};base64,{attachment['data']}"
        if mime in IMAGE_MIMES:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content.append({
                "type": "file",
                "file": {
                    "filename": attachment["name"],
                    "file_data": data_url,
                },
            })
    return content


def build_provider_payload(
    provider: str,
    *,
    question: str = "dry run",
    system_prompt: str | None = None,
    model_override: str | None = None,
    deep_search: bool = False,
    max_output_tokens: int | None = None,
    attachments: list[dict] | None = None,
    benchmark_mode: bool = False,
) -> dict:
    """Build the one OpenRouter Chat Completions payload used by every family."""
    provider_key = str(provider or "").lower()
    if provider_key not in _DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(f"Unsupported model family: {provider}")

    system = system_prompt if system_prompt is not None else get_system_prompt()
    if deep_search:
        system += "\n" + cfg.DEEP_THINK_PROMPT

    default_model = _DEFAULT_MODEL_BY_PROVIDER[provider_key]
    internal_model = (
        _DEEP_SEARCH_MODEL_BY_PROVIDER[provider_key]
        if deep_search
        else (model_override or default_model)
    )
    api_model, model_config = cfg.resolve_api_model(
        internal_model,
        default_model,
        provider_key,
    )
    max_tokens = int(max_output_tokens) if max_output_tokens is not None else cfg.get_output_token_limit(True, deep_search)

    provider_attachments = native_attachments_for_provider(attachments or [], provider_key)
    fallback_suffix = build_attachment_question_suffix(attachments or [], provider_key)
    if fallback_suffix:
        question = (question or "") + fallback_suffix

    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system or " "},
            {
                "role": "user",
                "content": _openrouter_user_content(question, provider_attachments),
            },
        ],
        "max_tokens": max_tokens,
        # Privacy policy, not provider pinning: only zero-retention endpoints may run.
        "provider": {"zdr": True},
    }
    if not benchmark_mode:
        max_uses = 5 if deep_search else 2
        payload["tools"] = [{
            "type": "openrouter:web_search",
            "parameters": {"engine": "auto", "max_uses": max_uses},
        }]
        payload["max_tool_calls"] = max_uses + 1

    request_config = dict(model_config.request_config or {})
    if provider_key == "mistral" and internal_model in cfg.MISTRAL_REASONING_MODELS:
        request_config.setdefault("reasoning", {"effort": "high"})
    if deep_search:
        request_config.setdefault("reasoning", {"effort": cfg.REASONING_EFFORT_FOR_DEEP})
    _merge_nested_config(payload, request_config)

    return {
        "provider": provider_key,
        "endpoint": "chat.completions",
        "internal_model": f"deep_search:{internal_model}" if deep_search else internal_model,
        "api_model": api_model,
        "is_low_reasoning": bool(model_config.is_low_reasoning) if model_config else False,
        "payload": payload,
    }


def query_model(
    provider: str,
    question: str,
    api_key: str,
    system_prompt: str | None = None,
    deep_search: bool = False,
    model_override: str | None = None,
    max_output_tokens: int | None = None,
    attachments: list[dict] | None = None,
    benchmark_mode: bool = False,
):
    """Run one non-streaming model request through OpenRouter."""
    provider_key = str(provider or "").lower()
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
            benchmark_mode=benchmark_mode,
        )
        _log_model_selection(label, request_data["api_model"], deep_search, model_override)
        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=openrouter_headers(api_key),
            json=request_data["payload"],
            timeout=PROVIDER_HTTP_TIMEOUT,
        )
        with managed_provider_resource(response):
            if response.status_code >= 400:
                _raise_provider_http_status(response)
            data = response.json()
        message = (((data.get("choices") or [{}])[0].get("message")) or {})
        result = parse_openrouter_response(
            coerce_text(message.get("content")),
            message.get("annotations") or data.get("citations") or [],
            provider_key,
        )
        if not result_text(result):
            return {
                "text": "",
                "sources": [],
                "error": "The model returned no answer. Please try again.",
                "error_code": "empty_response",
            }
        return result
    except Exception as exc:
        return _error(label, exc)
