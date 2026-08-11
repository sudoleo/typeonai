"""Provider transport registry used by every server-side Consensus caller.

This module knows how to call providers and normalize transport results.  It
does not decide which models to use, synthesize a Consensus, score agreement or
persist a product-domain result.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

import app.core.config as cfg
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import result_sources, result_text, to_plain
from app.services.llm.credentials import enable_gemini_adc, resolve_developer_api_keys
from app.services.llm.engines import (
    query_claude,
    query_deepseek,
    query_gemini,
    query_grok,
    query_mistral,
    query_openai,
)
from app.services.llm.mock_llm import mock_ask_result, mock_llm_enabled


PROVIDER_ORDER = ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok")
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
}
PROVIDER_FUNCTIONS = {
    "openai": query_openai,
    "mistral": query_mistral,
    "anthropic": query_claude,
    "gemini": query_gemini,
    "deepseek": query_deepseek,
    "grok": query_grok,
}


@dataclass(frozen=True)
class ProviderAnswer:
    provider: str
    model: str
    response: str
    sources: list


def developer_keys() -> dict:
    """Return the one canonical server credential map (including Gemini ADC)."""
    return enable_gemini_adc(resolve_developer_api_keys())


def provider_available(provider: str, keys: dict) -> bool:
    if mock_llm_enabled():
        return True
    label = PROVIDER_LABELS[provider]
    if keys.get(label):
        return True
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    return provider == "gemini" and bool(adc_path and os.path.isfile(adc_path))


def query_provider(
    provider: str,
    model: str,
    question: str,
    keys: dict,
    is_pro: bool,
    deep_think: bool = False,
):
    """Call one provider using the shared server-side transport contract."""
    label = PROVIDER_LABELS[provider]
    if mock_llm_enabled():
        return mock_ask_result(label, question)
    kwargs = {
        "system_prompt": get_system_prompt(),
        "deep_search": bool(deep_think),
        "model_override": model,
        "max_output_tokens": cfg.get_output_token_limit(is_pro, bool(deep_think)),
        "attachments": [],
    }
    key = keys.get(label) or ""
    if provider == "gemini":
        return PROVIDER_FUNCTIONS[provider](question, user_api_key=key, **kwargs)
    return PROVIDER_FUNCTIONS[provider](question, key, **kwargs)


def fan_out_provider_answers(
    *,
    question: str,
    provider_models: dict[str, str],
    keys: dict,
    is_pro: bool,
    deep_think: bool,
    provider_order: Iterable[str] = PROVIDER_ORDER,
    provider_call: Callable = query_provider,
    answer_char_limit: int | None = None,
    log_context: str = "Consensus",
) -> dict[str, ProviderAnswer]:
    """Run provider transports in parallel and collect results deterministically."""
    ordered = [name for name in provider_order if name in provider_models]
    answers: dict[str, ProviderAnswer] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(ordered))) as pool:
        futures = {
            provider: pool.submit(
                provider_call,
                provider,
                provider_models[provider],
                question,
                keys,
                is_pro,
                deep_think,
            )
            for provider in ordered
        }
        for provider in ordered:
            try:
                raw = futures[provider].result()
            except Exception:
                logging.warning("%s provider failed: %s", log_context, provider)
                continue
            text = result_text(raw).strip()
            if answer_char_limit:
                text = text[:answer_char_limit]
            if not text or text.lower().startswith("error") or (
                isinstance(raw, dict) and raw.get("error")
            ):
                continue
            answers[provider] = ProviderAnswer(
                provider=PROVIDER_LABELS[provider],
                model=provider_models[provider],
                response=text,
                sources=to_plain(result_sources(raw)),
            )
    return answers
