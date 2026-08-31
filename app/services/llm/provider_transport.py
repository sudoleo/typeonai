"""Provider transport registry used by every server-side Consensus caller.

This module knows how to call providers and normalize transport results.  It
does not decide which models to use, synthesize a Consensus, score agreement or
persist a product-domain result.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

import app.core.config as cfg
from app.core.observability import record_metric, safe_exception
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import result_sources, result_text, to_plain
from app.services.llm.credentials import openrouter_api_key, resolve_developer_api_keys
from app.services.llm.engines import query_model
from app.services.llm.mock_llm import mock_ask_result, mock_llm_enabled


# Familien und ihre Labels kommen aus der Provider-Registry; die Reihenfolge
# ist die Registry-Reihenfolge (Anzeige- und Prompt-Reihenfolge).
PROVIDER_ORDER = tuple(cfg.PROVIDERS)
PROVIDER_LABELS = dict(cfg.PROVIDER_LABEL_BY_ID)
@dataclass(frozen=True)
class ProviderAnswer:
    provider: str
    model: str
    response: str
    sources: list


def developer_keys() -> dict:
    """Return the one canonical OpenRouter server credential."""
    return resolve_developer_api_keys()


def provider_available(provider: str, keys: dict) -> bool:
    return mock_llm_enabled() or bool(openrouter_api_key(keys))


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
    return query_model(
        provider,
        question,
        openrouter_api_key(keys) or "",
        **kwargs,
    )


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

    def invoke(provider: str):
        started = time.monotonic()
        try:
            raw = provider_call(
                provider,
                provider_models[provider],
                question,
                keys,
                is_pro,
                deep_think,
            )
            return raw, (time.monotonic() - started) * 1000, None
        except Exception as exc:
            return None, (time.monotonic() - started) * 1000, exc

    def metric(provider: str, duration_ms: float, outcome: str) -> None:
        record_metric(
            "provider",
            f"{provider}:{log_context}",
            duration_ms=duration_ms,
            outcome=outcome,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(ordered))) as pool:
        futures = {
            provider: pool.submit(invoke, provider)
            for provider in ordered
        }
        for provider in ordered:
            raw, duration_ms, error = futures[provider].result()
            if error is not None:
                category = safe_exception(error)
                outcome = "timeout" if "timeout" in category.lower() else "failure"
                metric(provider, duration_ms, outcome)
                logging.warning(
                    "%s provider failed provider=%s category=%s",
                    log_context,
                    provider,
                    category,
                )
                continue
            text = result_text(raw).strip()
            if answer_char_limit:
                text = text[:answer_char_limit]
            if not text or text.lower().startswith("error") or (
                isinstance(raw, dict) and raw.get("error")
            ):
                outcome = (
                    "timeout"
                    if isinstance(raw, dict)
                    and raw.get("error_code") == "provider_timeout"
                    else "failure"
                )
                metric(provider, duration_ms, outcome)
                continue
            metric(provider, duration_ms, "success")
            answers[provider] = ProviderAnswer(
                provider=PROVIDER_LABELS[provider],
                model=provider_models[provider],
                response=text,
                sources=to_plain(result_sources(raw)),
            )
    return answers
