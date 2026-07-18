"""Server-selected, asynchronous Consensus API orchestration.

This module deliberately orchestrates the existing provider functions and the
existing Consensus/Differences engine. It does not contain an alternative
synthesis implementation.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import app.core.config as cfg
from app.core.security import db_firestore
from app.services.api_run_repository import FirestoreApiRunRepository
from app.services.llm.base import get_system_prompt, validate_model
from app.services.llm.citations import result_sources, result_text, to_plain
from app.services.llm.consensus_engine import (
    compute_agreement_score,
    is_consensus_error_text,
    query_consensus,
    query_differences,
)
from app.services.llm.credentials import (
    missing_credentials,
    resolve_developer_api_keys,
)
from app.services.llm.engines import (
    query_claude,
    query_deepseek,
    query_gemini,
    query_grok,
    query_mistral,
    query_openai,
)
from app.services.llm.mock_llm import mock_ask_result, mock_llm_enabled
from app.services.usage_repository import (
    FirestoreUsageRepository,
    RunKind,
    RunStatus,
    UsageLimits,
    UsageRunConflict,
)


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
PROVIDER_ALLOWED_ATTR = {
    "openai": "ALLOWED_OPENAI_MODELS",
    "mistral": "ALLOWED_MISTRAL_MODELS",
    "anthropic": "ALLOWED_ANTHROPIC_MODELS",
    "gemini": "ALLOWED_GEMINI_MODELS",
    "deepseek": "ALLOWED_DEEPSEEK_MODELS",
    "grok": "ALLOWED_GROK_MODELS",
}


api_run_repository = FirestoreApiRunRepository(db_firestore)
usage_repository = FirestoreUsageRepository(db_firestore)
_background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="consensus-api")


def build_server_model_plan(*, deep_think: bool, is_pro: bool) -> dict:
    preset = dict(cfg.CONSENSUS_PRESET_MODELS[cfg.DEFAULT_CONSENSUS_PRESET])
    providers = {provider: preset[provider] for provider in PROVIDER_ORDER}
    consensus_model = cfg.DEEP_THINK_CONSENSUS_MODEL if deep_think else preset["consensus"]
    is_early = bool(is_pro)
    for provider, model in providers.items():
        validate_model(
            model,
            getattr(cfg, PROVIDER_ALLOWED_ATTR[provider]),
            PROVIDER_LABELS[provider],
            is_pro=is_pro,
            is_early=is_early,
        )
    if consensus_model not in cfg.ALLOWED_CONSENSUS_MODELS:
        raise ValueError("Configured consensus model is not allowed")
    if cfg.is_premium_consensus_model(consensus_model) and not is_pro:
        raise PermissionError("Configured consensus model requires Pro")
    return {
        "preset": cfg.DEFAULT_CONSENSUS_PRESET,
        "providers": providers,
        "consensus_model": consensus_model,
        "deep_think": bool(deep_think),
    }


def validate_server_credentials(model_plan: dict) -> None:
    if mock_llm_enabled():
        return
    required = [PROVIDER_LABELS[name] for name in model_plan["providers"]]
    consensus_provider = _consensus_provider_label(model_plan["consensus_model"])
    if consensus_provider and consensus_provider not in required:
        required.append(consensus_provider)
    keys = resolve_developer_api_keys()
    missing = missing_credentials(keys, required)
    if missing:
        raise RuntimeError("Missing server credentials for: " + ", ".join(missing))


def usage_limits_for_run(run: dict) -> UsageLimits:
    is_pro = bool(run.get("is_pro_at_acceptance"))
    return UsageLimits(
        total=cfg.get_consensus_run_limit(is_pro),
        deep_think=cfg.get_deep_think_run_limit(is_pro),
    )


def usage_key_for_run(run: dict) -> str:
    return "consensus-api:" + str(run.get("idempotency_hash") or "")


def reserve_run(run: dict):
    deep_think = bool((run.get("request") or {}).get("deep_think"))
    result = usage_repository.reserve(
        str(run["uid"]),
        usage_key_for_run(run),
        RunKind.DEEP_THINK if deep_think else RunKind.REGULAR,
        usage_limits_for_run(run),
    )
    if result.status is RunStatus.RELEASED:
        raise UsageRunConflict("Idempotency-Key belongs to a released usage run")
    reserved, _ = api_run_repository.mark_reserved(str(run["run_id"]))
    return reserved, result


def release_run_reservation(run: dict) -> None:
    usage_repository.release(str(run["uid"]), usage_key_for_run(run))


def schedule_run(run_id: str) -> None:
    _background_executor.submit(execute_persisted_run, run_id)


def recover_persisted_runs() -> int:
    """Requeue only pre-provider runs; running work is never replayed."""
    recovered = 0
    try:
        for run in api_run_repository.list_by_status(("reserved",)):
            schedule_run(run["run_id"])
            recovered += 1
        for run in api_run_repository.list_by_status(("accepted",)):
            try:
                reserved, _usage = reserve_run(run)
            except Exception:
                logging.exception(
                    "Consensus API accepted run could not be recovered: %s",
                    run.get("run_id"),
                )
                continue
            if reserved.get("status") == "reserved":
                schedule_run(reserved["run_id"])
                recovered += 1
        for run in api_run_repository.list_by_status(("running",)):
            api_run_repository.fail_if_lease_expired(run["run_id"])
    except Exception:
        logging.exception("Consensus API recovery scan failed")
    return recovered


def execute_persisted_run(run_id: str) -> None:
    worker_id = uuid.uuid4().hex
    usage_consumed = False
    run = None
    try:
        run, claimed = api_run_repository.claim_running(run_id, worker_id)
        if not claimed:
            return
        # Usage is consumed once, immediately before any provider can start.
        # The usage transaction is separate from the API-run claim transaction.
        usage_repository.consume(run["uid"], usage_key_for_run(run))
        usage_consumed = True
        result = execute_consensus_pipeline(run)
        api_run_repository.succeed(run_id, result)
    except Exception:
        logging.exception("Consensus API run failed: %s", run_id)
        if run is not None and not usage_consumed:
            try:
                release_run_reservation(run)
            except Exception:
                logging.exception("Consensus API reservation release failed: %s", run_id)
        try:
            current = api_run_repository.get(run_id)
            if current.get("status") == "running":
                api_run_repository.fail(
                    run_id,
                    code="run_failed",
                    message="The Consensus run could not be completed.",
                )
        except Exception:
            logging.exception("Consensus API failure state could not be persisted: %s", run_id)


def execute_consensus_pipeline(run: dict) -> dict:
    request = run.get("request") or {}
    question = str(request.get("question") or "").strip()
    deep_think = bool(request.get("deep_think"))
    is_pro = bool(run.get("is_pro_at_acceptance"))
    plan = run.get("model_plan") or {}
    providers = plan.get("providers") or {}
    keys = resolve_developer_api_keys()
    if mock_llm_enabled():
        keys = {label: "mock" for label in PROVIDER_LABELS.values()}

    answers: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(providers))) as pool:
        futures = {
            provider: pool.submit(
                _provider_answer,
                provider,
                model,
                question,
                keys,
                is_pro,
                deep_think,
            )
            for provider, model in providers.items()
        }
        for provider in PROVIDER_ORDER:
            if provider not in futures:
                continue
            try:
                raw = futures[provider].result()
            except Exception:
                logging.exception("Consensus API provider failed: %s", provider)
                continue
            text = result_text(raw).strip()[: cfg.get_consensus_answer_char_limit()]
            if not text or text.lower().startswith("error") or (
                isinstance(raw, dict) and raw.get("error")
            ):
                continue
            answers[provider] = {
                "provider": PROVIDER_LABELS[provider],
                "model": providers[provider],
                "response": text,
                "sources": to_plain(result_sources(raw)),
            }

    if len(answers) < 2:
        raise RuntimeError("Fewer than two provider answers completed")

    slots = {
        provider: (answers.get(provider) or {}).get("response")
        for provider in PROVIDER_ORDER
    }
    excluded = [
        PROVIDER_LABELS[provider]
        for provider in PROVIDER_ORDER
        if provider not in answers
    ]
    model_sources = {
        PROVIDER_LABELS[provider]: answer["sources"] for provider, answer in answers.items()
    }
    consensus_model = str(plan.get("consensus_model") or "")
    consensus = query_consensus(
        question,
        slots["openai"],
        slots["mistral"],
        slots["anthropic"],
        slots["gemini"],
        slots["deepseek"],
        slots["grok"],
        excluded,
        consensus_model,
        keys,
        model_sources=model_sources,
    )
    if is_consensus_error_text(consensus):
        raise RuntimeError("Consensus synthesis failed")
    differences_text, differences_data = query_differences(
        slots["openai"],
        slots["mistral"],
        slots["anthropic"],
        slots["gemini"],
        slots["deepseek"],
        slots["grok"],
        consensus,
        keys,
        differences_model=consensus_model,
        excluded_models=excluded,
    )
    if not isinstance(differences_data, dict):
        raise RuntimeError("Differences analysis failed")
    differences_data["agreement"] = compute_agreement_score(differences_data)
    return to_plain(
        {
            "consensus_response": consensus,
            "differences": differences_text,
            "differences_data": differences_data,
            "model_answers": [answers[p] for p in PROVIDER_ORDER if p in answers],
        }
    )


def _provider_answer(
    provider: str,
    model: str,
    question: str,
    keys: dict,
    is_pro: bool,
    deep_think: bool,
):
    label = PROVIDER_LABELS[provider]
    if mock_llm_enabled():
        return mock_ask_result(label, question)
    kwargs = {
        "system_prompt": get_system_prompt(),
        "deep_search": deep_think,
        "model_override": model,
        "max_output_tokens": cfg.get_output_token_limit(is_pro, deep_think),
        "attachments": [],
    }
    key = keys.get(label) or ""
    if provider == "gemini":
        return PROVIDER_FUNCTIONS[provider](question, user_api_key=key, **kwargs)
    return PROVIDER_FUNCTIONS[provider](question, key, **kwargs)


def _consensus_provider_label(consensus_model: str) -> str | None:
    from app.services.llm.consensus_engine import resolve_consensus_engine_model

    resolved = resolve_consensus_engine_model(consensus_model)
    if not resolved:
        return None
    return PROVIDER_LABELS.get(resolved.provider)
