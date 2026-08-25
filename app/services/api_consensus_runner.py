"""Server-selected, asynchronous Consensus API orchestration.

This module deliberately orchestrates the existing provider functions and the
existing Consensus/Differences engine. It does not contain an alternative
synthesis implementation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import app.core.config as cfg
from app.core.background_tasks import task_succeeded
from app.core.observability import safe_exception
from app.core.security import db_firestore
from app.services.api_account_cleanup import (
    ApiAccountInactive,
    ApiAccountStatusUnavailable,
    FirestoreApiAccountCleanup,
)
from app.services.api_run_repository import FirestoreApiRunRepository
from app.services.consensus_pipeline import run_consensus_pipeline
from app.services.llm.base import validate_model
from app.services.llm.consensus_engine import (
    query_consensus,
    query_differences,
)
from app.services.llm.credentials import (
    enable_gemini_adc,
    missing_credentials,
    resolve_developer_api_keys,
)
from app.services.llm.mock_llm import mock_llm_enabled
from app.services.llm import provider_transport
from app.services.llm.provider_transport import PROVIDER_LABELS, PROVIDER_ORDER
from app.services.usage_repository import (
    FirestoreUsageRepository,
    RunKind,
    RunStatus,
    UsageLimits,
    UsageRunConflict,
    UsageRunNotFound,
    UsageTransitionError,
    canonical_request_fingerprint,
)


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
api_account_cleanup = FirestoreApiAccountCleanup(db_firestore)
API_RUN_WORKERS = 2
MAX_SCHEDULED_RUNS = 32
API_MAINTENANCE_INTERVAL_SECONDS = 60
_background_executor = ThreadPoolExecutor(
    max_workers=API_RUN_WORKERS, thread_name_prefix="consensus-api"
)
_schedule_lock = threading.Lock()
_scheduled_run_ids: set[str] = set()
_retention_backfilled = False


def build_server_model_plan(*, deep_think: bool, is_pro: bool) -> dict:
    preset = dict(cfg.CONSENSUS_PRESET_MODELS[cfg.DEFAULT_CONSENSUS_PRESET])
    providers = {provider: preset[provider] for provider in PROVIDER_ORDER}
    if len(providers) < 2:
        raise ValueError("At least two API providers are required")
    consensus_model = cfg.DEEP_THINK_CONSENSUS_MODEL if deep_think else preset["consensus"]
    for provider, model in providers.items():
        validate_model(
            model,
            getattr(cfg, PROVIDER_ALLOWED_ATTR[provider]),
            PROVIDER_LABELS[provider],
            is_pro=is_pro,
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
    keys = enable_gemini_adc(resolve_developer_api_keys())
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
        request_fingerprint=canonical_request_fingerprint(
            {
                "schema": 1,
                "request": run.get("request") or {},
                "model_plan": run.get("model_plan") or {},
            }
        ),
    )
    if result.status is RunStatus.RELEASED:
        raise UsageRunConflict("Idempotency-Key belongs to a released usage run")
    reserved, _ = api_run_repository.mark_reserved(str(run["run_id"]))
    return reserved, result


def release_run_reservation(run: dict) -> None:
    usage_repository.release(str(run["uid"]), usage_key_for_run(run))


def schedule_run(run_id: str) -> bool:
    """Deduplicate work and bound the shared active-plus-pending queue."""
    run_id = str(run_id or "").strip()
    with _schedule_lock:
        if run_id in _scheduled_run_ids:
            return True
        if len(_scheduled_run_ids) >= MAX_SCHEDULED_RUNS:
            return False
        _scheduled_run_ids.add(run_id)
    try:
        _background_executor.submit(_execute_scheduled_run, run_id)
    except Exception:
        with _schedule_lock:
            _scheduled_run_ids.discard(run_id)
        raise
    return True


def _execute_scheduled_run(run_id: str) -> None:
    try:
        execute_persisted_run(run_id)
    finally:
        with _schedule_lock:
            _scheduled_run_ids.discard(run_id)


def fail_expired_run(run_id: str) -> bool:
    """Fail an expired worker and release only a pre-provider reservation."""
    try:
        run = api_run_repository.get(run_id)
        changed = api_run_repository.fail_if_lease_expired(run_id)
    except Exception as exc:
        logging.error(
            "Consensus API lease check failed category=%s", safe_exception(exc)
        )
        return False
    if not changed:
        return False
    try:
        release_run_reservation(run)
    except (UsageRunNotFound, UsageTransitionError):
        # Missing is harmless. Consumed/released are terminal: consumed proves
        # a provider was allowed to start and must remain charged.
        pass
    except Exception as exc:
        logging.error(
            "Consensus API stale reservation reconciliation failed category=%s",
            safe_exception(exc),
        )
    return True


def cleanup_expired_runs() -> int:
    """Hard-delete API content after the fixed 30-day retention window."""
    deleted = 0
    try:
        expired_runs = api_run_repository.list_expired()
    except Exception as exc:
        logging.error(
            "Consensus API retention scan failed category=%s", safe_exception(exc)
        )
        return 0
    for run in expired_runs:
        run_id = str(run.get("run_id") or "")
        try:
            if run.get("status") in {"accepted", "reserved"}:
                try:
                    release_run_reservation(run)
                except (UsageRunNotFound, UsageTransitionError):
                    pass
            elif run.get("status") == "running":
                fail_expired_run(run_id)
            if api_run_repository.delete_expired(run_id):
                deleted += 1
        except Exception as exc:
            logging.error(
                "Consensus API expired run cleanup failed category=%s",
                safe_exception(exc),
            )
    return deleted


def recover_persisted_runs() -> int:
    """Requeue only pre-provider runs; running work is never replayed."""
    global _retention_backfilled
    # A MOCK_LLM instance would answer recovered production runs with fixtures,
    # so it never picks up durable work left behind by the live deployment.
    if mock_llm_enabled():
        return 0
    recovered = 0
    try:
        if not _retention_backfilled:
            backfill = getattr(api_run_repository, "backfill_retention", None)
            if backfill is not None:
                backfill()
            _retention_backfilled = True
        cleanup_expired_runs()
        for run in api_run_repository.list_by_status(("reserved",)):
            if schedule_run(run["run_id"]):
                recovered += 1
        for run in api_run_repository.list_by_status(("accepted",)):
            try:
                reserved, _usage = reserve_run(run)
            except Exception as exc:
                logging.error(
                    "Consensus API accepted run could not be recovered category=%s",
                    safe_exception(exc),
                )
                continue
            if reserved.get("status") == "reserved":
                if schedule_run(reserved["run_id"]):
                    recovered += 1
        for run in api_run_repository.list_by_status(("running",)):
            fail_expired_run(run["run_id"])
    except Exception as exc:
        logging.error(
            "Consensus API recovery scan failed category=%s", safe_exception(exc)
        )
        raise
    return recovered


async def api_run_maintenance_loop() -> None:
    """Periodically retry durable work and close expired worker leases."""
    while True:
        recovered = await asyncio.to_thread(recover_persisted_runs)
        task_succeeded("consensus-api-maintenance", recovered_runs=recovered)
        await asyncio.sleep(API_MAINTENANCE_INTERVAL_SECONDS)


def execute_persisted_run(run_id: str) -> None:
    worker_id = uuid.uuid4().hex
    usage_consumed = False
    run = None
    try:
        run, claimed = api_run_repository.claim_running(run_id, worker_id)
        if not claimed:
            return
        try:
            api_account_cleanup.ensure_active(str(run["uid"]))
        except ApiAccountInactive:
            release_run_reservation(run)
            api_run_repository.fail(
                run_id,
                code="account_inactive",
                message="The account is no longer allowed to use the Consensus API.",
            )
            return
        except ApiAccountStatusUnavailable:
            release_run_reservation(run)
            api_run_repository.fail(
                run_id,
                code="account_status_unavailable",
                message="The account status could not be verified before provider start.",
            )
            return
        # Usage is consumed once, immediately before any provider can start.
        # The usage transaction is separate from the API-run claim transaction.
        usage_repository.consume(run["uid"], usage_key_for_run(run))
        usage_consumed = True
        result = execute_consensus_pipeline(run)
        api_run_repository.succeed(run_id, result)
    except Exception as exc:
        logging.error("Consensus API run failed category=%s", safe_exception(exc))
        if run is not None and not usage_consumed:
            try:
                release_run_reservation(run)
            except Exception as release_exc:
                logging.error(
                    "Consensus API reservation release failed category=%s",
                    safe_exception(release_exc),
                )
        try:
            current = api_run_repository.get(run_id)
            if current.get("status") == "running":
                api_run_repository.fail(
                    run_id,
                    code="run_failed",
                    message="The Consensus run could not be completed.",
                )
        except Exception as persist_exc:
            logging.error(
                "Consensus API failure state could not be persisted category=%s",
                safe_exception(persist_exc),
            )


def execute_consensus_pipeline(run: dict) -> dict:
    request = run.get("request") or {}
    question = str(request.get("question") or "").strip()
    deep_think = bool(request.get("deep_think"))
    is_pro = bool(run.get("is_pro_at_acceptance"))
    plan = run.get("model_plan") or {}
    providers = dict(plan.get("providers") or {})
    # Publisher runs use the same six providers as every other API run. The
    # persisted model_plan is the single source of truth for who answers, so a
    # run never drops a provider that its own plan still advertises.
    keys = enable_gemini_adc(resolve_developer_api_keys())
    if mock_llm_enabled():
        keys = {label: "mock" for label in PROVIDER_LABELS.values()}

    consensus_model = str(plan.get("consensus_model") or "")
    result = run_consensus_pipeline(
        question=question,
        provider_models=providers,
        consensus_model=consensus_model,
        keys=keys,
        is_pro=is_pro,
        deep_think=deep_think,
        provider_order=PROVIDER_ORDER,
        provider_call=_provider_answer,
        synthesize=query_consensus,
        judge=query_differences,
        answer_char_limit=cfg.get_consensus_answer_char_limit(),
        log_context="Consensus API",
    )
    result.pop("agreement", None)
    return result


def _provider_answer(
    provider: str,
    model: str,
    question: str,
    keys: dict,
    is_pro: bool,
    deep_think: bool,
):
    return provider_transport.query_provider(
        provider, model, question, keys, is_pro, deep_think
    )


def _consensus_provider_label(consensus_model: str) -> str | None:
    from app.services.llm.consensus_engine import resolve_consensus_engine_model

    resolved = resolve_consensus_engine_model(consensus_model)
    if not resolved:
        return None
    return PROVIDER_LABELS.get(resolved.provider)
