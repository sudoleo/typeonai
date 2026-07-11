"""Sequential, Firestore-leased background runner for Consensus Watch."""

from __future__ import annotations

import asyncio
import logging
import os

from firebase_admin import auth

import app.core.config as cfg
from app.api.routers.pages import SITE_URL
from app.services import mailer, share_snapshots, watch_service
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import result_text
from app.services.llm.consensus_engine import (
    compute_agreement_score,
    is_consensus_error_text,
    query_consensus,
    query_consensus_change,
    query_differences,
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


TICK_SECONDS = 30 * 60
PROVIDER_ORDER = ("openai", "mistral", "gemini", "anthropic", "deepseek", "grok")
PROVIDER_LABELS = {
    "openai": "OpenAI", "mistral": "Mistral", "anthropic": "Anthropic",
    "gemini": "Gemini", "deepseek": "DeepSeek", "grok": "Grok",
}
PROVIDER_FUNCTIONS = {
    "openai": query_openai, "mistral": query_mistral, "anthropic": query_claude,
    "gemini": query_gemini, "deepseek": query_deepseek, "grok": query_grok,
}
PROVIDER_ENV = {
    "openai": "DEVELOPER_OPENAI_API_KEY", "mistral": "DEVELOPER_MISTRAL_API_KEY",
    "anthropic": "DEVELOPER_ANTHROPIC_API_KEY", "gemini": "DEVELOPER_GEMINI_API_KEY",
    "deepseek": "DEVELOPER_DEEPSEEK_API_KEY", "grok": "DEVELOPER_GROK_API_KEY",
}


def _developer_keys() -> dict:
    return {PROVIDER_LABELS[p]: os.environ.get(env, "").strip() for p, env in PROVIDER_ENV.items()}


def _selected_providers(keys: dict) -> list[str]:
    if mock_llm_enabled():
        return list(PROVIDER_ORDER[:3])
    return [p for p in PROVIDER_ORDER if keys.get(PROVIDER_LABELS[p])][:3]


def _provider_answer(provider: str, question: str, keys: dict):
    label = PROVIDER_LABELS[provider]
    if mock_llm_enabled():
        return mock_ask_result(label, question)
    kwargs = {
        "system_prompt": get_system_prompt(),
        "deep_search": False,
        "model_override": cfg.FREE_DEFAULT_MODEL_BY_PROVIDER[provider],
        "max_output_tokens": cfg.get_output_token_limit(False, False),
        "attachments": [],
    }
    key = keys.get(label) or ""
    if provider == "gemini":
        return PROVIDER_FUNCTIONS[provider](question, user_api_key=key, **kwargs)
    return PROVIDER_FUNCTIONS[provider](question, key, **kwargs)


def execute_watch(question: str, previous_consensus: str) -> dict:
    """Run at most three current free defaults; never touches usage counters."""
    keys = _developer_keys()
    providers = _selected_providers(keys)
    if mock_llm_enabled():
        for provider in providers:
            keys[PROVIDER_LABELS[provider]] = "mock"
    answers = {}
    for provider in providers:
        result = _provider_answer(provider, question, keys)
        text = result_text(result).strip()
        if text and not text.lower().startswith("error") and not (isinstance(result, dict) and result.get("error")):
            answers[provider] = text
    if len(answers) < 2:
        raise RuntimeError("Fewer than two provider answers completed.")

    slots = {provider: answers.get(provider, "") for provider in ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok")}
    excluded = [PROVIDER_LABELS[p] for p in slots if p not in answers]
    engine_provider = next(iter(answers))
    engine = PROVIDER_LABELS[engine_provider]
    consensus = query_consensus(
        question, slots["openai"], slots["mistral"], slots["anthropic"],
        slots["gemini"], slots["deepseek"], slots["grok"], excluded, engine, keys,
    )
    if is_consensus_error_text(consensus):
        raise RuntimeError("Consensus synthesis failed.")
    _legacy, differences = query_differences(
        slots["openai"], slots["mistral"], slots["anthropic"], slots["gemini"],
        slots["deepseek"], slots["grok"], consensus, keys, engine, excluded,
    )
    if not isinstance(differences, dict):
        raise RuntimeError("Differences Judge failed.")
    agreement = compute_agreement_score(differences)
    differences["agreement"] = agreement
    change = query_consensus_change(previous_consensus, consensus, keys, engine)
    return {
        "consensus": consensus,
        "agreement_score": agreement["score"],
        "verdict": agreement.get("level") or "",
        **change,
    }


def should_notify(old_score, new_score, changed: bool, severity: str) -> bool:
    try:
        delta = abs(float(new_score) - float(old_score))
    except (TypeError, ValueError):
        delta = 0
    return (bool(changed) and severity == "major") or delta >= 15


def _notification_context(watch_id: str, watch: dict):
    share_path = share_snapshots.share_path(watch.get("share_slug") or "", watch["share_id"])
    share_url = SITE_URL + share_path
    token = watch_service.make_unsubscribe_token(watch_id)
    return share_url, SITE_URL + "/watch/unsubscribe?token=" + token


async def _send_change_mail(watch_id: str, watch: dict, result: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        logging.warning("Watch %s owner has no verified e-mail; notification skipped", watch_id)
        return
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    summary = result.get("change_summary") or "The agreement score changed materially."
    message = mailer.build_change_message(
        recipient=user.email, question=watch.get("question") or "",
        old_score=watch.get("last_agreement_score"), new_score=result["agreement_score"],
        summary=summary, share_url=share_url, unsubscribe_url=unsubscribe_url,
    )
    await mailer.send_message(message)


async def _send_paused_mail(watch_id: str, watch: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        return
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    await mailer.send_message(mailer.build_paused_message(
        recipient=user.email, question=watch.get("question") or "",
        share_url=share_url, unsubscribe_url=unsubscribe_url,
    ))


async def run_watch_tick() -> int:
    now = watch_service.utcnow()
    if not await asyncio.to_thread(watch_service.acquire_worker_lease, now=now):
        return 0
    completed = 0
    try:
        due_ids = await asyncio.to_thread(watch_service.list_due_watch_ids, now=now)
        for watch_id in due_ids:
            claimed, reason = await asyncio.to_thread(watch_service.claim_watch, watch_id, now=now)
            if reason == "budget":
                break
            if not claimed:
                continue
            try:
                result = await asyncio.to_thread(
                    execute_watch, claimed.get("question") or "", claimed.get("last_consensus_text") or ""
                )
                await asyncio.to_thread(watch_service.complete_watch_run, watch_id, claimed, result, now=watch_service.utcnow())
            except Exception:
                logging.exception("Consensus Watch run failed for %s", watch_id)
                paused = await asyncio.to_thread(watch_service.fail_watch_run, watch_id, claimed, now=watch_service.utcnow())
                if paused:
                    try:
                        await _send_paused_mail(watch_id, claimed)
                    except Exception:
                        logging.exception("Consensus Watch pause mail failed for %s", watch_id)
            else:
                completed += 1
                if should_notify(claimed.get("last_agreement_score"), result["agreement_score"], result["changed"], result["severity"]):
                    try:
                        await _send_change_mail(watch_id, claimed, result)
                    except Exception:
                        # Mail is best-effort and must never turn a completed
                        # LLM run into a scheduler failure/history rollback.
                        logging.exception("Consensus Watch change mail failed for %s", watch_id)
    finally:
        try:
            await asyncio.to_thread(watch_service.release_worker_lease)
        except Exception:
            logging.exception("Consensus Watch worker lease release failed")
    return completed


async def watch_scheduler_loop():
    while True:
        try:
            await run_watch_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Consensus Watch scheduler tick failed")
        await asyncio.sleep(TICK_SECONDS)
