"""Sequential, Firestore-leased background runner for Consensus Watch."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from firebase_admin import auth

import app.core.config as cfg
from app.core import security
from app.api.routers.pages import SITE_URL
from app.services import (
    mailer, opinion_map, share_snapshots, telegram_watch, watch_brief,
    watch_followers, watch_service,
)
from app.services.llm.base import get_system_prompt
from app.services.llm.citations import result_sources, result_text
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
_scheduler_wake_event: asyncio.Event | None = None
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


def _selected_models(keys: dict, is_pro: bool, excluded_providers=None) -> list[tuple[str, str]]:
    configured = cfg.get_watch_models(is_pro)
    excluded = {
        str(provider or "").strip().lower() for provider in (excluded_providers or ())
    }
    if mock_llm_enabled():
        return [
            (provider, configured[provider])
            for provider in PROVIDER_ORDER
            if provider not in excluded and configured.get(provider)
        ]
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    return [
        (provider, configured[provider]) for provider in PROVIDER_ORDER
        if provider not in excluded and configured.get(provider) and (
            keys.get(PROVIDER_LABELS[provider])
            or (provider == "gemini" and adc_path and os.path.isfile(adc_path))
        )
    ]


def _provider_answer(provider: str, model: str, question: str, keys: dict, is_pro: bool):
    label = PROVIDER_LABELS[provider]
    if mock_llm_enabled():
        return mock_ask_result(label, question)
    kwargs = {
        "system_prompt": get_system_prompt(),
        "deep_search": False,
        "model_override": model,
        "max_output_tokens": cfg.get_output_token_limit(is_pro, False),
        "attachments": [],
    }
    key = keys.get(label) or ""
    if provider == "gemini":
        return PROVIDER_FUNCTIONS[provider](question, user_api_key=key, **kwargs)
    return PROVIDER_FUNCTIONS[provider](question, key, **kwargs)


def execute_watch(question: str, previous_consensus: str, condition: str = "",
                  previous_opinion_map=None, is_pro: bool = False,
                  excluded_providers=None, baseline_consensus: str = "") -> dict:
    """Run the configured tier models; never touches usage counters."""
    keys = _developer_keys()
    excluded = {
        str(provider or "").strip().lower() for provider in (excluded_providers or ())
    }
    for provider in excluded:
        if provider in PROVIDER_LABELS:
            keys[PROVIDER_LABELS[provider]] = ""
    selected_models = _selected_models(keys, is_pro, excluded)
    if mock_llm_enabled():
        for provider, _model in selected_models:
            keys[PROVIDER_LABELS[provider]] = "mock"
    answers = {}
    model_sources = {}
    with ThreadPoolExecutor(max_workers=max(1, len(selected_models))) as pool:
        futures = {
            provider: pool.submit(_provider_answer, provider, model, question, keys, is_pro)
            for provider, model in selected_models
        }
        # In konfigurierter Provider-Reihenfolge einsammeln, damit Engine-Wahl
        # und Quellen-Nummerierung trotz parallelem Fan-out deterministisch sind.
        for provider, _model in selected_models:
            try:
                result = futures[provider].result()
            except Exception:
                logging.exception("Consensus Watch provider failed: %s", provider)
                continue
            text = result_text(result).strip()
            if text and not text.lower().startswith("error") and not (isinstance(result, dict) and result.get("error")):
                answers[provider] = text
                sources = result_sources(result)
                if sources:
                    model_sources[PROVIDER_LABELS[provider]] = sources
    if len(answers) < 2:
        raise RuntimeError("Fewer than two provider answers completed.")

    slots = {provider: answers.get(provider, "") for provider in ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok")}
    excluded = [PROVIDER_LABELS[p] for p in slots if p not in answers]
    engine_provider = next(iter(answers))
    engine = PROVIDER_LABELS[engine_provider]
    consensus = query_consensus(
        question, slots["openai"], slots["mistral"], slots["anthropic"],
        slots["gemini"], slots["deepseek"], slots["grok"], excluded, engine, keys,
        model_sources=model_sources,
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
    if str(previous_consensus or "").strip():
        change = query_consensus_change(
            previous_consensus, consensus, keys, engine, condition=condition,
        )
    else:
        # A query-first Watch intentionally has no manual Consensus baseline.
        # Its first scheduled result establishes that baseline and must not be
        # reported as a material change merely because the old text was empty.
        change = {
            "changed": False,
            "severity": "minor",
            "change_summary": "First consensus established.",
        }
        if condition:
            condition_result = query_consensus_change(
                consensus, consensus, keys, engine, condition=condition,
            )
            change.update({
                key: condition_result[key]
                for key in ("condition_status", "condition_reason")
                if key in condition_result
            })
    baseline = str(baseline_consensus or previous_consensus or "")
    if baseline.strip() and baseline.strip() != str(previous_consensus or "").strip():
        baseline_change = query_consensus_change(baseline, consensus, keys, engine)
    else:
        baseline_change = change
    position_map = opinion_map.build_opinion_map(
        differences,
        previous_opinion_map,
        consensus_changed=bool(change.get("changed")),
    )
    included_providers = [PROVIDER_LABELS[provider] for provider in answers]
    model_labels = {
        PROVIDER_LABELS[provider]: model for provider, model in selected_models
        if provider in answers
    }
    return {
        "consensus": consensus,
        "agreement_score": agreement["score"],
        "verdict": agreement.get("level") or "",
        "opinion_map": position_map,
        "differences_data": differences,
        "differences_text": "",
        "sources": share_snapshots.sanitize_sources(model_sources),
        "included_models": share_snapshots.build_included_models(
            included_providers, model_labels,
        ),
        "consensus_model": engine,
        "baseline_changed": bool(baseline_change.get("changed")),
        "baseline_severity": baseline_change.get("severity") or "minor",
        "baseline_summary": baseline_change.get("change_summary") or "",
        **change,
    }


def should_notify(old_score, new_score, changed: bool, severity: str) -> bool:
    try:
        delta = abs(float(new_score) - float(old_score))
    except (TypeError, ValueError):
        delta = 0
    return (bool(changed) and severity == "major") or delta >= 15


def notification_kind(watch: dict, result: dict) -> str | None:
    email_mode = watch.get("email_mode") or "changes_only"
    if email_mode == "every_run":
        return "every_run"
    if email_mode == "condition":
        current_hash = watch_service.condition_hash(watch.get("condition") or "")
        previous_is_same_condition = watch.get("last_condition_hash") == current_hash
        if (result.get("condition_status") == "met"
                and (watch.get("last_condition_status") != "met"
                     or not previous_is_same_condition)):
            return "condition"
        return None
    if should_notify(
        watch.get("last_agreement_score"), result.get("agreement_score"),
        bool(result.get("changed")), result.get("severity") or "minor",
    ):
        return "change"
    return None


def _notification_context(watch_id: str, watch: dict):
    slug = "" if watch.get("visibility") == "private" else watch.get("share_slug") or ""
    share_path = share_snapshots.share_path(slug, watch["share_id"])
    share_url = SITE_URL + share_path
    token = watch_service.make_unsubscribe_token(watch_id)
    return share_url, SITE_URL + "/watch/unsubscribe?token=" + token


async def _send_change_mail(watch_id: str, watch: dict, result: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return False
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        logging.warning("Watch %s owner has no verified e-mail; notification skipped", watch_id)
        return False
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    summary = result.get("change_summary") or "The agreement score changed materially."
    message = mailer.build_change_message(
        recipient=user.email, question=watch.get("question") or "",
        old_score=watch.get("last_agreement_score"), new_score=result["agreement_score"],
        summary=summary, share_url=share_url, unsubscribe_url=unsubscribe_url,
    )
    return await mailer.send_message(message)


async def _send_run_mail(watch_id: str, watch: dict, result: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return False
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        logging.warning("Watch %s owner has no verified e-mail; notification skipped", watch_id)
        return False
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    return await mailer.send_message(mailer.build_run_message(
        recipient=user.email,
        question=watch.get("question") or "",
        agreement_score=result["agreement_score"],
        consensus=result.get("consensus") or "",
        changed=bool(result.get("changed")),
        severity=result.get("severity") or "minor",
        summary=result.get("change_summary") or "",
        share_url=share_url,
        unsubscribe_url=unsubscribe_url,
    ))


async def _send_condition_mail(watch_id: str, watch: dict, result: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return False
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        logging.warning("Watch %s owner has no verified e-mail; notification skipped", watch_id)
        return False
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    return await mailer.send_message(mailer.build_condition_message(
        recipient=user.email,
        question=watch.get("question") or "",
        condition=watch.get("condition") or "",
        reason=result.get("condition_reason") or "The condition is met by the new consensus.",
        agreement_score=result["agreement_score"],
        consensus=result.get("consensus") or "",
        share_url=share_url,
        unsubscribe_url=unsubscribe_url,
    ))


async def _send_follower_mails(watch_id: str, watch: dict, result: dict) -> int:
    """Bestätigte Seiten-Follower bei materiellen Änderungen benachrichtigen.

    Unabhängig vom email_mode des Owners; Schwelle ist dieselbe wie bei
    "changes_only". Best-effort – Fehler je Empfänger brechen nichts ab.
    """
    if str(watch.get("visibility") or "public") == "private":
        return 0
    if not mailer.is_configured():
        return 0
    if not should_notify(
        watch.get("last_agreement_score"), result.get("agreement_score"),
        bool(result.get("changed")), result.get("severity") or "minor",
    ):
        return 0
    followers = await asyncio.to_thread(watch_followers.list_followers, watch["share_id"])
    if not followers:
        return 0
    share_url = SITE_URL + share_snapshots.share_path(
        watch.get("share_slug") or "", watch["share_id"],
    )
    summary = result.get("change_summary") or "The agreement score changed materially."
    sent = 0
    for follower in followers:
        try:
            token = watch_followers.make_follow_unsubscribe_token(
                watch["share_id"], follower["email"],
            )
            message = mailer.build_follower_change_message(
                recipient=follower["email"], question=watch.get("question") or "",
                old_score=watch.get("last_agreement_score"),
                new_score=result["agreement_score"], summary=summary,
                share_url=share_url,
                unsubscribe_url=SITE_URL + "/watch/follow/unsubscribe?token=" + token,
            )
            if await mailer.send_message(message):
                sent += 1
        except Exception:
            logging.exception("Consensus Watch follower mail failed for %s", watch_id)
    return sent


async def _send_paused_mail(watch_id: str, watch: dict):
    if not mailer.is_configured():
        logging.info("Consensus Watch mail skipped: SMTP_HOST/MAIL_FROM not configured")
        return False
    user = await asyncio.to_thread(auth.get_user, watch["owner_uid"])
    if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
        return False
    share_url, unsubscribe_url = _notification_context(watch_id, watch)
    return await mailer.send_message(mailer.build_paused_message(
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
                share = await asyncio.to_thread(share_snapshots.get_share, claimed["share_id"])
                if not share or share.get("status") != "active":
                    raise RuntimeError("Watch share is unavailable.")
                claimed["question"] = share.get("question") or ""
                claimed["share_slug"] = share.get("slug") or ""
                claimed["initial_watch_run"] = bool(
                    share.get("awaiting_first_watch_run")
                    and not claimed.get("last_successful_run_id")
                )
                account_is_pro = await asyncio.to_thread(
                    security.is_user_pro, claimed["owner_uid"]
                )
                # Scheduled Publisher pages are deliberately pinned to the
                # Admin-configured Free Watch providers. All ordinary watches
                # continue to follow the owner's live account tier.
                is_pro = False if claimed.get("model_tier") == "free" else account_is_pro
                excluded_providers = claimed.get("excluded_providers") or (
                    ("deepseek",) if claimed.get("model_tier") == "free" else ()
                )
                try:
                    history = await asyncio.to_thread(
                        share_snapshots.list_watch_history, claimed["share_id"], max_items=1,
                    )
                except Exception:
                    logging.warning(
                        "Consensus Watch position baseline unavailable for %s",
                        watch_id, exc_info=True,
                    )
                    history = []
                previous_position_map = (
                    history[-1].get("opinion_map") if history
                    else opinion_map.build_opinion_map(share.get("differences_data") or {})
                )
                previous_version = None
                previous_run_id = str(claimed.get("last_successful_run_id") or "")
                try:
                    if previous_run_id:
                        previous_version = await asyncio.to_thread(
                            share_snapshots.get_watch_version,
                            claimed["share_id"], previous_run_id,
                        )
                except Exception:
                    logging.warning(
                        "Consensus Watch text baseline unavailable for %s",
                        watch_id, exc_info=True,
                    )
                original_consensus = share.get("consensus_md") or ""
                previous_consensus = (
                    previous_version.get("consensus_md")
                    if previous_version else original_consensus
                )
                result = await asyncio.to_thread(
                    execute_watch, claimed["question"], previous_consensus,
                    claimed.get("condition") if claimed.get("email_mode") == "condition" else "",
                    previous_position_map,
                    is_pro,
                    excluded_providers=excluded_providers,
                    baseline_consensus=original_consensus,
                )
                mail_kind = notification_kind(claimed, result)
                run_id = str(claimed.get("current_run_id") or "")
                await asyncio.to_thread(
                    watch_service.complete_watch_run, watch_id, claimed, result,
                    now=watch_service.utcnow(),
                    defer_condition_status=mail_kind == "condition",
                )
            except Exception:
                logging.exception("Consensus Watch run failed for %s", watch_id)
                paused = await asyncio.to_thread(watch_service.fail_watch_run, watch_id, claimed, now=watch_service.utcnow())
                if paused:
                    try:
                        if claimed.get("email_enabled") is not False:
                            await _send_paused_mail(watch_id, claimed)
                        await asyncio.to_thread(
                            telegram_watch.send_watch_notification,
                            watch_id, str(claimed.get("current_run_id") or "failed"),
                            "paused_error", claimed, {},
                        )
                    except Exception:
                        logging.exception("Consensus Watch pause notification failed for %s", watch_id)
            else:
                completed += 1
                if mail_kind:
                    notification_sent = False
                    try:
                        if claimed.get("email_enabled") is not False:
                            if mail_kind == "every_run":
                                notification_sent = bool(
                                    await _send_run_mail(watch_id, claimed, result)
                                )
                            elif mail_kind == "condition":
                                notification_sent = bool(
                                    await _send_condition_mail(watch_id, claimed, result)
                                )
                            else:
                                notification_sent = bool(
                                    await _send_change_mail(watch_id, claimed, result)
                                )
                    except Exception:
                        # Mail is best-effort and must never turn a completed
                        # LLM run into a scheduler failure/history rollback.
                        logging.exception("Consensus Watch result mail failed for %s", watch_id)
                    try:
                        telegram_sent = await asyncio.to_thread(
                            telegram_watch.send_watch_notification,
                            watch_id, run_id, mail_kind, claimed, result,
                        )
                        notification_sent = notification_sent or telegram_sent
                    except Exception:
                        logging.exception("Consensus Watch Telegram delivery failed for %s", watch_id)
                    if mail_kind == "condition" and notification_sent:
                        await asyncio.to_thread(
                            watch_service.set_condition_status, watch_id, "met",
                            claimed.get("condition") or "",
                        )
                try:
                    await _send_follower_mails(watch_id, claimed, result)
                except Exception:
                    logging.exception("Consensus Watch follower mails failed for %s", watch_id)
    finally:
        try:
            await asyncio.to_thread(watch_service.release_worker_lease)
        except Exception:
            logging.exception("Consensus Watch worker lease release failed")
    return completed


async def run_brief_tick() -> int:
    """Deliver due Morning Briefs. Claim-advance happens transactionally per
    brief BEFORE sending, so a crash can skip one digest but never double-send."""
    if not mailer.is_configured():
        return 0
    now = watch_brief.utcnow()
    sent = 0
    try:
        due_uids = await asyncio.to_thread(watch_brief.list_due_brief_uids, now=now)
    except Exception:
        logging.exception("Morning brief due-scan failed")
        return 0
    for uid in due_uids:
        try:
            claimed = await asyncio.to_thread(watch_brief.claim_brief, uid, now=now)
            if not claimed:
                continue
            user = await asyncio.to_thread(auth.get_user, uid)
            if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
                logging.warning("Morning brief for %s skipped: no verified e-mail", uid)
                continue
            items, changes = await asyncio.to_thread(
                watch_brief.collect_brief_items, uid, since=claimed["baseline"],
            )
            if not items:
                continue
            if claimed.get("mode") == "changes_only" and changes == 0:
                continue
            timezone_name = str(claimed.get("timezone") or "UTC")
            try:
                date_label = now.astimezone(ZoneInfo(timezone_name)).strftime("%A, %d %B %Y")
            except (ZoneInfoNotFoundError, ValueError):
                date_label = now.strftime("%A, %d %B %Y")
            message = mailer.build_brief_message(
                recipient=user.email, date_label=date_label, items=items,
                changes_count=changes, site_url=SITE_URL,
                unsubscribe_url=SITE_URL + "/watch/brief/unsubscribe?token="
                + watch_brief.make_brief_unsubscribe_token(uid),
            )
            if await mailer.send_message(message):
                await asyncio.to_thread(watch_brief.mark_brief_sent, uid, now=now)
                sent += 1
        except Exception:
            logging.exception("Morning brief delivery failed for %s", uid)
    return sent


def wake_watch_scheduler():
    """Wake the in-process scheduler so newly queued work starts promptly."""
    if _scheduler_wake_event is not None:
        _scheduler_wake_event.set()


async def watch_scheduler_loop():
    global _scheduler_wake_event
    wake_event = asyncio.Event()
    _scheduler_wake_event = wake_event
    try:
        while True:
            # Clear before the tick so a wake-up arriving during a long run is
            # retained and causes another immediate scan afterwards.
            wake_event.clear()
            try:
                await run_watch_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Consensus Watch scheduler tick failed")
            try:
                await run_brief_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Morning brief scheduler tick failed")
            try:
                await asyncio.wait_for(wake_event.wait(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                pass
    finally:
        if _scheduler_wake_event is wake_event:
            _scheduler_wake_event = None
