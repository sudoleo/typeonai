"""Sequential, Firestore-leased background runner for Consensus Watch."""

from __future__ import annotations

import asyncio
import logging
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from firebase_admin import auth

import app.core.config as cfg
from app.core import security
from app.core.background_tasks import task_succeeded
from app.core.observability import correlation_scope, record_metric, safe_exception
from app.core.site import SITE_URL
from app.services.consensus_pipeline import run_consensus_pipeline
from app.services import (
    drift_signal, mailer, opinion_map, share_snapshots, telegram_watch,
    watch_brief, watch_followers, watch_service,
)
from app.services.llm import provider_transport
from app.services.llm.consensus_engine import (
    query_consensus,
    query_consensus_change,
    query_differences,
)
from app.services.llm.mock_llm import mock_llm_enabled


TICK_SECONDS = 30 * 60
WATCH_LEASE_HEARTBEAT_SECONDS = 5 * 60
_scheduler_wake_event: asyncio.Event | None = None
# Preserve the established Watch engine preference. Topic/API use the shared
# canonical display order, while Watch historically preferred Gemini before
# Anthropic when both earlier engines failed.
_WATCH_ENGINE_PREFERENCE = ("openai", "mistral", "gemini", "anthropic", "deepseek", "grok")
PROVIDER_ORDER = tuple(dict.fromkeys(
    [provider for provider in _WATCH_ENGINE_PREFERENCE if provider in cfg.PROVIDERS]
    + list(cfg.PROVIDERS)
))
PROVIDER_LABELS = provider_transport.PROVIDER_LABELS


def _developer_keys() -> dict:
    return provider_transport.developer_keys()


def _selected_models(keys: dict, is_pro: bool,
                     model_overrides=None) -> list[tuple[str, str]]:
    """Every provider the tier configures, minus the ones with no credential.

    A missing server key is the only reason a configured provider can drop out
    of a run. There is deliberately no second, invisible filter on top of the
    Admin configuration.
    """
    configured = (
        dict(model_overrides)
        if isinstance(model_overrides, dict)
        else cfg.get_watch_models(is_pro)
    )
    if mock_llm_enabled():
        return [
            (provider, configured[provider])
            for provider in PROVIDER_ORDER
            if configured.get(provider)
        ]
    return [
        (provider, configured[provider]) for provider in PROVIDER_ORDER
        if configured.get(provider)
        and provider_transport.provider_available(provider, keys)
    ]


def _provider_answer(provider: str, model: str, question: str, keys: dict,
                     is_pro: bool, deep_think: bool = False):
    return provider_transport.query_provider(
        provider, model, question, keys, is_pro, deep_think
    )


def _configured_consensus_engine(keys: dict, is_pro: bool) -> str | None:
    """Return the configured Watch engine when its provider can be called."""
    chosen = cfg.get_watch_consensus_model(is_pro)
    resolved = cfg.get_consensus_model_config(chosen)
    if not resolved or not resolved.provider:
        return None
    if not provider_transport.provider_available(resolved.provider, keys):
        return None
    return chosen


def execute_watch(question: str, previous_consensus: str, condition: str = "",
                  previous_opinion_map=None, is_pro: bool = False,
                  baseline_consensus: str = "",
                  model_overrides=None) -> dict:
    """Run the configured tier models; never touches usage counters."""
    keys = _developer_keys()
    selected_models = _selected_models(
        keys, is_pro, model_overrides=model_overrides
    )
    configured_engine = _configured_consensus_engine(keys, is_pro)
    if mock_llm_enabled():
        for provider, _model in selected_models:
            keys[PROVIDER_LABELS[provider]] = "mock"
        if configured_engine:
            provider = cfg.get_consensus_model_config(configured_engine).provider
            keys[PROVIDER_LABELS[provider]] = "mock"
    provider_models = dict(selected_models)

    def first_successful_engine(answers) -> str:
        provider = next(name for name, _model in selected_models if name in answers)
        return PROVIDER_LABELS[provider]

    pipeline = run_consensus_pipeline(
        question=question,
        provider_models=provider_models,
        consensus_model=configured_engine or first_successful_engine,
        keys=keys,
        is_pro=is_pro,
        deep_think=False,
        provider_order=PROVIDER_ORDER,
        provider_call=_provider_answer,
        synthesize=query_consensus,
        judge=query_differences,
        log_context="Consensus Watch",
    )
    engine = configured_engine or first_successful_engine({
        provider: True
        for provider, _model in selected_models
        if any(
            item["provider"] == PROVIDER_LABELS[provider]
            for item in pipeline["model_answers"]
        )
    })
    consensus = pipeline["consensus_response"]
    differences = pipeline["differences_data"]
    agreement = pipeline["agreement"]
    model_answers = pipeline["model_answers"]
    model_sources = {
        item["provider"]: item.get("sources") or []
        for item in model_answers if item.get("sources")
    }
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
    included_providers = [item["provider"] for item in model_answers]
    model_labels = {
        item["provider"]: item["model"] for item in model_answers
    }
    return {
        "consensus": consensus,
        "agreement_score": agreement["score"],
        "verdict": agreement.get("level") or "",
        "opinion_map": position_map,
        "differences_data": differences,
        "differences_text": pipeline["differences"],
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


def should_notify(old_score, new_score, changed: bool, severity: str,
                  previous_scores=None) -> bool:
    """The mail bar, and since the drift rule was unified, the page bar too.

    ``previous_scores`` is the band of the recent checks; without it the
    predecessor alone forms the band, which is what a two-point history means
    anyway.
    """
    if previous_scores is None:
        previous_scores = [old_score] if isinstance(old_score, (int, float)) else []
    return drift_signal.is_material(changed, severity, new_score, previous_scores)


def _recent_scores(watch: dict) -> list:
    points = watch.get("history_points")
    return drift_signal.recent_scores(points if isinstance(points, list) else [])


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
        _recent_scores(watch) or None,
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
        severity=result.get("severity") or "major",
        direction=result.get("opinion_map"),
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
        old_score=watch.get("last_agreement_score"),
        direction=result.get("opinion_map"),
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
        old_score=watch.get("last_agreement_score"),
        direction=result.get("opinion_map"),
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
        _recent_scores(watch) or None,
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
                severity=result.get("severity") or "major",
                direction=result.get("opinion_map"),
            )
            if await mailer.send_message(message):
                sent += 1
        except Exception as exc:
            logging.error(
                "Consensus Watch follower mail failed for %s category=%s",
                watch_id,
                safe_exception(exc),
            )
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


async def _renew_watch_lease_until_stopped(
    watch_id: str, run_id: str, stop: asyncio.Event
) -> None:
    while True:
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=WATCH_LEASE_HEARTBEAT_SECONDS
            )
            return
        except asyncio.TimeoutError:
            renewed = await asyncio.to_thread(
                watch_service.renew_watch_lease,
                watch_id,
                run_id,
                now=watch_service.utcnow(),
            )
            if not renewed:
                logging.warning("Consensus Watch lease lost for %s", watch_id)
                return


async def run_watch_tick() -> int:
    # MOCK_LLM instances share the production Firestore: a mock tick would take
    # the worker lease from the live deployment and persist fixture answers as
    # real watch runs. execute_watch itself stays mockable for the test suite.
    if mock_llm_enabled():
        return 0
    now = watch_service.utcnow()
    if not await asyncio.to_thread(watch_service.acquire_worker_lease, now=now):
        return 0
    completed = 0
    try:
        due_ids = await asyncio.to_thread(watch_service.list_due_watch_ids, now=now)
        for watch_id in due_ids:
            claimed, reason = await asyncio.to_thread(
                watch_service.claim_watch,
                watch_id,
                now=watch_service.utcnow(),
            )
            if reason == "budget":
                break
            if not claimed:
                continue
            lease_stop = asyncio.Event()
            lease_heartbeat = asyncio.create_task(
                _renew_watch_lease_until_stopped(
                    watch_id,
                    str(claimed.get("current_run_id") or ""),
                    lease_stop,
                )
            )
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
                try:
                    history = await asyncio.to_thread(
                        share_snapshots.list_watch_history, claimed["share_id"], max_items=1,
                    )
                except Exception as exc:
                    logging.warning(
                        "Consensus Watch position baseline unavailable category=%s",
                        safe_exception(exc),
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
                except Exception as exc:
                    logging.warning(
                        "Consensus Watch text baseline unavailable category=%s",
                        safe_exception(exc),
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
                    baseline_consensus=original_consensus,
                )
                mail_kind = notification_kind(claimed, result)
                run_id = str(claimed.get("current_run_id") or "")
                persisted = await asyncio.to_thread(
                    watch_service.complete_watch_run, watch_id, claimed, result,
                    now=watch_service.utcnow(),
                    defer_condition_status=mail_kind == "condition",
                )
                if persisted is None:
                    logging.warning(
                        "Consensus Watch completion fenced out for %s", watch_id
                    )
                    continue
            except Exception as exc:
                logging.error(
                    "Consensus Watch run failed for %s category=%s",
                    watch_id,
                    safe_exception(exc),
                )
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
                    except Exception as exc:
                        logging.error(
                            "Consensus Watch pause notification failed for %s category=%s",
                            watch_id,
                            safe_exception(exc),
                        )
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
                    except Exception as exc:
                        # Mail is best-effort and must never turn a completed
                        # LLM run into a scheduler failure/history rollback.
                        logging.error(
                            "Consensus Watch result mail failed for %s category=%s",
                            watch_id,
                            safe_exception(exc),
                        )
                    try:
                        telegram_sent = await asyncio.to_thread(
                            telegram_watch.send_watch_notification,
                            watch_id, run_id, mail_kind, claimed, result,
                        )
                        notification_sent = notification_sent or telegram_sent
                    except Exception as exc:
                        logging.error(
                            "Consensus Watch Telegram delivery failed for %s category=%s",
                            watch_id,
                            safe_exception(exc),
                        )
                    if mail_kind == "condition" and notification_sent:
                        await asyncio.to_thread(
                            watch_service.set_condition_status, watch_id, "met",
                            claimed.get("condition") or "",
                            expected_run_id=run_id,
                        )
                try:
                    await _send_follower_mails(watch_id, claimed, result)
                except Exception as exc:
                    logging.error(
                        "Consensus Watch follower mails failed for %s category=%s",
                        watch_id,
                        safe_exception(exc),
                    )
            finally:
                lease_stop.set()
                await lease_heartbeat
    finally:
        try:
            await asyncio.to_thread(watch_service.release_worker_lease)
        except Exception as exc:
            logging.error(
                "Consensus Watch worker lease release failed category=%s",
                safe_exception(exc),
            )
    return completed


async def run_brief_tick() -> int:
    """Deliver due Morning Briefs. Claim-advance happens transactionally per
    brief BEFORE sending, so a crash can skip one digest but never double-send."""
    if mock_llm_enabled() or not mailer.is_configured():
        return 0
    now = watch_brief.utcnow()
    sent = 0
    try:
        due_uids = await asyncio.to_thread(watch_brief.list_due_brief_uids, now=now)
    except Exception as exc:
        logging.error(
            "Morning brief due-scan failed category=%s", safe_exception(exc)
        )
        return 0
    for uid in due_uids:
        try:
            claimed = await asyncio.to_thread(watch_brief.claim_brief, uid, now=now)
            if not claimed:
                continue
            user = await asyncio.to_thread(auth.get_user, uid)
            if not getattr(user, "email_verified", False) or not getattr(user, "email", None):
                logging.warning("Morning brief skipped: no verified e-mail")
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
        except Exception as exc:
            logging.error(
                "Morning brief delivery failed category=%s", safe_exception(exc)
            )
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
            started = time.monotonic()
            with correlation_scope(prefix="watch-tick"):
                try:
                    watches_ran = await run_watch_tick()
                    briefs_sent = await run_brief_tick()
                except Exception:
                    record_metric(
                        "scheduler", "consensus-watch",
                        duration_ms=(time.monotonic() - started) * 1000,
                        outcome="failure",
                    )
                    raise
                record_metric(
                    "scheduler", "consensus-watch",
                    duration_ms=(time.monotonic() - started) * 1000,
                    processed=int(watches_ran or 0) + int(briefs_sent or 0),
                )
            task_succeeded(
                "consensus-watch-scheduler",
                watches_ran=watches_ran,
                briefs_sent=briefs_sent,
            )
            try:
                await asyncio.wait_for(wake_event.wait(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                pass
    finally:
        if _scheduler_wake_event is wake_event:
            _scheduler_wake_event = None
