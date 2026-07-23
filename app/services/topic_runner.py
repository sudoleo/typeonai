"""Executable research and scheduling flow for public Topic tickers."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

from app.services import mailer, topics, watch_scheduler


TOPIC_SCHEDULER_INTERVAL_SECONDS = 60
SITE_URL = "https://www.consens.io"


def _research_question(topic: dict) -> str:
    rules = topic.get("source_rules") or {}
    preferred = ", ".join(rules.get("preferred_domains") or [])
    notes = str(rules.get("notes") or "").strip()
    guidance = [
        "Research the current state of this topic using live web sources.",
        "Prioritize primary, recent evidence such as official announcements, "
        "documentation, GitHub releases, and attributable public posts.",
        "Include source URLs through the provider's citation tools. Do not rely "
        "on links saved on an earlier Topic run.",
    ]
    if preferred:
        guidance.append("Prefer these domains when relevant: " + preferred + ".")
    if notes:
        guidance.append("Editorial source guidance: " + notes)
    return str(topic.get("lead_question") or "").strip() + "\n\n" + "\n".join(guidance)


def _source_type(url: str, allowed: list[str], preferred: list[str]) -> str:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    path = (urlsplit(url).path or "").lower()
    if host in {"x.com", "twitter.com"} or host.endswith(".x.com"):
        candidate = "x"
    elif host == "github.com" or host.endswith(".github.com"):
        candidate = "github"
    elif "docs" in host or "/docs" in path or "documentation" in path:
        candidate = "documentation"
    elif any(host == domain or host.endswith("." + domain) for domain in preferred):
        candidate = "official"
    else:
        candidate = "press"
    if candidate in allowed:
        return candidate
    return allowed[0] if allowed else "press"


def evidence_from_sources(sources, source_rules: dict) -> list[dict]:
    allowed = list(source_rules.get("allowed_types") or topics.EVIDENCE_TYPES)
    preferred = [
        str(domain or "").lower().removeprefix("www.")
        for domain in source_rules.get("preferred_domains") or []
    ]
    evidence = []
    for source in sources or []:
        if not isinstance(source, dict) or not source.get("url"):
            continue
        url = str(source["url"])
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        # Web-search tools sometimes hand back a bare citation index ("7") as
        # the title; fall back to the host so cards never show a lonely number.
        raw_title = str(source.get("title") or "").strip()
        title = raw_title if raw_title and not raw_title.isdigit() else host or url
        evidence.append({
            "type": _source_type(url, allowed, preferred),
            "title": title[:240],
            "url": url,
            "publisher": str(source.get("provider") or host)[:120],
            "published_at": "",
            "excerpt": "",
        })
    return evidence


def _model_stances(opinion_map: dict, provider: str) -> list[str]:
    stances = []
    for dimension in (opinion_map or {}).get("dimensions") or []:
        for position in dimension.get("positions") or []:
            if provider in (position.get("models") or []):
                stance = str(position.get("stance") or "").strip()
                if stance and stance not in stances:
                    stances.append(stance)
    return stances


def opinion_changes_from_maps(current: dict, previous: dict) -> list[dict]:
    changes = []
    for model in (current or {}).get("models") or []:
        if not model.get("moved"):
            continue
        provider = str(model.get("provider") or "").strip()
        old = _model_stances(previous, provider)
        new = _model_stances(current, provider)
        summary = str(model.get("summary") or "").strip()
        changes.append({
            "model": provider,
            "from": " · ".join(old)[:180],
            "to": " · ".join(new)[:180],
            "summary": (
                summary
                or f"{provider} materially changed its position on this Topic."
            )[:400],
        })
    return changes


def execute_claimed_topic(claimed: dict, *, actor_uid: str, db=None,
                          now=None, executor=None) -> dict:
    """Collect sources, run the selected models, and persist one immutable point."""
    db = db if db is not None else topics.db_firestore
    now = now or topics.utcnow()
    executor = executor or watch_scheduler.execute_watch
    previous = (
        topics.get_run(claimed["id"], str(claimed.get("latest_run_id") or ""), db=db)
        or {}
    )
    run_config = claimed.get("run_config") or {}
    try:
        result = executor(
            _research_question(claimed),
            str(previous.get("consensus_md") or ""),
            previous_opinion_map=previous.get("opinion_map"),
            model_overrides=run_config.get("provider_models"),
        )
        changed = bool(result.get("changed"))
        change_type = (
            str(result.get("severity") or "minor")
            if changed else "stable"
        )
        if change_type not in {"minor", "major"}:
            change_type = "minor" if changed else "stable"
        source_rules = topics.normalize_source_rules(claimed.get("source_rules") or {})
        run = topics.create_run(
            claimed["id"],
            {
                "observed_at": now,
                "consensus_md": result.get("consensus"),
                "agreement_score": result.get("agreement_score"),
                "change_type": change_type,
                "change_summary": (
                    result.get("change_summary")
                    or ("First consensus established." if not previous else "No material shift.")
                ),
                "opinion_changes": opinion_changes_from_maps(
                    result.get("opinion_map") or {},
                    previous.get("opinion_map") or {},
                ),
                "evidence": evidence_from_sources(
                    result.get("sources"), source_rules
                ),
                "models": result.get("included_models") or topics.topic_model_labels(
                    run_config.get("provider_models") or {}
                ),
                "differences_data": result.get("differences_data") or {},
                "opinion_map": result.get("opinion_map") or {},
                "run_mode": "automatic",
            },
            actor_uid=actor_uid,
            db=db,
            now=now,
            run_id=str(claimed.get("current_run_id") or ""),
        )
        return run
    except Exception as exc:
        topics.fail_topic_run(claimed["id"], str(exc), db=db, now=now)
        raise


def run_topic_now(topic_id: str, *, actor_uid: str, db=None, now=None,
                  executor=None) -> dict:
    db = db if db is not None else topics.db_firestore
    now = now or topics.utcnow()
    claimed = topics.claim_topic_run(topic_id, force=True, db=db, now=now)
    return execute_claimed_topic(
        claimed, actor_uid=actor_uid, db=db, now=now, executor=executor
    )


async def notify_topic_followers(topic: dict, run: dict, old_score) -> None:
    if run.get("change_type") not in {"minor", "major"} or not mailer.is_configured():
        return
    for follower in await asyncio.to_thread(topics.list_followers, topic["id"]):
        claimed = await asyncio.to_thread(
            topics.claim_delivery, topic["id"], run["id"], follower["id"]
        )
        if not claimed:
            continue
        unsubscribe_url = (
            SITE_URL + "/topic-follow/unsubscribe?token="
            + topics.make_unsubscribe_token(topic["id"], follower["email"])
        )
        sent = await mailer.send_message(mailer.build_topic_change_message(
            recipient=follower["email"],
            title=topic["title"],
            question=topic["lead_question"],
            old_score=old_score,
            new_score=run["agreement_score"],
            change_type=run["change_type"],
            summary=run.get("change_summary") or "The Topic consensus changed.",
            topic_url=f"{SITE_URL}/topics/{topic['slug']}",
            unsubscribe_url=unsubscribe_url,
        ))
        await asyncio.to_thread(
            topics.finish_delivery,
            topic["id"],
            run["id"],
            follower["id"],
            success=sent,
        )


async def run_due_topic_tick() -> int:
    ran = 0
    due_ids = await asyncio.to_thread(topics.list_due_topic_ids)
    for topic_id in due_ids:
        try:
            claimed = await asyncio.to_thread(
                topics.claim_topic_run, topic_id, force=False
            )
            await asyncio.to_thread(
                execute_claimed_topic, claimed, actor_uid="topic-scheduler"
            )
            current = await asyncio.to_thread(topics.get_topic, topic_id)
            run = await asyncio.to_thread(
                topics.get_run, topic_id, str(current.get("latest_run_id") or "")
            )
            await notify_topic_followers(
                current, run or {}, claimed.get("latest_agreement_score")
            )
            ran += 1
        except topics.TopicError as exc:
            if exc.code != "conflict":
                logging.warning("Topic scheduler skipped %s: %s", topic_id, exc.message)
        except Exception:
            logging.exception("Scheduled Topic run failed for %s", topic_id)
    return ran


async def topic_scheduler_loop() -> None:
    while True:
        try:
            await run_due_topic_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Topic scheduler tick failed")
        await asyncio.sleep(TOPIC_SCHEDULER_INTERVAL_SECONDS)
