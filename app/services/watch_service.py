"""Firestore-backed lifecycle helpers for Consensus Watch."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

import app.core.config as cfg
from app.core.entitlements import TIER_PRO
from app.core.security import db_firestore
from app.services import drift_signal, opinion_map, persistence_guard, share_snapshots


WATCHES_COLLECTION = "watches"
WATCH_INTERVALS = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=30),
}
WATCH_STATUSES = {"active", "paused"}
WATCH_EMAIL_MODES = {"changes_only", "condition", "every_run"}
WATCH_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
WATCH_CONDITION_MAX_CHARS = 500
WATCH_QUESTION_MIN_CHARS = 8
WATCH_RUN_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
UNSUBSCRIBE_MAX_AGE_DAYS = 90
WATCH_LEASE_MINUTES = 15
WORKER_LEASE_MINUTES = 29
RUNTIME_COLLECTION = "watch_runtime"
WATCH_HISTORY_POINTS = 16
WATCH_EVENT_CHECKED = "watch.checked"
WATCH_EVENT_CHANGED = "watch.changed"
WATCH_EVENT_CONDITION_MET = "watch.condition_met"
WATCH_EVENT_RUN_FAILED = "watch.run_failed"
PUBLISHER_SOURCE = "scheduled_publisher"
API_RUNS_COLLECTION = "api_consensus_runs"
WATCH_STATE_COLLECTION = "watch_state"
WATCH_UNIQUES_COLLECTION = "watch_uniques"
PUBLISHER_COUNTER_ID = "publisher_capacity"


def _where_equal(collection, field: str, value):
    """Use the current Firestore filter API with mock-compatible fallback."""
    try:
        return collection.where(filter=FieldFilter(field, "==", value))
    except TypeError:
        return collection.where(field, "==", value)


def _run_transaction(db, operation):
    fake_runner = getattr(db, "run_transaction", None)
    if callable(fake_runner):
        return fake_runner(operation)
    from firebase_admin import firestore

    transaction = db.transaction(max_attempts=12)

    @firestore.transactional
    def run(tx):
        return operation(tx)

    return run(transaction)


def _owner_state_ref(db, uid: str):
    return (
        db.collection("users").document(uid).collection(WATCH_STATE_COLLECTION)
        .document("quota")
    )


def _unique_ref(db, uid: str, uniqueness_key: str):
    digest = hashlib.sha256(
        f"{uid}\0{uniqueness_key}".encode("utf-8")
    ).hexdigest()
    return (
        db.collection("users").document(uid).collection(WATCH_UNIQUES_COLLECTION)
        .document(digest)
    )


def _publisher_counter_ref(db):
    return db.collection(RUNTIME_COLLECTION).document(PUBLISHER_COUNTER_ID)


def _safe_count(value) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


class WatchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _watch_id() -> str:
    return secrets.token_urlsafe(18).replace("-", "").replace("_", "")[:24]


def validate_interval(interval, tier) -> str:
    normalized = str(interval or "").strip().lower()
    if normalized not in WATCH_INTERVALS:
        raise WatchError("invalid_interval", "Interval must be daily, weekly, or monthly.")
    if normalized == "daily" and not cfg.is_watch_daily_allowed(tier):
        raise WatchError(
            "pro_required",
            "Daily watches require Plus or Pro."
            if cfg.watch_plus_daily_allowed()
            else "Daily watches require Pro.",
        )
    return normalized


def validate_email_mode(value) -> str:
    normalized = str(value or "changes_only").strip().lower()
    if normalized not in WATCH_EMAIL_MODES:
        raise WatchError(
            "invalid_email_mode",
            "Email mode must be changes_only, condition, or every_run.",
        )
    return normalized


def validate_notification_channel(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise WatchError("invalid_notification_channel", f"{name} must be true or false.")
    return value


def validate_condition(value, *, required=False) -> str:
    condition = " ".join(str(value or "").split()).strip()
    if required and not condition:
        raise WatchError("invalid_condition", "Enter a condition for this watch.")
    if len(condition) > WATCH_CONDITION_MAX_CHARS:
        raise WatchError(
            "invalid_condition",
            f"Condition must be at most {WATCH_CONDITION_MAX_CHARS} characters.",
        )
    return condition


def condition_hash(condition: str) -> str:
    normalized = validate_condition(condition)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def validate_run_schedule(run_time, timezone_name) -> tuple[str, str]:
    """Validate an optional local HH:MM + IANA timezone pair."""
    run_time = str(run_time or "").strip()
    timezone_name = str(timezone_name or "").strip()
    if not run_time and not timezone_name:
        return "", ""
    if not WATCH_RUN_TIME_RE.fullmatch(run_time):
        raise WatchError("invalid_run_time", "Run time must use HH:MM in 24-hour format.")
    if not timezone_name or len(timezone_name) > 64:
        raise WatchError("invalid_timezone", "A valid timezone is required.")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise WatchError("invalid_timezone", "A valid IANA timezone is required.") from exc
    return run_time, timezone_name


def validate_run_weekday(value, interval: str, *, has_run_time: bool) -> str:
    """Validate the optional local weekday used by weekly schedules."""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    if interval != "weekly":
        raise WatchError("invalid_run_weekday", "A run day can only be set for weekly watches.")
    if normalized not in WATCH_WEEKDAYS:
        raise WatchError("invalid_run_weekday", "Run day must be Monday through Sunday.")
    if not has_run_time:
        raise WatchError("invalid_run_weekday", "A weekly run day requires a run time.")
    return normalized


def next_scheduled_run(interval: str, run_time: str, timezone_name: str, run_weekday="", *,
                       now: datetime, previous_scheduled: datetime | None = None) -> datetime:
    """Advance by the existing interval while keeping the selected local time."""
    delta = WATCH_INTERVALS[interval]
    run_time, timezone_name = validate_run_schedule(run_time, timezone_name)
    run_weekday = validate_run_weekday(run_weekday, interval, has_run_time=bool(run_time))
    if not run_time:
        return now + delta
    zone = ZoneInfo(timezone_name)
    reference = previous_scheduled if isinstance(previous_scheduled, datetime) else now
    local_reference = reference.astimezone(zone)
    hour, minute = (int(part) for part in run_time.split(":"))
    if interval == "weekly" and run_weekday:
        target_weekday = WATCH_WEEKDAYS.index(run_weekday)
        days_ahead = (target_weekday - local_reference.weekday()) % 7
        if isinstance(previous_scheduled, datetime) and days_ahead == 0:
            days_ahead = 7
        candidate_date = local_reference.date() + timedelta(days=days_ahead)
        step = timedelta(days=7)
    else:
        candidate_date = (local_reference + delta).date()
        step = delta
    while True:
        local_candidate = datetime.combine(candidate_date, time(hour, minute), tzinfo=zone)
        candidate = local_candidate.astimezone(timezone.utc)
        if candidate > now:
            return candidate
        candidate_date += step


def _serialize_watch(watch_id: str, data: dict, share: dict | None = None) -> dict:
    def iso(value):
        return value.isoformat() if isinstance(value, datetime) else ""

    share_id = str(data.get("share_id") or "")
    share = share or {}
    slug = str(share.get("slug") or data.get("share_slug") or "")
    visibility = str(share.get("visibility") or data.get("visibility") or "public")
    baseline_agreement = (share.get("differences_data") or {}).get("agreement") or {}
    baseline_agreement_score = baseline_agreement.get("score")
    return {
        # Google-Listing-Status der Seite (Quelle: Share-Doc) fürs Dashboard.
        "indexed": bool(share.get("indexed")),
        "index_requested": bool(share.get("index_requested")),
        "index_eligible": bool(share.get("index_eligible")),
        "id": watch_id,
        "share_id": share_id,
        "share_path": share_snapshots.share_path("" if visibility == "private" else slug, share_id),
        "question": str(share.get("question") or data.get("question") or "")[:200],
        "interval": data.get("interval") or "weekly",
        "model_tier": "free" if data.get("model_tier") == "free" else "account",
        "publication_source": str(
            data.get("publication_source") or share.get("publication_source") or ""
        ),
        "run_weekday": str(data.get("run_weekday") or ""),
        "run_time": str(data.get("run_time") or ""),
        "timezone": str(data.get("timezone") or ""),
        "email_mode": data.get("email_mode") or "changes_only",
        # Legacy watches predate channel switches and remain e-mail enabled.
        "email_enabled": data.get("email_enabled") is not False,
        "telegram_enabled": data.get("telegram_enabled") is True,
        "telegram_muted_until": iso(data.get("telegram_muted_until")),
        "condition": str(data.get("condition") or ""),
        "last_condition_status": data.get("last_condition_status"),
        "visibility": visibility,
        "status": data.get("status") or "paused",
        "next_run_at": iso(data.get("next_run_at")),
        "last_run_at": iso(data.get("last_run_at")),
        "last_agreement_score": data.get("last_agreement_score"),
        "baseline_agreement_score": (
            baseline_agreement_score
            if isinstance(baseline_agreement_score, (int, float)) else None
        ),
        "last_successful_run_id": str(data.get("last_successful_run_id") or ""),
        "last_trigger": "changed" if data.get("last_trigger") == "changed" else "stable",
        "last_change_summary": str(data.get("last_change_summary") or "")[:400],
        "last_drift_score": data.get("last_drift_score"),
        "last_event_type": str(data.get("last_event_type") or ""),
        "query_first": data.get("query_first") is True,
        "awaiting_first_run": bool(
            share.get("awaiting_first_watch_run")
            and not data.get("last_successful_run_id")
        ),
        "created_at": iso(data.get("created_at")),
    }


def _owned_active_share(uid: str, share_id: str, db) -> dict:
    share = share_snapshots.get_share(share_id, db=db)
    if not share or share.get("status") != "active":
        raise WatchError("not_found", "Share not found.")
    if share.get("owner_uid") != uid:
        raise WatchError("forbidden", "You can only watch your own shares.")
    return share


def _check_active_limit(uid: str, tier, db, *, excluding_id: str | None = None):
    count = 0
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream():
        if doc.id == excluding_id:
            continue
        if (doc.to_dict() or {}).get("status") == "active":
            count += 1
    if count >= cfg.get_watch_active_limit(tier):
        raise WatchError("limit_reached", "Active watch limit reached.")


def _ensure_watch_indexes(
    uid: str,
    uniqueness_key: str,
    *,
    db,
    include_publisher: bool = False,
):
    """Lazily seed counters/uniqueness for watches created before Phase 2."""
    owner_ref = _owner_state_ref(db, uid)
    unique_ref = _unique_ref(db, uid, uniqueness_key)
    publisher_ref = _publisher_counter_ref(db)
    owner_exists = owner_ref.get().exists
    unique_exists = unique_ref.get().exists
    publisher_exists = publisher_ref.get().exists if include_publisher else True
    if owner_exists and unique_exists and publisher_exists:
        return

    owner_watches = list(
        _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream()
    )
    active_count = sum(
        1 for snapshot in owner_watches
        if (snapshot.to_dict() or {}).get("status") == "active"
    )
    if uniqueness_key.startswith("question:"):
        expected = uniqueness_key.removeprefix("question:")
        existing = next(
            (
                snapshot for snapshot in owner_watches
                if (snapshot.to_dict() or {}).get("query_first") is True
                and str((snapshot.to_dict() or {}).get("question_hash") or "")
                == expected
            ),
            None,
        )
    else:
        expected = uniqueness_key.removeprefix("share:")
        existing = next(
            (
                snapshot for snapshot in owner_watches
                if str((snapshot.to_dict() or {}).get("share_id") or "") == expected
            ),
            None,
        )
    publisher_active = 0
    if include_publisher and not publisher_exists:
        publisher_active = sum(
            1
            for snapshot in _where_equal(
                db.collection(WATCHES_COLLECTION), "model_tier", "free"
            ).stream()
            if (snapshot.to_dict() or {}).get("status") == "active"
        )

    def initialize(transaction):
        persistence_guard.ensure_account_write_allowed(
            uid=uid, db=db, transaction=transaction
        )
        owner_snapshot = owner_ref.get(transaction=transaction)
        unique_snapshot = unique_ref.get(transaction=transaction)
        publisher_snapshot = (
            publisher_ref.get(transaction=transaction) if include_publisher else None
        )
        if not owner_snapshot.exists:
            transaction.set(owner_ref, {
                "schema_version": 1,
                "active_count": active_count,
                "updated_at": utcnow(),
            })
        if not unique_snapshot.exists and existing is not None:
            existing_data = existing.to_dict() or {}
            transaction.set(unique_ref, {
                "watch_id": existing.id,
                "share_id": str(existing_data.get("share_id") or ""),
                "uniqueness_key": uniqueness_key,
            })
        if include_publisher and publisher_snapshot is not None and not publisher_snapshot.exists:
            transaction.set(publisher_ref, {
                "schema_version": 1,
                "active_count": publisher_active,
                "updated_at": utcnow(),
            })

    _run_transaction(db, initialize)


def create_watch(uid: str, *, interval, tier, email_mode="changes_only",
                 email_enabled=True, telegram_enabled=False,
                 condition="", visibility="public", run_time="", timezone_name="",
                 run_weekday="",
                 result_id=None,
                 share_id=None, question=None, model_tier="", return_existing=False,
                 bypass_active_limit=False,
                 publisher_active_limit=None, db=None) -> dict:
    db = db if db is not None else db_firestore
    interval = validate_interval(interval, tier)
    email_mode = validate_email_mode(email_mode)
    email_enabled = validate_notification_channel(email_enabled, "email_enabled")
    telegram_enabled = validate_notification_channel(telegram_enabled, "telegram_enabled")
    if not email_enabled and not telegram_enabled:
        raise WatchError("notification_channel_required", "Enable e-mail or Telegram notifications.")
    condition = validate_condition(condition, required=email_mode == "condition")
    run_time, timezone_name = validate_run_schedule(run_time, timezone_name)
    run_weekday = validate_run_weekday(run_weekday, interval, has_run_time=bool(run_time))
    try:
        visibility = share_snapshots.validate_share_visibility(visibility)
    except share_snapshots.ShareError as exc:
        raise WatchError(exc.code, exc.message) from exc
    normalized_model_tier = str(model_tier or "").strip().lower()
    if normalized_model_tier not in {"", "free"}:
        raise WatchError("invalid_model_tier", "Only the Free Watch model tier can be pinned.")
    if question is not None and not isinstance(question, str):
        raise WatchError("invalid_question", "Question must be text.")
    sources = [bool(result_id), bool(share_id), bool(str(question or "").strip())]
    if sum(sources) != 1:
        raise WatchError(
            "invalid_request",
            "Provide exactly one of result_id, share_id, or question.",
        )
    created_query_share = False
    query_share_doc = None
    if question:
        normalized_question = " ".join(str(question).split()).strip()
        if len(normalized_question) < WATCH_QUESTION_MIN_CHARS:
            raise WatchError("invalid_question", "Enter a complete question for this watch.")
        if len(normalized_question) > share_snapshots.MAX_QUESTION_CHARS:
            raise WatchError(
                "invalid_question",
                f"Question must be at most {share_snapshots.MAX_QUESTION_CHARS} characters.",
            )
        try:
            share_id, query_share_doc = share_snapshots.build_share_for_watch_query(
                uid, normalized_question, visibility=visibility,
            )
        except share_snapshots.ShareError as exc:
            raise WatchError(exc.code, exc.message) from exc
        created_query_share = True
    if result_id:
        try:
            created = share_snapshots.create_share_from_pending(
                uid, str(result_id), db=db, visibility=visibility,
            )
        except share_snapshots.ShareError as exc:
            raise WatchError(exc.code, exc.message) from exc
        share_id = created["share_id"]

    share_id = str(share_id)
    share = query_share_doc or _owned_active_share(uid, share_id, db)
    share_visibility = str(share.get("visibility") or "public")
    if share_visibility != visibility:
        raise WatchError("invalid_visibility", "The selected page visibility does not match this page.")
    watch_id = _watch_id()
    now = utcnow()
    agreement = (share.get("differences_data") or {}).get("agreement") or {}
    score = agreement.get("score")
    doc = {
        "owner_uid": uid,
        "share_id": share_id,
        # Denormalized so SEO/admin capacity checks do not have to fetch every
        # referenced share. The share remains authoritative for all mutations.
        "publication_source": str(share.get("publication_source") or "")[:40],
        "question_hash": share.get("question_hash") or share_snapshots.question_hash(share.get("question")),
        "interval": interval,
        "model_tier": normalized_model_tier,
        "run_weekday": run_weekday,
        "run_time": run_time,
        "timezone": timezone_name,
        "email_mode": email_mode,
        "email_enabled": email_enabled,
        "telegram_enabled": telegram_enabled,
        "telegram_muted_until": None,
        "condition": condition,
        "last_condition_status": None,
        "last_condition_hash": None,
        "visibility": visibility,
        "status": "active",
        "next_run_at": next_scheduled_run(
            interval, run_time, timezone_name, run_weekday, now=now,
        ),
        "claimed_until": None,
        "consecutive_failures": 0,
        "created_at": now,
        "last_run_at": None,
        "last_agreement_score": score if isinstance(score, (int, float)) else None,
        "last_successful_run_id": "",
        "last_trigger": "stable",
        "last_change_summary": "",
        "last_drift_score": None,
        "last_event_type": "",
        "query_first": created_query_share,
    }
    question_hash = str(doc.get("question_hash") or "")
    uniqueness_key = (
        f"question:{question_hash}" if created_query_share else f"share:{share_id}"
    )
    include_publisher = normalized_model_tier == "free"
    _ensure_watch_indexes(
        uid,
        uniqueness_key,
        db=db,
        include_publisher=include_publisher,
    )
    watch_ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    owner_ref = _owner_state_ref(db, uid)
    unique_ref = _unique_ref(db, uid, uniqueness_key)
    publisher_ref = _publisher_counter_ref(db)
    query_share_ref = (
        db.collection(share_snapshots.SHARES_COLLECTION).document(share_id)
        if query_share_doc is not None else None
    )

    def create(transaction):
        persistence_guard.ensure_account_write_allowed(
            uid=uid, db=db, transaction=transaction
        )
        owner_snapshot = owner_ref.get(transaction=transaction)
        unique_snapshot = unique_ref.get(transaction=transaction)
        existing_ref = None
        existing_snapshot = None
        unique_data = unique_snapshot.to_dict() if unique_snapshot.exists else {}
        existing_id = str((unique_data or {}).get("watch_id") or "")
        if existing_id:
            existing_ref = db.collection(WATCHES_COLLECTION).document(existing_id)
            existing_snapshot = existing_ref.get(transaction=transaction)
        publisher_snapshot = (
            publisher_ref.get(transaction=transaction) if include_publisher else None
        )

        if existing_snapshot is not None and existing_snapshot.exists:
            existing_data = existing_snapshot.to_dict() or {}
            if return_existing and str(existing_data.get("share_id") or "") == share_id:
                if normalized_model_tier == "free":
                    managed_updates = {
                        "model_tier": "free",
                        "publication_source": str(
                            share.get("publication_source") or ""
                        )[:40],
                        "interval": "weekly",
                        "run_weekday": run_weekday,
                        "run_time": run_time,
                        "timezone": timezone_name,
                    }
                    if any(
                        existing_data.get(key) != value
                        for key, value in managed_updates.items()
                    ):
                        managed_updates["next_run_at"] = next_scheduled_run(
                            "weekly", run_time, timezone_name, run_weekday, now=utcnow()
                        )
                        transaction.update(existing_ref, managed_updates)
                        existing_data.update(managed_updates)
                return existing_id, existing_data
            raise WatchError("already_exists", "This question is already watched.")

        owner_state = owner_snapshot.to_dict() if owner_snapshot.exists else {}
        active_count = _safe_count((owner_state or {}).get("active_count"))
        if not bypass_active_limit and active_count >= cfg.get_watch_active_limit(tier):
            raise WatchError("limit_reached", "Active watch limit reached.")
        publisher_count = 0
        if include_publisher:
            publisher_state = (
                publisher_snapshot.to_dict() if publisher_snapshot and publisher_snapshot.exists else {}
            )
            publisher_count = _safe_count((publisher_state or {}).get("active_count"))
            if (
                publisher_active_limit is not None
                and publisher_count >= int(publisher_active_limit)
            ):
                raise WatchError(
                    "publisher_capacity", "Active Publisher Watch limit reached."
                )

        if query_share_ref is not None:
            transaction.set(query_share_ref, query_share_doc)
        transaction.set(watch_ref, doc)
        transaction.set(unique_ref, {
            "watch_id": watch_id,
            "share_id": share_id,
            "uniqueness_key": uniqueness_key,
        })
        transaction.set(owner_ref, {
            "schema_version": 1,
            "active_count": active_count + 1,
            "updated_at": now,
        })
        if include_publisher:
            transaction.set(publisher_ref, {
                "schema_version": 1,
                "active_count": publisher_count + 1,
                "updated_at": now,
            })
        return watch_id, doc

    stored_watch_id, stored_doc = _run_transaction(db, create)
    return _serialize_watch(stored_watch_id, stored_doc, share)


def serialize_history_points(points, max_items=WATCH_HISTORY_POINTS) -> list[dict]:
    """Compact, JSON-safe view of the newest history points (ascending).

    The trigger is recomputed from the full series here instead of read from
    the stored field: checks written before the drift rule tightened carry a
    trigger that marked every wording difference as movement.
    """
    serialized = []
    for point in drift_signal.annotate_points(points)[-max_items:]:
        ts = point.get("ts")
        serialized.append({
            "run_id": str(point.get("run_id") or ""),
            "ts": ts.isoformat() if isinstance(ts, datetime) else "",
            "agreement_score": point.get("agreement_score"),
            "changed": bool(point.get("changed")),
            "severity": str(point.get("severity") or ""),
            "change_summary": str(point.get("change_summary") or ""),
            "trigger": point.get("trigger") if point.get("trigger") in {"changed", "stable"} else "stable",
            "restated": bool(point.get("restated")),
            "event_type": str(point.get("event_type") or ""),
            "baseline_changed": bool(point.get("baseline_changed")),
            "baseline_severity": str(point.get("baseline_severity") or ""),
            "baseline_summary": str(point.get("baseline_summary") or ""),
            "has_snapshot": bool(point.get("has_snapshot")),
            "opinion_map": opinion_map.sanitize_opinion_map(point.get("opinion_map")),
        })
    return serialized


def list_watches(uid: str, db=None, include_history=False) -> list[dict]:
    db = db if db is not None else db_firestore
    watch_docs = list(
        _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream()
    )
    share_ids = [str((doc.to_dict() or {}).get("share_id") or "") for doc in watch_docs]
    shares = _get_shares_by_id(db, share_ids)
    items = []
    for doc in watch_docs:
        data = doc.to_dict() or {}
        share_id = str(data.get("share_id") or "")
        share = shares.get(share_id) or {}
        item = _serialize_watch(doc.id, data, share)
        if include_history:
            points = data.get("history_points")
            history_status = "ok"
            if not isinstance(points, list):
                try:
                    points = share_snapshots.list_watch_history(
                        share_id, db=db, max_items=WATCH_HISTORY_POINTS,
                    )
                    # One-time lazy backfill: subsequent dashboard loads no
                    # longer issue a history query per Watch.
                    doc.reference.update({"history_points": points[-WATCH_HISTORY_POINTS:]})
                except Exception:
                    logging.warning("Watch history unavailable for watch_id=%s", doc.id)
                    points = []
                    history_status = "unavailable"
            item["history"] = serialize_history_points(points)
            item["history_status"] = history_status
            if item["history"]:
                latest_map = item["history"][-1].get("opinion_map") or {}
                item["last_drift_score"] = latest_map.get("shift_score")
        items.append(item)
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


def _get_shares_by_id(db, share_ids: list[str]) -> dict[str, dict]:
    unique_ids = list(dict.fromkeys(share_id for share_id in share_ids if share_id))
    refs = [db.collection(share_snapshots.SHARES_COLLECTION).document(share_id) for share_id in unique_ids]
    if not refs:
        return {}
    if hasattr(db, "get_all"):
        snapshots = list(db.get_all(refs))
        return {
            snapshot.id: (snapshot.to_dict() or {})
            for snapshot in snapshots
            if snapshot.exists
        }
    else:
        snapshots = [ref.get() for ref in refs]
        return {
            ref.id: (snapshot.to_dict() or {})
            for ref, snapshot in zip(refs, snapshots)
            if snapshot.exists
        }


def list_watches_for_admin_page(*, db=None, cursor="", max_items=100) -> dict:
    """Return one bounded operational page with batched Share reads."""
    db = db if db is not None else db_firestore
    max_items = max(1, min(200, int(max_items)))
    collection = db.collection(WATCHES_COLLECTION)
    try:
        query = collection.order_by(
            "created_at", direction=firestore.Query.DESCENDING
        )
        if cursor:
            cursor_snapshot = collection.document(str(cursor)).get()
            if cursor_snapshot.exists:
                query = query.start_after(cursor_snapshot)
        docs = list(query.limit(max_items + 1).stream())
    except (AttributeError, TypeError):
        # Compatibility for local unit doubles; production Firestore always
        # executes the bounded ordered query above.
        docs = list(collection.stream())
        docs.sort(
            key=lambda doc: (doc.to_dict() or {}).get("created_at")
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        if cursor:
            ids = [doc.id for doc in docs]
            docs = docs[ids.index(cursor) + 1:] if cursor in ids else docs
        docs = docs[:max_items + 1]
    has_more = len(docs) > max_items
    page = docs[:max_items]
    shares = _get_shares_by_id(
        db, [str((doc.to_dict() or {}).get("share_id") or "") for doc in page]
    )
    items = []
    for doc in page:
        data = doc.to_dict() or {}
        share = shares.get(str(data.get("share_id") or "")) or {}
        item = _serialize_watch(doc.id, data, share)
        item.update({
            "owner_uid": str(data.get("owner_uid") or ""),
            "consecutive_failures": int(data.get("consecutive_failures") or 0),
            "claimed_until": (
                data["claimed_until"].isoformat()
                if isinstance(data.get("claimed_until"), datetime) else ""
            ),
        })
        items.append(item)
    return {
        "items": items,
        "has_more": has_more,
        "next_cursor": page[-1].id if has_more and page else None,
    }


def list_watches_for_admin(db=None) -> list[dict]:
    return list_watches_for_admin_page(db=db, max_items=200)["items"]


def publisher_watch_counts(db=None) -> dict:
    """Count managed Publisher Watches, including pre-lineage records."""
    db = db if db is not None else db_firestore
    active = paused = 0
    # model_tier=free is only written by the admin-only Publisher Watch API.
    # It remains the compatibility marker for Watches created before the
    # explicit publication_source lineage field was introduced.
    docs = _where_equal(
        db.collection(WATCHES_COLLECTION), "model_tier", "free"
    ).stream()
    for doc in docs:
        data = doc.to_dict() or {}
        if data.get("status") == "active":
            active += 1
        elif data.get("status") in {"paused", "paused_error"}:
            paused += 1
    return {"active": active, "paused": paused}


def backfill_publisher_watch_lineage(db=None) -> dict:
    """Backfill explicit lineage when a legacy Watch points to a Publisher run.

    Free-tier Watches are sufficient for capacity counting. Destructive SEO
    safeguards still require verified Publisher lineage on the immutable Share,
    so ambiguous legacy records are deliberately left untouched.
    """
    db = db if db is not None else db_firestore
    checked = updated_watches = updated_shares = 0
    docs = _where_equal(
        db.collection(WATCHES_COLLECTION), "model_tier", "free"
    ).stream()
    for doc in docs:
        checked += 1
        watch = doc.to_dict() or {}
        share_id = str(watch.get("share_id") or "")
        if not share_id:
            continue
        share_ref = db.collection(share_snapshots.SHARES_COLLECTION).document(share_id)
        share_snap = share_ref.get()
        share = share_snap.to_dict() if share_snap.exists else None
        if not share:
            continue
        verified = str(share.get("publication_source") or "") == PUBLISHER_SOURCE
        if not verified:
            run_id = str(share.get("source_api_run_id") or "")
            if run_id:
                run_snap = db.collection(API_RUNS_COLLECTION).document(run_id).get()
                run = run_snap.to_dict() if run_snap.exists else {}
                verified = bool((run.get("request") or {}).get("publisher_mode"))
        if not verified:
            continue
        if str(watch.get("publication_source") or "") != PUBLISHER_SOURCE:
            doc.reference.update({"publication_source": PUBLISHER_SOURCE})
            updated_watches += 1
        if str(share.get("publication_source") or "") != PUBLISHER_SOURCE:
            share_ref.update({"publication_source": PUBLISHER_SOURCE})
            share_snapshots.invalidate_share_cache(share_id)
            updated_shares += 1
    return {
        "checked": checked,
        "updated_watches": updated_watches,
        "updated_shares": updated_shares,
    }


def find_watch_for_share(share_id: str, db=None, *, share=None) -> dict | None:
    db = db if db is not None else db_firestore
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "share_id", share_id).stream():
        data = doc.to_dict() or {}
        resolved_share = (
            share if isinstance(share, dict)
            else share_snapshots.get_share(share_id, db=db) or {}
        )
        return {
            **_serialize_watch(doc.id, data, resolved_share),
            "owner_uid": data.get("owner_uid") or "",
        }
    return None


def set_watch_status_admin(watch_id: str, status: str, *, db=None) -> dict:
    """Admin-only service primitive; deliberately changes no share/index fields."""
    db = db if db is not None else db_firestore
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data:
        raise WatchError("not_found", "Watch not found.")
    requested = str(status or "").strip().lower()
    if requested not in {"active", "paused"}:
        raise WatchError("invalid_status", "Status must be active or paused.")
    if data.get("status") == requested:
        share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
        return _serialize_watch(watch_id, data, share)
    updates = {"status": requested, "claimed_until": None}
    if requested == "active":
        now = utcnow()
        updates.update(
            next_run_at=next_scheduled_run(
                data.get("interval") or "weekly",
                data.get("run_time") or "",
                data.get("timezone") or "",
                data.get("run_weekday") or "",
                now=now,
            ),
            consecutive_failures=0,
        )
    data = _apply_watch_updates(
        str(data.get("owner_uid") or ""),
        watch_id,
        updates,
        tier=TIER_PRO,
        bypass_active_limit=data.get("model_tier") == "free",
        db=db,
    )
    share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
    return _serialize_watch(watch_id, data, share)


def queue_watch_run(watch_id: str, *, now=None, db=None) -> dict:
    """Make an active watch due for the normal leased scheduler path."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data:
        raise WatchError("not_found", "Watch not found.")
    if data.get("status") != "active":
        raise WatchError("invalid_status", "Only an active watch can be queued.")
    claimed_until = data.get("claimed_until")
    if isinstance(claimed_until, datetime) and claimed_until > now:
        raise WatchError("already_claimed", "This watch is currently running.")
    ref.update({"next_run_at": now})
    data["next_run_at"] = now
    share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
    return _serialize_watch(watch_id, data, share)


def _owned_watch(uid: str, watch_id: str, db):
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data:
        raise WatchError("not_found", "Watch not found.")
    if data.get("owner_uid") != uid:
        raise WatchError("forbidden", "You can only manage your own watches.")
    return ref, data


def _watch_uniqueness_key(data: dict) -> str:
    if data.get("query_first") is True:
        return f"question:{str(data.get('question_hash') or '')}"
    return f"share:{str(data.get('share_id') or '')}"


def _apply_watch_updates(
    uid: str,
    watch_id: str,
    updates: dict,
    *,
    tier,
    bypass_active_limit: bool = False,
    db,
) -> dict:
    initial_ref, initial = _owned_watch(uid, watch_id, db)
    uniqueness_key = _watch_uniqueness_key(initial)
    include_publisher = initial.get("model_tier") == "free"
    _ensure_watch_indexes(
        uid,
        uniqueness_key,
        db=db,
        include_publisher=include_publisher,
    )
    owner_ref = _owner_state_ref(db, uid)
    publisher_ref = _publisher_counter_ref(db)

    def mutate(transaction):
        persistence_guard.ensure_account_write_allowed(
            uid=uid, db=db, transaction=transaction
        )
        watch_snapshot = initial_ref.get(transaction=transaction)
        owner_snapshot = owner_ref.get(transaction=transaction)
        publisher_snapshot = (
            publisher_ref.get(transaction=transaction) if include_publisher else None
        )
        current = watch_snapshot.to_dict() if watch_snapshot.exists else None
        if not current or current.get("owner_uid") != uid:
            raise WatchError("not_found", "Watch not found.")
        old_active = current.get("status") == "active"
        new_active = updates.get("status", current.get("status")) == "active"
        owner_state = owner_snapshot.to_dict() if owner_snapshot.exists else {}
        active_count = _safe_count((owner_state or {}).get("active_count"))
        if (
            new_active
            and not old_active
            and not bypass_active_limit
            and active_count >= cfg.get_watch_active_limit(tier)
        ):
            raise WatchError("limit_reached", "Active watch limit reached.")
        delta = int(new_active) - int(old_active)
        transaction.update(initial_ref, updates)
        if delta:
            transaction.set(owner_ref, {
                "schema_version": 1,
                "active_count": max(0, active_count + delta),
                "updated_at": utcnow(),
            })
            if include_publisher:
                publisher_state = (
                    publisher_snapshot.to_dict()
                    if publisher_snapshot and publisher_snapshot.exists else {}
                )
                publisher_count = _safe_count(
                    (publisher_state or {}).get("active_count")
                )
                transaction.set(publisher_ref, {
                    "schema_version": 1,
                    "active_count": max(0, publisher_count + delta),
                    "updated_at": utcnow(),
                })
        return {**current, **updates}

    return _run_transaction(db, mutate)


def update_watch(uid: str, watch_id: str, changes: dict, tier, db=None) -> dict:
    db = db if db is not None else db_firestore
    ref, data = _owned_watch(uid, watch_id, db)
    allowed_changes = {
        "interval", "status", "email_mode", "email_enabled", "telegram_enabled",
        "condition", "run_weekday", "run_time", "timezone",
    }
    if not changes or any(key not in allowed_changes for key in changes):
        raise WatchError(
            "invalid_request",
            "Only interval, status, alert rule, channels, condition, run day, run time, and timezone can be changed.",
        )
    if data.get("model_tier") == "free" and any(
        key in changes for key in {"interval", "run_weekday", "run_time", "timezone"}
    ):
        raise WatchError(
            "managed_watch",
            "Scheduled Publisher Watch timing is managed from the Admin Publisher configuration.",
        )
    updates = {}
    now = utcnow()
    effective_interval = data.get("interval") or "weekly"
    if "interval" in changes:
        interval = validate_interval(changes["interval"], tier)
        effective_interval = interval
        updates["interval"] = interval
    schedule_changed = any(
        key in changes for key in {"interval", "run_weekday", "run_time", "timezone"}
    )
    if schedule_changed:
        effective_run_time = changes.get("run_time", data.get("run_time") or "")
        effective_timezone = changes.get("timezone", data.get("timezone") or "")
        effective_run_time, effective_timezone = validate_run_schedule(
            effective_run_time, effective_timezone,
        )
        requested_weekday = changes.get("run_weekday", data.get("run_weekday") or "")
        if effective_interval != "weekly" and "run_weekday" not in changes:
            requested_weekday = ""
        effective_run_weekday = validate_run_weekday(
            requested_weekday, effective_interval, has_run_time=bool(effective_run_time),
        )
        updates.update(
            run_weekday=effective_run_weekday,
            run_time=effective_run_time,
            timezone=effective_timezone,
            next_run_at=next_scheduled_run(
                effective_interval, effective_run_time, effective_timezone,
                effective_run_weekday, now=now,
            ),
        )
    if "email_mode" in changes:
        updates["email_mode"] = validate_email_mode(changes["email_mode"])
    for channel in ("email_enabled", "telegram_enabled"):
        if channel in changes:
            updates[channel] = validate_notification_channel(changes[channel], channel)
    effective_email = updates.get("email_enabled", data.get("email_enabled") is not False)
    effective_telegram = updates.get("telegram_enabled", data.get("telegram_enabled") is True)
    if not effective_email and not effective_telegram:
        raise WatchError("notification_channel_required", "Keep at least one notification channel enabled.")
    if "condition" in changes:
        updates["condition"] = validate_condition(changes["condition"])
        if updates["condition"] != str(data.get("condition") or ""):
            updates["last_condition_status"] = None
            updates["last_condition_hash"] = None
    effective_mode = updates.get("email_mode") or data.get("email_mode") or "changes_only"
    effective_condition = updates.get("condition", str(data.get("condition") or ""))
    if effective_mode == "condition":
        updates["condition"] = validate_condition(effective_condition, required=True)
        if data.get("email_mode") != "condition" and "last_condition_status" not in updates:
            updates["last_condition_status"] = None
            updates["last_condition_hash"] = None
    if "status" in changes:
        status = str(changes["status"] or "").strip().lower()
        if status not in WATCH_STATUSES:
            raise WatchError("invalid_status", "Status must be active or paused.")
        if status == "active" and data.get("status") != "active":
            if data.get("model_tier") != "free":
                _check_active_limit(uid, tier, db, excluding_id=watch_id)
            interval = updates.get("interval") or data.get("interval") or "weekly"
            validate_interval(interval, tier)
            run_time = updates.get("run_time", data.get("run_time") or "")
            timezone_name = updates.get("timezone", data.get("timezone") or "")
            run_weekday = updates.get("run_weekday", data.get("run_weekday") or "")
            updates.update(
                next_run_at=next_scheduled_run(
                    interval, run_time, timezone_name, run_weekday, now=now,
                ),
                consecutive_failures=0,
            )
        updates.update(status=status, claimed_until=None)
    data = _apply_watch_updates(
        uid,
        watch_id,
        updates,
        tier=tier,
        bypass_active_limit=data.get("model_tier") == "free",
        db=db,
    )
    share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
    return _serialize_watch(watch_id, data, share)


def _delete_watch_record(
    watch_id: str,
    *,
    expected_uid: str = "",
    allow_account_deletion: bool = False,
    db=None,
) -> bool:
    db = db if db is not None else db_firestore
    watch_ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    snapshot = watch_ref.get()
    initial = snapshot.to_dict() if snapshot.exists else None
    if not initial:
        return False
    uid = str(initial.get("owner_uid") or "")
    if expected_uid and uid != expected_uid:
        raise WatchError("forbidden", "You can only manage your own watches.")
    uniqueness_key = _watch_uniqueness_key(initial)
    include_publisher = initial.get("model_tier") == "free"
    # Normal mutations lazily backfill legacy indexes. Account-deletion cleanup
    # must never do that: a retry can run after the watch-index area was already
    # acknowledged, and recreating those documents would leave personal data
    # behind a completed cleanup marker.
    if not allow_account_deletion:
        _ensure_watch_indexes(
            uid,
            uniqueness_key,
            db=db,
            include_publisher=include_publisher,
        )
    owner_ref = _owner_state_ref(db, uid)
    unique_ref = _unique_ref(db, uid, uniqueness_key)
    publisher_ref = _publisher_counter_ref(db)
    share_id = str(initial.get("share_id") or "")
    share_ref = db.collection(share_snapshots.SHARES_COLLECTION).document(share_id)

    def remove(transaction):
        if not allow_account_deletion:
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=db, transaction=transaction
            )
        watch_snapshot = watch_ref.get(transaction=transaction)
        owner_snapshot = owner_ref.get(transaction=transaction)
        unique_snapshot = unique_ref.get(transaction=transaction)
        publisher_snapshot = (
            publisher_ref.get(transaction=transaction) if include_publisher else None
        )
        current = watch_snapshot.to_dict() if watch_snapshot.exists else None
        if not current:
            return False, False
        if expected_uid and current.get("owner_uid") != expected_uid:
            raise WatchError("forbidden", "You can only manage your own watches.")
        revoke_query_share = bool(
            current.get("query_first")
            and not current.get("last_successful_run_id")
        )
        share_snapshot = (
            share_ref.get(transaction=transaction) if revoke_query_share else None
        )
        owner_state = owner_snapshot.to_dict() if owner_snapshot.exists else {}
        active_count = _safe_count((owner_state or {}).get("active_count"))
        was_active = current.get("status") == "active"
        transaction.delete(watch_ref)
        unique_data = unique_snapshot.to_dict() if unique_snapshot.exists else {}
        if str((unique_data or {}).get("watch_id") or "") == watch_id:
            transaction.delete(unique_ref)
        if was_active:
            if owner_snapshot.exists:
                transaction.set(owner_ref, {
                    "schema_version": 1,
                    "active_count": max(0, active_count - 1),
                    "updated_at": utcnow(),
                })
            if (
                include_publisher
                and publisher_snapshot is not None
                and publisher_snapshot.exists
            ):
                publisher_state = publisher_snapshot.to_dict()
                publisher_count = _safe_count(
                    (publisher_state or {}).get("active_count")
                )
                transaction.set(publisher_ref, {
                    "schema_version": 1,
                    "active_count": max(0, publisher_count - 1),
                    "updated_at": utcnow(),
                })
        revoked = False
        if share_snapshot is not None and share_snapshot.exists:
            share = share_snapshot.to_dict() or {}
            if share.get("awaiting_first_watch_run"):
                transaction.update(share_ref, {"status": "revoked", "indexed": False})
                revoked = True
        return True, revoked

    deleted, revoked = _run_transaction(db, remove)
    if revoked:
        share_snapshots.invalidate_share_cache(share_id)
    return deleted


def delete_watch(uid: str, watch_id: str, db=None):
    db = db if db is not None else db_firestore
    if not _delete_watch_record(watch_id, expected_uid=uid, db=db):
        raise WatchError("not_found", "Watch not found.")


def pause_watch(uid: str, watch_id: str, db=None) -> dict:
    """Pause an owned watch from a signed-in surface or Telegram callback."""
    db = db if db is not None else db_firestore
    data = _apply_watch_updates(
        uid,
        watch_id,
        {"status": "paused", "claimed_until": None, "current_run_id": None},
        tier=TIER_PRO,
        db=db,
    )
    share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
    return _serialize_watch(watch_id, data, share)


def _unsubscribe_secret() -> bytes:
    secret = os.environ.get("WATCH_UNSUBSCRIBE_SECRET", "").strip()
    if not secret:
        raise RuntimeError("WATCH_UNSUBSCRIBE_SECRET is not configured")
    return secret.encode("utf-8")


def sign_token_payload(payload: dict, *, now=None, max_age_days=UNSUBSCRIBE_MAX_AGE_DAYS) -> str:
    """HMAC-signed, URL-safe token around a small JSON payload (adds ``exp``)."""
    now = now or utcnow()
    payload = dict(payload)
    payload["exp"] = int((now + timedelta(days=max_age_days)).timestamp())
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=")
    signature = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
    return (encoded + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode("ascii")


def parse_token_payload(token: str, *, now=None) -> dict:
    try:
        encoded, signature = str(token or "").encode("ascii").split(b".", 1)
        expected = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + b"=" * (-len(signature) % 4))
        if not hmac.compare_digest(actual, expected):
            raise ValueError("bad signature")
        payload = json.loads(base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4)))
        if int(payload["exp"]) < int((now or utcnow()).timestamp()):
            raise WatchError("expired_token", "This link has expired.")
        return payload
    except WatchError:
        raise
    except Exception as exc:
        raise WatchError("invalid_token", "This link is invalid.") from exc


def make_unsubscribe_token(watch_id: str, *, now=None, max_age_days=UNSUBSCRIBE_MAX_AGE_DAYS) -> str:
    return sign_token_payload({"wid": watch_id}, now=now, max_age_days=max_age_days)


def parse_unsubscribe_token(token: str, *, now=None) -> str:
    payload = parse_token_payload(token, now=now)
    return str(payload.get("wid") or "")


def unsubscribe(token: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    watch_id = parse_unsubscribe_token(token)
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data:
        raise WatchError("not_found", "This watch no longer exists.")
    data = _apply_watch_updates(
        str(data.get("owner_uid") or ""),
        watch_id,
        {"status": "paused", "claimed_until": None, "current_run_id": None},
        tier=TIER_PRO,
        db=db,
    )
    return {"watch_id": watch_id, "question": str(data.get("question") or "")[:200]}


def delete_watches_for_share(
    share_id: str, db=None, *, allow_account_deletion: bool = False
) -> int:
    db = db if db is not None else db_firestore
    deleted = 0
    docs = list(
        _where_equal(db.collection(WATCHES_COLLECTION), "share_id", share_id).stream()
    )
    for doc in docs:
        if _delete_watch_record(
            doc.id, db=db, allow_account_deletion=allow_account_deletion
        ):
            deleted += 1
    return deleted


def delete_watches_for_owner(uid: str, db=None) -> int:
    db = db if db is not None else db_firestore
    docs = list(
        _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream()
    )
    deleted = 0
    for doc in docs:
        if _delete_watch_record(
            doc.id,
            expected_uid=uid,
            allow_account_deletion=True,
            db=db,
        ):
            deleted += 1
    return deleted


def get_public_watch_meta(share_id: str, db=None) -> dict | None:
    """Public, text-free status metadata for a share's current watch."""
    db = db if db is not None else db_firestore
    candidates = []
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "share_id", share_id).stream():
        data = doc.to_dict() or {}
        if data.get("status") not in {"active", "paused", "paused_error"}:
            continue
        candidates.append(data)
    if not candidates:
        return None
    candidates.sort(
        key=lambda data: data.get("created_at") if isinstance(data.get("created_at"), datetime) else datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    data = candidates[0]
    return {
        "status": data.get("status") or "paused",
        "interval": data.get("interval") or "weekly",
        "run_weekday": str(data.get("run_weekday") or ""),
        "run_time": str(data.get("run_time") or ""),
        "timezone": str(data.get("timezone") or ""),
        "last_run_at": data.get("last_run_at"),
        "next_run_at": data.get("next_run_at"),
        "created_at": data.get("created_at"),
        "last_successful_run_id": str(data.get("last_successful_run_id") or ""),
        "last_trigger": "changed" if data.get("last_trigger") == "changed" else "stable",
        "last_change_summary": str(data.get("last_change_summary") or "")[:400],
        "last_drift_score": data.get("last_drift_score"),
        "last_event_type": str(data.get("last_event_type") or ""),
    }


def _claim_in_transaction(tx, watch_ref, budget_ref, now: datetime, daily_limit: int):
    """Pure transaction body, kept directly testable with the Firestore seam."""
    watch_snap = watch_ref.get(transaction=tx)
    data = watch_snap.to_dict() if watch_snap.exists else None
    if not data or data.get("status") != "active":
        return None, "not_due"
    next_run = data.get("next_run_at")
    claimed_until = data.get("claimed_until")
    if not isinstance(next_run, datetime) or next_run > now:
        return None, "not_due"
    if isinstance(claimed_until, datetime) and claimed_until > now:
        return None, "claimed"

    budget_snap = budget_ref.get(transaction=tx)
    budget = budget_snap.to_dict() if budget_snap.exists else {}
    count = budget.get("count", 0)
    count = count if isinstance(count, int) and count >= 0 else 0
    if count >= daily_limit:
        return None, "budget"

    run_id = secrets.token_hex(12)
    lease = now + timedelta(minutes=WATCH_LEASE_MINUTES)
    tx.update(watch_ref, {"claimed_until": lease, "current_run_id": run_id})
    tx.set(budget_ref, {"date": now.strftime("%Y-%m-%d"), "count": count + 1})
    claimed = dict(data)
    claimed.update({"claimed_until": lease, "current_run_id": run_id})
    return claimed, "claimed"


def claim_watch(watch_id: str, *, now=None, db=None):
    from firebase_admin import firestore

    db = db if db is not None else db_firestore
    now = now or utcnow()
    watch_ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    budget_ref = db.collection(RUNTIME_COLLECTION).document("daily_" + now.strftime("%Y%m%d"))
    tx = db.transaction()

    @firestore.transactional
    def consume(transaction):
        return _claim_in_transaction(transaction, watch_ref, budget_ref, now, cfg.get_watch_max_runs_per_day())

    return consume(tx)


def renew_watch_lease(
    watch_id: str, run_id: str, *, now=None, db=None
) -> bool:
    """Extend a lease only while the caller still owns ``current_run_id``."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)

    def renew(transaction):
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else None
        if (
            not data
            or data.get("status") != "active"
            or str(data.get("current_run_id") or "") != str(run_id or "")
        ):
            return False
        transaction.update(ref, {
            "claimed_until": now + timedelta(minutes=WATCH_LEASE_MINUTES)
        })
        return True

    return _run_transaction(db, renew)


def list_due_watch_ids(*, now=None, db=None, max_items=200) -> list[str]:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    collection = db.collection(WATCHES_COLLECTION)
    try:
        query = (
            collection
            .where(filter=FieldFilter("status", "==", "active"))
            .where(filter=FieldFilter("next_run_at", "<=", now))
        )
    except TypeError:
        query = collection.where("status", "==", "active")
        if not hasattr(query, "where"):
            docs = [
                doc for doc in query.stream()
                if isinstance((doc.to_dict() or {}).get("next_run_at"), datetime)
                and (doc.to_dict() or {})["next_run_at"] <= now
            ]
            docs.sort(key=lambda doc: (doc.to_dict() or {})["next_run_at"])
            return [doc.id for doc in docs[:max_items]]
        query = query.where("next_run_at", "<=", now)
    if not hasattr(query, "order_by"):
        docs = list(query.stream())
        docs.sort(key=lambda doc: (doc.to_dict() or {})["next_run_at"])
        return [doc.id for doc in docs[:max_items]]
    query = query.order_by("next_run_at").limit(max(1, min(500, int(max_items))))
    return [doc.id for doc in query.stream()]


def _worker_lease_transaction(tx, ref, now: datetime):
    snap = ref.get(transaction=tx)
    data = snap.to_dict() if snap.exists else {}
    until = data.get("claimed_until")
    if isinstance(until, datetime) and until > now:
        return False
    tx.set(ref, {"claimed_until": now + timedelta(minutes=WORKER_LEASE_MINUTES)})
    return True


def acquire_worker_lease(*, now=None, db=None) -> bool:
    from firebase_admin import firestore

    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref = db.collection(RUNTIME_COLLECTION).document("global_worker")
    tx = db.transaction()

    @firestore.transactional
    def acquire(transaction):
        return _worker_lease_transaction(transaction, ref, now)

    return acquire(tx)


def release_worker_lease(*, db=None):
    db = db if db is not None else db_firestore
    db.collection(RUNTIME_COLLECTION).document("global_worker").update({"claimed_until": None})


def complete_watch_run(watch_id: str, claimed: dict, result: dict, *, now=None,
                       db=None, defer_condition_status=False):
    """Persist one immutable Watch version, then advance the live pointer."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    interval = claimed.get("interval") if claimed.get("interval") in WATCH_INTERVALS else "weekly"
    previous_score = claimed.get("last_agreement_score")
    previous_points = claimed.get("history_points")
    previous_points = previous_points if isinstance(previous_points, list) else []
    # The band the recent checks held, so a score bouncing between two cap
    # steps is not announced as movement on every bounce.
    previous_scores = drift_signal.recent_scores(previous_points) or (
        [previous_score] if isinstance(previous_score, (int, float)) else []
    )
    trigger = drift_signal.classify(
        result.get("changed"), result.get("severity"),
        result.get("agreement_score"), previous_scores,
    )
    history = {
        "schema_version": 2,
        "ts": now,
        "agreement_score": int(result["agreement_score"]),
        "verdict": str(result.get("verdict") or "")[:80],
        "changed": bool(result.get("changed")),
        "severity": str(result.get("severity") or "minor")[:10],
        "change_summary": str(result.get("change_summary") or "")[:400],
        # Wenige, eindeutige Trigger fuer spaetere Webhooks: ein erfolgreicher
        # Lauf ist entweder stable oder changed. Bedingungen/Fehler bleiben
        # getrennte Delivery-Ereignisse und werden nicht semantisch ausgedeutet.
        "trigger": trigger,
        "event_type": WATCH_EVENT_CHANGED if trigger == "changed" else WATCH_EVENT_CHECKED,
        "baseline_changed": bool(result.get("baseline_changed")),
        "baseline_severity": str(result.get("baseline_severity") or "minor")[:10],
        "baseline_summary": str(result.get("baseline_summary") or "")[:400],
        "previous_run_id": str(claimed.get("last_successful_run_id") or ""),
        "consensus_md": str(result.get("consensus") or "")[:share_snapshots.MAX_CONSENSUS_CHARS],
        "differences_data": share_snapshots.sanitize_differences_data(
            result.get("differences_data")
        ),
        "differences_text": str(result.get("differences_text") or "")[
            :share_snapshots.MAX_DIFFERENCES_TEXT_CHARS
        ],
        "sources": share_snapshots.sanitize_sources(result.get("sources")),
        "included_models": list(result.get("included_models") or [])[:6],
        "consensus_model": str(result.get("consensus_model") or "")[:80],
    }
    position_map = opinion_map.sanitize_opinion_map(result.get("opinion_map"))
    if position_map:
        history["opinion_map"] = position_map
    share_ref = db.collection("shares").document(claimed["share_id"])
    history_ref = share_ref.collection("watch_history").document(claimed["current_run_id"])
    watch_ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    watch_updates = {
        "next_run_at": next_scheduled_run(
            interval, claimed.get("run_time") or "", claimed.get("timezone") or "",
            claimed.get("run_weekday") or "",
            now=now, previous_scheduled=claimed.get("next_run_at"),
        ),
        "claimed_until": None,
        "current_run_id": None,
        "consecutive_failures": 0,
        "last_run_at": now,
        "last_agreement_score": history["agreement_score"],
        "last_successful_run_id": str(claimed["current_run_id"]),
        "last_trigger": trigger,
        "last_drift_score": (
            position_map.get("shift_score") if isinstance(position_map, dict) else None
        ),
        "last_event_type": WATCH_EVENT_CHANGED if trigger == "changed" else WATCH_EVENT_CHECKED,
    }
    if trigger == "changed":
        watch_updates["last_change_summary"] = history["change_summary"]
    condition_status = str(result.get("condition_status") or "unknown")
    if condition_status in {"met", "not_met"} and not defer_condition_status:
        watch_updates["last_condition_status"] = condition_status
        watch_updates["last_condition_hash"] = condition_hash(claimed.get("condition") or "")
    # Frische-Signal für SEO (dateModified/sitemap-lastmod) direkt am Share.
    share_updates = {
        "last_watch_run_at": now,
        "latest_watch_run_id": str(claimed["current_run_id"]),
    }
    if claimed.get("initial_watch_run"):
        # A query-first Watch has no manual Consensus snapshot.  Its first
        # scheduled result becomes the immutable baseline used by all later
        # comparisons and keeps the page useful if the Watch is deleted.
        share_updates.update({
            "consensus_md": history["consensus_md"],
            "differences_data": history["differences_data"],
            "differences_text": history["differences_text"],
            "sources": history["sources"],
            "included_models": history["included_models"],
            "consensus_model": history["consensus_model"],
            "answered_at": now.isoformat(),
            "awaiting_first_watch_run": False,
            "index_eligible": share_snapshots.compute_index_eligible(
                claimed.get("question") or "",
                history["consensus_md"], history["sources"], history["included_models"],
            ),
        })
    def persist(transaction):
        current_snapshot = watch_ref.get(transaction=transaction)
        current = current_snapshot.to_dict() if current_snapshot.exists else None
        if (
            not current
            or str(current.get("current_run_id") or "")
            != str(claimed.get("current_run_id") or "")
        ):
            return False
        existing_points = current.get("history_points")
        existing_points = existing_points if isinstance(existing_points, list) else []
        compact_history = {
            key: value for key, value in history.items()
            if key in {
                "ts", "agreement_score", "changed", "severity", "change_summary",
                "trigger", "event_type", "baseline_changed", "baseline_severity",
                "baseline_summary", "opinion_map",
            }
        }
        updates = dict(watch_updates)
        updates["history_points"] = (existing_points + [compact_history])[-WATCH_HISTORY_POINTS:]
        transaction.set(history_ref, history)
        transaction.update(watch_ref, updates)
        transaction.update(share_ref, share_updates)
        return True

    if not _run_transaction(db, persist):
        return None
    share_snapshots.invalidate_share_cache(claimed["share_id"])
    return history


def set_condition_status(
    watch_id: str,
    status: str,
    condition: str,
    db=None,
    expected_run_id: str = "",
):
    """Persist a known condition state after its transition mail was accepted."""
    if status not in {"met", "not_met"}:
        raise ValueError("invalid condition status")
    db = db if db is not None else db_firestore
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)

    def persist(transaction):
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else None
        if not data:
            return False
        if expected_run_id and str(data.get("last_successful_run_id") or "") != expected_run_id:
            return False
        transaction.update(ref, {
            "last_condition_status": status,
            "last_condition_hash": condition_hash(condition),
            "last_event_type": (
                WATCH_EVENT_CONDITION_MET if status == "met" else WATCH_EVENT_CHECKED
            ),
        })
        return True

    return _run_transaction(db, persist)


def fail_watch_run(watch_id: str, claimed: dict, *, now=None, db=None) -> bool | None:
    """Record no history; pause after the third consecutive failure."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    interval = claimed.get("interval") if claimed.get("interval") in WATCH_INTERVALS else "weekly"
    ref = db.collection(WATCHES_COLLECTION).document(watch_id)
    uid = str(claimed.get("owner_uid") or "")
    include_publisher = claimed.get("model_tier") == "free"
    if uid:
        _ensure_watch_indexes(
            uid,
            _watch_uniqueness_key(claimed),
            db=db,
            include_publisher=include_publisher,
        )
    owner_ref = _owner_state_ref(db, uid) if uid else None
    publisher_ref = _publisher_counter_ref(db)

    def fail(transaction):
        snapshot = ref.get(transaction=transaction)
        owner_snapshot = (
            owner_ref.get(transaction=transaction) if owner_ref is not None else None
        )
        publisher_snapshot = (
            publisher_ref.get(transaction=transaction) if include_publisher else None
        )
        current = snapshot.to_dict() if snapshot.exists else None
        if (
            not current
            or str(current.get("current_run_id") or "")
            != str(claimed.get("current_run_id") or "")
        ):
            return None
        failures = _safe_count(current.get("consecutive_failures")) + 1
        paused = failures >= 3
        transaction.update(ref, {
            "status": "paused_error" if paused else "active",
            "next_run_at": next_scheduled_run(
                interval, claimed.get("run_time") or "", claimed.get("timezone") or "",
                claimed.get("run_weekday") or "",
                now=now, previous_scheduled=claimed.get("next_run_at"),
            ),
            "claimed_until": None,
            "current_run_id": None,
            "consecutive_failures": failures,
            "last_run_at": now,
            "last_event_type": WATCH_EVENT_RUN_FAILED,
        })
        if paused and current.get("status") == "active" and owner_ref is not None:
            owner_state = (
                owner_snapshot.to_dict() if owner_snapshot and owner_snapshot.exists else {}
            )
            transaction.set(owner_ref, {
                "schema_version": 1,
                "active_count": max(
                    0, _safe_count((owner_state or {}).get("active_count")) - 1
                ),
                "updated_at": now,
            })
            if include_publisher:
                publisher_state = (
                    publisher_snapshot.to_dict()
                    if publisher_snapshot and publisher_snapshot.exists else {}
                )
                transaction.set(publisher_ref, {
                    "schema_version": 1,
                    "active_count": max(
                        0,
                        _safe_count((publisher_state or {}).get("active_count")) - 1,
                    ),
                    "updated_at": now,
                })
        return paused

    return _run_transaction(db, fail)
