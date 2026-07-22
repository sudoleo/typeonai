"""Firestore-backed lifecycle helpers for Consensus Watch."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from google.cloud.firestore_v1.base_query import FieldFilter

import app.core.config as cfg
from app.core.security import db_firestore
from app.services import opinion_map, share_snapshots


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
WATCH_INTERNAL_EXCLUDED_PROVIDERS = {"deepseek"}
WATCH_EVENT_CHECKED = "watch.checked"
WATCH_EVENT_CHANGED = "watch.changed"
WATCH_EVENT_CONDITION_MET = "watch.condition_met"
WATCH_EVENT_RUN_FAILED = "watch.run_failed"
PUBLISHER_SOURCE = "scheduled_publisher"
API_RUNS_COLLECTION = "api_consensus_runs"


def _where_equal(collection, field: str, value):
    """Use the current Firestore filter API with mock-compatible fallback."""
    try:
        return collection.where(filter=FieldFilter(field, "==", value))
    except TypeError:
        return collection.where(field, "==", value)


class WatchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _watch_id() -> str:
    return secrets.token_urlsafe(18).replace("-", "").replace("_", "")[:24]


def validate_interval(interval, is_pro: bool) -> str:
    normalized = str(interval or "").strip().lower()
    if normalized not in WATCH_INTERVALS:
        raise WatchError("invalid_interval", "Interval must be daily, weekly, or monthly.")
    if not is_pro and normalized == "daily":
        raise WatchError("pro_required", "Daily watches require Pro.")
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
    excluded_providers = list(data.get("excluded_providers") or [])
    baseline_agreement = (share.get("differences_data") or {}).get("agreement") or {}
    baseline_agreement_score = baseline_agreement.get("score")
    if data.get("model_tier") == "free" and not excluded_providers:
        # Backward-compatible default for Publisher Watches created before the
        # explicit exclusion field was introduced.
        excluded_providers = ["deepseek"]
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
        "excluded_providers": [
            provider for provider in excluded_providers
            if provider in WATCH_INTERNAL_EXCLUDED_PROVIDERS
        ],
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


def _check_active_limit(uid: str, is_pro: bool, db, *, excluding_id: str | None = None):
    count = 0
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream():
        if doc.id == excluding_id:
            continue
        if (doc.to_dict() or {}).get("status") == "active":
            count += 1
    if count >= cfg.get_watch_active_limit(is_pro):
        raise WatchError("limit_reached", "Active watch limit reached.")


def create_watch(uid: str, *, interval, is_pro: bool, email_mode="changes_only",
                 email_enabled=True, telegram_enabled=False,
                 condition="", visibility="public", run_time="", timezone_name="",
                 run_weekday="",
                 result_id=None,
                 share_id=None, question=None, model_tier="", return_existing=False,
                 bypass_active_limit=False, excluded_providers=None, db=None) -> dict:
    db = db if db is not None else db_firestore
    interval = validate_interval(interval, is_pro)
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
    normalized_excluded = sorted({
        str(provider or "").strip().lower() for provider in (excluded_providers or ())
        if str(provider or "").strip()
    })
    if any(provider not in WATCH_INTERNAL_EXCLUDED_PROVIDERS for provider in normalized_excluded):
        raise WatchError("invalid_provider", "Unsupported Watch provider exclusion.")
    if question is not None and not isinstance(question, str):
        raise WatchError("invalid_question", "Question must be text.")
    sources = [bool(result_id), bool(share_id), bool(str(question or "").strip())]
    if sum(sources) != 1:
        raise WatchError(
            "invalid_request",
            "Provide exactly one of result_id, share_id, or question.",
        )
    created_query_share = False
    if question:
        normalized_question = " ".join(str(question).split()).strip()
        if len(normalized_question) < WATCH_QUESTION_MIN_CHARS:
            raise WatchError("invalid_question", "Enter a complete question for this watch.")
        if len(normalized_question) > share_snapshots.MAX_QUESTION_CHARS:
            raise WatchError(
                "invalid_question",
                f"Question must be at most {share_snapshots.MAX_QUESTION_CHARS} characters.",
            )
        question_hash = share_snapshots.question_hash(normalized_question)
        for existing in _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream():
            if (existing.to_dict() or {}).get("question_hash") == question_hash:
                raise WatchError("already_exists", "This question is already watched.")
        if not bypass_active_limit:
            _check_active_limit(uid, is_pro, db)
        try:
            created = share_snapshots.create_share_for_watch_query(
                uid, normalized_question, db=db, visibility=visibility,
            )
        except share_snapshots.ShareError as exc:
            raise WatchError(exc.code, exc.message) from exc
        share_id = created["share_id"]
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
    share = _owned_active_share(uid, share_id, db)
    share_visibility = str(share.get("visibility") or "public")
    if share_visibility != visibility:
        raise WatchError("invalid_visibility", "The selected page visibility does not match this page.")
    for existing in _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream():
        existing_data = existing.to_dict() or {}
        if existing_data.get("share_id") == share_id:
            if return_existing:
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
                        "excluded_providers": normalized_excluded,
                    }
                    if any(existing_data.get(key) != value for key, value in managed_updates.items()):
                        managed_updates["next_run_at"] = next_scheduled_run(
                            "weekly", run_time, timezone_name, run_weekday, now=utcnow()
                        )
                        existing.reference.update(managed_updates)
                        existing_data.update(managed_updates)
                return _serialize_watch(existing.id, existing_data, share)
            raise WatchError("already_exists", "This consensus is already watched.")
    if not bypass_active_limit and not created_query_share:
        _check_active_limit(uid, is_pro, db)

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
        "excluded_providers": normalized_excluded,
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
    db.collection(WATCHES_COLLECTION).document(watch_id).set(doc)
    return _serialize_watch(watch_id, doc, share)


def serialize_history_points(points, max_items=WATCH_HISTORY_POINTS) -> list[dict]:
    """Compact, JSON-safe view of the newest history points (ascending)."""
    serialized = []
    for point in points[-max_items:]:
        ts = point.get("ts")
        serialized.append({
            "run_id": str(point.get("run_id") or ""),
            "ts": ts.isoformat() if isinstance(ts, datetime) else "",
            "agreement_score": point.get("agreement_score"),
            "changed": bool(point.get("changed")),
            "severity": str(point.get("severity") or ""),
            "change_summary": str(point.get("change_summary") or ""),
            "trigger": "changed" if point.get("trigger") == "changed" else "stable",
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
    items = []
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "owner_uid", uid).stream():
        data = doc.to_dict() or {}
        share_id = str(data.get("share_id") or "")
        share = share_snapshots.get_share(share_id, db=db) or {}
        item = _serialize_watch(doc.id, data, share)
        if include_history:
            try:
                points = share_snapshots.list_watch_history(
                    share_id, db=db, max_items=WATCH_HISTORY_POINTS,
                )
            except Exception:
                points = []
            item["history"] = serialize_history_points(points)
            if item["history"]:
                latest_map = item["history"][-1].get("opinion_map") or {}
                item["last_drift_score"] = latest_map.get("shift_score")
        items.append(item)
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


def list_watches_for_admin(db=None) -> list[dict]:
    """Return operational watch metadata for the admin diagnostics page."""
    db = db if db is not None else db_firestore
    items = []
    for doc in db.collection(WATCHES_COLLECTION).stream():
        data = doc.to_dict() or {}
        share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
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
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


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
    ref.update(updates)
    data.update(updates)
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


def update_watch(uid: str, watch_id: str, changes: dict, is_pro: bool, db=None) -> dict:
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
        interval = validate_interval(changes["interval"], is_pro)
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
                _check_active_limit(uid, is_pro, db, excluding_id=watch_id)
            interval = updates.get("interval") or data.get("interval") or "weekly"
            validate_interval(interval, is_pro)
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
    ref.update(updates)
    data.update(updates)
    share = share_snapshots.get_share(str(data.get("share_id") or ""), db=db) or {}
    return _serialize_watch(watch_id, data, share)


def delete_watch(uid: str, watch_id: str, db=None):
    db = db if db is not None else db_firestore
    ref, data = _owned_watch(uid, watch_id, db)
    ref.delete()
    if data.get("query_first") and not data.get("last_successful_run_id"):
        share_id = str(data.get("share_id") or "")
        share = share_snapshots.get_share(share_id, db=db) or {}
        if share.get("awaiting_first_watch_run"):
            db.collection(share_snapshots.SHARES_COLLECTION).document(share_id).update({
                "status": "revoked",
                "indexed": False,
            })
            share_snapshots.invalidate_share_cache(share_id)


def pause_watch(uid: str, watch_id: str, db=None) -> dict:
    """Pause an owned watch from a signed-in surface or Telegram callback."""
    db = db if db is not None else db_firestore
    ref, data = _owned_watch(uid, watch_id, db)
    ref.update({"status": "paused", "claimed_until": None})
    data.update({"status": "paused", "claimed_until": None})
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
    ref.update({"status": "paused", "claimed_until": None})
    return {"watch_id": watch_id, "question": str(data.get("question") or "")[:200]}


def delete_watches_for_share(share_id: str, db=None) -> int:
    db = db if db is not None else db_firestore
    deleted = 0
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "share_id", share_id).stream():
        doc.reference.delete()
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


def list_due_watch_ids(*, now=None, db=None, max_items=200) -> list[str]:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    result = []
    for doc in _where_equal(db.collection(WATCHES_COLLECTION), "status", "active").stream():
        data = doc.to_dict() or {}
        if isinstance(data.get("next_run_at"), datetime) and data["next_run_at"] <= now:
            result.append(doc.id)
        if len(result) >= max_items:
            break
    return result


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
    try:
        score_delta = abs(float(result["agreement_score"]) - float(previous_score))
    except (TypeError, ValueError):
        score_delta = 0
    trigger = "changed" if bool(result.get("changed")) or score_delta >= 15 else "stable"
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
    # History + Scheduler-Fortschritt atomar: ein Restart kann nie einen
    # sichtbaren Punkt ohne vorgeruecktes next_run_at hinterlassen.
    if hasattr(db, "batch"):
        batch = db.batch()
        batch.set(history_ref, history)
        batch.update(watch_ref, watch_updates)
        batch.update(share_ref, share_updates)
        batch.commit()
    else:  # schlanker Unit-Test-Seam
        history_ref.set(history)
        watch_ref.update(watch_updates)
        share_ref.update(share_updates)
    share_snapshots.invalidate_share_cache(claimed["share_id"])
    return history


def set_condition_status(watch_id: str, status: str, condition: str, db=None):
    """Persist a known condition state after its transition mail was accepted."""
    if status not in {"met", "not_met"}:
        raise ValueError("invalid condition status")
    db = db if db is not None else db_firestore
    db.collection(WATCHES_COLLECTION).document(watch_id).update({
        "last_condition_status": status,
        "last_condition_hash": condition_hash(condition),
        "last_event_type": WATCH_EVENT_CONDITION_MET if status == "met" else WATCH_EVENT_CHECKED,
    })


def fail_watch_run(watch_id: str, claimed: dict, *, now=None, db=None) -> bool:
    """Record no history; pause after the third consecutive failure."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    failures = int(claimed.get("consecutive_failures") or 0) + 1
    interval = claimed.get("interval") if claimed.get("interval") in WATCH_INTERVALS else "weekly"
    paused = failures >= 3
    db.collection(WATCHES_COLLECTION).document(watch_id).update({
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
    return paused
