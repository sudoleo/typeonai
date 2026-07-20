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
WATCH_RUN_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
UNSUBSCRIBE_MAX_AGE_DAYS = 90
WATCH_LEASE_MINUTES = 15
WORKER_LEASE_MINUTES = 29
RUNTIME_COLLECTION = "watch_runtime"
WATCH_HISTORY_POINTS = 16
WATCH_INTERNAL_EXCLUDED_PROVIDERS = {"deepseek"}


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
        "excluded_providers": [
            provider for provider in excluded_providers
            if provider in WATCH_INTERNAL_EXCLUDED_PROVIDERS
        ],
        "run_weekday": str(data.get("run_weekday") or ""),
        "run_time": str(data.get("run_time") or ""),
        "timezone": str(data.get("timezone") or ""),
        "email_mode": data.get("email_mode") or "changes_only",
        "condition": str(data.get("condition") or ""),
        "last_condition_status": data.get("last_condition_status"),
        "visibility": visibility,
        "status": data.get("status") or "paused",
        "next_run_at": iso(data.get("next_run_at")),
        "last_run_at": iso(data.get("last_run_at")),
        "last_agreement_score": data.get("last_agreement_score"),
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
                 condition="", visibility="public", run_time="", timezone_name="",
                 run_weekday="",
                 result_id=None,
                 share_id=None, model_tier="", return_existing=False,
                 bypass_active_limit=False, excluded_providers=None, db=None) -> dict:
    db = db if db is not None else db_firestore
    interval = validate_interval(interval, is_pro)
    email_mode = validate_email_mode(email_mode)
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
    if bool(result_id) == bool(share_id):
        raise WatchError("invalid_request", "Provide exactly one of result_id or share_id.")
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
    if not bypass_active_limit:
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
    }
    db.collection(WATCHES_COLLECTION).document(watch_id).set(doc)
    return _serialize_watch(watch_id, doc, share)


def serialize_history_points(points, max_items=WATCH_HISTORY_POINTS) -> list[dict]:
    """Compact, JSON-safe view of the newest history points (ascending)."""
    serialized = []
    for point in points[-max_items:]:
        ts = point.get("ts")
        serialized.append({
            "ts": ts.isoformat() if isinstance(ts, datetime) else "",
            "agreement_score": point.get("agreement_score"),
            "changed": bool(point.get("changed")),
            "severity": str(point.get("severity") or ""),
            "change_summary": str(point.get("change_summary") or ""),
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
    """Count only watches whose share has explicit scheduled-publisher lineage."""
    db = db if db is not None else db_firestore
    active = paused = 0
    # Publisher watches are the only watches pinned to the internal Free model
    # tier. Query that bounded subset instead of scanning every user watch and
    # issuing one additional share read for each result.
    docs = _where_equal(
        db.collection(WATCHES_COLLECTION), "model_tier", "free"
    ).stream()
    for doc in docs:
        data = doc.to_dict() or {}
        publication_source = str(data.get("publication_source") or "")
        if publication_source != "scheduled_publisher":
            # Compatibility for Publisher watches created before the
            # denormalized lineage field existed. The candidate set is capped
            # by the Publisher Watch limit, rather than all application watches.
            share = share_snapshots.get_share(
                str(data.get("share_id") or ""), db=db
            ) or {}
            publication_source = str(share.get("publication_source") or "")
        if publication_source != "scheduled_publisher":
            continue
        if data.get("status") == "active":
            active += 1
        elif data.get("status") in {"paused", "paused_error"}:
            paused += 1
    return {"active": active, "paused": paused}


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
        "interval", "status", "email_mode", "condition", "run_weekday", "run_time", "timezone",
    }
    if not changes or any(key not in allowed_changes for key in changes):
        raise WatchError(
            "invalid_request",
            "Only interval, status, email_mode, condition, run_weekday, run_time, and timezone can be changed.",
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
    ref, _ = _owned_watch(uid, watch_id, db)
    ref.delete()


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
    """Persist one compact history point, then advance the schedule."""
    db = db if db is not None else db_firestore
    now = now or utcnow()
    interval = claimed.get("interval") if claimed.get("interval") in WATCH_INTERVALS else "weekly"
    history = {
        "ts": now,
        "agreement_score": int(result["agreement_score"]),
        "verdict": str(result.get("verdict") or "")[:80],
        "changed": bool(result.get("changed")),
        "severity": str(result.get("severity") or "minor")[:10],
        "change_summary": str(result.get("change_summary") or "")[:400],
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
    }
    condition_status = str(result.get("condition_status") or "unknown")
    if condition_status in {"met", "not_met"} and not defer_condition_status:
        watch_updates["last_condition_status"] = condition_status
        watch_updates["last_condition_hash"] = condition_hash(claimed.get("condition") or "")
    # Frische-Signal für SEO (dateModified/sitemap-lastmod) direkt am Share.
    share_updates = {"last_watch_run_at": now}
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
    return history


def set_condition_status(watch_id: str, status: str, condition: str, db=None):
    """Persist a known condition state after its transition mail was accepted."""
    if status not in {"met", "not_met"}:
        raise ValueError("invalid condition status")
    db = db if db is not None else db_firestore
    db.collection(WATCHES_COLLECTION).document(watch_id).update({
        "last_condition_status": status,
        "last_condition_hash": condition_hash(condition),
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
    })
    return paused
