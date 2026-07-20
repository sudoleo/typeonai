"""User-level Morning Brief for Consensus Watch.

One opt-in daily digest e-mail per user that summarizes all of their watches
(current agreement score, changes since the last brief, upcoming runs). Pure
aggregation of already-persisted watch/history data — never triggers LLM runs,
so it is available on every tier.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.core.security import db_firestore
from app.services import watch_service
from app.services.watch_service import WatchError


BRIEFS_COLLECTION = "watch_briefs"
BRIEF_MODES = {"always", "changes_only"}
DEFAULT_SEND_TIME = "07:00"
# Fallback window for the very first brief when no baseline exists yet.
FIRST_BRIEF_WINDOW = timedelta(days=1)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def validate_mode(value) -> str:
    normalized = str(value or "always").strip().lower()
    if normalized not in BRIEF_MODES:
        raise WatchError("invalid_mode", "Brief mode must be always or changes_only.")
    return normalized


def next_brief_send_at(send_time: str, timezone_name: str, *, now: datetime) -> datetime:
    """Next strictly-future occurrence of the local HH:MM, DST-safe."""
    send_time, timezone_name = watch_service.validate_run_schedule(send_time, timezone_name)
    if not send_time:
        raise WatchError("invalid_run_time", "A send time is required for the brief.")
    zone = ZoneInfo(timezone_name)
    hour, minute = (int(part) for part in send_time.split(":"))
    candidate_date = now.astimezone(zone).date()
    while True:
        candidate = datetime.combine(candidate_date, time(hour, minute), tzinfo=zone).astimezone(timezone.utc)
        if candidate > now:
            return candidate
        candidate_date += timedelta(days=1)


def _serialize_brief(data: dict | None) -> dict:
    def iso(value):
        return value.isoformat() if isinstance(value, datetime) else ""

    data = data or {}
    return {
        "enabled": bool(data.get("enabled")),
        "send_time": str(data.get("send_time") or DEFAULT_SEND_TIME),
        "timezone": str(data.get("timezone") or ""),
        "mode": data.get("mode") if data.get("mode") in BRIEF_MODES else "always",
        "next_send_at": iso(data.get("next_send_at")),
        "last_sent_at": iso(data.get("last_sent_at")),
    }


def get_brief(uid: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    snap = db.collection(BRIEFS_COLLECTION).document(uid).get()
    return _serialize_brief(snap.to_dict() if snap.exists else None)


def has_watches(uid: str, db=None) -> bool:
    db = db if db is not None else db_firestore
    for _doc in watch_service._where_equal(
        db.collection(watch_service.WATCHES_COLLECTION), "owner_uid", uid
    ).stream():
        return True
    return False


def disable_if_no_watches(uid: str, db=None) -> bool:
    """Disable a stale brief after its owner's final watch is removed."""
    db = db if db is not None else db_firestore
    if has_watches(uid, db=db):
        return False
    ref = db.collection(BRIEFS_COLLECTION).document(uid)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data or not data.get("enabled"):
        return False
    ref.update({"enabled": False, "next_send_at": None, "updated_at": utcnow()})
    return True


def update_brief(uid: str, changes: dict, db=None) -> dict:
    """Owner-scoped upsert of the brief settings document."""
    db = db if db is not None else db_firestore
    allowed = {"enabled", "send_time", "timezone", "mode"}
    if not changes or any(key not in allowed for key in changes):
        raise WatchError(
            "invalid_request",
            "Only enabled, send_time, timezone, and mode can be changed.",
        )
    ref = db.collection(BRIEFS_COLLECTION).document(uid)
    snap = ref.get()
    data = (snap.to_dict() if snap.exists else None) or {}
    now = utcnow()

    updates: dict = {"updated_at": now}
    if "mode" in changes:
        updates["mode"] = validate_mode(changes["mode"])
    schedule_changed = "send_time" in changes or "timezone" in changes
    send_time = str(changes.get("send_time", data.get("send_time") or DEFAULT_SEND_TIME))
    timezone_name = str(changes.get("timezone", data.get("timezone") or ""))
    if "enabled" in changes:
        updates["enabled"] = bool(changes["enabled"])
    effective_enabled = updates.get("enabled", bool(data.get("enabled")))
    if effective_enabled and not has_watches(uid, db=db):
        raise WatchError(
            "watch_required",
            "Create at least one watch before enabling the Morning Brief.",
        )
    if effective_enabled and (schedule_changed or "enabled" in changes):
        send_time, timezone_name = watch_service.validate_run_schedule(send_time, timezone_name)
        if not send_time or not timezone_name:
            raise WatchError("invalid_timezone", "A send time and timezone are required.")
        updates.update(
            send_time=send_time,
            timezone=timezone_name,
            next_send_at=next_brief_send_at(send_time, timezone_name, now=now),
        )
        if updates.get("enabled") and not data.get("enabled"):
            updates["enabled_at"] = now
    elif schedule_changed:
        send_time, timezone_name = watch_service.validate_run_schedule(send_time, timezone_name)
        updates.update(send_time=send_time, timezone=timezone_name)
    if not data:
        updates.setdefault("created_at", now)
        updates.setdefault("mode", "always")
        updates.setdefault("send_time", DEFAULT_SEND_TIME)
    if snap.exists:
        ref.update(updates)
    else:
        ref.set(updates)
    data.update(updates)
    return _serialize_brief(data)


def list_due_brief_uids(*, now=None, db=None, max_items=200) -> list[str]:
    db = db if db is not None else db_firestore
    now = now or utcnow()
    due = []
    for doc in watch_service._where_equal(
        db.collection(BRIEFS_COLLECTION), "enabled", True
    ).stream():
        data = doc.to_dict() or {}
        if isinstance(data.get("next_send_at"), datetime) and data["next_send_at"] <= now:
            due.append(doc.id)
        if len(due) >= max_items:
            break
    return due


def _claim_in_transaction(tx, ref, now: datetime):
    """Advance the schedule atomically BEFORE sending: at-most-once delivery."""
    snap = ref.get(transaction=tx)
    data = snap.to_dict() if snap.exists else None
    if not data or not data.get("enabled"):
        return None
    next_send = data.get("next_send_at")
    if not isinstance(next_send, datetime) or next_send > now:
        return None
    try:
        advanced = next_brief_send_at(
            str(data.get("send_time") or DEFAULT_SEND_TIME),
            str(data.get("timezone") or ""), now=now,
        )
    except WatchError:
        # Unschedulable settings: disable instead of hot-looping every tick.
        tx.update(ref, {"enabled": False, "next_send_at": None})
        return None
    tx.update(ref, {"next_send_at": advanced, "last_evaluated_at": now})
    baseline = data.get("last_evaluated_at") or data.get("enabled_at")
    claimed = dict(data)
    claimed["baseline"] = baseline if isinstance(baseline, datetime) else now - FIRST_BRIEF_WINDOW
    return claimed


def claim_brief(uid: str, *, now=None, db=None):
    from firebase_admin import firestore

    db = db if db is not None else db_firestore
    now = now or utcnow()
    ref = db.collection(BRIEFS_COLLECTION).document(uid)
    tx = db.transaction()

    @firestore.transactional
    def consume(transaction):
        return _claim_in_transaction(transaction, ref, now)

    return consume(tx)


def mark_brief_sent(uid: str, *, now=None, db=None):
    db = db if db is not None else db_firestore
    db.collection(BRIEFS_COLLECTION).document(uid).update({"last_sent_at": now or utcnow()})


SCORE_EVENT_DELTA = 15


def collect_brief_items(uid: str, *, since: datetime, db=None) -> tuple[list[dict], int]:
    """Digest rows for every watch of the user + count of notable new events.

    Notable = a history point after `since` whose change flag is set or whose
    agreement score moved by >= SCORE_EVENT_DELTA vs its predecessor — the same
    signal the changes_only watch mail uses.
    """
    db = db if db is not None else db_firestore
    watches = watch_service.list_watches(uid, db=db, include_history=True)
    items, changes = [], 0
    for watch in watches:
        history = watch.get("history") or []
        new_points = []
        for index, point in enumerate(history):
            try:
                ts = datetime.fromisoformat(point.get("ts") or "")
            except ValueError:
                continue
            if ts <= since:
                continue
            previous = history[index - 1].get("agreement_score") if index else None
            score = point.get("agreement_score")
            score_event = (
                isinstance(previous, (int, float)) and isinstance(score, (int, float))
                and abs(score - previous) >= SCORE_EVENT_DELTA
            )
            notable = bool(point.get("changed")) or score_event
            changes += 1 if notable else 0
            new_points.append({**point, "notable": notable})
        previous_score = history[-2].get("agreement_score") if len(history) >= 2 else None
        items.append({
            "question": watch.get("question") or "",
            "share_path": watch.get("share_path") or "",
            "status": watch.get("status") or "paused",
            "interval": watch.get("interval") or "weekly",
            "run_weekday": watch.get("run_weekday") or "",
            "run_time": watch.get("run_time") or "",
            "timezone": watch.get("timezone") or "",
            "score": watch.get("last_agreement_score"),
            "previous_score": previous_score,
            "new_points": new_points,
            "next_run_at": watch.get("next_run_at") or "",
            "last_run_at": watch.get("last_run_at") or "",
        })
    return items, changes


def _unsubscribe_secret() -> bytes:
    return watch_service._unsubscribe_secret()


def make_brief_unsubscribe_token(uid: str, *, now=None,
                                 max_age_days=watch_service.UNSUBSCRIBE_MAX_AGE_DAYS) -> str:
    now = now or utcnow()
    payload = {"buid": uid, "exp": int((now + timedelta(days=max_age_days)).timestamp())}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=")
    signature = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
    return (encoded + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode("ascii")


def parse_brief_unsubscribe_token(token: str, *, now=None) -> str:
    try:
        encoded, signature = str(token or "").encode("ascii").split(b".", 1)
        expected = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + b"=" * (-len(signature) % 4))
        if not hmac.compare_digest(actual, expected):
            raise ValueError("bad signature")
        payload = json.loads(base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4)))
        if int(payload["exp"]) < int((now or utcnow()).timestamp()):
            raise WatchError("expired_token", "This unsubscribe link has expired.")
        return str(payload["buid"])
    except WatchError:
        raise
    except Exception as exc:
        raise WatchError("invalid_token", "This unsubscribe link is invalid.") from exc


def unsubscribe_brief(token: str, db=None) -> dict:
    db = db if db is not None else db_firestore
    uid = parse_brief_unsubscribe_token(token)
    ref = db.collection(BRIEFS_COLLECTION).document(uid)
    snap = ref.get()
    if not snap.exists:
        raise WatchError("not_found", "This morning brief no longer exists.")
    ref.update({"enabled": False})
    return {"uid": uid}
