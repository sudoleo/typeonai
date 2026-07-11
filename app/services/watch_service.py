"""Firestore-backed lifecycle helpers for Consensus Watch."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone

import app.core.config as cfg
from app.core.security import db_firestore
from app.services import share_snapshots


WATCHES_COLLECTION = "watches"
WATCH_INTERVALS = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=30),
}
WATCH_STATUSES = {"active", "paused"}
UNSUBSCRIBE_MAX_AGE_DAYS = 90


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


def _serialize_watch(watch_id: str, data: dict) -> dict:
    def iso(value):
        return value.isoformat() if isinstance(value, datetime) else ""

    share_id = str(data.get("share_id") or "")
    return {
        "id": watch_id,
        "share_id": share_id,
        "share_path": share_snapshots.share_path(str(data.get("share_slug") or ""), share_id),
        "question": str(data.get("question") or "")[:200],
        "interval": data.get("interval") or "weekly",
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
    for doc in db.collection(WATCHES_COLLECTION).where("owner_uid", "==", uid).stream():
        if doc.id == excluding_id:
            continue
        if (doc.to_dict() or {}).get("status") == "active":
            count += 1
    if count >= cfg.get_watch_active_limit(is_pro):
        raise WatchError("limit_reached", "Active watch limit reached.")


def create_watch(uid: str, *, interval, is_pro: bool, result_id=None, share_id=None, db=None) -> dict:
    db = db if db is not None else db_firestore
    interval = validate_interval(interval, is_pro)
    _check_active_limit(uid, is_pro, db)
    if bool(result_id) == bool(share_id):
        raise WatchError("invalid_request", "Provide exactly one of result_id or share_id.")
    if result_id:
        try:
            created = share_snapshots.create_share_from_pending(uid, str(result_id), db=db)
        except share_snapshots.ShareError as exc:
            raise WatchError(exc.code, exc.message) from exc
        share_id = created["share_id"]

    share_id = str(share_id)
    share = _owned_active_share(uid, share_id, db)
    for existing in db.collection(WATCHES_COLLECTION).where("owner_uid", "==", uid).stream():
        if (existing.to_dict() or {}).get("share_id") == share_id:
            raise WatchError("already_exists", "This consensus is already watched.")

    watch_id = _watch_id()
    now = utcnow()
    agreement = (share.get("differences_data") or {}).get("agreement") or {}
    score = agreement.get("score")
    doc = {
        "owner_uid": uid,
        "share_id": share_id,
        "share_slug": share.get("slug") or "",
        "question": share.get("question") or "",
        "question_hash": share.get("question_hash") or share_snapshots.question_hash(share.get("question")),
        "interval": interval,
        "status": "active",
        "next_run_at": now + WATCH_INTERVALS[interval],
        "claimed_until": None,
        "consecutive_failures": 0,
        "created_at": now,
        "last_run_at": None,
        "last_agreement_score": score if isinstance(score, (int, float)) else None,
    }
    db.collection(WATCHES_COLLECTION).document(watch_id).set(doc)
    return _serialize_watch(watch_id, doc)


def list_watches(uid: str, db=None) -> list[dict]:
    db = db if db is not None else db_firestore
    items = [_serialize_watch(doc.id, doc.to_dict() or {}) for doc in db.collection(WATCHES_COLLECTION).where("owner_uid", "==", uid).stream()]
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


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
    if not changes or any(key not in {"interval", "status"} for key in changes):
        raise WatchError("invalid_request", "Only interval and status can be changed.")
    updates = {}
    if "interval" in changes:
        interval = validate_interval(changes["interval"], is_pro)
        updates.update(interval=interval, next_run_at=utcnow() + WATCH_INTERVALS[interval])
    if "status" in changes:
        status = str(changes["status"] or "").strip().lower()
        if status not in WATCH_STATUSES:
            raise WatchError("invalid_status", "Status must be active or paused.")
        if status == "active" and data.get("status") != "active":
            _check_active_limit(uid, is_pro, db, excluding_id=watch_id)
            interval = updates.get("interval") or data.get("interval") or "weekly"
            validate_interval(interval, is_pro)
            updates.update(next_run_at=utcnow() + WATCH_INTERVALS[interval], consecutive_failures=0)
        updates.update(status=status, claimed_until=None)
    ref.update(updates)
    data.update(updates)
    return _serialize_watch(watch_id, data)


def delete_watch(uid: str, watch_id: str, db=None):
    db = db if db is not None else db_firestore
    ref, _ = _owned_watch(uid, watch_id, db)
    ref.delete()


def _unsubscribe_secret() -> bytes:
    secret = os.environ.get("WATCH_UNSUBSCRIBE_SECRET", "").strip()
    if not secret:
        raise RuntimeError("WATCH_UNSUBSCRIBE_SECRET is not configured")
    return secret.encode("utf-8")


def make_unsubscribe_token(watch_id: str, *, now=None, max_age_days=UNSUBSCRIBE_MAX_AGE_DAYS) -> str:
    now = now or utcnow()
    payload = {"wid": watch_id, "exp": int((now + timedelta(days=max_age_days)).timestamp())}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=")
    signature = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
    return (encoded + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")).decode("ascii")


def parse_unsubscribe_token(token: str, *, now=None) -> str:
    try:
        encoded, signature = str(token or "").encode("ascii").split(b".", 1)
        expected = hmac.new(_unsubscribe_secret(), encoded, hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + b"=" * (-len(signature) % 4))
        if not hmac.compare_digest(actual, expected):
            raise ValueError("bad signature")
        payload = json.loads(base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4)))
        if int(payload["exp"]) < int((now or utcnow()).timestamp()):
            raise WatchError("expired_token", "This unsubscribe link has expired.")
        return str(payload["wid"])
    except WatchError:
        raise
    except Exception as exc:
        raise WatchError("invalid_token", "This unsubscribe link is invalid.") from exc


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
    for doc in db.collection(WATCHES_COLLECTION).where("share_id", "==", share_id).stream():
        doc.reference.delete()
        deleted += 1
    return deleted
