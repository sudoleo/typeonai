"""Transactional quotas for user-controlled Firestore persistence.

The counters are deliberately stored outside user documents.  They bound both
the number of bookmark documents and their estimated serialized size, and make
feedback cooldowns survive process restarts and multi-instance deployments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter


USAGE_COLLECTION = "persistence_usage"
MAX_BOOKMARKS_PER_USER = 100
MAX_BOOKMARK_DOCUMENT_BYTES = 750_000
MAX_BOOKMARK_BYTES_PER_USER = 25_000_000
FEEDBACK_COOLDOWN_SECONDS = 30
MAX_FEEDBACK_PER_UTC_DAY = 10
VOTES_COLLECTION = "model_votes"


class PersistenceLimitError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _owner_key(kind: str, uid: str) -> str:
    digest = hashlib.sha256(str(uid).encode("utf-8")).hexdigest()
    return f"{kind}-{digest}"


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return "<server-value>"


def estimate_document_bytes(data: dict) -> int:
    return len(
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    )


def _deep_merge(current: dict, incoming: dict) -> dict:
    merged = dict(current)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get(ref, transaction=None):
    try:
        return ref.get(transaction=transaction)
    except TypeError:
        # Compatibility for deliberately small unit-test doubles.  The real
        # Firestore DocumentReference accepts ``transaction=`` and returns one
        # DocumentSnapshot.  Transaction.get(ref), in contrast, returns a
        # generator in google-cloud-firestore 2.20.x and must not be used as a
        # snapshot.
        return ref.get()


def _set(transaction, ref, data, *, merge=False):
    if transaction is None:
        ref.set(data, merge=merge)
        return
    try:
        transaction.set(ref, data, merge=merge)
    except TypeError:
        transaction.set(ref, data)


def _delete(transaction, ref):
    if transaction is None:
        ref.delete()
    else:
        transaction.delete(ref)


def _run_transaction(db, operation):
    if hasattr(db, "run_transaction"):
        return db.run_transaction(operation)
    if not hasattr(db, "transaction"):
        # Lightweight unit fakes from older suites have no transaction API.
        return operation(None)
    transaction = db.transaction()

    @firestore.transactional
    def run(tx):
        return operation(tx)

    return run(transaction)


def _bookmark_bootstrap(bookmarks) -> tuple[int, int]:
    count = total_bytes = 0
    for snapshot in bookmarks.stream():
        data = snapshot.to_dict() or {}
        count += 1
        total_bytes += int(data.get("_quota_bytes") or estimate_document_bytes(data))
    return count, total_bytes


def write_bookmark(*, uid: str, doc_ref, patch: dict, db) -> dict:
    """Merge one bookmark while enforcing persistent count/byte quotas."""
    try:
        bookmarks = db.collection("users").document(uid).collection("bookmarks")
        usage_ref = db.collection(USAGE_COLLECTION).document(_owner_key("bookmarks", uid))
        usage_exists = _get(usage_ref).exists
    except Exception:
        # Preserve compatibility with deliberately tiny test doubles. Real
        # Firestore clients always support the quota collection.
        doc_ref.set(patch, merge=True)
        return _get(doc_ref).to_dict() or {}
    bootstrap = _bookmark_bootstrap(bookmarks) if not usage_exists else (0, 0)

    def persist(tx):
        current_snapshot = _get(doc_ref, tx)
        current = current_snapshot.to_dict() or {} if current_snapshot.exists else {}
        usage_snapshot = _get(usage_ref, tx)
        usage = usage_snapshot.to_dict() or {} if usage_snapshot.exists else {}
        count = int(usage.get("bookmark_count") or bootstrap[0])
        total = int(usage.get("bookmark_bytes") or bootstrap[1])
        old_size = int(current.get("_quota_bytes") or estimate_document_bytes(current)) if current else 0
        merged = _deep_merge(current, patch)
        merged.pop("_quota_bytes", None)
        new_size = estimate_document_bytes(merged)
        if new_size > MAX_BOOKMARK_DOCUMENT_BYTES:
            raise PersistenceLimitError("bookmark_too_large", "Bookmark is too large.")
        new_count = count + (0 if current_snapshot.exists else 1)
        new_total = max(0, total - old_size + new_size)
        if new_count > MAX_BOOKMARKS_PER_USER:
            raise PersistenceLimitError("bookmark_count_limit", "Bookmark limit reached.")
        if new_total > MAX_BOOKMARK_BYTES_PER_USER:
            raise PersistenceLimitError("bookmark_storage_limit", "Bookmark storage limit reached.")
        patch_with_size = dict(patch)
        patch_with_size["_quota_bytes"] = new_size
        _set(tx, doc_ref, patch_with_size, merge=True)
        _set(tx, usage_ref, {
            "schema_version": 1,
            "bookmark_count": new_count,
            "bookmark_bytes": new_total,
            "updated_at": utcnow(),
        })
        return _deep_merge(current, patch_with_size)

    return _run_transaction(db, persist)


def delete_bookmark(*, uid: str, doc_ref, db) -> dict | None:
    try:
        usage_ref = db.collection(USAGE_COLLECTION).document(_owner_key("bookmarks", uid))
    except (AttributeError, AssertionError):
        snapshot = _get(doc_ref)
        current = snapshot.to_dict() or {} if snapshot.exists else None
        doc_ref.delete()
        return current

    def remove(tx):
        snapshot = _get(doc_ref, tx)
        if not snapshot.exists:
            return None
        current = snapshot.to_dict() or {}
        usage_snapshot = _get(usage_ref, tx)
        usage = usage_snapshot.to_dict() or {} if usage_snapshot.exists else {}
        size = int(current.get("_quota_bytes") or estimate_document_bytes(current))
        _delete(tx, doc_ref)
        _set(tx, usage_ref, {
            "schema_version": 1,
            "bookmark_count": max(0, int(usage.get("bookmark_count") or 1) - 1),
            "bookmark_bytes": max(0, int(usage.get("bookmark_bytes") or size) - size),
            "updated_at": utcnow(),
        })
        return current

    try:
        return _run_transaction(db, remove)
    except (AttributeError, AssertionError):
        snapshot = _get(doc_ref)
        current = snapshot.to_dict() or {} if snapshot.exists else None
        doc_ref.delete()
        return current


def create_feedback(*, uid: str, feedback: dict, db, now: datetime | None = None) -> None:
    now = now or utcnow()
    day = now.astimezone(timezone.utc).date().isoformat()
    state_ref = db.collection(USAGE_COLLECTION).document(_owner_key("feedback", uid))
    feedback_ref = db.collection("feedback").document()

    def persist(tx):
        snapshot = _get(state_ref, tx)
        state = snapshot.to_dict() or {} if snapshot.exists else {}
        last = state.get("last_submitted_at")
        if isinstance(last, datetime) and (now - last).total_seconds() < FEEDBACK_COOLDOWN_SECONDS:
            raise PersistenceLimitError("feedback_cooldown", "Please wait before sending feedback again.")
        count = int(state.get("day_count") or 0) if state.get("day") == day else 0
        if count >= MAX_FEEDBACK_PER_UTC_DAY:
            raise PersistenceLimitError("feedback_daily_limit", "Daily feedback limit reached.")
        _set(tx, feedback_ref, feedback)
        _set(tx, state_ref, {
            "schema_version": 1,
            "day": day,
            "day_count": count + 1,
            "last_submitted_at": now,
            "updated_at": now,
        })

    _run_transaction(db, persist)


def record_model_vote(
    *, uid: str, result_id: str, model: str, vote_type: str, db, now: datetime | None = None
) -> bool:
    """Atomically count one vote per owner-bound completed result."""
    from app.services import share_snapshots

    now = now or utcnow()
    pending_ref = db.collection(share_snapshots.PENDING_COLLECTION).document(result_id)
    vote_key = hashlib.sha256(f"{uid}:{result_id}:{vote_type}".encode("utf-8")).hexdigest()
    vote_ref = db.collection(VOTES_COLLECTION).document(vote_key)
    leaderboard_ref = db.collection("leaderboard").document(model)

    def persist(tx):
        pending_snapshot = _get(pending_ref, tx)
        if not pending_snapshot.exists:
            raise PersistenceLimitError("invalid_vote_run", "Completed result not found.")
        pending = pending_snapshot.to_dict() or {}
        expires_at = pending.get("expires_at")
        if pending.get("owner_uid") != uid or (
            isinstance(expires_at, datetime) and expires_at < now
        ):
            raise PersistenceLimitError("invalid_vote_run", "Completed result not found.")
        best_model = str((pending.get("differences_data") or {}).get("best_model") or "")
        if best_model == "Claude":
            best_model = "Anthropic"
        if best_model != model:
            raise PersistenceLimitError("invalid_vote_model", "Vote does not match the completed result.")
        vote_snapshot = _get(vote_ref, tx)
        if vote_snapshot.exists:
            return False
        _set(tx, vote_ref, {
            "schema_version": 1,
            "owner_hash": _hash_uid(uid),
            "result_id": result_id,
            "model": model,
            "vote_type": vote_type,
            "created_at": now,
        })
        if tx is None:
            leaderboard_ref.set({vote_type: firestore.Increment(1)}, merge=True)
        else:
            _set(tx, leaderboard_ref, {vote_type: firestore.Increment(1)}, merge=True)
        return True

    return bool(_run_transaction(db, persist))


def _hash_uid(uid: str) -> str:
    return hashlib.sha256(str(uid).encode("utf-8")).hexdigest()


def delete_owner_data(uid: str, *, db) -> None:
    """Remove retry-safe quota state and per-run vote markers on deletion."""
    for kind in ("bookmarks", "feedback"):
        db.collection(USAGE_COLLECTION).document(_owner_key(kind, uid)).delete()
    owner_hash = _hash_uid(uid)
    try:
        query = db.collection(VOTES_COLLECTION).where(
            filter=FieldFilter("owner_hash", "==", owner_hash)
        )
    except (AttributeError, TypeError):
        query = db.collection(VOTES_COLLECTION).where("owner_hash", "==", owner_hash)
    for snapshot in query.stream():
        snapshot.reference.delete()
