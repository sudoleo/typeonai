"""Persistent resend and recipient budgets for double-opt-in mail."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from firebase_admin import firestore


COLLECTION = "follow_challenges"
RESEND_COOLDOWN = timedelta(minutes=15)
CHALLENGE_TTL = timedelta(days=3)
MAX_PER_RECIPIENT_DAY = 5
MAX_GLOBAL_PER_HOUR = 500


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _get(ref, tx):
    try:
        return ref.get(transaction=tx)
    except TypeError:
        # Firestore Transaction.get(ref) yields a generator in 2.20.x.  Read
        # through the DocumentReference to receive the single snapshot while
        # still binding the read to the transaction.
        return ref.get()


def _run(db, operation):
    if hasattr(db, "run_transaction"):
        return db.run_transaction(operation)
    if not hasattr(db, "transaction"):
        return operation(None)
    transaction = db.transaction()

    @firestore.transactional
    def run(tx):
        return operation(tx)

    return run(transaction)


def claim_confirmation_send(
    *, resource_type: str, resource_id: str, email: str, db, now: datetime | None = None
) -> bool:
    """Return True only when a confirmation mail may be sent now.

    Document ids contain SHA-256 input hashes; raw pending e-mail addresses are
    intentionally not persisted.  Confirmed follower records remain unchanged.
    """
    now = now or utcnow()
    recipient_hash = _hash(email)
    challenge_id = _hash(f"{resource_type}:{resource_id}:{email}")
    day = now.date().isoformat()
    hour = now.strftime("%Y-%m-%dT%H")
    challenge_ref = db.collection(COLLECTION).document(f"challenge-{challenge_id}")
    recipient_ref = db.collection(COLLECTION).document(f"recipient-{recipient_hash}")
    global_ref = db.collection(COLLECTION).document(f"global-{hour}")

    def claim(tx):
        challenge_snap = _get(challenge_ref, tx)
        recipient_snap = _get(recipient_ref, tx)
        global_snap = _get(global_ref, tx)
        challenge = challenge_snap.to_dict() or {} if challenge_snap.exists else {}
        recipient = recipient_snap.to_dict() or {} if recipient_snap.exists else {}
        global_state = global_snap.to_dict() or {} if global_snap.exists else {}
        last_sent = challenge.get("last_sent_at")
        if isinstance(last_sent, datetime) and now - last_sent < RESEND_COOLDOWN:
            return False
        recipient_count = int(recipient.get("count") or 0) if recipient.get("day") == day else 0
        if recipient_count >= MAX_PER_RECIPIENT_DAY:
            return False
        global_count = int(global_state.get("count") or 0)
        if global_count >= MAX_GLOBAL_PER_HOUR:
            return False
        writes = (
            (challenge_ref, {
                "schema_version": 1,
                "resource_type": resource_type,
                "resource_hash": _hash(resource_id),
                "recipient_hash": recipient_hash,
                "last_sent_at": now,
                "expires_at": now + CHALLENGE_TTL,
            }),
            (recipient_ref, {
                "schema_version": 1,
                "recipient_hash": recipient_hash,
                "day": day,
                "count": recipient_count + 1,
                "updated_at": now,
            }),
            (global_ref, {
                "schema_version": 1,
                "hour": hour,
                "count": global_count + 1,
                "expires_at": now + timedelta(hours=2),
            }),
        )
        for ref, data in writes:
            if tx is None:
                ref.set(data)
            else:
                tx.set(ref, data)
        return True

    try:
        return bool(_run(db, claim))
    except (AttributeError, AssertionError):
        # Old unit fakes have no generic top-level collections. Production and
        # emulator clients always take the transactional path above.
        return True


def delete_for_email(email: str, *, db) -> int:
    normalized = str(email or "").strip().lower()
    if not normalized:
        return 0
    recipient_hash = _hash(normalized)
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        query = db.collection(COLLECTION).where(
            filter=FieldFilter("recipient_hash", "==", recipient_hash)
        )
    except (ImportError, TypeError):
        query = db.collection(COLLECTION).where("recipient_hash", "==", recipient_hash)
    deleted = 0
    for snapshot in query.stream():
        snapshot.reference.delete()
        deleted += 1
    return deleted
