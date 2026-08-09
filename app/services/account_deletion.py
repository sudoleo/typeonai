"""Persistent, idempotent and retryable full-account deletion jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from firebase_admin import auth, firestore

from app.core.security import invalidate_auth_tombstone_cache
from app.services import share_snapshots
from app.services.api_account_cleanup import FirestoreApiAccountCleanup
from app.services.chat_store import ChatStore


ACCOUNT_DELETION_JOBS_COLLECTION = "account_deletion_jobs"
ACCOUNT_DELETION_RETRY_INTERVAL_SECONDS = 5 * 60
AUTH_TOMBSTONE_RETENTION = timedelta(hours=2)


class AccountDeletionError(Exception):
    pass


def _where_equal(collection, field: str, value):
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter

        return collection.where(filter=FieldFilter(field, "==", value))
    except (ImportError, TypeError):
        return collection.where(field, "==", value)


class FirestoreAccountDeletion:
    """Coordinates every personal-data area behind one durable tombstone."""

    def __init__(self, db):
        self._db = db
        self._api_cleanup = FirestoreApiAccountCleanup(db)

    def start(self, uid: str, *, email: str = "") -> dict:
        uid = self._validate_uid(uid)
        now = datetime.now(timezone.utc)
        ref = self._job_ref(uid)
        snap = ref.get()
        existing = snap.to_dict() if snap.exists else {}
        if existing.get("status") == "completed":
            invalidate_auth_tombstone_cache(uid, blocked=True)
            return existing
        payload = {
            "schema_version": 1,
            "uid": uid,
            "status": "pending",
            "cleanup_pending": True,
            "tombstone_expires_at": now + AUTH_TOMBSTONE_RETENTION,
            "updated_at": now,
        }
        if not existing:
            payload["created_at"] = now
            payload["completed_areas"] = {}
        normalized_email = str(email or "").strip().lower()
        if normalized_email:
            payload["email"] = normalized_email
        ref.set(payload, merge=True)
        invalidate_auth_tombstone_cache(uid, blocked=True)
        return {**existing, **payload}

    def cleanup_uid(self, uid: str) -> list[str]:
        uid = self._validate_uid(uid)
        ref = self._job_ref(uid)
        snap = ref.get()
        if not snap.exists:
            raise AccountDeletionError("Account deletion job does not exist")
        job = snap.to_dict() or {}
        completed = job.get("completed_areas")
        completed = dict(completed) if isinstance(completed, dict) else {}
        email = str(job.get("email") or "").strip().lower()
        errors: list[str] = []

        areas = (
            ("api_access", lambda: self._delete_api_access(uid)),
            ("user_subcollections", lambda: self._delete_user_subcollections(uid)),
            ("chats", lambda: ChatStore(self._db).delete_all_chats(uid)),
            ("waitlist_feedback", lambda: self._delete_uid_queries(uid)),
            ("owned_shares", lambda: self._delete_owned_shares(uid)),
            ("pending_results", lambda: self._delete_query("pending_results", "owner_uid", uid)),
            ("orphan_watches", lambda: self._delete_query("watches", "owner_uid", uid)),
            ("watch_brief", lambda: self._db.collection("watch_briefs").document(uid).delete()),
            ("email_follows", lambda: self._delete_email_follows(email)),
            ("profile", lambda: self._db.collection("users").document(uid).delete()),
            ("firebase_auth", lambda: self._delete_auth_user(uid)),
        )
        for name, operation in areas:
            if completed.get(name) is True:
                continue
            try:
                operation()
                completed[name] = True
                ref.set(
                    {
                        "completed_areas": completed,
                        "last_completed_area": name,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    merge=True,
                )
            except Exception:
                logging.exception("Account deletion area %s failed for UID %s", name, uid)
                errors.append(name)

        now = datetime.now(timezone.utc)
        if errors:
            ref.set(
                {
                    "status": "pending",
                    "cleanup_pending": True,
                    "cleanup_errors": errors,
                    "last_attempt_at": now,
                    "updated_at": now,
                },
                merge=True,
            )
            return errors

        # The minimal UID tombstone remains for longer than a Firebase ID token,
        # but the email needed for follower cleanup is removed immediately.
        ref.set(
            {
                "status": "completed",
                "cleanup_pending": False,
                "cleanup_errors": [],
                "email": firestore.DELETE_FIELD,
                "completed_at": now,
                "tombstone_expires_at": now + AUTH_TOMBSTONE_RETENTION,
                "updated_at": now,
            },
            merge=True,
        )
        try:
            self._api_cleanup.clear_completed_block(uid)
        except Exception:
            logging.exception("Completed API tombstone cleanup failed for UID %s", uid)
            try:
                self._api_cleanup.mark_cleanup_pending(uid)
            except Exception:
                logging.exception(
                    "API tombstone retry marker failed for UID %s", uid
                )
        invalidate_auth_tombstone_cache(uid, blocked=True)
        return []

    def retry_pending(self) -> int:
        completed = 0
        query = _where_equal(
            self._db.collection(ACCOUNT_DELETION_JOBS_COLLECTION),
            "status",
            "pending",
        )
        for snap in query.stream():
            try:
                if not self.cleanup_uid(snap.id):
                    completed += 1
            except Exception:
                logging.exception("Account deletion retry failed for UID %s", snap.id)
        self.delete_expired_tombstones()
        return completed

    def delete_expired_tombstones(self) -> int:
        now = datetime.now(timezone.utc)
        deleted = 0
        query = _where_equal(
            self._db.collection(ACCOUNT_DELETION_JOBS_COLLECTION),
            "status",
            "completed",
        )
        for snap in query.stream():
            data = snap.to_dict() or {}
            expires_at = data.get("tombstone_expires_at")
            if isinstance(expires_at, datetime) and expires_at.tzinfo is not None:
                if expires_at.astimezone(timezone.utc) <= now:
                    snap.reference.delete()
                    invalidate_auth_tombstone_cache(snap.id, blocked=False)
                    deleted += 1
        return deleted

    async def retry_loop(self) -> None:
        while True:
            await asyncio.sleep(ACCOUNT_DELETION_RETRY_INTERVAL_SECONDS)
            await asyncio.to_thread(self.retry_pending)

    def _delete_api_access(self, uid: str) -> None:
        self._api_cleanup.block(uid)
        errors = self._api_cleanup.cleanup_uid(uid)
        if errors:
            raise AccountDeletionError("API cleanup failed: " + ", ".join(errors))

    def _delete_user_subcollections(self, uid: str) -> None:
        user_ref = self._db.collection("users").document(uid)
        for name in ("bookmarks", "counters", "usage_days", "usage_runs"):
            for snap in user_ref.collection(name).stream():
                snap.reference.delete()

    def _delete_uid_queries(self, uid: str) -> None:
        for name in ("pro_waitlist", "feedback"):
            self._delete_query(name, "uid", uid)

    def _delete_owned_shares(self, uid: str) -> None:
        docs = list(
            _where_equal(self._db.collection("shares"), "owner_uid", uid).stream()
        )
        for snap in docs:
            try:
                share_snapshots.hard_delete_share(snap.id, db=self._db)
            except share_snapshots.ShareError as exc:
                if exc.code != "not_found":
                    raise

    def _delete_email_follows(self, email: str) -> None:
        if not email:
            return
        self._delete_query("watch_followers", "email", email)
        topic_followers = list(
            _where_equal(self._db.collection("topic_followers"), "email", email).stream()
        )
        for follower in topic_followers:
            deliveries = _where_equal(
                self._db.collection("topic_follower_deliveries"),
                "follower_id",
                follower.id,
            )
            for delivery in deliveries.stream():
                delivery.reference.delete()
            follower.reference.delete()

    def _delete_query(self, collection: str, field: str, value) -> None:
        query = _where_equal(self._db.collection(collection), field, value)
        for snap in query.stream():
            snap.reference.delete()

    @staticmethod
    def _delete_auth_user(uid: str) -> None:
        try:
            auth.revoke_refresh_tokens(uid)
            auth.delete_user(uid)
        except auth.UserNotFoundError:
            return

    def _job_ref(self, uid: str):
        return self._db.collection(ACCOUNT_DELETION_JOBS_COLLECTION).document(uid)

    @staticmethod
    def _validate_uid(uid: str) -> str:
        normalized = str(uid or "").strip()
        if not normalized:
            raise AccountDeletionError("uid must not be empty")
        return normalized
