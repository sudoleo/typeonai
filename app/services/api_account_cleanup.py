"""Fail-closed API account blocks and retryable account-data cleanup."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from firebase_admin import auth
from google.cloud.firestore_v1.base_query import FieldFilter

from app.services import telegram_watch


API_ACCOUNT_BLOCKS_COLLECTION = "api_consensus_account_blocks"
API_IDEMPOTENCY_COLLECTION = "api_consensus_idempotency"
API_KEYS_COLLECTION = "api_consensus_keys"
API_RUNS_COLLECTION = "api_consensus_runs"
API_ACCOUNT_CLEANUP_INTERVAL_SECONDS = 5 * 60


class ApiAccountCleanupError(Exception):
    pass


class ApiAccountInactive(ApiAccountCleanupError):
    pass


class ApiAccountStatusUnavailable(ApiAccountCleanupError):
    pass


class FirestoreApiAccountCleanup:
    def __init__(self, db):
        self._db = db

    def block(self, uid: str, *, reason: str = "account_deleted") -> None:
        """Persist the deny decision before any destructive cleanup starts."""
        uid = _validate_uid(uid)
        now = datetime.now(timezone.utc)
        self._block_ref(uid).set(
            {
                "schema_version": 1,
                "uid": uid,
                "blocked": True,
                "reason": str(reason or "account_deleted")[:80],
                "blocked_at": now,
                "cleanup_pending": True,
                "updated_at": now,
            },
            merge=True,
        )

    def is_blocked(self, uid: str) -> bool:
        uid = _validate_uid(uid)
        snap = self._block_ref(uid).get()
        return bool(snap.exists and (snap.to_dict() or {}).get("blocked") is True)

    def ensure_active(self, uid: str) -> None:
        uid = _validate_uid(uid)
        try:
            if self.is_blocked(uid):
                raise ApiAccountInactive("API account is blocked")
            user = auth.get_user(uid)
        except ApiAccountInactive:
            raise
        except auth.UserNotFoundError:
            raise ApiAccountInactive("API account no longer exists") from None
        except Exception as exc:
            raise ApiAccountStatusUnavailable(
                "API account status is temporarily unavailable"
            ) from exc
        if user.disabled or not user.email_verified:
            raise ApiAccountInactive("API account is disabled or unverified")

    def cleanup_uid(self, uid: str) -> list[str]:
        """Best-effort, idempotent cleanup while the UID remains blocked."""
        uid = _validate_uid(uid)
        errors: list[str] = []

        try:
            user_ref = self._db.collection("users").document(uid)
            for snap in user_ref.collection(API_IDEMPOTENCY_COLLECTION).stream():
                snap.reference.delete()
        except Exception:
            logging.exception("API idempotency cleanup failed for blocked UID %s", uid)
            errors.append(API_IDEMPOTENCY_COLLECTION)

        for collection_name in (API_KEYS_COLLECTION, API_RUNS_COLLECTION):
            try:
                query = self._db.collection(collection_name).where(
                    filter=FieldFilter("uid", "==", uid)
                )
                for snap in query.stream():
                    snap.reference.delete()
            except Exception:
                logging.exception("%s cleanup failed for blocked UID %s", collection_name, uid)
                errors.append(collection_name)

        try:
            telegram_watch.delete_user_data(uid, db=self._db)
        except Exception:
            logging.exception("Telegram cleanup failed for blocked UID %s", uid)
            errors.append("telegram")

        now = datetime.now(timezone.utc)
        self._block_ref(uid).set(
            {
                "cleanup_pending": bool(errors),
                "cleanup_errors": errors,
                "cleanup_checked_at": now,
                "updated_at": now,
            },
            merge=True,
        )
        return errors

    def retry_pending(self) -> int:
        query = self._db.collection(API_ACCOUNT_BLOCKS_COLLECTION).where(
            filter=FieldFilter("cleanup_pending", "==", True)
        )
        completed = 0
        for snap in query.stream():
            try:
                if not self.cleanup_uid(snap.id):
                    completed += 1
                    try:
                        auth.get_user(snap.id)
                        if (snap.to_dict() or {}).get("reason") == "account_deleted":
                            auth.delete_user(snap.id)
                            self.clear_completed_block(snap.id)
                    except auth.UserNotFoundError:
                        self.clear_completed_block(snap.id)
                    except Exception:
                        # Keep the fail-closed tombstone until Firebase status
                        # can be checked reliably on a later maintenance pass.
                        logging.exception(
                            "Blocked API account Auth check failed: %s", snap.id
                        )
                        self._block_ref(snap.id).set(
                            {
                                "cleanup_pending": True,
                                "updated_at": datetime.now(timezone.utc),
                            },
                            merge=True,
                        )
            except Exception:
                logging.exception("Blocked API account cleanup retry failed: %s", snap.id)
        return completed

    async def retry_loop(self) -> None:
        while True:
            await asyncio.sleep(API_ACCOUNT_CLEANUP_INTERVAL_SECONDS)
            await asyncio.to_thread(self.retry_pending)

    def clear_completed_block(self, uid: str) -> bool:
        """Remove the temporary tombstone after cleanup and Auth deletion."""
        uid = _validate_uid(uid)
        ref = self._block_ref(uid)
        snap = ref.get()
        if not snap.exists:
            return False
        if (snap.to_dict() or {}).get("cleanup_pending") is True:
            return False
        ref.delete()
        return True

    def mark_cleanup_pending(self, uid: str) -> None:
        uid = _validate_uid(uid)
        self._block_ref(uid).set(
            {
                "cleanup_pending": True,
                "updated_at": datetime.now(timezone.utc),
            },
            merge=True,
        )

    def _block_ref(self, uid: str):
        return self._db.collection(API_ACCOUNT_BLOCKS_COLLECTION).document(uid)


def _validate_uid(uid: str) -> str:
    value = str(uid or "").strip()
    if not value:
        raise ApiAccountCleanupError("uid must not be empty")
    return value
