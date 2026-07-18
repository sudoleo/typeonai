"""Persistent idempotency and state transitions for asynchronous API runs."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter


API_RUNS_COLLECTION = "api_consensus_runs"
API_IDEMPOTENCY_COLLECTION = "api_consensus_idempotency"
MAX_IDEMPOTENCY_KEY_BYTES = 256
RUN_LEASE_SECONDS = 60 * 60
RUN_STATUSES = {"accepted", "reserved", "running", "succeeded", "failed"}


class ApiRunError(Exception):
    pass


class ApiRunNotFound(ApiRunError):
    pass


class ApiRunConflict(ApiRunError):
    pass


class ApiRunTransitionError(ApiRunError):
    pass


T = TypeVar("T")
TransactionRunner = Callable[[Callable[[object], T]], T]


def idempotency_hash(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Idempotency-Key must not be empty")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise ValueError(
            f"Idempotency-Key must not exceed {MAX_IDEMPOTENCY_KEY_BYTES} bytes"
        )
    return hashlib.sha256(encoded).hexdigest()


def request_hash(request_payload: dict) -> str:
    canonical = json.dumps(
        request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FirestoreApiRunRepository:
    def __init__(self, db, *, transaction_runner: TransactionRunner | None = None):
        self._db = db
        self._transaction_runner = transaction_runner

    def create_or_get(
        self,
        *,
        uid: str,
        api_key_id: str,
        idempotency_key: str,
        request_payload: dict,
        model_plan: dict,
        is_pro: bool,
    ) -> tuple[dict, bool]:
        key_hash = idempotency_hash(idempotency_key)
        payload_hash = request_hash(request_payload)
        run_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        mapping_ref = self._idempotency_ref(uid, key_hash)
        run_ref = self._run_ref(run_id)

        def operation(tx):
            mapping_snap = mapping_ref.get(transaction=tx)
            if mapping_snap.exists:
                mapping = mapping_snap.to_dict() or {}
                if mapping.get("request_hash") != payload_hash:
                    raise ApiRunConflict(
                        "Idempotency-Key is already bound to a different request"
                    )
                existing_ref = self._run_ref(str(mapping.get("run_id") or ""))
                existing_snap = existing_ref.get(transaction=tx)
                if not existing_snap.exists:
                    raise ApiRunConflict("Idempotency mapping is inconsistent")
                return self._with_id(existing_snap), False

            run_data = {
                "schema_version": 1,
                "run_id": run_id,
                "uid": uid,
                "api_key_id": api_key_id,
                "idempotency_hash": key_hash,
                "request_hash": payload_hash,
                "status": "accepted",
                "request": dict(request_payload),
                "model_plan": dict(model_plan),
                "is_pro_at_acceptance": bool(is_pro),
                "accepted_at": now,
                "created_at": now,
                "updated_at": now,
            }
            tx.set(run_ref, run_data)
            tx.set(
                mapping_ref,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "request_hash": payload_hash,
                    "created_at": now,
                },
            )
            return dict(run_data), True

        return self._transaction(operation)

    def get_by_idempotency(
        self, *, uid: str, idempotency_key: str, request_payload: dict
    ) -> dict | None:
        key_hash = idempotency_hash(idempotency_key)
        snap = self._idempotency_ref(uid, key_hash).get()
        if not snap.exists:
            return None
        mapping = snap.to_dict() or {}
        if mapping.get("request_hash") != request_hash(request_payload):
            raise ApiRunConflict(
                "Idempotency-Key is already bound to a different request"
            )
        return self.get_for_uid(str(mapping.get("run_id") or ""), uid)

    def mark_reserved(self, run_id: str) -> tuple[dict, bool]:
        return self._transition(run_id, "accepted", "reserved")

    def claim_running(self, run_id: str, worker_id: str) -> tuple[dict, bool]:
        now = datetime.now(timezone.utc)
        extra = {
            "worker_id": worker_id,
            "lease_expires_at": now + timedelta(seconds=RUN_LEASE_SECONDS),
        }
        return self._transition(run_id, "reserved", "running", extra=extra)

    def succeed(self, run_id: str, result: dict) -> dict:
        run, changed = self._transition(
            run_id, "running", "succeeded", extra={"result": result}
        )
        if not changed and run.get("status") != "succeeded":
            raise ApiRunTransitionError("Run is not running")
        return run

    def fail(self, run_id: str, *, code: str, message: str) -> dict:
        run, changed = self._transition(
            run_id,
            "running",
            "failed",
            extra={"error": {"code": code, "message": message}},
        )
        if not changed and run.get("status") != "failed":
            raise ApiRunTransitionError("Run is not running")
        return run

    def fail_if_lease_expired(
        self, run_id: str, *, now: datetime | None = None
    ) -> bool:
        """Terminally fail stale running work without ever replaying providers."""
        run_ref = self._run_ref(_validate_run_id(run_id))
        check_at = now or datetime.now(timezone.utc)

        def operation(tx):
            snap = run_ref.get(transaction=tx)
            if not snap.exists:
                return False
            data = snap.to_dict() or {}
            lease_expires_at = data.get("lease_expires_at")
            if data.get("status") != "running" or not isinstance(
                lease_expires_at, datetime
            ):
                return False
            if lease_expires_at > check_at:
                return False
            updates = {
                "status": "failed",
                "failed_at": check_at,
                "updated_at": check_at,
                "lease_expires_at": None,
                "error": {
                    "code": "worker_interrupted",
                    "message": "The worker stopped before the run completed.",
                },
            }
            tx.update(run_ref, updates)
            return True

        return self._transaction(operation)

    def delete_accepted(self, run_id: str) -> bool:
        run_ref = self._run_ref(run_id)

        def operation(tx):
            snap = run_ref.get(transaction=tx)
            if not snap.exists:
                return False
            data = snap.to_dict() or {}
            if data.get("status") != "accepted":
                return False
            mapping_ref = self._idempotency_ref(
                str(data.get("uid") or ""), str(data.get("idempotency_hash") or "")
            )
            mapping_snap = mapping_ref.get(transaction=tx)
            if (
                mapping_snap.exists
                and (mapping_snap.to_dict() or {}).get("run_id") == run_id
            ):
                tx.delete(mapping_ref)
            tx.delete(run_ref)
            return True

        return self._transaction(operation)

    def get(self, run_id: str) -> dict:
        snap = self._run_ref(_validate_run_id(run_id)).get()
        if not snap.exists:
            raise ApiRunNotFound("Run not found")
        return self._with_id(snap)

    def get_for_uid(self, run_id: str, uid: str) -> dict:
        run = self.get(run_id)
        if run.get("uid") != uid:
            raise ApiRunNotFound("Run not found")
        return run

    def list_by_status(self, statuses: tuple[str, ...]) -> list[dict]:
        if not statuses:
            return []
        query = self._db.collection(API_RUNS_COLLECTION).where(
            filter=FieldFilter("status", "in", list(statuses))
        )
        return [self._with_id(snap) for snap in query.stream()]

    def _transition(
        self,
        run_id: str,
        expected: str,
        target: str,
        *,
        extra: dict | None = None,
    ) -> tuple[dict, bool]:
        if expected not in RUN_STATUSES or target not in RUN_STATUSES:
            raise ValueError("Unsupported API run status")
        run_ref = self._run_ref(_validate_run_id(run_id))

        def operation(tx):
            snap = run_ref.get(transaction=tx)
            if not snap.exists:
                raise ApiRunNotFound("Run not found")
            data = snap.to_dict() or {}
            status = data.get("status")
            if status == target:
                return self._with_id(snap), False
            if status != expected:
                return self._with_id(snap), False
            now = datetime.now(timezone.utc)
            updates = {
                "status": target,
                "updated_at": now,
                f"{target}_at": now,
            }
            updates.update(extra or {})
            if target in {"succeeded", "failed"}:
                updates["lease_expires_at"] = None
            tx.update(run_ref, updates)
            data.update(updates)
            data["run_id"] = run_ref.id
            return data, True

        return self._transaction(operation)

    def _transaction(self, operation: Callable[[object], T]) -> T:
        if self._transaction_runner is not None:
            return self._transaction_runner(operation)
        transaction = self._db.transaction()

        @firestore.transactional
        def run(tx):
            return operation(tx)

        return run(transaction)

    def _run_ref(self, run_id: str):
        return self._db.collection(API_RUNS_COLLECTION).document(run_id)

    def _idempotency_ref(self, uid: str, key_hash: str):
        return (
            self._db.collection("users")
            .document(uid)
            .collection(API_IDEMPOTENCY_COLLECTION)
            .document(key_hash)
        )

    @staticmethod
    def _with_id(snap) -> dict:
        data = snap.to_dict() or {}
        data["run_id"] = snap.id
        return data


def _validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip().lower()
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise ApiRunNotFound("Run not found")
    return value
