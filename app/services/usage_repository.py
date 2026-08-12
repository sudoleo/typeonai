"""Persistente, run-basierte Usage-Reservierungen fuer kuenftige APIs.

Ein logischer Consensus-Run belegt genau einen Integer-Slot. Die Anzahl der
Provider/Modelle ist absichtlich kein Teil dieser Schnittstelle. Deep-Think-
Runs verwenden einen separaten Zaehler.

Firestore-Datenmodell (unter ``users/{uid}``):

* ``usage_days/{YYYY-MM-DD}`` enthaelt die aggregierten Integer-Zaehler
  ``total_reserved``, ``total_consumed``, ``deep_think_reserved`` und
  ``deep_think_consumed`` fuer den UTC-Tag der Reservierung.
* ``usage_runs/{sha256(idempotency_key)}`` enthaelt Run-Typ, UTC-Tag, Ablauf,
  kanonischen Request-Fingerprint, Status und transaktionale Operations-Claims.
  Der Klartext-Idempotency-Key wird nicht persistiert; die UID ist bereits Teil
  des Dokumentpfads, wodurch die Idempotenz aus UID + Key entsteht.

Jeder Run belegt einen Total-Slot; Deep Think belegt zusaetzlich einen Slot im
separaten Deep-Think-Kontingent. Statusuebergaenge sind ``reserved -> consumed``
oder ``reserved -> released``.
``consumed`` und ``released`` sind terminal. Wiederholungen derselben Operation
sind idempotent; ein Key darf nicht fuer einen anderen Run-Typ wiederverwendet
werden. Reservierungen zaehlen bereits gegen das Limit, werden aber erst durch
``consume`` als verbraucht markiert. Kostenpflichtige Arbeit darf erst nach
``consume`` und einem erfolgreichen Operations-Claim beginnen; Provider-Aufrufe
gehoeren niemals in eine Firestore-Transaktion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Callable, Protocol, TypeVar

from firebase_admin import firestore

from app.services import persistence_guard


USAGE_DAYS_COLLECTION = "usage_days"
USAGE_RUNS_COLLECTION = "usage_runs"
USAGE_SCHEMA_VERSION = 2
MAX_IDEMPOTENCY_KEY_BYTES = 256
MAX_OPERATION_NAME_BYTES = 80
FINGERPRINT_HEX_LENGTH = 64


class RunKind(str, Enum):
    REGULAR = "regular"
    DEEP_THINK = "deep_think"


class RunStatus(str, Enum):
    RESERVED = "reserved"
    CONSUMED = "consumed"
    RELEASED = "released"


@dataclass(frozen=True)
class UsageLimits:
    total: int
    deep_think: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.total, "total limit")
        _require_non_negative_int(self.deep_think, "deep_think limit")


@dataclass(frozen=True)
class UsageBucketSnapshot:
    limit: int
    reserved: int
    consumed: int
    remaining: int


@dataclass(frozen=True)
class UsageSnapshot:
    uid: str
    utc_date: str
    total: UsageBucketSnapshot
    deep_think: UsageBucketSnapshot


@dataclass(frozen=True)
class UsageRunResult:
    uid: str
    idempotency_hash: str
    kind: RunKind
    status: RunStatus
    utc_date: str
    snapshot: UsageSnapshot
    idempotent: bool


@dataclass(frozen=True)
class UsageOperationClaim:
    uid: str
    idempotency_hash: str
    operation: str
    request_fingerprint: str
    claimed_at: datetime
    expires_at: datetime
    idempotent: bool


class UsageRepositoryError(Exception):
    """Basisklasse fuer erwartbare Usage-Repository-Fehler."""


class UsageLimitExceeded(UsageRepositoryError):
    def __init__(
        self,
        *,
        uid: str,
        kind: RunKind,
        utc_date: str,
        snapshot: UsageSnapshot,
        limiting_bucket: str,
    ):
        super().__init__(f"{limiting_bucket} usage limit reached")
        self.uid = uid
        self.kind = kind
        self.utc_date = utc_date
        self.snapshot = snapshot
        self.limiting_bucket = limiting_bucket


class UsageRunNotFound(UsageRepositoryError):
    pass


class UsageRunConflict(UsageRepositoryError):
    pass


class UsageTransitionError(UsageRepositoryError):
    pass


class UsageRunExpired(UsageRepositoryError):
    pass


class UsageOperationAlreadyClaimed(UsageRepositoryError):
    pass


class UsageDataError(UsageRepositoryError):
    pass


class UsageRepository(Protocol):
    def reserve(
        self,
        uid: str,
        idempotency_key: str,
        kind: RunKind,
        limits: UsageLimits,
        *,
        request_fingerprint: str | None = None,
        now: datetime | None = None,
    ) -> UsageRunResult: ...

    def consume(self, uid: str, idempotency_key: str) -> UsageRunResult: ...

    def release(self, uid: str, idempotency_key: str) -> UsageRunResult: ...

    def get_run(
        self,
        uid: str,
        idempotency_key: str,
        *,
        now: datetime | None = None,
    ) -> UsageRunResult: ...

    def bind_context_target(
        self,
        uid: str,
        idempotency_key: str,
        target_scope: str,
        *,
        now: datetime | None = None,
    ) -> None: ...

    def claim_operation(
        self,
        uid: str,
        idempotency_key: str,
        operation: str,
        request_fingerprint: str,
        *,
        now: datetime | None = None,
    ) -> UsageOperationClaim: ...

    def snapshot(
        self,
        uid: str,
        limits: UsageLimits,
        *,
        now: datetime | None = None,
    ) -> UsageSnapshot: ...


T = TypeVar("T")
TransactionRunner = Callable[[Callable[[object], T]], T]


class FirestoreUsageRepository:
    """Firestore-Implementierung mit atomarem Check-and-reserve.

    ``transaction_runner`` ist ein Test-Seam. In Produktion wird immer der
    Retry-faehige ``firebase_admin.firestore.transactional``-Wrapper benutzt.
    """

    def __init__(self, db, *, transaction_runner: TransactionRunner | None = None):
        self._db = db
        self._transaction_runner = transaction_runner

    def reserve(
        self,
        uid: str,
        idempotency_key: str,
        kind: RunKind,
        limits: UsageLimits,
        *,
        request_fingerprint: str | None = None,
        now: datetime | None = None,
    ) -> UsageRunResult:
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        kind = _coerce_kind(kind)
        request_fingerprint = _validate_fingerprint(
            request_fingerprint
            or canonical_request_fingerprint({"internal_idempotency_hash": key_hash})
        )
        now = _as_utc(now)
        utc_date = now.date().isoformat()
        expires_at = datetime.combine(
            now.date() + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        run_ref = self._run_ref(uid, key_hash)

        def operation(tx):
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=self._db, transaction=tx
            )
            run_snap = run_ref.get(transaction=tx)
            if run_snap.exists:
                run_data = run_snap.to_dict() or {}
                existing_kind = _stored_kind(run_data)
                if existing_kind is not kind:
                    raise UsageRunConflict(
                        "Idempotency key is already bound to a different run kind"
                    )
                run_date = _stored_utc_date(run_data)
                stored_fingerprint = _stored_request_fingerprint(run_data)
                if not hmac.compare_digest(stored_fingerprint, request_fingerprint):
                    raise UsageRunConflict(
                        "Idempotency key is already bound to a different request"
                    )
                stored_expires_at = _stored_expires_at(run_data)
                if now >= stored_expires_at:
                    raise UsageRunExpired("Usage run has expired")
                day_data = self._read_day(tx, uid, run_date)
                return _result(
                    uid,
                    key_hash,
                    existing_kind,
                    _stored_status(run_data),
                    run_date,
                    day_data,
                    _stored_limits(run_data),
                    idempotent=True,
                )

            day_ref = self._day_ref(uid, utc_date)
            day_data = self._read_day(tx, uid, utc_date)
            snapshot = _snapshot(uid, utc_date, day_data, limits)
            if snapshot.total.remaining < 1:
                raise UsageLimitExceeded(
                    uid=uid,
                    kind=kind,
                    utc_date=utc_date,
                    snapshot=snapshot,
                    limiting_bucket="total",
                )
            if kind is RunKind.DEEP_THINK and snapshot.deep_think.remaining < 1:
                raise UsageLimitExceeded(
                    uid=uid,
                    kind=kind,
                    utc_date=utc_date,
                    snapshot=snapshot,
                    limiting_bucket="deep_think",
                )

            day_data["total_reserved"] += 1
            if kind is RunKind.DEEP_THINK:
                day_data["deep_think_reserved"] += 1
            day_data.update(
                {
                    "schema_version": USAGE_SCHEMA_VERSION,
                    "utc_date": utc_date,
                    "updated_at": now,
                }
            )
            tx.set(day_ref, day_data, merge=True)
            tx.set(
                run_ref,
                {
                    "schema_version": USAGE_SCHEMA_VERSION,
                    "kind": kind.value,
                    "status": RunStatus.RESERVED.value,
                    "utc_date": utc_date,
                    "total_limit_at_reservation": limits.total,
                    "deep_think_limit_at_reservation": limits.deep_think,
                    "request_fingerprint": request_fingerprint,
                    "expires_at": expires_at,
                    "operation_claims": {},
                    "created_at": now,
                    "updated_at": now,
                },
            )
            return _result(
                uid,
                key_hash,
                kind,
                RunStatus.RESERVED,
                utc_date,
                day_data,
                limits,
                idempotent=False,
            )

        return self._transaction(operation)

    def claim_operation(
        self,
        uid: str,
        idempotency_key: str,
        operation: str,
        request_fingerprint: str,
        *,
        now: datetime | None = None,
    ) -> UsageOperationClaim:
        """Atomically authorize one billable logical operation once.

        Accounting idempotency and execution authorization are deliberately
        separate. Repeating a consumed run may read its counters, but it may
        never acquire the same operation slot twice.
        """
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        operation = _validate_operation(operation)
        request_fingerprint = _validate_fingerprint(request_fingerprint)
        now = _as_utc(now)
        run_ref = self._run_ref(uid, key_hash)

        def claim(tx):
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=self._db, transaction=tx
            )
            run_snap = run_ref.get(transaction=tx)
            if not run_snap.exists:
                raise UsageRunNotFound("Usage reservation does not exist")
            run_data = run_snap.to_dict() or {}
            if _stored_status(run_data) is not RunStatus.CONSUMED:
                raise UsageTransitionError(
                    "Only a consumed usage run can authorize provider work"
                )
            expires_at = _stored_expires_at(run_data)
            if now >= expires_at:
                raise UsageRunExpired("Usage run has expired")

            claims = run_data.get("operation_claims")
            if claims is None:
                claims = {}
            if not isinstance(claims, dict):
                raise UsageDataError("Invalid operation claims in Firestore")
            existing = claims.get(operation)
            if existing is not None:
                if not isinstance(existing, dict):
                    raise UsageDataError("Invalid operation claim in Firestore")
                stored_fingerprint = _validate_fingerprint(
                    existing.get("request_fingerprint"),
                    error_type=UsageDataError,
                )
                if not hmac.compare_digest(stored_fingerprint, request_fingerprint):
                    raise UsageRunConflict(
                        "Operation is already bound to a different request"
                    )
                claimed_at = _stored_datetime(existing.get("claimed_at"), "claim time")
                return UsageOperationClaim(
                    uid=uid,
                    idempotency_hash=key_hash,
                    operation=operation,
                    request_fingerprint=stored_fingerprint,
                    claimed_at=claimed_at,
                    expires_at=expires_at,
                    idempotent=True,
                )

            updated_claims = dict(claims)
            updated_claims[operation] = {
                "request_fingerprint": request_fingerprint,
                "claimed_at": now,
            }
            tx.update(
                run_ref,
                {
                    "operation_claims": updated_claims,
                    "updated_at": now,
                },
            )
            return UsageOperationClaim(
                uid=uid,
                idempotency_hash=key_hash,
                operation=operation,
                request_fingerprint=request_fingerprint,
                claimed_at=now,
                expires_at=expires_at,
                idempotent=False,
            )

        return self._transaction(claim)

    def consume(self, uid: str, idempotency_key: str) -> UsageRunResult:
        return self._finish(uid, idempotency_key, RunStatus.CONSUMED)

    def release(self, uid: str, idempotency_key: str) -> UsageRunResult:
        return self._finish(uid, idempotency_key, RunStatus.RELEASED)

    def get_run(
        self,
        uid: str,
        idempotency_key: str,
        *,
        now: datetime | None = None,
    ) -> UsageRunResult:
        """Read a logical run without changing its lifecycle or counters."""
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        now = _as_utc(now)
        snap = self._run_ref(uid, key_hash).get()
        if not snap.exists:
            raise UsageRunNotFound("Usage reservation does not exist")
        run_data = snap.to_dict() or {}
        if now >= _stored_expires_at(run_data):
            raise UsageRunExpired("Usage run has expired")
        kind = _stored_kind(run_data)
        status = _stored_status(run_data)
        utc_date = _stored_utc_date(run_data)
        limits = _stored_limits(run_data)
        day_snap = self._day_ref(uid, utc_date).get()
        day_data = _parse_day_data(day_snap.to_dict() if day_snap.exists else {})
        return _result(
            uid,
            key_hash,
            kind,
            status,
            utc_date,
            day_data,
            limits,
            idempotent=True,
        )

    def bind_context_target(
        self,
        uid: str,
        idempotency_key: str,
        target_scope: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Bind one consumed logical run to one chat-context target.

        Only a hash of the target scope is stored. This does not reserve or
        consume another slot; it prevents one historical consumed key from
        financing context builds for multiple turns.
        """
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        now = _as_utc(now)
        if not isinstance(target_scope, str) or not target_scope.strip():
            raise ValueError("target_scope must not be empty")
        encoded_scope = target_scope.encode("utf-8")
        if len(encoded_scope) > 512:
            raise ValueError("target_scope is too long")
        target_hash = hashlib.sha256(encoded_scope).hexdigest()
        run_ref = self._run_ref(uid, key_hash)

        def operation(tx):
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=self._db, transaction=tx
            )
            run_snap = run_ref.get(transaction=tx)
            if not run_snap.exists:
                raise UsageRunNotFound("Usage reservation does not exist")
            run_data = run_snap.to_dict() or {}
            if now >= _stored_expires_at(run_data):
                raise UsageRunExpired("Usage run has expired")
            status = _stored_status(run_data)
            if status is not RunStatus.CONSUMED:
                raise UsageTransitionError(
                    "Only a consumed usage run can fund chat context"
                )
            existing = run_data.get("context_target_hash")
            if isinstance(existing, str) and existing:
                if not hmac.compare_digest(existing, target_hash):
                    raise UsageRunConflict(
                        "Usage run is already bound to a different context target"
                    )
                return
            tx.update(
                run_ref,
                {
                    "context_target_hash": target_hash,
                    "context_bound_at": now,
                    "updated_at": now,
                },
            )

        self._transaction(operation)

    def snapshot(
        self,
        uid: str,
        limits: UsageLimits,
        *,
        now: datetime | None = None,
    ) -> UsageSnapshot:
        uid = _validate_uid(uid)
        utc_date = _as_utc(now).date().isoformat()
        snap = self._day_ref(uid, utc_date).get()
        day_data = _parse_day_data(snap.to_dict() if snap.exists else {})
        return _snapshot(uid, utc_date, day_data, limits)

    def _finish(
        self, uid: str, idempotency_key: str, target: RunStatus
    ) -> UsageRunResult:
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        run_ref = self._run_ref(uid, key_hash)

        def operation(tx):
            persistence_guard.ensure_account_write_allowed(
                uid=uid, db=self._db, transaction=tx
            )
            run_snap = run_ref.get(transaction=tx)
            if not run_snap.exists:
                raise UsageRunNotFound("Usage reservation does not exist")
            run_data = run_snap.to_dict() or {}
            kind = _stored_kind(run_data)
            status = _stored_status(run_data)
            utc_date = _stored_utc_date(run_data)
            limits = _stored_limits(run_data)
            day_ref = self._day_ref(uid, utc_date)
            day_data = self._read_day(tx, uid, utc_date)

            if status is target:
                return _result(
                    uid,
                    key_hash,
                    kind,
                    status,
                    utc_date,
                    day_data,
                    limits,
                    idempotent=True,
                )
            if status is not RunStatus.RESERVED:
                raise UsageTransitionError(
                    f"Cannot transition usage run from {status.value} to {target.value}"
                )

            if day_data["total_reserved"] < 1:
                raise UsageDataError("Reserved counter is inconsistent with usage run")
            if kind is RunKind.DEEP_THINK and day_data["deep_think_reserved"] < 1:
                raise UsageDataError("Deep Think counter is inconsistent with usage run")
            day_data["total_reserved"] -= 1
            if target is RunStatus.CONSUMED:
                day_data["total_consumed"] += 1
            if kind is RunKind.DEEP_THINK:
                day_data["deep_think_reserved"] -= 1
                if target is RunStatus.CONSUMED:
                    day_data["deep_think_consumed"] += 1

            updated_at = datetime.now(timezone.utc)
            day_data.update(
                {
                    "schema_version": USAGE_SCHEMA_VERSION,
                    "utc_date": utc_date,
                    "updated_at": updated_at,
                }
            )
            tx.set(day_ref, day_data, merge=True)
            tx.update(
                run_ref,
                {
                    "status": target.value,
                    "updated_at": updated_at,
                    f"{target.value}_at": updated_at,
                },
            )
            return _result(
                uid,
                key_hash,
                kind,
                target,
                utc_date,
                day_data,
                limits,
                idempotent=False,
            )

        return self._transaction(operation)

    def _transaction(self, operation: Callable[[object], T]) -> T:
        if self._transaction_runner is not None:
            return self._transaction_runner(operation)
        # Ein UI-Lauf fannt mehrere /ask_* Requests parallel mit demselben
        # Idempotency-Key aus. Der erste Request konsumiert den Run, die
        # restlichen muessen danach idempotent den CONSUMED-Stand lesen. Fuenf
        # Firestore-Versuche reichen bei sechs gleichzeitigen Transaktionen
        # nicht verlaesslich; ein hoeheres SDK-Retry-Budget laesst die kurze
        # Hot-Document-Kollision auslaufen, ohne den Run mehrfach zu zaehlen.
        transaction = self._db.transaction(max_attempts=12)

        @firestore.transactional
        def run(tx):
            return operation(tx)

        return run(transaction)

    def _user_ref(self, uid: str):
        return self._db.collection("users").document(uid)

    def _day_ref(self, uid: str, utc_date: str):
        return self._user_ref(uid).collection(USAGE_DAYS_COLLECTION).document(utc_date)

    def _run_ref(self, uid: str, key_hash: str):
        return self._user_ref(uid).collection(USAGE_RUNS_COLLECTION).document(key_hash)

    def _read_day(self, tx, uid: str, utc_date: str) -> dict:
        snap = self._day_ref(uid, utc_date).get(transaction=tx)
        return _parse_day_data(snap.to_dict() if snap.exists else {})


def _require_non_negative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_uid(uid: str) -> str:
    value = str(uid or "").strip()
    if not value:
        raise ValueError("uid must not be empty")
    return value


def _idempotency_hash(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("idempotency_key must not be empty")
    encoded = key.encode("utf-8")
    if len(encoded) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise ValueError(
            f"idempotency_key must not exceed {MAX_IDEMPOTENCY_KEY_BYTES} bytes"
        )
    return hashlib.sha256(encoded).hexdigest()


def canonical_request_fingerprint(value) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible request."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Request fingerprint input must be canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_fingerprint(value, *, error_type=ValueError) -> str:
    if not isinstance(value, str) or len(value) != FINGERPRINT_HEX_LENGTH:
        raise error_type("Invalid request fingerprint")
    try:
        int(value, 16)
    except ValueError:
        raise error_type("Invalid request fingerprint") from None
    return value.lower()


def _validate_operation(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation must not be empty")
    normalized = value.strip().lower()
    if len(normalized.encode("utf-8")) > MAX_OPERATION_NAME_BYTES:
        raise ValueError("operation is too long")
    if not all(char.isalnum() or char in {":", "_", "-"} for char in normalized):
        raise ValueError("operation contains invalid characters")
    return normalized


def _coerce_kind(kind: RunKind) -> RunKind:
    try:
        return RunKind(kind)
    except (TypeError, ValueError):
        raise ValueError("Unsupported usage run kind") from None


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_day_data(raw: dict | None) -> dict:
    data = raw if isinstance(raw, dict) else {}
    parsed = {}
    for field in (
        "total_reserved",
        "total_consumed",
        "deep_think_reserved",
        "deep_think_consumed",
    ):
        value = data.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UsageDataError(f"Invalid Firestore usage counter: {field}")
        parsed[field] = value
    return parsed


def _bucket_snapshot(data: dict, prefix: str, limit: int) -> UsageBucketSnapshot:
    _require_non_negative_int(limit, f"{prefix} limit")
    reserved = data[f"{prefix}_reserved"]
    consumed = data[f"{prefix}_consumed"]
    return UsageBucketSnapshot(
        limit=limit,
        reserved=reserved,
        consumed=consumed,
        remaining=max(0, limit - reserved - consumed),
    )


def _snapshot(
    uid: str, utc_date: str, day_data: dict, limits: UsageLimits
) -> UsageSnapshot:
    return UsageSnapshot(
        uid=uid,
        utc_date=utc_date,
        total=_bucket_snapshot(day_data, "total", limits.total),
        deep_think=_bucket_snapshot(day_data, "deep_think", limits.deep_think),
    )


def _stored_kind(data: dict) -> RunKind:
    try:
        return RunKind(data.get("kind"))
    except (TypeError, ValueError):
        raise UsageDataError("Invalid usage run kind in Firestore") from None


def _stored_status(data: dict) -> RunStatus:
    try:
        return RunStatus(data.get("status"))
    except (TypeError, ValueError):
        raise UsageDataError("Invalid usage run status in Firestore") from None


def _stored_utc_date(data: dict) -> str:
    value = data.get("utc_date")
    if not isinstance(value, str):
        raise UsageDataError("Invalid usage run UTC date in Firestore")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise UsageDataError("Invalid usage run UTC date in Firestore") from None
    if parsed.isoformat() != value:
        raise UsageDataError("Invalid usage run UTC date in Firestore")
    return value


def _stored_request_fingerprint(data: dict) -> str:
    return _validate_fingerprint(
        data.get("request_fingerprint"), error_type=UsageDataError
    )


def _stored_datetime(value, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise UsageDataError(f"Invalid usage run {label} in Firestore")
    if value.tzinfo is None:
        raise UsageDataError(f"Invalid usage run {label} in Firestore")
    return value.astimezone(timezone.utc)


def _stored_expires_at(data: dict) -> datetime:
    return _stored_datetime(data.get("expires_at"), "expiry")


def _stored_limits(data: dict) -> UsageLimits:
    try:
        return UsageLimits(
            total=data.get("total_limit_at_reservation"),
            deep_think=data.get("deep_think_limit_at_reservation"),
        )
    except ValueError as exc:
        raise UsageDataError(str(exc)) from None


def _result(
    uid: str,
    key_hash: str,
    kind: RunKind,
    status: RunStatus,
    utc_date: str,
    day_data: dict,
    limits: UsageLimits,
    *,
    idempotent: bool,
) -> UsageRunResult:
    return UsageRunResult(
        uid=uid,
        idempotency_hash=key_hash,
        kind=kind,
        status=status,
        utc_date=utc_date,
        snapshot=_snapshot(uid, utc_date, day_data, limits),
        idempotent=idempotent,
    )
