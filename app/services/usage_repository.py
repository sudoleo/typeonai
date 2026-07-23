"""Persistente, run-basierte Usage-Reservierungen fuer kuenftige APIs.

Ein logischer Consensus-Run belegt genau einen Integer-Slot. Die Anzahl der
Provider/Modelle ist absichtlich kein Teil dieser Schnittstelle. Deep-Think-
Runs verwenden einen separaten Zaehler.

Firestore-Datenmodell (unter ``users/{uid}``):

* ``usage_days/{YYYY-MM-DD}`` enthaelt die aggregierten Integer-Zaehler
  ``total_reserved``, ``total_consumed``, ``deep_think_reserved`` und
  ``deep_think_consumed`` fuer den UTC-Tag der Reservierung.
* ``usage_runs/{sha256(idempotency_key)}`` enthaelt Run-Typ, UTC-Tag und Status.
  Der Klartext-Idempotency-Key wird nicht persistiert; die UID ist bereits Teil
  des Dokumentpfads, wodurch die Idempotenz aus UID + Key entsteht.

Jeder Run belegt einen Total-Slot; Deep Think belegt zusaetzlich einen Slot im
separaten Deep-Think-Kontingent. Statusuebergaenge sind ``reserved -> consumed``
oder ``reserved -> released``.
``consumed`` und ``released`` sind terminal. Wiederholungen derselben Operation
sind idempotent; ein Key darf nicht fuer einen anderen Run-Typ wiederverwendet
werden. Reservierungen zaehlen bereits gegen das Limit, werden aber erst durch
``consume`` als verbraucht markiert. Provider-Aufrufe gehoeren deshalb zwischen
``reserve`` und ``consume``/``release`` und niemals in eine Firestore-
Transaktion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol, TypeVar

from firebase_admin import firestore


USAGE_DAYS_COLLECTION = "usage_days"
USAGE_RUNS_COLLECTION = "usage_runs"
USAGE_SCHEMA_VERSION = 1
MAX_IDEMPOTENCY_KEY_BYTES = 256


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
        now: datetime | None = None,
    ) -> UsageRunResult: ...

    def consume(self, uid: str, idempotency_key: str) -> UsageRunResult: ...

    def release(self, uid: str, idempotency_key: str) -> UsageRunResult: ...

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
        now: datetime | None = None,
    ) -> UsageRunResult:
        uid = _validate_uid(uid)
        key_hash = _idempotency_hash(idempotency_key)
        kind = _coerce_kind(kind)
        now = _as_utc(now)
        utc_date = now.date().isoformat()
        run_ref = self._run_ref(uid, key_hash)

        def operation(tx):
            run_snap = run_ref.get(transaction=tx)
            if run_snap.exists:
                run_data = run_snap.to_dict() or {}
                existing_kind = _stored_kind(run_data)
                if existing_kind is not kind:
                    raise UsageRunConflict(
                        "Idempotency key is already bound to a different run kind"
                    )
                run_date = _stored_utc_date(run_data)
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

    def consume(self, uid: str, idempotency_key: str) -> UsageRunResult:
        return self._finish(uid, idempotency_key, RunStatus.CONSUMED)

    def release(self, uid: str, idempotency_key: str) -> UsageRunResult:
        return self._finish(uid, idempotency_key, RunStatus.RELEASED)

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
