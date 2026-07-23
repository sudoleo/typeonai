"""Run-basierte Firestore-Usage: Limits, Idempotenz und Parallelitaet."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

import app.core.config as cfg
from usage_test_support import FakeFirestore, make_usage_repository
from app.services.usage_repository import (
    FirestoreUsageRepository,
    RunKind,
    RunStatus,
    UsageLimitExceeded,
    UsageLimits,
    UsageRunConflict,
    UsageRunNotFound,
    UsageTransitionError,
)


@pytest.fixture
def usage_repo():
    return make_usage_repository()


LIMITS = UsageLimits(total=3, deep_think=1)
UTC_NOON = datetime(2026, 7, 18, 12, tzinfo=timezone.utc)


def test_new_free_consensus_run_limit_defaults_to_three():
    assert cfg.DEFAULT_LIMITS["free_consensus_run_limit"] == 3
    assert cfg.get_consensus_run_limit(False) == 3
    assert cfg.get_deep_think_run_limit(False) == 0


def test_production_path_wraps_reserve_in_firestore_transaction(monkeypatch):
    db = FakeFirestore()
    calls = []
    transaction_options = []
    original_transaction = db.transaction

    def capture_transaction(**kwargs):
        transaction_options.append(kwargs)
        return original_transaction(**kwargs)

    db.transaction = capture_transaction

    def fake_transactional(function):
        calls.append("decorated")

        def execute(transaction):
            with db.lock:
                result = function(transaction)
                transaction.commit()
                return result

        return execute

    monkeypatch.setattr(
        "app.services.usage_repository.firestore.transactional", fake_transactional
    )
    repo = FirestoreUsageRepository(db)

    repo.reserve("firestore-user", "tx-key", RunKind.REGULAR, LIMITS, now=UTC_NOON)

    assert calls == ["decorated"]
    assert transaction_options == [{"max_attempts": 12}]
    assert repo.snapshot("firestore-user", LIMITS, now=UTC_NOON).total.reserved == 1


def test_usage_limits_reject_floats_and_booleans():
    with pytest.raises(ValueError):
        UsageLimits(total=3.0, deep_think=1)
    with pytest.raises(ValueError):
        UsageLimits(total=3, deep_think=True)


def test_regular_limit_reserves_exactly_three_integer_slots(usage_repo):
    repo, db = usage_repo

    for index in range(3):
        result = repo.reserve(
            "user-1", f"run-{index}", RunKind.REGULAR, LIMITS, now=UTC_NOON
        )
        assert result.status is RunStatus.RESERVED

    with pytest.raises(UsageLimitExceeded) as exc_info:
        repo.reserve("user-1", "run-3", RunKind.REGULAR, LIMITS, now=UTC_NOON)

    assert exc_info.value.limiting_bucket == "total"
    assert exc_info.value.snapshot.total.remaining == 0
    snapshot = repo.snapshot("user-1", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 3
    assert snapshot.total.consumed == 0
    assert snapshot.total.remaining == 0
    assert all(
        not isinstance(value, float)
        for document in db.documents.values()
        for value in document.values()
    )


def test_parallel_unique_reservations_cannot_oversubscribe_limit(usage_repo):
    repo, _ = usage_repo

    def reserve(index):
        try:
            repo.reserve(
                "parallel-user",
                f"parallel-{index}",
                RunKind.REGULAR,
                LIMITS,
                now=UTC_NOON,
            )
            return "reserved"
        except UsageLimitExceeded:
            return "limited"

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(24)))

    assert results.count("reserved") == 3
    assert results.count("limited") == 21
    snapshot = repo.snapshot("parallel-user", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 3


def test_parallel_same_idempotency_key_reserves_only_once(usage_repo):
    repo, _ = usage_repo

    def reserve(_):
        return repo.reserve(
            "same-key-user", "one-logical-run", RunKind.REGULAR, LIMITS, now=UTC_NOON
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(reserve, range(20)))

    assert sum(not result.idempotent for result in results) == 1
    assert all(result.status is RunStatus.RESERVED for result in results)
    snapshot = repo.snapshot("same-key-user", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 1


def test_idempotency_is_scoped_by_uid_and_bound_to_run_kind(usage_repo):
    repo, _ = usage_repo
    first = repo.reserve("user-a", "shared-key", RunKind.REGULAR, LIMITS, now=UTC_NOON)
    repeated = repo.reserve("user-a", "shared-key", RunKind.REGULAR, LIMITS, now=UTC_NOON)
    other_user = repo.reserve("user-b", "shared-key", RunKind.REGULAR, LIMITS, now=UTC_NOON)

    assert first.idempotent is False
    assert repeated.idempotent is True
    assert other_user.idempotent is False
    with pytest.raises(UsageRunConflict):
        repo.reserve("user-a", "shared-key", RunKind.DEEP_THINK, LIMITS, now=UTC_NOON)


def test_consume_moves_one_slot_and_is_idempotent(usage_repo):
    repo, _ = usage_repo
    repo.reserve("consumer", "consume-me", RunKind.REGULAR, LIMITS, now=UTC_NOON)

    consumed = repo.consume("consumer", "consume-me")
    repeated = repo.consume("consumer", "consume-me")

    assert consumed.status is RunStatus.CONSUMED
    assert consumed.idempotent is False
    assert repeated.status is RunStatus.CONSUMED
    assert repeated.idempotent is True
    snapshot = repo.snapshot("consumer", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 0
    assert snapshot.total.consumed == 1
    assert snapshot.total.remaining == 2
    with pytest.raises(UsageTransitionError):
        repo.release("consumer", "consume-me")


def test_release_frees_slot_and_is_idempotent(usage_repo):
    repo, _ = usage_repo
    repo.reserve("releaser", "release-me", RunKind.REGULAR, LIMITS, now=UTC_NOON)

    released = repo.release("releaser", "release-me")
    repeated = repo.release("releaser", "release-me")

    assert released.status is RunStatus.RELEASED
    assert released.idempotent is False
    assert repeated.status is RunStatus.RELEASED
    assert repeated.idempotent is True
    snapshot = repo.snapshot("releaser", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 0
    assert snapshot.total.consumed == 0
    assert snapshot.total.remaining == 3
    with pytest.raises(UsageTransitionError):
        repo.consume("releaser", "release-me")


def test_deep_think_has_separate_limit_and_counters(usage_repo):
    repo, _ = usage_repo
    repo.reserve("deep-user", "regular", RunKind.REGULAR, LIMITS, now=UTC_NOON)
    repo.reserve("deep-user", "deep", RunKind.DEEP_THINK, LIMITS, now=UTC_NOON)

    with pytest.raises(UsageLimitExceeded) as exc_info:
        repo.reserve("deep-user", "deep-2", RunKind.DEEP_THINK, LIMITS, now=UTC_NOON)

    assert exc_info.value.limiting_bucket == "deep_think"
    snapshot = repo.snapshot("deep-user", LIMITS, now=UTC_NOON)
    # Beide logischen Runs zaehlen je genau einmal gegen das Total-Limit; nur
    # der Deep-Think-Run belegt zusaetzlich das separate Deep-Kontingent.
    assert snapshot.total.reserved == 2
    assert snapshot.total.remaining == 1
    assert snapshot.deep_think.reserved == 1
    assert snapshot.deep_think.remaining == 0


def test_consuming_deep_think_moves_total_and_deep_counters(usage_repo):
    repo, _ = usage_repo
    repo.reserve("deep-consumer", "deep", RunKind.DEEP_THINK, LIMITS, now=UTC_NOON)

    repo.consume("deep-consumer", "deep")

    snapshot = repo.snapshot("deep-consumer", LIMITS, now=UTC_NOON)
    assert snapshot.total.reserved == 0
    assert snapshot.total.consumed == 1
    assert snapshot.deep_think.reserved == 0
    assert snapshot.deep_think.consumed == 1


def test_total_limit_blocks_deep_think_even_when_deep_quota_remains(usage_repo):
    repo, _ = usage_repo
    tight_limits = UsageLimits(total=1, deep_think=2)
    repo.reserve("total-user", "regular", RunKind.REGULAR, tight_limits, now=UTC_NOON)

    with pytest.raises(UsageLimitExceeded) as exc_info:
        repo.reserve(
            "total-user", "deep", RunKind.DEEP_THINK, tight_limits, now=UTC_NOON
        )

    assert exc_info.value.limiting_bucket == "total"


def test_reservation_is_charged_to_utc_day_of_reserve(usage_repo):
    repo, _ = usage_repo
    berlin = timezone(timedelta(hours=2))
    local_after_midnight = datetime(2026, 7, 19, 1, 30, tzinfo=berlin)

    result = repo.reserve(
        "utc-user", "utc-run", RunKind.REGULAR, LIMITS, now=local_after_midnight
    )

    assert result.utc_date == "2026-07-18"
    assert repo.snapshot("utc-user", LIMITS, now=UTC_NOON).total.reserved == 1
    next_day = datetime(2026, 7, 19, 12, tzinfo=timezone.utc)
    assert repo.snapshot("utc-user", LIMITS, now=next_day).total.reserved == 0


def test_missing_reservation_cannot_be_consumed_or_released(usage_repo):
    repo, _ = usage_repo
    with pytest.raises(UsageRunNotFound):
        repo.consume("missing-user", "missing")
    with pytest.raises(UsageRunNotFound):
        repo.release("missing-user", "missing")
