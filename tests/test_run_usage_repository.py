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
    UsageRunExpired,
    UsageRunNotFound,
    UsageTransitionError,
    canonical_request_fingerprint,
)


@pytest.fixture
def usage_repo():
    return make_usage_repository()


LIMITS = UsageLimits(total=3, deep_think=1)
UTC_NOON = datetime(2026, 7, 18, 12, tzinfo=timezone.utc)


def test_free_consensus_run_limit_allows_a_daily_habit():
    # Drei Runs waren ein Test, keine Gewohnheit. Der Wert darf steigen, aber
    # nicht wieder unter das zurueckfallen, was eine Session unbrauchbar macht.
    assert cfg.DEFAULT_LIMITS["free_consensus_run_limit"] == 12
    assert cfg.get_consensus_run_limit(False) >= 10
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


def test_get_run_is_read_only_and_reports_consumed_status(usage_repo):
    repo, db = usage_repo
    repo.reserve("reader", "read-me", RunKind.REGULAR, LIMITS, now=UTC_NOON)
    repo.consume("reader", "read-me")
    before = {path: dict(data) for path, data in db.documents.items()}

    result = repo.get_run("reader", "read-me")

    assert result.status is RunStatus.CONSUMED
    assert result.idempotent is True
    assert db.documents == before


def test_consumed_run_context_binding_is_idempotent_and_target_specific(usage_repo):
    repo, db = usage_repo
    repo.reserve("binder", "bind-me", RunKind.REGULAR, LIMITS, now=UTC_NOON)
    repo.consume("binder", "bind-me")
    before_snapshot = repo.snapshot("binder", LIMITS, now=UTC_NOON)

    repo.bind_context_target("binder", "bind-me", "chat-context\0chat-a\0turn-a")
    first_documents = {path: dict(data) for path, data in db.documents.items()}
    repo.bind_context_target("binder", "bind-me", "chat-context\0chat-a\0turn-a")

    assert db.documents == first_documents
    after_snapshot = repo.snapshot("binder", LIMITS, now=UTC_NOON)
    assert after_snapshot == before_snapshot
    with pytest.raises(UsageRunConflict):
        repo.bind_context_target(
            "binder", "bind-me", "chat-context\0chat-b\0turn-b"
        )


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


def test_request_fingerprint_binds_reused_key(usage_repo):
    repo, _ = usage_repo
    first = canonical_request_fingerprint({"question": "first"})
    second = canonical_request_fingerprint({"question": "second"})
    repo.reserve(
        "bound-user",
        "bound-key",
        RunKind.REGULAR,
        LIMITS,
        request_fingerprint=first,
        now=UTC_NOON,
    )

    with pytest.raises(UsageRunConflict):
        repo.reserve(
            "bound-user",
            "bound-key",
            RunKind.REGULAR,
            LIMITS,
            request_fingerprint=second,
            now=UTC_NOON,
        )


def test_parallel_operation_claim_allows_exactly_one_winner(usage_repo):
    repo, _ = usage_repo
    fingerprint = canonical_request_fingerprint({"question": "same"})
    repo.reserve(
        "claim-user",
        "claim-key",
        RunKind.REGULAR,
        LIMITS,
        request_fingerprint=fingerprint,
        now=UTC_NOON,
    )
    repo.consume("claim-user", "claim-key")
    operation_fingerprint = canonical_request_fingerprint({"model": "gpt"})

    def claim(_):
        return repo.claim_operation(
            "claim-user",
            "claim-key",
            "ask:openai",
            operation_fingerprint,
            now=UTC_NOON,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        claims = list(pool.map(claim, range(24)))

    assert sum(not item.idempotent for item in claims) == 1
    assert sum(item.idempotent for item in claims) == 23


def test_cross_operation_claims_are_independent_and_payload_bound(usage_repo):
    repo, _ = usage_repo
    run_fingerprint = canonical_request_fingerprint({"question": "same"})
    repo.reserve(
        "cross-operation",
        "claim-key",
        RunKind.REGULAR,
        LIMITS,
        request_fingerprint=run_fingerprint,
        now=UTC_NOON,
    )
    repo.consume("cross-operation", "claim-key")
    ask = canonical_request_fingerprint({"provider": "openai"})
    consensus = canonical_request_fingerprint({"engine": "gemini"})

    assert repo.claim_operation(
        "cross-operation", "claim-key", "ask:openai", ask, now=UTC_NOON
    ).idempotent is False
    assert repo.claim_operation(
        "cross-operation", "claim-key", "consensus", consensus, now=UTC_NOON
    ).idempotent is False
    with pytest.raises(UsageRunConflict):
        repo.claim_operation(
            "cross-operation",
            "claim-key",
            "ask:openai",
            canonical_request_fingerprint({"provider": "openai", "changed": True}),
            now=UTC_NOON,
        )


def test_cross_day_replay_cannot_claim_provider_work(usage_repo):
    repo, _ = usage_repo
    run_fingerprint = canonical_request_fingerprint({"question": "today"})
    repo.reserve(
        "expired-user",
        "expired-key",
        RunKind.REGULAR,
        LIMITS,
        request_fingerprint=run_fingerprint,
        now=UTC_NOON,
    )
    repo.consume("expired-user", "expired-key")
    tomorrow = UTC_NOON + timedelta(days=1)

    with pytest.raises(UsageRunExpired):
        repo.claim_operation(
            "expired-user",
            "expired-key",
            "ask:openai",
            canonical_request_fingerprint({"model": "gpt"}),
            now=tomorrow,
        )
