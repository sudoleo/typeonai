"""Tests fuer die atomaren Usage-Zaehler in app/core/state.py.

Die /ask_*-Endpoints laufen beim Frage-Fan-out parallel in Threadpool-Workern;
check_and_increment_usage muss Check + Inkrement unter einem Lock ausfuehren,
sonst gehen Inkremente verloren (faktische Limit-Umgehung).
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.state import (
    check_and_increment_usage,
    deep_search_usage,
    get_usage_snapshot,
    reset_usage,
    usage_counter,
)


@pytest.fixture
def uid():
    uid = "uid-usage-store-tests"
    reset_usage(uid)
    yield uid
    reset_usage(uid)


def test_ok_increments_regular_only(uid):
    status, usage, deep = check_and_increment_usage(
        uid, limit_regular=10, limit_deep=5, increment=1
    )
    assert status == "ok"
    assert usage == 1
    assert deep == 0
    assert uid not in deep_search_usage


def test_deep_search_increments_both_counters(uid):
    status, usage, deep = check_and_increment_usage(
        uid, limit_regular=10, limit_deep=5, increment=1, deep_search=True
    )
    assert status == "ok"
    assert usage == 1
    assert deep == 1


def test_regular_limit_blocks_without_incrementing(uid):
    usage_counter[uid] = 10
    status, usage, deep = check_and_increment_usage(
        uid, limit_regular=10, limit_deep=5, increment=1
    )
    assert status == "limit_regular"
    assert usage == 10
    assert usage_counter[uid] == 10


def test_deep_limit_blocks_without_incrementing(uid):
    deep_search_usage[uid] = 5
    status, usage, deep = check_and_increment_usage(
        uid, limit_regular=10, limit_deep=5, increment=1, deep_search=True
    )
    assert status == "limit_deep"
    assert deep == 5
    assert uid not in usage_counter
    assert deep_search_usage[uid] == 5


def test_parallel_increments_do_not_undercount(uid):
    # 120 parallele Fan-out-Inkremente a 1/6 (wie 20 Fragen an 6 Modelle).
    def worker(_):
        return check_and_increment_usage(
            uid, limit_regular=1000, limit_deep=1000, increment=1.0 / 6, deep_search=True
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(worker, range(120)))

    assert all(status == "ok" for status, _, _ in results)
    usage, deep = get_usage_snapshot(uid)
    assert usage == pytest.approx(20.0)
    assert deep == pytest.approx(20.0)


def test_limit_enforced_exactly_under_contention(uid):
    limit = 10

    def worker(_):
        return check_and_increment_usage(
            uid, limit_regular=limit, limit_deep=limit, increment=1
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(50)))

    ok_count = sum(1 for status, _, _ in results if status == "ok")
    assert ok_count == limit
    assert get_usage_snapshot(uid)[0] == limit


def test_reset_usage_clears_both_counters(uid):
    check_and_increment_usage(
        uid, limit_regular=10, limit_deep=10, increment=1, deep_search=True
    )
    reset_usage(uid)
    assert uid not in usage_counter
    assert uid not in deep_search_usage
    assert get_usage_snapshot(uid) == (0, 0)
