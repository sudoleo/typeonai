"""The synchronous SSE endpoints must not run out of worker threads.

``/ask_*`` and ``/consensus`` return sync generators, so Starlette holds one
anyio worker thread per in-flight provider stream. With anyio's default of 40
threads a single consensus run (six providers) let roughly six concurrent users
saturate the pool. These tests pin the raised budget and its bounds.
"""

from __future__ import annotations

import anyio
import anyio.to_thread
import pytest

from app.core import concurrency


def test_default_budget_carries_more_than_six_concurrent_runs(monkeypatch):
    monkeypatch.delenv("MAX_WORKER_THREADS", raising=False)

    budget = concurrency.configured_max_worker_threads()

    assert budget == concurrency.DEFAULT_MAX_WORKER_THREADS
    # Six providers per run: the whole point is to clear anyio's default of 40.
    assert budget // 6 >= 20


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("240", 240),
        ("1", concurrency.MIN_MAX_WORKER_THREADS),  # below the floor
        ("100000", concurrency.MAX_MAX_WORKER_THREADS),  # above the ceiling
        ("not-a-number", concurrency.DEFAULT_MAX_WORKER_THREADS),
        ("", concurrency.DEFAULT_MAX_WORKER_THREADS),
    ],
)
def test_budget_is_bounded_and_survives_garbage(monkeypatch, configured, expected):
    monkeypatch.setenv("MAX_WORKER_THREADS", configured)

    assert concurrency.configured_max_worker_threads() == expected


def test_apply_raises_the_running_loops_limiter(monkeypatch):
    monkeypatch.setenv("MAX_WORKER_THREADS", "144")

    async def scenario():
        before = anyio.to_thread.current_default_thread_limiter().total_tokens
        applied = concurrency.apply_worker_thread_budget()
        after = anyio.to_thread.current_default_thread_limiter().total_tokens
        return before, applied, after

    before, applied, after = anyio.run(scenario)

    assert before == 40, "anyio's untouched default; if this changes, re-check the sizing"
    assert applied == 144
    assert after == 144


def test_apply_never_lowers_an_already_larger_budget(monkeypatch):
    monkeypatch.setenv("MAX_WORKER_THREADS", "64")

    async def scenario():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 300
        concurrency.apply_worker_thread_budget()
        return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(scenario) == 300


def test_lifespan_applies_the_budget_before_the_e2e_early_return(monkeypatch):
    """The E2E profile skips every maintenance loop -- but not this."""

    import main

    applied: list[int] = []
    monkeypatch.setattr(
        main, "apply_worker_thread_budget", lambda: applied.append(1) or 1
    )
    monkeypatch.setattr(main, "e2e_test_mode_enabled", lambda: True)

    async def scenario():
        async with main.lifespan(main.app):
            pass

    anyio.run(scenario)

    assert applied == [1]
