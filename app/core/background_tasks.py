"""Supervision and health state for long-running application tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import logging
import threading
from typing import Awaitable, Callable

from app.core.observability import safe_exception


_lock = threading.Lock()
_health: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(name: str, **changes) -> None:
    with _lock:
        current = dict(_health.get(name) or {})
        current.update(changes)
        current["name"] = name
        _health[name] = current


def task_succeeded(name: str, **details) -> None:
    """Record a completed scheduler/maintenance tick."""
    _update(
        name,
        state="running",
        last_success_at=_now(),
        consecutive_failures=0,
        details=details or {},
    )


def mark_task_disabled(name: str, reason: str) -> None:
    _update(name, state="disabled", disabled_reason=reason)


def task_health_snapshot() -> dict[str, dict]:
    with _lock:
        return {name: dict(value) for name, value in sorted(_health.items())}


def reset_task_health() -> None:
    """Test helper; production code never clears health history."""
    with _lock:
        _health.clear()


async def _send_alert(alert: Callable[[dict], object] | None, report: dict) -> None:
    if alert is None:
        return
    try:
        result = await asyncio.to_thread(alert, report)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logging.error(
            "Background task alert delivery failed category=%s",
            safe_exception(exc),
        )


async def supervise_background_task(
    name: str,
    factory: Callable[[], Awaitable[None]],
    *,
    alert: Callable[[dict], object] | None = None,
    restart: bool = True,
    alert_after_failures: int = 3,
    initial_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 60.0,
) -> None:
    """Run one task, restart crashes with backoff, and expose health state."""
    restart_count = 0
    initial_backoff = max(0.01, float(initial_backoff_seconds))
    backoff = initial_backoff
    while True:
        _update(
            name,
            state="running",
            last_started_at=_now(),
            restart_count=restart_count,
        )
        try:
            await factory()
            if not restart:
                _update(name, state="completed", last_success_at=_now())
                return
            raise RuntimeError("background task returned unexpectedly")
        except asyncio.CancelledError:
            _update(name, state="stopped", stopped_at=_now())
            raise
        except Exception as exc:
            with _lock:
                previous = dict(_health.get(name) or {})
            failures = int(previous.get("consecutive_failures") or 0) + 1
            if failures == 1:
                backoff = initial_backoff
            restart_count += 1
            _update(
                name,
                state="restarting" if restart else "failed",
                last_failure_at=_now(),
                last_error_type=type(exc).__name__,
                consecutive_failures=failures,
                restart_count=restart_count,
                next_retry_seconds=backoff if restart else None,
            )
            logging.error(
                "Background task %s crashed category=%s",
                name,
                safe_exception(exc),
            )
            alert_threshold = 1 if not restart else max(1, int(alert_after_failures))
            if failures == alert_threshold:
                await _send_alert(
                    alert,
                    {
                        "source": "server",
                        "type": "background_task_repeated_failure",
                        "phase": "background",
                        "message": f"Background task {name} failed repeatedly.",
                        "path": name,
                        "details": f"error_type={type(exc).__name__}; failures={failures}",
                    },
                )
            if not restart:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_seconds)
