"""PII-free correlation context and bounded in-process operational metrics."""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import threading
import time
from contextlib import contextmanager


_CORRELATION_ID = contextvars.ContextVar("correlation_id", default="-")
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9_.:-]")
_LOCK = threading.Lock()
_METRICS: dict[str, dict] = {}


def correlation_id() -> str:
    return _CORRELATION_ID.get()


def new_correlation_id(prefix: str = "req") -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


@contextmanager
def correlation_scope(value: str | None = None, *, prefix: str = "job"):
    token = _CORRELATION_ID.set(value or new_correlation_id(prefix))
    try:
        yield correlation_id()
    finally:
        _CORRELATION_ID.reset(token)


class CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id()
        return True


def configure_logging() -> None:
    root = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [corr=%(correlation_id)s] %(name)s: %(message)s"
    )
    for handler in root.handlers:
        handler.addFilter(CorrelationFilter())
        handler.setFormatter(formatter)


def _label(value: str) -> str:
    return _SAFE_LABEL.sub("_", str(value or "unknown"))[:80]


def record_metric(
    family: str,
    name: str,
    *,
    duration_ms: float = 0,
    outcome: str = "success",
    processed: int = 0,
    retries: int = 0,
) -> None:
    key = f"{_label(family)}:{_label(name)}"
    with _LOCK:
        item = _METRICS.setdefault(key, {
            "count": 0,
            "failures": 0,
            "timeouts": 0,
            "cancellations": 0,
            "retries": 0,
            "processed": 0,
            "duration_ms_total": 0,
            "duration_ms_max": 0,
        })
        item["count"] += 1
        item["failures"] += int(outcome == "failure")
        item["timeouts"] += int(outcome == "timeout")
        item["cancellations"] += int(outcome == "cancelled")
        item["retries"] += max(0, int(retries))
        item["processed"] += max(0, int(processed))
        duration = max(0, int(duration_ms))
        item["duration_ms_total"] += duration
        item["duration_ms_max"] = max(item["duration_ms_max"], duration)


def metrics_snapshot() -> dict:
    with _LOCK:
        return {key: dict(value) for key, value in sorted(_METRICS.items())}


def safe_exception(exc: BaseException) -> str:
    """Stable error category without exception messages or response bodies."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, (int, str)):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    # Numeric HTTP/gRPC status values are operational metadata. Arbitrary
    # string ``code`` attributes are not: SDKs (and application exceptions)
    # may populate them with upstream text or identifiers.
    suffix = f":{status}" if isinstance(status, int) else ""
    return f"{type(exc).__name__}{suffix}"


class CorrelationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status_code = 500
        with correlation_scope(prefix="req") as current:
            async def send_with_correlation(message):
                nonlocal status_code
                if message.get("type") == "http.response.start":
                    status_code = int(message.get("status") or 500)
                    headers = list(message.get("headers") or [])
                    headers.append((b"x-correlation-id", current.encode("ascii")))
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, send_with_correlation)
            finally:
                record_metric(
                    "http",
                    f"{scope.get('method', 'GET')}:{status_code // 100}xx",
                    duration_ms=(time.monotonic() - started) * 1000,
                    outcome="failure" if status_code >= 500 else "success",
                )
