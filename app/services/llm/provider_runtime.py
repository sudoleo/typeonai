"""Shared transport budgets and cooperative cancellation for LLM providers.

Provider calls deliberately have no automatic transport retry. A logical
operation claim authorizes one paid call; schema-level fallbacks and retries
remain explicit in the consensus layer and therefore auditable.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from typing import Callable, Iterator

import httpx
import openai


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


PROVIDER_CONNECT_TIMEOUT_SECONDS = _bounded_env_float(
    "PROVIDER_CONNECT_TIMEOUT_SECONDS", 10.0, 1.0, 30.0
)
PROVIDER_READ_TIMEOUT_SECONDS = _bounded_env_float(
    "PROVIDER_READ_TIMEOUT_SECONDS", 120.0, 10.0, 300.0
)
PROVIDER_KEY_CHECK_TIMEOUT_SECONDS = 15.0


def provider_http_timeout(read_seconds: float | None = None) -> tuple[float, float]:
    read_budget = float(read_seconds or PROVIDER_READ_TIMEOUT_SECONDS)
    return (min(PROVIDER_CONNECT_TIMEOUT_SECONDS, read_budget), read_budget)


PROVIDER_HTTP_TIMEOUT = provider_http_timeout()
PROVIDER_SDK_MAX_RETRIES = 0


def openai_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
) -> openai.OpenAI:
    """Build an OpenAI-compatible client with explicit, shared budgets."""
    read_budget = float(timeout_seconds or PROVIDER_READ_TIMEOUT_SECONDS)
    kwargs = {
        "api_key": api_key,
        "timeout": httpx.Timeout(
            read_budget,
            connect=min(PROVIDER_CONNECT_TIMEOUT_SECONDS, read_budget),
        ),
        "max_retries": PROVIDER_SDK_MAX_RETRIES,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


class ProviderCancelled(Exception):
    """Raised inside a provider producer after its downstream client left."""


class ProviderCancellation:
    """Thread-safe cancellation signal that closes active streaming resources."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._closers: set[Callable[[], None]] = set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProviderCancelled("provider stream cancelled")

    def register(self, resource) -> Callable[[], None]:
        close = getattr(resource, "close", None)
        if not callable(close):
            return lambda: None
        with self._lock:
            if self.cancelled:
                close_now = True
            else:
                self._closers.add(close)
                close_now = False
        if close_now:
            _close_safely(close)

        def unregister() -> None:
            with self._lock:
                self._closers.discard(close)

        return unregister

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            closers = tuple(self._closers)
            self._closers.clear()
        for close in closers:
            _close_safely(close)


def _close_safely(close: Callable[[], None]) -> None:
    try:
        close()
    except Exception:
        # Cancellation is best effort. The producer still observes the event.
        pass


_thread_context = threading.local()


@contextmanager
def bind_provider_cancellation(
    cancellation: ProviderCancellation,
) -> Iterator[ProviderCancellation]:
    previous = getattr(_thread_context, "cancellation", None)
    _thread_context.cancellation = cancellation
    try:
        yield cancellation
    finally:
        _thread_context.cancellation = previous


def current_provider_cancellation() -> ProviderCancellation | None:
    return getattr(_thread_context, "cancellation", None)


def raise_if_provider_cancelled() -> None:
    cancellation = current_provider_cancellation()
    if cancellation:
        cancellation.raise_if_cancelled()


@contextmanager
def managed_provider_resource(resource):
    """Register a response/SDK stream for cancellation and always close it."""
    cancellation = current_provider_cancellation()
    unregister = cancellation.register(resource) if cancellation else (lambda: None)
    try:
        if cancellation:
            cancellation.raise_if_cancelled()
        yield resource
    finally:
        unregister()
        close = getattr(resource, "close", None)
        if callable(close):
            _close_safely(close)
