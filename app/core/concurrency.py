"""Worker-thread budget for the synchronous streaming endpoints.

``/ask_*`` and ``/consensus`` are declared ``def`` (not ``async def``) and hand
back a ``StreamingResponse`` wrapped around a *synchronous* generator. Starlette
therefore pulls every SSE chunk through ``iterate_in_threadpool``, so an
in-flight model stream occupies one anyio worker thread for as long as the
provider keeps writing -- up to ``PROVIDER_READ_TIMEOUT_SECONDS`` (120s by
default).

anyio's default budget is 40 threads. One consensus run fans out to six
providers at once, so the pool saturated at roughly six concurrent users and
every further request queued behind a free thread instead of failing loudly.
Raising the limiter is the cheap half of the fix; the structural half is
replacing ``requests`` with an async client so streams stop costing a thread at
all. That rewrite touches every engine, so it stays deliberately out of scope
here.

The budget is applied once per event loop at startup. ``anyio``'s limiter lives
in a ``RunVar``, i.e. it is bound to the running loop -- setting it at import
time would land on the wrong loop (or none at all).
"""

from __future__ import annotations

import logging
import os

import anyio.to_thread


logger = logging.getLogger(__name__)

# Six provider streams per run, plus headroom for the synchronous Firestore and
# mailer work that shares the same pool. 160 carries ~24 concurrent runs.
DEFAULT_MAX_WORKER_THREADS = 160
MIN_MAX_WORKER_THREADS = 40
MAX_MAX_WORKER_THREADS = 512


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def configured_max_worker_threads() -> int:
    """Resolve the deployment's thread budget, clamped to a sane range."""

    return _bounded_env_int(
        "MAX_WORKER_THREADS",
        DEFAULT_MAX_WORKER_THREADS,
        MIN_MAX_WORKER_THREADS,
        MAX_MAX_WORKER_THREADS,
    )


def apply_worker_thread_budget() -> int:
    """Raise the running loop's thread limiter. Returns the applied value.

    Never lowers an already larger budget: a deployment that tuned the limiter
    by other means keeps its value.
    """

    budget = configured_max_worker_threads()
    limiter = anyio.to_thread.current_default_thread_limiter()
    if limiter.total_tokens < budget:
        limiter.total_tokens = budget
    logger.info(
        "Worker thread budget: %s (concurrent provider streams: ~%s runs)",
        limiter.total_tokens,
        int(limiter.total_tokens // 6),
    )
    return int(limiter.total_tokens)
