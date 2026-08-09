import os

import pytest

# Die Playwright-E2E-Suite (tests/e2e/) braucht einen Chromium-Browser und
# startet einen eigenen uvicorn-Server; sie darf die schnelle Backend-Baseline
# ("python -m pytest tests") nicht mit einsammeln. Lauf nur mit RUN_E2E=1
# (siehe tests/e2e/README.md).
if os.environ.get("RUN_E2E") != "1":
    # The regular suite uses in-memory repository fakes and must be collectable
    # on a clean CI checkout without the gitignored production credential.
    os.environ.setdefault("UNIT_TEST_MODE", "1")
    collect_ignore = ["e2e"]


@pytest.fixture(autouse=True)
def _clear_resolved_chat_context_cache():
    """Der /ask_*-Kontext-Cache ist prozessweit und lebt 120 s — ohne Reset
    koennte ein Test den aufgeloesten Kontext des naechsten beantworten."""
    from app.api.routers import chat as chat_router

    chat_router.resolved_context_cache.clear()
    yield
    chat_router.resolved_context_cache.clear()
