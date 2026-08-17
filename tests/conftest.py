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
def _neutral_user_memory_profile():
    """Jeder authentifizierte /ask_* liest jetzt das User-Memory-Profil.

    Ohne diesen Ersatz liefe der Read gegen den geschlossenen Loopback-Endpunkt
    aus UNIT_TEST_MODE und jeder /ask_*-Test wartete das Firestore-Budget ab.
    Der Default ist ein leeres Profil -- also exakt das Verhalten vor dem
    Feature. tests/test_user_memory.py setzt bewusst eigene Stubs darueber."""
    from app.api.routers import chat as chat_router
    from app.services.user_memory import empty_profile

    class _EmptyProfileRepository:
        def get(self, uid):
            return empty_profile()

    original = chat_router.user_memory_repository
    chat_router.user_memory_repository = _EmptyProfileRepository()
    yield
    chat_router.user_memory_repository = original


@pytest.fixture(autouse=True)
def _clear_resolved_chat_context_cache():
    """Der /ask_*-Kontext-Cache ist prozessweit und lebt 120 s — ohne Reset
    koennte ein Test den aufgeloesten Kontext des naechsten beantworten."""
    from app.api.routers import chat as chat_router

    chat_router.resolved_context_cache.clear()
    yield
    chat_router.resolved_context_cache.clear()
