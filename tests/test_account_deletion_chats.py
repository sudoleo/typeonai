"""Kontoloeschung muss die Chat-Kaskade ausloesen (DSGVO Art. 17).

Firestore kaskadiert nicht: ein geloeschtes chats/{id} liesse turns,
model_answers und context_versions als unerreichbare Waisen zurueck — samt der
Fragen des Nutzers und der vollstaendigen Modellantworten.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import users as users_router
from app.core.rate_limit import limiter


UID = "deleted-owner"
AUTH = {"Authorization": "Bearer owner-token"}


class EmptyQuery:
    def stream(self):
        return iter(())


class EmptyCollection:
    """Every other cleanup path finds nothing, so the test isolates chats."""

    def where(self, *_args, **_kwargs):
        return EmptyQuery()

    def stream(self):
        return iter(())

    def document(self, _document_id):
        return EmptyDocument()


class EmptyDocument:
    def collection(self, _name):
        return EmptyCollection()

    def delete(self):
        return None


class EmptyDatabase:
    def collection(self, _name):
        return EmptyCollection()


class RecordingChatStore:
    instances = []

    def __init__(self, db):
        self.db = db
        self.deleted = []
        self.error = None
        RecordingChatStore.instances.append(self)

    def delete_all_chats(self, uid):
        self.deleted.append(uid)
        if self.error:
            raise self.error


@pytest.fixture
def delete_account_api(monkeypatch):
    limiter.reset()
    RecordingChatStore.instances = []
    monkeypatch.setattr(users_router, "db_firestore", EmptyDatabase())
    monkeypatch.setattr(users_router, "verify_user_token", lambda token, **kw: UID)
    monkeypatch.setattr(users_router, "invalidate_tier_cache", lambda uid: None)
    monkeypatch.setattr(
        users_router,
        "api_account_cleanup",
        SimpleNamespace(
            block=lambda uid: None,
            cleanup_uid=lambda uid: [],
            clear_completed_block=lambda uid: None,
        ),
    )
    monkeypatch.setattr(users_router.auth, "delete_user", lambda uid: None)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(users_router.router)
    return TestClient(app)


def test_delete_account_cascades_into_the_owner_chats(delete_account_api):
    with patch.object(users_router, "ChatStore", RecordingChatStore):
        response = delete_account_api.post("/delete_account", headers=AUTH, json={})

    assert response.status_code == 200
    assert [store.deleted for store in RecordingChatStore.instances] == [[UID]]


def test_delete_account_reports_a_failed_chat_cascade_instead_of_hiding_it(
    delete_account_api, caplog
):
    class FailingChatStore(RecordingChatStore):
        def __init__(self, db):
            super().__init__(db)
            self.error = RuntimeError("firestore unavailable")

    with caplog.at_level("WARNING"):
        with patch.object(users_router, "ChatStore", FailingChatStore):
            response = delete_account_api.post("/delete_account", headers=AUTH, json={})

    # The auth account still goes away so the user is never locked out of a
    # retry, but the chat cascade lands in the same partial-cleanup report as
    # every other subcollection instead of failing silently.
    assert response.status_code == 200
    assert "chats" in caplog.text
    assert "partial cleanup" in caplog.text
