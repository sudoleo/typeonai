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
    deletion = SimpleNamespace(
        started=[],
        cleaned=[],
        errors=[],
    )
    deletion.start = lambda uid, email="": deletion.started.append((uid, email))
    deletion.cleanup_uid = lambda uid: deletion.cleaned.append(uid) or deletion.errors
    monkeypatch.setattr(users_router, "account_deletion", deletion)
    monkeypatch.setattr(
        users_router.auth,
        "get_user",
        lambda uid: SimpleNamespace(email="owner@example.test"),
    )

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(users_router.router)
    return TestClient(app), deletion


def test_delete_account_cascades_into_the_owner_chats(delete_account_api):
    client, deletion = delete_account_api
    response = client.post("/delete_account", headers=AUTH, json={})

    assert response.status_code == 200
    assert deletion.started == [(UID, "owner@example.test")]
    assert deletion.cleaned == [UID]


def test_delete_account_reports_a_failed_chat_cascade_instead_of_hiding_it(
    delete_account_api, caplog
):
    client, deletion = delete_account_api
    deletion.errors = ["chats"]
    response = client.post("/delete_account", headers=AUTH, json={})

    assert response.status_code == 202
    assert response.json()["cleanup_pending"] is True
    assert response.json()["status"] == "cleanup_pending"


def test_delete_account_reports_durable_job_when_immediate_cleanup_crashes(
    delete_account_api,
):
    client, deletion = delete_account_api

    def fail_cleanup(_uid):
        raise RuntimeError("temporary Firestore outage")

    deletion.cleanup_uid = fail_cleanup
    response = client.post("/delete_account", headers=AUTH, json={})

    assert response.status_code == 202
    assert response.json()["status"] == "cleanup_pending"
    assert deletion.started == [(UID, "owner@example.test")]
