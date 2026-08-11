"""Failure injection for durable, per-area account deletion retries."""

from types import SimpleNamespace
from unittest.mock import patch

from firebase_admin import firestore

from app.services.account_deletion import FirestoreAccountDeletion


class Snapshot:
    def __init__(self, data, reference=None, document_id="owner"):
        self.exists = data is not None
        self._data = dict(data or {})
        self.reference = reference
        self.id = document_id

    def to_dict(self):
        return dict(self._data)


class MemoryDocument:
    def __init__(self):
        self.data = None
        self.deleted = 0

    def get(self):
        return Snapshot(self.data, self)

    def set(self, values, merge=False):
        current = dict(self.data or {}) if merge else {}
        for key, value in values.items():
            if value is firestore.DELETE_FIELD:
                current.pop(key, None)
            else:
                current[key] = value
        self.data = current

    def delete(self):
        self.deleted += 1
        self.data = None

    def collection(self, _name):
        return EmptyCollection()


class EmptyCollection:
    def stream(self):
        return iter(())


class Collection:
    def __init__(self, db, name):
        self.db = db
        self.name = name

    def document(self, document_id):
        return self.db.documents.setdefault((self.name, document_id), MemoryDocument())


class MemoryDatabase:
    def __init__(self):
        self.documents = {}

    def collection(self, name):
        return Collection(self, name)


def test_failed_area_remains_pending_and_only_that_area_is_retried(monkeypatch):
    db = MemoryDatabase()
    service = FirestoreAccountDeletion(db)
    calls = {
        "api": 0,
        "subcollections": 0,
        "chats": 0,
        "uid_queries": 0,
        "shares": 0,
        "pending": 0,
        "watches": 0,
        "watch_indexes": 0,
        "guards": 0,
        "follows": 0,
        "auth": 0,
    }

    monkeypatch.setattr(
        service,
        "_delete_persistence_guards",
        lambda uid, email: calls.__setitem__("guards", calls["guards"] + 1),
    )
    monkeypatch.setattr(
        service,
        "_delete_api_access",
        lambda uid: calls.__setitem__("api", calls["api"] + 1),
    )
    monkeypatch.setattr(
        service,
        "_delete_user_subcollections",
        lambda uid: calls.__setitem__("subcollections", calls["subcollections"] + 1),
    )
    monkeypatch.setattr(
        service,
        "_delete_uid_queries",
        lambda uid: calls.__setitem__("uid_queries", calls["uid_queries"] + 1),
    )

    def shares(uid):
        calls["shares"] += 1
        if calls["shares"] == 1:
            raise RuntimeError("injected share failure")

    monkeypatch.setattr(service, "_delete_owned_shares", shares)

    def delete_query(collection, field, value):
        key = "pending" if collection == "pending_results" else "watches"
        calls[key] += 1

    monkeypatch.setattr(service, "_delete_query", delete_query)
    monkeypatch.setattr(
        service,
        "_delete_orphan_watches",
        lambda uid: calls.__setitem__("watches", calls["watches"] + 1),
    )
    monkeypatch.setattr(
        service,
        "_delete_watch_indexes",
        lambda uid: calls.__setitem__(
            "watch_indexes", calls["watch_indexes"] + 1
        ),
    )
    monkeypatch.setattr(
        service,
        "_delete_email_follows",
        lambda email: calls.__setitem__("follows", calls["follows"] + 1),
    )
    monkeypatch.setattr(
        service,
        "_delete_auth_user",
        lambda uid: calls.__setitem__("auth", calls["auth"] + 1),
    )

    class Chats:
        def __init__(self, _db):
            pass

        def delete_all_chats(self, uid):
            calls["chats"] += 1

    with patch("app.services.account_deletion.ChatStore", Chats):
        service.start("owner", email="Owner@Example.test")
        first_errors = service.cleanup_uid("owner")
        second_errors = service.cleanup_uid("owner")

    assert first_errors == ["owned_shares"]
    assert second_errors == []
    assert calls["shares"] == 2
    assert calls["api"] == 1
    assert calls["subcollections"] == 1
    assert calls["chats"] == 1
    assert calls["watches"] == 1
    assert calls["watch_indexes"] == 1
    assert calls["guards"] == 1
    assert calls["follows"] == 1
    assert calls["auth"] == 1
    job = db.collection("account_deletion_jobs").document("owner").data
    assert job["status"] == "completed"
    assert job["cleanup_pending"] is False
    assert "email" not in job
    assert all(job["completed_areas"].values())
