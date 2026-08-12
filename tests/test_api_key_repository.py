from __future__ import annotations

from copy import deepcopy

import pytest

from app.services.api_key_repository import (
    DEFAULT_API_KEY_SCOPES,
    FirestoreApiKeyRepository,
    InvalidApiKey,
)
from app.services import persistence_guard


class Snapshot:
    def __init__(self, data):
        self.exists = data is not None
        self._data = deepcopy(data)

    def to_dict(self):
        return deepcopy(self._data)


class Document:
    def __init__(self, db, collection, key):
        self.db = db
        self.collection = collection
        self.key = key

    @property
    def storage_key(self):
        if self.collection == "api_consensus_keys":
            return self.key
        return (self.collection, self.key)

    def set(self, data):
        self.db[self.storage_key] = deepcopy(data)

    def get(self, transaction=None):
        return Snapshot(self.db.get(self.storage_key))

    def update(self, data):
        self.db[self.storage_key].update(deepcopy(data))


class Collection:
    def __init__(self, db, name):
        self.db = db
        self.name = name

    def document(self, key):
        return Document(self.db, self.name, key)


class Transaction:
    def set(self, ref, data):
        ref.set(data)


class FakeDb:
    def __init__(self):
        self.documents = {}

    def collection(self, name):
        return Collection(self.documents, name)

    def run_transaction(self, operation):
        return operation(Transaction())


def test_plaintext_key_is_returned_once_but_never_persisted():
    db = FakeDb()
    repo = FirestoreApiKeyRepository(db)
    issued = repo.issue("user-1", label="CI")

    assert issued["api_key"].startswith("cns_live_")
    assert issued["key_id"] in db.documents
    assert issued["api_key"] not in repr(db.documents)
    assert "api_key" not in db.documents[issued["key_id"]]
    identity = repo.authenticate(issued["api_key"])
    assert identity.uid == "user-1"
    assert identity.key_id == issued["key_id"]
    assert identity.scopes == tuple(sorted(DEFAULT_API_KEY_SCOPES))
    first_last_used = db.documents[issued["key_id"]]["last_used_at"]
    repo.authenticate(issued["api_key"])
    assert db.documents[issued["key_id"]]["last_used_at"] == first_last_used


def test_account_deletion_tombstone_fences_api_key_issue():
    db = FakeDb()
    db.documents[(persistence_guard.ACCOUNT_DELETION_JOBS_COLLECTION, "user-1")] = {
        "status": "pending"
    }
    repo = FirestoreApiKeyRepository(db)

    with pytest.raises(persistence_guard.AccountDeletionInProgress):
        repo.issue("user-1", label="must-not-survive")

    assert not any(isinstance(key, str) for key in db.documents)


def test_revoked_key_cannot_authenticate():
    db = FakeDb()
    repo = FirestoreApiKeyRepository(db)
    issued = repo.issue("user-1")
    repo.revoke(issued["key_id"])

    with pytest.raises(InvalidApiKey):
        repo.authenticate(issued["api_key"])


def test_scopes_are_validated_persisted_and_authenticated():
    db = FakeDb()
    repo = FirestoreApiKeyRepository(db)
    issued = repo.issue(
        "admin-1", scopes=["share:index", "consensus:run", "share:index"]
    )

    assert issued["scopes"] == ["consensus:run", "share:index"]
    identity = repo.authenticate(issued["api_key"])
    assert identity.has_scope("share:index")
    assert not identity.has_scope("share:write")

    with pytest.raises(ValueError, match="Unknown API key scope"):
        repo.issue("user-1", scopes=["root"])


def test_legacy_key_without_scopes_gets_safe_defaults():
    db = FakeDb()
    repo = FirestoreApiKeyRepository(db)
    issued = repo.issue("user-1")
    db.documents[issued["key_id"]].pop("scopes")

    identity = repo.authenticate(issued["api_key"])
    assert identity.has_scope("consensus:run")
    assert identity.has_scope("share:write")
    assert not identity.has_scope("share:index")
