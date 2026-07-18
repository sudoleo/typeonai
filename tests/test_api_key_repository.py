from __future__ import annotations

from copy import deepcopy

import pytest

from app.services.api_key_repository import (
    FirestoreApiKeyRepository,
    InvalidApiKey,
)


class Snapshot:
    def __init__(self, data):
        self.exists = data is not None
        self._data = deepcopy(data)

    def to_dict(self):
        return deepcopy(self._data)


class Document:
    def __init__(self, db, key):
        self.db = db
        self.key = key

    def set(self, data):
        self.db[self.key] = deepcopy(data)

    def get(self):
        return Snapshot(self.db.get(self.key))

    def update(self, data):
        self.db[self.key].update(deepcopy(data))


class Collection:
    def __init__(self, db):
        self.db = db

    def document(self, key):
        return Document(self.db, key)


class FakeDb:
    def __init__(self):
        self.documents = {}

    def collection(self, name):
        assert name == "api_consensus_keys"
        return Collection(self.documents)


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


def test_revoked_key_cannot_authenticate():
    db = FakeDb()
    repo = FirestoreApiKeyRepository(db)
    issued = repo.issue("user-1")
    repo.revoke(issued["key_id"])

    with pytest.raises(InvalidApiKey):
        repo.authenticate(issued["api_key"])
