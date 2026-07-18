from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta

import pytest

from app.services.api_run_repository import (
    ApiRunConflict,
    FirestoreApiRunRepository,
    idempotency_hash,
)


class Snapshot:
    def __init__(self, ref, data):
        self.reference = ref
        self.id = ref.id
        self.exists = data is not None
        self._data = deepcopy(data)

    def to_dict(self):
        return deepcopy(self._data)


class Document:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)
        self.id = self.path[-1]

    def collection(self, name):
        return Collection(self.db, self.path + (name,))

    def get(self, transaction=None):
        if transaction:
            return transaction.get(self)
        with self.db.lock:
            return Snapshot(self, self.db.documents.get(self.path))


class Collection:
    def __init__(self, db, path):
        self.db = db
        self.path = tuple(path)

    def document(self, doc_id):
        return Document(self.db, self.path + (doc_id,))


class Transaction:
    def __init__(self, db):
        self.db = db
        self.writes = []

    def get(self, ref):
        return Snapshot(ref, self.db.documents.get(ref.path))

    def set(self, ref, data, merge=False):
        self.writes.append(("set", ref.path, deepcopy(data), merge))

    def update(self, ref, data):
        self.writes.append(("update", ref.path, deepcopy(data), False))

    def delete(self, ref):
        self.writes.append(("delete", ref.path, None, False))

    def commit(self):
        for operation, path, data, merge in self.writes:
            if operation == "delete":
                self.db.documents.pop(path, None)
            elif operation == "update":
                self.db.documents[path].update(data)
            elif merge and path in self.db.documents:
                self.db.documents[path].update(data)
            else:
                self.db.documents[path] = data


class FakeDb:
    def __init__(self):
        self.documents = {}
        self.lock = threading.RLock()

    def collection(self, name):
        return Collection(self, (name,))

    def run_transaction(self, operation):
        with self.lock:
            tx = Transaction(self)
            result = operation(tx)
            tx.commit()
            return result


def make_repo():
    db = FakeDb()
    return FirestoreApiRunRepository(db, transaction_runner=db.run_transaction), db


def create(repo, question="Why?"):
    return repo.create_or_get(
        uid="user-1",
        api_key_id="a" * 64,
        idempotency_key="request-123",
        request_payload={"question": question, "deep_think": False},
        model_plan={"providers": {"openai": "model"}, "consensus_model": "OpenAI"},
        is_pro=False,
    )


def test_create_is_idempotent_and_never_stores_plaintext_key():
    repo, db = make_repo()
    first, created = create(repo)
    second, created_again = create(repo)

    assert created is True
    assert created_again is False
    assert second["run_id"] == first["run_id"]
    serialized = repr(db.documents)
    assert "request-123" not in serialized
    assert idempotency_hash("request-123") in serialized


def test_same_key_with_different_request_conflicts():
    repo, _db = make_repo()
    create(repo)
    with pytest.raises(ApiRunConflict):
        create(repo, question="Different")


def test_only_one_concurrent_worker_can_claim_running():
    repo, _db = make_repo()
    run, _ = create(repo)
    repo.mark_reserved(run["run_id"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda worker: repo.claim_running(run["run_id"], worker)[1],
                [f"worker-{i}" for i in range(8)],
            )
        )

    assert results.count(True) == 1
    assert repo.get(run["run_id"])["status"] == "running"


def test_full_state_sequence_and_terminal_idempotency():
    repo, _db = make_repo()
    run, _ = create(repo)
    reserved, changed = repo.mark_reserved(run["run_id"])
    running, claimed = repo.claim_running(run["run_id"], "worker")
    succeeded = repo.succeed(run["run_id"], {"consensus_response": "ok"})
    succeeded_again = repo.succeed(run["run_id"], {"ignored": True})

    assert changed and reserved["status"] == "reserved"
    assert claimed and running["status"] == "running"
    assert succeeded["status"] == "succeeded"
    assert succeeded_again["result"] == {"consensus_response": "ok"}


def test_expired_running_lease_fails_without_requeueing():
    repo, _db = make_repo()
    run, _ = create(repo)
    repo.mark_reserved(run["run_id"])
    running, _ = repo.claim_running(run["run_id"], "worker")

    changed = repo.fail_if_lease_expired(
        run["run_id"], now=running["lease_expires_at"] + timedelta(seconds=1)
    )

    failed = repo.get(run["run_id"])
    assert changed is True
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "worker_interrupted"
    assert repo.fail_if_lease_expired(run["run_id"]) is False
