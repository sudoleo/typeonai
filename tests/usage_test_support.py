"""Thread-safe Firestore fake for run-usage endpoint tests."""

import threading
from copy import deepcopy

from app.services.usage_repository import FirestoreUsageRepository


class FakeSnapshot:
    def __init__(self, data):
        self.exists = data is not None
        self._data = deepcopy(data)

    def to_dict(self):
        return deepcopy(self._data)


class FakeDocumentReference:
    def __init__(self, db, path):
        self._db = db
        self.path = tuple(path)

    def collection(self, name):
        return FakeCollectionReference(self._db, self.path + (name,))

    def get(self, transaction=None):
        if transaction is not None:
            return transaction.get(self)
        with self._db.lock:
            return FakeSnapshot(self._db.documents.get(self.path))


class FakeCollectionReference:
    def __init__(self, db, path):
        self._db = db
        self.path = tuple(path)

    def document(self, document_id):
        return FakeDocumentReference(self._db, self.path + (document_id,))


class FakeTransaction:
    def __init__(self, db):
        self._db = db
        self._writes = []

    def get(self, ref):
        return FakeSnapshot(self._db.documents.get(ref.path))

    def set(self, ref, data, merge=False):
        self._writes.append(("set", ref.path, deepcopy(data), merge))

    def update(self, ref, data):
        self._writes.append(("update", ref.path, deepcopy(data), False))

    def commit(self):
        for operation, path, data, merge in self._writes:
            current = deepcopy(self._db.documents.get(path))
            if operation == "update":
                if current is None:
                    raise AssertionError(f"Cannot update missing fake document: {path}")
                current.update(data)
                self._db.documents[path] = current
            elif merge and current is not None:
                current.update(data)
                self._db.documents[path] = current
            else:
                self._db.documents[path] = data


class FakeFirestore:
    def __init__(self):
        self.documents = {}
        self.lock = threading.RLock()

    def collection(self, name):
        return FakeCollectionReference(self, (name,))

    def transaction(self):
        return FakeTransaction(self)

    def run_transaction(self, operation):
        with self.lock:
            transaction = FakeTransaction(self)
            result = operation(transaction)
            transaction.commit()
            return result


def make_usage_repository():
    db = FakeFirestore()
    return FirestoreUsageRepository(db, transaction_runner=db.run_transaction), db
