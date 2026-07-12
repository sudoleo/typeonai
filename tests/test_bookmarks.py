from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import bookmarks as bookmarks_router
from app.core.rate_limit import limiter


class FakeSnapshot:
    def __init__(self, bookmark_id, data):
        self.id = bookmark_id
        self._data = data

    def to_dict(self):
        return self._data.copy()


class FakeBookmarkRef:
    def __init__(self, bookmark_id, data):
        self.id = bookmark_id
        self.data = data

    def set(self, incoming, merge=False):
        assert merge is True
        for key, value in incoming.items():
            if key == "responses":
                self.data.setdefault("responses", {}).update(value)
            else:
                self.data[key] = value

    def get(self):
        return FakeSnapshot(self.id, self.data)


class FakeFirestore:
    def __init__(self, bookmark_ref):
        self.bookmark_ref = bookmark_ref

    def collection(self, name):
        if name == "users":
            return self
        assert name == "bookmarks"
        return self

    def document(self, document_id):
        if document_id == "uid-1":
            return self
        assert document_id == self.bookmark_ref.id
        return self.bookmark_ref


def test_consensus_save_returns_complete_merged_bookmark():
    bookmark_id = "V2h5Pw__"
    bookmark_ref = FakeBookmarkRef(
        bookmark_id,
        {
            "query": "Why?",
            "mode": "Normal",
            "responses": {"OpenAI": "Existing model answer"},
        },
    )
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)

    with patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"), patch.object(
        bookmarks_router, "db_firestore", fake_db
    ):
        response = TestClient(app).post(
            "/bookmark/consensus",
            json={
                "id_token": "token",
                "question": "Why?",
                "consensusText": "Merged consensus",
                "differencesText": "Merged differences",
            },
        )

    assert response.status_code == 200
    bookmark = response.json()["bookmark"]
    assert bookmark["id"] == bookmark_id
    assert bookmark["query"] == "Why?"
    assert bookmark["responses"] == {
        "OpenAI": "Existing model answer",
        "consensus": "Merged consensus",
        "differences": "Merged differences",
    }
