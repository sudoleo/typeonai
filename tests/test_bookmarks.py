from pathlib import Path
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

    @property
    def exists(self):
        return self._data is not None


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


class FakeListSnapshot:
    def __init__(self, bookmark_id, data):
        self.id = bookmark_id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data or {})


class FakeListRef:
    def __init__(self, collection, bookmark_id):
        self.collection = collection
        self.id = bookmark_id

    def get(self):
        return FakeListSnapshot(self.id, self.collection.items.get(self.id))


class FakeListQuery:
    def __init__(self, collection):
        self.collection = collection
        self.after_id = None
        self.page_limit = None

    def start_after(self, snapshot):
        self.after_id = snapshot.id
        return self

    def limit(self, value):
        self.page_limit = value
        return self

    def stream(self):
        ids = list(self.collection.items)
        if self.after_id in ids:
            ids = ids[ids.index(self.after_id) + 1:]
        if self.page_limit is not None:
            ids = ids[:self.page_limit]
        return [FakeListSnapshot(item_id, self.collection.items[item_id]) for item_id in ids]


class FakeListCollection:
    def __init__(self, items):
        self.items = items

    def order_by(self, *_args, **_kwargs):
        return FakeListQuery(self)

    def document(self, bookmark_id):
        return FakeListRef(self, bookmark_id)


class FakeOwnerDocument:
    def __init__(self, database, uid):
        self.database = database
        self.uid = uid

    def collection(self, name):
        assert name == "bookmarks"
        return FakeListCollection(self.database.by_uid.get(self.uid, {}))


class FakeUsersCollection:
    def __init__(self, database):
        self.database = database

    def document(self, uid):
        return FakeOwnerDocument(self.database, uid)


class FakeBookmarkDatabase:
    def __init__(self, by_uid):
        self.by_uid = by_uid

    def collection(self, name):
        assert name == "users"
        return FakeUsersCollection(self)


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

    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(
            bookmarks_router.share_snapshots,
            "pending_result_is_available",
            return_value=True,
        ),
    ):
        response = TestClient(app).post(
            "/bookmark/consensus",
            json={
                "id_token": "token",
                "question": "Why?",
                "consensusText": "Merged consensus",
                "differencesText": "Merged differences",
                "resultId": "N" * 16,
                "consensusModel": "Gemini-Pro",
                "modelLabels": {"OpenAI": "GPT-5.4 mini"},
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
    assert bookmark["share_result_id"] == "N" * 16
    assert bookmark["consensus_model"] == "Gemini-Pro"
    assert bookmark["model_labels"] == {"OpenAI": "GPT-5.4 mini"}


def test_legacy_consensus_bookmark_gets_new_share_result():
    bookmark_id = "V2h5Pw__"
    bookmark_ref = FakeBookmarkRef(
        bookmark_id,
        {
            "query": "Why?",
            "responses": {
                "OpenAI": "Existing model answer",
                "Gemini": "Another model answer",
                "consensus": "Stored consensus",
                "differences": "Stored differences",
            },
            "sources": [{"id": "S1", "url": "https://example.test/source"}],
        },
    )
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    new_result_id = "R" * 16

    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(
            bookmarks_router.share_snapshots,
            "pending_result_is_available",
            return_value=False,
        ),
        patch.object(
            bookmarks_router.share_snapshots,
            "save_pending_result",
            return_value=new_result_id,
        ) as save_pending,
    ):
        response = TestClient(app).post(
            "/bookmark/consensus/share-result",
            json={"id_token": "token", "bookmarkId": bookmark_id},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success", "result_id": new_result_id, "created": True,
    }
    assert bookmark_ref.data["share_result_id"] == new_result_id
    payload = save_pending.call_args.args[0]
    assert payload["owner_uid"] == "uid-1"
    assert payload["question"] == "Why?"
    assert payload["consensus_md"] == "Stored consensus"
    assert payload["included_models"] == ["OpenAI", "Google Gemini"]


def test_consensus_bookmark_reuses_live_share_result():
    bookmark_id = "V2h5Pw__"
    existing_result_id = "E" * 16
    bookmark_ref = FakeBookmarkRef(
        bookmark_id,
        {
            "query": "Why?",
            "share_result_id": existing_result_id,
            "responses": {"consensus": "Stored consensus"},
        },
    )
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)

    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(
            bookmarks_router.share_snapshots,
            "pending_result_is_available",
            return_value=True,
        ),
        patch.object(bookmarks_router.share_snapshots, "save_pending_result") as save_pending,
    ):
        response = TestClient(app).post(
            "/bookmark/consensus/share-result",
            json={"id_token": "token", "bookmarkId": bookmark_id},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success", "result_id": existing_result_id, "created": False,
    }
    save_pending.assert_not_called()


def test_bookmark_frontend_prepares_share_and_watch_result():
    root = Path(__file__).resolve().parents[1]
    firebase = (root / "static" / "firebase.js").read_text(encoding="utf-8")
    share_dialog = (root / "static" / "js" / "share-dialog.js").read_text(encoding="utf-8")
    watch = (root / "static" / "js" / "watch.js").read_text(encoding="utf-8")
    assert 'fetch("/bookmark/consensus/share-result"' in firebase
    assert "prepareBookmarkShareResult(bookmark);" in firebase
    assert "resolveCurrentShareResultId" in share_dialog
    assert "resolveCurrentShareResultId" in watch


def test_bookmark_list_is_compact_and_cursor_paginated():
    items = {
        "first_id": {"query": "First", "mode": "Normal", "responses": {"OpenAI": "large", "consensus": "full"}},
        "second_id": {"query": "Second", "responses": {"Gemini": "large"}},
        "third_id": {"query": "Third", "responses": {"Grok": "large"}},
    }
    fake_db = FakeBookmarkDatabase({"uid-1": items})
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    with patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"), \
            patch.object(bookmarks_router, "db_firestore", fake_db):
        client = TestClient(app)
        first = client.get("/bookmarks?limit=2", headers={"Authorization": "Bearer token"})
        cursor = first.json()["next_cursor"]
        second = client.get(
            "/bookmarks?limit=2&cursor=" + cursor,
            headers={"Authorization": "Bearer token"},
        )
    assert first.status_code == 200
    assert [item["id"] for item in first.json()["bookmarks"]] == ["first_id", "second_id"]
    assert "responses" not in first.json()["bookmarks"][0]
    assert first.json()["bookmarks"][0]["has_consensus"] is True
    assert second.json()["bookmarks"] == [{
        "id": "third_id", "query": "Third", "mode": "", "timestamp": None,
        "has_consensus": False, "model_count": 1, "source_count": 0,
        "attachment_count": 0,
    }]
    assert second.json()["has_more"] is False


def test_bookmark_detail_is_owner_scoped_and_frontend_loads_on_open():
    fake_db = FakeBookmarkDatabase({
        "owner": {"owned_id": {"query": "Owned", "responses": {"consensus": "Full answer"}}},
        "other": {},
    })
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)

    def verify(token):
        return "owner" if token == "owner-token" else "other"

    with patch.object(bookmarks_router, "verify_user_token", side_effect=verify), \
            patch.object(bookmarks_router, "db_firestore", fake_db):
        client = TestClient(app)
        owned = client.get("/bookmarks/owned_id", headers={"Authorization": "Bearer owner-token"})
        hidden = client.get("/bookmarks/owned_id", headers={"Authorization": "Bearer other-token"})
    assert owned.status_code == 200
    assert owned.json()["bookmark"]["responses"]["consensus"] == "Full answer"
    assert hidden.status_code == 404

    firebase = (Path(__file__).resolve().parents[1] / "static" / "firebase.js").read_text(encoding="utf-8")
    assert 'fetch("/bookmarks/" + encodeURIComponent(bookmarkId)' in firebase
    assert 'const path = "/bookmarks?limit=30"' in firebase
    assert "bookmarkDetailCache.clear()" in firebase
    assert "window.openBookmark(bookmark.id)" in firebase
