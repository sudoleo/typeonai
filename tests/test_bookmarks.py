from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import bookmarks as bookmarks_router
from app.core.rate_limit import limiter
from app.core.security import CustomSecurityMiddleware
from app.services.chat_store import ChatCursorUnavailable, InvalidChatCursor


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


def test_followup_bookmark_keeps_complete_previous_turn_for_restore():
    current = "What should I prioritize first?"
    previous = "How should I plan the migration?"
    bookmark_id = "V2hhdCBzaG91bGQgSSBwcmlvcml0aXplIGZpcnN0Pw__"
    bookmark_ref = FakeBookmarkRef(bookmark_id, {"query": current, "responses": {}})
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
                "question": current,
                "previousQuestion": previous,
                "previousTurn": {
                    "question": previous,
                    "consensus": "Original consensus",
                    "differences": "Original differences",
                    "differences_data": {
                        "agreement": {"score": 82, "model_count": 6},
                        "claims": [],
                        "differences": [],
                        "best_model": None,
                        "models_compared": [],
                    },
                },
                "consensusText": "Follow-up consensus",
                "differencesText": "No material differences",
                "resultId": "R" * 16,
            },
        )

    assert response.status_code == 200
    bookmark = response.json()["bookmark"]
    assert bookmark["query"] == current
    assert bookmark["previous_question"] == previous
    assert bookmark["previous_turn"]["question"] == previous
    assert bookmark["previous_turn"]["consensus"] == "Original consensus"
    assert bookmark["previous_turn"]["differences_data"]["agreement"]["score"] == 82
    assert bookmark["share_result_id"] == ""
    meta = bookmarks_router._bookmark_meta(bookmark_id, bookmark)
    assert meta["query"] == current


def test_followup_consensus_updates_one_stable_chat_bookmark():
    bookmark_id = "stable_chat_bookmark"
    chat_id = "c" * 32
    turn_id = "2" * 32
    bookmark_ref = FakeBookmarkRef(
        bookmark_id,
        {
            "query": "First question",
            "responses": {"DeepSeek": "stale answer", "consensus": "First consensus"},
        },
    )
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)

    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
    ):
        response = TestClient(app).post(
            "/bookmark/consensus",
            json={
                "id_token": "token",
                "bookmarkId": bookmark_id,
                "chatId": chat_id,
                "turnId": turn_id,
                "question": "Second question",
                "previousQuestion": "First question",
                "consensusText": "Second consensus",
                "differencesText": "Second differences",
                "modelResponses": {
                    "OpenAI": "new answer",
                    "Gemini": "other answer",
                },
            },
        )

    assert response.status_code == 200
    bookmark = response.json()["bookmark"]
    assert bookmark["id"] == bookmark_id
    assert bookmark["query"] == "Second question"
    assert bookmark["chat_id"] == chat_id
    assert bookmark["turn_id"] == turn_id
    assert bookmark["responses"]["OpenAI"] == "new answer"
    assert bookmark["responses"]["Gemini"] == "other answer"
    assert bookmark["responses"]["DeepSeek"] == ""
    assert bookmark["responses"]["consensus"] == "Second consensus"


def test_bookmark_name_stays_the_first_question_of_the_conversation():
    """Der Sidebar-Name gehoert der ERSTEN Frage; `query` wandert weiter."""
    bookmark_id = "stable_chat_bookmark"
    chat_id = "c" * 32
    turn_id = "2" * 32
    bookmark_ref = FakeBookmarkRef(bookmark_id, {"responses": {}})
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    limiter.reset()
    app.include_router(bookmarks_router.router)

    class FakeChatStore:
        def get_chat(self, uid, requested_chat_id):
            assert (uid, requested_chat_id) == ("uid-1", chat_id)
            return {"id": chat_id, "title": "First question"}

    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(bookmarks_router, "_chat_store", return_value=FakeChatStore()),
    ):
        client = TestClient(app)
        opening = client.post(
            "/bookmark/consensus",
            json={
                "id_token": "token",
                "bookmarkId": bookmark_id,
                "chatId": chat_id,
                "turnId": turn_id,
                "question": "First question",
                "consensusText": "First consensus",
                "differencesText": "First differences",
            },
        )
        followup = client.post(
            "/bookmark/consensus",
            json={
                "id_token": "token",
                "bookmarkId": bookmark_id,
                "chatId": chat_id,
                "turnId": turn_id,
                "question": "Second question",
                "previousQuestion": "First question",
                "consensusText": "Second consensus",
                "differencesText": "Second differences",
            },
        )

    assert opening.json()["bookmark"]["title"] == "First question"
    assert followup.status_code == 200
    bookmark = followup.json()["bookmark"]
    assert bookmark["title"] == "First question"
    assert bookmark["query"] == "Second question"
    assert bookmarks_router._bookmark_meta(bookmark_id, bookmark)["title"] == (
        "First question"
    )


def test_a_legacy_bookmark_without_a_title_falls_back_to_its_question():
    meta = bookmarks_router._bookmark_meta("legacy_id", {"query": "Only question"})
    assert meta["title"] == "Only question"


def test_a_followup_without_a_chat_never_renames_its_bookmark():
    bookmark_id = "legacy_followup"
    bookmark_ref = FakeBookmarkRef(
        bookmark_id, {"query": "First", "title": "First", "responses": {}}
    )
    fake_db = FakeFirestore(bookmark_ref)
    app = FastAPI()
    app.state.limiter = limiter
    limiter.reset()
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
                "bookmarkId": bookmark_id,
                "question": "Second",
                "previousQuestion": "First",
                "consensusText": "Second consensus",
                "differencesText": "Second differences",
            },
        )

    assert response.status_code == 200
    assert response.json()["bookmark"]["title"] == "First"
    assert response.json()["bookmark"]["query"] == "Second"


def test_chat_bookmark_conversation_returns_complete_owner_bound_turns():
    bookmark_id = "stable_chat_bookmark"
    chat_id = "c" * 32
    fake_db = FakeBookmarkDatabase({
        "uid-1": {bookmark_id: {"query": "Latest", "chat_id": chat_id}},
    })

    class FakeChatStore:
        def list_turn_details(
            self, uid, requested_chat_id, *, cursor, limit, status=None
        ):
            # One owner-bound page call, filtered in the store: the failed turn
            # is never fetched in full, and the chat is verified only once.
            assert (uid, requested_chat_id, cursor, limit, status) == (
                "uid-1", chat_id, "", 50, "completed",
            )
            return {
                "turns": [
                    {
                        "id": "1" * 32,
                        "status": "completed",
                        "question": "First",
                        "consensus": "Answer",
                        "model_answers": {},
                    },
                    {
                        "id": "3" * 32,
                        "status": "completed",
                        "question": "Third",
                        "consensus": "Answer",
                        "model_answers": {},
                    },
                ],
                "next_cursor": None,
                "has_more": False,
            }

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    app.add_middleware(CustomSecurityMiddleware)
    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(bookmarks_router, "_chat_store", return_value=FakeChatStore()),
    ):
        response = TestClient(app).get(
            f"/bookmarks/{bookmark_id}/conversation?limit=50",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert [turn["question"] for turn in response.json()["turns"]] == ["First", "Third"]


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


def test_bookmark_frontend_restores_followup_as_two_complete_turns():
    root = Path(__file__).resolve().parents[1]
    firebase = (root / "static" / "firebase.js").read_text(encoding="utf-8")
    query_send = (root / "static" / "js" / "query-send.js").read_text(encoding="utf-8")
    consensus_run = (root / "static" / "js" / "consensus-run.js").read_text(encoding="utf-8")

    assert "function bookmarkDisplayQuestion(bookmark)" in firebase
    assert "window.App?.setThreadQuestion?.(displayQuestion);" in firebase
    assert "renderStoredTurns?.(materialized.historyTurns)" in firebase
    assert '"/conversation?limit=50"' in firebase
    assert "restoreCompletedChat?.(" in firebase
    assert "window.App.bookmarkSession" in firebase
    assert "question: displayQuestion" in firebase
    assert query_send.count("bookmarkPreviousQuestion)") == 6
    assert "previousQuestion: bookmarkPreviousQuestion" in consensus_run
    assert "previousTurn: bookmarkPreviousTurn" in consensus_run
    assert "buildStoredAgreement(differencesData)" in consensus_run


def test_bookmark_restore_uses_historical_model_labels_without_mutating_picker_state():
    root = Path(__file__).resolve().parents[1]
    firebase = (root / "static" / "firebase.js").read_text(encoding="utf-8")
    query_send = (root / "static" / "js" / "query-send.js").read_text(encoding="utf-8")

    helper = firebase.split("function applyBookmarkModelPresentation(bookmark) {", 1)[1]
    helper = helper.split("// Diese Funktion füllt die UI", 1)[0]

    assert "bookmark?.model_labels" in helper
    assert "labelEl.textContent = visibleLabel" in helper
    assert "select.value" not in helper
    assert "localStorage.setItem" not in helper
    assert "localStorage.removeItem" not in helper
    assert "bookmarkCitationModels = applyBookmarkModelPresentation(bookmark);" in firebase
    assert 'consensusModel: bookmark.consensus_model || ""' in firebase
    assert "window.App.updateDeepThinkText?.();" in query_send
    assert firebase.index("if (bookmark.mode)") < firebase.index(
        "bookmarkCitationModels = applyBookmarkModelPresentation(bookmark);"
    )
    assert query_send.index("window.App.updateDeepThinkText?.();") < query_send.index(
        "window.App?.consensusPipeline?.onPrepare?.();"
    )


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
        "id": "third_id", "query": "Third", "title": "Third", "mode": "",
        "timestamp": None, "has_consensus": False, "model_count": 1,
        "source_count": 0, "attachment_count": 0,
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
    assert 'const path = "/bookmarks?limit=35"' in firebase
    assert "bookmarkDetailCache.clear()" in firebase
    assert "window.openBookmark(bookmark.id)" in firebase


# ---------------------------------------------------------------------------
# Ein Bookmark ist der einzige Griff an seinem Chat. Wird es geloescht, muss
# der Chat mitgehen — sonst bleibt das Transcript unerreichbar liegen.
# ---------------------------------------------------------------------------


class DeletableBookmarkRef(FakeBookmarkRef):
    def __init__(self, bookmark_id, data):
        super().__init__(bookmark_id, data)
        self.deleted = False

    def delete(self):
        self.deleted = True


class RecordingChatStore:
    def __init__(self, error=None):
        self.deleted = []
        self.error = error

    def delete_chat(self, uid, chat_id):
        self.deleted.append((uid, chat_id))
        if self.error:
            raise self.error


def _delete_bookmark(bookmark_ref, chat_store):
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", FakeFirestore(bookmark_ref)),
        patch.object(bookmarks_router, "_chat_store", return_value=chat_store),
    ):
        return TestClient(app).request(
            "DELETE",
            "/bookmark",
            json={"id_token": "token", "bookmarkId": bookmark_ref.id},
        )


def test_deleting_a_chat_bookmark_also_deletes_its_chat():
    chat_id = "c" * 32
    bookmark_ref = DeletableBookmarkRef("chat_bookmark", {"query": "Q", "chat_id": chat_id})
    chat_store = RecordingChatStore()

    response = _delete_bookmark(bookmark_ref, chat_store)

    assert response.status_code == 200
    assert bookmark_ref.deleted is True
    assert chat_store.deleted == [("uid-1", chat_id)]


def test_deleting_a_legacy_bookmark_touches_no_chat():
    bookmark_ref = DeletableBookmarkRef("legacy_bookmark", {"query": "Q"})
    chat_store = RecordingChatStore()

    response = _delete_bookmark(bookmark_ref, chat_store)

    assert response.status_code == 200
    assert bookmark_ref.deleted is True
    assert chat_store.deleted == []


def test_a_failing_chat_cascade_does_not_fail_the_bookmark_deletion():
    chat_id = "c" * 32
    bookmark_ref = DeletableBookmarkRef("chat_bookmark", {"query": "Q", "chat_id": chat_id})
    chat_store = RecordingChatStore(error=RuntimeError("firestore unavailable"))

    response = _delete_bookmark(bookmark_ref, chat_store)

    # The bookmark is already gone; reporting a 500 would only make the user
    # retry a deletion that already happened. The chat stays reachable for the
    # account-level cascade.
    assert response.status_code == 200
    assert bookmark_ref.deleted is True


def test_unavailable_cursor_signing_is_reported_as_a_server_error():
    bookmark_id = "stable_chat_bookmark"
    chat_id = "c" * 32
    fake_db = FakeBookmarkDatabase({
        "uid-1": {bookmark_id: {"query": "Latest", "chat_id": chat_id}},
    })

    class UnavailableChatStore:
        def list_turn_details(self, *_args, **_kwargs):
            raise ChatCursorUnavailable("Chat cursor signing is not configured")

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    app.add_middleware(CustomSecurityMiddleware)
    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(bookmarks_router, "_chat_store", return_value=UnavailableChatStore()),
    ):
        response = TestClient(app).get(
            f"/bookmarks/{bookmark_id}/conversation",
            headers={"Authorization": "Bearer token"},
        )

    # A missing signing secret is a deployment problem, not a bad request.
    assert response.status_code == 503


def test_a_malformed_cursor_is_still_a_client_error():
    bookmark_id = "stable_chat_bookmark"
    chat_id = "c" * 32
    fake_db = FakeBookmarkDatabase({
        "uid-1": {bookmark_id: {"query": "Latest", "chat_id": chat_id}},
    })

    class InvalidCursorChatStore:
        def list_turn_details(self, *_args, **_kwargs):
            raise InvalidChatCursor("Invalid chat cursor")

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(bookmarks_router.router)
    app.add_middleware(CustomSecurityMiddleware)
    with (
        patch.object(bookmarks_router, "verify_user_token", return_value="uid-1"),
        patch.object(bookmarks_router, "db_firestore", fake_db),
        patch.object(bookmarks_router, "_chat_store", return_value=InvalidCursorChatStore()),
    ):
        response = TestClient(app).get(
            f"/bookmarks/{bookmark_id}/conversation?cursor=garbage",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 400
