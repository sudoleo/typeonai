from __future__ import annotations

import re
import unicodedata
import copy
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import chat_history as chat_history_router
from app.core.rate_limit import limiter
from app.core.security import CustomSecurityMiddleware
from app.services import chat_store
from google.api_core.datetime_helpers import DatetimeWithNanoseconds


AUTH_OWNER = {"Authorization": "Bearer owner-token"}
AUTH_OTHER = {"Authorization": "Bearer other-token"}
TURN_PAYLOAD = {
    "question": "What changed?",
    "mode": "regular",
    "deep_search": False,
    "selected_models": ["OpenAI", "Gemini"],
    "consensus_model": "GPT Consensus",
}
PROVIDERS = ("OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok")


class FakeSnapshot:
    def __init__(self, reference, data):
        self.reference = reference
        self.id = reference.id
        self._data = None if data is None else dict(data)

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class FakeDocumentRef:
    def __init__(self, database, path):
        self.database = database
        self.path = tuple(path)
        self.id = self.path[-1]

    def collection(self, name):
        return FakeCollectionRef(self.database, (*self.path, name))

    def get(self, transaction=None):
        if transaction is not None:
            return transaction.get(self)
        self.database.read_log.append(self.path)
        return FakeSnapshot(self, self.database.documents.get(self.path))

    def set(self, data, merge=False):
        self.database.apply_set(self, data, merge=merge)

    def update(self, data):
        self.database.apply_update(self, data)


class FakeCollectionRef:
    def __init__(self, database, path):
        self.database = database
        self.path = tuple(path)

    def document(self, document_id):
        return FakeDocumentRef(self.database, (*self.path, document_id))

    def order_by(self, field, direction=None):
        return FakeQuery(self, [(field, direction)])


class FakeQuery:
    def __init__(self, collection, orders):
        self.collection = collection
        self.orders = list(orders)
        self.after_id = None
        self.after_values = None
        self.page_limit = None

    def order_by(self, field, direction=None):
        self.orders.append((field, direction))
        return self

    def start_after(self, boundary):
        if isinstance(boundary, dict):
            self.after_values = {str(key): value for key, value in boundary.items()}
        else:
            self.after_id = boundary.id
        return self

    def limit(self, value):
        self.page_limit = value
        return self

    def stream(self):
        depth = len(self.collection.path) + 1
        items = []
        for path, data in self.collection.database.documents.items():
            if len(path) == depth and path[:-1] == self.collection.path:
                items.append(FakeSnapshot(FakeDocumentRef(self.collection.database, path), data))

        for field, direction in reversed(self.orders):
            reverse = str(direction).upper().endswith("DESCENDING")
            if str(field) != "__name__":
                items.sort(
                    key=lambda snapshot: snapshot.to_dict().get(field),
                    reverse=reverse,
                )
            else:
                items.sort(key=lambda snapshot: snapshot.id, reverse=reverse)

        if self.after_id is not None:
            ids = [snapshot.id for snapshot in items]
            if self.after_id in ids:
                items = items[ids.index(self.after_id) + 1:]
        if self.after_values is not None:
            items = [snapshot for snapshot in items if self._is_after(snapshot)]
        if self.page_limit is not None:
            items = items[:self.page_limit]
        return items

    def _is_after(self, snapshot):
        data = snapshot.to_dict()
        for field, direction in self.orders:
            field_name = str(field)
            document_value = snapshot.id if field_name == "__name__" else data.get(field_name)
            cursor_value = self.after_values[field_name]
            if document_value == cursor_value:
                continue
            descending = str(direction).upper().endswith("DESCENDING")
            return document_value < cursor_value if descending else document_value > cursor_value
        return False


class FakeTransaction:
    def __init__(self, database):
        self.database = database
        self.operations = []
        self.writes_started = False

    def get(self, ref):
        if self.writes_started:
            raise AssertionError("Firestore transaction read attempted after write")
        self.database.read_log.append(ref.path)
        return FakeSnapshot(ref, self.database.documents.get(ref.path))

    def _stage(self, operation, ref, data):
        self.writes_started = True
        self.operations.append((operation, ref, dict(data)))
        fail_after = self.database.fail_transaction_after_staged_writes
        if fail_after is not None and len(self.operations) >= fail_after:
            raise RuntimeError("injected transaction failure")

    def set(self, ref, data):
        self._stage("set", ref, data)

    def update(self, ref, data):
        self._stage("update", ref, data)

    def commit(self):
        paths = [ref.path for _operation, ref, _data in self.operations]
        documents_before = copy.deepcopy(self.database.documents)
        clock_before = self.database.clock
        write_log_length = len(self.database.write_log)
        try:
            for operation, ref, data in self.operations:
                if operation == "set":
                    self.database.apply_set(ref, data)
                else:
                    self.database.apply_update(ref, data)
        except Exception:
            self.database.documents = documents_before
            self.database.clock = clock_before
            del self.database.write_log[write_log_length:]
            raise
        self.database.transaction_commits.append(paths)


class FakeChatDatabase:
    def __init__(self):
        self.documents = {}
        self.clock = datetime(2026, 8, 5, tzinfo=timezone.utc)
        self.read_log = []
        self.write_log = []
        self.transaction_commits = []
        self.fail_transaction_after_staged_writes = None

    def collection(self, name):
        return FakeCollectionRef(self, (name,))

    def run_transaction(self, operation):
        transaction = FakeTransaction(self)
        result = operation(transaction)
        transaction.commit()
        return result

    def apply_set(self, ref, data, merge=False):
        incoming = self.resolve_timestamps(data)
        if merge and ref.path in self.documents:
            self.documents[ref.path].update(incoming)
        else:
            self.documents[ref.path] = dict(incoming)
        self.write_log.append(("set", ref.path))

    def apply_update(self, ref, data):
        if ref.path not in self.documents:
            raise RuntimeError("missing document")
        self.documents[ref.path].update(self.resolve_timestamps(data))
        self.write_log.append(("update", ref.path))

    def resolve_timestamps(self, data):
        resolved = {}
        for key, value in data.items():
            if value.__class__.__name__ == "Sentinel":
                self.clock += timedelta(microseconds=1)
                resolved[key] = self.clock
            else:
                resolved[key] = value
        return resolved

    def chats(self, uid):
        prefix = ("users", uid, "chats")
        return {
            path[-1]: data
            for path, data in self.documents.items()
            if len(path) == 4 and path[:-1] == prefix
        }

    def turns(self, uid, chat_id):
        prefix = ("users", uid, "chats", chat_id, "turns")
        return {
            path[-1]: data
            for path, data in self.documents.items()
            if len(path) == 6 and path[:-1] == prefix
        }

    def model_answers(self, uid, chat_id, turn_id):
        prefix = ("users", uid, "chats", chat_id, "turns", turn_id, "model_answers")
        return {
            path[-1]: data
            for path, data in self.documents.items()
            if len(path) == 8 and path[:-1] == prefix
        }


@pytest.fixture
def chat_api(monkeypatch):
    database = FakeChatDatabase()
    limiter._storage.reset()
    monkeypatch.setenv("CHAT_CURSOR_SECRET", "test-chat-cursor-secret-32-bytes")
    monkeypatch.setattr(chat_history_router, "db_firestore", database)

    def verify(token):
        if token == "owner-token":
            return "owner-uid"
        if token == "other-token":
            return "other-uid"
        raise RuntimeError("invalid token")

    monkeypatch.setattr(chat_history_router, "verify_user_token", verify)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_history_router.router)
    app.add_middleware(CustomSecurityMiddleware)
    return TestClient(app), database


def create_chat(client, *, title=None, headers=AUTH_OWNER):
    body = {} if title is None else {"title": title}
    response = client.post("/chats", json=body, headers=headers)
    assert response.status_code == 201
    return response.json()["chat"]


def create_turn(client, chat_id, payload=None, *, headers=AUTH_OWNER):
    response = client.post(
        f"/chats/{chat_id}/turns",
        json=payload or TURN_PAYLOAD,
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()["turn"]


def completion_payload(question=TURN_PAYLOAD["question"], providers=PROVIDERS):
    return {
        "question": question,
        "model_answers": {
            provider: {
                "model_label": f"{provider} Model",
                "answer": f"Answer from {provider}",
                "sources": [{
                    "id": f"S{index}",
                    "title": f"Source {index}",
                    "url": f"https://example.test/{provider.lower()}",
                    "provider": provider,
                    "token": "must-not-survive",
                }],
                "api_key": "sk-must-not-survive",
                "raw_attachment": "must-not-survive",
            }
            for index, provider in enumerate(providers, 1)
        },
        "consensus": "Shared conclusion",
        "differences": "Small differences remain",
        "differences_data": {
            "claims": [{
                "anchor": "Shared conclusion",
                "agree": list(providers),
                "dissent": [],
                "secret": "must-not-survive",
            }],
            "differences": [],
            "models_compared": list(providers),
            "agreement": {"score": 83, "level": "high", "unknown": "drop"},
            "api_key": "sk-must-not-survive",
        },
        "sources": [
            {
                "id": "S1",
                "title": "Turn source",
                "url": "https://example.test/turn",
                "provider": "OpenAI",
                "credential": "must-not-survive",
            },
            {"id": "S2", "title": "Unsafe", "url": "javascript:alert(1)"},
        ],
        "result_id": "AbCdEf0123456789",
    }


def test_all_chat_endpoints_require_verified_authentication(chat_api):
    client, _database = chat_api
    requests = (
        client.post("/chats", json={}),
        client.get("/chats"),
        client.get("/chats/" + "a" * 32),
        client.post("/chats/" + "a" * 32 + "/turns", json=TURN_PAYLOAD),
        client.get("/chats/" + "a" * 32 + "/turns"),
        client.get("/chats/" + "a" * 32 + "/turns/" + "b" * 32),
        client.get("/chats", headers={"Authorization": "Bearer invalid"}),
    )
    assert all(response.status_code == 401 for response in requests)


def test_chat_creation_uses_server_id_and_storage_allowlist(chat_api):
    client, database = chat_api
    chat = create_chat(client, title="  Project   Alpha  ")

    assert re.fullmatch(r"[0-9a-f]{32}", chat["id"])
    assert chat["id"] not in {"Project Alpha", "project-alpha"}
    assert chat["title"] == "Project Alpha"
    assert chat["status"] == "active"
    assert chat["turn_count"] == 0
    assert chat["latest_question"] == ""
    assert set(database.chats("owner-uid")[chat["id"]]) == {
        "schema_version", "title", "status", "created_at", "updated_at",
        "turn_count", "latest_question",
    }

    rejected = client.post(
        "/chats",
        json={"title": "Unsafe", "api_key": "sk-do-not-store", "owner_uid": "other"},
        headers=AUTH_OWNER,
    )
    assert rejected.status_code == 422
    assert len(database.chats("owner-uid")) == 1


def test_chat_detail_is_owner_scoped_and_list_is_compact(chat_api):
    client, _database = chat_api
    first = create_chat(client, title="First")
    second = create_chat(client, title="Second")

    owner_detail = client.get(f"/chats/{first['id']}", headers=AUTH_OWNER)
    foreign_detail = client.get(f"/chats/{first['id']}", headers=AUTH_OTHER)
    foreign_turns = client.get(f"/chats/{first['id']}/turns", headers=AUTH_OTHER)
    listed = client.get("/chats", headers=AUTH_OWNER)

    assert owner_detail.status_code == 200
    assert owner_detail.headers["cache-control"] == "private, no-store"
    assert owner_detail.json()["chat"]["schema_version"] == 1
    assert foreign_detail.status_code == 404
    assert foreign_turns.status_code == 404
    assert [item["id"] for item in listed.json()["chats"]] == [second["id"], first["id"]]
    assert "schema_version" not in listed.json()["chats"][0]
    assert "turns" not in listed.json()["chats"][0]


def test_chat_list_enforces_limit_and_owner_bound_cursor(chat_api):
    client, _database = chat_api
    chats = [create_chat(client, title=f"Chat {index}") for index in range(3)]

    first_page = client.get("/chats?limit=2", headers=AUTH_OWNER)
    cursor = first_page.json()["next_cursor"]
    second_page = client.get(f"/chats?limit=2&cursor={cursor}", headers=AUTH_OWNER)
    foreign_cursor = client.get(f"/chats?limit=2&cursor={cursor}", headers=AUTH_OTHER)
    tampered = client.get(f"/chats?limit=2&cursor={cursor[:-1]}x", headers=AUTH_OWNER)
    over_limit = client.get("/chats?limit=51", headers=AUTH_OWNER)

    assert [item["id"] for item in first_page.json()["chats"]] == [
        chats[2]["id"], chats[1]["id"]
    ]
    assert [item["id"] for item in second_page.json()["chats"]] == [chats[0]["id"]]
    assert second_page.json()["has_more"] is False
    assert foreign_cursor.status_code == 400
    assert tampered.status_code == 400
    assert over_limit.status_code == 422


def test_chat_cursor_keeps_original_boundary_when_boundary_chat_moves(chat_api):
    client, _database = chat_api
    chats = [create_chat(client, title=f"Chat {index}") for index in range(4)]

    first_page = client.get("/chats?limit=2", headers=AUTH_OWNER)
    delivered_ids = [item["id"] for item in first_page.json()["chats"]]
    cursor = first_page.json()["next_cursor"]
    boundary_chat_id = delivered_ids[-1]

    create_turn(
        client,
        boundary_chat_id,
        {**TURN_PAYLOAD, "question": "Move the boundary chat to the top"},
    )
    second_page = client.get(f"/chats?limit=2&cursor={cursor}", headers=AUTH_OWNER)
    second_ids = [item["id"] for item in second_page.json()["chats"]]

    assert delivered_ids == [chats[3]["id"], chats[2]["id"]]
    assert second_ids == [chats[1]["id"], chats[0]["id"]]
    assert set(delivered_ids).isdisjoint(second_ids)


def test_chat_cursor_preserves_nanoseconds_and_rejects_invalid_timestamp(chat_api):
    _client, _database = chat_api
    timestamp = DatetimeWithNanoseconds.from_rfc3339(
        "2026-08-05T12:34:56.123456789Z"
    )
    document_id = "a" * 32
    cursor = chat_store._encode_chat_cursor(
        timestamp, document_id, owner_scope="owner-uid"
    )

    decoded_timestamp, decoded_id = chat_store._decode_chat_cursor(
        cursor, owner_scope="owner-uid"
    )
    invalid = chat_store._encode_cursor_payload(
        {"v": 2, "k": "chats", "updated_at": 123, "id": document_id},
        owner_scope="owner-uid",
    )
    out_of_range = chat_store._encode_cursor_payload(
        {
            "v": 2,
            "k": "chats",
            "updated_at": "0000-01-01T00:00:00.000000Z",
            "id": document_id,
        },
        owner_scope="owner-uid",
    )
    wrong_resource = chat_store._encode_cursor_payload(
        {"v": 2, "k": "turns", "position": 1, "id": document_id},
        owner_scope="owner-uid",
    )

    assert decoded_timestamp.rfc3339() == "2026-08-05T12:34:56.123456789Z"
    assert decoded_timestamp.nanosecond == 123456789
    assert copy.deepcopy(decoded_timestamp).nanosecond == 123456789
    assert decoded_id == document_id
    with pytest.raises(chat_store.InvalidChatCursor):
        chat_store._decode_chat_cursor(invalid, owner_scope="owner-uid")
    with pytest.raises(chat_store.InvalidChatCursor):
        chat_store._decode_chat_cursor(out_of_range, owner_scope="owner-uid")
    with pytest.raises(chat_store.InvalidChatCursor):
        chat_store._decode_chat_cursor(wrong_resource, owner_scope="owner-uid")


def test_chat_cursor_signing_secret_is_required_for_followup_page(chat_api, monkeypatch):
    client, _database = chat_api
    create_chat(client, title="First")
    create_chat(client, title="Second")
    monkeypatch.delenv("CHAT_CURSOR_SECRET", raising=False)
    monkeypatch.delenv("WATCH_UNSUBSCRIBE_SECRET", raising=False)

    response = client.get("/chats?limit=1", headers=AUTH_OWNER)

    assert response.status_code == 503


def test_first_and_second_turn_are_monotone_and_update_chat(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    first = create_turn(
        client,
        chat["id"],
        {
            **TURN_PAYLOAD,
            "question": "  First question\nwith context?  ",
            "selected_models": [" OpenAI ", "OpenAI", " Gemini "],
        },
    )
    second = create_turn(
        client,
        chat["id"],
        {**TURN_PAYLOAD, "question": "Second question?", "deep_search": True},
    )

    stored_chat = database.chats("owner-uid")[chat["id"]]
    stored_turns = database.turns("owner-uid", chat["id"])
    assert first["position"] == 1
    assert second["position"] == 2
    assert first["status"] == second["status"] == "pending"
    assert first["selected_models"] == ["OpenAI", "Gemini"]
    assert stored_chat["turn_count"] == 2
    assert stored_chat["latest_question"] == "Second question?"
    assert stored_chat["title"] == "First question with context?"
    assert all(
        set(turn) <= {
            "schema_version", "position", "status", "question", "mode",
            "deep_search", "selected_models", "consensus_model", "created_at",
            "updated_at", "client_request_id",
        }
        for turn in stored_turns.values()
    )


def test_turn_client_request_id_returns_original_for_normalized_identical_retry(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    payload = {
        **TURN_PAYLOAD,
        "question": "  What changed?  ",
        "mode": " regular ",
        "selected_models": [" OpenAI ", "OpenAI", " Gemini "],
        "client_request_id": "request-123",
    }
    first = create_turn(client, chat["id"], payload)
    second = create_turn(client, chat["id"], {
        **TURN_PAYLOAD,
        "client_request_id": "request-123",
    })

    assert second == first
    assert first["client_request_id"] == "request-123"
    assert len(database.turns("owner-uid", chat["id"])) == 1
    assert database.chats("owner-uid")[chat["id"]]["turn_count"] == 1


def test_turn_client_request_id_rejects_changed_payload(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    payload = {**TURN_PAYLOAD, "client_request_id": "request-123"}
    original = create_turn(client, chat["id"], payload)
    conflict = client.post(
        f"/chats/{chat['id']}/turns",
        headers=AUTH_OWNER,
        json={
            **payload,
            "question": "A changed question must not reuse the original turn",
        },
    )

    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "Idempotency conflict"}
    assert database.turns("owner-uid", chat["id"])[original["id"]]["question"] == (
        TURN_PAYLOAD["question"]
    )
    assert len(database.turns("owner-uid", chat["id"])) == 1
    assert database.chats("owner-uid")[chat["id"]]["turn_count"] == 1


def test_latest_question_is_bounded_preview_but_turn_keeps_full_question(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    question = ("  Café\n\tＡＩ   context " * 30) + "complete ending  "
    expected_full = unicodedata.normalize("NFKC", question).strip()
    expected_preview = " ".join(expected_full.split())[
        :chat_store.LATEST_QUESTION_PREVIEW_MAX_LENGTH
    ]
    turn = create_turn(
        client,
        chat["id"],
        {**TURN_PAYLOAD, "question": question},
    )
    detail = client.get(f"/chats/{chat['id']}", headers=AUTH_OWNER).json()["chat"]

    assert len(expected_preview) == chat_store.LATEST_QUESTION_PREVIEW_MAX_LENGTH
    assert turn["question"] == expected_full
    assert turn["question"].endswith("complete ending")
    assert len(turn["question"]) > chat_store.LATEST_QUESTION_PREVIEW_MAX_LENGTH
    assert detail["latest_question"] == expected_preview
    assert database.chats("owner-uid",)[chat["id"]]["latest_question"] == expected_preview


def test_turn_list_is_sorted_bounded_and_cursor_paginated(chat_api):
    client, _database = chat_api
    chat = create_chat(client)
    turns = [
        create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": f"Question {index}"})
        for index in range(3)
    ]

    first_page = client.get(f"/chats/{chat['id']}/turns?limit=2", headers=AUTH_OWNER)
    cursor = first_page.json()["next_cursor"]
    second_page = client.get(
        f"/chats/{chat['id']}/turns?limit=2&cursor={cursor}", headers=AUTH_OWNER
    )
    over_limit = client.get(f"/chats/{chat['id']}/turns?limit=101", headers=AUTH_OWNER)

    assert [item["id"] for item in first_page.json()["turns"]] == [
        turns[0]["id"], turns[1]["id"]
    ]
    assert [item["id"] for item in second_page.json()["turns"]] == [turns[2]["id"]]
    assert second_page.json()["has_more"] is False
    assert over_limit.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {**TURN_PAYLOAD, "question": "   "},
        {**TURN_PAYLOAD, "mode": "x" * 41},
        {**TURN_PAYLOAD, "deep_search": "false"},
        {**TURN_PAYLOAD, "selected_models": []},
        {**TURN_PAYLOAD, "selected_models": [f"model-{index}" for index in range(9)]},
        {**TURN_PAYLOAD, "consensus_model": ""},
        {**TURN_PAYLOAD, "client_request_id": "contains spaces"},
        {**TURN_PAYLOAD, "api_key": "sk-do-not-store"},
        {**TURN_PAYLOAD, "id_token": "firebase-token-do-not-store"},
    ],
)
def test_invalid_or_secret_bearing_turn_payloads_are_not_stored(chat_api, payload):
    client, database = chat_api
    chat = create_chat(client)
    response = client.post(
        f"/chats/{chat['id']}/turns", json=payload, headers=AUTH_OWNER
    )
    assert response.status_code == 422
    assert database.turns("owner-uid", chat["id"]) == {}
    assert database.chats("owner-uid")[chat["id"]]["turn_count"] == 0


def test_unknown_and_foreign_chat_are_indistinguishable_for_turn_creation(chat_api):
    client, _database = chat_api
    owned = create_chat(client)
    unknown_id = "f" * 32

    unknown = client.post(
        f"/chats/{unknown_id}/turns", json=TURN_PAYLOAD, headers=AUTH_OWNER
    )
    foreign = client.post(
        f"/chats/{owned['id']}/turns", json=TURN_PAYLOAD, headers=AUTH_OTHER
    )
    malformed = client.get("/chats/not-a-valid-id", headers=AUTH_OWNER)

    assert unknown.status_code == foreign.status_code == malformed.status_code == 404
    assert unknown.json() == foreign.json()


def test_completion_preflight_is_owner_bound_read_only_and_skips_answers(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    database.read_log.clear()

    validated = store.validate_turn_for_completion(
        "owner-uid",
        chat["id"],
        turn["id"],
        question=TURN_PAYLOAD["question"],
    )

    assert validated["status"] == "pending"
    assert database.turns("owner-uid", chat["id"])[turn["id"]]["status"] == "pending"
    assert not any("model_answers" in path for path in database.read_log)
    with pytest.raises(chat_store.ChatNotFound):
        store.validate_turn_for_completion(
            "other-uid",
            chat["id"],
            turn["id"],
            question=TURN_PAYLOAD["question"],
        )
    with pytest.raises(chat_store.TurnQuestionConflict):
        store.validate_turn_for_completion(
            "owner-uid",
            chat["id"],
            turn["id"],
            question="Different question",
        )


def test_complete_turn_atomically_persists_six_separate_answer_documents(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    chat_before = dict(database.chats("owner-uid")[chat["id"]])

    completed = store.complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload()
    )

    stored_turn = database.turns("owner-uid", chat["id"])[turn["id"]]
    stored_answers = database.model_answers("owner-uid", chat["id"], turn["id"])
    stored_chat = database.chats("owner-uid")[chat["id"]]
    assert completed["status"] == "completed"
    assert completed["answer_count"] == 6
    assert completed["agreement_score"] == 83
    assert completed["included_models"] == list(PROVIDERS)
    assert set(completed["model_answers"]) == set(PROVIDERS)
    assert set(stored_answers) == set(chat_store.PROVIDER_DOCUMENT_IDS.values())
    assert all(
        set(answer) == {
            "schema_version", "provider", "model_label", "answer", "sources",
            "created_at", "updated_at",
        }
        for answer in stored_answers.values()
    )
    assert all("api_key" not in answer for answer in stored_answers.values())
    assert all("raw_attachment" not in answer for answer in stored_answers.values())
    assert "model_answers" not in stored_turn
    assert not any(provider in stored_turn for provider in PROVIDERS)
    assert stored_turn["completion_fingerprint"]
    assert stored_turn["result_id"] == "AbCdEf0123456789"
    assert stored_turn["sources"] == [{
        "id": "S1", "title": "Turn source", "url": "https://example.test/turn",
        "provider": "OpenAI",
    }]
    assert "secret" not in stored_turn["differences_data"]["claims"][0]
    assert "api_key" not in stored_turn["differences_data"]
    assert stored_chat["turn_count"] == chat_before["turn_count"] == 1
    assert stored_chat["latest_question"] == chat_before["latest_question"]
    assert stored_chat["updated_at"] > chat_before["updated_at"]
    committed_paths = database.transaction_commits[-1]
    assert len(committed_paths) == 8
    assert committed_paths[-2:] == [
        ("users", "owner-uid", "chats", chat["id"], "turns", turn["id"]),
        ("users", "owner-uid", "chats", chat["id"]),
    ]


def test_complete_turn_skips_empty_answers_and_derives_count_server_side(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    payload = completion_payload(providers=("OpenAI", "Gemini", "Grok"))
    payload["model_answers"]["Gemini"]["answer"] = "   "
    payload["model_answers"]["Grok"]["answer"] = None
    payload["differences_data"] = {"agreement": {"level": "unknown"}}

    completed = chat_store.ChatStore(database).complete_turn(
        "owner-uid", chat["id"], turn["id"], **payload
    )

    assert completed["answer_count"] == 1
    assert completed["included_models"] == ["OpenAI"]
    assert "agreement_score" not in completed
    assert "score" not in completed["differences_data"]["agreement"]
    assert set(database.model_answers("owner-uid", chat["id"], turn["id"])) == {
        "openai"
    }


def test_complete_turn_rejects_unknown_or_too_many_providers_before_writes(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    before = len(database.write_log)

    unknown = completion_payload(providers=("OpenAI",))
    unknown["model_answers"]["Untrusted/Path"] = {"answer": "unsafe"}
    with pytest.raises(ValueError, match="supported provider"):
        store.complete_turn("owner-uid", chat["id"], turn["id"], **unknown)
    with pytest.raises(ValueError, match="at most 6"):
        store.complete_turn(
            "owner-uid",
            chat["id"],
            turn["id"],
            **{
                **completion_payload(),
                "model_answers": [
                    {"provider": provider, "answer": "ok"} for provider in PROVIDERS
                ] + [{"provider": "OpenAI", "answer": "duplicate"}],
            },
        )

    assert len(database.write_log) == before
    assert database.turns("owner-uid", chat["id"])[turn["id"]]["status"] == "pending"
    assert database.model_answers("owner-uid", chat["id"], turn["id"]) == {}


def test_completion_applies_text_model_label_and_source_limits(chat_api, monkeypatch):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    monkeypatch.setattr(chat_store.cfg, "get_consensus_answer_char_limit", lambda: 12)
    payload = completion_payload(providers=("OpenAI",))
    payload["model_answers"]["OpenAI"].update({
        "answer": "a" * 100,
        "model_label": "unsafe,label" + "x" * 200,
        "sources": [
            {
                "id": f"S{index}", "title": "t" * 500,
                "url": f"https://example.test/{index}", "provider": "p" * 100,
            }
            for index in range(chat_store.MODEL_SOURCES_MAX_ITEMS + 5)
        ],
    })
    payload["consensus"] = "c" * (chat_store.CONSENSUS_MAX_LENGTH + 10)
    payload["differences"] = "d" * (chat_store.DIFFERENCES_MAX_LENGTH + 10)

    chat_store.ChatStore(database).complete_turn(
        "owner-uid", chat["id"], turn["id"], **payload
    )

    answer = database.model_answers("owner-uid", chat["id"], turn["id"])["openai"]
    stored_turn = database.turns("owner-uid", chat["id"])[turn["id"]]
    assert answer["answer"] == "a" * 12
    assert answer["model_label"] == "OpenAI"
    assert len(answer["sources"]) == chat_store.MODEL_SOURCES_MAX_ITEMS
    assert len(answer["sources"][0]["title"]) == 300
    assert len(answer["sources"][0]["provider"]) == 40
    assert len(stored_turn["consensus"]) == chat_store.CONSENSUS_MAX_LENGTH
    assert len(stored_turn["differences"]) == chat_store.DIFFERENCES_MAX_LENGTH


def test_completion_retry_is_idempotent_but_changed_payload_conflicts(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    payload = completion_payload(providers=("OpenAI", "Gemini"))
    first = store.complete_turn("owner-uid", chat["id"], turn["id"], **payload)
    writes_after_first = len(database.write_log)
    reordered = dict(payload)
    reordered["model_answers"] = {
        "Gemini": {
            **payload["model_answers"]["Gemini"],
            "api_key": "a-different-ignored-secret",
        },
        "OpenAI": payload["model_answers"]["OpenAI"],
    }

    second = store.complete_turn(
        "owner-uid", chat["id"], turn["id"], **reordered
    )
    assert second == first
    assert len(database.write_log) == writes_after_first

    with pytest.raises(chat_store.TurnCompletionConflict):
        store.complete_turn(
            "owner-uid", chat["id"], turn["id"],
            **{**payload, "consensus": "Different conclusion"},
        )
    assert len(database.write_log) == writes_after_first


def test_completion_rejects_question_mismatch_failed_turn_and_invalid_result_id(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    mismatch_turn = create_turn(client, chat["id"])
    failed_turn = create_turn(
        client, chat["id"], {**TURN_PAYLOAD, "question": "Second turn"}
    )
    invalid_result_turn = create_turn(
        client, chat["id"], {**TURN_PAYLOAD, "question": "Third turn"}
    )
    store = chat_store.ChatStore(database)

    with pytest.raises(chat_store.TurnQuestionConflict):
        store.complete_turn(
            "owner-uid", chat["id"], mismatch_turn["id"],
            **completion_payload(question="Different question"),
        )
    store.fail_turn(
        "owner-uid", chat["id"], failed_turn["id"], error_code="cancelled"
    )
    with pytest.raises(chat_store.TurnStatusConflict):
        store.complete_turn(
            "owner-uid", chat["id"], failed_turn["id"],
            **completion_payload(question="Second turn"),
        )
    with pytest.raises(ValueError, match="valid result identifier"):
        store.complete_turn(
            "owner-uid", chat["id"], invalid_result_turn["id"],
            **completion_payload(question="Third turn") | {"result_id": "../secret"},
        )
    assert all(
        item["status"] in {"pending", "failed"}
        for item in database.turns("owner-uid", chat["id"]).values()
    )


def test_fail_turn_is_allowlisted_idempotent_and_terminal(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    failed = create_turn(client, chat["id"])
    completed = create_turn(
        client, chat["id"], {**TURN_PAYLOAD, "question": "Completed question"}
    )
    store = chat_store.ChatStore(database)

    first = store.fail_turn(
        "owner-uid", chat["id"], failed["id"], error_code="cancelled"
    )
    writes_after_first = len(database.write_log)
    second = store.fail_turn(
        "owner-uid", chat["id"], failed["id"], error_code="cancelled"
    )
    assert second == first
    assert len(database.write_log) == writes_after_first
    assert set(database.turns("owner-uid", chat["id"])[failed["id"]]) == {
        "schema_version", "position", "status", "question", "mode", "deep_search",
        "selected_models", "consensus_model", "created_at", "updated_at",
        "error_code", "failed_at",
    }
    with pytest.raises(chat_store.TurnStatusConflict):
        store.fail_turn(
            "owner-uid", chat["id"], failed["id"],
            error_code="insufficient_answers",
        )
    with pytest.raises(ValueError, match="allowed code"):
        store.fail_turn(
            "owner-uid", chat["id"], failed["id"],
            error_code="provider stacktrace and sk-secret",
        )

    store.complete_turn(
        "owner-uid", chat["id"], completed["id"],
        **completion_payload(question="Completed question"),
    )
    with pytest.raises(chat_store.TurnStatusConflict):
        store.fail_turn(
            "owner-uid", chat["id"], completed["id"], error_code="cancelled"
        )


def test_completion_transaction_failure_leaves_no_partial_state(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    chat_before = dict(database.chats("owner-uid")[chat["id"]])
    database.fail_transaction_after_staged_writes = 3

    with pytest.raises(RuntimeError, match="injected transaction failure"):
        chat_store.ChatStore(database).complete_turn(
            "owner-uid", chat["id"], turn["id"], **completion_payload()
        )

    assert database.model_answers("owner-uid", chat["id"], turn["id"]) == {}
    assert database.turns("owner-uid", chat["id"])[turn["id"]]["status"] == "pending"
    assert database.chats("owner-uid")[chat["id"]] == chat_before


def test_turn_detail_is_owner_scoped_whitelisted_and_reads_six_known_docs(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    chat_store.ChatStore(database).complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload()
    )
    database.read_log.clear()

    owner = client.get(
        f"/chats/{chat['id']}/turns/{turn['id']}", headers=AUTH_OWNER
    )
    foreign_turn = client.get(
        f"/chats/{chat['id']}/turns/{turn['id']}", headers=AUTH_OTHER
    )
    unknown_turn = client.get(
        f"/chats/{chat['id']}/turns/{'f' * 32}", headers=AUTH_OWNER
    )
    foreign_chat = client.get(
        f"/chats/{chat['id']}/turns/{'f' * 32}", headers=AUTH_OTHER
    )
    malformed = client.get(
        f"/chats/{chat['id']}/turns/not-a-valid-id", headers=AUTH_OWNER
    )

    assert owner.status_code == 200
    assert owner.headers["cache-control"] == "private, no-store"
    returned = owner.json()["turn"]
    assert "completion_fingerprint" not in returned
    assert set(returned["model_answers"]) == set(PROVIDERS)
    assert all("api_key" not in answer for answer in returned["model_answers"].values())
    assert (
        foreign_turn.status_code
        == unknown_turn.status_code
        == foreign_chat.status_code
        == malformed.status_code
        == 404
    )
    assert foreign_turn.json() == unknown_turn.json() == foreign_chat.json() == malformed.json()
    answer_reads = [path for path in database.read_log if "model_answers" in path]
    assert len(answer_reads) == 6
    assert {path[-1] for path in answer_reads} == set(
        chat_store.PROVIDER_DOCUMENT_IDS.values()
    )


def test_turn_list_stays_compact_after_completion_and_failure(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    completed = create_turn(client, chat["id"])
    failed = create_turn(
        client, chat["id"], {**TURN_PAYLOAD, "question": "Failed question"}
    )
    store = chat_store.ChatStore(database)
    store.complete_turn(
        "owner-uid", chat["id"], completed["id"], **completion_payload()
    )
    store.fail_turn(
        "owner-uid", chat["id"], failed["id"], error_code="consensus_failed"
    )

    listed = client.get(f"/chats/{chat['id']}/turns", headers=AUTH_OWNER).json()["turns"]
    completed_meta, failed_meta = listed
    assert completed_meta["answer_count"] == 6
    assert completed_meta["agreement_score"] == 83
    assert completed_meta["completed_at"]
    assert failed_meta["error_code"] == "consensus_failed"
    assert failed_meta["failed_at"]
    forbidden = {
        "consensus", "differences", "differences_data", "sources",
        "model_answers", "completion_fingerprint",
    }
    assert all(forbidden.isdisjoint(item) for item in listed)
