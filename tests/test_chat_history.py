from __future__ import annotations

import re
import unicodedata
import copy
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import chat_history as chat_history_router
from app.core.rate_limit import api_uid_limiter, limiter
from app.core.security import CustomSecurityMiddleware
from app.services import chat_store
from google.api_core.datetime_helpers import DatetimeWithNanoseconds


AUTH_OWNER = {"Authorization": "Bearer owner-token"}
AUTH_OTHER = {"Authorization": "Bearer other-token"}
# Eine echte Engine aus ALLOWED_CONSENSUS_MODELS. Frueher stand hier der
# Fantasiestring "GPT Consensus" - der Endpoint nahm ihn an, weil er nur die
# Laenge prueft. Seit die Turn-Anlage dieselbe Allowlist wie /consensus
# durchsetzt, muss die Fixture ein Modell verwenden, das es wirklich gibt.
TURN_PAYLOAD = {
    "question": "What changed?",
    "mode": "regular",
    "deep_search": False,
    "selected_models": ["OpenAI", "Gemini"],
    "consensus_model": "OpenAI",
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

    def delete(self):
        self.database.documents.pop(self.path, None)
        self.database.write_log.append(("delete", self.path))


class FakeAggregationResult:
    def __init__(self, value):
        self.value = value


class FakeAggregation:
    def __init__(self, value):
        self._value = value

    def get(self):
        return [[FakeAggregationResult(self._value)]]


class FakeCollectionRef:
    def __init__(self, database, path):
        self.database = database
        self.path = tuple(path)

    def document(self, document_id):
        return FakeDocumentRef(self.database, (*self.path, document_id))

    def order_by(self, field, direction=None):
        return FakeQuery(self, [(field, direction)])

    def where(self, filter=None):
        return FakeQuery(self, [], filters=[filter])

    def limit(self, value):
        # Echte CollectionReference kann das auch ohne vorheriges order_by.
        return FakeQuery(self, []).limit(value)

    def count(self):
        # Bildet die Firestore-Aggregation nach, inklusive der verschachtelten
        # Ergebnisform [[AggregationResult]], die der Store auspacken muss.
        return FakeAggregation(len(self._child_paths()))

    def _child_paths(self):
        depth = len(self.path) + 1
        return sorted(
            path
            for path in self.database.documents
            if len(path) == depth and path[:-1] == self.path
        )

    def stream(self):
        for path in self._child_paths():
            ref = FakeDocumentRef(self.database, path)
            self.database.read_log.append(path)
            yield FakeSnapshot(ref, self.database.documents.get(path))

    def list_documents(self):
        for path in self._child_paths():
            self.database.read_log.append(path)
            yield FakeDocumentRef(self.database, path)


class FakeQuery:
    def __init__(self, collection, orders, filters=None):
        self.collection = collection
        self.orders = list(orders)
        self.filters = [f for f in (filters or []) if f is not None]
        self.after_id = None
        self.after_values = None
        self.page_limit = None

    def order_by(self, field, direction=None):
        self.orders.append((field, direction))
        return self

    def where(self, filter=None):
        if filter is not None:
            self.filters.append(filter)
        return self

    def _matches_filters(self, snapshot):
        data = snapshot.to_dict() or {}
        for condition in self.filters:
            field = getattr(condition, "field_path", None)
            op = getattr(condition, "op_string", None)
            value = getattr(condition, "value", None)
            if op != "==":
                raise AssertionError(f"unsupported fake filter operator: {op}")
            if data.get(field) != value:
                return False
        return True

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

        if self.filters:
            items = [item for item in items if self._matches_filters(item)]

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
        # Emulates SERVER_TIMESTAMP, which is always ~now. A fixed date in the
        # past would make freshly written documents look stale to any age-based
        # logic (e.g. the abandoned-turn sweep).
        self.clock = datetime.now(timezone.utc).replace(microsecond=0)
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
    api_uid_limiter.reset()
    monkeypatch.setenv("CHAT_CURSOR_SECRET", "test-chat-cursor-secret-32-bytes")
    monkeypatch.setattr(chat_history_router, "db_firestore", database)

    def verify(token):
        if token == "owner-token":
            return "owner-uid"
        if token == "other-token":
            return "other-uid"
        raise RuntimeError("invalid token")

    monkeypatch.setattr(chat_history_router, "verify_user_token", verify)
    # Ohne diesen Stub liefe das Tier-Gate der Turn-Anlage gegen das ECHTE
    # Firestore (is_user_pro haengt am Client aus app.core.security, nicht am
    # gepatchten db_firestore dieses Moduls). Default ist Free; ein Test, der
    # Pro braucht, patcht die Funktion selbst.
    monkeypatch.setattr(chat_history_router, "is_user_pro", lambda uid: False)
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
        # Kein Modell ausserhalb der Allowlist: der Turn haelt die Engine fest,
        # mit der spaeter der Memory-Call laeuft.
        {**TURN_PAYLOAD, "consensus_model": "GPT Consensus"},
        {**TURN_PAYLOAD, "consensus_model": "gpt-5-turbo-unlimited"},
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


def test_premium_engine_needs_pro_at_turn_creation_like_at_consensus(
    chat_api, monkeypatch
):
    """Das Tier-Gate von /consensus gilt schon beim Anlegen des Turns.

    Der Turn ist die Quelle, aus der _memory_credentials spaeter die Engine
    liest. Ohne dieses Gate koennte ein Free-Konto eine Premium-Engine im
    Dokument hinterlegen, die dann - sobald die Admin-Zuordnung
    chat_memory_models fuer diese Familie leer ist - den Kompressions-Call auf
    dem Developer-Key faehrt.
    """
    client, database = chat_api
    chat = create_chat(client)
    premium = {**TURN_PAYLOAD, "consensus_model": "Anthropic-Pro"}

    blocked = client.post(
        f"/chats/{chat['id']}/turns", json=premium, headers=AUTH_OWNER
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["error_code"] == "pro_required"
    # Kein halb angelegter Turn und kein hochgezaehlter Chat.
    assert database.turns("owner-uid", chat["id"]) == {}
    assert database.chats("owner-uid")[chat["id"]]["turn_count"] == 0

    # Free-Engines bleiben fuer dasselbe Konto offen.
    create_turn(client, chat["id"])

    # Mit Pro geht die Premium-Engine durch.
    monkeypatch.setattr(chat_history_router, "is_user_pro", lambda uid: True)
    allowed = client.post(
        f"/chats/{chat['id']}/turns", json=premium, headers=AUTH_OWNER
    )
    assert allowed.status_code == 201
    assert allowed.json()["turn"]["consensus_model"] == "Anthropic-Pro"


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


# ---------------------------------------------------------------------------
# Question-Limit: /consensus kappt die Frage auf
# cfg.get_consensus_question_char_limit(). Wird ein Turn mit einer laengeren
# Frage angelegt, kann er NIE abgeschlossen werden — die gekappte Frage passt
# dann nicht mehr, und jeder Retry scheitert dauerhaft an 409.
# ---------------------------------------------------------------------------


def test_turn_question_limit_matches_the_consensus_cap():
    assert chat_store._question_max_length() == cfg.get_consensus_question_char_limit()


def test_turn_rejects_a_question_the_consensus_cap_would_truncate(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    limit = cfg.get_consensus_question_char_limit()
    payload = {**TURN_PAYLOAD, "question": "q" * (limit + 1)}

    response = client.post(f"/chats/{chat['id']}/turns", json=payload, headers=AUTH_OWNER)

    assert response.status_code == 422
    assert database.turns("owner-uid", chat["id"]) == {}


def test_longest_accepted_question_still_validates_after_the_consensus_cap(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    limit = cfg.get_consensus_question_char_limit()
    question = "q" * limit
    turn = create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": question})

    # Exactly what /consensus does to the question before it validates the turn.
    capped = question[:limit].rstrip()
    validated = chat_store.ChatStore(database).validate_turn_for_completion(
        "owner-uid", chat["id"], turn["id"], question=capped
    )

    assert validated["id"] == turn["id"]
    assert validated["status"] == "pending"


# ---------------------------------------------------------------------------
# Read-Amplification: ein Turn-Detail darf nicht sechs Einzel-Gets auf
# model_answers kosten, und eine Detail-Seite darf den Chat nicht pro Turn neu
# pruefen.
# ---------------------------------------------------------------------------


def test_get_turn_reads_model_answers_with_one_query(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    store.complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload(providers=("OpenAI",))
    )
    database.read_log.clear()

    detail = store.get_turn("owner-uid", chat["id"], turn["id"])

    assert set(detail["model_answers"]) == {"OpenAI"}
    answer_reads = [
        path for path in database.read_log if "model_answers" in path
    ]
    # One collection read for the single stored answer — not one get per
    # provider in PROVIDER_ORDER.
    assert len(answer_reads) == 1


def test_model_answer_under_a_foreign_document_id_is_ignored(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    store.complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload(providers=("OpenAI",))
    )
    # Same payload, wrong document id: the provider must not be trusted over
    # the path it was stored under.
    smuggled = ("users", "owner-uid", "chats", chat["id"], "turns", turn["id"],
                "model_answers", "gemini")
    database.documents[smuggled] = {
        **database.model_answers("owner-uid", chat["id"], turn["id"])["openai"],
    }

    detail = store.get_turn("owner-uid", chat["id"], turn["id"])

    assert set(detail["model_answers"]) == {"OpenAI"}


def test_list_turn_details_checks_the_chat_once_for_the_whole_page(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    store = chat_store.ChatStore(database)
    for index in range(3):
        turn = create_turn(
            client, chat["id"], {**TURN_PAYLOAD, "question": f"Question {index}"}
        )
        if index != 1:
            store.complete_turn(
                "owner-uid",
                chat["id"],
                turn["id"],
                **completion_payload(
                    question=f"Question {index}", providers=("OpenAI", "Gemini")
                ),
            )
    database.read_log.clear()

    page = store.list_turn_details("owner-uid", chat["id"], status="completed")

    assert [turn["question"] for turn in page["turns"]] == ["Question 0", "Question 2"]
    assert page["has_more"] is False
    assert all(turn["model_answers"] for turn in page["turns"])
    chat_reads = [path for path in database.read_log if len(path) == 4]
    assert len(chat_reads) == 1


def test_list_turn_details_keeps_pagination_identical_to_list_turns(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    store = chat_store.ChatStore(database)
    for index in range(3):
        create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": f"Question {index}"})

    metadata_page = store.list_turns("owner-uid", chat["id"], limit=2)
    detail_page = store.list_turn_details("owner-uid", chat["id"], limit=2)

    assert detail_page["has_more"] == metadata_page["has_more"] is True
    assert detail_page["next_cursor"] == metadata_page["next_cursor"]
    assert [turn["id"] for turn in detail_page["turns"]] == [
        turn["id"] for turn in metadata_page["turns"]
    ]


def test_list_turn_details_rejects_a_foreign_owner(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    create_turn(client, chat["id"])

    with pytest.raises(chat_store.ChatNotFound):
        chat_store.ChatStore(database).list_turn_details("other-uid", chat["id"])


# ---------------------------------------------------------------------------
# DSGVO Art. 17: Firestore kaskadiert nicht. Chats, Turns, Modellantworten und
# Context-Versionen muessen bei der Kontoloeschung alle verschwinden.
# ---------------------------------------------------------------------------


def test_delete_all_chats_removes_every_nested_level(chat_api):
    client, database = chat_api
    store = chat_store.ChatStore(database)
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store.complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload()
    )
    version_path = (
        "users", "owner-uid", "chats", chat["id"], "context_versions", "d" * 32,
    )
    database.documents[version_path] = {"status": "ready", "memory": {}}
    assert database.model_answers("owner-uid", chat["id"], turn["id"])

    store.delete_all_chats("owner-uid")

    survivors = [
        path for path in database.documents
        if len(path) > 2 and path[:2] == ("users", "owner-uid") and "chats" in path
    ]
    assert survivors == []


def test_delete_all_chats_descends_into_missing_parent_documents(chat_api):
    _client, database = chat_api
    chat_id = "a" * 32
    turn_id = "b" * 32
    chat_path = ("users", "owner-uid", "chats", chat_id)
    turn_path = (*chat_path, "turns", turn_id)
    answer_path = (*turn_path, "model_answers", "openai")
    context_path = (*chat_path, "context_versions", "c" * 32)
    # A None parent models Firestore's missing document that remains visible
    # through list_documents() because it still owns subcollections.
    database.documents[chat_path] = None
    database.documents[turn_path] = {"status": "completed", "position": 1}
    database.documents[answer_path] = {"answer": "private answer"}
    database.documents[context_path] = {"memory": {"private": "context"}}

    chat_store.ChatStore(database).delete_all_chats("owner-uid")

    assert not any(path[: len(chat_path)] == chat_path for path in database.documents)


def test_delete_all_chats_leaves_other_owners_untouched(chat_api):
    client, database = chat_api
    store = chat_store.ChatStore(database)
    owner_chat = create_chat(client)
    create_turn(client, owner_chat["id"])
    other_chat = create_chat(client, headers=AUTH_OTHER)
    create_turn(client, other_chat["id"], headers=AUTH_OTHER)

    store.delete_all_chats("owner-uid")

    assert database.chats("owner-uid") == {}
    assert set(database.chats("other-uid")) == {other_chat["id"]}
    assert database.turns("other-uid", other_chat["id"])


def test_delete_all_chats_is_idempotent(chat_api):
    client, database = chat_api
    store = chat_store.ChatStore(database)
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store.complete_turn("owner-uid", chat["id"], turn["id"], **completion_payload())

    store.delete_all_chats("owner-uid")
    store.delete_all_chats("owner-uid")

    assert database.chats("owner-uid") == {}


# ---------------------------------------------------------------------------
# Loeschpfad: ohne DELETE war ein Bookmark der einzige Griff an einem Chat —
# war es weg, blieb das Transcript unerreichbar liegen.
# ---------------------------------------------------------------------------


def test_delete_chat_removes_the_whole_tree(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    chat_store.ChatStore(database).complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload()
    )
    version_path = (
        "users", "owner-uid", "chats", chat["id"], "context_versions", "d" * 32,
    )
    database.documents[version_path] = {"status": "ready"}

    response = client.delete(f"/chats/{chat['id']}", headers=AUTH_OWNER)

    assert response.status_code == 200
    assert database.chats("owner-uid") == {}
    assert database.turns("owner-uid", chat["id"]) == {}
    assert database.model_answers("owner-uid", chat["id"], turn["id"]) == {}
    assert version_path not in database.documents


def test_delete_chat_is_owner_scoped_and_uniformly_404(chat_api):
    client, database = chat_api
    chat = create_chat(client)

    foreign = client.delete(f"/chats/{chat['id']}", headers=AUTH_OTHER)
    unknown = client.delete(f"/chats/{'f' * 32}", headers=AUTH_OWNER)
    malformed = client.delete("/chats/not-a-chat-id", headers=AUTH_OWNER)

    assert foreign.status_code == unknown.status_code == malformed.status_code == 404
    # The foreign attempt must not have touched the owner's chat.
    assert set(database.chats("owner-uid")) == {chat["id"]}


def test_delete_chat_requires_authentication(chat_api):
    client, database = chat_api
    chat = create_chat(client)

    response = client.delete(f"/chats/{chat['id']}")

    assert response.status_code == 401
    assert set(database.chats("owner-uid")) == {chat["id"]}


def test_deleting_a_chat_twice_reports_404_and_changes_nothing(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    create_turn(client, chat["id"])

    assert client.delete(f"/chats/{chat['id']}", headers=AUTH_OWNER).status_code == 200
    writes = len(database.write_log)
    assert client.delete(f"/chats/{chat['id']}", headers=AUTH_OWNER).status_code == 404
    assert len(database.write_log) == writes


def test_deleting_chat_state_rejects_a_late_turn_before_it_can_write(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    chat_path = ("users", "owner-uid", "chats", chat["id"])
    database.documents[chat_path]["status"] = "deleting"
    before = dict(database.documents)

    response = client.post(
        f"/chats/{chat['id']}/turns",
        json=TURN_PAYLOAD,
        headers=AUTH_OWNER,
    )

    assert response.status_code == 404
    assert database.turns("owner-uid", chat["id"]) == {}
    assert database.documents == before


def test_deleting_chat_state_rejects_late_completion_and_failure_writes(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    chat_path = ("users", "owner-uid", "chats", chat["id"])
    database.documents[chat_path]["status"] = "deleting"
    before = copy.deepcopy(database.documents)
    store = chat_store.ChatStore(database)

    with pytest.raises(chat_store.ChatNotFound):
        store.complete_turn(
            "owner-uid", chat["id"], turn["id"], **completion_payload()
        )
    with pytest.raises(chat_store.ChatNotFound):
        store.fail_turn(
            "owner-uid",
            chat["id"],
            turn["id"],
            error_code="consensus_failed",
        )

    assert database.documents == before
    assert database.model_answers("owner-uid", chat["id"], turn["id"]) == {}


def test_account_deletion_tombstone_fences_late_chat_completion(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    database.documents[("account_deletion_jobs", "owner-uid")] = {
        "status": "pending"
    }
    before = copy.deepcopy(database.documents)

    with pytest.raises(chat_store.persistence_guard.AccountDeletionInProgress):
        chat_store.ChatStore(database).complete_turn(
            "owner-uid", chat["id"], turn["id"], **completion_payload()
        )

    assert database.documents == before
    assert database.model_answers("owner-uid", chat["id"], turn["id"]) == {}


def test_account_deletion_tombstone_fences_late_chat_and_turn_creation(chat_api):
    client, database = chat_api
    existing_chat = create_chat(client)
    database.documents[("account_deletion_jobs", "owner-uid")] = {
        "status": "pending"
    }
    before = copy.deepcopy(database.documents)
    store = chat_store.ChatStore(database)

    with pytest.raises(chat_store.persistence_guard.AccountDeletionInProgress):
        store.create_chat("owner-uid", title="must not survive")
    with pytest.raises(chat_store.persistence_guard.AccountDeletionInProgress):
        store.create_turn(
            "owner-uid",
            existing_chat["id"],
            **TURN_PAYLOAD,
        )

    assert database.documents == before
    assert database.turns("owner-uid", existing_chat["id"]) == {}


def test_account_deletion_fences_normal_chat_delete_but_cleanup_can_continue(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(client, chat["id"])
    store = chat_store.ChatStore(database)
    database.documents[("account_deletion_jobs", "owner-uid")] = {
        "status": "pending"
    }
    before = copy.deepcopy(database.documents)

    with pytest.raises(chat_store.persistence_guard.AccountDeletionInProgress):
        store.delete_chat("owner-uid", chat["id"])

    assert database.documents == before
    assert database.turns("owner-uid", chat["id"])[turn["id"]]["status"] == "pending"

    store.delete_all_chats("owner-uid")

    assert database.chats("owner-uid") == {}
    assert database.turns("owner-uid", chat["id"]) == {}


# ---------------------------------------------------------------------------
# Verwaiste pending Turns: ein Reload verliert die Browser-Bindung, danach kann
# kein Lauf den Turn je abschliessen.
# ---------------------------------------------------------------------------


def _age_turn(database, chat_id, turn_id, *, seconds):
    path = ("users", "owner-uid", "chats", chat_id, "turns", turn_id)
    database.documents[path]["created_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    )


def test_stale_pending_turn_is_retired_as_abandoned(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    stale = create_turn(client, chat["id"])
    _age_turn(
        database,
        chat["id"],
        stale["id"],
        seconds=chat_store.ABANDONED_TURN_MAX_AGE_SECONDS + 60,
    )

    retired = chat_store.ChatStore(database).purge_abandoned_turns("owner-uid", chat["id"])

    assert retired == [stale["id"]]
    stored = database.turns("owner-uid", chat["id"])[stale["id"]]
    assert stored["status"] == "failed"
    assert stored["error_code"] == "abandoned"


def test_a_recent_pending_turn_is_never_retired(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    fresh = create_turn(client, chat["id"])
    _age_turn(
        database,
        chat["id"],
        fresh["id"],
        seconds=chat_store.ABANDONED_TURN_MAX_AGE_SECONDS - 60,
    )

    retired = chat_store.ChatStore(database).purge_abandoned_turns("owner-uid", chat["id"])

    assert retired == []
    assert database.turns("owner-uid", chat["id"])[fresh["id"]]["status"] == "pending"


def test_sweep_never_touches_completed_or_failed_turns(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    store = chat_store.ChatStore(database)
    done = create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "Done"})
    store.complete_turn(
        "owner-uid", chat["id"], done["id"], **completion_payload(question="Done")
    )
    broken = create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "Broken"})
    store.fail_turn("owner-uid", chat["id"], broken["id"], error_code="cancelled")
    for turn_id in (done["id"], broken["id"]):
        _age_turn(
            database,
            chat["id"],
            turn_id,
            seconds=chat_store.ABANDONED_TURN_MAX_AGE_SECONDS + 600,
        )

    assert store.purge_abandoned_turns("owner-uid", chat["id"]) == []
    stored = database.turns("owner-uid", chat["id"])
    assert stored[done["id"]]["status"] == "completed"
    assert stored[broken["id"]]["error_code"] == "cancelled"


def test_chat_count_is_capped_per_owner_and_scoped_to_that_owner(chat_api, monkeypatch):
    """Chats kosten keinen Usage-Slot - ohne Deckel waechst Firestore beliebig."""
    monkeypatch.setattr(chat_store, "MAX_CHATS_PER_OWNER", 3)
    client, database = chat_api

    for _ in range(3):
        create_chat(client)
    blocked = client.post("/chats", json={}, headers=AUTH_OWNER)
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["error_code"] == "chat_limit_reached"
    assert len(database.chats("owner-uid")) == 3

    # Das Limit ist pro Konto, nicht global.
    other = client.post("/chats", json={}, headers=AUTH_OTHER)
    assert other.status_code == 201

    # Platz schaffen macht wieder Platz.
    victim = next(iter(database.chats("owner-uid")))
    assert client.delete(f"/chats/{victim}", headers=AUTH_OWNER).status_code == 200
    assert client.post("/chats", json={}, headers=AUTH_OWNER).status_code == 201


def test_chat_count_falls_back_to_a_bounded_scan_without_aggregation(
    chat_api, monkeypatch
):
    """Ohne count()-Aggregation greift der begrenzte Scan - und zaehlt gleich."""
    monkeypatch.setattr(chat_store, "MAX_CHATS_PER_OWNER", 2)
    monkeypatch.delattr(FakeCollectionRef, "count")
    client, database = chat_api

    create_chat(client)
    create_chat(client)
    blocked = client.post("/chats", json={}, headers=AUTH_OWNER)

    assert blocked.status_code == 403
    assert blocked.json()["detail"]["error_code"] == "chat_limit_reached"
    assert len(database.chats("owner-uid")) == 2


def test_turns_per_chat_are_capped_without_corrupting_the_counter(
    chat_api, monkeypatch
):
    monkeypatch.setattr(chat_store, "MAX_TURNS_PER_CHAT", 2)
    client, database = chat_api
    chat = create_chat(client)

    create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "One"})
    create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "Two"})
    blocked = client.post(
        f"/chats/{chat['id']}/turns",
        json={**TURN_PAYLOAD, "question": "Three"},
        headers=AUTH_OWNER,
    )

    assert blocked.status_code == 403
    assert blocked.json()["detail"]["error_code"] == "turn_limit_reached"
    # Die abgelehnte Anlage darf weder einen Turn noch einen Zaehlerstand
    # hinterlassen - sonst waere der Chat danach dauerhaft inkonsistent.
    assert len(database.turns("owner-uid", chat["id"])) == 2
    assert database.chats("owner-uid")[chat["id"]]["turn_count"] == 2


def test_write_endpoints_are_rate_limited_per_account_not_only_per_ip(
    chat_api, monkeypatch
):
    """Das IP-Limit ist hier die falsche Achse (IPv6 umgeht es, NAT teilt es).

    Der UID-Limiter sitzt hinter der Token-Pruefung, also trifft er genau das
    Konto, das die Schreibzugriffe ausloest - und nur dieses.
    """
    monkeypatch.setitem(chat_history_router.CHAT_UID_RATE_LIMITS, "create_chat", 3)
    client, _database = chat_api

    for _ in range(3):
        assert client.post("/chats", json={}, headers=AUTH_OWNER).status_code == 201
    throttled = client.post("/chats", json={}, headers=AUTH_OWNER)
    assert throttled.status_code == 429
    assert throttled.json()["detail"]["error_code"] == "chat_rate_limited"

    # Ein anderes Konto von derselben IP bleibt unberuehrt.
    assert client.post("/chats", json={}, headers=AUTH_OTHER).status_code == 201

    # Lesende Endpoints haengen nicht am Schreib-Budget.
    assert client.get("/chats", headers=AUTH_OWNER).status_code == 200


def test_context_uid_budget_is_charged_on_post_not_turn_get(chat_api, monkeypatch):
    client, _database = chat_api
    operations = []

    def uid_for(_request, operation=""):
        operations.append(operation)
        return "owner-uid"

    class Store:
        def get_turn(self, uid, chat_id, turn_id):
            return {"id": turn_id, "status": "pending"}

    class Repository:
        def load_target_and_predecessors(self, uid, chat_id, turn_id):
            return {"status": "completed"}, []

    class ContextService:
        def __init__(self, repository):
            self.repository = repository

        def build_for_turn(self, uid, chat_id, turn_id, compressor=None, **_kwargs):
            return {"id": "context-version"}

    monkeypatch.setattr(chat_history_router, "_chat_uid", uid_for)
    monkeypatch.setattr(chat_history_router, "_store", lambda: Store())
    monkeypatch.setattr(
        chat_history_router, "_context_repository", lambda: Repository()
    )
    monkeypatch.setattr(chat_history_router, "ChatContextService", ContextService)

    chat_id = "a" * 32
    turn_id = "b" * 32
    get_response = client.get(
        f"/chats/{chat_id}/turns/{turn_id}", headers=AUTH_OWNER
    )
    post_response = client.post(
        f"/chats/{chat_id}/turns/{turn_id}/context",
        json={},
        headers=AUTH_OWNER,
    )

    assert get_response.status_code == 200
    assert post_response.status_code == 200
    assert operations == ["", "build_context"]


def test_creating_a_turn_retires_an_abandoned_predecessor(chat_api):
    client, database = chat_api
    chat = create_chat(client)
    abandoned = create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "Lost"})
    _age_turn(
        database,
        chat["id"],
        abandoned["id"],
        seconds=chat_store.ABANDONED_TURN_MAX_AGE_SECONDS + 60,
    )

    fresh = create_turn(client, chat["id"], {**TURN_PAYLOAD, "question": "Next"})

    stored = database.turns("owner-uid", chat["id"])
    assert stored[abandoned["id"]]["error_code"] == "abandoned"
    assert stored[fresh["id"]]["status"] == "pending"


def test_a_failing_sweep_never_blocks_turn_creation(chat_api, monkeypatch):
    client, database = chat_api
    chat = create_chat(client)

    def explode(self, uid, chat_id, *, now=None):
        raise RuntimeError("firestore unavailable")

    monkeypatch.setattr(chat_store.ChatStore, "purge_abandoned_turns", explode)
    turn = create_turn(client, chat["id"])

    assert database.turns("owner-uid", chat["id"])[turn["id"]]["status"] == "pending"


def test_turn_keeps_the_attachment_meta_of_its_own_question(chat_api):
    """Ein Anhang gehoert zu der Frage, mit der er rausgegangen ist.

    Vorher hing er am Bookmark-Dokument, das nur EINE Fassung kennt: die
    naechste Frage ohne Datei ueberschrieb sie und der Anhang war aus dem
    gespeicherten Chat verschwunden. Am Turn ueberlebt er jede Folgefrage.
    """
    client, database = chat_api
    chat = create_chat(client)

    with_file = create_turn(
        client,
        chat["id"],
        {
            **TURN_PAYLOAD,
            "question": "What does this chart show?",
            "attachments": [
                # Der Browser meldet .csv je nach System unterschiedlich; der
                # Server kennt nur den kanonischen Typ.
                {"name": "chart.png", "mime": "image/png", "size": 2048},
                {"name": "rows.csv", "mime": "text/csv", "size": 900},
            ],
        },
    )
    without_file = create_turn(
        client, chat["id"], {**TURN_PAYLOAD, "question": "And in euros?"}
    )

    assert with_file["attachments"] == [
        {"name": "chart.png", "mime": "image/png", "size": 2048},
        {"name": "rows.csv", "mime": "text/plain", "size": 900},
    ]
    # Die Folgefrage sagt nichts ueber die Datei der vorigen aus - und raeumt
    # sie auch nicht weg.
    assert "attachments" not in without_file
    stored = database.turns("owner-uid", chat["id"])
    assert stored[with_file["id"]]["attachments"][0]["name"] == "chart.png"
    assert "attachments" not in stored[without_file["id"]]

    page = client.get(f"/chats/{chat['id']}/turns", headers=AUTH_OWNER)
    listed = {turn["id"]: turn for turn in page.json()["turns"]}
    assert listed[with_file["id"]]["attachments"][1]["mime"] == "text/plain"


def test_turn_attachments_never_carry_file_data_or_unknown_types(chat_api):
    """Datei-Bytes gehoeren an die Modelle, nicht nach Firestore."""
    client, database = chat_api
    chat = create_chat(client)

    turn = create_turn(
        client,
        chat["id"],
        {
            **TURN_PAYLOAD,
            "attachments": [
                {
                    "name": "secret.png",
                    "mime": "image/png",
                    "size": 10,
                    "data": "must-not-survive",
                },
                {"name": "payload.exe", "mime": "application/x-msdownload", "size": 5},
                {"name": "", "mime": "image/png", "size": 5},
            ],
        },
    )

    assert turn["attachments"] == [
        {"name": "secret.png", "mime": "image/png", "size": 10}
    ]
    stored = database.turns("owner-uid", chat["id"])[turn["id"]]
    assert "must-not-survive" not in repr(stored)


def test_turn_attachment_meta_survives_completion(chat_api):
    """Die Antwort ueberschreibt die Frage nicht.

    complete_turn schreibt in dasselbe Dokument; ein set() statt update()
    haette den Anhang der Frage still mitgeloescht.
    """
    client, database = chat_api
    chat = create_chat(client)
    turn = create_turn(
        client,
        chat["id"],
        {
            **TURN_PAYLOAD,
            "attachments": [{"name": "chart.png", "mime": "image/png", "size": 2048}],
        },
    )
    chat_store.ChatStore(database).complete_turn(
        "owner-uid", chat["id"], turn["id"], **completion_payload()
    )

    detail = client.get(
        f"/chats/{chat['id']}/turns/{turn['id']}", headers=AUTH_OWNER
    ).json()["turn"]
    assert detail["attachments"] == [
        {"name": "chart.png", "mime": "image/png", "size": 2048}
    ]
