from __future__ import annotations

import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter
from app.services.chat_store import ChatNotFound, TurnQuestionConflict


UID = "chat-consensus-owner"
CHAT_ID = "a" * 32
TURN_ID = "b" * 32
CONTEXT_VERSION_ID = "c" * 32
RESULT_ID = "AbCdEf0123456789"
AUTH = {"Authorization": "Bearer test-token"}


class RecordingStore:
    def __init__(self):
        self.validations = []
        self.reads = []
        self.completions = []
        self.failures = []
        self.validation_error = None
        self.completion_error = None
        self.validation_result = {
            "id": TURN_ID,
            "status": "pending",
            "question": "What changed?",
        }
        self.turn_detail = None

    def validate_turn_for_completion(self, uid, chat_id, turn_id, *, question):
        self.validations.append((uid, chat_id, turn_id, question))
        if self.validation_error:
            raise self.validation_error
        return self.validation_result

    def get_turn(self, uid, chat_id, turn_id):
        self.reads.append((uid, chat_id, turn_id))
        if self.turn_detail is None:
            raise AssertionError("unexpected completed-turn read")
        return self.turn_detail

    def complete_turn(self, uid, chat_id, turn_id, **payload):
        self.completions.append((uid, chat_id, turn_id, payload))
        if self.completion_error:
            raise self.completion_error
        return {"id": turn_id, "status": "completed"}

    def fail_turn(self, uid, chat_id, turn_id, *, error_code):
        self.failures.append((uid, chat_id, turn_id, error_code))
        return {"id": turn_id, "status": "failed"}


def _base_payload(**updates):
    payload = {
        "question": "What changed?",
        "consensus_model": "Gemini",
        "answer_openai": "OpenAI answer",
        "answer_mistral": "Mistral answer",
        "model_labels": {
            "OpenAI": "gpt-test",
            "Mistral": "mistral-test",
        },
        "model_sources": {
            "OpenAI": [{
                "url": "https://openai.example/source",
                "title": "OpenAI source",
                "provider": "OpenAI",
            }],
            "Mistral": [{
                "url": "https://mistral.example/source",
                "title": "Mistral source",
                "provider": "Mistral",
            }],
        },
        "chat_id": CHAT_ID,
        "turn_id": TURN_ID,
        "useOwnKeys": True,
        "gemini_key": "own-key-secret",
    }
    payload.update(updates)
    return payload


def _final_sse_payload(response):
    matches = re.findall(r"event: final\r?\ndata: (.+?)\r?\n\r?\n", response.text)
    assert matches, response.text
    return json.loads(matches[-1])


@pytest.fixture
def chat_consensus_api(monkeypatch):
    limiter.reset()
    store = RecordingStore()
    monkeypatch.setattr(chat_router, "chat_store", store)
    monkeypatch.setattr(chat_router, "verify_user_token", lambda token: UID)
    monkeypatch.setattr(chat_router, "is_user_pro", lambda uid: False)
    monkeypatch.setattr(chat_router, "query_consensus", lambda *args, **kwargs: "Consensus")
    monkeypatch.setattr(
        chat_router,
        "query_differences",
        lambda *args, **kwargs: ("Differences", {"agreement": {"score": 88}}),
    )
    monkeypatch.setattr(chat_router, "persist_pending_result", lambda **kwargs: RESULT_ID)
    monkeypatch.setattr(chat_router, "record_differences_stats", lambda *args, **kwargs: None)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_router.router)
    return TestClient(app), store, monkeypatch


def test_consensus_without_chat_ids_remains_legacy_compatible(chat_consensus_api):
    client, store, _ = chat_consensus_api
    payload = _base_payload()
    payload.pop("chat_id")
    payload.pop("turn_id")

    response = client.post("/consensus", headers=AUTH, json=payload)

    assert response.status_code == 200
    assert response.json()["consensus_response"] == "Consensus"
    assert "chat_persisted" not in response.json()
    assert store.validations == []
    assert store.completions == []


@pytest.mark.parametrize("missing", ["chat_id", "turn_id"])
def test_consensus_requires_chat_and_turn_ids_together(chat_consensus_api, missing):
    client, store, _ = chat_consensus_api
    payload = _base_payload()
    payload.pop(missing)

    response = client.post("/consensus", headers=AUTH, json=payload)

    assert response.status_code == 400
    assert store.validations == []


@pytest.mark.parametrize("field,value", [
    ("chat_id", "A" * 32),
    ("turn_id", "../turn"),
    ("chat_id", "a" * 31),
])
def test_consensus_rejects_noncanonical_chat_ids(chat_consensus_api, field, value):
    client, store, _ = chat_consensus_api

    response = client.post(
        "/consensus", headers=AUTH, json=_base_payload(**{field: value})
    )

    assert response.status_code == 400
    assert store.validations == []


def test_consensus_context_version_requires_ids_and_canonical_value(chat_consensus_api):
    client, store, _ = chat_consensus_api
    no_ids = _base_payload(context_version_id=CONTEXT_VERSION_ID)
    no_ids.pop("chat_id")
    no_ids.pop("turn_id")

    response = client.post("/consensus", headers=AUTH, json=no_ids)
    assert response.status_code == 400
    assert store.validations == []

    response = client.post(
        "/consensus", headers=AUTH, json=_base_payload(context_version_id="invalid")
    )
    assert response.status_code == 400
    assert store.validations == []


@pytest.mark.parametrize("supplied", [None, "d" * 32])
def test_consensus_requires_exact_context_version_link_before_engine(
    chat_consensus_api, supplied
):
    client, store, monkeypatch = chat_consensus_api
    store.validation_result = {
        "id": TURN_ID,
        "status": "pending",
        "question": "What changed?",
        "context_version_id": CONTEXT_VERSION_ID,
    }
    engine_calls = []
    monkeypatch.setattr(
        chat_router, "query_consensus", lambda *args, **kwargs: engine_calls.append(True)
    )
    payload = _base_payload()
    if supplied is not None:
        payload["context_version_id"] = supplied

    response = client.post("/consensus", headers=AUTH, json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["chat_turn_state"] == "pending"
    assert store.failures == []
    assert store.completions == []
    assert engine_calls == []


def test_consensus_accepts_exact_linked_context_version(chat_consensus_api):
    client, store, _ = chat_consensus_api
    store.validation_result["context_version_id"] = CONTEXT_VERSION_ID

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(context_version_id=CONTEXT_VERSION_ID),
    )

    assert response.status_code == 200
    assert response.json()["chat_turn_state"] == "completed"
    assert len(store.completions) == 1


def test_consensus_rejects_foreign_or_unknown_turn_before_engine(chat_consensus_api):
    client, store, monkeypatch = chat_consensus_api
    engine_calls = []
    store.validation_error = ChatNotFound("not found")
    monkeypatch.setattr(
        chat_router,
        "query_consensus",
        lambda *args, **kwargs: engine_calls.append(True),
    )

    response = client.post("/consensus", headers=AUTH, json=_base_payload())

    assert response.status_code == 404
    assert response.json()["detail"]["chat_turn_state"] == "failed"
    assert response.json()["detail"]["chat_id"] == CHAT_ID
    assert response.json()["detail"]["turn_id"] == TURN_ID
    assert engine_calls == []
    assert store.completions == []


def test_consensus_rejects_question_mismatch_before_engine(chat_consensus_api):
    client, store, monkeypatch = chat_consensus_api
    engine_calls = []
    store.validation_error = TurnQuestionConflict("mismatch")
    monkeypatch.setattr(
        chat_router,
        "query_consensus",
        lambda *args, **kwargs: engine_calls.append(True),
    )

    response = client.post("/consensus", headers=AUTH, json=_base_payload())

    assert response.status_code == 409
    assert response.json()["detail"]["chat_turn_state"] == "failed"
    assert response.json()["detail"]["chat_id"] == CHAT_ID
    assert response.json()["detail"]["turn_id"] == TURN_ID
    assert engine_calls == []
    assert store.completions == []


def test_non_streaming_completion_maps_answers_labels_sources_and_result(chat_consensus_api):
    client, store, _ = chat_consensus_api
    turn_sources = [{
        "url": "https://global.example/source",
        "title": "Global source",
        "provider": "Search",
        "ignored": "discard me",
    }]

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(turn_sources=turn_sources),
    )

    assert response.status_code == 200
    assert response.json()["chat_id"] == CHAT_ID
    assert response.json()["turn_id"] == TURN_ID
    assert response.json()["chat_persisted"] is True
    assert response.json()["chat_turn_state"] == "completed"
    assert len(store.completions) == 1
    uid, chat_id, turn_id, completion = store.completions[0]
    assert (uid, chat_id, turn_id) == (UID, CHAT_ID, TURN_ID)
    assert completion["question"] == "What changed?"
    assert completion["consensus"] == "Consensus"
    assert completion["differences"] == "Differences"
    assert completion["differences_data"]["agreement"]["score"] == 88
    assert completion["result_id"] == RESULT_ID
    assert completion["sources"] == [{
        "id": "",
        "url": "https://global.example/source",
        "title": "Global source",
        "provider": "Search",
    }]
    assert set(completion["model_answers"]) == {"OpenAI", "Mistral"}
    assert completion["model_answers"]["OpenAI"] == {
        "provider": "OpenAI",
        "answer": "OpenAI answer",
        "model_label": "gpt-test",
        "sources": _base_payload()["model_sources"]["OpenAI"],
    }
    assert "own-key-secret" not in json.dumps(completion)


def test_excluded_and_empty_providers_are_not_completed(chat_consensus_api):
    client, store, _ = chat_consensus_api

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(
            answer_gemini="Gemini answer",
            answer_grok="   ",
            excluded_models=["Gemini"],
            model_labels={"OpenAI": "o", "Mistral": "m", "Gemini": "g"},
        ),
    )

    assert response.status_code == 200
    completed = store.completions[0][3]["model_answers"]
    assert set(completed) == {"OpenAI", "Mistral"}


def test_missing_turn_sources_falls_back_to_deduplicated_model_sources(chat_consensus_api):
    client, store, _ = chat_consensus_api
    duplicate = {
        "url": "https://same.example/source",
        "title": "Same",
        "provider": "Search",
    }
    payload = _base_payload(model_sources={
        "OpenAI": [duplicate],
        "Mistral": [duplicate],
    })

    response = client.post("/consensus", headers=AUTH, json=payload)

    assert response.status_code == 200
    assert store.completions[0][3]["sources"] == [{"id": "", **duplicate}]


def test_streaming_success_completes_exactly_once(chat_consensus_api):
    client, store, monkeypatch = chat_consensus_api

    def consensus_stream(*args, **kwargs):
        yield {"type": "delta", "text": "Con"}
        yield {"type": "final", "text": "Consensus"}

    def differences_stream(*args, **kwargs):
        yield {"type": "final", "text": "Differences", "data": {"agreement": {"score": 91}}}

    monkeypatch.setattr(chat_router, "stream_consensus", consensus_stream)
    monkeypatch.setattr(chat_router, "stream_differences", differences_stream)

    response = client.post(
        "/consensus", headers=AUTH, json=_base_payload(stream=True)
    )
    final = _final_sse_payload(response)

    assert response.status_code == 200
    assert final["consensus_response"] == "Consensus"
    assert final["chat_id"] == CHAT_ID
    assert final["turn_id"] == TURN_ID
    assert final["chat_persisted"] is True
    assert final["chat_turn_state"] == "completed"
    assert len(store.completions) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_terminal_consensus_error_marks_pending_turn_failed_best_effort(
    chat_consensus_api, stream
):
    client, store, monkeypatch = chat_consensus_api
    if stream:
        monkeypatch.setattr(
            chat_router,
            "stream_consensus",
            lambda *args, **kwargs: iter([{
                "type": "final",
                "text": "Consensus error: unavailable",
                "error": True,
            }]),
        )
    else:
        monkeypatch.setattr(
            chat_router,
            "query_consensus",
            lambda *args, **kwargs: "Consensus error: unavailable",
        )

    response = client.post(
        "/consensus", headers=AUTH, json=_base_payload(stream=stream)
    )
    body = _final_sse_payload(response) if stream else response.json()

    assert body["chat_persisted"] is False
    assert body["chat_turn_state"] == "failed"
    assert store.completions == []
    assert store.failures == [(UID, CHAT_ID, TURN_ID, "consensus_failed")]


def test_server_side_insufficient_answers_fails_pending_turn(chat_consensus_api):
    client, store, _ = chat_consensus_api

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(answer_mistral=""),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "Missing parameters: at least two selected model answers",
        "chat_id": CHAT_ID,
        "turn_id": TURN_ID,
        "chat_turn_state": "failed",
        "chat_persisted": False,
    }
    assert store.failures == [(UID, CHAT_ID, TURN_ID, "insufficient_answers")]
    assert store.completions == []


def test_insufficient_answers_fail_before_own_key_or_engine_checks(
    chat_consensus_api,
):
    client, store, monkeypatch = chat_consensus_api
    forbidden = []

    def forbid(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError(f"{name} must not run for a disposition-only failure")
        return call

    monkeypatch.setattr(chat_router, "build_engine_api_keys", forbid("keys"))
    monkeypatch.setattr(chat_router, "query_consensus", forbid("consensus"))
    monkeypatch.setattr(chat_router, "query_differences", forbid("differences"))
    monkeypatch.setattr(chat_router, "persist_pending_result", forbid("share"))
    monkeypatch.setattr(chat_router, "record_differences_stats", forbid("stats"))
    monkeypatch.setattr(chat_router, "reserve_usage_run", forbid("usage_reserve"))
    monkeypatch.setattr(chat_router, "consume_usage_run", forbid("usage_consume"))

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(answer_mistral="", gemini_key=""),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["chat_turn_state"] == "failed"
    assert store.failures == [(UID, CHAT_ID, TURN_ID, "insufficient_answers")]
    assert store.completions == []
    assert forbidden == []


def test_insufficient_answers_disposition_precedes_current_model_and_tier_checks(
    chat_consensus_api,
):
    client, store, monkeypatch = chat_consensus_api
    forbidden = []

    def forbid_keys(*args, **kwargs):
        forbidden.append("keys")
        raise AssertionError("credentials must not be resolved")

    monkeypatch.setattr(chat_router, "build_engine_api_keys", forbid_keys)

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(
            consensus_model="Retired-Pro",
            deep_search=True,
            answer_mistral="",
            gemini_key="",
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["chat_turn_state"] == "failed"
    assert store.failures == [(UID, CHAT_ID, TURN_ID, "insufficient_answers")]
    assert forbidden == []


@pytest.mark.parametrize("stream", [False, True])
def test_completion_storage_failure_never_replaces_successful_consensus(
    chat_consensus_api, stream
):
    client, store, monkeypatch = chat_consensus_api
    store.completion_error = RuntimeError("storage unavailable")
    if stream:
        monkeypatch.setattr(
            chat_router,
            "stream_consensus",
            lambda *args, **kwargs: iter([{"type": "final", "text": "Consensus"}]),
        )
        monkeypatch.setattr(
            chat_router,
            "stream_differences",
            lambda *args, **kwargs: iter([{
                "type": "final", "text": "Differences", "data": None
            }]),
        )

    response = client.post(
        "/consensus", headers=AUTH, json=_base_payload(stream=stream)
    )
    body = _final_sse_payload(response) if stream else response.json()

    assert response.status_code == 200
    assert body["consensus_response"] == "Consensus"
    assert body["chat_persisted"] is False
    assert body["chat_turn_state"] == "pending"
    assert store.failures == []
    assert len(store.completions) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_completed_turn_replays_without_engine_writes_or_usage(
    chat_consensus_api, stream
):
    client, store, monkeypatch = chat_consensus_api
    store.validation_result = {
        "id": TURN_ID,
        "status": "completed",
        "question": "What changed?",
    }
    store.turn_detail = {
        "id": TURN_ID,
        "status": "completed",
        "question": "What changed?",
        "consensus": "Stored consensus",
        "differences": "Stored differences",
        "differences_data": {"agreement": {"score": 94}},
        "result_id": RESULT_ID,
        "model_answers": {},
    }
    forbidden = []

    def forbid(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError(f"{name} must not run during replay")
        return call

    monkeypatch.setattr(chat_router, "query_consensus", forbid("consensus"))
    monkeypatch.setattr(chat_router, "stream_consensus", forbid("consensus_stream"))
    monkeypatch.setattr(chat_router, "query_differences", forbid("differences"))
    monkeypatch.setattr(chat_router, "stream_differences", forbid("differences_stream"))
    monkeypatch.setattr(chat_router, "persist_pending_result", forbid("share"))
    monkeypatch.setattr(chat_router, "record_differences_stats", forbid("stats"))
    monkeypatch.setattr(chat_router, "reserve_usage_run", forbid("usage_reserve"))
    monkeypatch.setattr(chat_router, "consume_usage_run", forbid("usage_consume"))

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(stream=stream, useOwnKeys=True, gemini_key=""),
    )
    body = _final_sse_payload(response) if stream else response.json()

    assert response.status_code == 200
    assert body == {
        "consensus_response": "Stored consensus",
        "differences": "Stored differences",
        "differences_data": {"agreement": {"score": 94}},
        "sources": [],
        "model_answers": {},
        "chat_id": CHAT_ID,
        "turn_id": TURN_ID,
        "chat_turn_state": "completed",
        "chat_persisted": True,
        "chat_replayed": True,
        "result_id": RESULT_ID,
    }
    if stream:
        assert "event: consensus.final" in response.text
        assert "Stored consensus" in response.text
    assert store.reads == [(UID, CHAT_ID, TURN_ID)]
    assert store.completions == []
    assert forbidden == []


def test_completed_replay_precedes_current_model_tier_and_credentials(
    chat_consensus_api,
):
    client, store, monkeypatch = chat_consensus_api
    store.validation_result = {
        "id": TURN_ID,
        "status": "completed",
        "question": "What changed?",
    }
    store.turn_detail = {
        "id": TURN_ID,
        "status": "completed",
        "question": "What changed?",
        "consensus": "Historical premium consensus",
        "differences": "Stored differences",
        "model_answers": {},
    }
    forbidden = []

    def forbid(name):
        def call(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError(f"{name} must not run during completed replay")
        return call

    monkeypatch.setattr(chat_router, "build_engine_api_keys", forbid("keys"))
    monkeypatch.setattr(chat_router, "query_consensus", forbid("consensus"))
    monkeypatch.setattr(chat_router, "query_differences", forbid("differences"))
    monkeypatch.setattr(chat_router, "reserve_usage_run", forbid("usage_reserve"))
    monkeypatch.setattr(chat_router, "consume_usage_run", forbid("usage_consume"))

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(
            consensus_model="historical-model-no-longer-allowed",
            deep_search=True,
            useOwnKeys=True,
            gemini_key="",
        ),
    )

    assert response.status_code == 200
    assert response.json()["consensus_response"] == "Historical premium consensus"
    assert response.json()["chat_replayed"] is True
    assert store.completions == []
    assert forbidden == []


def test_completed_turn_without_stored_consensus_fails_closed(chat_consensus_api):
    client, store, monkeypatch = chat_consensus_api
    store.validation_result = {"id": TURN_ID, "status": "completed"}
    store.turn_detail = {
        "id": TURN_ID,
        "status": "completed",
        "consensus": "",
        "model_answers": {},
    }
    engine_calls = []
    monkeypatch.setattr(
        chat_router,
        "query_consensus",
        lambda *args, **kwargs: engine_calls.append(True),
    )

    response = client.post("/consensus", headers=AUTH, json=_base_payload())

    assert response.status_code == 409
    assert response.json()["detail"]["chat_turn_state"] == "failed"
    assert response.json()["detail"]["chat_persisted"] is False
    assert engine_calls == []
    assert store.completions == []


def test_correctable_missing_own_key_keeps_turn_retryable(chat_consensus_api):
    client, store, _ = chat_consensus_api

    response = client.post(
        "/consensus",
        headers=AUTH,
        json=_base_payload(gemini_key=""),
    )

    assert response.status_code == 400
    assert "chat_turn_state" not in response.json()
    assert store.failures == []
    assert store.completions == []
