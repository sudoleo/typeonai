"""Der Antwort-Vertrag von /consensus.

Die Antworten kommen als Familien-Mapping (`answers`) herein; die historischen
`answer_<familie>`-Felder bleiben lesbar, damit ein alter Client-Stand
weiterlaeuft. Und ein Lauf vergleicht hoechstens cfg.MAX_RUN_FAMILIES Modelle,
egal wie viele Familien konfiguriert sind.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter

UID = "user-answers"
AUTH = {"Authorization": "Bearer token"}


@pytest.fixture
def consensus_api(monkeypatch):
    limiter.reset()
    seen = {}

    def fake_consensus(question, answers, excluded_models, consensus_model, api_keys, **kwargs):
        # Die Domaenen-Pipeline fuellt jede konfigurierte Familie auf; fuer den
        # Vertrag zaehlen die Familien, die tatsaechlich eine Antwort tragen.
        seen["answers"] = {
            provider: text for provider, text in dict(answers).items() if text
        }
        seen["excluded"] = list(excluded_models or [])
        return "Consensus"

    monkeypatch.setattr(chat_router, "verify_user_token", lambda token: UID)
    monkeypatch.setattr(chat_router, "is_user_pro", lambda uid: False)
    monkeypatch.setattr(chat_router, "query_consensus", fake_consensus)
    monkeypatch.setattr(
        chat_router,
        "query_differences",
        lambda *args, **kwargs: ("Differences", {"agreement": {"score": 80}}),
    )
    monkeypatch.setattr(chat_router, "persist_pending_result", lambda **kwargs: "result-id")
    monkeypatch.setattr(chat_router, "record_differences_stats", lambda *args, **kwargs: None)

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_router.router)
    return TestClient(app), seen


def _payload(**updates):
    payload = {
        "question": "What changed?",
        "consensus_model": "Gemini",
        "useOwnKeys": True,
        "openrouter_key": "own-key-secret",
    }
    payload.update(updates)
    return payload


def test_answers_field_keyed_by_family(consensus_api):
    client, seen = consensus_api

    response = client.post("/consensus", headers=AUTH, json=_payload(
        answers={"openai": "first answer", "grok": "second answer"},
    ))

    assert response.status_code == 200
    assert seen["answers"] == {"openai": "first answer", "grok": "second answer"}


def test_answers_field_keyed_by_display_name(consensus_api):
    client, seen = consensus_api

    response = client.post("/consensus", headers=AUTH, json=_payload(
        answers={"OpenAI": "first answer", "DeepSeek": "second answer"},
    ))

    assert response.status_code == 200
    assert seen["answers"] == {"openai": "first answer", "deepseek": "second answer"}


def test_legacy_answer_fields_still_work(consensus_api):
    client, seen = consensus_api

    response = client.post("/consensus", headers=AUTH, json=_payload(
        answer_openai="first answer",
        answer_claude="second answer",
    ))

    assert response.status_code == 200
    assert seen["answers"] == {"openai": "first answer", "anthropic": "second answer"}


def test_excluded_family_is_dropped_from_the_run(consensus_api):
    client, seen = consensus_api

    response = client.post("/consensus", headers=AUTH, json=_payload(
        answers={"openai": "first answer", "grok": "second answer", "gemini": "third"},
        excluded_models=["Gemini"],
    ))

    assert response.status_code == 200
    assert "gemini" not in seen["answers"]


def test_a_run_never_compares_more_than_the_configured_cap(consensus_api):
    client, _ = consensus_api
    answers = {
        provider: f"answer {index}"
        for index, provider in enumerate(cfg.PROVIDERS)
    }
    assert len(answers) <= cfg.MAX_RUN_FAMILIES

    response = client.post("/consensus", headers=AUTH, json=_payload(answers=answers))
    assert response.status_code == 200

    original = cfg.MAX_RUN_FAMILIES
    try:
        cfg.MAX_RUN_FAMILIES = len(answers) - 1
        chat_router.cfg.MAX_RUN_FAMILIES = cfg.MAX_RUN_FAMILIES
        limiter.reset()
        response = client.post("/consensus", headers=AUTH, json=_payload(answers=answers))
        assert response.status_code == 400
        assert "at most" in response.json()["detail"].lower()
    finally:
        cfg.MAX_RUN_FAMILIES = original
        chat_router.cfg.MAX_RUN_FAMILIES = original
