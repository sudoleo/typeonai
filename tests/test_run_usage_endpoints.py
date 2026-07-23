"""Integration der persistenten Run-Usage in die bestehenden API-Flows."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.api_core.exceptions import Aborted

import app.core.config as cfg
from app.api.routers import chat as chat_router
from app.api.routers import users as users_router
from app.core.rate_limit import limiter
from app.services.usage_repository import UsageLimits
from usage_test_support import make_usage_repository


UID = "run-endpoint-user"
AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def run_api(monkeypatch):
    limiter.reset()
    repository, _ = make_usage_repository()
    monkeypatch.setattr(chat_router, "run_usage_repository", repository)
    monkeypatch.setattr(users_router, "run_usage_repository", repository)
    monkeypatch.setattr(chat_router, "verify_user_token", lambda token: UID)
    monkeypatch.setattr(users_router, "verify_user_token", lambda token, **kwargs: UID)
    monkeypatch.setattr(chat_router, "is_user_pro", lambda uid: False)
    monkeypatch.setattr(users_router, "is_user_pro", lambda uid: False)
    for env_name in (
        "DEVELOPER_OPENAI_API_KEY",
        "DEVELOPER_MISTRAL_API_KEY",
        "DEVELOPER_ANTHROPIC_API_KEY",
        "DEVELOPER_GEMINI_API_KEY",
        "DEVELOPER_DEEPSEEK_API_KEY",
        "DEVELOPER_GROK_API_KEY",
    ):
        monkeypatch.setenv(env_name, "test-key")

    def fake_run_ask(provider, **kwargs):
        return {
            "response": f"{provider.label} answer",
            "sources": [],
            **kwargs["extras"],
        }

    monkeypatch.setattr(chat_router, "_run_ask", fake_run_ask)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_router.router)
    app.include_router(users_router.router)
    return TestClient(app), repository


def _limits(is_pro=False):
    return UsageLimits(
        total=cfg.get_consensus_run_limit(is_pro),
        deep_think=cfg.get_deep_think_run_limit(is_pro),
    )


def _prepare(client, key, *, deep=False):
    return client.post(
        "/prepare",
        headers=AUTH,
        json={
            "question": "What changed?",
            "usage_run_key": key,
            "deep_search": deep,
        },
    )


def _ask(client, route, provider, key, *, deep=False):
    return client.post(
        route,
        headers=AUTH,
        json={
            "question": "What changed?",
            "model": cfg.FREE_DEFAULT_MODEL_BY_PROVIDER[provider],
            "usage_run_key": key,
            "deep_search": deep,
        },
    )


def test_prepare_and_parallel_models_consume_exactly_one_run(run_api):
    client, repository = run_api
    key = "one-logical-run"

    prepared = _prepare(client, key)
    assert prepared.status_code == 200
    assert prepared.json()["usage_run_status"] == "reserved"
    assert prepared.json()["free_usage_remaining"] == 2

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda args: _ask(client, *args, key),
                [
                    ("/ask_openai", "openai"),
                    ("/ask_mistral", "mistral"),
                ],
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["usage_run_status"] == "consumed" for response in responses)
    snapshot = repository.snapshot(UID, _limits())
    assert snapshot.total.reserved == 0
    assert snapshot.total.consumed == 1
    assert snapshot.total.remaining == 2


def test_consensus_reuses_consumed_run_without_second_charge(run_api):
    client, repository = run_api
    key = "answers-plus-consensus"
    assert _prepare(client, key).status_code == 200
    assert _ask(client, "/ask_openai", "openai", key).status_code == 200
    assert _ask(client, "/ask_mistral", "mistral", key).status_code == 200

    payload = {
        "usage_run_key": key,
        "question": "What changed?",
        "consensus_model": "Gemini",
        "answer_openai": "OpenAI answer",
        "answer_mistral": "Mistral answer",
    }
    with patch.object(chat_router, "query_consensus", return_value="Consensus"), \
         patch.object(chat_router, "query_differences", return_value=("Differences", None)), \
         patch.object(chat_router, "persist_pending_result", return_value=None), \
         patch.object(chat_router, "record_differences_stats"):
        response = client.post("/consensus", headers=AUTH, json=payload)

    assert response.status_code == 200
    assert response.json()["usage_run_status"] == "consumed"
    snapshot = repository.snapshot(UID, _limits())
    assert snapshot.total.consumed == 1
    assert snapshot.total.remaining == 2


def test_usage_endpoint_reads_persistent_snapshot_and_release_frees_reservation(run_api):
    client, repository = run_api
    key = "unused-reservation"
    assert _prepare(client, key).status_code == 200

    usage = client.post("/usage", json={"id_token": "test-token"})
    assert usage.status_code == 200
    assert usage.json()["remaining"] == 2
    assert usage.json()["reserved"] == 1
    assert usage.json()["consumed"] == 0

    released = client.post(
        "/usage/run/release",
        json={"id_token": "test-token", "usage_run_key": key},
    )
    assert released.status_code == 200
    assert released.json()["status"] == "released"
    snapshot = repository.snapshot(UID, _limits())
    assert snapshot.total.reserved == 0
    assert snapshot.total.consumed == 0
    assert snapshot.total.remaining == 3


def test_requests_without_run_key_are_rejected_before_provider_call(run_api):
    client, repository = run_api
    response = client.post(
        "/ask_openai",
        headers=AUTH,
        json={
            "question": "What changed?",
            "model": cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["openai"],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "usage_run_key_required"
    assert repository.snapshot(UID, _limits()).total.consumed == 0


def test_exhausted_firestore_contention_returns_structured_503(run_api, monkeypatch):
    client, repository = run_api
    key = "contention-run"
    assert _prepare(client, key).status_code == 200
    monkeypatch.setattr(repository, "consume", lambda *_args, **_kwargs: (_ for _ in ()).throw(Aborted("contention")))

    response = _ask(client, "/ask_gemini", "gemini", key)

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "usage_storage_busy"


def test_deep_think_counts_once_total_and_once_in_deep_quota(run_api, monkeypatch):
    client, repository = run_api
    monkeypatch.setattr(chat_router, "is_user_pro", lambda uid: True)
    key = "deep-think-run"

    assert _prepare(client, key, deep=True).status_code == 200
    response = _ask(client, "/ask_openai", "openai", key, deep=True)

    assert response.status_code == 200
    snapshot = repository.snapshot(UID, _limits(is_pro=True))
    assert snapshot.total.consumed == 1
    assert snapshot.deep_think.consumed == 1
