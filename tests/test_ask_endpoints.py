"""Endpoint-Tests fuer die deduplizierten /ask_*-Handler (handle_ask).

Nagelt die Provider-Eigenheiten fest, die beim Refactoring erhalten bleiben
mussten: Gemini-Sonderpfade (useOwnKeys, Service Account, 401 statt 400),
Own-Key-Bypass der Usage-Zaehlung und die Usage-Limit-Antworten.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter
from app.services.usage_repository import RunKind, UsageLimits
from usage_test_support import make_usage_repository


@pytest.fixture(autouse=True)
def reset_rate_limiter(monkeypatch):
    # Die /ask_*-Routen sind mit 3-5/minute limitiert; mehrere Tests teilen
    # sich denselben In-Memory-Limiter (Key: Test-Client-IP).
    limiter.reset()
    repository, _ = make_usage_repository()
    monkeypatch.setattr(chat_router, "run_usage_repository", repository)
    monkeypatch.setattr(
        chat_router,
        "get_usage_run_key",
        lambda data: str(data.get("usage_run_key") or "test-run-key"),
    )
    yield repository


def make_client():
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_router.router)
    return TestClient(app)


def free_model(provider: str) -> str:
    return cfg.FREE_DEFAULT_MODEL_BY_PROVIDER[provider]


def auth_patches(uid="uid-ask-tests", is_pro=False, is_early=False):
    return (
        patch.object(chat_router, "verify_user_token", return_value=uid),
        patch.object(chat_router, "is_user_pro", return_value=is_pro),
        patch.object(chat_router, "is_user_early", return_value=is_early),
    )


AUTH_HEADER = {"Authorization": "Bearer test-token"}


def test_no_auth_error_is_provider_specific():
    client = make_client()

    response = client.post(
        "/ask_mistral",
        json={"question": "hello", "model": free_model("mistral")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "No auth provided."

    response = client.post(
        "/ask_gemini",
        json={"question": "hello", "model": free_model("gemini")},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_gemini_own_keys_flag_without_key_is_rejected():
    client = make_client()
    p1, p2, p3 = auth_patches()
    with p1, p2, p3:
        response = client.post(
            "/ask_gemini",
            headers=AUTH_HEADER,
            json={
                "question": "hello",
                "model": free_model("gemini"),
                "useOwnKeys": "true",
            },
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing user API key for Gemini."


def test_deep_search_is_pro_only():
    client = make_client()
    p1, p2, p3 = auth_patches(is_pro=False)
    with p1, p2, p3:
        response = client.post(
            "/ask_grok",
            headers=AUTH_HEADER,
            json={
                "question": "hello",
                "model": free_model("grok"),
                "deep_search": "true",
            },
        )
    assert response.status_code == 403
    assert "Pro users" in response.json()["detail"]


def test_usage_limit_blocks_developer_key_path(reset_rate_limiter):
    client = make_client()
    uid = "uid-limit-reached"
    limits = UsageLimits(total=cfg.get_consensus_run_limit(False), deep_think=0)
    for index in range(limits.total):
        key = f"used-{index}"
        reset_rate_limiter.reserve(uid, key, RunKind.REGULAR, limits)
        reset_rate_limiter.consume(uid, key)
    p1, p2, p3 = auth_patches(uid=uid)
    with p1, p2, p3:
        response = client.post(
            "/ask_deepseek",
            headers=AUTH_HEADER,
            json={"question": "hello", "model": free_model("deepseek")},
        )
    assert response.status_code == 403
    body = response.json()["detail"]
    assert body["error_code"] == "total_usage_limit_exceeded"
    assert body["free_usage_remaining"] == 0


def test_gemini_developer_path_uses_service_account_and_counts_usage(reset_rate_limiter):
    client = make_client()
    uid = "uid-gemini-dev"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return {"ok": True}

    try:
        p1, p2, p3 = auth_patches(uid=uid)
        with p1, p2, p3, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_gemini",
                headers=AUTH_HEADER,
                json={"question": "hello", "model": free_model("gemini")},
            )
        assert response.status_code == 200
        assert captured["provider"].label == "Gemini"
        # Gemini hat keinen Pflicht-Dev-Key: der Engine-Layer entscheidet
        # zwischen DEVELOPER_GEMINI_API_KEY und Service Account/ADC.
        assert captured["key"] is None
        assert captured["extras"]["key_used"] == "Service Account"
        snapshot = reset_rate_limiter.snapshot(
            uid,
            UsageLimits(
                total=cfg.get_consensus_run_limit(False),
                deep_think=cfg.get_deep_think_run_limit(False),
            ),
        )
        assert snapshot.total.consumed == 1
        assert isinstance(snapshot.total.consumed, int)
    finally:
        pass


def test_own_key_path_bypasses_usage_counting(reset_rate_limiter):
    client = make_client()
    uid = "uid-own-key"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    try:
        p1, p2, p3 = auth_patches(uid=uid)
        with p1, p2, p3, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_claude",
                headers=AUTH_HEADER,
                json={
                    "question": "hello",
                    "model": free_model("anthropic"),
                    "api_key": "sk-user-key",
                },
            )
        assert response.status_code == 200
        assert captured["key"] == "sk-user-key"
        assert captured["extras"]["free_usage_remaining"] == "Unlimited"
        assert captured["extras"]["key_used"] == "User API Key"
        snapshot = reset_rate_limiter.snapshot(
            uid,
            UsageLimits(
                total=cfg.get_consensus_run_limit(False),
                deep_think=cfg.get_deep_think_run_limit(False),
            ),
        )
        assert snapshot.total.consumed == 0
    finally:
        pass


def test_own_key_without_login_is_rejected_for_every_provider():
    client = make_client()
    for route, provider in [
        ("/ask_openai", "openai"),
        ("/ask_mistral", "mistral"),
        ("/ask_gemini", "gemini"),
    ]:
        response = client.post(
            route,
            json={
                "question": "hello",
                "model": free_model(provider),
                "api_key": "sk-user-key",
            },
        )
        assert response.status_code == 401, route
        assert response.json()["detail"] == chat_router.OWN_KEYS_LOGIN_REQUIRED
