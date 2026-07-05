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
from app.core.state import usage_counter, deep_search_usage


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    # Die /ask_*-Routen sind mit 3-5/minute limitiert; mehrere Tests teilen
    # sich denselben In-Memory-Limiter (Key: Test-Client-IP).
    limiter.reset()
    yield


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
        json={"question": "hello", "model": free_model("mistral"), "active_count": 1},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "No auth provided."

    response = client.post(
        "/ask_gemini",
        json={"question": "hello", "model": free_model("gemini"), "active_count": 1},
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
                "active_count": 1,
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
                "active_count": 1,
                "deep_search": "true",
            },
        )
    assert response.status_code == 403
    assert "Pro users" in response.json()["detail"]


def test_usage_limit_blocks_developer_key_path():
    client = make_client()
    uid = "uid-limit-reached"
    usage_counter[uid] = cfg.get_usage_limit(False)
    try:
        p1, p2, p3 = auth_patches(uid=uid)
        with p1, p2, p3:
            response = client.post(
                "/ask_deepseek",
                headers=AUTH_HEADER,
                json={"question": "hello", "model": free_model("deepseek"), "active_count": 1},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["error"] == "Free usage limit reached. Upgrade to Pro."
        assert body["free_usage_remaining"] == 0
        # Der abgelehnte Request darf nichts hochzaehlen.
        assert usage_counter[uid] == cfg.get_usage_limit(False)
    finally:
        usage_counter.pop(uid, None)


def test_gemini_developer_path_uses_service_account_and_counts_usage():
    client = make_client()
    uid = "uid-gemini-dev"
    usage_counter.pop(uid, None)
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
                json={"question": "hello", "model": free_model("gemini"), "active_count": 1},
            )
        assert response.status_code == 200
        assert captured["provider"].label == "Gemini"
        # Gemini hat keinen Pflicht-Dev-Key: der Engine-Layer entscheidet
        # zwischen DEVELOPER_GEMINI_API_KEY und Service Account/ADC.
        assert captured["key"] is None
        assert captured["extras"]["key_used"] == "Service Account"
        assert usage_counter[uid] == 1.0
    finally:
        usage_counter.pop(uid, None)
        deep_search_usage.pop(uid, None)


def test_own_key_path_bypasses_usage_counting():
    client = make_client()
    uid = "uid-own-key"
    usage_counter.pop(uid, None)
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
                    "active_count": 1,
                    "api_key": "sk-user-key",
                },
            )
        assert response.status_code == 200
        assert captured["key"] == "sk-user-key"
        assert captured["extras"]["free_usage_remaining"] == "Unlimited"
        assert captured["extras"]["key_used"] == "User API Key"
        assert uid not in usage_counter
    finally:
        usage_counter.pop(uid, None)


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
                "active_count": 1,
                "api_key": "sk-user-key",
            },
        )
        assert response.status_code == 401, route
        assert response.json()["detail"] == chat_router.OWN_KEYS_LOGIN_REQUIRED
