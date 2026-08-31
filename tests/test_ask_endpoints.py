"""Endpoint-Tests fuer die deduplizierten /ask_*-Handler (handle_ask).

Nagelt die gemeinsamen OpenRouter-Vertraege fest: Own-Key-Bypass der
Usage-Zaehlung, Auth-Verhalten und die Usage-Limit-Antworten.
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


def auth_patches(uid="uid-ask-tests", is_pro=False):
    return (
        patch.object(chat_router, "verify_user_token", return_value=uid),
        patch.object(chat_router, "is_user_pro", return_value=is_pro),
    )


AUTH_HEADER = {"Authorization": "Bearer test-token"}
PNG_ATTACHMENT = {
    "name": "pixel.png",
    "data": "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",
}


def test_no_auth_error_is_uniform_across_model_families():
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
    assert response.status_code == 400
    assert response.json()["detail"] == "No auth provided."


def test_own_keys_flag_without_openrouter_key_is_rejected():
    client = make_client()
    p1, p2 = auth_patches()
    with p1, p2:
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
    assert response.json()["detail"] == "Missing user OpenRouter API key."


def test_deep_search_is_pro_only():
    client = make_client()
    p1, p2 = auth_patches(is_pro=False)
    with p1, p2:
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


def test_glm_attachment_support_depends_on_the_effective_model():
    client = make_client()
    p1, p2 = auth_patches(is_pro=True)
    with p1, p2, patch.object(chat_router, "_run_ask", return_value={"ok": True}) as run:
        flash = client.post(
            "/ask_glm",
            headers=AUTH_HEADER,
            json={
                "question": "describe it",
                "model": cfg.GLM_BASE_MODEL,
                "useOwnKeys": True,
                "openrouter_key": "sk-user-key",
                "attachments": [PNG_ATTACHMENT],
            },
        )
        pro = client.post(
            "/ask_glm",
            headers=AUTH_HEADER,
            json={
                "question": "describe it",
                "model": cfg.GLM_BASE_MODEL,
                "deep_search": True,
                "useOwnKeys": True,
                "openrouter_key": "sk-user-key",
                "attachments": [PNG_ATTACHMENT],
            },
        )

    assert flash.status_code == 200
    assert pro.status_code == 400
    assert pro.json()["detail"] == "GLM 5.3 cannot read attachments."
    assert run.call_count == 1


def test_megabyte_style_one_word_question_is_rejected_before_provider_work():
    client = make_client()
    p1, p2 = auth_patches()
    with p1, p2, patch.object(chat_router, "_run_ask") as provider_call:
        response = client.post(
            "/ask_openai",
            headers=AUTH_HEADER,
            json={
                "question": "x" * (chat_router.MAX_QUESTION_CHARS + 1),
                "model": free_model("openai"),
                "openrouter_key": "own-key",
            },
        )

    assert response.status_code == 400
    provider_call.assert_not_called()


def test_multibyte_question_and_system_prompt_obey_utf8_byte_caps():
    client = make_client()
    p1, p2 = auth_patches()
    with p1, p2, patch.object(chat_router, "_run_ask") as provider_call:
        question_response = client.post(
            "/ask_openai",
            headers=AUTH_HEADER,
            json={
                "question": "🙂" * 4_001,
                "model": free_model("openai"),
                "openrouter_key": "own-key",
            },
        )
        prompt_response = client.post(
            "/ask_openai",
            headers=AUTH_HEADER,
            json={
                "question": "small",
                "system_prompt": "🙂" * 8_001,
                "model": free_model("openai"),
                "openrouter_key": "own-key",
            },
        )

    assert question_response.status_code == 400
    assert prompt_response.status_code == 400
    provider_call.assert_not_called()


def test_usage_limit_blocks_developer_key_path(reset_rate_limiter):
    client = make_client()
    uid = "uid-limit-reached"
    limits = UsageLimits(total=cfg.get_consensus_run_limit(False), deep_think=0)
    for index in range(limits.total):
        key = f"used-{index}"
        reset_rate_limiter.reserve(uid, key, RunKind.REGULAR, limits)
        reset_rate_limiter.consume(uid, key)
    p1, p2 = auth_patches(uid=uid)
    with p1, p2:
        response = client.post(
            "/ask_deepseek",
            headers=AUTH_HEADER,
            json={"question": "hello", "model": free_model("deepseek")},
        )
    assert response.status_code == 403
    body = response.json()["detail"]
    assert body["error_code"] == "total_usage_limit_exceeded"
    assert body["free_usage_remaining"] == 0


def test_gemini_developer_path_uses_openrouter_and_counts_usage(reset_rate_limiter):
    client = make_client()
    uid = "uid-gemini-dev"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return {"ok": True}

    try:
        p1, p2 = auth_patches(uid=uid)
        with p1, p2, \
             patch.object(chat_router, "resolve_developer_api_keys", return_value={"OpenRouter": "server-key"}), \
             patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_gemini",
                headers=AUTH_HEADER,
                json={"question": "hello", "model": free_model("gemini")},
            )
        assert response.status_code == 200
        assert captured["provider"].label == "Gemini"
        assert captured["key"] == "server-key"
        assert captured["extras"]["key_used"] == "Developer API Key"
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
        p1, p2 = auth_patches(uid=uid)
        with p1, p2, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_claude",
                headers=AUTH_HEADER,
                json={
                    "question": "hello",
                    "model": free_model("anthropic"),
                    "useOwnKeys": True,
                    "openrouter_key": "sk-user-key",
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
                "useOwnKeys": True,
                "openrouter_key": "sk-user-key",
            },
        )
        assert response.status_code == 401, route
        assert response.json()["detail"] == chat_router.OWN_KEYS_LOGIN_REQUIRED
