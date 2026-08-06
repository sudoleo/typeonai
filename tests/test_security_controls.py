from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import chat as chat_router
from app.api.routers import pages as pages_router
from app.core.rate_limit import limiter


def make_client(router):
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    return TestClient(app)


def test_check_keys_requires_verified_login():
    client = make_client(pages_router.router)

    response = client.post("/check_keys", json={"openai_key": "sk-test-value"})

    assert response.status_code == 401


def test_check_keys_requires_at_least_one_key_after_auth():
    client = make_client(pages_router.router)

    with patch.object(pages_router, "verify_user_token", return_value="uid-1"):
        response = client.post(
            "/check_keys",
            headers={"Authorization": "Bearer token"},
            json={},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Enter at least one API key to test."


def test_user_api_key_requests_require_login():
    client = make_client(chat_router.router)

    response = client.post(
        "/ask_openai",
        json={
            "question": "hello",
            "api_key": "sk-user-key",
            "model": "gpt-5.4-mini",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == chat_router.OWN_KEYS_LOGIN_REQUIRED


def test_validation_errors_stay_422_and_never_echo_the_submitted_value():
    """Der globale Handler in main.py muss ValueError-Validatoren ueberleben.

    Fast jeder Validator in diesem Projekt meldet ungueltige Eingaben als
    ValueError. Pydantic v2 haengt dieses ROHE Exception-Objekt in errors()
    unter "ctx" -- der Handler serialisierte das direkt und lief in einen
    TypeError, sodass die Antwort ein 500er statt eines 422ers wurde. Die
    Unit-Tests der einzelnen Router sehen das nicht: sie bauen eigene
    FastAPI-Apps ohne diesen Handler.
    """
    import main

    client = TestClient(main.app)
    response = client.post(
        "/chats/" + "a" * 32 + "/turns",
        json={
            "question": "Test",
            "mode": "x" * 41,  # ValueError aus normalize_mode
            "deep_search": False,
            "selected_models": ["OpenAI"],
            "consensus_model": "OpenAI",
        },
        headers={"Authorization": "Bearer not-a-real-token"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "Validation failed"
    assert payload["details"], "der Grund der Ablehnung muss erkennbar bleiben"
    for entry in payload["details"]:
        assert set(entry) == {"loc", "type", "msg"}
        assert all(isinstance(part, str) for part in entry["loc"])
    # Weder das rohe Exception-Objekt noch der eingesendete Wert gehen zurueck.
    body = response.text
    assert "ctx" not in body
    assert "x" * 41 not in body
