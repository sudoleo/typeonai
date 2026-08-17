from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import client_errors
from app.core.rate_limit import limiter
from tests.frontend_order import group_of, loads_before


def _client(monkeypatch, captured):
    monkeypatch.setattr(
        client_errors,
        "send_critical_error_notification",
        lambda report: captured.append(report) or {"status": "sent"},
    )
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(client_errors.router)
    return TestClient(app)


def test_client_error_report_is_accepted_and_sanitized(monkeypatch):
    captured = []
    client = _client(monkeypatch, captured)

    response = client.post(
        "/api/client-errors",
        headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
        json={
            "type": "run_failed",
            "phase": "model_fanout",
            "message": "private.prompt@example.test token=browser-secret",
            "details": "OpenAI body: private provider response",
            "stack": "stack containing browser-secret",
            "path": "/s/private-share-id",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    assert captured == [{
        "source": "browser",
        "type": "run_failed",
        "phase": "model_fanout",
        "message": "A browser run failed.",
        "path": "/s/{share_id}",
    }]


def test_client_error_report_replaces_unknown_phase_and_route(monkeypatch):
    captured = []
    client = _client(monkeypatch, captured)

    response = client.post(
        "/api/client-errors",
        headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
        json={
            "type": "unhandled_error",
            "phase": "secret-phase@example.test",
            "message": "private runtime text",
            "path": "/account/private-user-id",
        },
    )

    assert response.status_code == 202
    assert captured == [{
        "source": "browser",
        "type": "unhandled_error",
        "phase": "browser",
        "message": "An unhandled browser error occurred.",
        "path": "/other",
    }]


def test_client_error_report_rejects_cross_origin(monkeypatch):
    captured = []
    client = _client(monkeypatch, captured)

    response = client.post(
        "/api/client-errors",
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
        json={"type": "unhandled_error", "message": "boom", "path": "/app"},
    )

    assert response.status_code == 403
    assert captured == []


def test_client_error_report_rejects_foreign_origin_without_fetch_metadata(
    monkeypatch,
):
    captured = []
    client = _client(monkeypatch, captured)

    response = client.post(
        "/api/client-errors",
        headers={"Origin": "https://attacker.example"},
        json={"type": "unhandled_error", "message": "boom", "path": "/app"},
    )

    assert response.status_code == 403
    assert captured == []


def test_error_reporter_loads_before_app_modules():
    # Must be in place before anything can throw: the head group is
    # render-blocking, the app group is deferred.
    assert group_of("error-reporter.js") == "head"
    assert loads_before("error-reporter.js", "app-core.js")
