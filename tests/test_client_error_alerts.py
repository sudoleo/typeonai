from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers import client_errors
from app.core.rate_limit import limiter


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
            "message": "All selected models failed.",
            "details": "OpenAI: connection lost",
            "stack": "stack",
            "path": "/app",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    assert captured == [{
        "source": "browser",
        "type": "run_failed",
        "phase": "model_fanout",
        "message": "All selected models failed.",
        "details": "OpenAI: connection lost",
        "stack": "stack",
        "path": "/app",
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


def test_error_reporter_loads_before_app_modules():
    html = open("templates/index.html", encoding="utf-8").read()
    assert "error-reporter.js?v=20260731-criticalalerts1" in html
    assert html.index("error-reporter.js") < html.index("app-core.js")
