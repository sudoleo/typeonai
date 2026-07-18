from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from app.api.routers import admin as admin_router
from app.api.routers import api_v1
from app.services import api_consensus_runner
from app.services.usage_repository import RunKind, RunStatus


NOW = datetime.now(timezone.utc)


class KeyRepo:
    def authenticate(self, key):
        if key != "cns_test":
            from app.services.api_key_repository import InvalidApiKey

            raise InvalidApiKey()
        return SimpleNamespace(uid="user-1", key_id="f" * 64, label="test")


class RunRepo:
    def __init__(self):
        self.run = None

    def create_or_get(self, **kwargs):
        if self.run is not None:
            return self.run, False
        self.run = {
            "run_id": "a" * 32,
            "uid": kwargs["uid"],
            "idempotency_hash": "b" * 64,
            "status": "accepted",
            "request": kwargs["request_payload"],
            "accepted_at": NOW,
        }
        return self.run, True

    def get_by_idempotency(self, **kwargs):
        return self.run

    def delete_accepted(self, run_id):
        return False

    def get_for_uid(self, run_id, uid):
        if not self.run or self.run["run_id"] != run_id or self.run["uid"] != uid:
            from app.services.api_run_repository import ApiRunNotFound

            raise ApiRunNotFound()
        return self.run


def setup_api(monkeypatch):
    repo = RunRepo()
    scheduled = []
    monkeypatch.setattr(api_v1, "api_key_repository", KeyRepo())
    monkeypatch.setattr(api_v1, "api_run_repository", repo)
    monkeypatch.setattr(api_v1, "is_user_pro", lambda uid: False)
    monkeypatch.setattr(
        api_v1,
        "build_server_model_plan",
        lambda **kwargs: {"providers": {}, "consensus_model": "OpenAI"},
    )
    monkeypatch.setattr(api_v1, "validate_server_credentials", lambda plan: None)

    def reserve(run):
        run["status"] = "reserved"
        run["reserved_at"] = NOW
        return run, object()

    monkeypatch.setattr(api_v1, "reserve_run", reserve)
    monkeypatch.setattr(api_v1, "schedule_run", scheduled.append)
    return repo, scheduled


def test_openapi_contract_declares_api_key_and_idempotency_header():
    schema = main.app.openapi()
    operation = schema["paths"]["/api/v1/consensus/runs"]["post"]

    assert schema["components"]["securitySchemes"]["ConsensusApiKey"] == {
        "type": "apiKey",
        "description": "User-bound Consensus API key. The plaintext value is shown only when issued.",
        "in": "header",
        "name": "X-API-Key",
    }
    idempotency = next(
        param for param in operation["parameters"] if param["name"] == "Idempotency-Key"
    )
    assert idempotency["required"] is True
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["$ref"].endswith("/ConsensusRunRequest")


def test_admin_dashboard_exposes_safe_api_key_management_section():
    template = open("templates/admin.html", encoding="utf-8").read()

    assert 'data-tab="api"' in template
    assert 'id="tab-api"' in template
    assert 'id="apiKeyUid"' in template
    assert 'id="issuedApiKeyPanel"' in template
    assert 'id="issuedApiKeyValue"' in template
    assert "'/api/admin/api-keys'" in template
    assert "`/api/admin/api-keys/${encodeURIComponent(key.key_id)}`" in template
    assert "input.value = '';" in template
    assert "Only a SHA-256 hash is stored" in template


def test_admin_can_issue_list_and_revoke_api_keys(monkeypatch):
    calls = []

    class AdminKeyRepo:
        def issue(self, uid, *, label, created_by):
            calls.append(("issue", uid, label, created_by))
            return {
                "key_id": "f" * 64,
                "api_key": "cns_live_once",
                "uid": uid,
                "label": label,
                "prefix": "cns_live_once",
                "status": "active",
                "created_at": NOW,
            }

        def list(self, *, uid=None):
            calls.append(("list", uid))
            return [{"key_id": "f" * 64, "uid": uid, "status": "active"}]

        def revoke(self, key_id):
            calls.append(("revoke", key_id))
            return {"status": "revoked"}

    monkeypatch.setattr(admin_router, "_require_admin", lambda request, data: "admin-1")
    monkeypatch.setattr(admin_router.auth, "get_user", lambda uid: SimpleNamespace(uid=uid))
    monkeypatch.setattr(admin_router, "api_key_repository", AdminKeyRepo())
    client = TestClient(main.app)

    issued = client.post(
        "/api/admin/api-keys", json={"uid": "user-1", "label": "Production"}
    )
    listed = client.get("/api/admin/api-keys?uid=user-1")
    revoked = client.delete("/api/admin/api-keys/" + "f" * 64)

    assert issued.status_code == 201
    assert issued.json()["api_key"] == "cns_live_once"
    assert listed.status_code == 200
    assert listed.json()["keys"][0]["uid"] == "user-1"
    assert revoked.json() == {"key_id": "f" * 64, "status": "revoked"}
    assert calls == [
        ("issue", "user-1", "Production", "admin-1"),
        ("list", "user-1"),
        ("revoke", "f" * 64),
    ]


def test_post_returns_accepted_run_and_duplicate_run_id(monkeypatch):
    _repo, scheduled = setup_api(monkeypatch)
    client = TestClient(main.app)
    headers = {"X-API-Key": "cns_test", "Idempotency-Key": "same"}

    first = client.post(
        "/api/v1/consensus/runs", headers=headers, json={"question": "Why?"}
    )
    second = client.post(
        "/api/v1/consensus/runs", headers=headers, json={"question": "Why?"}
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"] == "a" * 32
    assert first.json()["status"] == "reserved"
    assert first.headers["location"].endswith("/" + "a" * 32)
    assert scheduled == ["a" * 32, "a" * 32]


def test_request_rejects_client_model_and_limit_fields(monkeypatch):
    setup_api(monkeypatch)
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/consensus/runs",
        headers={"X-API-Key": "cns_test", "Idempotency-Key": "new"},
        json={"question": "Why?", "model_count": 2, "cost": 1},
    )
    assert response.status_code == 422


def test_deep_think_is_uid_tier_gated(monkeypatch):
    setup_api(monkeypatch)
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/consensus/runs",
        headers={"X-API-Key": "cns_test", "Idempotency-Key": "deep"},
        json={"question": "Why?", "deep_think": True},
    )
    assert response.status_code == 403


def test_runner_claim_prevents_duplicate_usage_and_provider_start(monkeypatch):
    class ClaimRepo:
        def __init__(self):
            self.claimed = False
            self.succeeded = 0

        def claim_running(self, run_id, worker_id):
            if self.claimed:
                return {"run_id": run_id, "status": "running"}, False
            self.claimed = True
            return {
                "run_id": run_id,
                "uid": "user-1",
                "idempotency_hash": "b" * 64,
                "status": "running",
            }, True

        def succeed(self, run_id, result):
            self.succeeded += 1

        def get(self, run_id):
            return {"status": "succeeded"}

    class UsageRepo:
        def __init__(self):
            self.consumed = 0

        def consume(self, uid, key):
            self.consumed += 1

    run_repo = ClaimRepo()
    usage_repo = UsageRepo()
    provider_starts = []
    monkeypatch.setattr(api_consensus_runner, "api_run_repository", run_repo)
    monkeypatch.setattr(api_consensus_runner, "usage_repository", usage_repo)
    monkeypatch.setattr(
        api_consensus_runner,
        "execute_consensus_pipeline",
        lambda run: provider_starts.append(run["run_id"]) or {"consensus_response": "ok"},
    )

    api_consensus_runner.execute_persisted_run("a" * 32)
    api_consensus_runner.execute_persisted_run("a" * 32)

    assert usage_repo.consumed == 1
    assert provider_starts == ["a" * 32]
    assert run_repo.succeeded == 1


def test_deep_think_reservation_uses_separate_usage_kind(monkeypatch):
    captured = {}

    class UsageRepo:
        def reserve(self, uid, key, kind, limits):
            captured.update(uid=uid, key=key, kind=kind, limits=limits)
            return SimpleNamespace(status=RunStatus.RESERVED)

    class RunRepo:
        def mark_reserved(self, run_id):
            return {"run_id": run_id, "status": "reserved"}, True

    monkeypatch.setattr(api_consensus_runner, "usage_repository", UsageRepo())
    monkeypatch.setattr(api_consensus_runner, "api_run_repository", RunRepo())
    run = {
        "run_id": "a" * 32,
        "uid": "user-1",
        "idempotency_hash": "b" * 64,
        "is_pro_at_acceptance": True,
        "request": {"deep_think": True},
    }

    api_consensus_runner.reserve_run(run)

    assert captured["kind"] is RunKind.DEEP_THINK
    assert captured["key"] == "consensus-api:" + "b" * 64
