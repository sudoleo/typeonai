from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from app.api.routers import admin as admin_router
from app.api.routers import api_v1
from app.services import api_consensus_runner
from app.services.usage_repository import RunKind, RunStatus
from app.services.usage_repository import UsageTransitionError


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

    def delete_terminal_for_uid(self, run_id, uid):
        if not self.run or self.run["run_id"] != run_id or self.run["uid"] != uid:
            from app.services.api_run_repository import ApiRunNotFound

            raise ApiRunNotFound()
        if self.run["status"] not in {"succeeded", "failed"}:
            from app.services.api_run_repository import ApiRunTransitionError

            raise ApiRunTransitionError("Only terminal runs can be deleted")
        self.run = None
        return True


def setup_api(monkeypatch):
    repo = RunRepo()
    scheduled = []
    monkeypatch.setattr(api_v1, "api_key_repository", KeyRepo())
    monkeypatch.setattr(api_v1, "ensure_api_account_active", lambda uid: None)
    monkeypatch.setattr(api_v1, "enforce_uid_rate_limit", lambda *args: None)
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
    publisher_header = next(
        param for param in operation["parameters"] if param["name"] == "X-Consensus-Publisher"
    )
    assert publisher_header["required"] is False
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["$ref"].endswith("/ConsensusRunRequest")
    run_path = schema["paths"]["/api/v1/consensus/runs/{run_id}"]
    assert "get" in run_path and "delete" in run_path
    assert "post" in schema["paths"]["/api/v1/consensus/runs/{run_id}/share"]
    assert "get" in schema["paths"]["/api/v1/shares"]
    assert "get" in schema["paths"]["/api/v1/publisher/config"]
    share_path = schema["paths"]["/api/v1/shares/{share_id}"]
    assert "get" in share_path and "delete" in share_path
    assert "post" in schema["paths"]["/api/v1/shares/{share_id}/watch"]
    assert "put" in schema["paths"]["/api/v1/shares/{share_id}/indexing"]
    response_properties = schema["components"]["schemas"]["ConsensusRunResponse"][
        "properties"
    ]
    assert "expires_at" in response_properties


def test_admin_dashboard_exposes_safe_api_key_management_section():
    template = open("templates/admin.html", encoding="utf-8").read()

    assert 'data-tab="api"' in template
    assert 'id="tab-api"' in template
    assert 'id="apiKeyUid"' in template
    assert 'id="issuedApiKeyPanel"' in template
    assert 'id="issuedApiKeyValue"' in template
    assert 'id="apiKeyDirectIndex"' in template
    assert "'/api/admin/api-keys'" in template
    assert "`/api/admin/api-keys/${encodeURIComponent(key.key_id)}`" in template
    assert "input.value = '';" in template
    assert "Only a SHA-256 hash is stored" in template
    assert 'id="publisherEnabled"' in template
    assert 'id="publisherTopicBrief"' in template
    assert "Free Watch providers" in template
    assert "DeepSeek is excluded from both" in template
    assert "'/api/admin/publisher-config'" in template


def test_admin_can_load_and_save_publisher_configuration(monkeypatch):
    saved = []
    config = {
        "enabled": True,
        "topic_brief": "Choose a useful evidence-rich topic.",
        "auto_index": True,
        "weekly_watch_enabled": True,
        "watch_weekday": "tuesday",
        "watch_time": "09:00",
        "watch_timezone": "Europe/Berlin",
    }
    monkeypatch.setattr(admin_router, "_require_admin", lambda request, data: "admin-1")
    monkeypatch.setattr(admin_router.publisher_config, "get_config", lambda: dict(config))
    monkeypatch.setattr(
        admin_router.publisher_config,
        "save_config",
        lambda data, *, updated_by: saved.append((data, updated_by)) or dict(data),
    )
    client = TestClient(main.app)

    loaded = client.get("/api/admin/publisher-config")
    updated = client.put("/api/admin/publisher-config", json={**config, "enabled": False})

    assert loaded.status_code == 200
    assert loaded.json()["config"]["watch_model_tier"] == "free"
    assert updated.status_code == 200
    assert updated.json()["config"]["enabled"] is False
    assert saved[0][1] == "admin-1"


def test_admin_can_issue_list_and_revoke_api_keys(monkeypatch):
    calls = []

    class AdminKeyRepo:
        def issue(self, uid, *, label, created_by, scopes):
            calls.append(("issue", uid, label, created_by, scopes))
            return {
                "key_id": "f" * 64,
                "api_key": "cns_live_once",
                "uid": uid,
                "label": label,
                "prefix": "cns_live_once",
                "status": "active",
                "scopes": list(scopes),
                "created_at": NOW,
            }

        def list(self, *, uid=None):
            calls.append(("list", uid))
            return [{"key_id": "f" * 64, "uid": uid, "status": "active"}]

        def revoke(self, key_id):
            calls.append(("revoke", key_id))
            return {"status": "revoked"}

    monkeypatch.setattr(admin_router, "_require_admin", lambda request, data: "admin-1")
    monkeypatch.setattr(
        admin_router.auth,
        "get_user",
        lambda uid: SimpleNamespace(
            uid=uid, disabled=False, email_verified=True
        ),
    )
    monkeypatch.setattr(
        admin_router,
        "api_account_cleanup",
        SimpleNamespace(is_blocked=lambda uid: False),
    )
    monkeypatch.setattr(admin_router, "api_key_repository", AdminKeyRepo())
    client = TestClient(main.app)

    issued = client.post(
        "/api/admin/api-keys", json={"uid": "user-1", "label": "Production"}
    )
    listed = client.get("/api/admin/api-keys?uid=user-1")
    revoked = client.delete("/api/admin/api-keys/" + "f" * 64)

    assert issued.status_code == 201
    assert issued.json()["api_key"] == "cns_live_once"
    assert issued.headers["cache-control"] == "private, no-store"
    assert listed.status_code == 200
    assert listed.json()["keys"][0]["uid"] == "user-1"
    assert revoked.json() == {"key_id": "f" * 64, "status": "revoked"}
    assert calls == [
        ("issue", "user-1", "Production", "admin-1", ["consensus:run", "share:write"]),
        ("list", "user-1"),
        ("revoke", "f" * 64),
    ]


def test_admin_cannot_issue_direct_index_scope_to_non_admin_uid(monkeypatch):
    monkeypatch.setattr(admin_router, "_require_admin", lambda request, data: "admin-1")
    monkeypatch.setattr(
        admin_router.auth,
        "get_user",
        lambda uid: SimpleNamespace(uid=uid, disabled=False, email_verified=True),
    )
    monkeypatch.setattr(
        admin_router,
        "api_account_cleanup",
        SimpleNamespace(is_blocked=lambda uid: False),
    )
    monkeypatch.setattr(admin_router, "is_user_admin", lambda uid: False)
    client = TestClient(main.app)

    response = client.post(
        "/api/admin/api-keys",
        json={
            "uid": "regular-user",
            "label": "unsafe",
            "scopes": ["consensus:run", "share:write", "share:index"],
        },
    )

    assert response.status_code == 409
    assert "admin UID" in response.text


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
    assert first.headers["cache-control"] == "private, no-store"
    assert first.headers["pragma"] == "no-cache"
    assert scheduled == ["a" * 32, "a" * 32]


def test_admin_publisher_run_excludes_deepseek_from_persisted_plan(monkeypatch):
    repo, _scheduled = setup_api(monkeypatch)
    captured = {}
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: True)

    def build_plan(**kwargs):
        captured.update(kwargs)
        return {
            "providers": {"openai": "openai-model", "gemini": "gemini-model"},
            "consensus_model": "OpenAI",
            "excluded_providers": ["deepseek"],
        }

    monkeypatch.setattr(api_v1, "build_server_model_plan", build_plan)
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/consensus/runs",
        headers={
            "X-API-Key": "cns_test",
            "Idempotency-Key": "publisher-no-deepseek",
            "X-Consensus-Publisher": "true",
        },
        json={"question": "Which evidence should be compared?"},
    )

    assert response.status_code == 202
    assert captured["excluded_providers"] == ("deepseek",)
    assert repo.run["request"]["publisher_mode"] is True


def test_publisher_mode_requires_admin(monkeypatch):
    setup_api(monkeypatch)
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: False)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/consensus/runs",
        headers={
            "X-API-Key": "cns_test",
            "Idempotency-Key": "publisher-denied",
            "X-Consensus-Publisher": "true",
        },
        json={"question": "Which evidence should be compared?"},
    )

    assert response.status_code == 403


def test_publisher_model_plan_has_no_deepseek_provider_or_engine():
    plan = api_consensus_runner.build_server_model_plan(
        deep_think=False, is_pro=True, excluded_providers=("deepseek",)
    )

    assert "deepseek" not in plan["providers"]
    assert plan["excluded_providers"] == ["deepseek"]
    assert api_consensus_runner._consensus_provider_label(plan["consensus_model"]) != "DeepSeek"


def test_terminal_run_can_be_deleted_early(monkeypatch):
    repo, _scheduled = setup_api(monkeypatch)
    repo.run = {
        "run_id": "a" * 32,
        "uid": "user-1",
        "status": "succeeded",
        "request": {"question": "Why?", "deep_think": False},
        "accepted_at": NOW,
        "result": {"consensus_response": "ok"},
    }
    client = TestClient(main.app)

    response = client.delete(
        "/api/v1/consensus/runs/" + "a" * 32,
        headers={"X-API-Key": "cns_test"},
    )

    assert response.status_code == 204
    assert response.headers["cache-control"] == "private, no-store"
    assert repo.run is None


def test_disabled_account_cannot_use_still_active_key(monkeypatch):
    monkeypatch.setattr(api_v1, "api_key_repository", KeyRepo())
    from app.services.api_account_cleanup import ApiAccountInactive

    def reject(_uid):
        raise ApiAccountInactive("disabled")

    monkeypatch.setattr(
        api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=reject)
    )
    client = TestClient(main.app)

    response = client.get(
        "/api/v1/consensus/runs/" + "a" * 32,
        headers={"X-API-Key": "cns_test"},
    )

    assert response.status_code == 401
    assert response.json() == {"error": "Invalid API key"}


def test_locally_blocked_account_fails_before_firebase_lookup(monkeypatch):
    monkeypatch.setattr(api_v1, "api_key_repository", KeyRepo())
    from app.services.api_account_cleanup import ApiAccountInactive

    def reject(_uid):
        raise ApiAccountInactive("blocked")

    monkeypatch.setattr(
        api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=reject)
    )
    client = TestClient(main.app)

    response = client.get(
        "/api/v1/consensus/runs/" + "a" * 32,
        headers={"X-API-Key": "cns_test"},
    )

    assert response.status_code == 401


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


def test_post_is_rate_limited_per_api_key(monkeypatch):
    setup_api(monkeypatch)
    unique_key = "cns_live_rate-limit-test-key"
    authentications = []

    class RateKeyRepo:
        def authenticate(self, key):
            if key != unique_key:
                raise AssertionError("unexpected key")
            authentications.append(key)
            return SimpleNamespace(uid="rate-user", key_id="e" * 64, label="rate")

    monkeypatch.setattr(api_v1, "api_key_repository", RateKeyRepo())
    client = TestClient(main.app)
    headers = {"X-API-Key": unique_key, "Idempotency-Key": "same"}

    responses = [
        client.post("/api/v1/consensus/runs", headers=headers, json={"question": "Why?"})
        for _ in range(11)
    ]

    assert [response.status_code for response in responses[:10]] == [202] * 10
    assert responses[10].status_code == 429
    assert len(authentications) == 10


def test_random_invalid_keys_are_ip_limited_before_firestore_auth(monkeypatch):
    setup_api(monkeypatch)
    authentications = []

    class InvalidKeyRepo:
        def authenticate(self, key):
            from app.services.api_key_repository import InvalidApiKey

            authentications.append(key)
            raise InvalidApiKey()

    monkeypatch.setattr(api_v1, "api_key_repository", InvalidKeyRepo())
    client = TestClient(main.app)
    responses = []
    for index in range(31):
        responses.append(
            client.post(
                "/api/v1/consensus/runs",
                headers={
                    "X-Forwarded-For": "198.51.100.199",
                    "X-API-Key": f"cns_live_invalid-{index}",
                    "Idempotency-Key": f"invalid-{index}",
                },
                json={"question": "Why?"},
            )
        )

    assert [response.status_code for response in responses[:30]] == [401] * 30
    assert responses[30].status_code == 429
    assert len(authentications) == 30


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
        "api_account_cleanup",
        SimpleNamespace(ensure_active=lambda uid: None),
    )
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


def test_publisher_pipeline_removes_deepseek_provider_and_judge_key(monkeypatch):
    provider_calls = []
    consensus_keys = []
    differences_keys = []
    run = {
        "request": {
            "question": "Do data centers raise electricity prices?",
            "deep_think": False,
            "publisher_mode": True,
        },
        "is_pro_at_acceptance": True,
        "model_plan": {
            "providers": {
                "openai": "openai-model",
                "mistral": "mistral-model",
                "deepseek": "deepseek-model",
            },
            "consensus_model": "OpenAI",
        },
    }
    monkeypatch.setattr(
        api_consensus_runner,
        "resolve_developer_api_keys",
        lambda: {"OpenAI": "o", "Mistral": "m", "DeepSeek": "d"},
    )
    monkeypatch.setattr(api_consensus_runner, "mock_llm_enabled", lambda: True)
    monkeypatch.setattr(
        api_consensus_runner,
        "_provider_answer",
        lambda provider, *args: provider_calls.append(provider) or f"{provider} answer",
    )
    monkeypatch.setattr(api_consensus_runner, "result_text", lambda value: value)
    monkeypatch.setattr(api_consensus_runner, "result_sources", lambda value: [])

    def consensus(*args, **kwargs):
        consensus_keys.append(args[9])
        return "Consensus"

    def differences(*args, **kwargs):
        differences_keys.append(args[7])
        return "Differences", {"claims": []}

    monkeypatch.setattr(api_consensus_runner, "query_consensus", consensus)
    monkeypatch.setattr(api_consensus_runner, "query_differences", differences)
    monkeypatch.setattr(api_consensus_runner, "compute_agreement_score", lambda data: {"score": 75})

    result = api_consensus_runner.execute_consensus_pipeline(run)

    assert provider_calls == ["openai", "mistral"]
    assert consensus_keys[0]["DeepSeek"] is None
    assert differences_keys[0]["DeepSeek"] is None
    assert [item["provider"] for item in result["model_answers"]] == ["OpenAI", "Mistral"]


def test_queued_run_rechecks_account_before_provider_start(monkeypatch):
    from app.services.api_account_cleanup import ApiAccountInactive

    failures = []
    releases = []
    provider_starts = []
    run = {
        "run_id": "a" * 32,
        "uid": "user-1",
        "idempotency_hash": "b" * 64,
        "status": "running",
    }

    class RunRepo:
        def claim_running(self, run_id, worker_id):
            return run, True

        def fail(self, run_id, *, code, message):
            failures.append((code, message))

    class UsageRepo:
        def release(self, uid, key):
            releases.append((uid, key))

        def consume(self, uid, key):
            raise AssertionError("inactive account must not consume usage")

    def reject(_uid):
        raise ApiAccountInactive("blocked")

    monkeypatch.setattr(api_consensus_runner, "api_run_repository", RunRepo())
    monkeypatch.setattr(api_consensus_runner, "usage_repository", UsageRepo())
    monkeypatch.setattr(
        api_consensus_runner,
        "api_account_cleanup",
        SimpleNamespace(ensure_active=reject),
    )
    monkeypatch.setattr(
        api_consensus_runner,
        "execute_consensus_pipeline",
        lambda persisted: provider_starts.append(persisted["run_id"]),
    )

    api_consensus_runner.execute_persisted_run(run["run_id"])

    assert releases == [("user-1", "consensus-api:" + "b" * 64)]
    assert failures[0][0] == "account_inactive"
    assert provider_starts == []


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


def test_scheduler_deduplicates_and_bounds_pending_work(monkeypatch):
    submitted = []

    class Executor:
        def submit(self, fn, run_id):
            submitted.append((fn, run_id))

    monkeypatch.setattr(api_consensus_runner, "_background_executor", Executor())
    monkeypatch.setattr(api_consensus_runner, "MAX_SCHEDULED_RUNS", 1)
    with api_consensus_runner._schedule_lock:
        api_consensus_runner._scheduled_run_ids.clear()

    try:
        assert api_consensus_runner.schedule_run("a" * 32) is True
        assert api_consensus_runner.schedule_run("a" * 32) is True
        assert api_consensus_runner.schedule_run("b" * 32) is False
        assert [item[1] for item in submitted] == ["a" * 32]
    finally:
        with api_consensus_runner._schedule_lock:
            api_consensus_runner._scheduled_run_ids.clear()


def test_expired_pre_provider_worker_releases_reserved_usage(monkeypatch):
    run = {
        "run_id": "a" * 32,
        "uid": "user-1",
        "idempotency_hash": "b" * 64,
        "status": "running",
    }
    releases = []

    class RunRepo:
        def get(self, run_id):
            return run

        def fail_if_lease_expired(self, run_id):
            return True

    class UsageRepo:
        def release(self, uid, key):
            releases.append((uid, key))

    monkeypatch.setattr(api_consensus_runner, "api_run_repository", RunRepo())
    monkeypatch.setattr(api_consensus_runner, "usage_repository", UsageRepo())

    assert api_consensus_runner.fail_expired_run(run["run_id"]) is True
    assert releases == [("user-1", "consensus-api:" + "b" * 64)]


def test_expired_post_provider_worker_keeps_consumed_usage(monkeypatch):
    run = {
        "run_id": "a" * 32,
        "uid": "user-1",
        "idempotency_hash": "b" * 64,
        "status": "running",
    }

    class RunRepo:
        def get(self, run_id):
            return run

        def fail_if_lease_expired(self, run_id):
            return True

    class UsageRepo:
        def release(self, uid, key):
            raise UsageTransitionError("consumed is terminal")

    monkeypatch.setattr(api_consensus_runner, "api_run_repository", RunRepo())
    monkeypatch.setattr(api_consensus_runner, "usage_repository", UsageRepo())

    assert api_consensus_runner.fail_expired_run(run["run_id"]) is True


def test_api_key_can_publish_list_read_and_revoke_own_share(monkeypatch):
    share_id = "A" * 16
    run_id = "a" * 32
    share_doc = {
        "owner_uid": "user-publisher",
        "slug": "evidence-question",
        "question": "Which evidence should be compared?",
        "status": "active",
        "visibility": "public",
        "index_eligible": True,
        "indexed": False,
    }
    publication_calls = []
    revoked = []

    class PublishKeyRepo:
        def authenticate(self, key):
            assert key == "cns_publish_test"
            return SimpleNamespace(
                uid="user-publisher",
                key_id="f" * 64,
                scopes=("consensus:run", "share:write"),
            )

    class PublishRunRepo:
        def get_for_uid(self, requested_run_id, uid):
            assert (requested_run_id, uid) == (run_id, "user-publisher")
            return {"run_id": run_id, "uid": uid, "status": "succeeded"}

    monkeypatch.setattr(api_v1, "api_key_repository", PublishKeyRepo())
    monkeypatch.setattr(
        api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=lambda uid: None)
    )
    monkeypatch.setattr(api_v1, "api_run_repository", PublishRunRepo())
    monkeypatch.setattr(
        api_v1.share_snapshots,
        "create_share_from_api_run",
        lambda uid, run: publication_calls.append((uid, run["run_id"])) or {
            "share_id": share_id,
            "slug": share_doc["slug"],
            "created": True,
        },
    )
    monkeypatch.setattr(
        api_v1.share_snapshots,
        "get_share",
        lambda requested: dict(share_doc) if requested == share_id else None,
    )
    monkeypatch.setattr(
        api_v1.share_snapshots,
        "list_shares_for_owner",
        lambda uid, max_items: [{"share_id": share_id}],
    )
    monkeypatch.setattr(
        api_v1.share_snapshots,
        "revoke_share",
        lambda requested, uid: revoked.append((requested, uid)),
    )
    client = TestClient(main.app)
    headers = {"X-API-Key": "cns_publish_test"}

    published = client.post(f"/api/v1/consensus/runs/{run_id}/share", headers=headers)
    fetched = client.get(f"/api/v1/shares/{share_id}", headers=headers)
    listed = client.get("/api/v1/shares?limit=10", headers=headers)
    deleted = client.delete(f"/api/v1/shares/{share_id}", headers=headers)

    assert published.status_code == 201
    assert published.json()["url"].endswith(f"/s/evidence-question-{share_id}")
    assert published.json()["indexing_status"] == "noindex"
    assert fetched.status_code == 200
    assert listed.json()["shares"][0]["share_id"] == share_id
    assert deleted.status_code == 204
    assert publication_calls == [("user-publisher", run_id)]
    assert revoked == [(share_id, "user-publisher")]


def test_admin_api_configures_weekly_watch_with_free_provider_tier(monkeypatch):
    identity = SimpleNamespace(
        uid="admin-publisher",
        key_id="e" * 64,
        scopes=("share:write",),
    )
    captured = {}

    class PublisherKeyRepo:
        def authenticate(self, key):
            return identity

    config = {
        "enabled": True,
        "topic_brief": "Choose a useful topic.",
        "auto_index": True,
        "weekly_watch_enabled": True,
        "watch_weekday": "wednesday",
        "watch_time": "08:30",
        "watch_timezone": "Europe/Berlin",
    }

    def create_watch(uid, **kwargs):
        captured.update(uid=uid, **kwargs)
        return {
            "id": "watch-1",
            "share_id": "C" * 16,
            "status": "active",
            "interval": kwargs["interval"],
            "model_tier": kwargs["model_tier"],
        }

    monkeypatch.setattr(api_v1, "api_key_repository", PublisherKeyRepo())
    monkeypatch.setattr(
        api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=lambda uid: None)
    )
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: True)
    monkeypatch.setattr(api_v1, "is_user_pro", lambda uid: True)
    monkeypatch.setattr(api_v1.publisher_config, "get_config", lambda: dict(config))
    monkeypatch.setattr(api_v1.watch_service, "create_watch", create_watch)
    client = TestClient(main.app)
    headers = {"X-API-Key": "cns_publisher"}

    loaded = client.get("/api/v1/publisher/config", headers=headers)
    watched = client.post(f"/api/v1/shares/{'C' * 16}/watch", headers=headers)

    assert loaded.status_code == 200
    assert loaded.json()["watch_interval"] == "weekly"
    assert loaded.json()["watch_model_tier"] == "free"
    assert loaded.json()["excluded_providers"] == ["deepseek"]
    assert watched.status_code == 200
    assert watched.json()["watch"]["model_tier"] == "free"
    assert captured["interval"] == "weekly"
    assert captured["model_tier"] == "free"
    assert captured["return_existing"] is True
    assert captured["bypass_active_limit"] is True
    assert captured["excluded_providers"] == ("deepseek",)
    assert captured["run_weekday"] == "wednesday"


def test_publisher_watch_capacity_returns_successful_skip(monkeypatch):
    identity = SimpleNamespace(uid="admin-publisher", key_id="e" * 64, scopes=("share:write",))
    monkeypatch.setattr(
        api_v1, "api_key_repository",
        SimpleNamespace(authenticate=lambda key: identity),
    )
    monkeypatch.setattr(api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=lambda uid: None))
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: True)
    monkeypatch.setattr(api_v1.publisher_config, "get_config", lambda: {
        **api_v1.publisher_config.DEFAULT_CONFIG,
        "max_active_publisher_watches": 12,
    })
    monkeypatch.setattr(api_v1.watch_service, "publisher_watch_counts", lambda: {"active": 12, "paused": 3})
    monkeypatch.setattr(api_v1.watch_service, "find_watch_for_share", lambda share_id: None)
    monkeypatch.setattr(
        api_v1.watch_service, "create_watch",
        lambda *args, **kwargs: pytest.fail("capacity skip must not create a Watch"),
    )
    response = TestClient(main.app).post(
        f"/api/v1/shares/{'C' * 16}/watch", headers={"X-API-Key": "cns_publisher"}
    )
    assert response.status_code == 200
    assert response.json()["watch_status"] == "watch_skipped_capacity"
    assert response.json()["watch"] is None


def test_direct_indexing_requires_scope_admin_and_returns_indexed_state(monkeypatch):
    share_id = "B" * 16
    identity = SimpleNamespace(
        uid="admin-publisher",
        key_id="e" * 64,
        scopes=("consensus:run", "share:write", "share:index"),
    )

    class IndexKeyRepo:
        def authenticate(self, key):
            return identity

    monkeypatch.setattr(api_v1, "api_key_repository", IndexKeyRepo())
    monkeypatch.setattr(
        api_v1, "api_account_cleanup", SimpleNamespace(ensure_active=lambda uid: None)
    )
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: True)
    monkeypatch.setattr(
        api_v1.share_snapshots,
        "set_api_share_indexing",
        lambda requested, uid, *, indexed, actor_key_id: {
            "owner_uid": uid,
            "slug": "indexed-page",
            "question": "An indexable question?",
            "status": "active",
            "visibility": "public",
            "index_eligible": True,
            "indexed": indexed,
        },
    )
    client = TestClient(main.app)
    response = client.put(
        f"/api/v1/shares/{share_id}/indexing",
        headers={"X-API-Key": "cns_index_test"},
        json={"indexed": True},
    )

    assert response.status_code == 200
    assert response.json()["indexing_status"] == "indexed"
    assert response.json()["robots"] == "index, follow"
    assert response.json()["in_sitemap"] is True

    identity.scopes = ("share:write",)
    denied_scope = client.put(
        f"/api/v1/shares/{share_id}/indexing",
        headers={"X-API-Key": "cns_index_test"},
        json={"indexed": True},
    )
    assert denied_scope.status_code == 403

    identity.scopes = ("share:index",)
    monkeypatch.setattr(api_v1, "is_user_admin", lambda uid: False)
    denied_admin = client.put(
        f"/api/v1/shares/{share_id}/indexing",
        headers={"X-API-Key": "cns_index_test"},
        json={"indexed": True},
    )
    assert denied_admin.status_code == 403
