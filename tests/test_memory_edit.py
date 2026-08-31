from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import users as users_router
from app.core.rate_limit import limiter
from app.services import memory_edit, user_memory


UID = "memory-edit-owner"
NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


class Snapshot:
    def __init__(self, doc):
        self._doc = doc
        self.exists = doc.data is not None

    def to_dict(self):
        return dict(self._doc.data or {})


class Document:
    def __init__(self):
        self.data = None
        self.children = {}

    def get(self, transaction=None):
        return Snapshot(self)

    def set(self, value, merge=False):
        self.data = ({**(self.data or {}), **value} if merge else dict(value))

    def delete(self):
        self.data = None

    def collection(self, name):
        return Collection(self.children.setdefault(name, {}))


class Collection:
    def __init__(self, docs):
        self.docs = docs

    def document(self, name):
        return self.docs.setdefault(name, Document())


class Transaction:
    def set(self, ref, value, merge=False):
        ref.set(value, merge=merge)


class Database:
    def __init__(self):
        self.roots = {}

    def collection(self, name):
        return Collection(self.roots.setdefault(name, {}))

    def run_transaction(self, operation):
        return operation(Transaction())


def config(**updates):
    return {**cfg.DEFAULT_MEMORY_EDIT_CONFIG, **updates}


@pytest.fixture
def repository(monkeypatch):
    monkeypatch.setattr(
        memory_edit.persistence_guard,
        "ensure_account_write_allowed",
        lambda **kwargs: None,
    )
    db = Database()
    profile = (
        db.collection("users").document(UID)
        .collection(user_memory.MEMORY_COLLECTION)
        .document(user_memory.PROFILE_DOCUMENT_ID)
    )
    profile.set({
        **user_memory.empty_profile(),
        "role": "Works at Firma X.",
        "notes": "Prefers short answers.",
        "revision": 4,
    })
    return memory_edit.FirestoreMemoryEditRepository(db)


def test_model_patch_schema_is_strict_and_passage_bounded():
    assert memory_edit.parse_and_validate_patch({
        "operation": "replace",
        "target": "Works at Firma X.",
        "replacement": "Works at Firma Y.",
    })["operation"] == "replace"
    with pytest.raises(memory_edit.MemoryEditError):
        memory_edit.parse_and_validate_patch({
            "operation": "replace",
            "target": "x",
            "replacement": "y",
            "explanation": "trust me",
        })
    with pytest.raises(memory_edit.MemoryEditError):
        memory_edit.parse_and_validate_patch({
            "operation": "rewrite",
            "target": "",
            "replacement": "all memory",
        })


def test_replace_is_revision_checked_and_undo_restores_exact_content(repository):
    request_id = "request-00000001"
    fingerprint = "f" * 64
    reserved = repository.reserve(
        UID,
        client_request_id=request_id,
        fingerprint=fingerprint,
        tier="free",
        config=config(),
        now=NOW,
    )
    assert reserved["baseline_revision"] == 4

    result = repository.apply_patch(
        UID,
        client_request_id=request_id,
        fingerprint=fingerprint,
        patch={
            "operation": "replace",
            "target": "Works at Firma X.",
            "replacement": "Works at Firma Y.",
        },
        memory_limit=12_000,
        now=NOW + timedelta(seconds=1),
    )
    assert result["status"] == "applied"
    profile, revision = user_memory.FirestoreUserMemoryRepository(
        repository.db
    ).get_with_revision(UID)
    assert profile["role"] == "Works at Firma Y."
    assert revision == 5

    undone = repository.undo(
        UID,
        result["revision_id"],
        memory_limit=12_000,
        now=NOW + timedelta(seconds=2),
    )
    restored, restored_revision = user_memory.FirestoreUserMemoryRepository(
        repository.db
    ).get_with_revision(UID)
    assert undone["status"] == "undone"
    assert restored["role"] == "Works at Firma X."
    assert restored["notes"] == "Prefers short answers."
    assert restored_revision == 6


def test_non_unique_target_is_never_overwritten(repository):
    profile_ref = repository._profile_ref(UID)
    profile_ref.data["notes"] = "Firma X. then Firma X."
    reservation = repository.reserve(
        UID,
        client_request_id="request-00000002",
        fingerprint="a" * 64,
        tier="free",
        config=config(),
        now=NOW,
    )
    with pytest.raises(memory_edit.MemoryEditError, match="unambiguous"):
        repository.apply_patch(
            UID,
            client_request_id="request-00000002",
            fingerprint="a" * 64,
            patch={"operation": "delete", "target": "Firma X.", "replacement": ""},
            memory_limit=12_000,
            now=NOW + timedelta(seconds=1),
        )
    assert reservation["memory"]["notes"] == "Firma X. then Firma X."
    assert profile_ref.data["notes"] == "Firma X. then Firma X."


def test_same_client_request_never_calls_provider_twice(repository):
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return {
            "operation": "replace",
            "target": "Works at Firma X.",
            "replacement": "Works at Firma Y.",
        }

    service = memory_edit.MemoryEditService(repository, provider=provider)
    payload = dict(
        tier="free",
        client_request_id="request-00000003",
        source_kind="consensus",
        selected_text="You work at Firma X.",
        correction="I work at Firma Y.",
        config=config(),
    )
    first = service.edit(UID, **payload)
    second = service.edit(UID, **payload)
    assert first["revision_id"] == second["revision_id"]
    assert len(calls) == 1


def test_remember_intent_appends_when_no_related_entry_exists(repository):
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return {
            "operation": "append",
            "target": "",
            "replacement": "Lives in Berlin.",
        }

    service = memory_edit.MemoryEditService(repository, provider=provider)
    result = service.edit(
        UID,
        tier="free",
        client_request_id="request-add-0001",
        source_kind="model_answer",
        selected_text="You live in Berlin.",
        correction="I live in Berlin.",
        intent="add",
        config=config(),
    )
    profile, _ = user_memory.FirestoreUserMemoryRepository(
        repository.db
    ).get_with_revision(UID)
    assert result["status"] == "applied"
    assert result["operation"] == "append"
    assert profile["notes"] == "Prefers short answers.\nLives in Berlin."
    assert calls[0]["intent"] == "add"


def test_remember_intent_replaces_one_unique_conflicting_passage(repository):
    service = memory_edit.MemoryEditService(
        repository,
        provider=lambda **kwargs: {
            "operation": "replace",
            "target": "Works at Firma X.",
            "replacement": "Works at Firma Y.",
        },
    )
    result = service.edit(
        UID,
        tier="free",
        client_request_id="request-add-0002",
        source_kind="consensus",
        selected_text="You work at Firma Y.",
        correction="I work at Firma Y.",
        intent="add",
        config=config(),
    )
    profile, _ = user_memory.FirestoreUserMemoryRepository(
        repository.db
    ).get_with_revision(UID)
    assert result["operation"] == "replace"
    assert profile["role"] == "Works at Firma Y."
    assert profile["notes"] == "Prefers short answers."


def test_remember_intent_rejects_delete_patch(repository):
    service = memory_edit.MemoryEditService(
        repository,
        provider=lambda **kwargs: {
            "operation": "delete",
            "target": "Works at Firma X.",
            "replacement": "",
        },
    )
    with pytest.raises(memory_edit.MemoryEditError) as exc:
        service.edit(
            UID,
            tier="free",
            client_request_id="request-add-0003",
            source_kind="consensus",
            selected_text="You work at Firma Y.",
            correction="I work at Firma Y.",
            intent="add",
            config=config(),
        )
    assert exc.value.code == "invalid_model_patch"


def test_smallest_replace_preserves_unrelated_details(repository):
    profile_ref = repository._profile_ref(UID)
    profile_ref.data["role"] = "Works at Continental and lives in Hanover."
    repository.reserve(
        UID,
        client_request_id="request-merge-0001",
        fingerprint="7" * 64,
        tier="free",
        config=config(),
        now=NOW,
    )
    repository.apply_patch(
        UID,
        client_request_id="request-merge-0001",
        fingerprint="7" * 64,
        patch={
            "operation": "replace",
            "target": "Works at Continental",
            "replacement": "Works at VHV",
        },
        memory_limit=12_000,
        now=NOW + timedelta(seconds=1),
    )
    profile, _ = user_memory.FirestoreUserMemoryRepository(
        repository.db
    ).get_with_revision(UID)
    assert profile["role"] == "Works at VHV and lives in Hanover."
    assert profile["notes"] == "Prefers short answers."


def test_persistent_daily_budget_is_shared_by_repository_instances(repository):
    other_process = memory_edit.FirestoreMemoryEditRepository(repository.db)
    limited = config(memory_free_ai_edits_daily=1)
    repository.reserve(
        UID,
        client_request_id="request-00000004",
        fingerprint="b" * 64,
        tier="free",
        config=limited,
        now=NOW,
    )
    with pytest.raises(memory_edit.MemoryEditError) as exc:
        other_process.reserve(
            UID,
            client_request_id="request-00000005",
            fingerprint="c" * 64,
            tier="free",
            config=limited,
            now=NOW + timedelta(seconds=46),
        )
    assert exc.value.code == "daily_limit"


def test_over_plan_memory_is_not_truncated_or_charged_by_ai_edit(repository):
    profile_ref = repository._profile_ref(UID)
    profile_ref.data["notes"] = "x" * 12_001
    with pytest.raises(memory_edit.MemoryEditError) as exc:
        repository.reserve(
            UID,
            client_request_id="request-over-plan",
            fingerprint="9" * 64,
            tier="free",
            config=config(),
            now=NOW,
        )
    assert exc.value.code == "memory_limit"
    assert profile_ref.data["notes"] == "x" * 12_001
    usage = repository.db.collection(memory_edit.USAGE_COLLECTION).document(
        memory_edit._hash(UID)
    )
    assert not usage.get().exists


def test_invalid_admin_values_fall_back_to_safe_defaults():
    normalized = cfg.normalize_memory_edit_config({
        "memory_edit_enabled": "yes",
        "memory_edit_model": "unknown-model",
        "memory_free_chars": 999_999,
        "memory_pro_chars": -1,
        "memory_global_calls_daily": "not-a-number",
    })
    assert normalized == cfg.DEFAULT_MEMORY_EDIT_CONFIG


def test_provider_call_is_schema_bound_no_reasoning_and_output_capped(monkeypatch):
    captured = {}

    client_config = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"operation":"append","target":"","replacement":"Works at Firma Y."}'
            ))])

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        memory_edit,
        "openai_client",
        lambda **kwargs: (
            client_config.update(kwargs)
            or SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        ),
    )
    patch = memory_edit.request_memory_patch(
        model="gpt-5.6-luna",
        memory=user_memory.empty_profile(),
        source_kind="consensus",
        selected_text="Works at Firma X.",
        correction="Works at Firma Y.",
        max_output_tokens=150,
        timeout_seconds=10,
        intent="add",
    )
    assert patch["operation"] == "append"
    assert client_config["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["model"] == "openai/gpt-5.6-luna"
    assert captured["reasoning_effort"] == "none"
    assert captured["max_tokens"] == 150
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["extra_body"] == {"provider": {"zdr": True}}
    assert '"intent":"add"' in captured["messages"][1]["content"]
    assert "smallest exact, uniquely occurring substring" in captured["messages"][0]["content"]
    assert "preserve every detail" in captured["messages"][0]["content"]


def test_one_in_flight_edit_and_global_budget_are_persistent(repository):
    repository.reserve(
        UID,
        client_request_id="request-00000006",
        fingerprint="d" * 64,
        tier="free",
        config=config(),
        now=NOW,
    )
    with pytest.raises(memory_edit.MemoryEditError) as in_flight:
        memory_edit.FirestoreMemoryEditRepository(repository.db).reserve(
            UID,
            client_request_id="request-00000007",
            fingerprint="e" * 64,
            tier="free",
            config=config(),
            now=NOW + timedelta(seconds=1),
        )
    assert in_flight.value.code == "edit_in_progress"

    other_uid = "second-owner"
    with pytest.raises(memory_edit.MemoryEditError) as global_limit:
        repository.reserve(
            other_uid,
            client_request_id="request-00000008",
            fingerprint="f" * 64,
            tier="free",
            config=config(memory_global_calls_daily=1),
            now=NOW + timedelta(seconds=1),
        )
    assert global_limit.value.code == "global_limit"


def test_edit_endpoint_applies_explicit_feedback_without_confirmation(monkeypatch):
    calls = []

    class StubService:
        def edit(self, uid, **kwargs):
            calls.append((uid, kwargs))
            return {
                "status": "applied",
                "revision_id": "a" * 32,
                "revision": 2,
                "undo_expires_at": NOW.isoformat(),
            }

    limiter.reset()
    monkeypatch.setattr(users_router, "verify_user_token", lambda token: UID)
    monkeypatch.setattr(users_router, "get_user_tier", lambda uid: "free")
    monkeypatch.setattr(users_router, "memory_edit_service", StubService())
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(users_router.router)
    client = TestClient(app)
    response = client.post(
        "/api/my/memory/edit",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "client_request_id": "request-00000009",
            "source_kind": "model_answer",
            "selected_text": "You work at Firma X.",
            "correction": "I work at Firma Y.",
            "intent": "add",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "applied"
    assert calls[0][0] == UID
    assert calls[0][1]["correction"] == "I work at Firma Y."
    assert calls[0][1]["intent"] == "add"
