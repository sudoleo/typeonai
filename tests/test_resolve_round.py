"""Tests fuer die Resolve-Runde (resolve_engine + POST /resolve)."""

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import bookmarks as bookmarks_router
from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter
from app.services.usage_repository import RunKind, UsageLimits
from app.services.llm.resolve_engine import (
    InvalidResolvePayload,
    normalize_resolve_positions,
    run_resolve_round,
)
from usage_test_support import make_usage_repository


@pytest.fixture(autouse=True)
def fake_run_usage(monkeypatch):
    repository, _ = make_usage_repository()
    monkeypatch.setattr(chat_router, "run_usage_repository", repository)
    monkeypatch.setattr(
        chat_router,
        "get_usage_run_key",
        lambda data: str(data.get("usage_run_key") or "test-resolve-run"),
    )
    yield repository


def make_positions():
    return [
        {"stance": "The bridge opened in 1937.", "models": ["OpenAI"], "quote": "opened in 1937"},
        {"stance": "The bridge opened in 1936.", "models": ["Gemini"], "quote": "completed in 1936"},
    ]


class TestNormalizeResolvePositions:
    def test_valid_payload_is_normalized(self):
        claim, positions = normalize_resolve_positions("Opening year", make_positions())
        assert claim == "Opening year"
        assert [p["models"] for p in positions] == [["OpenAI"], ["Gemini"]]

    def test_model_aliases_are_canonicalized(self):
        raw = make_positions()
        raw[0]["models"] = ["claude"]
        _, positions = normalize_resolve_positions("c", raw)
        assert positions[0]["models"] == ["Anthropic"]

    def test_unknown_models_are_dropped(self):
        raw = make_positions()
        raw[0]["models"] = ["NotAModel", "OpenAI"]
        _, positions = normalize_resolve_positions("c", raw)
        assert positions[0]["models"] == ["OpenAI"]

    def test_missing_claim_is_rejected(self):
        with pytest.raises(InvalidResolvePayload):
            normalize_resolve_positions("", make_positions())

    def test_single_position_is_rejected(self):
        with pytest.raises(InvalidResolvePayload):
            normalize_resolve_positions("c", make_positions()[:1])

    def test_same_model_on_both_sides_is_rejected(self):
        raw = make_positions()
        raw[1]["models"] = ["OpenAI"]  # Duplikat wird dedupliziert -> Position leer
        with pytest.raises(InvalidResolvePayload):
            normalize_resolve_positions("c", raw)

    def test_oversized_texts_are_clipped(self):
        raw = make_positions()
        raw[0]["stance"] = "x" * 5000
        _, positions = normalize_resolve_positions("c", raw)
        assert len(positions[0]["stance"]) <= 400


def run_round_with(fake_engine):
    _, positions = normalize_resolve_positions("Opening year", make_positions())
    with patch("app.services.llm.resolve_engine._call_engine_text", side_effect=fake_engine):
        return run_resolve_round(
            "When did it open?", "Opening year", positions,
            api_keys={"OpenRouter": "sk-or-test"},
        )


class TestRunResolveRound:
    def test_all_maintain_is_standoff(self):
        def fake(provider, *args, **kwargs):
            return json.dumps({"decision": "maintain", "position": "p", "reason": "r"})

        result = run_round_with(fake)
        assert result["outcome"] == "standoff"
        assert {r["model"] for r in result["results"]} == {"OpenAI", "Gemini"}

    def test_one_revision_is_resolved(self):
        def fake(provider, *args, **kwargs):
            decision = "revise" if provider == "gemini" else "maintain"
            return json.dumps({"decision": decision, "position": "p", "reason": "r"})

        result = run_round_with(fake)
        assert result["outcome"] == "resolved"
        by_model = {r["model"]: r["decision"] for r in result["results"]}
        assert by_model == {"OpenAI": "maintain", "Gemini": "revise"}

    def test_all_revise_is_mutual_revision(self):
        def fake(provider, *args, **kwargs):
            return json.dumps({"decision": "revise", "position": "p", "reason": "r"})

        assert run_round_with(fake)["outcome"] == "mutual_revision"

    def test_provider_errors_do_not_break_the_round(self):
        def fake(provider, *args, **kwargs):
            if provider == "openai":
                raise RuntimeError("503 - UNAVAILABLE")
            return json.dumps({"decision": "maintain", "position": "p", "reason": "r"})

        result = run_round_with(fake)
        assert result["outcome"] == "standoff"
        by_model = {r["model"]: r["decision"] for r in result["results"]}
        assert by_model["OpenAI"] == "error"

    def test_all_failures_yield_error_outcome(self):
        def fake(provider, *args, **kwargs):
            return "not json at all"

        assert run_round_with(fake)["outcome"] == "error"

    def test_invalid_decision_counts_as_error(self):
        def fake(provider, *args, **kwargs):
            return json.dumps({"decision": "shrug", "position": "p"})

        assert run_round_with(fake)["outcome"] == "error"

    def test_missing_shared_key_skips_all_engine_calls(self):
        _, positions = normalize_resolve_positions("c", make_positions())
        with patch("app.services.llm.resolve_engine._call_engine_text") as engine:
            engine.return_value = json.dumps({"decision": "maintain", "position": "p", "reason": "r"})
            result = run_resolve_round("q", "c", positions, api_keys={"OpenRouter": ""})
        by_model = {r["model"]: r for r in result["results"]}
        assert by_model["OpenAI"]["decision"] == "error"
        assert by_model["OpenAI"]["reason"] == "missing API key"
        assert by_model["Gemini"]["decision"] == "error"
        assert by_model["Gemini"]["reason"] == "missing API key"
        assert engine.call_count == 0


# ---------------------------------------------------------------------------
# Endpoint-Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter.reset()
    yield


def make_client():
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat_router.router)
    return TestClient(app)


AUTH_HEADER = {"Authorization": "Bearer test-token"}


def resolve_payload():
    return {
        "question": "When did it open?",
        "claim": "Opening year",
        "positions": make_positions(),
    }


class ResolveBookmarkSnapshot:
    def __init__(self, bookmark_id, data):
        self.id = bookmark_id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data or {})


class ResolveBookmarkRef:
    def __init__(self, bookmark_id, data):
        self.id = bookmark_id
        self.data = data

    def get(self):
        return ResolveBookmarkSnapshot(self.id, self.data)

    def set(self, patch_data, merge=False):
        assert merge is True
        for key, value in patch_data.items():
            if key == "responses":
                self.data.setdefault("responses", {}).update(value)
            else:
                self.data[key] = value


class ResolveBookmarkDb:
    def __init__(self, uid, bookmark_ref):
        self.uid = uid
        self.bookmark_ref = bookmark_ref

    def collection(self, name):
        assert name in {"users", "bookmarks"}
        return self

    def document(self, document_id):
        if document_id == self.uid:
            return self
        assert document_id == self.bookmark_ref.id
        return self.bookmark_ref


def test_resolve_requires_auth():
    client = make_client()
    response = client.post("/resolve", json=resolve_payload())
    assert response.status_code == 401


def test_resolve_requires_pro():
    client = make_client()
    with patch.object(chat_router, "verify_user_token", return_value="uid-free"), \
         patch.object(chat_router, "is_user_pro", return_value=False):
        response = client.post("/resolve", headers=AUTH_HEADER, json=resolve_payload())
    assert response.status_code == 403
    # Bare-App ohne main.py-Exception-Handler: detail bleibt verschachtelt.
    assert response.json()["detail"]["error_code"] == "pro_required"


def test_resolve_rejects_invalid_positions():
    client = make_client()
    payload = resolve_payload()
    payload["positions"] = payload["positions"][:1]
    with patch.object(chat_router, "verify_user_token", return_value="uid-r"), \
         patch.object(chat_router, "is_user_pro", return_value=True):
        response = client.post("/resolve", headers=AUTH_HEADER, json=payload)
    assert response.status_code == 400
    assert "two positions" in response.json()["detail"]


def test_resolve_counts_usage_and_returns_result(fake_run_usage):
    client = make_client()
    uid = "uid-resolve-usage"
    fake_result = {"claim": "Opening year", "outcome": "standoff", "results": []}
    with patch.object(chat_router, "verify_user_token", return_value=uid), \
         patch.object(chat_router, "is_user_pro", return_value=True), \
         patch.object(chat_router, "run_resolve_round", return_value=dict(fake_result)) as round_mock:
        response = client.post("/resolve", headers=AUTH_HEADER, json=resolve_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "standoff"
    assert body["is_pro_user"] is True
    snapshot = fake_run_usage.snapshot(
        uid,
        UsageLimits(
            total=cfg.get_consensus_run_limit(True),
            deep_think=cfg.get_deep_think_run_limit(True),
        ),
    )
    assert snapshot.total.consumed == 1
    # Positionen kommen normalisiert bei der Engine an.
    args = round_mock.call_args.args
    assert args[1] == "Opening year"


def test_resolve_rejects_a_key_reserved_for_a_normal_consensus(fake_run_usage):
    client = make_client()
    uid = "uid-resolve-key-purpose"
    payload = {**resolve_payload(), "usage_run_key": "normal-consensus-key"}
    limits = UsageLimits(
        total=cfg.get_consensus_run_limit(True),
        deep_think=cfg.get_deep_think_run_limit(True),
    )
    fake_run_usage.reserve(
        uid,
        payload["usage_run_key"],
        RunKind.REGULAR,
        limits,
        request_fingerprint=chat_router.usage_run_fingerprint(
            payload,
            question=payload["question"],
            deep_think=False,
        ),
    )

    with patch.object(chat_router, "verify_user_token", return_value=uid), \
         patch.object(chat_router, "is_user_pro", return_value=True), \
         patch.object(chat_router, "run_resolve_round") as round_mock:
        response = client.post("/resolve", headers=AUTH_HEADER, json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "usage_run_conflict"
    round_mock.assert_not_called()
    snapshot = fake_run_usage.snapshot(uid, limits)
    assert snapshot.total.reserved == 1
    assert snapshot.total.consumed == 0


def test_resolve_blocks_when_usage_limit_reached(fake_run_usage):
    client = make_client()
    uid = "uid-resolve-limit"
    limits = UsageLimits(
        total=cfg.get_consensus_run_limit(True),
        deep_think=cfg.get_deep_think_run_limit(True),
    )
    for index in range(limits.total):
        key = f"used-{index}"
        fake_run_usage.reserve(uid, key, RunKind.REGULAR, limits)
        fake_run_usage.consume(uid, key)
    with patch.object(chat_router, "verify_user_token", return_value=uid), \
         patch.object(chat_router, "is_user_pro", return_value=True):
        response = client.post("/resolve", headers=AUTH_HEADER, json=resolve_payload())
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "total_usage_limit_exceeded"


def test_resolve_persists_only_the_server_result_on_the_bound_bookmark_revision():
    uid = "uid-resolve-bookmark"
    bookmark_id = "bookmark_resolve"
    bookmark = {
        "query": "When did it open?",
        "chat_id": "a" * 32,
        "turn_id": "b" * 32,
        "share_result_id": "old-share-result",
        "responses": {
            "consensus": "Stored consensus",
            "differences_data": {
                "claims": [],
                "differences": [{
                    "claim": "Opening year",
                    "type": "contradiction",
                    "severity": "major",
                    "positions": make_positions(),
                }],
                "best_model": "",
                "models_compared": ["OpenAI", "Gemini"],
            },
        },
    }
    bookmark_ref = ResolveBookmarkRef(bookmark_id, bookmark)
    database = ResolveBookmarkDb(uid, bookmark_ref)
    payload = {
        **resolve_payload(),
        "bookmarkId": bookmark_id,
        "expectedBookmarkVersion": bookmarks_router._bookmark_share_version(bookmark),
    }
    server_result = {
        "claim": "Opening year",
        "outcome": "resolved",
        "results": [
            {"model": "OpenAI", "decision": "maintain", "position": "1937", "reason": "Evidence", "prompt": "private"},
            {"model": "Gemini", "decision": "revise", "position": "1937", "reason": "Corrected", "prompt": "private"},
        ],
    }

    with patch.object(chat_router, "db_firestore", database):
        persisted = chat_router._persist_resolve_bookmark(
            uid,
            payload,
            "Opening year",
            normalize_resolve_positions("Opening year", make_positions())[1],
            server_result,
        )

    assert persisted is True
    resolution = bookmark_ref.data["responses"]["differences_data"]["differences"][0]["resolution"]
    assert resolution["outcome"] == "resolved"
    assert [item["decision"] for item in resolution["results"]] == ["maintain", "revise"]
    assert all("prompt" not in item for item in resolution["results"])
    assert bookmark_ref.data["share_result_id"] == ""


def test_resolve_does_not_persist_after_the_bookmark_revision_advanced():
    uid = "uid-resolve-stale"
    bookmark_id = "bookmark_resolve_stale"
    old_bookmark = {
        "query": "Old question",
        "responses": {
            "consensus": "Old consensus",
            "differences_data": {
                "claims": [],
                "differences": [{
                    "claim": "Opening year",
                    "type": "contradiction",
                    "severity": "major",
                    "positions": make_positions(),
                }],
                "best_model": "",
                "models_compared": ["OpenAI", "Gemini"],
            },
        },
    }
    newer_bookmark = {**old_bookmark, "query": "Newer follow-up"}
    bookmark_ref = ResolveBookmarkRef(bookmark_id, newer_bookmark)
    database = ResolveBookmarkDb(uid, bookmark_ref)
    payload = {
        **resolve_payload(),
        "bookmarkId": bookmark_id,
        "expectedBookmarkVersion": bookmarks_router._bookmark_share_version(old_bookmark),
    }

    with patch.object(chat_router, "db_firestore", database):
        persisted = chat_router._persist_resolve_bookmark(
            uid,
            payload,
            "Opening year",
            normalize_resolve_positions("Opening year", make_positions())[1],
            {"outcome": "resolved", "results": []},
        )

    assert persisted is False
    assert "resolution" not in bookmark_ref.data["responses"]["differences_data"]["differences"][0]
