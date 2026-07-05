"""Tests fuer die Resolve-Runde (resolve_engine + POST /resolve)."""

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import chat as chat_router
from app.core.rate_limit import limiter
from app.core.state import usage_counter
from app.services.llm.resolve_engine import (
    InvalidResolvePayload,
    normalize_resolve_positions,
    run_resolve_round,
)


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
            api_keys={"OpenAI": "sk-1", "Gemini": "g-1"},
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

    def test_missing_key_skips_engine_call(self):
        _, positions = normalize_resolve_positions("c", make_positions())
        with patch("app.services.llm.resolve_engine._call_engine_text") as engine:
            engine.return_value = json.dumps({"decision": "maintain", "position": "p", "reason": "r"})
            result = run_resolve_round("q", "c", positions, api_keys={"Gemini": "g-1"})
        by_model = {r["model"]: r for r in result["results"]}
        assert by_model["OpenAI"]["decision"] == "error"
        assert by_model["OpenAI"]["reason"] == "missing API key"
        # Nur Gemini durfte den Engine-Call ausloesen.
        assert engine.call_count == 1


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


def test_resolve_requires_auth():
    client = make_client()
    response = client.post("/resolve", json=resolve_payload())
    assert response.status_code == 401


def test_resolve_rejects_invalid_positions():
    client = make_client()
    payload = resolve_payload()
    payload["positions"] = payload["positions"][:1]
    with patch.object(chat_router, "verify_user_token", return_value="uid-r"), \
         patch.object(chat_router, "is_user_pro", return_value=False):
        response = client.post("/resolve", headers=AUTH_HEADER, json=payload)
    assert response.status_code == 400
    assert "two positions" in response.json()["detail"]


def test_resolve_counts_usage_and_returns_result():
    client = make_client()
    uid = "uid-resolve-usage"
    usage_counter.pop(uid, None)
    fake_result = {"claim": "Opening year", "outcome": "standoff", "results": []}
    try:
        with patch.object(chat_router, "verify_user_token", return_value=uid), \
             patch.object(chat_router, "is_user_pro", return_value=False), \
             patch.object(chat_router, "run_resolve_round", return_value=dict(fake_result)) as round_mock:
            response = client.post("/resolve", headers=AUTH_HEADER, json=resolve_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "standoff"
        assert body["is_pro_user"] is False
        assert usage_counter[uid] == 1
        # Positionen kommen normalisiert bei der Engine an.
        _, kwargs_or_args = round_mock.call_args
        args = round_mock.call_args.args
        assert args[1] == "Opening year"
    finally:
        usage_counter.pop(uid, None)


def test_resolve_blocks_when_usage_limit_reached():
    client = make_client()
    uid = "uid-resolve-limit"
    usage_counter[uid] = cfg.get_usage_limit(False)
    try:
        with patch.object(chat_router, "verify_user_token", return_value=uid), \
             patch.object(chat_router, "is_user_pro", return_value=False):
            response = client.post("/resolve", headers=AUTH_HEADER, json=resolve_payload())
        assert response.status_code == 403
        # Bare-App ohne main.py-Exception-Handler: detail bleibt verschachtelt.
        assert response.json()["detail"]["error_code"] == "usage_limit_exceeded"
    finally:
        usage_counter.pop(uid, None)
