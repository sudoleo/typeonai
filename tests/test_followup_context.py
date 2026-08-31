"""Tests fuer Follow-up-Fragen: Kontext-Normalisierung/-Kappung und den
System-Prompt-Aufbau.

Nagelt die Vertraege fest: genau eine Kontext-Ebene ({previous_question,
previous_consensus}), serverseitige Caps (Kostenkontrolle — der Kontext geht
in alle /ask_*-Prompts gleichzeitig), KEIN Tier-Gate (ein Follow-up ist ein
normaler Lauf und zaehlt gegen das Tagesbudget), Injektion nur in handle_ask
(nicht doppelt via /prepare).
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app.core.config as cfg
from app.api.routers import chat as chat_router
from app.api.routers.chat import normalize_followup_context
from app.core.rate_limit import limiter
from app.services.llm.base import FOLLOWUP_CONTEXT_HEADER, build_followup_system_prompt
from usage_test_support import make_usage_repository


@pytest.fixture(autouse=True)
def reset_rate_limiter(monkeypatch):
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


def auth_patches(uid="uid-followup-tests", tier="free"):
    return (
        patch.object(chat_router, "verify_user_token", return_value=uid),
        patch.object(chat_router, "get_user_tier", return_value=tier),
    )


AUTH_HEADER = {"Authorization": "Bearer test-token"}

VALID_CONTEXT = {
    "previous_question": "What is quantum entanglement?",
    "previous_consensus": "Quantum entanglement is a correlation between particles.",
}


# ---------------------------------------------------------------------------
# normalize_followup_context: Validierung + Kappung
# ---------------------------------------------------------------------------

class TestNormalizeFollowupContext:
    def test_non_dict_payloads_are_ignored(self):
        assert normalize_followup_context(None) is None
        assert normalize_followup_context("context") is None
        assert normalize_followup_context(["a", "b"]) is None
        assert normalize_followup_context(42) is None

    def test_missing_or_empty_fields_are_ignored(self):
        assert normalize_followup_context({}) is None
        assert normalize_followup_context({"previous_question": "q"}) is None
        assert normalize_followup_context({"previous_consensus": "c"}) is None
        assert normalize_followup_context(
            {"previous_question": "   ", "previous_consensus": "c"}
        ) is None
        assert normalize_followup_context(
            {"previous_question": "q", "previous_consensus": ""}
        ) is None
        # Nicht-Strings zaehlen nicht als Kontext.
        assert normalize_followup_context(
            {"previous_question": ["q"], "previous_consensus": "c"}
        ) is None

    def test_valid_context_is_stripped_and_passed_through(self):
        ctx = normalize_followup_context(
            {"previous_question": "  q  ", "previous_consensus": "  c  "}
        )
        assert ctx == {"previous_question": "q", "previous_consensus": "c"}

    def test_oversized_texts_are_rejected(self):
        q_limit = cfg.get_followup_question_char_limit()
        c_limit = cfg.get_followup_consensus_char_limit()
        with pytest.raises(HTTPException) as exc_info:
            normalize_followup_context(
                {
                    "previous_question": "q" * (q_limit + 5_000),
                    "previous_consensus": "c" * (c_limit + 50_000),
                }
            )
        assert exc_info.value.status_code == 400

    def test_exactly_one_context_level_no_history(self):
        # Zusaetzliche Felder (z.B. ein verschachtelter Verlauf) werden
        # verworfen: das Ergebnis enthaelt genau das eine Frage/Konsens-Paar.
        ctx = normalize_followup_context(
            {
                "previous_question": "q",
                "previous_consensus": "c",
                "context": {"previous_question": "older q", "previous_consensus": "older c"},
                "history": ["turn1", "turn2"],
            }
        )
        assert set(ctx.keys()) == {"previous_question", "previous_consensus"}

    def test_limits_are_admin_overridable(self):
        original = cfg.get_limits_config()
        try:
            overrides = dict(original)
            overrides["followup_max_question_chars"] = 111
            overrides["followup_max_consensus_chars"] = 222
            cfg.apply_limits(overrides)
            assert cfg.get_followup_question_char_limit() == 111
            assert cfg.get_followup_consensus_char_limit() == 222
        finally:
            cfg.apply_limits(original)


# ---------------------------------------------------------------------------
# build_followup_system_prompt: Prompt-Aufbau
# ---------------------------------------------------------------------------

class TestBuildFollowupSystemPrompt:
    def test_contains_context_and_base_prompt(self):
        prompt = build_followup_system_prompt("BASE PROMPT", "prev q", "prev consensus")
        assert FOLLOWUP_CONTEXT_HEADER in prompt
        assert "prev q" in prompt
        assert "prev consensus" in prompt
        assert prompt.endswith("BASE PROMPT")

    def test_context_block_precedes_base_prompt(self):
        # Gleiche Konvention wie die REAL-TIME-DATA-Injektion in /prepare:
        # Kontextblock vor dem eigentlichen System-Prompt.
        prompt = build_followup_system_prompt("BASE PROMPT", "prev q", "prev consensus")
        assert prompt.index(FOLLOWUP_CONTEXT_HEADER) < prompt.index("BASE PROMPT")
        assert prompt.index("prev q") < prompt.index("prev consensus")


# ---------------------------------------------------------------------------
# Endpoint-Gates + Injektion in handle_ask
# ---------------------------------------------------------------------------

def test_ask_with_context_works_for_free_users():
    """Follow-ups sind nicht mehr Pro-only: der Kontext wird auch fuer Free
    injiziert. Bezahlt wird er ueber das normale Tagesbudget."""
    client = make_client()
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    p1, p2 = auth_patches(tier="free")
    with p1, p2, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
        response = client.post(
            "/ask_gemini",
            headers=AUTH_HEADER,
            json={
                "question": "and how is it used?",
                "model": free_model("gemini"),
                "system_prompt": "BASE PROMPT",
                "context": VALID_CONTEXT,
            },
        )
    assert response.status_code == 200
    assert FOLLOWUP_CONTEXT_HEADER in captured["system_prompt"]
    assert "What is quantum entanglement?" in captured["system_prompt"]


def test_ask_with_context_version_loads_authoritative_context_without_compressing():
    client = make_client()
    captured = {}
    ids = {
        "chat_id": "a" * 32,
        "turn_id": "b" * 32,
        "context_version_id": "c" * 32,
    }

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    def resolve(uid, chat_id, turn_id, version_id, *, question, provider=""):
        assert uid == "uid-followup-tests"
        assert (chat_id, turn_id, version_id) == (
            ids["chat_id"], ids["turn_id"], ids["context_version_id"]
        )
        assert question == "Does that decision still hold?"
        # Der Provider entscheidet, wessen Vorantwort im Kontext landet.
        assert provider == "Gemini"
        return "AUTHORITATIVE CHAT CONTEXT: decision=PostgreSQL"

    p1, p2 = auth_patches(tier="free")
    with p1, p2:
        with patch.object(
            chat_router.chat_context_service, "resolve_for_ask", side_effect=resolve
        ), patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_gemini",
                headers=AUTH_HEADER,
                json={
                    "question": "Does that decision still hold?",
                    "model": free_model("gemini"),
                    "system_prompt": "BASE PROMPT",
                    **ids,
                },
            )

    assert response.status_code == 200
    assert "decision=PostgreSQL" in captured["system_prompt"]
    assert captured["system_prompt"].index("BASE PROMPT") < captured["system_prompt"].index(
        "Persistent Memory is managed only"
    )
    assert captured["system_prompt"].endswith("for future requests.")
    assert FOLLOWUP_CONTEXT_HEADER not in captured["system_prompt"]


def test_prepare_with_context_works_for_free_users():
    client = make_client()
    p1, p2 = auth_patches(tier="free")
    with p1, p2:
        response = client.post(
            "/prepare",
            headers=AUTH_HEADER,
            json={"question": "and how is it used?", "context": VALID_CONTEXT},
        )
    assert response.status_code == 200


def test_ask_rejects_oversized_context_before_provider_call():
    client = make_client()
    uid = "uid-followup-pro"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    oversized_consensus = "c" * (cfg.get_followup_consensus_char_limit() + 10_000)
    try:
        p1, p2 = auth_patches(uid=uid, tier="pro")
        with p1, p2, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_gemini",
                headers=AUTH_HEADER,
                json={
                    "question": "and how is it used?",
                    "model": free_model("gemini"),
                    "system_prompt": "BASE PROMPT",
                    "context": {
                        "previous_question": "What is quantum entanglement?",
                        "previous_consensus": oversized_consensus,
                    },
                },
            )
        assert response.status_code == 400
        assert captured == {}
    finally:
        pass


def test_ask_without_context_only_adds_the_memory_write_boundary():
    client = make_client()
    uid = "uid-followup-none"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    try:
        p1, p2 = auth_patches(uid=uid, tier="pro")
        with p1, p2, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
            response = client.post(
                "/ask_gemini",
                headers=AUTH_HEADER,
                json={
                    "question": "hello",
                    "model": free_model("gemini"),
                    "system_prompt": "BASE PROMPT",
                },
            )
        assert response.status_code == 200
        assert captured["system_prompt"].startswith("BASE PROMPT\n\nPersistent Memory")
        assert "Never say or imply that you saved, changed, or will remember" in captured[
            "system_prompt"
        ]
    finally:
        pass


def test_prepare_validates_but_does_not_inject_context():
    # Die Injektion passiert ausschliesslich in handle_ask — sonst stuende der
    # Kontextblock doppelt im Prompt (Client schickt system_prompt + context
    # an /ask_*).
    client = make_client()
    p1, p2 = auth_patches(uid="uid-followup-prepare", tier="pro")
    with p1, p2:
        response = client.post(
            "/prepare",
            headers=AUTH_HEADER,
            json={
                "question": "and how is it used?",
                "system_prompt": "BASE PROMPT",
                "context": VALID_CONTEXT,
            },
        )
    assert response.status_code == 200
    assert FOLLOWUP_CONTEXT_HEADER not in response.json()["system_prompt"]
