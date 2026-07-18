"""Tests fuer Follow-up-Fragen: Kontext-Normalisierung/-Kappung und den
System-Prompt-Aufbau.

Nagelt die Vertraege fest: genau eine Kontext-Ebene ({previous_question,
previous_consensus}), serverseitige Caps (Kostenkontrolle — der Kontext geht
in alle /ask_*-Prompts gleichzeitig), Pro-Gate in /prepare UND handle_ask,
Injektion nur in handle_ask (nicht doppelt via /prepare).
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
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


def auth_patches(uid="uid-followup-tests", is_pro=False, is_early=False):
    return (
        patch.object(chat_router, "verify_user_token", return_value=uid),
        patch.object(chat_router, "is_user_pro", return_value=is_pro),
        patch.object(chat_router, "is_user_early", return_value=is_early),
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

    def test_oversized_texts_are_capped_at_the_limits(self):
        q_limit = cfg.get_followup_question_char_limit()
        c_limit = cfg.get_followup_consensus_char_limit()
        ctx = normalize_followup_context(
            {
                "previous_question": "q" * (q_limit + 5_000),
                "previous_consensus": "c" * (c_limit + 50_000),
            }
        )
        assert len(ctx["previous_question"]) == q_limit
        assert len(ctx["previous_consensus"]) == c_limit

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

def test_ask_with_context_is_pro_only():
    client = make_client()
    p1, p2, p3 = auth_patches(is_pro=False)
    with p1, p2, p3:
        response = client.post(
            "/ask_gemini",
            headers=AUTH_HEADER,
            json={
                "question": "and how is it used?",
                "model": free_model("gemini"),
                "context": VALID_CONTEXT,
            },
        )
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "pro_required"


def test_prepare_with_context_is_pro_only():
    client = make_client()
    p1, p2, p3 = auth_patches(is_pro=False)
    with p1, p2, p3:
        response = client.post(
            "/prepare",
            headers=AUTH_HEADER,
            json={"question": "and how is it used?", "context": VALID_CONTEXT},
        )
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "pro_required"


def test_ask_injects_capped_context_into_system_prompt_for_pro():
    client = make_client()
    uid = "uid-followup-pro"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    oversized_consensus = "c" * (cfg.get_followup_consensus_char_limit() + 10_000)
    try:
        p1, p2, p3 = auth_patches(uid=uid, is_pro=True)
        with p1, p2, p3, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
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
        assert response.status_code == 200
        system_prompt = captured["system_prompt"]
        assert FOLLOWUP_CONTEXT_HEADER in system_prompt
        assert "What is quantum entanglement?" in system_prompt
        assert system_prompt.endswith("BASE PROMPT")
        # Der Konsens-Text wurde serverseitig gekappt, nicht 1:1 uebernommen.
        assert len(system_prompt) < len(oversized_consensus)
    finally:
        pass


def test_ask_without_context_leaves_system_prompt_untouched():
    client = make_client()
    uid = "uid-followup-none"
    captured = {}

    def fake_run_ask(provider, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    try:
        p1, p2, p3 = auth_patches(uid=uid, is_pro=True)
        with p1, p2, p3, patch.object(chat_router, "_run_ask", side_effect=fake_run_ask):
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
        assert captured["system_prompt"] == "BASE PROMPT"
    finally:
        pass


def test_prepare_validates_but_does_not_inject_context():
    # Die Injektion passiert ausschliesslich in handle_ask — sonst stuende der
    # Kontextblock doppelt im Prompt (Client schickt system_prompt + context
    # an /ask_*).
    client = make_client()
    p1, p2, p3 = auth_patches(uid="uid-followup-prepare", is_pro=True)
    with p1, p2, p3:
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
