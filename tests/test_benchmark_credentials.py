"""Benchmark uses exactly one OpenRouter credential (no provider/ADC keys)."""

from app.services.llm import credentials


def test_resolve_reads_only_openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setenv("UNRELATED_API_KEY", "provider-key")
    assert credentials.resolve_developer_api_keys(["OpenAI", "Gemini"]) == {
        "OpenRouter": "sk-openrouter"
    }


def test_resolve_blank_is_none(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert credentials.resolve_developer_api_keys()["OpenRouter"] is None


def test_missing_credentials_is_one_shared_key():
    assert credentials.missing_credentials({"OpenRouter": None}, ["OpenAI", "Gemini"]) == [
        "OpenRouter"
    ]
    assert credentials.missing_credentials({"OpenRouter": "k"}, ["OpenAI", "Gemini"]) == []


def test_openrouter_api_key_accepts_shared_mapping():
    assert credentials.openrouter_api_key({"OpenRouter": "  k  "}) == "k"
    assert credentials.openrouter_api_key({"OpenAI": "provider-key"}) is None
