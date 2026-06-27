"""Credential-Aufloesung + Validierung (Phase 2.5a). Keine echten Keys, kein HTTP."""

from app.services.llm import credentials

ALL = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]


def test_env_names_cover_all_six_providers():
    assert set(credentials.DEVELOPER_API_KEY_ENV) == set(ALL)
    assert credentials.DEVELOPER_API_KEY_ENV["Gemini"] == "DEVELOPER_GEMINI_API_KEY"


def test_resolve_reads_env(monkeypatch):
    monkeypatch.setenv("DEVELOPER_OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("DEVELOPER_GROK_API_KEY", raising=False)
    keys = credentials.resolve_developer_api_keys(["OpenAI", "Grok"])
    assert keys["OpenAI"] == "sk-openai"
    assert keys["Grok"] is None


def test_resolve_blank_is_none(monkeypatch):
    monkeypatch.setenv("DEVELOPER_MISTRAL_API_KEY", "   ")
    assert credentials.resolve_developer_api_keys(["Mistral"])["Mistral"] is None


def test_missing_credentials_lists_absent(monkeypatch):
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: False)
    api_keys = {"OpenAI": "k", "Mistral": None, "Anthropic": "k",
                "Gemini": None, "DeepSeek": "k", "Grok": "k"}
    assert set(credentials.missing_credentials(api_keys, ALL)) == {"Mistral", "Gemini"}


def test_gemini_ok_with_key_even_without_adc(monkeypatch):
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: False)
    assert credentials.missing_credentials({"Gemini": "k"}, ["Gemini"]) == []


def test_gemini_ok_with_adc_without_key(monkeypatch):
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: True)
    assert credentials.missing_credentials({"Gemini": None}, ["Gemini"]) == []


def test_gemini_missing_without_key_and_adc(monkeypatch):
    monkeypatch.setattr(credentials, "gemini_adc_available", lambda: False)
    assert credentials.missing_credentials({"Gemini": None}, ["Gemini"]) == ["Gemini"]
