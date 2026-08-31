"""Unified OpenRouter benchmark transport (mocked; no live calls)."""

from benchmark import transport
from app.services.llm.engines import OPENROUTER_CHAT_COMPLETIONS_URL


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "" if status_code < 400 else "error body"

    def json(self):
        return self._payload


def make_post(captured, response):
    def _post(url, json=None, headers=None, params=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, params=params, timeout=timeout)
        return response

    return _post


OPENROUTER_RESPONSE = {
    "choices": [{
        "message": {
            "content": "The answer is (C).",
            "annotations": [{
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.test/source",
                    "title": "Source",
                    "content": "Evidence",
                    "start_index": 0,
                    "end_index": 3,
                },
            }],
        },
    }],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
}


def request_data(provider="openai"):
    return {
        "provider": provider,
        "api_model": f"{provider}/test-model",
        "payload": {"model": "stale-model", "messages": []},
    }


def test_all_model_families_use_one_openrouter_transport():
    for provider in ("openai", "mistral", "anthropic", "gemini", "deepseek", "grok"):
        captured = {}
        post = make_post(captured, FakeResponse(OPENROUTER_RESPONSE))
        result = transport.execute(request_data(provider), "fake-key", http_post=post)

        assert result["error"] is None
        assert result["text"] == "The [S1] answer is (C)."
        assert result["usage"] == {"prompt": 100, "completion": 20, "total": 120}
        assert result["raw"] is OPENROUTER_RESPONSE
        assert captured["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
        assert captured["params"] is None
        assert captured["json"]["model"] == f"{provider}/test-model"
        assert captured["headers"] == {
            "Authorization": "Bearer fake-key",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://consens.io",
            "X-Title": "consens.io",
        }


def test_shared_credential_mapping_is_accepted(monkeypatch):
    captured = {}
    post = make_post(captured, FakeResponse(OPENROUTER_RESPONSE))
    result = transport.execute(
        request_data(), {"OpenRouter": "secret"}, http_post=post
    )
    assert result["error"] is None
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_no_provider_specific_adc_or_auth_parameters_exist():
    captured = {}
    post = make_post(captured, FakeResponse(OPENROUTER_RESPONSE))
    transport.execute(request_data("gemini"), "secret", http_post=post)
    assert captured["params"] is None
    assert not hasattr(transport, "_gemini_adc_headers")
    assert not hasattr(transport, "parse_gemini_response")
    assert not hasattr(transport, "parse_anthropic_response")


def test_http_error_is_structured():
    post = make_post({}, FakeResponse({}, status_code=500))
    result = transport.execute(request_data(), "k", http_post=post)
    assert result["error_code"] == "provider_http_error"
    assert result["text"] == ""
    assert result["status"] == 500


def test_transport_exception_is_structured():
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    result = transport.call_provider(request_data(), "k", http_post=boom)
    assert result["error_code"] == "transport_request_failed"
    assert "network down" in result["error"]


def test_malformed_response_is_structured():
    post = make_post({}, FakeResponse({"choices": []}))
    result = transport.execute(request_data(), "k", http_post=post)
    assert result["error"] is None
    assert result["text"] == ""
    assert result["usage"] == {"prompt": 0, "completion": 0, "total": 0}
