"""§9.5 – Usage + Text-Extraktion aus kanonischen Provider-JSONs (gemockt),
je Provider (E1-Drift-Absicherung). Keine echten Calls."""

import unittest

from benchmark import transport


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "" if status_code < 400 else "error body"

    def json(self):
        return self._payload


def make_post(captured, response):
    def _post(url, json=None, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["params"] = params
        return response

    return _post


# Kanonische, minimale Roh-JSONs je Provider.
CANONICAL = {
    "openai": (
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "The answer is (C).", "annotations": []}]}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        },
        "The answer is (C).",
        {"prompt": 100, "completion": 20, "total": 120},
    ),
    "grok": (
        {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "The answer is (F).", "annotations": []}]}
            ],
            "usage": {"input_tokens": 70, "output_tokens": 11, "total_tokens": 81},
        },
        "The answer is (F).",
        {"prompt": 70, "completion": 11, "total": 81},
    ),
    "anthropic": (
        {
            "content": [{"type": "text", "text": "The answer is (B)."}],
            "usage": {"input_tokens": 50, "output_tokens": 10},
        },
        "The answer is (B).",
        {"prompt": 50, "completion": 10, "total": 60},  # total fehlt -> summiert
    ),
    "gemini": (
        {
            "candidates": [{"content": {"parts": [{"text": "The answer is (A)."}]}}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 5, "totalTokenCount": 35},
        },
        "The answer is (A).",
        {"prompt": 30, "completion": 5, "total": 35},
    ),
    "mistral": (
        {
            "outputs": [
                {"type": "message.output", "content": [{"type": "text", "text": "The answer is (D)."}]}
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
        },
        "The answer is (D).",
        {"prompt": 40, "completion": 8, "total": 48},
    ),
    "deepseek": (
        {
            "choices": [{"message": {"content": "The answer is (E)."}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 6, "total_tokens": 31},
        },
        "The answer is (E).",
        {"prompt": 25, "completion": 6, "total": 31},
    ),
}


class TransportTests(unittest.TestCase):
    def _request_data(self, provider):
        return {"provider": provider, "api_model": "test-model", "payload": {"x": 1}}

    def test_text_and_usage_per_provider(self):
        for provider, (raw, expected_text, expected_usage) in CANONICAL.items():
            with self.subTest(provider=provider):
                captured = {}
                post = make_post(captured, FakeResponse(raw))
                result = transport.execute(
                    self._request_data(provider), "fake-key", http_post=post
                )
                self.assertIsNone(result["error"])
                self.assertEqual(result["text"], expected_text)
                self.assertEqual(result["usage"], expected_usage)
                self.assertEqual(result["status"], 200)
                self.assertIs(result["raw"], raw)

    def test_gemini_key_goes_into_params_not_headers(self):
        captured = {}
        raw = CANONICAL["gemini"][0]
        post = make_post(captured, FakeResponse(raw))
        transport.execute(self._request_data("gemini"), "secret", http_post=post)
        self.assertEqual(captured["params"], {"key": "secret"})
        self.assertNotIn("Authorization", captured["headers"])

    def test_anthropic_uses_x_api_key_header(self):
        captured = {}
        raw = CANONICAL["anthropic"][0]
        post = make_post(captured, FakeResponse(raw))
        transport.execute(self._request_data("anthropic"), "secret", http_post=post)
        self.assertEqual(captured["headers"]["x-api-key"], "secret")

    def test_http_error_is_structured(self):
        post = make_post({}, FakeResponse({}, status_code=500))
        result = transport.execute(self._request_data("openai"), "k", http_post=post)
        self.assertEqual(result["error_code"], "provider_http_error")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["status"], 500)

    def test_transport_exception_is_structured(self):
        def boom(*args, **kwargs):
            raise RuntimeError("network down")

        result = transport.execute(self._request_data("openai"), "k", http_post=boom)
        self.assertEqual(result["error_code"], "transport_request_failed")
        self.assertIn("network down", result["error"])


if __name__ == "__main__":
    unittest.main()
