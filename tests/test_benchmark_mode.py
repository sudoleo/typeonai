"""§9.1 – benchmark_mode entfernt Web-Tools fuer alle 6 Provider; False laesst die
Tool-Injektion (Status quo) unveraendert (Regression)."""

import unittest

from app.services.llm.engines import build_provider_payload
from benchmark.audit import assert_no_web_tools, find_web_tool_violations

PROVIDERS = ["openai", "mistral", "anthropic", "gemini", "deepseek", "grok"]
# Provider, die in der normalen App ein Web-Tool injizieren – seit DeepSeek die
# serverseitige Suche auf dem Anthropic-Endpoint anbietet, sind das alle sechs.
TOOL_PROVIDERS = PROVIDERS


def _build(provider, benchmark_mode):
    return build_provider_payload(
        provider,
        question="What is 2+2?",
        system_prompt="system",
        max_output_tokens=128,
        benchmark_mode=benchmark_mode,
    )["payload"]


class BenchmarkModeTests(unittest.TestCase):
    def test_benchmark_mode_removes_web_tools_for_all_providers(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                payload = _build(provider, benchmark_mode=True)
                # darf nirgends ein Web-Tool tragen
                assert_no_web_tools(payload, context=provider)
                self.assertNotIn("tools", payload)
                self.assertNotIn("tool_choice", payload)
                self.assertNotIn("include", payload)

    def test_normal_mode_still_injects_web_tools(self):
        for provider in TOOL_PROVIDERS:
            with self.subTest(provider=provider):
                payload = _build(provider, benchmark_mode=False)
                self.assertTrue(
                    find_web_tool_violations(payload),
                    f"{provider} should still inject a web tool in normal mode",
                )

    def test_deepseek_benchmark_mode_keeps_openrouter_compatible_payload(self):
        """Closed book remains on the shared OpenRouter Chat Completions path."""
        request = build_provider_payload(
            "deepseek",
            question="What is 2+2?",
            system_prompt="system",
            max_output_tokens=128,
            benchmark_mode=True,
        )
        self.assertEqual(request["endpoint"], "chat.completions")
        self.assertIn("messages", request["payload"])
        self.assertEqual(request["payload"]["messages"][0]["role"], "system")
        self.assertFalse(find_web_tool_violations(request["payload"]))

    def test_deepseek_normal_mode_uses_shared_openrouter_search(self):
        """Normal mode may search, but still uses the one OpenRouter endpoint."""
        request = build_provider_payload(
            "deepseek",
            question="What is 2+2?",
            system_prompt="system",
            max_output_tokens=128,
        )
        self.assertEqual(request["endpoint"], "chat.completions")
        self.assertEqual(request["payload"]["messages"][0]["content"], "system")
        self.assertEqual(
            request["payload"]["tools"],
            [{"type": "openrouter:web_search", "parameters": {"engine": "auto", "max_uses": 2}}],
        )

    def test_default_matches_normal_mode(self):
        # Default (benchmark_mode weggelassen) == explizit False (Produktion unveraendert).
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                default_payload = build_provider_payload(
                    provider, question="What is 2+2?", system_prompt="system", max_output_tokens=128
                )["payload"]
                self.assertEqual(default_payload, _build(provider, benchmark_mode=False))


if __name__ == "__main__":
    unittest.main()
