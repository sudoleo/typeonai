"""§9.1 – benchmark_mode entfernt Web-Tools fuer alle 6 Provider; False laesst die
Tool-Injektion (Status quo) unveraendert (Regression)."""

import unittest

from app.services.llm.engines import (
    DEEPSEEK_SEARCH_MAX_USES,
    build_provider_payload,
)
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

    def test_deepseek_benchmark_mode_keeps_openai_compatible_payload(self):
        """Closed book bleibt auf /chat/completions: der Benchmark darf weder
        Endpoint noch Prompt-Format wechseln, sonst sind die V1-Laeufe nicht
        mehr vergleichbar."""
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

    def test_deepseek_normal_mode_uses_anthropic_endpoint_with_search(self):
        """Die Suche laeuft nur ueber /anthropic/v1/messages – /chat/completions
        lehnt `web_search` als Tool-Typ ab."""
        request = build_provider_payload(
            "deepseek",
            question="What is 2+2?",
            system_prompt="system",
            max_output_tokens=128,
        )
        self.assertEqual(request["endpoint"], "anthropic.messages")
        self.assertEqual(request["payload"]["system"], "system")
        self.assertEqual(
            request["payload"]["tools"],
            [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": DEEPSEEK_SEARCH_MAX_USES,
            }],
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
