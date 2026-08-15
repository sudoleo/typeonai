"""DeepSeek-Websuche ueber den Anthropic-kompatiblen Endpoint.

DeepSeek fuehrt die Suche serverseitig aus, liefert die Treffer als
``web_search_tool_result``-Bloecke, haengt aber **keine** ``citations`` an den
Textblock. Ohne den Fallback aus diesen Bloecken haette eine DeepSeek-Antwort
trotz Websuche gar keine Quellen.
"""

import unittest

from app.services.llm.citations import (
    parse_anthropic_response,
    result_sources,
    result_text,
)


def _deepseek_payload():
    """Antwortform, wie sie /anthropic/v1/messages fuer DeepSeek liefert."""
    return {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "..."},
            {
                "type": "server_tool_use",
                "name": "web_search",
                "input": {"query": "who won the last race"},
            },
            {
                "type": "web_search_tool_result",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "F1 2026 results",
                        "url": "https://www.espn.com/f1/story/_/id/1",
                        "encrypted_content": "opaque",
                    },
                    {
                        "type": "web_search_result",
                        "title": "Race report",
                        "url": "https://www.formula1.com/en/results/2026/races",
                    },
                    # Duplikat mit Slash – muss zusammenfallen.
                    {
                        "type": "web_search_result",
                        "title": "F1 2026 results",
                        "url": "https://www.espn.com/f1/story/_/id/1/",
                    },
                ],
            },
            {"type": "text", "text": "Lando Norris won the Hungarian Grand Prix."},
        ],
    }


class DeepSeekWebSearchParsingTests(unittest.TestCase):
    def test_search_results_become_sources(self):
        parsed = parse_anthropic_response(_deepseek_payload(), "deepseek")
        urls = [s["url"] for s in result_sources(parsed)]
        self.assertEqual(
            urls,
            [
                "https://www.espn.com/f1/story/_/id/1",
                "https://www.formula1.com/en/results/2026/races",
            ],
        )
        self.assertEqual(result_sources(parsed)[0]["provider"], "deepseek")

    def test_text_stays_free_of_inline_tags(self):
        """Die Trefferliste sagt "das wurde gesucht", nicht "dieser Satz stammt
        aus S3" – ein Inline-Tag wuerde eine Belegtiefe vortaeuschen, die
        DeepSeek nicht liefert."""
        parsed = parse_anthropic_response(_deepseek_payload(), "deepseek")
        self.assertEqual(
            result_text(parsed), "Lando Norris won the Hungarian Grand Prix."
        )

    def test_encrypted_content_is_not_leaked(self):
        parsed = parse_anthropic_response(_deepseek_payload(), "deepseek")
        for source in result_sources(parsed):
            self.assertNotIn("opaque", repr(source))

    def test_no_search_block_yields_no_sources(self):
        parsed = parse_anthropic_response(
            {"content": [{"type": "text", "text": "2 + 2 is 4."}]}, "deepseek"
        )
        self.assertEqual(result_sources(parsed), [])
        self.assertEqual(result_text(parsed), "2 + 2 is 4.")

    def test_real_citations_still_win(self):
        """Anthropic selbst liefert citations – dort bleibt es bei Inline-Tags,
        der Suchtreffer-Fallback darf die Liste nicht aufblaehen."""
        parsed = parse_anthropic_response(
            {
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "content": [
                            {"type": "web_search_result", "url": "https://unused.example"}
                        ],
                    },
                    {
                        "type": "text",
                        "text": "Paris is the capital.",
                        "citations": [
                            {
                                "url": "https://cited.example",
                                "title": "Cited",
                                "cited_text": "Paris",
                            }
                        ],
                    },
                ]
            }
        )
        self.assertEqual(
            [s["url"] for s in result_sources(parsed)], ["https://cited.example"]
        )
        self.assertIn("[S1]", result_text(parsed))


if __name__ == "__main__":
    unittest.main()
