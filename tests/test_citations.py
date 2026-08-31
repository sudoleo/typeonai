import unittest

from app.services.llm.citations import parse_openrouter_response, result_sources, result_text


class OpenRouterCitationParsingTests(unittest.TestCase):
    def test_url_annotations_become_deduplicated_tagged_sources(self):
        text = "The first claim is supported. The second claim is also supported."
        parsed = parse_openrouter_response(
            text,
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.com/article/",
                        "title": "Example article",
                        "content": "first claim",
                        "start_index": 0,
                        "end_index": 27,
                    },
                },
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.com/article",
                        "title": "Duplicate title",
                        "content": "second claim",
                        "start_index": 28,
                        "end_index": len(text),
                    },
                },
                {"type": "text", "text": "ignored"},
            ],
            "openai",
        )

        self.assertEqual(len(result_sources(parsed)), 1)
        self.assertEqual(result_sources(parsed)[0]["id"], "S1")
        self.assertEqual(result_sources(parsed)[0]["title"], "Example article")
        self.assertEqual(result_sources(parsed)[0]["snippet"], "first claim")
        self.assertEqual(result_sources(parsed)[0]["provider"], "openai")
        self.assertEqual(result_text(parsed).count("[S1]"), 2)

    def test_invalid_annotations_are_ignored_and_text_is_preserved(self):
        text = "No usable citation here."
        parsed = parse_openrouter_response(
            text,
            [
                {"type": "citation", "url_citation": {"url": "https://ignored.example"}},
                {"type": "url_citation", "url_citation": {"title": "Missing URL"}},
                {"type": "url_citation", "url_citation": {"url": ""}},
            ],
        )
        self.assertEqual(result_text(parsed), text)
        self.assertEqual(result_sources(parsed), [])


if __name__ == "__main__":
    unittest.main()
