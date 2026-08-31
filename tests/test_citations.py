import unittest

from app.services.llm.citations import (
    SOURCE_SNIPPET_MAX_CHARS,
    parse_openrouter_response,
    result_sources,
    result_text,
)


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

    def test_page_sized_annotation_content_is_clipped_to_a_teaser(self):
        """OpenRouter schickt in `content` den Seitentext, nicht den Beleg.

        Gemessen bis ~4.000 Zeichen je Quelle. Ungekuerzt stand die halbe
        Seite in der Quellenliste unter der Antwort.
        """
        page = " ".join("wort%d" % i for i in range(1200))
        parsed = parse_openrouter_response(
            "Belegter Satz.",
            [{
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/lang",
                    "title": "Lange Seite",
                    "content": page,
                    "end_index": 14,
                },
            }],
        )

        snippet = result_sources(parsed)[0]["snippet"]
        self.assertLessEqual(len(snippet), SOURCE_SNIPPET_MAX_CHARS + 1)
        self.assertTrue(snippet.endswith("…"))
        self.assertTrue(snippet.startswith("wort0 wort1 "))
        # `extract` ist derselbe Teaser, nicht die ungekuerzte Fassung.
        self.assertEqual(result_sources(parsed)[0]["extract"], snippet)

    def test_short_snippets_stay_untouched(self):
        parsed = parse_openrouter_response(
            "Belegter Satz.",
            [{
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/kurz",
                    "content": "  Ein   kurzer Beleg.  ",
                    "end_index": 14,
                },
            }],
        )
        self.assertEqual(result_sources(parsed)[0]["snippet"], "Ein kurzer Beleg.")


if __name__ == "__main__":
    unittest.main()
