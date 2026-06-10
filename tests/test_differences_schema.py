import json
import unittest

from app.services.llm.consensus_engine import (
    _build_differences_prompt,
    _legacy_differences_text,
    parse_differences_payload,
)

ANON_MAP = {
    "Model A": "OpenAI",
    "Model B": "Gemini",
    "Model C": "Grok",
}


def valid_payload():
    return {
        "claims": [
            {
                "anchor": "the capital of France is Paris",
                "agree": ["Model A", "Model B"],
                "dissent": [{"model": "Model C", "quote": "the capital is Lyon"}],
            },
            {
                "anchor": "founded in the third century BC",
                "agree": ["Model A", "Model B", "Model C"],
                "dissent": [],
            },
        ],
        "differences": [
            {
                "claim": "The models disagree on the capital city.",
                "type": "contradiction",
                "positions": [
                    {"stance": "Paris is the capital.", "models": ["Model A", "Model B"], "quote": "Paris is the capital"},
                    {"stance": "Lyon is the capital.", "models": ["Model C"], "quote": "the capital is Lyon"},
                ],
                "verify": "Check the official government source for the capital.",
            }
        ],
        "best_model": "Model A",
    }


class ParseDifferencesPayloadTests(unittest.TestCase):
    def test_valid_json_is_parsed_and_translated(self):
        data, legacy = parse_differences_payload(json.dumps(valid_payload()), ANON_MAP)

        self.assertIsNotNone(data)
        self.assertEqual(data["best_model"], "OpenAI")
        self.assertEqual(data["models_compared"], ["Gemini", "Grok", "OpenAI"])

        claim = data["claims"][0]
        self.assertEqual(claim["agree"], ["OpenAI", "Gemini"])
        self.assertEqual(claim["dissent"], [{"model": "Grok", "quote": "the capital is Lyon"}])

        diff = data["differences"][0]
        self.assertEqual(diff["type"], "contradiction")
        self.assertEqual(diff["positions"][0]["models"], ["OpenAI", "Gemini"])
        self.assertEqual(diff["positions"][1]["models"], ["Grok"])

        self.assertIn("partially", legacy)
        self.assertIn("BestModel: OpenAI", legacy)

    def test_json_inside_markdown_fences(self):
        raw = "Here is my analysis:\n```json\n" + json.dumps(valid_payload()) + "\n```\n"
        data, _ = parse_differences_payload(raw, ANON_MAP)
        self.assertIsNotNone(data)
        self.assertEqual(data["best_model"], "OpenAI")

    def test_hallucinated_labels_are_dropped(self):
        payload = valid_payload()
        payload["claims"][0]["agree"].append("Model Z")
        payload["claims"][0]["dissent"].append({"model": "Model Q", "quote": "made up"})
        payload["differences"][0]["positions"].append(
            {"stance": "invented", "models": ["Model Z"], "quote": "x"}
        )
        payload["best_model"] = "Model Z"

        data, legacy = parse_differences_payload(json.dumps(payload), ANON_MAP)

        self.assertEqual(data["claims"][0]["agree"], ["OpenAI", "Gemini"])
        self.assertEqual([d["model"] for d in data["claims"][0]["dissent"]], ["Grok"])
        # Position ohne bekannte Modelle fliegt komplett raus
        self.assertEqual(len(data["differences"][0]["positions"]), 2)
        self.assertEqual(data["best_model"], "")
        self.assertNotIn("BestModel:", legacy)

    def test_unparsable_output_falls_back_to_raw_text(self):
        raw = "The consensus answer is **largely** credible.\n\nBestModel: Model B"
        data, legacy = parse_differences_payload(raw, ANON_MAP)
        self.assertIsNone(data)
        # Rohtext bleibt erhalten, BestModel wird rückübersetzt
        self.assertIn("largely", legacy)
        self.assertIn("BestModel: Gemini", legacy)

    def test_empty_string_returns_none(self):
        data, legacy = parse_differences_payload("", ANON_MAP)
        self.assertIsNone(data)
        self.assertEqual(legacy, "")

    def test_unknown_type_defaults_to_emphasis(self):
        payload = valid_payload()
        payload["differences"][0]["type"] = "stylistic"
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(data["differences"][0]["type"], "emphasis")

    def test_dissent_wins_over_agree_for_same_model(self):
        payload = valid_payload()
        payload["claims"][0]["agree"] = ["Model A", "Model C"]
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        claim = data["claims"][0]
        self.assertEqual(claim["agree"], ["OpenAI"])
        self.assertEqual([d["model"] for d in claim["dissent"]], ["Grok"])


class LegacyTextSynthesisTests(unittest.TestCase):
    def test_no_differences_is_very_credible(self):
        legacy = _legacy_differences_text({"claims": [], "differences": [], "best_model": "Gemini"})
        self.assertIn("**very** credible", legacy)
        self.assertIn("No substantive contradictions", legacy)
        self.assertIn("BestModel: Gemini", legacy)

    def test_only_emphasis_is_largely_credible(self):
        legacy = _legacy_differences_text({
            "differences": [{"claim": "Different focus on costs.", "type": "emphasis", "positions": []}],
            "best_model": "",
        })
        self.assertIn("**largely** credible", legacy)
        self.assertIn("Different focus on costs.", legacy)

    def test_multiple_contradictions_are_hardly_credible(self):
        legacy = _legacy_differences_text({
            "differences": [
                {"claim": "A", "type": "contradiction", "positions": []},
                {"claim": "B", "type": "contradiction", "positions": []},
            ],
            "best_model": "",
        })
        self.assertIn("**hardly** credible", legacy)


class DifferencesPromptTests(unittest.TestCase):
    def test_prompt_requests_json_and_anonymizes(self):
        built = _build_differences_prompt(
            "answer one", "answer two", None, None, None, None,
            consensus_answer="the consensus",
            excluded_models=[],
        )
        self.assertIsNotNone(built)
        prompt, anon_map = built
        self.assertIn("JSON", prompt)
        self.assertIn('"claims"', prompt)
        self.assertIn('"differences"', prompt)
        # Echte Modellnamen tauchen im Prompt nicht auf
        self.assertNotIn("OpenAI", prompt)
        self.assertNotIn("Mistral", prompt)
        self.assertEqual(sorted(anon_map.values()), ["Mistral", "OpenAI"])


if __name__ == "__main__":
    unittest.main()
