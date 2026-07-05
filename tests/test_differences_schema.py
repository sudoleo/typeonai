import json
import unittest
from unittest import mock

import app.core.config as cfg
from app.services.llm.consensus_engine import (
    _build_differences_prompt,
    _differences_attempts,
    _legacy_differences_text,
    _resolve_differences_engine,
    compute_agreement_score,
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
        # Ohne Severity-Angabe gilt der konservative Default "major"
        self.assertEqual(diff["severity"], "major")
        self.assertEqual(diff["positions"][0]["models"], ["OpenAI", "Gemini"])
        self.assertEqual(diff["positions"][1]["models"], ["Grok"])

        # Agreement-Score: base (2/3 + 3/3)/2 = 0.833, -0.25 major = 0.583
        agreement = data["agreement"]
        self.assertEqual(agreement["score"], 58)
        self.assertEqual(agreement["level"], "partially")
        self.assertEqual(agreement["major_contradictions"], 1)
        self.assertEqual(agreement["model_count"], 3)

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


class JsonRepairAndShapeTests(unittest.TestCase):
    def test_truncated_json_is_repaired(self):
        # Abbruch mitten im "verify"-String (max_tokens-Szenario): die
        # vollständigen Claims/Differences davor bleiben erhalten.
        raw = json.dumps(valid_payload())
        truncated = raw[: raw.index("Check the official") + 9]
        data, legacy = parse_differences_payload(truncated, ANON_MAP)

        self.assertIsNotNone(data)
        self.assertEqual(len(data["claims"]), 2)
        self.assertEqual(len(data["differences"]), 1)
        # best_model war abgeschnitten
        self.assertEqual(data["best_model"], "")
        self.assertIn("partially", legacy)

    def test_incomplete_object_shape_is_rejected(self):
        # Reparierbar, aber ohne "differences"-Liste: fehlende Widersprüche
        # dürfen nicht als "keine Widersprüche" durchgehen.
        data, legacy = parse_differences_payload('{"claims": [{"anchor": "x", "agree": [', ANON_MAP)
        self.assertIsNone(data)
        self.assertEqual(legacy, "")

    def test_json_garbage_never_leaks_raw_text(self):
        data, legacy = parse_differences_payload("```json\n{\"claims\": bro", ANON_MAP)
        self.assertIsNone(data)
        self.assertEqual(legacy, "")


class QuoteVerificationTests(unittest.TestCase):
    def test_found_anchor_and_quotes_use_original_text(self):
        consensus = (
            "Intro. The Capital of  France is Paris. "
            "It was founded in the third century BC."
        )
        model_answers = {"Grok": "I disagree: THE CAPITAL IS LYON. More text."}
        data, _ = parse_differences_payload(
            json.dumps(valid_payload()), ANON_MAP,
            consensus_answer=consensus, model_answers=model_answers,
        )

        # Anchor wird durch den Original-Wortlaut ersetzt (Casing/Whitespace)
        self.assertEqual(data["claims"][0]["anchor"], "The Capital of  France is Paris")
        self.assertEqual(data["claims"][1]["anchor"], "founded in the third century BC")
        # Dissent-Quote wird gegen die Grok-Antwort verifiziert
        self.assertEqual(data["claims"][0]["dissent"][0]["quote"], "THE CAPITAL IS LYON")
        # Grok-Position ebenso; die OpenAI/Gemini-Position ist nicht belegbar
        # (keine Antworttexte vorhanden) und wird geleert
        self.assertEqual(data["differences"][0]["positions"][0]["quote"], "")
        self.assertEqual(data["differences"][0]["positions"][1]["quote"], "THE CAPITAL IS LYON")

    def test_fuzzy_anchor_match(self):
        payload = valid_payload()
        payload["claims"][0]["anchor"] = "the capital of France is certainly Paris"
        consensus = "Well. The capital of France is Paris. End."
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=consensus, model_answers={},
        )
        self.assertTrue(data["claims"][0]["anchor"].startswith("The capital of France is"))

    def test_unfindable_anchor_is_kept_for_fallback_box(self):
        payload = valid_payload()
        payload["claims"][0]["anchor"] = "completely unrelated hallucinated sentence here"
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer="Something else entirely.", model_answers={},
        )
        self.assertEqual(
            data["claims"][0]["anchor"],
            "completely unrelated hallucinated sentence here",
        )


class JudgePolicyTests(unittest.TestCase):
    def test_pro_engines_use_default_judge(self):
        provider, api_model, _model_ref = _resolve_differences_engine("OpenAI-Pro")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, cfg.DEFAULT_OPENAI_MODEL)

        provider, api_model, _model_ref = _resolve_differences_engine("Anthropic-Pro")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(api_model, cfg.DEFAULT_ANTHROPIC_MODEL)

    def test_invalid_engine_returns_none(self):
        self.assertIsNone(_resolve_differences_engine("DoesNotExist"))
        self.assertIsNone(_differences_attempts("DoesNotExist", {}))

    def test_attempts_are_primary_retry_fallback(self):
        with mock.patch.dict("os.environ", {"DEVELOPER_GEMINI_API_KEY": ""}):
            attempts = _differences_attempts("OpenAI", {"OpenAI": "sk-1", "Mistral": "sk-2"})
        self.assertEqual(len(attempts), 3)
        (p1, _, _), retry1 = attempts[0]
        (p2, _, _), retry2 = attempts[1]
        (p3, _, _), retry3 = attempts[2]
        self.assertEqual((p1, retry1), ("openai", False))
        self.assertEqual((p2, retry2), ("openai", True))
        self.assertEqual((p3, retry3), ("mistral", True))

    def test_attempts_without_other_keys_have_no_fallback(self):
        with mock.patch.dict("os.environ", {"DEVELOPER_GEMINI_API_KEY": ""}):
            attempts = _differences_attempts("OpenAI", {"OpenAI": "sk-1"})
        self.assertEqual(len(attempts), 2)


FOUR_MODELS = ["OpenAI", "Gemini", "Grok", "Mistral"]


class LegacyTextSynthesisTests(unittest.TestCase):
    """Der Credibility-Satz leitet sich jetzt aus dem Agreement-Score ab."""

    def test_no_differences_is_very_credible(self):
        legacy = _legacy_differences_text({
            "claims": [], "differences": [], "best_model": "Gemini",
            "models_compared": FOUR_MODELS,
        })
        self.assertIn("**very** credible", legacy)
        self.assertIn("No substantive contradictions", legacy)
        self.assertIn("BestModel: Gemini", legacy)

    def test_only_emphasis_is_largely_credible(self):
        legacy = _legacy_differences_text({
            "differences": [{"claim": "Different focus on costs.", "type": "emphasis", "positions": []}],
            "best_model": "",
            "models_compared": FOUR_MODELS,
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
            "models_compared": FOUR_MODELS,
        })
        self.assertIn("**hardly** credible", legacy)


class AgreementScoreTests(unittest.TestCase):
    def test_clean_run_with_four_models_is_perfect(self):
        agreement = compute_agreement_score({
            "claims": [{"anchor": "a", "agree": ["OpenAI", "Gemini"], "dissent": []}],
            "differences": [],
            "models_compared": FOUR_MODELS,
        })
        self.assertEqual(agreement["score"], 100)
        self.assertEqual(agreement["level"], "very")

    def test_two_models_cannot_reach_very(self):
        agreement = compute_agreement_score({
            "claims": [], "differences": [],
            "models_compared": ["OpenAI", "Gemini"],
        })
        self.assertEqual(agreement["score"], 75)
        self.assertEqual(agreement["level"], "largely")

    def test_minor_contradiction_hurts_less_than_major(self):
        base = {
            "claims": [{"anchor": "a", "agree": ["OpenAI", "Gemini", "Grok"], "dissent": []}],
            "models_compared": FOUR_MODELS,
        }
        minor = compute_agreement_score({
            **base,
            "differences": [{"claim": "x", "type": "contradiction", "severity": "minor", "positions": []}],
        })
        major = compute_agreement_score({
            **base,
            "differences": [{"claim": "x", "type": "contradiction", "severity": "major", "positions": []}],
        })
        # Minor: 1.0 - 0.10 = 0.90, Cap 0.84 -> largely
        self.assertEqual(minor["score"], 84)
        self.assertEqual(minor["level"], "largely")
        self.assertEqual(minor["minor_contradictions"], 1)
        # Major: 1.0 - 0.25 = 0.75, Cap 0.64 -> partially
        self.assertEqual(major["score"], 64)
        self.assertEqual(major["level"], "partially")
        self.assertEqual(major["major_contradictions"], 1)

    def test_severity_minor_is_parsed_from_payload(self):
        payload = valid_payload()
        payload["differences"][0]["severity"] = "minor"
        data, legacy = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(data["differences"][0]["severity"], "minor")
        self.assertEqual(data["agreement"]["minor_contradictions"], 1)
        self.assertEqual(data["agreement"]["major_contradictions"], 0)
        # base 0.833 - 0.10 = 0.733 -> 73 -> largely (statt partially bei major)
        self.assertEqual(data["agreement"]["score"], 73)
        self.assertIn("**largely** credible", legacy)

    def test_emphasis_has_no_severity(self):
        payload = valid_payload()
        payload["differences"][0]["type"] = "emphasis"
        payload["differences"][0]["severity"] = "major"
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(data["differences"][0]["severity"], "")


class DifferencesPromptTests(unittest.TestCase):
    def test_prompt_requests_json_and_anonymizes(self):
        built = _build_differences_prompt(
            "answer one", "answer two", None, None, None, None,
            consensus_answer="the consensus",
            excluded_models=[],
        )
        self.assertIsNotNone(built)
        prompt, anon_map, answers_by_model = built
        self.assertIn("JSON", prompt)
        self.assertIn('"claims"', prompt)
        self.assertIn('"differences"', prompt)
        self.assertIn('"severity"', prompt)
        # Echte Modellnamen tauchen im Prompt nicht auf
        self.assertNotIn("OpenAI", prompt)
        self.assertNotIn("Mistral", prompt)
        self.assertEqual(sorted(anon_map.values()), ["Mistral", "OpenAI"])
        # answers_by_model liefert die Texte für die Zitat-Verifikation
        self.assertEqual(
            answers_by_model,
            {"OpenAI": "answer one", "Mistral": "answer two"},
        )


if __name__ == "__main__":
    unittest.main()
