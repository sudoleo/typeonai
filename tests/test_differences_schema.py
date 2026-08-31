import json
import random
import unittest
from unittest import mock

import app.core.config as cfg
from app.services.llm.consensus_engine import (
    _build_differences_prompt,
    _call_engine_text,
    _differences_attempts,
    _enumerate_consensus_sentences,
    _legacy_differences_text,
    _judge_effort,
    _provider_error_is_retryable,
    _resolve_differences_engine,
    _stream_engine_text,
    compute_agreement_score,
    DIFFERENCES_JSON_SCHEMA,
    parse_differences_payload,
)
from app.services.llm.engines import _ProviderHTTPStatusError

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

    def test_truncation_before_any_difference_is_not_reported_as_agreement(self):
        """Eine am Token-Limit abgeschnittene Ausgabe hat den Rest VERLOREN.
        Ein dabei leeres "differences" als "keine Widersprueche" zu rendern
        (und die Score-Strafpunkte wegfallen zu lassen) waere eine erfundene
        Entwarnung - Retry/Fallback-Judge sind die richtige Antwort."""
        raw = json.dumps(valid_payload())
        truncated = raw[: raw.index('"differences"') + len('"differences": [')]
        data, legacy = parse_differences_payload(truncated, ANON_MAP)
        self.assertIsNone(data)
        self.assertEqual(legacy, "")

    def test_truncation_after_a_difference_keeps_what_was_written(self):
        """Steht mindestens ein Widerspruch vollstaendig da, bleibt er gueltig -
        nur der abgeschnittene Rest faellt weg."""
        payload = valid_payload()
        payload["differences"].append({
            "claim": "Second disputed point.",
            "type": "contradiction",
            "positions": [{"stance": "x", "models": ["Model A"], "quote": "q"}],
        })
        raw = json.dumps(payload)
        truncated = raw[: raw.index("Second disputed point") + 5]
        data, _ = parse_differences_payload(truncated, ANON_MAP)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["differences"]), 1)

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

    def test_difference_consensus_anchor_is_verified_against_the_consensus(self):
        """Der Widerspruchs-Anker zeigt in die Konsensantwort (nicht in eine
        Modellantwort) und wird wie claims[].anchor auf den Originalwortlaut
        normalisiert - das Frontend markiert damit den Satz inline."""
        payload = valid_payload()
        payload["differences"][0]["consensus_anchor"] = "the capital of  france is paris"
        consensus = "Intro. The Capital of  France is Paris. End."
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=consensus, model_answers={},
        )
        self.assertEqual(
            data["differences"][0]["consensus_anchor"],
            "The Capital of  France is Paris",
        )

    def test_unfindable_difference_anchor_is_cleared(self):
        """Anders als beim Claim-Anker (der in die Fallback-Box wandert) waere
        ein nicht auffindbarer Widerspruchs-Anker eine falsche Markierung im
        Text - er wird deshalb geleert, die Karte bleibt."""
        payload = valid_payload()
        payload["differences"][0]["consensus_anchor"] = "hallucinated passage nobody wrote"
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer="Something else entirely.", model_answers={},
        )
        self.assertEqual(data["differences"][0]["consensus_anchor"], "")

    def test_missing_difference_anchor_defaults_to_empty(self):
        """Alte Bookmarks/Snapshots ohne das Feld degradieren sauber."""
        data, _ = parse_differences_payload(
            json.dumps(valid_payload()), ANON_MAP,
            consensus_answer="Anything.", model_answers={},
        )
        self.assertEqual(data["differences"][0]["consensus_anchor"], "")

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
    """Judge-Familie ist immer eine andere als die der Consensus-Engine;
    die Judge-Stufe (standard/pro) folgt der gewählten Engine."""

    ALL_KEYS = {"OpenRouter": "sk-or"}

    def test_judge_family_differs_from_consensus_family(self):
        # Gemini ist die erste Familie der Priorität; für eine Gemini-Engine
        # muss der Judge trotzdem auf eine andere Familie ausweichen.
        (provider, api_model, _), tier = _resolve_differences_engine("Gemini", self.ALL_KEYS)
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, cfg.openrouter_model_id(cfg.DEFAULT_OPENAI_MODEL, "openai"))
        self.assertEqual(tier, "standard")

        (provider, api_model, _), tier = _resolve_differences_engine("OpenAI", self.ALL_KEYS)
        self.assertEqual(provider, "gemini")
        self.assertEqual(api_model, cfg.openrouter_model_id(cfg.GEMINI_FLASH_MODEL, "gemini"))
        self.assertEqual(tier, "standard")

    def test_pro_engine_gets_pro_judge_of_other_family(self):
        (provider, api_model, _), tier = _resolve_differences_engine("OpenAI-Pro", self.ALL_KEYS)
        self.assertEqual(provider, "gemini")
        self.assertEqual(api_model, cfg.openrouter_model_id(cfg.GEMINI_PRO_MODEL, "gemini"))
        self.assertEqual(tier, "pro")

        (provider, api_model, _), tier = _resolve_differences_engine("Gemini-Pro", self.ALL_KEYS)
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "openai/gpt-5.5")
        self.assertEqual(tier, "pro")

    def test_missing_common_key_fails_open_to_own_standard_judge(self):
        (provider, api_model, _), tier = _resolve_differences_engine(
            "OpenAI-Pro", {}
        )
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, cfg.openrouter_model_id(cfg.DEFAULT_OPENAI_MODEL, "openai"))
        self.assertEqual(tier, "standard")

    def test_invalid_engine_returns_none(self):
        self.assertIsNone(_resolve_differences_engine("DoesNotExist", {}))
        self.assertIsNone(_differences_attempts("DoesNotExist", {}))

    def test_attempts_are_primary_retry_fallback(self):
        attempts = _differences_attempts("OpenAI", self.ALL_KEYS)
        self.assertEqual(len(attempts), 3)
        (p1, _, _), retry1, tier1 = attempts[0]
        (p2, _, _), retry2, tier2 = attempts[1]
        (p3, _, _), retry3, tier3 = attempts[2]
        self.assertEqual((p1, retry1, tier1), ("gemini", False, "standard"))
        self.assertEqual((p2, retry2, tier2), ("gemini", True, "standard"))
        self.assertEqual((p3, retry3, tier3), ("deepseek", True, "standard"))

    def test_pro_attempts_fail_open_to_standard_judge(self):
        attempts = _differences_attempts("OpenAI-Pro", self.ALL_KEYS)
        self.assertEqual(len(attempts), 4)
        (p1, m1, _), _, tier1 = attempts[0]
        (p3, m3, _), _, tier3 = attempts[2]
        (p4, m4, _), _, tier4 = attempts[3]
        self.assertEqual((p1, m1, tier1), ("gemini", cfg.openrouter_model_id(cfg.GEMINI_PRO_MODEL, "gemini"), "pro"))
        self.assertEqual((p3, m3, tier3), ("deepseek", cfg.openrouter_model_id(cfg.DEEPSEEK_PRO_MODEL, "deepseek"), "pro"))
        # Letzte Stufe: Standard-Judge der Fallback-Familie
        self.assertEqual((p4, m4, tier4), ("deepseek", cfg.openrouter_model_id(cfg.DEFAULT_DEEPSEEK_MODEL, "deepseek"), "standard"))

    def test_attempts_without_any_cross_family_key(self):
        attempts = _differences_attempts("OpenAI", {})
        self.assertEqual(len(attempts), 2)
        for (provider, api_model, _), _, tier in attempts:
            self.assertEqual((provider, api_model, tier), (
                "openai", cfg.openrouter_model_id(cfg.DEFAULT_OPENAI_MODEL, "openai"), "standard"
            ))

    def test_differences_judge_uses_openrouter_json_schema(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        with mock.patch("app.services.llm.consensus_engine.requests.post", return_value=response) as post:
            raw = _call_engine_text(
                "gemini", cfg.openrouter_model_id(cfg.DEFAULT_GEMINI_MODEL, "gemini"),
                cfg.DEFAULT_GEMINI_MODEL, self.ALL_KEYS,
                system="system", prompt="prompt", max_tokens=2048,
                json_mode=True, effort="low", json_schema=DIFFERENCES_JSON_SCHEMA,
            )
        self.assertEqual(raw, "{}")
        payload = post.call_args.kwargs["json"]
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["schema"],
            DIFFERENCES_JSON_SCHEMA,
        )
        self.assertIs(response_format["json_schema"]["strict"], True)
        self.assertEqual(payload["provider"], {"zdr": True})

    def test_differences_schema_is_strict_mode_compatible(self):
        def assert_strict_object_schema(schema):
            if not isinstance(schema, dict):
                return
            if schema.get("type") == "object":
                properties = schema.get("properties", {})
                self.assertEqual(set(schema.get("required", [])), set(properties))
                self.assertIs(schema.get("additionalProperties"), False)
            for value in schema.values():
                if isinstance(value, dict):
                    assert_strict_object_schema(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_strict_object_schema(item)

        assert_strict_object_schema(DIFFERENCES_JSON_SCHEMA)

    def test_streaming_differences_judge_uses_same_json_schema(self):
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            yield {"type": "delta", "text": "{}"}

        with mock.patch(
            "app.services.llm.streaming.stream_chat_completion_text",
            side_effect=fake_stream,
        ):
            events = list(_stream_engine_text(
                "gemini",
                cfg.openrouter_model_id(cfg.DEFAULT_GEMINI_MODEL, "gemini"),
                cfg.DEFAULT_GEMINI_MODEL,
                self.ALL_KEYS,
                system="system",
                prompt="prompt",
                max_tokens=2048,
                json_mode=True,
                effort="low",
                json_schema=DIFFERENCES_JSON_SCHEMA,
            ))

        self.assertEqual(events, [{"type": "delta", "text": "{}"}])
        self.assertEqual(captured["response_format"]["type"], "json_schema")
        self.assertEqual(
            captured["response_format"]["json_schema"]["schema"],
            DIFFERENCES_JSON_SCHEMA,
        )

    def test_one_openrouter_key_makes_every_judge_family_available(self):
        (provider, _, _), _tier = _resolve_differences_engine("OpenAI", self.ALL_KEYS)
        self.assertEqual(provider, "gemini")

    def test_mistral_judge_uses_supported_none_effort(self):
        self.assertEqual(
            _judge_effort("mistral", cfg.MISTRAL_PRO_MODEL, "pro"),
            "none",
        )
        self.assertEqual(
            _judge_effort("gemini", cfg.GEMINI_PRO_MODEL, "pro"),
            "low",
        )

    def test_only_retryable_provider_errors_repeat_same_call(self):
        for message, expected in (
            ("Gemini: 400 - invalid schema", False),
            ("401 - invalid API key", False),
            ("404 - model not found", False),
            ("429 - rate limited", True),
            ("503 - temporarily unavailable", True),
            ("connection reset", True),
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    _provider_error_is_retryable(RuntimeError(message)),
                    expected,
                )

        for status_code, expected in (
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (429, True),
            (503, True),
        ):
            with self.subTest(status_code=status_code):
                self.assertEqual(
                    _provider_error_is_retryable(
                        _ProviderHTTPStatusError(status_code)
                    ),
                    expected,
                )


class JudgeMetadataTests(unittest.TestCase):
    """data['judges'] weist den Judge aus, der das Ergebnis TATSÄCHLICH
    geliefert hat (auch nach Fallbacks)."""

    def _run_query(self, api_keys, side_effect):
        from app.services.llm.consensus_engine import query_differences
        with mock.patch(
            "app.services.llm.consensus_engine._call_engine_text",
            side_effect=side_effect,
        ):
            return query_differences(
                {"openai": "answer one", "mistral": "answer two"},
                "the consensus", api_keys,
                differences_model="OpenAI",
                excluded_models=[],
            )

    def test_query_differences_reports_actual_judge(self):
        payload = json.dumps({"claims": [], "differences": [], "best_model": ""})
        _, data = self._run_query(
            {"OpenRouter": "sk-or"},
            lambda provider, *a, **kw: payload,
        )
        self.assertIsNotNone(data)
        judge = data["judges"]["differences"]
        self.assertEqual(judge["provider"], "Gemini")
        self.assertEqual(judge["model"], cfg.openrouter_model_id(cfg.GEMINI_FLASH_MODEL, "gemini"))
        self.assertEqual(judge["tier"], "standard")
        # v3-Metadaten: erster Versuch traf, Dauer ist eine nichtnegative Zahl.
        self.assertEqual(judge["attempts"], 1)
        self.assertIsInstance(judge["duration_ms"], int)
        self.assertGreaterEqual(judge["duration_ms"], 0)

    def test_fallback_judge_is_reported(self):
        payload = json.dumps({"claims": [], "differences": [], "best_model": ""})

        def flaky(provider, *args, **kwargs):
            if provider == "gemini":
                raise RuntimeError("503")
            return payload

        _, data = self._run_query(
            {"OpenRouter": "sk-or"}, flaky,
        )
        self.assertIsNotNone(data)
        self.assertEqual(data["judges"]["differences"]["provider"], "DeepSeek")

    def test_non_retryable_primary_error_skips_duplicate_call(self):
        payload = json.dumps({"claims": [], "differences": [], "best_model": ""})
        providers = []

        def invalid_key_then_fallback(provider, *args, **kwargs):
            providers.append(provider)
            if provider == "gemini":
                raise RuntimeError("OpenRouter: 401 - invalid API key")
            return payload

        _, data = self._run_query(
            {"OpenRouter": "sk-or"},
            invalid_key_then_fallback,
        )
        self.assertIsNotNone(data)
        self.assertEqual(providers, ["gemini", "deepseek"])
        self.assertEqual(data["judges"]["differences"]["provider"], "DeepSeek")
        self.assertEqual(data["judges"]["differences"]["attempts"], 2)

    def test_stream_differences_reports_judge(self):
        from app.services.llm.consensus_engine import stream_differences
        payload = json.dumps({"claims": [], "differences": [], "best_model": ""})
        efforts = []

        def fake_stream(provider, *args, **kwargs):
            efforts.append((provider, kwargs.get("effort")))
            # _stream_engine_text liefert Event-Dicts; Reasoning-Marker
            # dürfen das Parsen des JSON-Ergebnisses nicht stören.
            yield {"type": "reasoning"}
            yield {"type": "delta", "text": payload}

        with mock.patch(
            "app.services.llm.consensus_engine._stream_engine_text",
            side_effect=fake_stream,
        ):
            events = list(stream_differences(
                {"openai": "answer one", "mistral": "answer two"},
                "the consensus", {"OpenRouter": "sk-or"},
                differences_model="OpenAI",
                excluded_models=[],
            ))
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["data"]["judges"]["differences"]["provider"], "Gemini")
        self.assertEqual(efforts, [("gemini", "low")])


FOUR_MODELS = ["OpenAI", "Gemini", "Grok", "Mistral"]


class LegacyTextSynthesisTests(unittest.TestCase):
    """Der Credibility-Satz leitet sich jetzt aus dem Agreement-Score ab."""

    def test_no_differences_is_very_credible(self):
        legacy = _legacy_differences_text({
            "claims": [{"anchor": "a", "agree": FOUR_MODELS, "dissent": []}],
            "differences": [], "best_model": "Gemini",
            "models_compared": FOUR_MODELS,
        })
        self.assertIn("**very** credible", legacy)
        self.assertIn("No substantive contradictions", legacy)
        self.assertIn("BestModel: Gemini", legacy)

    def test_nothing_measured_is_not_very_credible(self):
        """Ohne einen einzigen belegten Satz gibt es nichts, worauf sich
        Zuversicht stuetzen koennte - auch wenn kein Widerspruch etwas
        abgezogen hat."""
        legacy = _legacy_differences_text({
            "claims": [], "differences": [], "best_model": "Gemini",
            "models_compared": FOUR_MODELS,
        })
        self.assertIn("**partially** credible", legacy)

    def test_only_emphasis_is_largely_credible(self):
        legacy = _legacy_differences_text({
            "claims": [{"anchor": "a", "agree": FOUR_MODELS, "dissent": []}],
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
            "claims": [{"anchor": "a", "agree": ["OpenAI", "Gemini"], "dissent": []}],
            "differences": [],
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
            {"openai": "answer one", "mistral": "answer two"},
            consensus_answer="This is the consensus answer.",
            excluded_models=[],
        )
        self.assertIsNotNone(built)
        prompt, anon_map, answers_by_model, sentences = built
        self.assertIn("JSON", prompt)
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
        # Die Konsensantwort steht nummeriert im Prompt; der Judge referenziert
        # Saetze ueber "s" statt sie abzuschreiben.
        self.assertIn("[1] This is the consensus answer.", prompt)
        self.assertEqual(sentences, ["This is the consensus answer."])
        self.assertIn('"s"', prompt)

    def test_follow_up_reading_reaches_the_judge_without_touching_single_runs(self):
        """Der Judge sah die Frage bisher gar nicht und wertete Antworten auf
        verschiedene Lesarten derselben Frage als inhaltlichen Widerspruch."""
        # Gleiche Anonymisierungs-Reihenfolge fuer beide Prompts erzwingen.
        random.seed(7)
        base = _build_differences_prompt(
            {"openai": "answer one", "mistral": "answer two"},
            consensus_answer="This is the consensus answer.",
            excluded_models=[],
        )[0]
        random.seed(7)
        with_question = _build_differences_prompt(
            {"openai": "answer one", "mistral": "answer two"},
            consensus_answer="This is the consensus answer.",
            excluded_models=[],
            resolved_question="How would you rate consens.io from 1 to 10?",
        )[0]

        self.assertIn("How would you rate consens.io from 1 to 10?", with_question)
        self.assertIn("not a factual", with_question)
        # Einzellauf: unveraenderter Prompt, damit die Score-Kalibrierung haelt.
        self.assertNotIn("resolved against the conversation", base)
        self.assertTrue(with_question.endswith(base))

    def test_differences_judge_no_longer_asks_for_the_claim_list(self):
        """Die Belegliste ist ein eigener Call (Coverage-Judge). Steht sie hier
        wieder im Prompt oder im Schema, konkurriert sie erneut mit den
        Widerspruechen um Aufmerksamkeit und Output-Tokens - genau der Grund,
        aus dem sie regelmaessig verkuerzt ankam."""
        built = _build_differences_prompt(
            {"openai": "answer one", "mistral": "answer two"},
            consensus_answer="This is the consensus answer.",
            excluded_models=[],
        )
        prompt = built[0]
        self.assertNotIn('"claims"', prompt)
        self.assertNotIn('"agree"', prompt)

        self.assertNotIn("claims", DIFFERENCES_JSON_SCHEMA["properties"])
        self.assertEqual(
            DIFFERENCES_JSON_SCHEMA["required"],
            ["differences", "best_model"],
        )

    def test_long_consensus_is_numbered_sentence_by_sentence(self):
        consensus = (
            "# Overview\n\n"
            "The tower is 330 metres tall. It was finished in 1889, i.e. for the fair.\n\n"
            "- Tickets cost about 30 euros for adults.\n"
            "- Short.\n"
        )
        built = _build_differences_prompt(
            {"openai": "answer one"},
            consensus_answer=consensus,
            excluded_models=[],
        )
        prompt, _anon_map, _answers, sentences = built

        self.assertEqual(sentences, [
            "The tower is 330 metres tall.",
            "It was finished in 1889, i.e. for the fair.",
            "Tickets cost about 30 euros for adults.",
        ])
        # Ueberschrift bleibt unnummeriert, der Listenzaehler steht vor der Marke
        self.assertIn("# Overview", prompt)
        self.assertNotIn("[1] # Overview", prompt)
        self.assertIn("[1] The tower", prompt)
        self.assertIn("[2] It was finished", prompt)
        self.assertIn("- [3] Tickets cost", prompt)
        # Jeder Satz ist ein exakter Ausschnitt der Konsensantwort
        for sentence in sentences:
            self.assertIn(sentence, consensus)


class ConsensusSentenceSplitTests(unittest.TestCase):
    """Jeder nummerierte Satz muss ein EXAKTER Ausschnitt der Konsensantwort
    sein - nur dann findet ihn das Frontend im gerenderten Text wieder."""

    def sentences(self, text):
        numbered, sentences = _enumerate_consensus_sentences(text)
        for sentence in sentences:
            self.assertIn(sentence, text, "Anker muss ein Substring des Konsens sein")
        return numbered, sentences

    def test_plain_sentences_are_split(self):
        _numbered, sentences = self.sentences(
            "The tower is 330 metres tall. It was finished in 1889."
        )
        self.assertEqual(sentences, [
            "The tower is 330 metres tall.",
            "It was finished in 1889.",
        ])

    def test_year_at_the_end_is_a_sentence_end(self):
        """Der haeufigste Faktensatz ueberhaupt endet auf eine Zahl - er darf
        nicht mit dem Folgesatz verschmelzen."""
        _numbered, sentences = self.sentences(
            "Es wurde 1889 fertiggestellt. Der Bau dauerte gut zwei Jahre."
        )
        self.assertEqual(len(sentences), 2)

    def test_source_tag_neither_blocks_the_split_nor_enters_the_anchor(self):
        numbered, sentences = self.sentences(
            "Der Turm ist 330 m hoch.[S1] Er wurde 1889 fertiggestellt."
        )
        self.assertEqual(sentences[0], "Der Turm ist 330 m hoch.")
        self.assertEqual(sentences[1], "Er wurde 1889 fertiggestellt.")
        self.assertIn("[1] Der Turm", numbered)
        self.assertIn("[2] Er wurde", numbered)

    def test_abbreviations_and_initials_do_not_split(self):
        _numbered, sentences = self.sentences(
            "Laut J. R. R. Tolkien ist das anders und u.a. deshalb umstritten."
        )
        self.assertEqual(len(sentences), 1)

    def test_currency_abbreviations_do_not_break_markdown_claims(self):
        """Mrd./Mio. stehen vor dem Waehrungszeichen, nicht am Satzende.

        Ein Split an dieser Stelle erzeugt Fragmente mit verwaisten **, die in
        der Key-Claims-Liste als rohe Markdown-Zeichen sichtbar werden.
        """
        _numbered, sentences = self.sentences(
            "**+18 % QoQ** = Anstieg von **5,7 Mrd. $ in Q1** gegenueber "
            "**6,7 Mrd. $ in Q2**. Die **40 Mio. EUR ARR** sind annualisiert."
        )
        self.assertEqual(sentences, [
            "**+18 % QoQ** = Anstieg von **5,7 Mrd. $ in Q1** gegenueber "
            "**6,7 Mrd. $ in Q2**.",
            "Die **40 Mio. EUR ARR** sind annualisiert.",
        ])

    def test_quantity_abbreviation_can_still_end_a_sentence(self):
        """Die Waehrungs-Sonderregel darf echte Satzenden nicht verschlucken."""
        _numbered, sentences = self.sentences(
            "Der Quartalsumsatz liegt bei 40 Mio. Danach steigt die Prognose weiter."
        )
        self.assertEqual(sentences, [
            "Der Quartalsumsatz liegt bei 40 Mio.",
            "Danach steigt die Prognose weiter.",
        ])

    def test_display_math_blocks_are_not_numbered(self):
        """Eine abgesetzte Formel ist kein Satz.

        Als Anker waere sie im gerenderten Konsens unauffindbar - dort ist sie
        ein KaTeX-Block - und landete deshalb samt LaTeX-Quelltext
        ("6{,}7 / 5{,}7 - 1 \\approx 17{,}5%") in der Key-claims-Liste. Ein
        "[n] " zwischen ihren Zeilen wuerde sie zusaetzlich zerschneiden.
        """
        text = (
            "Zur Einordnung der beiden Zahlen:\n\n"
            "$$\n"
            "6{,}7 / 5{,}7 - 1 \\approx 17{,}5\\%\n"
            "$$\n\n"
            "Der durchschnittliche Monatsumsatz lag bei 2,23 Mrd. $ im Quartal."
        )
        numbered, sentences = self.sentences(text)
        self.assertEqual(sentences, [
            "Zur Einordnung der beiden Zahlen:",
            "Der durchschnittliche Monatsumsatz lag bei 2,23 Mrd. $ im Quartal.",
        ])
        self.assertIn("$$\n6{,}7 / 5{,}7 - 1 \\approx 17{,}5\\%\n$$", numbered)

    def test_single_line_display_math_does_not_swallow_the_next_paragraph(self):
        """"$$...$$" auf einer Zeile ist bereits geschlossen - alles danach
        bleibt normaler Fliesstext."""
        text = (
            "$$6{,}7 / 5{,}7 - 1 \\approx 17{,}5\\%$$\n\n"
            "Das entspricht rund 18 Prozent Wachstum.\n\n"
            "\\[ \\det(DF) \\equiv -2 \\]\n\n"
            "Die Determinante bleibt dabei konstant."
        )
        _numbered, sentences = self.sentences(text)
        self.assertEqual(sentences, [
            "Das entspricht rund 18 Prozent Wachstum.",
            "Die Determinante bleibt dabei konstant.",
        ])

    def test_inline_math_stays_part_of_its_sentence(self):
        """Nur ABGESETZTE Formeln fallen weg. Eine Formel im Satz gehoert zum
        Satz - der Anker traegt sie mit."""
        _numbered, sentences = self.sentences(
            "Der Zuwachs betraegt $17{,}5\\%$ gegenueber dem Vorquartal."
        )
        self.assertEqual(sentences, [
            "Der Zuwachs betraegt $17{,}5\\%$ gegenueber dem Vorquartal.",
        ])

    def test_headings_tables_and_code_are_not_numbered(self):
        text = (
            "# Titel\n\n"
            "Der erste pruefbare Satz steht hier.\n\n"
            "| a | b |\n|---|---|\n\n"
            "```python\nx = 1. Foo bar baz\n```\n"
        )
        numbered, sentences = self.sentences(text)
        self.assertEqual(sentences, ["Der erste pruefbare Satz steht hier."])
        self.assertIn("# Titel", numbered)
        self.assertNotIn("[1] # Titel", numbered)
        self.assertNotIn("[2]", numbered)

    def test_list_counter_stays_out_of_the_anchor(self):
        _numbered, sentences = self.sentences(
            "1. **Weltklasse:** ca. 1.300 Watt Dauerleistung.\n"
            "2. Ein Hobbyfahrer schafft dagegen rund 200 Watt.\n"
        )
        self.assertEqual(sentences, [
            "**Weltklasse:** ca. 1.300 Watt Dauerleistung.",
            "Ein Hobbyfahrer schafft dagegen rund 200 Watt.",
        ])

    def test_sentence_count_is_capped(self):
        text = " ".join(f"Dies ist der Satz Nummer {i} im Text." for i in range(200))
        _numbered, sentences = self.sentences(text)
        self.assertEqual(len(sentences), 80)

    def test_empty_answer_yields_no_sentences(self):
        numbered, sentences = _enumerate_consensus_sentences("")
        self.assertEqual(sentences, [])
        self.assertEqual(numbered, "")


class SentenceAnchorTests(unittest.TestCase):
    """Der Anker kommt aus der Satznummer statt aus einer Abschrift - damit
    ist er per Konstruktion im Konsenstext auffindbar."""

    CONSENSUS = (
        "The capital of France is Paris. "
        "It was founded in the third century BC."
    )

    def numbered_payload(self):
        payload = valid_payload()
        payload["claims"][0] = {
            "s": 1,
            "agree": ["Model A", "Model B"],
            "dissent": [{"model": "Model C", "quote": "the capital is Lyon"}],
        }
        payload["claims"][1] = {"s": 2, "agree": ["Model A", "Model B"], "dissent": []}
        payload["differences"][0]["s"] = 1
        return payload

    def test_sentence_numbers_resolve_to_the_exact_sentence(self):
        data, _ = parse_differences_payload(
            json.dumps(self.numbered_payload()), ANON_MAP,
            consensus_answer=self.CONSENSUS, model_answers={},
        )
        self.assertEqual(data["claims"][0]["anchor"], "The capital of France is Paris.")
        self.assertEqual(
            data["claims"][1]["anchor"], "It was founded in the third century BC.")
        self.assertEqual(
            data["differences"][0]["consensus_anchor"], "The capital of France is Paris.")

    def test_unknown_sentence_number_is_ignored(self):
        payload = self.numbered_payload()
        payload["claims"][0]["s"] = 99
        payload["differences"][0]["s"] = 99
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=self.CONSENSUS, model_answers={},
        )
        # Ohne gueltige Nummer und ohne Abschrift bleibt kein Anker uebrig -
        # der Claim faellt damit ganz weg, statt einen falschen Satz zu
        # markieren. Die Widerspruchs-Karte bleibt, nur ohne Inline-Marke.
        self.assertEqual(len(data["claims"]), 1)
        self.assertEqual(
            data["claims"][0]["anchor"], "It was founded in the third century BC.")
        self.assertEqual(data["differences"][0]["consensus_anchor"], "")

    def test_zero_means_the_consensus_does_not_state_it(self):
        payload = self.numbered_payload()
        payload["differences"][0]["s"] = 0
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=self.CONSENSUS, model_answers={},
        )
        self.assertEqual(data["differences"][0]["consensus_anchor"], "")

    def test_verbatim_anchor_still_works_for_older_payloads(self):
        """Alte Bookmarks und ein Judge, der doch abschreibt, bleiben gueltig."""
        data, _ = parse_differences_payload(
            json.dumps(valid_payload()), ANON_MAP,
            consensus_answer=self.CONSENSUS, model_answers={},
        )
        self.assertEqual(data["claims"][0]["anchor"], "The capital of France is Paris")

    def test_duplicate_sentence_keeps_the_more_conservative_claim(self):
        payload = self.numbered_payload()
        payload["claims"][1] = {"s": 1, "agree": ["Model A"], "dissent": []}
        payload["claims"].append(
            {"s": 1, "agree": ["Model A"], "dissent": [{"model": "Model C", "quote": "no"}]}
        )
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=self.CONSENSUS, model_answers={},
        )
        # Ein Satz, ein Claim - und zwar der am wenigsten gestuetzte (1/2)
        self.assertEqual(len(data["claims"]), 1)
        self.assertEqual(data["claims"][0]["agree"], ["OpenAI"])
        self.assertEqual([d["model"] for d in data["claims"][0]["dissent"]], ["Grok"])

    def test_identical_sentence_occurrences_keep_distinct_ids(self):
        consensus = "This remains uncertain. This remains uncertain."
        payload = self.numbered_payload()
        payload["claims"] = [
            {"s": 1, "agree": ["Model A", "Model B"], "dissent": []},
            {"s": 2, "agree": ["Model A"],
             "dissent": [{"model": "Model C", "quote": "no"}]},
        ]
        payload["differences"][0]["s"] = 2
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=consensus, model_answers={},
        )
        self.assertEqual([claim["sentence_id"] for claim in data["claims"]], [1, 2])
        self.assertEqual([claim["anchor_occurrence"] for claim in data["claims"]], [0, 1])
        self.assertEqual(data["differences"][0]["sentence_id"], 2)
        self.assertEqual(data["differences"][0]["anchor_occurrence"], 1)

    def test_visibly_identical_sentences_ignore_citations_and_markdown(self):
        consensus = (
            "**This remains uncertain.**[S1]\n\n"
            "This remains uncertain.[S2]"
        )
        payload = self.numbered_payload()
        payload["claims"] = [
            {"s": 1, "agree": ["Model A", "Model B"], "dissent": []},
            {"s": 2, "agree": ["Model A"],
             "dissent": [{"model": "Model C", "quote": "no"}]},
        ]
        payload["differences"][0]["s"] = 2
        data, _ = parse_differences_payload(
            json.dumps(payload), ANON_MAP,
            consensus_answer=consensus, model_answers={},
        )
        self.assertEqual(
            [claim["anchor_occurrence"] for claim in data["claims"]],
            [0, 1],
        )
        self.assertEqual(data["differences"][0]["anchor_occurrence"], 1)


class ClaimSupportThresholdTests(unittest.TestCase):
    """Eine einzelne Stimme belegt nichts: "1/1 - all models agree" liest sich
    wie eine Bestaetigung, ist aber nur ein Modell."""

    def test_single_voice_claim_is_dropped(self):
        payload = valid_payload()
        payload["claims"][1]["agree"] = ["Model A"]
        payload["claims"][1]["dissent"] = []
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(len(data["claims"]), 1)
        self.assertEqual(data["claims"][0]["agree"], ["OpenAI", "Gemini"])

    def test_two_voices_are_kept(self):
        payload = valid_payload()
        payload["claims"][1]["agree"] = ["Model A", "Model B"]
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(len(data["claims"]), 2)

    def test_single_dissent_alone_is_dropped(self):
        payload = valid_payload()
        payload["claims"][1]["agree"] = []
        payload["claims"][1]["dissent"] = [{"model": "Model C", "quote": "nope"}]
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        self.assertEqual(len(data["claims"]), 1)

    def test_duplicate_dissent_from_one_model_counts_once(self):
        payload = valid_payload()
        payload["claims"][1] = {
            "anchor": "founded in the third century BC",
            "agree": ["Model A"],
            "dissent": [
                {"model": "Model C", "quote": ""},
                {"model": "Model C", "quote": "better quote"},
            ],
        }
        data, _ = parse_differences_payload(json.dumps(payload), ANON_MAP)
        claim = data["claims"][1]
        self.assertEqual(claim["agree"], ["OpenAI"])
        self.assertEqual(claim["dissent"], [{"model": "Grok", "quote": "better quote"}])


if __name__ == "__main__":
    unittest.main()
