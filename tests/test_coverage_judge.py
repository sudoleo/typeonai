"""Der Coverage-Judge belegt JEDEN Konsens-Satz - und darf keinen verschweigen.

Der ganze Sinn dieses zweiten Judges ist, dass Auslassen unmoeglich wird. Die
Tests hier pruefen deshalb vor allem die Zwangsmechanik: Schema, verbindliche
ID-Liste, serverseitige Vollstaendigkeitspruefung, gezielte Nachforderung - und
dass eine duenn belegte Aussage sichtbar bleibt, statt zu verschwinden.
"""

import json
import unittest
from unittest import mock

import app.core.config as cfg
from app.services.llm import coverage_judge as coverage
from app.services.llm.consensus_engine import (
    _build_judge_context,
    _coverage_attempts,
    _coverage_claims,
    _run_coverage_judge,
    query_differences,
)
from app.services.llm.consensus_scoring import compute_agreement_score

ALL_KEYS = {"OpenRouter": "sk-or"}

CONSENSUS = (
    "The tower stands 330 metres tall. It was completed in 1889. "
    "That is the whole story."
)


def judge_context(consensus=CONSENSUS):
    return _build_judge_context(
        {"openai": "answer one about the tower", "mistral": "answer two about the tower"},
        consensus,
        [],
        "",
    )


def coverage_payload(context, stances_by_id, classifications=None, quotes=None):
    """Ein vollstaendiges Coverage-JSON fuer die Saetze eines Kontexts."""
    classifications = classifications or {}
    quotes = quotes or {}
    sentences = []
    for number in range(1, len(context.sentences) + 1):
        key = coverage.sentence_id(number)
        stances = stances_by_id.get(key) or {
            label: "supports" for label in context.labels
        }
        sentences.append({
            "id": key,
            "classification": classifications.get(key, "claim"),
            "models": stances,
            "counter_quotes": [
                {"model": label, "quote": quote}
                for label, quote in (quotes.get(key) or {}).items()
            ],
        })
    return json.dumps({"sentences": sentences})


class CoverageSchemaTests(unittest.TestCase):
    def test_schema_is_strict_mode_compatible(self):
        def assert_strict(schema):
            if not isinstance(schema, dict):
                return
            if schema.get("type") == "object":
                self.assertEqual(
                    set(schema.get("required", [])), set(schema.get("properties", {}))
                )
                self.assertIs(schema.get("additionalProperties"), False)
            for value in schema.values():
                if isinstance(value, dict):
                    assert_strict(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_strict(item)

        assert_strict(coverage.build_coverage_schema(["Model A", "Model B"], ["s1", "s2"]))

    def test_every_model_is_a_required_property(self):
        """Eine LISTE haette dem Judge freigestellt, Modelle wegzulassen. Als
        Pflichtfelder erzwingt schon das Schema eine Aussage je Modell."""
        schema = coverage.build_coverage_schema(["Model A", "Model B", "Model C"], ["s1"])
        models = schema["properties"]["sentences"]["items"]["properties"]["models"]
        self.assertEqual(models["required"], ["Model A", "Model B", "Model C"])
        self.assertEqual(
            models["properties"]["Model B"]["enum"], list(coverage.STANCES)
        )

    def test_sentence_ids_are_an_enum(self):
        schema = coverage.build_coverage_schema(["Model A"], ["s1", "s2"])
        item = schema["properties"]["sentences"]["items"]
        self.assertEqual(item["properties"]["id"]["enum"], ["s1", "s2"])


class CoveragePromptTests(unittest.TestCase):
    def test_prompt_carries_the_binding_id_list(self):
        prompt = coverage.build_coverage_prompt(
            labels=["Model A", "Model B"],
            responses_text="- Model A: one\n- Model B: two",
            numbered_answer="[1] One. [2] Two.",
            ids=["s1", "s2"],
        )
        self.assertIn("Binding list of sentence ids", prompt)
        self.assertIn('["s1", "s2"]', prompt)
        self.assertIn("all 2 of them", prompt)
        # Der haeufigste Fehlschluss eines Klassifikators: aus dem eigenen
        # Wissen ergaenzen, was die Antwort gar nicht sagt.
        self.assertIn("never fill a gap from your own knowledge", prompt)

    def test_repair_prompt_asks_only_for_the_missing_ids(self):
        prompt = coverage.build_coverage_prompt(
            labels=["Model A"],
            responses_text="- Model A: one",
            numbered_answer="[1] One. [2] Two. [3] Three.",
            ids=["s2"],
            missing_only=True,
        )
        self.assertIn("EXACTLY the sentence IDs listed below", prompt)
        self.assertIn('["s2"]', prompt)


class CoverageParsingTests(unittest.TestCase):
    LABELS = ["Model A", "Model B"]

    def parse(self, payload, ids=("s1", "s2")):
        return coverage.parse_coverage_payload(payload, self.LABELS, list(ids))

    def test_unmentioned_model_counts_as_not_addressed(self):
        """Die konservative Lesart: ein Modell, das der Judge weglaesst, hat
        sich nicht geaeussert - es hat nicht zugestimmt."""
        parsed = self.parse(json.dumps({"sentences": [{
            "id": "s1", "classification": "claim",
            "models": {"Model A": "supports"}, "counter_quotes": [],
        }]}))
        self.assertEqual(parsed["s1"]["models"]["Model B"], "not_addressed")

    def test_unknown_stance_and_classification_are_coerced(self):
        parsed = self.parse(json.dumps({"sentences": [{
            "id": "s1", "classification": "probably",
            "models": {"Model A": "agrees", "Model B": "contradicts"},
            "counter_quotes": [],
        }]}))
        # Unbekannte Klassifikation -> "claim": lieber einen Satz zu viel
        # belegen als einen stillschweigend fallen lassen.
        self.assertEqual(parsed["s1"]["classification"], "claim")
        self.assertEqual(parsed["s1"]["models"]["Model A"], "unclear")
        self.assertEqual(parsed["s1"]["models"]["Model B"], "contradicts")

    def test_ids_outside_the_binding_list_are_dropped(self):
        parsed = self.parse(json.dumps({"sentences": [
            {"id": "s9", "classification": "claim", "models": {}, "counter_quotes": []},
            {"id": "s1", "classification": "claim", "models": {}, "counter_quotes": []},
        ]}))
        self.assertEqual(list(parsed), ["s1"])

    def test_model_list_form_is_accepted(self):
        """Nicht im Schema, aber ein haeufiger Freiheitsgrad schwaecherer
        Modelle."""
        parsed = self.parse(json.dumps({"sentences": [{
            "id": "s1", "classification": "claim",
            "models": [{"model": "Model A", "stance": "contradicts"}],
            "counter_quotes": [],
        }]}))
        self.assertEqual(parsed["s1"]["models"]["Model A"], "contradicts")

    def test_structurally_broken_output_is_none(self):
        self.assertIsNone(self.parse("not json at all"))
        self.assertIsNone(self.parse(json.dumps({"sentences": "s1"})))

    def test_missing_ids_are_reported(self):
        parsed = self.parse(json.dumps({"sentences": [{
            "id": "s1", "classification": "claim", "models": {}, "counter_quotes": [],
        }]}))
        self.assertEqual(coverage.missing_sentence_ids(parsed, ["s1", "s2"]), ["s2"])


class CoverageClaimTests(unittest.TestCase):
    def setUp(self):
        self.context = judge_context()
        self.labels = list(self.context.labels)

    def test_stances_become_agree_and_dissent(self):
        result = coverage.parse_coverage_payload(
            coverage_payload(
                self.context,
                {"s2": {self.labels[0]: "supports", self.labels[1]: "contradicts"}},
                quotes={"s2": {self.labels[1]: "completed in 1887"}},
            ),
            self.labels,
            coverage.sentence_ids(self.context.sentences),
        )
        claims = _coverage_claims(result, self.context)
        by_sentence = {claim["sentence_id"]: claim for claim in claims}
        disputed = by_sentence[2]
        self.assertEqual(len(disputed["agree"]), 1)
        self.assertEqual(disputed["dissent"][0]["quote"], "completed in 1887")
        self.assertEqual(disputed["coverage"], "split")
        self.assertEqual(by_sentence[1]["coverage"], "supported")

    def test_a_non_claim_sentence_is_skipped_on_purpose(self):
        """Der EINZIGE legitime Weg, einen Satz zu ueberspringen: eine
        ausgesprochene Klassifikation, kein Verschwinden."""
        result = coverage.parse_coverage_payload(
            coverage_payload(self.context, {}, classifications={"s3": "not_a_claim"}),
            self.labels,
            coverage.sentence_ids(self.context.sentences),
        )
        claims = _coverage_claims(result, self.context)
        self.assertEqual([claim["sentence_id"] for claim in claims], [1, 2])

    def test_an_id_the_judge_never_answered_stays_visible_as_thin(self):
        """Nach der Nachforderung uebrige Luecken werden neutral behandelt -
        grau statt weg. Ein unmarkierter Satz saehe aus wie ungeprueft."""
        claims = _coverage_claims({}, self.context)
        self.assertEqual(len(claims), len(self.context.sentences))
        self.assertTrue(all(claim["coverage"] == "thin" for claim in claims))
        self.assertTrue(all(claim["agree"] == [] for claim in claims))

    def test_a_single_voice_is_thin_not_supported(self):
        result = coverage.parse_coverage_payload(
            coverage_payload(self.context, {"s1": {
                self.labels[0]: "supports", self.labels[1]: "not_addressed",
            }}),
            self.labels,
            coverage.sentence_ids(self.context.sentences),
        )
        claims = {c["sentence_id"]: c for c in _coverage_claims(result, self.context)}
        self.assertEqual(claims[1]["coverage"], "thin")
        self.assertEqual(len(claims[1]["agree"]), 1)

    def test_anchor_is_an_exact_excerpt_of_the_consensus(self):
        claims = _coverage_claims({}, self.context)
        for claim in claims:
            self.assertIn(claim["anchor"], CONSENSUS)


class CoverageScoringTests(unittest.TestCase):
    def test_thin_claims_do_not_lift_the_agreement_score(self):
        """Eine einzelne Stimme ist 100 % Zustimmung der Stimmen, die es gibt -
        und trotzdem kein Beleg. Sie darf den Score nicht heben."""
        data = {
            "claims": [
                {"agree": ["A", "B"], "dissent": [{"model": "C"}], "coverage": "split"},
                {"agree": ["A"], "dissent": [], "coverage": "thin"},
            ],
            "differences": [],
            "models_compared": ["A", "B", "C"],
        }
        scored = compute_agreement_score(data)
        self.assertEqual(scored["scored_claims"], 1)
        self.assertEqual(scored["thin_claims"], 1)

        without_thin = compute_agreement_score({**data, "claims": data["claims"][:1]})
        self.assertEqual(scored["score"], without_thin["score"])


class CoverageJudgePolicyTests(unittest.TestCase):
    def test_coverage_judge_stays_on_the_cheap_standard_tier(self):
        """Die Aufgabe ist kontrollierte Klassifikation, kein Denken. Das
        Pro-Modell bleibt dem Differences-Judge vorbehalten - auch wenn die
        Engine selbst eine Pro-Engine ist."""
        attempts = _coverage_attempts("OpenAI-Pro", ALL_KEYS)
        (provider, api_model, _), is_retry = attempts[0]
        self.assertEqual(provider, "gemini")
        self.assertEqual(
            api_model, cfg.openrouter_model_id(cfg.GEMINI_FLASH_MODEL, "gemini")
        )
        self.assertFalse(is_retry)

    def test_coverage_judge_avoids_the_consensus_family(self):
        attempts = _coverage_attempts("Gemini", ALL_KEYS)
        self.assertTrue(all(engine[0] != "gemini" for engine, _ in attempts))

    def test_invalid_engine_has_no_attempts(self):
        self.assertIsNone(_coverage_attempts("Nonexistent", ALL_KEYS))


class CoverageRunTests(unittest.TestCase):
    def setUp(self):
        self.context = judge_context()
        self.ids = coverage.sentence_ids(self.context.sentences)

    def test_missing_ids_trigger_exactly_one_targeted_repair_call(self):
        first = json.dumps({"sentences": [{
            "id": "s1", "classification": "claim",
            "models": {label: "supports" for label in self.context.labels},
            "counter_quotes": [],
        }]})
        repair = json.dumps({"sentences": [
            {
                "id": key, "classification": "claim",
                "models": {label: "supports" for label in self.context.labels},
                "counter_quotes": [],
            }
            for key in ("s2", "s3")
        ]})
        prompts = []

        def engine(provider, api_model, model_ref, api_keys, **kwargs):
            prompts.append(kwargs["prompt"])
            return repair if len(prompts) > 1 else first

        with mock.patch(
            "app.services.llm.consensus_engine._call_engine_text", side_effect=engine
        ):
            result, meta = _run_coverage_judge(self.context, ALL_KEYS, "OpenAI")

        self.assertEqual(len(prompts), 2)
        self.assertIn("EXACTLY the sentence IDs listed below", prompts[1])
        self.assertIn('["s2", "s3"]', prompts[1])
        self.assertEqual(sorted(result), ["s1", "s2", "s3"])
        self.assertEqual(meta["missing"], 0)
        self.assertEqual(meta["repaired"], 2)

    def test_an_unfixable_gap_is_reported_instead_of_hidden(self):
        payload = json.dumps({"sentences": [{
            "id": "s1", "classification": "claim",
            "models": {label: "supports" for label in self.context.labels},
            "counter_quotes": [],
        }]})
        with mock.patch(
            "app.services.llm.consensus_engine._call_engine_text",
            return_value=payload,
        ):
            _result, meta = _run_coverage_judge(self.context, ALL_KEYS, "OpenAI")
        self.assertEqual(meta["sentences"], 3)
        self.assertEqual(meta["covered"], 1)
        self.assertEqual(meta["missing"], 2)

    def test_a_failing_coverage_judge_never_breaks_the_run(self):
        with mock.patch(
            "app.services.llm.consensus_engine._call_engine_text",
            side_effect=RuntimeError("503"),
        ):
            result, meta = _run_coverage_judge(self.context, ALL_KEYS, "OpenAI")
        self.assertIsNone(result)
        self.assertIsNone(meta)


class CoverageIntegrationTests(unittest.TestCase):
    """Beide Judges zusammen: der Differences-Call liefert die Widersprueche,
    der Coverage-Call die Belegliste - und der Server legt sie zusammen."""

    def run_pair(self, consensus=CONSENSUS):
        differences_payload = json.dumps({
            "differences": [{
                "claim": "The completion year is disputed.",
                "s": 2,
                "type": "contradiction",
                "severity": "major",
                "positions": [
                    {"stance": "1889", "models": ["Model A"], "quote": "answer one"},
                    {"stance": "1887", "models": ["Model B"], "quote": "answer two"},
                ],
                "verify": "Check the year.",
            }],
            "best_model": "Model A",
        })

        def engine(provider, api_model, model_ref, api_keys, **kwargs):
            prompt = kwargs["prompt"]
            if '"counter_quotes"' in prompt:
                labels = [
                    label for label in ("Model A", "Model B", "Model C")
                    if f"- {label}:" in prompt
                ]
                sentences = []
                for index, key in enumerate(
                    coverage.sentence_ids(range(prompt.count("] ")))
                ):
                    sentences.append({
                        "id": key,
                        "classification": "claim",
                        "models": {label: "supports" for label in labels},
                        "counter_quotes": [],
                    })
                return json.dumps({"sentences": sentences})
            return differences_payload

        with mock.patch(
            "app.services.llm.consensus_engine._call_engine_text", side_effect=engine
        ):
            return query_differences(
                {"openai": "answer one", "mistral": "answer two"},
                consensus, ALL_KEYS,
                differences_model="OpenAI",
                excluded_models=[],
            )

    def test_claims_come_from_the_coverage_judge(self):
        _text, data = self.run_pair()
        self.assertIsNotNone(data)
        # Jeder pruefbare Satz der Konsensantwort ist belegt - nicht nur die
        # drei bis sechs, die der alte Differences-Judge fuer wichtig hielt.
        self.assertEqual(len(data["claims"]), 3)
        self.assertEqual(
            [claim["sentence_id"] for claim in data["claims"]], [1, 2, 3]
        )
        self.assertTrue(all(c["coverage"] == "supported" for c in data["claims"]))
        # Der Differences-Judge behaelt seinen eigenen Befund.
        self.assertEqual(len(data["differences"]), 1)
        self.assertEqual(data["differences"][0]["sentence_id"], 2)

    def test_both_judges_are_reported_separately(self):
        _text, data = self.run_pair()
        self.assertIn("differences", data["judges"])
        self.assertIn("coverage", data["judges"])
        self.assertEqual(data["judges"]["coverage"]["tier"], "standard")
        self.assertEqual(data["judges"]["coverage"]["covered"], 3)

    def test_the_credibility_sentence_follows_the_recomputed_score(self):
        """Der Freitext haengt am Agreement-Score. Der Score aendert sich durch
        die Coverage-Claims - der Satz muss danach neu gebaut werden, sonst
        widersprechen sich Text und Zahl."""
        text, data = self.run_pair()
        self.assertEqual(data["agreement"]["scored_claims"], 3)
        self.assertIn(data["agreement"]["level"], text.split("**")[1::2])


if __name__ == "__main__":
    unittest.main()
