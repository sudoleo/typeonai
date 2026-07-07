"""Tests für die anonyme Differences-Telemetrie (differences_stats).

Kernvertrag: Das Statistik-Dokument enthält ausschließlich Zähl-/Strukturdaten
und Modell-Metadaten — niemals Frage-/Antwort-/Claim-Texte, Zitate oder
nutzerbezogene Identifikatoren (DSGVO-Anonymität, siehe Modul-Docstring).
"""

import unittest

from app.services.differences_stats import (
    DIFFERENCES_STATS_SCHEMA_VERSION,
    build_differences_stats_doc,
)


def make_differences_data():
    return {
        "claims": [
            {"anchor": "The sky is blue", "agree": ["OpenAI", "Gemini", "Grok"], "dissent": ["Mistral"]},
            {"anchor": "Water boils at 100C", "agree": ["OpenAI", "Mistral"], "dissent": []},
        ],
        "differences": [
            {
                "claim": "Disagreement about the boiling point",
                "type": "contradiction",
                "severity": "major",
                "positions": [
                    {"stance": "It is 100C", "models": ["OpenAI", "Gemini"], "quote": "boils at 100"},
                    {"stance": "It depends on pressure", "models": ["Mistral"], "quote": "depends on"},
                ],
                "verify": "Check a physics source",
            },
            {
                "claim": "Different emphasis on caveats",
                "type": "emphasis",
                "severity": "",
                "positions": [
                    {"stance": "Mentions caveats", "models": ["Grok"], "quote": ""},
                ],
                "verify": "",
            },
        ],
        "best_model": "OpenAI",
        "models_compared": ["Gemini", "Grok", "Mistral", "OpenAI"],
        "agreement": {
            "score": 55,
            "level": "partially",
            "model_count": 4,
            "major_contradictions": 1,
            "minor_contradictions": 0,
            "emphases": 1,
        },
    }


class BuildDifferencesStatsDocTests(unittest.TestCase):
    def test_none_for_missing_data(self):
        self.assertIsNone(build_differences_stats_doc(None))
        self.assertIsNone(build_differences_stats_doc("not a dict"))

    def test_counts_and_metadata(self):
        doc = build_differences_stats_doc(
            make_differences_data(),
            consensus_model="Gemini-Pro",
            model_labels={"OpenAI": "GPT-5.4 mini", "Gemini": "Gemini 3.1 Flash"},
            excluded_count=2,
            is_pro_user=True,
            used_own_keys=False,
            question_word_count=17,
        )

        self.assertEqual(doc["schema_version"], DIFFERENCES_STATS_SCHEMA_VERSION)
        self.assertEqual(doc["consensus_model"], "Gemini-Pro")
        self.assertEqual(doc["model_count"], 4)
        self.assertEqual(doc["models_compared"], ["Gemini", "Grok", "Mistral", "OpenAI"])
        self.assertEqual(doc["best_model"], "OpenAI")
        self.assertEqual(doc["excluded_count"], 2)
        self.assertTrue(doc["is_pro_user"])
        self.assertFalse(doc["used_own_keys"])
        self.assertEqual(doc["question_word_count"], 17)
        self.assertEqual(doc["agreement"]["score"], 55)
        self.assertEqual(doc["agreement"]["level"], "partially")

        # Claims: nur Zähler
        self.assertEqual(doc["claims"], [
            {"agree": 3, "dissent": 1},
            {"agree": 2, "dissent": 0},
        ])

        # Differences: Typ/Severity/Positionsstruktur, Modelle als Provider-Gruppen
        self.assertEqual(doc["differences"][0]["type"], "contradiction")
        self.assertEqual(doc["differences"][0]["severity"], "major")
        self.assertEqual(doc["differences"][0]["position_count"], 2)
        self.assertEqual(doc["differences"][0]["positions"], [
            {"models": ["Gemini", "OpenAI"]},
            {"models": ["Mistral"]},
        ])
        self.assertEqual(doc["differences"][1]["type"], "emphasis")

        # Modell-Labels laufen durch sanitize_model_labels (nur beteiligte Provider)
        self.assertEqual(doc["model_ids"], {"OpenAI": "GPT-5.4 mini", "Gemini": "Gemini 3.1 Flash"})

    def test_firestore_compatible_no_nested_arrays(self):
        """Firestore lehnt Arrays direkt in Arrays ab (400 invalid nested
        entity) — das Dokument darf nirgends list-in-list enthalten."""
        def assert_no_nested_arrays(value, path="doc"):
            if isinstance(value, dict):
                for k, v in value.items():
                    assert_no_nested_arrays(v, f"{path}.{k}")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    self.assertNotIsInstance(
                        item, (list, tuple),
                        f"Verschachteltes Array bei {path}[{i}] — Firestore-Write würde fehlschlagen",
                    )
                    assert_no_nested_arrays(item, f"{path}[{i}]")

        assert_no_nested_arrays(build_differences_stats_doc(make_differences_data()))

    def test_no_content_leaks_into_doc(self):
        """Kein Text aus Frage/Claims/Stances/Quotes darf im Dokument landen."""
        doc = build_differences_stats_doc(make_differences_data())
        flat = repr(doc)
        for forbidden in (
            "The sky is blue",
            "Water boils",
            "Disagreement about",
            "It is 100C",
            "boils at 100",
            "Mentions caveats",
            "Check a physics source",
        ):
            self.assertNotIn(forbidden, flat)


if __name__ == "__main__":
    unittest.main()
