"""§9.3 – Sampling deterministisch (gleiche Seeds -> gleiche IDs), offline;
Pilot- und Final-Sample disjunkt; finales Sample = per_cat x Kategorie.

Die Sampling-Logik braucht kein Parquet-Engine (arbeitet auf plain-Records).
Der Parquet-Lesepfad wird nur getestet, wenn ein Engine + Fixture vorhanden ist.
"""

import importlib.util
import unittest
from pathlib import Path

from benchmark import config, dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mmlu_pro_mini.parquet"

_HAS_PARQUET = bool(
    importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
)


def make_records(n_categories=3, per_category=10):
    records = []
    qid = 1
    for c in range(n_categories):
        category = f"cat_{c}"
        for _ in range(per_category):
            records.append(
                {
                    "question_id": qid,
                    "question": f"Question {qid}?",
                    "options": ["A-opt", "B-opt", "C-opt", "D-opt"],
                    "answer": "B",
                    "answer_index": 1,
                    "category": category,
                }
            )
            qid += 1
    return records


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = make_records()

    def test_pilot_is_deterministic(self):
        a = dataset.sample_pilot(self.records, seed=42, size=5)
        b = dataset.sample_pilot(self.records, seed=42, size=5)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        self.assertEqual(a, sorted(a))

    def test_different_seed_changes_pilot(self):
        a = dataset.sample_pilot(self.records, seed=1, size=5)
        b = dataset.sample_pilot(self.records, seed=2, size=5)
        self.assertNotEqual(a, b)

    def test_final_is_per_category_and_deterministic(self):
        pilot = dataset.sample_pilot(self.records, seed=42, size=5)
        final_a = dataset.sample_final(self.records, set(pilot), seed=7, per_cat=3)
        final_b = dataset.sample_final(self.records, set(pilot), seed=7, per_cat=3)
        self.assertEqual(final_a, final_b)
        # 3 Kategorien x 3 = 9
        self.assertEqual(len(final_a), 9)
        by_cat = {}
        for rec in dataset.records_for_ids(self.records, final_a):
            by_cat[rec["category"]] = by_cat.get(rec["category"], 0) + 1
        self.assertTrue(all(count == 3 for count in by_cat.values()))
        self.assertEqual(len(by_cat), 3)

    def test_pilot_and_final_are_disjoint(self):
        pilot, final = dataset.build_samples(
            self.records, pilot_seed=42, final_seed=7, pilot_size=5, per_cat=3
        )
        self.assertEqual(set(pilot) & set(final), set())

    def test_category_shortfall_raises(self):
        records = make_records(n_categories=2, per_category=4)
        with self.assertRaises(ValueError):
            dataset.sample_final(records, set(), seed=1, per_cat=7)

    def test_records_from_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "question_id": 5,
                    "question": "Q?",
                    "options": ["x", "y", "z"],
                    "answer": "a",
                    "answer_index": 0,
                    "category": "math",
                }
            ]
        )
        records = dataset.records_from_dataframe(df)
        self.assertEqual(records[0]["question_id"], 5)
        self.assertEqual(records[0]["options"], ["x", "y", "z"])
        self.assertEqual(records[0]["answer"], "A")  # normalisiert auf Upper

    @unittest.skipUnless(
        _HAS_PARQUET and FIXTURE.exists(), "needs a parquet engine + committed fixture"
    )
    def test_parquet_fixture_roundtrip(self):
        df = dataset.load_dataframe([FIXTURE])
        records = dataset.records_from_dataframe(df)
        self.assertGreater(len(records), 0)
        self.assertIn("question_id", records[0])


if __name__ == "__main__":
    unittest.main()
