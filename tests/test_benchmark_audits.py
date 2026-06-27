"""E4-Audits im Pilot-Flow + run_pilot-Orchestrierung (Phase 2.5b).
Alles mit Fake-Transport/Fake-Consensus, ohne HTTP/Keys."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from benchmark.prompt import LETTERS
from benchmark.runner import BenchmarkRunner, _permutation_consistent

QUESTIONS = [
    {"question_id": 1, "question": "Q1?", "options": ["o0", "o1", "o2", "o3"],
     "answer": "B", "answer_index": 1, "category": "math"},
    {"question_id": 2, "question": "Q2?", "options": ["p0", "p1", "p2", "p3"],
     "answer": "C", "answer_index": 2, "category": "law"},
]


def fake_transport_fixed(letter="B"):
    def _execute(request_data, api_key, **kwargs):
        return {"text": f"The answer is ({letter}).", "sources": [],
                "usage": {"prompt": 5, "completion": 2, "total": 7},
                "raw": {}, "status": 200, "latency_ms": 1.0,
                "error": None, "error_code": None}
    return _execute


def fake_consensus_fixed(question, answers, model_sources=None):
    return "The answer is (B)."


class PermutationConsistencyUnitTests(unittest.TestCase):
    def test_position_invariant_answer_is_consistent(self):
        # order[newpos] = oldindex; B (old idx1) landet auf newpos2 -> Buchstabe C.
        order = [2, 0, 1, 3]
        self.assertTrue(_permutation_consistent("B", "C", order))

    def test_position_biased_answer_is_inconsistent(self):
        order = [2, 0, 1, 3]
        self.assertFalse(_permutation_consistent("B", "B", order))

    def test_missing_letter_is_inconclusive(self):
        self.assertIsNone(_permutation_consistent(None, "B", [0, 1, 2, 3]))
        self.assertIsNone(_permutation_consistent("B", None, [0, 1, 2, 3]))


class ConsensusOrderAuditTests(unittest.TestCase):
    def _seed_run(self, run_dir, consensus_fn):
        runner = BenchmarkRunner()
        runner.run(QUESTIONS, run_dir=run_dir, api_keys={},
                   transport_execute=fake_transport_fixed("B"), consensus_fn=consensus_fn)
        return runner

    def test_order_invariant_consensus_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            runner = self._seed_run(run_dir, fake_consensus_fixed)
            audit_res = runner.audit_consensus_order(
                QUESTIONS, run_dir, consensus_fn=fake_consensus_fixed, rng=random.Random(1)
            )
            self.assertEqual(audit_res["total"], 2)
            self.assertEqual(audit_res["stable"], 2)

    def test_order_sensitive_consensus_is_unstable(self):
        # Consensus, dessen Buchstabe von der Reihenfolge abhaengt -> instabil.
        def order_sensitive(question, answers, model_sources=None):
            first = next(iter(answers))
            letter = LETTERS[hash(first) % 4]
            return f"The answer is ({letter})."

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            runner = self._seed_run(run_dir, fake_consensus_fixed)
            audit_res = runner.audit_consensus_order(
                QUESTIONS, run_dir, consensus_fn=order_sensitive, rng=random.Random(1)
            )
            self.assertEqual(audit_res["total"], 2)
            self.assertLess(audit_res["stable"], 2)


class RunPilotTests(unittest.TestCase):
    def test_run_pilot_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            runner = BenchmarkRunner()
            result, audits, summary = runner.run_pilot(
                QUESTIONS, run_dir=run_dir, api_keys={},
                transport_execute=fake_transport_fixed("B"),
                consensus_fn=fake_consensus_fixed,
                permutation_subset=2, rng=random.Random(7),
            )
            self.assertFalse(result.stopped)
            for name in ("calls.jsonl", "manifest.json", "audits.json", "results.json"):
                self.assertTrue((run_dir / name).exists(), f"missing {name}")

            # Audits wurden ausgefuehrt + gespeichert.
            self.assertIn("option_permutation", audits)
            self.assertIn("consensus_order", audits)
            saved = json.loads((run_dir / "audits.json").read_text(encoding="utf-8"))
            self.assertEqual(set(saved), {"option_permutation", "consensus_order"})
            # Permutation lief ueber subset(2) x 6 Modelle.
            self.assertEqual(audits["option_permutation"]["total"], 12)

            # Auswertung vorhanden.
            self.assertEqual(summary["n_questions"], 2)
            self.assertIn("majority_vote", summary["systems"])
            self.assertIn("consensus", summary["systems"])

    def test_run_pilot_skips_audits_when_budget_stops(self):
        big = fake_transport_fixed("B")

        def _execute(request_data, api_key, **kwargs):
            out = big(request_data, api_key)
            out["usage"] = {"prompt": 1_000_000, "completion": 1_000_000, "total": 2_000_000}
            return out

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            runner = BenchmarkRunner()
            result, audits, summary = runner.run_pilot(
                QUESTIONS, run_dir=run_dir, api_keys={},
                transport_execute=_execute, consensus_fn=fake_consensus_fixed, budget=25.0,
            )
            self.assertTrue(result.stopped)
            self.assertIsNone(audits)
            self.assertIsNone(summary)
            self.assertFalse((run_dir / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
