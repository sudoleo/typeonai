"""Auswertung calls.jsonl -> results.json (Phase 2.5b). Keine echten Calls."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark import results
from benchmark.runner import append_jsonl


def cell(qid, role, provider, letter, gt, *, correct=None, error=None,
         abstain=False, cost=0.01, latency=100.0, category="math"):
    if correct is None:
        correct = bool(letter) and (letter == gt)
    return {
        "run_id": "t", "question_id": qid, "category": category, "role": role,
        "provider": provider, "internal_model": provider, "api_model": provider,
        "extracted_letter": letter, "ground_truth": gt,
        "correct": correct, "abstain": abstain,
        "usage": {"prompt": 1, "completion": 1, "total": 2},
        "est_cost_usd": cost, "latency_ms": (None if error else latency),
        "error": error, "error_code": ("e" if error else None),
        "parsed_text": "", "request_payload": {},
    }


# Q1 (gt B): alle 6 Modelle B (Einigkeit). Q2 (gt C): C,C,C,A,error,abstain.
def build_calls():
    rows = []
    for p in ["openai", "mistral", "anthropic", "gemini", "deepseek", "grok"]:
        rows.append(cell(1, "model", p, "B", "B"))
    rows.append(cell(1, "consensus", "Gemini", "B", "B", cost=0.05))
    rows.append(cell(1, "synth_alone", "Gemini", "B", "B", cost=0.02))

    rows.append(cell(2, "model", "openai", "C", "C", category="law"))
    rows.append(cell(2, "model", "mistral", "C", "C", category="law"))
    rows.append(cell(2, "model", "anthropic", "C", "C", category="law"))
    rows.append(cell(2, "model", "gemini", "A", "C", category="law"))
    rows.append(cell(2, "model", "deepseek", None, "C", error="boom", category="law"))
    rows.append(cell(2, "model", "grok", None, "C", abstain=True, category="law"))
    rows.append(cell(2, "consensus", "Gemini", "A", "C", cost=0.05, category="law"))
    rows.append(cell(2, "synth_alone", "Gemini", "C", "C", cost=0.02, category="law"))
    return rows


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.agg = results.aggregate(build_calls())
        self.sys = self.agg["systems"]

    def test_question_and_disagreement_counts(self):
        self.assertEqual(self.agg["n_questions"], 2)
        self.assertEqual(self.agg["n_disagreement"], 1)  # nur Q2
        self.assertEqual(self.agg["disagreement_question_ids"], [2])

    def test_single_model_accuracy_overall_and_disagreement(self):
        self.assertEqual(self.sys["model:openai"]["accuracy_overall"], 1.0)
        self.assertEqual(self.sys["model:openai"]["accuracy_disagreement"], 1.0)
        self.assertEqual(self.sys["model:gemini"]["accuracy_overall"], 0.5)
        self.assertEqual(self.sys["model:gemini"]["accuracy_disagreement"], 0.0)

    def test_error_abstain_and_parse_rate(self):
        deepseek = self.sys["model:deepseek"]
        self.assertEqual(deepseek["error"], 1)
        self.assertEqual(deepseek["error_rate"], 0.5)
        self.assertEqual(deepseek["attempted"], 1)  # nur Q1 (Q2 Fehler)
        self.assertEqual(deepseek["parse_rate"], 1.0)

        grok = self.sys["model:grok"]
        self.assertEqual(grok["abstain"], 1)        # Q2 ohne Buchstabe
        self.assertEqual(grok["attempted"], 2)
        self.assertEqual(grok["parse_rate"], 0.5)

    def test_majority_vote(self):
        maj = self.sys["majority_vote"]
        self.assertEqual(maj["accuracy_overall"], 1.0)  # Q1 B, Q2 C beide korrekt
        self.assertEqual(maj["accuracy_disagreement"], 1.0)
        self.assertEqual(maj["no_majority"], 0)

    def test_consensus_and_synth(self):
        cons = self.sys["consensus"]
        self.assertEqual(cons["accuracy_overall"], 0.5)        # Q1 B richtig, Q2 A falsch
        self.assertEqual(cons["accuracy_disagreement"], 0.0)
        self.assertIn("synth_alone", self.sys)
        self.assertEqual(self.sys["synth_alone"]["accuracy_overall"], 1.0)

    def test_totals_costs_and_latency(self):
        totals = self.agg["totals"]
        self.assertGreater(totals["cost_usd"], 0.0)
        self.assertIn("model", totals["by_role"])
        self.assertEqual(totals["errors"], 1)
        self.assertIsNotNone(self.sys["model:openai"]["latency_ms"]["avg"])

    def test_no_majority_bucket(self):
        # Vier Modelle, 2:2 Gleichstand -> no_majority.
        rows = [
            cell(9, "model", "openai", "A", "A"),
            cell(9, "model", "mistral", "A", "A"),
            cell(9, "model", "anthropic", "B", "A"),
            cell(9, "model", "gemini", "B", "A"),
        ]
        agg = results.aggregate(rows)
        self.assertEqual(agg["systems"]["majority_vote"]["no_majority"], 1)
        self.assertEqual(agg["systems"]["majority_vote"]["accuracy_overall"], 0.0)


class WriteResultsTests(unittest.TestCase):
    def test_write_results_dedupes_retry_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "calls.jsonl"
            # Erst Fehlversuch, dann Erfolg fuer dieselbe Zelle -> einmal zaehlen.
            append_jsonl(path, cell(1, "model", "openai", None, "B", error="boom"))
            append_jsonl(path, cell(1, "model", "openai", "B", "B"))
            for p in ["mistral", "anthropic", "gemini", "deepseek", "grok"]:
                append_jsonl(path, cell(1, "model", p, "B", "B"))
            append_jsonl(path, cell(1, "consensus", "Gemini", "B", "B"))

            summary = results.write_results(run_dir)
            self.assertTrue((run_dir / "results.json").exists())
            self.assertEqual(summary["systems"]["model:openai"]["error"], 0)  # Erfolg gewinnt
            self.assertEqual(summary["systems"]["model:openai"]["accuracy_overall"], 1.0)
            self.assertEqual(summary["run_id"], run_dir.name)


if __name__ == "__main__":
    unittest.main()
