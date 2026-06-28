"""report_reader: stdlib-only Lesepfad fuer die Admin-Benchmark-Visualisierung."""

import json
import sys
import unittest
from pathlib import Path

from benchmark import report_reader


class SafeRunDirTests(unittest.TestCase):
    def test_rejects_traversal_and_missing(self):
        self.assertIsNone(report_reader._safe_run_dir("../secrets"))
        self.assertIsNone(report_reader._safe_run_dir("a/b"))
        self.assertIsNone(report_reader._safe_run_dir(".."))
        self.assertIsNone(report_reader._safe_run_dir(""))
        self.assertIsNone(report_reader.get_run("does-not-exist-xyz"))


class ReaderWithTempRunTests(unittest.TestCase):
    def setUp(self):
        # Isolierter Runs-Ordner, damit der Test nicht von committeten Runs abhaengt.
        self.tmp = Path(self.id())  # placeholder, overwritten below
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.runs = Path(self._dir.name)
        self._orig = report_reader.RUNS_DIR
        report_reader.RUNS_DIR = self.runs
        run = self.runs / "demo_run"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({
            "sample_role": "pilot", "label_mode": "names",
            "consensus_model": "gemini-x", "created": "2026-06-28T10:00:00+00:00",
            "output_token_limit": 12288,
        }), encoding="utf-8")
        (run / "results.json").write_text(json.dumps({
            "n_questions": 1, "n_disagreement": 0,
            "systems": {"consensus": {"accuracy_overall": 1.0}},
            "totals": {"cost_usd": 0.12, "cells": 8, "errors": 0},
        }), encoding="utf-8")
        (run / "audits.json").write_text(json.dumps({
            "option_permutation": {"consistent": 6, "conclusive": 6, "total": 6},
        }), encoding="utf-8")
        rows = [
            {"question_id": 1, "category": "math", "ground_truth": "C", "role": "model",
             "provider": "openai", "extracted_letter": "C", "correct": True, "abstain": False, "error": None},
            {"question_id": 1, "category": "math", "ground_truth": "C", "role": "model",
             "provider": "grok", "extracted_letter": "A", "correct": False, "abstain": False, "error": None},
            # Fehlversuch + spaeterer Erfolg derselben Zelle -> Erfolg gewinnt.
            {"question_id": 1, "category": "math", "ground_truth": "C", "role": "consensus",
             "provider": "gemini-x", "extracted_letter": None, "correct": False, "abstain": False, "error": "boom"},
            {"question_id": 1, "category": "math", "ground_truth": "C", "role": "consensus",
             "provider": "gemini-x", "extracted_letter": "C", "correct": True, "abstain": False, "error": None},
        ]
        (run / "calls.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def tearDown(self):
        report_reader.RUNS_DIR = self._orig
        self._dir.cleanup()

    def test_list_runs(self):
        runs = report_reader.list_runs()
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r["run_id"], "demo_run")
        self.assertEqual(r["sample_role"], "pilot")
        self.assertEqual(r["total_cost_usd"], 0.12)
        self.assertTrue(r["has_results"] and r["has_audits"])

    def test_get_run_matrix_and_dedupe(self):
        run = report_reader.get_run("demo_run")
        self.assertEqual(run["manifest"]["output_token_limit"], 12288)
        self.assertEqual(len(run["questions"]), 1)
        q = run["questions"][0]
        self.assertEqual(q["ground_truth"], "C")
        self.assertEqual(q["models"]["openai"]["letter"], "C")
        self.assertTrue(q["disagreement"])  # C vs A
        # Majority mit Gleichstand (C,A) -> no_majority.
        self.assertEqual(q["majority"]["letter"], "no_majority")
        self.assertFalse(q["majority"]["correct"])
        # Consensus: spaeterer Erfolg ueberschreibt Fehlversuch.
        self.assertEqual(q["consensus"]["letter"], "C")
        self.assertTrue(q["consensus"]["correct"])

    def test_reader_does_not_import_llm_engines(self):
        # Frischer Interpreter: report_reader darf den schweren LLM-Importgraphen
        # nicht ziehen (Admin-Lesepfad bleibt isoliert vom Ausfuehrungspfad).
        import subprocess
        code = (
            "import sys; from benchmark import report_reader; report_reader.list_runs(); "
            "assert 'app.services.llm.engines' not in sys.modules; "
            "assert 'app.services.llm.consensus_engine' not in sys.modules; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(report_reader.REPO_ROOT), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
