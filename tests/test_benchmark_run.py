"""End-to-end-Test des realen runner.run()-Pfads – **ohne HTTP, ohne echte
API-Keys** (Transport + Consensus injiziert). Deckt ab: JSONL-Append pro Zelle,
Resume ueberspringt nur erfolgreiche Zellen, Fehler werden gespeichert und sind
kontrolliert erneut behandelbar, Budget-Cap stoppt vor dem naechsten Call."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.runner import BenchmarkRunner, cell_key, load_done_keys

QUESTION = {
    "question_id": 101,
    "question": "What is the capital of France?",
    "options": ["London", "Paris", "Berlin", "Rome"],
    "answer": "B",
    "answer_index": 1,
    "category": "geography",
}

# 6 Modelle + Consensus + Synth-allein = 8 Zellen pro Frage.
CELLS_PER_QUESTION = 8


def fake_transport(*, fail_providers=(), prompt_tokens=10, completion_tokens=5):
    """Deterministischer Transport-Mock: jedes Modell 'antwortet' (B), ausser den
    in ``fail_providers`` genannten (HTTP-Fehler)."""

    def _execute(request_data, api_key, **kwargs):
        provider = request_data["provider"]
        if provider in fail_providers:
            return {
                "text": "", "sources": [],
                "usage": {"prompt": 0, "completion": 0, "total": 0},
                "raw": None, "status": 500, "latency_ms": 1.0,
                "error": "boom", "error_code": "provider_http_error",
            }
        return {
            "text": "Paris is the only listed capital of France.\nFINAL_ANSWER: B", "sources": [],
            "usage": {"prompt": prompt_tokens, "completion": completion_tokens,
                      "total": prompt_tokens + completion_tokens},
            "raw": {}, "status": 200, "latency_ms": 1.0,
            "error": None, "error_code": None,
        }

    return _execute


def fake_consensus(question, answers, model_sources=None):
    return "Most candidate answers support Paris.\nFINAL_ANSWER: B"


def read_cells(run_dir):
    path = Path(run_dir) / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RunEndToEndTests(unittest.TestCase):
    def _runner(self):
        return BenchmarkRunner()

    def test_jsonl_append_one_cell_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            result = self._runner().run(
                [QUESTION], run_dir=run_dir, api_keys={},
                transport_execute=fake_transport(), consensus_fn=fake_consensus,
            )
            cells = read_cells(run_dir)
            self.assertEqual(len(cells), CELLS_PER_QUESTION)
            self.assertEqual(result.cells_written, CELLS_PER_QUESTION)
            self.assertEqual(result.cells_failed, 0)
            self.assertFalse(result.stopped)

            roles = sorted(c["role"] for c in cells)
            self.assertEqual(roles.count("model"), 6)
            self.assertEqual(roles.count("consensus"), 1)
            self.assertEqual(roles.count("synth_alone"), 1)

            model_cell = next(c for c in cells if c["role"] == "model")
            self.assertEqual(model_cell["extracted_letter"], "B")
            self.assertTrue(model_cell["correct"])
            self.assertEqual(model_cell["ground_truth"], "B")
            self.assertTrue(model_cell["benchmark_mode"])
            self.assertEqual(model_cell["label_mode"], "anon_shuffled")
            self.assertIn("usage", model_cell)
            self.assertIn("est_cost_usd", model_cell)
            # manifest.json wird mitgeschrieben
            self.assertTrue((run_dir / "manifest.json").exists())

    def test_consensus_receives_benchmark_final_answer_contract(self):
        seen = {}

        def capturing_consensus(question, answers, model_sources=None):
            seen["question"] = question
            return fake_consensus(question, answers, model_sources)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_contract"
            self._runner().run(
                [QUESTION], run_dir=run_dir, api_keys={},
                transport_execute=fake_transport(), consensus_fn=capturing_consensus,
            )

        self.assertIn("Compare the candidate answers", seen["question"])
        self.assertIn("FINAL_ANSWER: X", seen["question"])
        self.assertIn("(B) Paris", seen["question"])

    def test_resume_skips_successful_cells_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run2"
            runner = self._runner()
            runner.run([QUESTION], run_dir=run_dir, api_keys={},
                       transport_execute=fake_transport(), consensus_fn=fake_consensus)
            first = read_cells(run_dir)

            result2 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                                 transport_execute=fake_transport(), consensus_fn=fake_consensus)
            second = read_cells(run_dir)

            self.assertEqual(len(second), len(first))  # keine Duplikate
            self.assertEqual(result2.cells_written, 0)
            self.assertEqual(result2.cells_skipped, CELLS_PER_QUESTION)

    def test_errors_are_stored_and_controllably_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run3"
            runner = self._runner()

            # 1) mistral schlaegt fehl -> Fehlerzelle wird gespeichert.
            r1 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                            transport_execute=fake_transport(fail_providers={"mistral"}),
                            consensus_fn=fake_consensus)
            self.assertEqual(r1.cells_failed, 1)
            cells = read_cells(run_dir)
            mistral_err = [c for c in cells if c["role"] == "model" and c["provider"] == "mistral"]
            self.assertEqual(len(mistral_err), 1)
            self.assertEqual(mistral_err[0]["error_code"], "provider_http_error")
            self.assertIsNone(mistral_err[0]["extracted_letter"])
            self.assertFalse(mistral_err[0]["correct"])
            # mistral gilt NICHT als erledigt
            self.assertNotIn(cell_key(101, "model", "mistral"), load_done_keys(run_dir / "calls.jsonl"))

            # 2) Resume ohne retry_failed: mistral wird NICHT erneut versucht.
            r2 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                            transport_execute=fake_transport(), consensus_fn=fake_consensus,
                            retry_failed=False)
            self.assertEqual(r2.cells_written, 0)
            self.assertNotIn(cell_key(101, "model", "mistral"), load_done_keys(run_dir / "calls.jsonl"))

            # 3) Resume mit retry_failed + jetzt funktionierendem Transport: mistral
            #    wird erneut versucht und gelingt; eine neue Erfolgszeile kommt dazu.
            before = len(read_cells(run_dir))
            r3 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                            transport_execute=fake_transport(), consensus_fn=fake_consensus,
                            retry_failed=True)
            self.assertEqual(r3.cells_written, 1)
            self.assertEqual(len(read_cells(run_dir)), before + 1)
            self.assertIn(cell_key(101, "model", "mistral"), load_done_keys(run_dir / "calls.jsonl"))

    def test_budget_cap_stops_before_first_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run4"
            result = self._runner().run(
                [QUESTION], run_dir=run_dir, api_keys={},
                transport_execute=fake_transport(), consensus_fn=fake_consensus,
                budget=0.0,
            )
            self.assertTrue(result.stopped)
            self.assertEqual(result.cells_written, 0)
            self.assertEqual(read_cells(run_dir), [])
            self.assertIn("budget", result.stop_reason.lower())

    def test_budget_cap_stops_midway_and_is_resumable(self):
        # Grosse Usage -> echte Ist-Kosten akkumulieren; moderates Cap stoppt mitten
        # in der Frage. Anschliessend laeuft der Rest mit groesserem/keinem Cap durch.
        big = fake_transport(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run5"
            runner = self._runner()
            r1 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                            transport_execute=big, consensus_fn=fake_consensus, budget=25.0)
            self.assertTrue(r1.stopped)
            self.assertGreaterEqual(r1.cells_written, 1)
            self.assertLess(r1.cells_written, CELLS_PER_QUESTION)
            self.assertEqual(len(read_cells(run_dir)), r1.cells_written)

            # Resume ohne Cap: Rest wird ergaenzt, keine Duplikate, alles erledigt.
            r2 = runner.run([QUESTION], run_dir=run_dir, api_keys={},
                            transport_execute=fake_transport(), consensus_fn=fake_consensus)
            self.assertFalse(r2.stopped)
            done = load_done_keys(run_dir / "calls.jsonl")
            expected_keys = {cell_key(101, "model", p) for p in
                             ["openai", "mistral", "anthropic", "gemini", "deepseek", "grok"]}
            expected_keys.add(cell_key(101, "consensus", runner.consensus_model))
            expected_keys.add(cell_key(101, "synth_alone", runner.consensus_model))
            self.assertTrue(expected_keys.issubset(done))


if __name__ == "__main__":
    unittest.main()
