"""§9.4 – Resume-Skip-Logik, Budget-Estimator-Stopp, Dry-Run-Audit."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark import audit, config
from benchmark.runner import (
    BenchmarkRunner,
    cell_key,
    load_done_keys,
    should_stop_for_budget,
)

ROWS = [
    {
        "question_id": 101,
        "question": "What is the capital of France?",
        "options": ["London", "Paris", "Berlin", "Rome"],
        "answer": "B",
        "answer_index": 1,
        "category": "geography",
    }
]


class ResumeTests(unittest.TestCase):
    def test_load_done_keys_skips_errors_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.jsonl"
            lines = [
                json.dumps({"question_id": 1, "role": "model", "provider": "openai", "error": None}),
                "",
                json.dumps({"question_id": 1, "role": "model", "provider": "mistral", "error": "boom"}),
                json.dumps({"question_id": 1, "role": "consensus", "provider": "Gemini", "error": None}),
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            done = load_done_keys(path)
            self.assertIn(cell_key(1, "model", "openai"), done)
            self.assertIn(cell_key(1, "consensus", "Gemini"), done)
            # fehlerhafte Zelle gilt als nicht erledigt
            self.assertNotIn(cell_key(1, "model", "mistral"), done)

    def test_load_done_keys_missing_file(self):
        self.assertEqual(load_done_keys(Path("does-not-exist.jsonl")), set())


class BudgetTests(unittest.TestCase):
    def test_no_cap_never_stops(self):
        self.assertFalse(should_stop_for_budget(spent=999.0, next_estimate=999.0, cap=None))

    def test_stops_when_next_cell_would_exceed(self):
        self.assertTrue(should_stop_for_budget(spent=0.95, next_estimate=0.10, cap=1.0))

    def test_continues_when_within_cap(self):
        self.assertFalse(should_stop_for_budget(spent=0.5, next_estimate=0.10, cap=1.0))


class DryRunTests(unittest.TestCase):
    def test_dry_run_audits_all_model_payloads_and_projects_cost(self):
        report = BenchmarkRunner().dry_run(ROWS)
        self.assertEqual(report.model_cells, 6)
        self.assertEqual(report.consensus_cells, 1)
        self.assertEqual(report.audited_payloads, 6)
        self.assertGreater(report.projected_cost_usd, 0.0)
        self.assertEqual(report.missing_pricing, set())

    def test_dry_run_audit_catches_injected_web_tool(self):
        # Simuliert eine Regression: build_provider_payload haette ein Tool injiziert.
        runner = BenchmarkRunner()
        original = runner.build_model_request

        def leaky(model, user_prompt):
            request = original(model, user_prompt)
            request["payload"]["tools"] = [{"type": "web_search"}]
            return request

        runner.build_model_request = leaky
        with self.assertRaises(AssertionError):
            runner.dry_run(ROWS)

    def test_assert_no_web_tools_passes_for_clean_payload(self):
        audit.assert_no_web_tools({"model": "x", "messages": [{"role": "user", "content": "hi"}]})


class ManifestModelConfigTests(unittest.TestCase):
    def test_manifest_records_regular_model_matrix_and_effective_settings(self):
        runner = BenchmarkRunner(sample_role="smoke", sample_manifest="mmlu_pro_smoke_v1.json")
        manifest = runner.build_manifest("smoke")
        models = {row["provider"]: row for row in manifest["models"]}

        self.assertEqual(models["openai"]["internal_id"], "gpt-5.5")
        self.assertEqual(models["openai"]["resolved_api_model"], "gpt-5.5")
        self.assertIsNone(models["openai"]["reasoning_settings"])
        self.assertFalse(models["openai"]["alias_status"]["internal_alias"])

        self.assertEqual(models["mistral"]["internal_id"], config.cfg.MISTRAL_PRO_MODEL)
        self.assertEqual(models["mistral"]["resolved_api_model"], config.cfg.MISTRAL_PRO_MODEL)
        self.assertEqual(models["mistral"]["reasoning_settings"], {"reasoning_effort": "high"})
        self.assertFalse(models["mistral"]["alias_status"]["latest_alias"])

        self.assertEqual(models["anthropic"]["internal_id"], config.cfg.ANTHROPIC_PRO_MODEL)
        self.assertIsNone(models["anthropic"]["reasoning_settings"])
        self.assertEqual(models["gemini"]["internal_id"], config.BENCHMARK_GEMINI_MODEL)
        self.assertIsNone(models["gemini"]["reasoning_settings"])
        self.assertFalse(models["gemini"]["alias_status"]["preview"])
        self.assertEqual(models["grok"]["internal_id"], "grok-4.3")
        self.assertIsNone(models["grok"]["reasoning_settings"])

        self.assertEqual(manifest["sample_role"], "smoke")
        self.assertEqual(manifest["consensus"]["internal_id"], config.BENCHMARK_GEMINI_MODEL)
        self.assertEqual(manifest["consensus"]["output_token_limit"], manifest["consensus_output_token_limit"])
        self.assertIsNone(manifest["consensus"]["temperature"])
        self.assertEqual(manifest["synth_alone"]["internal_id"], manifest["consensus"]["internal_id"])
        self.assertEqual(manifest["synth_alone"]["reasoning_settings"], manifest["consensus"]["reasoning_settings"])
        self.assertEqual(manifest["synth_alone"]["temperature"], manifest["consensus"]["temperature"])


if __name__ == "__main__":
    unittest.main()
