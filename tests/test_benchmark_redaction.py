"""_redact_payload darf kein No-op sein: Secrets/Header werden rekursiv entfernt,
und kein Secret-String kann aus Payload in einen Call-Record gelangen."""

import json
import unittest

from benchmark.runner import BenchmarkRunner, _redact_payload

QUESTION = {
    "question_id": 1,
    "question": "Q?",
    "options": ["a", "b", "c", "d"],
    "answer": "B",
    "answer_index": 1,
    "category": "math",
}


class RedactionTests(unittest.TestCase):
    def test_redacts_secret_fields_recursively(self):
        payload = {
            "model": "gpt-5.5",
            "Authorization": "Bearer sk-secret1",
            "api_key": "sk-secret2",
            "nested": {"x-api-key": "sk-secret3", "ok": "keep-me"},
            "list": [{"token": "sk-secret4"}, {"bearer": "sk-secret5"}],
            "headers": {"Authorization": "sk-secret6"},
        }
        red = _redact_payload(payload)
        blob = json.dumps(red)
        for secret in (f"sk-secret{i}" for i in range(1, 7)):
            self.assertNotIn(secret, blob)
        # Nicht-Secrets bleiben erhalten
        self.assertEqual(red["model"], "gpt-5.5")
        self.assertEqual(red["nested"]["ok"], "keep-me")
        self.assertEqual(red["headers"], "[REDACTED]")

    def test_original_payload_is_not_mutated(self):
        payload = {"api_key": "sk-orig", "model": "m"}
        _redact_payload(payload)
        self.assertEqual(payload["api_key"], "sk-orig")  # Original unangetastet

    def test_cell_record_cannot_leak_secret(self):
        runner = BenchmarkRunner()
        payload = {"model": "gpt-5.5", "Authorization": "Bearer sk-LEAK", "input": "q"}
        outcome = {
            "text": "The answer is (B).",
            "usage": {"prompt": 1, "completion": 1, "total": 2},
            "latency_ms": 1.0, "status": 200, "error": None, "error_code": None,
        }
        cell = runner._make_cell_record(
            run_id="r", qrecord=QUESTION, role="model", provider="openai",
            internal_model="x", api_model="gpt-5.5", user_prompt="q",
            payload=payload, outcome=outcome,
        )
        self.assertNotIn("sk-LEAK", json.dumps(cell))
        self.assertEqual(cell["request_payload"]["Authorization"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
