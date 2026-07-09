"""Unit-Tests fuer die Consensus-Engine: anonymisierter Prompt-Builder,
OpenAI-Reasoning-Erkennung (Token-Param/Temperatur) und Fallback-Provider."""

import re
import unittest
from unittest import mock

import app.core.config as cfg
from app.services.llm.consensus_engine import (
    CONSENSUS_TEMPERATURE,
    _build_consensus_prompt,
    _effective_temperature,
    _openai_token_param,
    query_consensus,
    stream_consensus,
)

REAL_MODEL_NAMES = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]

ANSWERS = {
    "answer_openai": "first candidate text",
    "answer_mistral": "second candidate text",
    "answer_claude": "third candidate text",
    "answer_gemini": "fourth candidate text",
    "answer_deepseek": "fifth candidate text",
    "answer_grok": "sixth candidate text",
}


def build_prompt(excluded_models=None, model_sources=None, shuffle=True, **overrides):
    answers = dict(ANSWERS)
    answers.update(overrides)
    return _build_consensus_prompt(
        "What is the answer?",
        answers["answer_openai"],
        answers["answer_mistral"],
        answers["answer_claude"],
        answers["answer_gemini"],
        answers["answer_deepseek"],
        answers["answer_grok"],
        excluded_models or [],
        model_sources=model_sources,
        shuffle=shuffle,
    )


class ConsensusPromptAnonymizationTests(unittest.TestCase):
    def test_prompt_contains_no_real_model_names(self):
        prompt = build_prompt()
        for name in REAL_MODEL_NAMES:
            self.assertNotIn(name, prompt)

    def test_all_answers_appear_under_contiguous_expert_labels(self):
        prompt = build_prompt()
        for text in ANSWERS.values():
            self.assertEqual(prompt.count(text), 1)
        labels = re.findall(r"Expert opinion from (Expert [A-Z]):", prompt)
        self.assertEqual(sorted(labels), [f"Expert {c}" for c in "ABCDEF"])

    def test_excluded_and_empty_answers_are_filtered(self):
        prompt = build_prompt(excluded_models=["OpenAI"], answer_mistral="")
        self.assertNotIn(ANSWERS["answer_openai"], prompt)
        self.assertNotIn(ANSWERS["answer_mistral"], prompt)
        labels = re.findall(r"Expert opinion from (Expert [A-Z]):", prompt)
        self.assertEqual(sorted(labels), [f"Expert {c}" for c in "ABCD"])

    def test_shuffle_false_keeps_fixed_model_order(self):
        prompt = build_prompt(shuffle=False)
        positions = [prompt.index(ANSWERS[key]) for key in (
            "answer_openai", "answer_mistral", "answer_claude",
            "answer_gemini", "answer_deepseek", "answer_grok",
        )]
        self.assertEqual(positions, sorted(positions))

    def test_shuffle_reorders_expert_labels(self):
        # random.shuffle deterministisch durch reverse ersetzen: Expert A
        # muss dann die Grok-Antwort tragen.
        with mock.patch(
            "app.services.llm.consensus_engine.random.shuffle",
            side_effect=lambda items: items.reverse(),
        ):
            prompt = build_prompt()
        first_block = prompt.split("Expert opinion from Expert B:")[0]
        self.assertIn(ANSWERS["answer_grok"], first_block)

    def test_sources_are_looked_up_by_real_name_but_stay_anonymous(self):
        sources = {"Gemini": [{"id": "S1", "title": "Example Title", "url": "https://example.com/a"}]}
        prompt = build_prompt(model_sources=sources)
        self.assertIn("[S1] Example Title", prompt)
        self.assertNotIn("Gemini", prompt)


class OpenAITokenParamTests(unittest.TestCase):
    def test_reasoning_models_use_max_completion_tokens(self):
        for model in ("gpt-5", "gpt-5.5", "gpt-5.4-mini", "gpt-5-nano", "o1", "o3-mini", "o4-mini"):
            self.assertEqual(_openai_token_param(model), "max_completion_tokens", model)

    def test_non_reasoning_models_use_max_tokens(self):
        # Regression: '"o" in model' matchte frueher fast jedes Modell.
        for model in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"):
            self.assertEqual(_openai_token_param(model), "max_tokens", model)

    def test_temperature_is_only_suppressed_for_reasoning_models(self):
        self.assertIsNone(_effective_temperature("openai", "gpt-5.5", 0.3))
        self.assertIsNone(_effective_temperature("openai", "o3-mini", 0.3))
        self.assertEqual(_effective_temperature("openai", "gpt-4o", 0.3), 0.3)
        self.assertEqual(_effective_temperature("gemini", "gemini-3.1-pro-preview", 0.3), 0.3)


class QueryConsensusFallbackTests(unittest.TestCase):
    def _query(self, engine, api_keys):
        # DEVELOPER_GEMINI_API_KEY leeren, damit nur die explizit uebergebenen
        # Keys den Fallback-Provider bestimmen.
        with mock.patch.dict("os.environ", {"DEVELOPER_GEMINI_API_KEY": ""}), mock.patch(
            "app.services.llm.consensus_engine._call_engine_text",
            side_effect=engine,
        ) as patched:
            result = query_consensus(
                "Q?", "a", "b", None, None, None, None,
                excluded_models=[],
                consensus_model="OpenAI",
                api_keys=api_keys,
            )
        return result, patched

    def test_fallback_provider_rescues_run_after_two_failures(self):
        calls = []

        def engine(provider, api_model, model_ref, api_keys, **kwargs):
            calls.append((provider, kwargs.get("temperature")))
            if len(calls) <= 2:
                raise RuntimeError("503 - UNAVAILABLE")
            return "rescued answer"

        result, patched = self._query(engine, {"OpenAI": "sk-1", "Mistral": "sk-2"})
        self.assertEqual(result, "rescued answer")
        self.assertEqual(patched.call_count, 3)
        self.assertEqual([provider for provider, _ in calls], ["openai", "openai", "mistral"])
        self.assertTrue(all(t == CONSENSUS_TEMPERATURE for _, t in calls))

    def test_empty_results_also_trigger_fallback(self):
        outputs = iter(["", "", "rescued answer"])

        def engine(*args, **kwargs):
            return next(outputs)

        result, patched = self._query(engine, {"OpenAI": "sk-1", "Mistral": "sk-2"})
        self.assertEqual(result, "rescued answer")
        self.assertEqual(patched.call_count, 3)

    def test_without_other_keys_there_is_no_fallback(self):
        def engine(*args, **kwargs):
            raise RuntimeError("503 - UNAVAILABLE")

        result, patched = self._query(engine, {"OpenAI": "sk-1"})
        self.assertEqual(patched.call_count, 2)
        self.assertEqual(result, "Consensus error: 503 - UNAVAILABLE")

    def test_failed_fallback_yields_error_text(self):
        def engine(*args, **kwargs):
            raise RuntimeError("503 - UNAVAILABLE")

        result, patched = self._query(engine, {"OpenAI": "sk-1", "Mistral": "sk-2"})
        self.assertEqual(patched.call_count, 3)
        self.assertEqual(result, "Consensus error: 503 - UNAVAILABLE")


class StreamConsensusFallbackTests(unittest.TestCase):
    def test_fallback_engine_delivers_final_answer(self):
        calls = []

        def fake_engine(engine_model, api_keys, prompt):
            calls.append(engine_model)
            if len(calls) <= 2:
                raise RuntimeError("503 - UNAVAILABLE")
            yield "rescued "
            yield "answer."

        with mock.patch.dict("os.environ", {"DEVELOPER_GEMINI_API_KEY": ""}), mock.patch(
            "app.services.llm.consensus_engine._stream_consensus_engine",
            side_effect=fake_engine,
        ):
            events = list(stream_consensus(
                "Q?", "a", "b", None, None, None, None,
                excluded_models=[],
                consensus_model="OpenAI",
                api_keys={"OpenAI": "sk-1", "Mistral": "sk-2"},
            ))

        self.assertEqual(events[-1], {"type": "final", "text": "rescued answer."})
        # Der dritte Versuch laeuft auf dem Fallback-Judge des naechsten
        # Providers mit Key (Mistral), adressiert als interne Modell-ID.
        self.assertEqual(calls, ["OpenAI", "OpenAI", cfg.DEFAULT_MISTRAL_MODEL])


if __name__ == "__main__":
    unittest.main()
