import unittest
from pathlib import Path

from fastapi import HTTPException

import app.core.config as cfg
from app.api.routers.admin import normalize_models_document
from app.api.routers.chat import parse_boolean_flag, validate_question_word_limit
from app.services.llm.base import validate_model
from app.services.llm.engines import build_provider_payload
from app.services.llm.citations import source_response


ROOT = Path(__file__).resolve().parents[1]


def contains_value(value, needle):
    if isinstance(value, dict):
        return any(contains_value(k, needle) or contains_value(v, needle) for k, v in value.items())
    if isinstance(value, list):
        return any(contains_value(item, needle) for item in value)
    return value == needle


class FrontierModelPayloadTests(unittest.TestCase):
    def assert_allowed_keys(self, payload, allowed):
        self.assertLessEqual(set(payload), allowed)

    def assert_internal_id_not_sent(self, request):
        self.assertFalse(contains_value(request["payload"], request["internal_model"]))
        if "model" in request["payload"]:
            self.assertNotEqual(request["payload"]["model"], request["internal_model"])

    def test_early_ids_map_to_api_models_and_low_payloads(self):
        cases = [
            (
                "openai",
                cfg.OPENAI_FRONTIER_LOW_MODEL,
                "gpt-5.5",
                {"reasoning": {"effort": "low"}},
                {"model", "instructions", "input", "tools", "tool_choice", "include", "max_output_tokens", "reasoning"},
            ),
            (
                "anthropic",
                cfg.ANTHROPIC_FRONTIER_LOW_MODEL,
                cfg.ANTHROPIC_PRO_MODEL,
                {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
                {"model", "max_tokens", "system", "messages", "tools", "thinking", "output_config"},
            ),
            (
                "gemini",
                cfg.GEMINI_FRONTIER_LOW_MODEL,
                cfg.GEMINI_PRO_MODEL,
                {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}},
                {"systemInstruction", "contents", "tools", "generationConfig", "safetySettings"},
            ),
            (
                "grok",
                cfg.GROK_FRONTIER_LOW_MODEL,
                "grok-4.3",
                {"reasoning": {"effort": "low"}},
                {"model", "instructions", "input", "tools", "tool_choice", "include", "max_output_tokens", "reasoning"},
            ),
        ]

        for provider, internal_model, api_model, expected_low_config, allowed_keys in cases:
            with self.subTest(provider=provider):
                request = build_provider_payload(
                    provider,
                    question="payload dry run",
                    system_prompt="system",
                    model_override=internal_model,
                    max_output_tokens=123,
                )
                self.assertEqual(request["provider"], provider)
                self.assertEqual(request["internal_model"], internal_model)
                self.assertEqual(request["api_model"], api_model)
                if "model" in request["payload"]:
                    self.assertEqual(request["payload"]["model"], api_model)
                self.assertTrue(request["is_low_reasoning"])
                self.assert_internal_id_not_sent(request)
                self.assert_allowed_keys(request["payload"], allowed_keys)

                for key, value in expected_low_config.items():
                    if isinstance(value, dict) and isinstance(request["payload"].get(key), dict):
                        for nested_key, nested_value in value.items():
                            self.assertEqual(request["payload"][key][nested_key], nested_value)
                    else:
                        self.assertEqual(request["payload"][key], value)

                self.assertNotIn("budget_tokens", str(request["payload"]))
                self.assertNotIn("reasoning_effort", request["payload"])

    def test_normal_pro_models_keep_api_names_without_low_payloads(self):
        cases = [
            ("openai", "gpt-5.5", "gpt-5.5"),
            ("anthropic", cfg.ANTHROPIC_PRO_MODEL, cfg.ANTHROPIC_PRO_MODEL),
            ("gemini", cfg.GEMINI_PRO_MODEL, cfg.GEMINI_PRO_MODEL),
            ("grok", "grok-4.3", "grok-4.3"),
        ]

        for provider, model_id, api_model in cases:
            with self.subTest(provider=provider):
                request = build_provider_payload(
                    provider,
                    question="payload dry run",
                    system_prompt="system",
                    model_override=model_id,
                    max_output_tokens=123,
                )
                self.assertEqual(request["provider"], provider)
                self.assertEqual(request["internal_model"], model_id)
                self.assertEqual(request["api_model"], api_model)
                if "model" in request["payload"]:
                    self.assertEqual(request["payload"]["model"], api_model)
                self.assertFalse(request["is_low_reasoning"])
                self.assertNotIn("thinking", request["payload"])
                self.assertNotIn("output_config", request["payload"])
                self.assertNotIn("reasoning", request["payload"])
                self.assertNotIn("reasoning_effort", request["payload"])

    def test_free_defaults_use_frontier_variants(self):
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["openai"], cfg.OPENAI_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["mistral"], cfg.DEFAULT_MISTRAL_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["anthropic"], cfg.ANTHROPIC_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["gemini"], cfg.GEMINI_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["grok"], cfg.GROK_FRONTIER_LOW_MODEL)

    def test_mistral_defaults_use_supported_reasoning_models(self):
        default_request = build_provider_payload(
            "mistral",
            question="payload dry run",
            system_prompt="system",
            max_output_tokens=123,
        )
        self.assertEqual(default_request["api_model"], cfg.DEFAULT_MISTRAL_MODEL)
        self.assertEqual(
            default_request["payload"]["completion_args"]["reasoning_effort"],
            "high",
        )

        deep_request = build_provider_payload(
            "mistral",
            question="payload dry run",
            system_prompt="system",
            deep_search=True,
            max_output_tokens=123,
        )
        self.assertEqual(deep_request["api_model"], cfg.MISTRAL_PRO_MODEL)
        self.assertEqual(
            deep_request["payload"]["completion_args"]["reasoning_effort"],
            "high",
        )
        self.assertNotIn("pixtral-large-latest", cfg.ALLOWED_MISTRAL_MODELS)

    def test_required_pro_models_include_normal_opus_48(self):
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, cfg.REQUIRED_PRO_MODELS)
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, cfg.PREMIUM_MODELS)
        self.assertNotIn(cfg.ANTHROPIC_PRO_MODEL, cfg.EARLY_FREE_MODELS)

        normalized = normalize_models_document({
            "anthropic": [cfg.ANTHROPIC_FRONTIER_LOW_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "deepseek": [],
            "premium": [],
        })
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, normalized["anthropic"])
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, normalized["premium"])

    def test_server_side_access_control(self):
        validate_model(cfg.OPENAI_FRONTIER_LOW_MODEL, cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=False)
        validate_model(cfg.ANTHROPIC_FRONTIER_LOW_MODEL, cfg.ALLOWED_ANTHROPIC_MODELS, "Anthropic", is_pro=False)
        validate_model(cfg.GEMINI_FRONTIER_LOW_MODEL, cfg.ALLOWED_GEMINI_MODELS, "Gemini", is_pro=False)
        validate_model(cfg.GROK_FRONTIER_LOW_MODEL, cfg.ALLOWED_GROK_MODELS, "Grok", is_pro=False)

        with self.assertRaises(HTTPException) as free_pro:
            validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=False)
        self.assertEqual(free_pro.exception.status_code, 403)

        validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=True)

        with self.assertRaises(HTTPException) as invalid:
            validate_model("not-a-real-model", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=True)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_firestore_sync_keeps_frontier_models_allowed_and_not_premium(self):
        snapshots = {
            "openai": set(cfg.ALLOWED_OPENAI_MODELS),
            "anthropic": set(cfg.ALLOWED_ANTHROPIC_MODELS),
            "gemini": set(cfg.ALLOWED_GEMINI_MODELS),
            "grok": set(cfg.ALLOWED_GROK_MODELS),
            "premium": set(cfg.PREMIUM_MODELS),
        }
        try:
            cfg.ALLOWED_OPENAI_MODELS.clear()
            cfg.ALLOWED_ANTHROPIC_MODELS.clear()
            cfg.ALLOWED_GEMINI_MODELS.clear()
            cfg.ALLOWED_GROK_MODELS.clear()
            cfg.PREMIUM_MODELS.update(cfg.FRONTIER_LOW_MODELS)

            cfg.ensure_default_models_allowed()
            cfg.PREMIUM_MODELS.difference_update(cfg.FRONTIER_LOW_MODELS)
            cfg.rebuild_model_configs()

            self.assertIn(cfg.OPENAI_FRONTIER_LOW_MODEL, cfg.ALLOWED_OPENAI_MODELS)
            self.assertIn(cfg.ANTHROPIC_FRONTIER_LOW_MODEL, cfg.ALLOWED_ANTHROPIC_MODELS)
            self.assertIn(cfg.GEMINI_FRONTIER_LOW_MODEL, cfg.ALLOWED_GEMINI_MODELS)
            self.assertIn(cfg.GROK_FRONTIER_LOW_MODEL, cfg.ALLOWED_GROK_MODELS)
            self.assertTrue(cfg.FRONTIER_LOW_MODELS.isdisjoint(cfg.PREMIUM_MODELS))
        finally:
            cfg.ALLOWED_OPENAI_MODELS.clear()
            cfg.ALLOWED_OPENAI_MODELS.update(snapshots["openai"])
            cfg.ALLOWED_ANTHROPIC_MODELS.clear()
            cfg.ALLOWED_ANTHROPIC_MODELS.update(snapshots["anthropic"])
            cfg.ALLOWED_GEMINI_MODELS.clear()
            cfg.ALLOWED_GEMINI_MODELS.update(snapshots["gemini"])
            cfg.ALLOWED_GROK_MODELS.clear()
            cfg.ALLOWED_GROK_MODELS.update(snapshots["grok"])
            cfg.PREMIUM_MODELS.clear()
            cfg.PREMIUM_MODELS.update(snapshots["premium"])
            cfg.rebuild_model_configs()

    def test_saved_preferences_are_not_overwritten_when_present(self):
        # applyTierDefaultModels lebt seit dem index.html-Refactor in
        # static/js/model-picker.js. Der Guard (gespeicherte Auswahl wird nicht
        # von Tier-Defaults ueberschrieben) muss dort erhalten bleiben.
        module = (ROOT / "static" / "js" / "model-picker.js").read_text(encoding="utf-8")
        self.assertIn('localStorage.getItem("pref_select_" + pref.key) !== null', module)
        self.assertIn("return;", module)

    def test_consensus_picker_defaults_to_gemini_frontier_low(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="consensusModelDropdown"', template)
        self.assertIn('value="{{ default_models[\'gemini\'] }}" class="early-option"', template)
        self.assertIn("selected>{{ model_labels.get(default_models['gemini']", template)

    def test_gemini_frontier_low_consensus_payload_uses_api_model(self):
        request = build_provider_payload(
            "gemini",
            question="consensus dry run",
            system_prompt="",
            model_override=cfg.GEMINI_FRONTIER_LOW_MODEL,
            max_output_tokens=456,
        )
        self.assertEqual(request["internal_model"], cfg.GEMINI_FRONTIER_LOW_MODEL)
        self.assertEqual(request["api_model"], cfg.GEMINI_PRO_MODEL)
        self.assertFalse(contains_value(request["payload"], cfg.GEMINI_FRONTIER_LOW_MODEL))
        self.assertEqual(
            request["payload"]["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "low"},
        )

    def test_provider_errors_are_structured_without_fallback_response(self):
        response = source_response({
            "text": "",
            "sources": [],
            "error": "OpenAI could not complete this request. Please try again later.",
            "error_detail": "400 - invalid parameter",
        })
        self.assertEqual(response["response"], "")
        self.assertIn("could not complete", response["error"])
        self.assertIn("invalid parameter", response["error_detail"])

    def test_question_validation_rejects_empty_input(self):
        for question in (None, "", "   "):
            with self.subTest(question=question):
                with self.assertRaises(HTTPException) as exc:
                    validate_question_word_limit(question, is_pro=False, deep_search=False)
                self.assertEqual(exc.exception.status_code, 400)

    def test_boolean_flag_parser_is_whitespace_tolerant(self):
        self.assertTrue(parse_boolean_flag(" true "))
        self.assertFalse(parse_boolean_flag("false"))
        self.assertFalse(parse_boolean_flag(None))


if __name__ == "__main__":
    unittest.main()
