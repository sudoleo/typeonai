import unittest
from pathlib import Path

from fastapi import HTTPException

import app.core.config as cfg
from app.api.routers.admin import (
    _model_dependencies,
    _server_enforced_models,
    normalize_models_document,
)
from app.api.routers.chat import parse_boolean_flag, validate_question_word_limit
from app.services.llm.base import validate_model
from app.services.llm.citations import source_response
from app.services.llm.engines import build_provider_payload


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigurationTests(unittest.TestCase):
    def test_removed_low_reasoning_aliases_are_not_runtime_models(self):
        self.assertFalse(hasattr(cfg, "EARLY_DEFAULT_MODEL_BY_PROVIDER"))
        self.assertFalse(hasattr(cfg, "EARLY_MODELS"))
        self.assertFalse(hasattr(cfg, "FRONTIER_LOW_MODELS"))
        for model_id in cfg.REMOVED_MODEL_IDS:
            with self.subTest(model=model_id):
                self.assertNotIn(model_id, cfg.ALL_ALLOWED_MODELS)
                self.assertNotIn(model_id, cfg.ALLOWED_CONSENSUS_MODELS)
                self.assertNotIn(model_id, cfg.PREMIUM_MODELS)

    def test_admin_drops_removed_aliases_everywhere(self):
        removed = next(iter(cfg.REMOVED_MODEL_IDS))
        normalized = normalize_models_document({
            "openai": [removed, cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [removed],
            "consensus": [removed, "Gemini"],
            "defaults": {"openai": removed},
        })
        self.assertNotIn(removed, normalized["openai"])
        self.assertNotIn(removed, normalized["premium"])
        self.assertNotIn(removed, normalized["consensus"])
        self.assertNotEqual(normalized["defaults"].get("openai"), removed)

    def test_new_gemini_models_are_direct_and_temperature_free(self):
        for model_id in (cfg.DEFAULT_GEMINI_MODEL, cfg.GEMINI_35_FLASH_MODEL, cfg.GEMINI_36_FLASH_MODEL):
            with self.subTest(model=model_id):
                request = build_provider_payload(
                    "gemini",
                    question="payload dry run",
                    system_prompt="system",
                    model_override=model_id,
                    max_output_tokens=123,
                )
                self.assertEqual(request["internal_model"], model_id)
                self.assertEqual(request["api_model"], model_id)
                generation = request["payload"]["generationConfig"]
                self.assertEqual(generation["maxOutputTokens"], 123)
                self.assertNotIn("temperature", generation)
                self.assertNotIn("thinkingConfig", generation)

    def test_gemini_models_are_available_to_admin(self):
        enforced = _server_enforced_models()["gemini"]
        self.assertEqual(enforced, [])
        self.assertIn(cfg.DEFAULT_GEMINI_MODEL, cfg.ALLOWED_GEMINI_MODELS)
        self.assertIn(cfg.GEMINI_36_FLASH_MODEL, cfg.ALLOWED_GEMINI_MODELS)

    def test_admin_premium_is_limited_to_configured_provider_models(self):
        normalized = normalize_models_document({
            "openai": [cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": ["gpt-5.5", cfg.DEFAULT_GEMINI_MODEL],
        })
        self.assertEqual(normalized["premium"], [cfg.DEFAULT_GEMINI_MODEL])

    def test_admin_dependencies_are_informative_not_server_enforced(self):
        normalized = normalize_models_document({
            "openai": [cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "defaults": {"openai": cfg.DEFAULT_OPENAI_MODEL},
        })
        dependencies = _model_dependencies(normalized)
        self.assertIn("Free default", dependencies["openai"][cfg.DEFAULT_OPENAI_MODEL])
        self.assertEqual(_server_enforced_models()["openai"], [])

    def test_retired_grok_aliases_are_canonicalized(self):
        normalized = normalize_models_document({
            "grok": ["grok-4-fast-reasoning-latest", "grok-4-1-fast-non-reasoning-latest"],
            "premium": ["grok-4-fast-reasoning-latest"],
        })
        self.assertIn("grok-4.3", normalized["grok"])
        self.assertIn(cfg.GROK_NO_REASONING_MODEL, normalized["grok"])
        self.assertNotIn("grok-4-fast-reasoning-latest", normalized["grok"])

    def test_grok_no_reasoning_and_high_reasoning_payloads(self):
        no_reasoning = build_provider_payload(
            "grok", question="q", system_prompt="s",
            model_override=cfg.GROK_NO_REASONING_MODEL, max_output_tokens=123,
        )
        self.assertEqual(no_reasoning["api_model"], "grok-4.3")
        self.assertEqual(no_reasoning["payload"]["reasoning"], {"effort": "none"})

        high = build_provider_payload(
            "grok", question="q", system_prompt="s",
            model_override="grok-4.3", max_output_tokens=123,
        )
        self.assertEqual(high["payload"]["reasoning"], {"effort": "high"})

    def test_access_control_only_has_free_and_pro_models(self):
        validate_model(
            cfg.DEFAULT_OPENAI_MODEL, cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=False
        )
        with self.assertRaises(HTTPException) as denied:
            validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=False)
        self.assertEqual(denied.exception.status_code, 403)
        validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=True)

        # Nach Entfernen des Early-Gates darf DeepSeek Pro nicht ueber das
        # fruehere is_free-Override am Backend-Gate vorbeikommen.
        with self.assertRaises(HTTPException) as deepseek_denied:
            validate_model(
                cfg.DEEPSEEK_PRO_MODEL,
                cfg.ALLOWED_DEEPSEEK_MODELS,
                "DeepSeek",
                is_pro=False,
            )
        self.assertEqual(deepseek_denied.exception.status_code, 403)

    def test_presets_are_complete_and_free_presets_stay_free(self):
        presets = {preset["id"]: preset for preset in cfg.get_consensus_presets()}
        self.assertEqual(set(presets), {"fast", "balanced", "thorough"})
        for preset in presets.values():
            self.assertEqual(set(preset["models"]), set(cfg.DEFAULT_MODEL_BY_PROVIDER))
        self.assertFalse(presets["fast"]["pro_only"])
        self.assertTrue(presets["thorough"]["pro_only"])

        normalized = normalize_models_document({
            "openai": [cfg.OPENAI_LUNA_MODEL, "gpt-5.5"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL, cfg.GROK_FAST_MODEL],
            "premium": ["gpt-5.5"],
            "preset_models": {
                "balanced": {
                    **cfg._BASE_CONSENSUS_PRESET_MODELS["balanced"],
                    "openai": "gpt-5.5",
                    "consensus": "OpenAI-Pro",
                },
            },
        })
        self.assertEqual(
            normalized["preset_models"]["balanced"]["openai"], cfg.OPENAI_LUNA_MODEL
        )
        self.assertEqual(
            normalized["preset_models"]["balanced"]["consensus"], cfg.OPENAI_LUNA_MODEL
        )

    def test_admin_and_picker_have_no_early_contract(self):
        admin = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
        index = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        picker = (ROOT / "static" / "js" / "model-picker.js").read_text(encoding="utf-8")
        tier = (ROOT / "static" / "js" / "user-tier.js").read_text(encoding="utf-8")
        combined = "\n".join((admin, index, picker, tier))
        self.assertNotIn("EARLY_DEFAULT_MODELS", combined)
        self.assertNotIn("early-option", combined)
        self.assertNotIn("isUserEarly", combined)
        self.assertIn("'In use'", admin)
        self.assertNotIn("re-added automatically", admin)
        self.assertNotIn("Server-enforced Pro model", admin)

    def test_provider_errors_are_structured_without_fallback_response(self):
        response = source_response({
            "text": "",
            "sources": [],
            "error": "OpenAI could not complete this request. Please try again later.",
            "error_code": "provider_request_failed",
        })
        self.assertEqual(response["response"], "")
        self.assertEqual(response["error_code"], "provider_request_failed")
        self.assertNotIn("error_detail", response)

    def test_input_helpers(self):
        for question in (None, "", "   "):
            with self.assertRaises(HTTPException) as exc:
                validate_question_word_limit(question, is_pro=False, deep_search=False)
            self.assertEqual(exc.exception.status_code, 400)
        self.assertTrue(parse_boolean_flag(" true "))
        self.assertFalse(parse_boolean_flag("false"))
        self.assertFalse(parse_boolean_flag(None))


if __name__ == "__main__":
    unittest.main()
