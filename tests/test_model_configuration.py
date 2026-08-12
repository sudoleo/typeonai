import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

import app.core.config as cfg
from app.services.llm.credentials import GEMINI_ADC_ALLOWED
from app.api.routers import admin as admin_router
from app.api.routers.admin import (
    get_models,
    _model_dependencies,
    _server_enforced_models,
    normalize_models_document,
)
from app.api.routers.chat import (
    build_engine_api_keys,
    parse_boolean_flag,
    validate_question_word_limit,
)
from app.services.llm.base import validate_model
from app.services.llm.citations import source_response
from app.services.llm.engines import build_provider_payload


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigurationTests(unittest.TestCase):
    def _valid_admin_payload(self):
        return {
            "openai": list(cfg.ALLOWED_OPENAI_MODELS),
            "mistral": list(cfg.ALLOWED_MISTRAL_MODELS),
            "anthropic": list(cfg.ALLOWED_ANTHROPIC_MODELS),
            "gemini": list(cfg.ALLOWED_GEMINI_MODELS),
            "deepseek": list(cfg.ALLOWED_DEEPSEEK_MODELS),
            "grok": list(cfg.ALLOWED_GROK_MODELS),
            "premium": list(cfg.PREMIUM_MODELS),
            "consensus": list(cfg.ALLOWED_CONSENSUS_MODELS),
            "preset_models": cfg.get_consensus_preset_models(),
            "defaults": dict(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER),
            "watch_models": {
                tier: dict(models) for tier, models in cfg.WATCH_MODELS_BY_TIER.items()
            },
            "deep_think_model": cfg.get_deep_think_consensus_model(),
            "judge_models": cfg.get_judge_models(),
            "judge_models_pro": cfg.get_pro_judge_models(),
            "judge_families": cfg.get_judge_families(),
            "chat_memory_models": cfg.get_chat_memory_models(),
            "limits": cfg.get_limits_config(),
        }

    def test_rejected_admin_document_cannot_mutate_runtime_limits(self):
        payload = self._valid_admin_payload()
        payload["limits"] = {**payload["limits"], "free_consensus_run_limit": 999}
        payload["preset_models"] = {}
        before = cfg.get_limits_config()
        fake_document = mock.Mock()
        fake_db = mock.Mock()
        fake_db.collection.return_value.document.return_value = fake_document

        with (
            mock.patch.object(admin_router, "_require_admin"),
            mock.patch.object(admin_router, "db_firestore", fake_db),
            self.assertRaises(HTTPException) as exc_info,
        ):
            admin_router.update_models(mock.Mock(), payload)

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertEqual(cfg.get_limits_config(), before)
        fake_document.set.assert_not_called()

    def test_runtime_reload_rolls_back_all_mutations_on_activation_error(self):
        payload = self._valid_admin_payload()

        class Snapshot:
            exists = True

            def to_dict(self):
                return payload

        document = mock.Mock()
        document.get.return_value = Snapshot()
        database = mock.Mock()
        database.collection.return_value.document.return_value = document
        before_limits = cfg.get_limits_config()
        before_openai = set(cfg.ALLOWED_OPENAI_MODELS)

        with (
            mock.patch("app.core.security.db_firestore", database),
            mock.patch.object(cfg, "apply_watch_models", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            cfg.load_models_from_db(strict=True)

        self.assertEqual(cfg.get_limits_config(), before_limits)
        self.assertEqual(set(cfg.ALLOWED_OPENAI_MODELS), before_openai)

    def test_admin_update_restores_persisted_document_on_activation_error(self):
        payload = self._valid_admin_payload()
        previous = {"schema_version": 7, "openai": ["previous-model"]}

        class Snapshot:
            exists = True

            def to_dict(self):
                return dict(previous)

        fake_document = mock.Mock()
        fake_document.get.return_value = Snapshot()
        fake_db = mock.Mock()
        fake_db.collection.return_value.document.return_value = fake_document

        with (
            mock.patch.object(admin_router, "_require_admin"),
            mock.patch.object(admin_router, "db_firestore", fake_db),
            mock.patch.object(
                admin_router,
                "load_models_from_db",
                side_effect=RuntimeError("activation failed"),
            ),
            self.assertRaises(HTTPException) as exc_info,
        ):
            admin_router.update_models(mock.Mock(), payload)

        self.assertEqual(exc_info.exception.status_code, 500)
        self.assertGreaterEqual(fake_document.set.call_count, 2)
        self.assertEqual(fake_document.set.call_args_list[-1].args[0], previous)
        fake_document.delete.assert_not_called()

    def test_readiness_config_load_never_creates_missing_document(self):
        snapshot = mock.Mock(exists=False)
        document = mock.Mock()
        document.get.return_value = snapshot
        database = mock.Mock()
        database.collection.return_value.document.return_value = document

        with mock.patch("app.core.security.db_firestore", database):
            loaded = cfg.load_models_from_db(strict=True, persist_backfill=False)

        self.assertTrue(loaded)
        document.get.assert_called_once_with(timeout=5.0, retry=None)
        document.set.assert_not_called()

    def test_engine_developer_keys_use_shared_credentials_source(self):
        expected = {
            "OpenAI": "openai-dev",
            "Mistral": "mistral-dev",
            "Anthropic": None,
            "Gemini": "gemini-dev",
            "DeepSeek": None,
            "Grok": "grok-dev",
        }
        with mock.patch(
            "app.api.routers.chat.resolve_developer_api_keys",
            return_value=expected,
        ) as resolver:
            resolved = build_engine_api_keys({}, False)
            self.assertEqual(
                {key: value for key, value in resolved.items() if key != GEMINI_ADC_ALLOWED},
                expected,
            )
            self.assertTrue(resolved[GEMINI_ADC_ALLOWED])
        resolver.assert_called_once_with()

    def test_engine_own_keys_are_stripped_and_never_use_developer_keys(self):
        with mock.patch(
            "app.api.routers.chat.resolve_developer_api_keys",
        ) as resolver:
            keys = build_engine_api_keys(
                {
                    "openai_key": "  user-openai  ",
                    "gemini_key": "",
                    "mistral_key": "   ",
                },
                True,
            )
        resolver.assert_not_called()
        self.assertEqual(keys["OpenAI"], "user-openai")
        self.assertIsNone(keys["Gemini"])
        self.assertIsNone(keys["Mistral"])
        self.assertNotIn(GEMINI_ADC_ALLOWED, keys)

    def test_admin_models_get_is_read_only_and_preserves_judge_family(self):
        raw = {
            "openai": [cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "consensus": ["Gemini"],
            "judge_families": {"openai": "gemini"},
        }

        class FakeSnapshot:
            exists = True

            def to_dict(self):
                return dict(raw)

        class FakeDocument:
            def __init__(self):
                self.set_calls = []

            def get(self):
                return FakeSnapshot()

            def set(self, *args, **kwargs):
                self.set_calls.append((args, kwargs))

        class FakeCollection:
            def __init__(self, document):
                self._document = document

            def document(self, name):
                self.name = name
                return self._document

        class FakeDB:
            def __init__(self, document):
                self._document = document

            def collection(self, name):
                self.name = name
                return FakeCollection(self._document)

        document = FakeDocument()
        with (
            mock.patch.object(admin_router, "db_firestore", FakeDB(document)),
            mock.patch.object(admin_router, "_require_admin"),
        ):
            response = get_models(mock.Mock())

        self.assertEqual(response["judge_families"], {"openai": "gemini"})
        self.assertEqual(document.set_calls, [])

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
        admin_js = (ROOT / "static" / "js" / "admin.js").read_text(encoding="utf-8")
        index = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        picker = (ROOT / "static" / "js" / "model-picker.js").read_text(encoding="utf-8")
        tier = (ROOT / "static" / "js" / "user-tier.js").read_text(encoding="utf-8")
        combined = "\n".join((admin, admin_js, index, picker, tier))
        self.assertNotIn("EARLY_DEFAULT_MODELS", combined)
        self.assertNotIn("early-option", combined)
        self.assertNotIn("isUserEarly", combined)
        self.assertIn("'In use'", admin_js)
        self.assertNotIn("re-added automatically", combined)
        self.assertNotIn("Server-enforced Pro model", combined)

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
