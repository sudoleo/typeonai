import unittest
from pathlib import Path

from fastapi import HTTPException

import app.core.config as cfg
from app.api.routers.admin import _server_enforced_models, normalize_models_document
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

    def test_grok_43_no_reasoning_uses_canonical_api_model_and_none_effort(self):
        request = build_provider_payload(
            "grok",
            question="payload dry run",
            system_prompt="system",
            model_override=cfg.GROK_NO_REASONING_MODEL,
            max_output_tokens=123,
        )
        self.assertEqual(request["internal_model"], cfg.GROK_NO_REASONING_MODEL)
        self.assertEqual(request["api_model"], "grok-4.3")
        self.assertEqual(request["payload"]["reasoning"], {"effort": "none"})
        self.assertFalse(request["is_low_reasoning"])
        self.assertNotIn(cfg.GROK_NO_REASONING_MODEL, cfg.PREMIUM_MODELS)

        high_request = build_provider_payload(
            "grok",
            question="payload dry run",
            system_prompt="system",
            model_override="grok-4.3",
            max_output_tokens=123,
        )
        self.assertEqual(high_request["api_model"], "grok-4.3")
        self.assertEqual(high_request["payload"]["reasoning"], {"effort": "high"})
        self.assertIn("grok-4.3", cfg.PREMIUM_MODELS)

    def test_admin_migrates_retired_grok_aliases_without_enforcing_fast_variant(self):
        old_non_reasoning = "grok-4-1-fast-non-reasoning-latest"
        old_reasoning = "grok-4-fast-reasoning-latest"
        normalized = normalize_models_document({
            "grok": [old_non_reasoning, old_reasoning, cfg.DEFAULT_GROK_MODEL, "grok-4.3"],
            "premium": [old_non_reasoning, old_reasoning, "grok-4.3"],
            "preset_models": {
                "fast": {
                    **cfg._BASE_CONSENSUS_PRESET_MODELS["fast"],
                    "grok": old_non_reasoning,
                },
            },
        })
        self.assertNotIn(old_non_reasoning, normalized["grok"])
        self.assertNotIn(old_reasoning, normalized["grok"])
        self.assertIn(cfg.GROK_NO_REASONING_MODEL, normalized["grok"])
        self.assertIn(cfg.GROK_FRONTIER_LOW_MODEL, normalized["grok"])
        self.assertNotIn(cfg.GROK_NO_REASONING_MODEL, normalized["premium"])
        self.assertEqual(
            normalized["preset_models"]["fast"]["grok"],
            cfg.GROK_NO_REASONING_MODEL,
        )
        self.assertNotIn(cfg.GROK_NO_REASONING_MODEL, _server_enforced_models()["grok"])

    def test_free_defaults_use_cheap_base_models(self):
        # Nicht-Early-Nutzer fallen auf die guenstigen Basis-Modelle zurueck;
        # keines davon ist ein tag-gated Early-Modell.
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["openai"], cfg.DEFAULT_OPENAI_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["mistral"], cfg.DEFAULT_MISTRAL_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["anthropic"], cfg.DEFAULT_ANTHROPIC_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["gemini"], cfg.DEFAULT_GEMINI_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["deepseek"], cfg.DEEPSEEK_FLASH_MODEL)
        self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["grok"], cfg.DEFAULT_GROK_MODEL)
        for model_id in cfg.FREE_DEFAULT_MODEL_BY_PROVIDER.values():
            self.assertNotIn(model_id, cfg.EARLY_MODELS)

    def test_early_defaults_use_frontier_variants(self):
        # Early-Nutzer behalten die Frontier-Low-Varianten als Default.
        self.assertEqual(cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER["openai"], cfg.OPENAI_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER["anthropic"], cfg.ANTHROPIC_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER["gemini"], cfg.GEMINI_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER["grok"], cfg.GROK_FRONTIER_LOW_MODEL)
        self.assertEqual(cfg.EARLY_DEFAULT_MODEL_BY_PROVIDER["deepseek"], cfg.DEEPSEEK_PRO_MODEL)

    def test_mistral_small_is_a_normal_free_model(self):
        # Mistral Small ist guenstig -> kein Early-/Pro-Gate.
        self.assertNotIn(cfg.DEFAULT_MISTRAL_MODEL, cfg.EARLY_MODELS)
        self.assertNotIn(cfg.DEFAULT_MISTRAL_MODEL, cfg.PREMIUM_MODELS)
        config = cfg.get_model_config(cfg.DEFAULT_MISTRAL_MODEL)
        self.assertTrue(config.is_free)
        self.assertFalse(config.is_frontier)
        # Ohne Early-/Pro-Tag waehlbar.
        validate_model(cfg.DEFAULT_MISTRAL_MODEL, cfg.ALLOWED_MISTRAL_MODELS, "Mistral", is_pro=False, is_early=False)

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
        self.assertNotIn(cfg.ANTHROPIC_PRO_MODEL, cfg.EARLY_MODELS)

        normalized = normalize_models_document({
            "anthropic": [cfg.ANTHROPIC_FRONTIER_LOW_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "deepseek": [],
            "premium": [],
        })
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, normalized["anthropic"])
        self.assertIn(cfg.ANTHROPIC_PRO_MODEL, normalized["premium"])

    def test_admin_models_normalizes_consensus_models_from_posted_provider_lists(self):
        normalized = normalize_models_document({
            "openai": ["gpt-test-consensus"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.GEMINI_FRONTIER_LOW_MODEL],
            "deepseek": [cfg.DEEPSEEK_PRO_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "consensus": ["Gemini-Pro", "gpt-test-consensus", "not-in-provider-list"],
        })
        self.assertIn(cfg.GEMINI_FRONTIER_LOW_MODEL, normalized["consensus"])
        self.assertIn("Gemini-Pro", normalized["consensus"])
        self.assertIn("gpt-test-consensus", normalized["consensus"])
        self.assertNotIn("not-in-provider-list", normalized["consensus"])

    def test_server_side_access_control(self):
        # Early-Modelle sind tag-gated: ohne Early-Zugang -> 403.
        early_cases = [
            (cfg.OPENAI_FRONTIER_LOW_MODEL, cfg.ALLOWED_OPENAI_MODELS, "OpenAI"),
            (cfg.ANTHROPIC_FRONTIER_LOW_MODEL, cfg.ALLOWED_ANTHROPIC_MODELS, "Anthropic"),
            (cfg.GEMINI_FRONTIER_LOW_MODEL, cfg.ALLOWED_GEMINI_MODELS, "Gemini"),
            (cfg.GROK_FRONTIER_LOW_MODEL, cfg.ALLOWED_GROK_MODELS, "Grok"),
            (cfg.DEEPSEEK_PRO_MODEL, cfg.ALLOWED_DEEPSEEK_MODELS, "DeepSeek"),
        ]
        for model_id, allowed, provider in early_cases:
            with self.subTest(model=model_id):
                # Mit Early-Zugang erlaubt.
                validate_model(model_id, allowed, provider, is_pro=False, is_early=True)
                # Pro schliesst Early ein (is_early wird an der Aufrufstelle so gesetzt).
                validate_model(model_id, allowed, provider, is_pro=True, is_early=True)
                # Ohne Early-Zugang gesperrt.
                with self.assertRaises(HTTPException) as denied:
                    validate_model(model_id, allowed, provider, is_pro=False, is_early=False)
                self.assertEqual(denied.exception.status_code, 403)

        with self.assertRaises(HTTPException) as free_pro:
            validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=False)
        self.assertEqual(free_pro.exception.status_code, 403)

        validate_model("gpt-5.5", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=True)

        with self.assertRaises(HTTPException) as invalid:
            validate_model("not-a-real-model", cfg.ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=True)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_early_consensus_engine_is_tag_gated(self):
        self.assertTrue(cfg.is_early_consensus_model(cfg.GEMINI_FRONTIER_LOW_MODEL))
        self.assertFalse(cfg.is_early_consensus_model("Gemini"))
        self.assertFalse(cfg.is_early_consensus_model("OpenAI-Pro"))

    def test_deep_think_consensus_model_is_always_available_and_pro_gated(self):
        self.assertIn(cfg.GEMINI_35_FLASH_MODEL, cfg.DEFAULT_CONSENSUS_MODELS)
        self.assertIn(cfg.GEMINI_35_FLASH_MODEL, cfg.REQUIRED_PRO_MODELS)
        self.assertTrue(cfg.is_premium_consensus_model(cfg.GEMINI_35_FLASH_MODEL))
        normalized = cfg.normalize_consensus_models(["Grok"])
        self.assertIn(cfg.GEMINI_35_FLASH_MODEL, normalized)

    def test_consensus_presets_are_complete_model_sets(self):
        presets = {preset["id"]: preset for preset in cfg.get_consensus_presets()}
        self.assertEqual(set(presets), {"fast", "balanced", "thorough"})
        for preset in presets.values():
            self.assertEqual(set(preset["models"]), set(cfg.DEFAULT_MODEL_BY_PROVIDER))
            self.assertTrue(preset["consensus_model"])
        self.assertEqual(presets["balanced"]["models"]["openai"], cfg.OPENAI_LUNA_MODEL)
        self.assertEqual(presets["balanced"]["consensus_model"], cfg.OPENAI_LUNA_MODEL)
        self.assertEqual(presets["thorough"]["models"]["openai"], cfg.OPENAI_SOL_MODEL)
        self.assertEqual(presets["thorough"]["label"], "High Quality")
        self.assertIn(cfg.OPENAI_SOL_MODEL, cfg.PREMIUM_MODELS)
        self.assertEqual(
            presets["fast"]["models"]["grok"],
            cfg.GROK_FAST_MODEL,
        )
        self.assertTrue(presets["thorough"]["pro_only"])

    def test_admin_normalizes_preset_models_and_keeps_free_presets_free(self):
        normalized = normalize_models_document({
            "openai": [cfg.OPENAI_LUNA_MODEL, "gpt-5.5"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL, cfg.GROK_FAST_MODEL],
            "premium": ["gpt-5.5"],
            "consensus": ["Gemini"],
            "preset_models": {
                "balanced": {
                    **cfg._BASE_CONSENSUS_PRESET_MODELS["balanced"],
                    "openai": "gpt-5.5",
                    "consensus": "OpenAI-Pro",
                },
                "thorough": {
                    **cfg._BASE_CONSENSUS_PRESET_MODELS["thorough"],
                    "openai": "gpt-5.5",
                },
            },
        })
        self.assertEqual(
            normalized["preset_models"]["balanced"]["openai"],
            cfg.OPENAI_LUNA_MODEL,
        )
        self.assertEqual(
            normalized["preset_models"]["balanced"]["consensus"],
            cfg.OPENAI_LUNA_MODEL,
        )
        self.assertIn(cfg.OPENAI_LUNA_MODEL, normalized["consensus"])
        self.assertEqual(
            normalized["preset_models"]["thorough"],
            cfg._BASE_CONSENSUS_PRESET_MODELS["thorough"],
        )

    def test_apply_deep_think_model_validates_and_falls_back(self):
        snapshot = cfg.get_deep_think_consensus_model()
        try:
            # Gueltiger Alias wird uebernommen und in der Consensus-Liste gesichert.
            cfg.apply_deep_think_model("Gemini-Pro")
            self.assertEqual(cfg.get_deep_think_consensus_model(), "Gemini-Pro")
            self.assertIn("Gemini-Pro", cfg.normalize_consensus_models(["Grok"]))
            # Unbekannte Werte fallen auf die Basis zurueck.
            cfg.apply_deep_think_model("not-a-model")
            self.assertEqual(cfg.get_deep_think_consensus_model(), cfg.GEMINI_35_FLASH_MODEL)
            # Leer/None ebenfalls.
            cfg.apply_deep_think_model(None)
            self.assertEqual(cfg.get_deep_think_consensus_model(), cfg.GEMINI_35_FLASH_MODEL)
        finally:
            cfg.apply_deep_think_model(snapshot)

    def test_normalize_models_document_validates_deep_think_model(self):
        base = {
            "openai": [cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "consensus": ["Gemini"],
        }
        # Alias ist gueltig und bleibt in der Consensus-Liste erhalten.
        normalized = normalize_models_document({**base, "deep_think_model": "Anthropic-Pro"})
        self.assertEqual(normalized["deep_think_model"], "Anthropic-Pro")
        self.assertIn("Anthropic-Pro", normalized["consensus"])
        # Direkte Modell-ID aus einer Provider-Liste ist gueltig.
        normalized = normalize_models_document({**base, "deep_think_model": cfg.DEFAULT_OPENAI_MODEL})
        self.assertEqual(normalized["deep_think_model"], cfg.DEFAULT_OPENAI_MODEL)
        self.assertIn(cfg.DEFAULT_OPENAI_MODEL, normalized["consensus"])
        # Unbekannte Werte und fehlendes Feld fallen auf die Basis zurueck.
        for payload in ({**base, "deep_think_model": "not-a-model"}, dict(base)):
            normalized = normalize_models_document(payload)
            self.assertEqual(normalized["deep_think_model"], cfg.GEMINI_35_FLASH_MODEL)
            self.assertIn(cfg.GEMINI_35_FLASH_MODEL, normalized["consensus"])

    def test_apply_judge_models_validates_and_falls_back(self):
        snapshot = cfg.get_judge_models()
        try:
            # Gueltiges Provider-Modell wird uebernommen; das Modul-Alias in
            # consensus_engine sieht die Aenderung live (in-place Mutation).
            from app.services.llm import consensus_engine
            cfg.apply_judge_models({"openai": "gpt-5.5"})
            self.assertEqual(cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["openai"], "gpt-5.5")
            self.assertEqual(
                consensus_engine.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["openai"], "gpt-5.5"
            )
            # Andere Provider bleiben auf der Basis.
            self.assertEqual(
                cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["mistral"],
                cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER["mistral"],
            )
            # Frontier-Low-IDs (interne Aliasse) und unbekannte Modelle -> Basis.
            cfg.apply_judge_models({
                "openai": cfg.OPENAI_FRONTIER_LOW_MODEL,
                "gemini": "not-a-model",
            })
            self.assertEqual(
                cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["openai"],
                cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER["openai"],
            )
            self.assertEqual(
                cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["gemini"],
                cfg._BASE_DIFFERENCES_JUDGE_BY_PROVIDER["gemini"],
            )
        finally:
            cfg.apply_judge_models(snapshot)

    def test_apply_pro_judge_models_and_judge_engine(self):
        from app.services.llm import consensus_engine
        snapshot = cfg.get_pro_judge_models()
        try:
            # Basis = API-Modelle der "<Familie>-Pro"-Aliasse.
            self.assertEqual(cfg._BASE_PRO_JUDGE_BY_PROVIDER["anthropic"], cfg.ANTHROPIC_PRO_MODEL)
            # Gueltiges Modell wird uebernommen und vom Pro-Judge-Pfad genutzt.
            cfg.apply_pro_judge_models({"anthropic": "claude-opus-4-7"})
            self.assertEqual(cfg.PRO_JUDGE_MODEL_BY_PROVIDER["anthropic"], "claude-opus-4-7")
            provider, api_model, model_ref = consensus_engine._judge_engine("anthropic", "pro")
            self.assertEqual((provider, api_model, model_ref), ("anthropic", "claude-opus-4-7", "claude-opus-4-7"))
            # Standard-Stufe bleibt unberuehrt.
            self.assertEqual(
                consensus_engine._judge_engine("anthropic", "standard")[1],
                cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER["anthropic"],
            )
            # Frontier-Low/unbekannt -> Basis.
            cfg.apply_pro_judge_models({"anthropic": cfg.ANTHROPIC_FRONTIER_LOW_MODEL})
            self.assertEqual(cfg.PRO_JUDGE_MODEL_BY_PROVIDER["anthropic"], cfg.ANTHROPIC_PRO_MODEL)
        finally:
            cfg.apply_pro_judge_models(snapshot)

    def test_apply_judge_families_and_family_preference(self):
        from app.services.llm import consensus_engine
        snapshot = cfg.get_judge_families()
        all_keys = {name: "key" for name in
                    ("OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok")}
        try:
            # Ohne Mapping: Prioritaetsliste (gemini zuerst, eigene Familie nie).
            cfg.apply_judge_families({})
            self.assertEqual(
                consensus_engine._judge_families("openai", all_keys, count=2),
                ["gemini", "mistral"],
            )
            # Mapping bevorzugt die gewaehlte Familie vor der Prioritaet.
            cfg.apply_judge_families({"openai": "anthropic"})
            self.assertEqual(
                consensus_engine._judge_families("openai", all_keys, count=2),
                ["anthropic", "gemini"],
            )
            # Kein Key fuer die bevorzugte Familie -> Auto-Fallback.
            keys_without_anthropic = dict(all_keys)
            keys_without_anthropic.pop("Anthropic")
            self.assertEqual(
                consensus_engine._judge_families("openai", keys_without_anthropic, count=1),
                ["gemini"],
            )
            # Self-Judging und unbekannte Provider werden verworfen.
            cfg.apply_judge_families({"openai": "openai", "gemini": "nope"})
            self.assertEqual(cfg.get_judge_families(), {})
        finally:
            cfg.apply_judge_families(snapshot)

    def test_normalize_models_document_validates_pro_judges_and_families(self):
        normalized = normalize_models_document({
            "openai": ["gpt-5.5"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "judge_models_pro": {
                "openai": "gpt-5.5",                      # gueltig
                "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL,  # frontier-low -> verworfen
            },
            "judge_families": {
                "openai": "anthropic",   # gueltig
                "gemini": "gemini",      # Self-Judging -> verworfen
                "grok": "not-a-provider" # unbekannt -> verworfen
            },
        })
        self.assertEqual(normalized["judge_models_pro"].get("openai"), "gpt-5.5")
        self.assertNotIn("gemini", normalized["judge_models_pro"])
        self.assertEqual(normalized["judge_families"], {"openai": "anthropic"})

    def test_normalize_models_document_validates_judge_models(self):
        normalized = normalize_models_document({
            "openai": ["gpt-5.5", cfg.DEFAULT_OPENAI_MODEL],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": [],
            "judge_models": {
                "openai": "gpt-5.5",                        # gueltig
                "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL,    # frontier-low -> verworfen
                "mistral": "not-in-list",                   # unbekannt -> verworfen
            },
        })
        self.assertEqual(normalized["judge_models"].get("openai"), "gpt-5.5")
        self.assertNotIn("gemini", normalized["judge_models"])
        self.assertNotIn("mistral", normalized["judge_models"])

    def test_admin_template_has_tabs_and_deep_think_control(self):
        template = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
        self.assertIn('deepThinkModelSelect', template)
        self.assertIn('deep_think_model: currentDeepThinkModel()', template)
        self.assertIn('judge_models: currentJudgeModels()', template)
        self.assertIn('judge_models_pro: currentProJudgeModels()', template)
        self.assertIn('judge_families: currentJudgeFamilies()', template)
        self.assertIn('judgeModelsContainer', template)
        self.assertIn('judgeFamiliesContainer', template)
        self.assertIn('data-tab="consensus"', template)
        self.assertIn('data-tab="models"', template)
        self.assertIn('id="presetModelsContainer"', template)
        self.assertIn('preset_models: currentPresetModels()', template)
        self.assertIn('.top-bar-favicon {', template)
        self.assertIn('width: 30px;', template)

    def test_consensus_preset_picker_applies_answers_and_pro_gates_thorough(self):
        module = (ROOT / "static" / "js" / "model-picker.js").read_text(encoding="utf-8")
        self.assertIn('preset.models?.[pref.provider]', module)
        self.assertIn('preset.consensus_model', module)
        self.assertIn('preset.pro_only && window.isUserPro !== true', module)
        self.assertIn('badge.textContent = "Pro"', module)
        self.assertIn('window.App.markConsensusPresetCustom', module)

    def test_empty_app_and_consensus_picker_css_prevent_overflow(self):
        base_css = (ROOT / "static" / "css" / "base.css").read_text(encoding="utf-8")
        picker_css = (ROOT / "static" / "css" / "components-model-picker.css").read_text(encoding="utf-8")
        self.assertIn("box-sizing: border-box;", base_css)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", picker_css)
        self.assertIn("overflow-x: hidden;", picker_css)

    def test_index_injects_deep_think_consensus_model(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn("window.DEEP_THINK_CONSENSUS_MODEL", template)
        module = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")
        self.assertIn('window.DEEP_THINK_CONSENSUS_MODEL || "gemini-3.5-flash"', module)

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
        self.assertIn("{% for model in consensus_models %}", template)
        self.assertIn('value="{{ model.value }}"', template)
        self.assertIn("{% if loop.first %}selected{% endif %}", template)

    def test_normalize_preserves_provider_order_and_validates_defaults(self):
        normalized = normalize_models_document({
            "openai": ["gpt-5.5", "gpt-5.4-mini"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL],
            "gemini": [cfg.GEMINI_FRONTIER_LOW_MODEL, cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": ["gpt-5.5"],
            "defaults": {
                "openai": "gpt-5.4-mini",          # gueltiger Free-Default
                "mistral": "gpt-5.5",              # premium -> verworfen
                "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL,  # early -> verworfen
            },
        })
        # Reihenfolge der eingegebenen Modelle bleibt erhalten (Pflichtmodelle hinten).
        self.assertEqual(normalized["openai"][:2], ["gpt-5.5", "gpt-5.4-mini"])
        self.assertIn(cfg.OPENAI_FRONTIER_LOW_MODEL, normalized["openai"])
        # Free-Default: nur das gueltige bleibt, Premium/Early werden verworfen.
        self.assertEqual(normalized["defaults"].get("openai"), "gpt-5.4-mini")
        self.assertNotIn("mistral", normalized["defaults"])
        self.assertNotIn("gemini", normalized["defaults"])

    def test_model_order_and_default_overrides_apply(self):
        order_snapshot = {p: list(v) for p, v in cfg.MODEL_ORDER_BY_PROVIDER.items()}
        free_snapshot = dict(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER)
        try:
            # get_ordered_models: Admin-Reihenfolge gewinnt, Rest wird angehaengt.
            cfg.apply_model_order({"openai": ["gpt-5.5"]})
            ordered = cfg.get_ordered_models("openai")
            self.assertEqual(ordered[0], "gpt-5.5")
            self.assertEqual(set(ordered), set(cfg.ALLOWED_OPENAI_MODELS))

            # Gueltiger Free-Default greift.
            cfg.apply_default_models({"openai": "gpt-4o"})
            self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["openai"], "gpt-4o")

            # Premium/Early als Free-Default -> Fallback auf Basis.
            cfg.apply_default_models({"openai": "gpt-5.5", "gemini": cfg.GEMINI_FRONTIER_LOW_MODEL})
            self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["openai"], cfg._BASE_FREE_DEFAULTS["openai"])
            self.assertEqual(cfg.FREE_DEFAULT_MODEL_BY_PROVIDER["gemini"], cfg._BASE_FREE_DEFAULTS["gemini"])
        finally:
            cfg.MODEL_ORDER_BY_PROVIDER.clear()
            cfg.MODEL_ORDER_BY_PROVIDER.update(order_snapshot)
            cfg.FREE_DEFAULT_MODEL_BY_PROVIDER.clear()
            cfg.FREE_DEFAULT_MODEL_BY_PROVIDER.update(free_snapshot)

    def test_watch_models_are_separate_for_free_and_pro(self):
        normalized = normalize_models_document({
            "openai": [cfg.DEFAULT_OPENAI_MODEL, "gpt-5.5"],
            "mistral": [cfg.DEFAULT_MISTRAL_MODEL, cfg.MISTRAL_PRO_MODEL],
            "anthropic": [cfg.DEFAULT_ANTHROPIC_MODEL, cfg.ANTHROPIC_PRO_MODEL],
            "gemini": [cfg.DEFAULT_GEMINI_MODEL],
            "deepseek": [cfg.DEEPSEEK_FLASH_MODEL],
            "grok": [cfg.DEFAULT_GROK_MODEL],
            "premium": ["gpt-5.5", cfg.MISTRAL_PRO_MODEL, cfg.ANTHROPIC_PRO_MODEL],
            "watch_models": {
                "free": {
                    "openai": cfg.DEFAULT_OPENAI_MODEL,
                    "mistral": cfg.DEFAULT_MISTRAL_MODEL,
                    "anthropic": cfg.ANTHROPIC_PRO_MODEL,
                },
                "pro": {
                    "openai": "gpt-5.5",
                    "mistral": cfg.MISTRAL_PRO_MODEL,
                    "anthropic": cfg.ANTHROPIC_PRO_MODEL,
                },
            },
        })
        self.assertEqual(set(normalized["watch_models"]["free"]), {"openai", "mistral"})
        self.assertEqual(len(normalized["watch_models"]["pro"]), 3)

        snapshot = {tier: dict(models) for tier, models in cfg.WATCH_MODELS_BY_TIER.items()}
        try:
            cfg.apply_watch_models(normalized["watch_models"])
            self.assertEqual(cfg.get_watch_models(False), normalized["watch_models"]["free"])
            self.assertEqual(cfg.get_watch_models(True), normalized["watch_models"]["pro"])
        finally:
            for tier, models in snapshot.items():
                cfg.WATCH_MODELS_BY_TIER[tier].clear()
                cfg.WATCH_MODELS_BY_TIER[tier].update(models)

    def test_admin_model_rows_have_order_and_default_controls(self):
        template = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
        self.assertIn("default-radio", template)
        self.assertIn("moveRow(row", template)
        self.assertIn("data.defaults[p] = modelName", template)

    def test_admin_model_rows_have_separate_premium_and_consensus_toggles(self):
        template = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
        self.assertIn("premium-checkbox", template)
        self.assertIn("consensus-checkbox", template)
        self.assertIn("addConsensusValue(modelName)", template)
        self.assertIn("moveConsensusRow", template)
        self.assertIn("consensusListValues()", template)
        self.assertIn("globalModelsData.consensus", template)

    def test_admin_exposes_free_and_pro_watch_model_config(self):
        template = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
        self.assertIn('id="watchModelConfig"', template)
        self.assertIn("watch_models: { free: {}, pro: {} }", template)
        self.assertIn("data.watch_models[select.dataset.watchTier]", template)

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
            "error_code": "provider_request_failed",
        })
        self.assertEqual(response["response"], "")
        self.assertIn("could not complete", response["error"])
        self.assertEqual(response["error_code"], "provider_request_failed")
        self.assertNotIn("error_detail", response)

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
