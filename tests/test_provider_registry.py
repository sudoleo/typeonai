"""Die Provider-Registry ist die einzige Quelle fuer Modellfamilien.

Eine neue Familie soll ein Eintrag in cfg.PROVIDERS plus ihre Modelle in der
Admin-DB sein -- kein weiteres paralleles Dict, keine weitere Sechser-Liste und
keine weitere Stelle im Prompt-Bau. Diese Tests halten genau das fest: alle
abgeleiteten Strukturen decken die Registry vollstaendig ab, und die
Consensus-/Differences-Pipeline haengt an keiner festen Anzahl von Familien.
"""

import unittest

import app.core.config as cfg
from app.services import chat_store, opinion_map, share_snapshots, topics, watch_scheduler
from app.services.llm import engines, provider_transport
from app.services.llm.consensus_engine import (
    _build_consensus_prompt,
    _build_differences_prompt,
    _model_answer_items,
)


class RegistryCoverageTests(unittest.TestCase):
    """Jede abgeleitete Struktur kennt genau die Familien der Registry."""

    def test_derived_model_maps_cover_every_family(self):
        families = set(cfg.PROVIDERS)
        for name, mapping in (
            ("DEFAULT_MODEL_BY_PROVIDER", cfg.DEFAULT_MODEL_BY_PROVIDER),
            ("FREE_DEFAULT_MODEL_BY_PROVIDER", cfg.FREE_DEFAULT_MODEL_BY_PROVIDER),
            ("PROVIDER_LABEL_BY_ID", cfg.PROVIDER_LABEL_BY_ID),
            ("OPENROUTER_MODEL_PREFIXES", cfg.OPENROUTER_MODEL_PREFIXES),
            ("DIFFERENCES_JUDGE_MODEL_BY_PROVIDER", cfg.DIFFERENCES_JUDGE_MODEL_BY_PROVIDER),
            ("PRO_JUDGE_MODEL_BY_PROVIDER", cfg.PRO_JUDGE_MODEL_BY_PROVIDER),
            ("CHAT_MEMORY_MODEL_BY_PROVIDER", cfg.CHAT_MEMORY_MODEL_BY_PROVIDER),
            ("MODEL_ORDER_BY_PROVIDER", cfg.MODEL_ORDER_BY_PROVIDER),
            ("_provider_allowed_sets()", cfg._provider_allowed_sets()),
        ):
            with self.subTest(mapping=name):
                self.assertEqual(set(mapping), families)

    def test_allowed_model_aliases_are_the_registry_sets(self):
        """Die ALLOWED_*-Namen sind Aliasse auf DASSELBE Set-Objekt: der
        Firestore-Load mutiert in place, sonst laufen Importe auf alten Daten."""
        for provider, alias in (
            ("openai", cfg.ALLOWED_OPENAI_MODELS),
            ("mistral", cfg.ALLOWED_MISTRAL_MODELS),
            ("anthropic", cfg.ALLOWED_ANTHROPIC_MODELS),
            ("gemini", cfg.ALLOWED_GEMINI_MODELS),
            ("deepseek", cfg.ALLOWED_DEEPSEEK_MODELS),
            ("grok", cfg.ALLOWED_GROK_MODELS),
        ):
            with self.subTest(provider=provider):
                self.assertIs(alias, cfg.PROVIDERS[provider].models)

    def test_every_family_has_both_consensus_aliases(self):
        for provider in cfg.PROVIDERS.values():
            with self.subTest(provider=provider.key):
                self.assertEqual(
                    cfg.CONSENSUS_ENGINE_ALIASES.get(provider.label),
                    (provider.key, provider.base_model),
                )
                self.assertEqual(
                    cfg.CONSENSUS_ENGINE_ALIASES.get(f"{provider.label}-Pro"),
                    (provider.key, provider.pro_model),
                )
                self.assertIn(provider.label, cfg.VALID_LEADERBOARD_MODELS)
                self.assertIn(f"{provider.label}-Pro", cfg.VALID_LEADERBOARD_MODELS)

    def test_base_model_is_free_and_pro_model_is_premium(self):
        for provider in cfg.PROVIDERS.values():
            with self.subTest(provider=provider.key):
                self.assertIn(provider.base_model, provider.models)
                self.assertIn(provider.pro_model, provider.models)
                self.assertNotIn(provider.base_model, cfg.PREMIUM_MODELS)
                self.assertIn(provider.pro_model, cfg.PREMIUM_MODELS)

    def test_model_configs_resolve_every_allowed_model(self):
        for model_id in cfg.ALL_ALLOWED_MODELS:
            with self.subTest(model=model_id):
                config = cfg.MODEL_CONFIGS[model_id]
                self.assertIn(config.provider, cfg.PROVIDERS)
                self.assertTrue(config.api_model.startswith(
                    cfg.PROVIDERS[config.provider].openrouter_prefix
                ))


class ConsumerCoverageTests(unittest.TestCase):
    """Die Module rund um den Lauf ziehen ihre Familien aus der Registry."""

    def test_transport_order_and_labels_come_from_the_registry(self):
        self.assertEqual(provider_transport.PROVIDER_ORDER, tuple(cfg.PROVIDERS))
        self.assertEqual(provider_transport.PROVIDER_LABELS, cfg.PROVIDER_LABEL_BY_ID)

    def test_engine_model_maps_cover_every_family(self):
        self.assertEqual(set(engines._DEFAULT_MODEL_BY_PROVIDER), set(cfg.PROVIDERS))
        self.assertEqual(set(engines._DEEP_SEARCH_MODEL_BY_PROVIDER), set(cfg.PROVIDERS))

    def test_product_surfaces_cover_every_family(self):
        labels = set(cfg.PROVIDER_LABEL_BY_ID.values())
        self.assertEqual(set(share_snapshots.PROVIDER_ORDER), labels)
        self.assertEqual(set(share_snapshots.PROVIDER_CITATION_LABELS), labels)
        self.assertEqual(set(chat_store.PROVIDER_DOCUMENT_IDS), labels)
        self.assertEqual(set(opinion_map.PROVIDERS), labels)
        self.assertEqual(set(topics.PROVIDER_ORDER), set(cfg.PROVIDERS))
        self.assertEqual(set(topics.PROVIDER_LABELS), set(cfg.PROVIDERS))

    def test_ask_endpoints_and_judge_vocabulary_cover_every_family(self):
        from app.api.routers.chat import ASK_PROVIDERS
        from app.services.llm.consensus_engine import (
            CANONICAL_MODEL_NAMES,
            normalize_model_name,
        )
        from app.services.llm.resolve_engine import PROVIDER_BY_LABEL

        self.assertEqual(set(ASK_PROVIDERS), set(cfg.PROVIDERS))
        self.assertEqual(
            set(PROVIDER_BY_LABEL), set(cfg.PROVIDER_LABEL_BY_ID.values())
        )
        for provider, label in cfg.PROVIDER_LABEL_BY_ID.items():
            with self.subTest(provider=provider):
                self.assertIs(ASK_PROVIDERS[provider].allowed_models,
                              cfg.PROVIDERS[provider].models)
                # Der Judge darf die Familie als Label, klein oder als
                # "<Label>-Pro" benennen; alles landet auf demselben Namen.
                self.assertEqual(CANONICAL_MODEL_NAMES[provider], label)
                self.assertEqual(normalize_model_name(label), label)
                self.assertEqual(normalize_model_name(f"{label}-Pro"), label)

    def test_preference_orders_stay_inside_the_registry(self):
        """Bewusst abweichende Reihenfolgen sind erlaubt, duerfen aber keine
        Familie verlieren -- sonst faellt eine neue Familie still hinten weg."""
        self.assertEqual(set(watch_scheduler.PROVIDER_ORDER), set(cfg.PROVIDERS))
        self.assertEqual(
            {provider.key for provider in cfg._consensus_alias_providers()},
            set(cfg.PROVIDERS),
        )


class PipelineIsFamilyCountAgnosticTests(unittest.TestCase):
    """Prompt-Bau und Judge haengen an der uebergebenen Menge, nicht an sechs."""

    def test_consensus_prompt_grows_and_shrinks_with_the_answers(self):
        for count in (2, 3, 6, 8):
            answers = {f"family{i}": f"answer {i}" for i in range(count)}
            with self.subTest(count=count):
                prompt = _build_consensus_prompt("Q?", answers, [], shuffle=False)
                for i in range(count):
                    self.assertIn(f"answer {i}", prompt)
                self.assertEqual(prompt.count("Expert opinion from"), count)

    def test_differences_prompt_anonymizes_any_number_of_families(self):
        answers = {f"family{i}": f"answer {i}" for i in range(8)}
        built = _build_differences_prompt(answers, "consensus text", excluded_models=[])
        self.assertIsNotNone(built)
        _prompt, anon_map, answers_by_model, _sentences = built
        self.assertEqual(len(anon_map), 8)
        self.assertEqual(len(answers_by_model), 8)
        self.assertEqual(sorted(anon_map), [f"Model {chr(65 + i)}" for i in range(8)])

    def test_answers_may_be_keyed_by_family_id_or_display_name(self):
        by_id = _model_answer_items({"openai": "a", "deepseek": "b"}, [])
        by_label = _model_answer_items({"OpenAI": "a", "DeepSeek": "b"}, [])
        self.assertEqual(by_id, [("OpenAI", "a"), ("DeepSeek", "b")])
        self.assertEqual(by_id, by_label)

    def test_empty_and_excluded_answers_drop_out(self):
        items = _model_answer_items(
            {"openai": "a", "mistral": "", "gemini": None, "grok": "d"},
            ["Grok"],
        )
        self.assertEqual(items, [("OpenAI", "a")])


if __name__ == "__main__":
    unittest.main()
