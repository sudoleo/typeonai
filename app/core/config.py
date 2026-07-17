from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    internal_id: str
    provider: str
    api_model: str
    label: str
    is_free: bool = True
    is_pro: bool = False
    is_frontier: bool = False
    is_low_reasoning: bool = False
    low_config: dict[str, Any] = field(default_factory=dict)

DEFAULT_LIMITS = {
    "free_usage_limit": 25,
    "pro_usage_limit": 500,
    "free_deep_search_limit": 0,
    "pro_deep_search_limit": 50,
    "free_max_words": 500,
    "pro_max_words": 500,
    "free_deep_search_max_words": 0,
    "pro_deep_search_max_words": 1000,
    "free_max_tokens": 4096,
    "pro_max_tokens": 4096,
    "free_deep_search_max_tokens": 0,
    "pro_deep_search_max_tokens": 8192,
    "consensus_max_tokens": 8192,
    "differences_max_tokens": 8192,
    # Serverseitige Eingabe-Caps fuer /consensus: Antworten/Frage kommen vom
    # Client und muessen begrenzt werden, bevor sie in den Engine-Prompt
    # fliessen (Kosten-/Abuse-Schutz). Grosszuegig gewaehlt, damit legitime
    # Deep-Search-Antworten (8192 Output-Tokens) nie gekappt werden.
    "consensus_max_answer_chars": 40_000,
    "consensus_max_question_chars": 8_000,
    # Serverseitige Caps fuer den Follow-up-Kontext (previous_question +
    # previous_consensus kommen vom Client). Bewusst enger als die
    # /consensus-Caps: der Kontext geht bei jeder Follow-up-Frage in alle
    # /ask_*-Prompts gleichzeitig ein (Kostenkontrolle).
    "followup_max_question_chars": 4_000,
    "followup_max_consensus_chars": 12_000,
    # Qualitätsfilter für index_eligible von Share-Snapshots (Etappe 3):
    # steuert nur die Eligibility-Anzeige, indexed setzt weiterhin der Admin.
    "share_min_consensus_chars": 600,
    "share_min_sources": 2,
    "share_min_models": 3,
    "share_question_min_chars": 15,
    "share_question_max_chars": 300,
    "watch_free_active_limit": 1,
    "watch_pro_active_limit": 5,
    "watch_max_runs_per_day": 50,
}

LIMITS = DEFAULT_LIMITS.copy()

FREE_USAGE_LIMIT = LIMITS["free_usage_limit"]
MAX_WORDS = LIMITS["free_max_words"]
DEEP_SEARCH_MAX_WORDS = LIMITS["pro_deep_search_max_words"]
MAX_TOKENS = LIMITS["pro_max_tokens"]
DEEP_SEARCH_MAX_TOKENS = LIMITS["pro_deep_search_max_tokens"]
CONSENSUS_MAX_TOKENS = LIMITS["consensus_max_tokens"]
DIFFERENCES_MAX_TOKENS = LIMITS["differences_max_tokens"]
REASONING_EFFORT_FOR_DEEP = "low"
GEMINI_MAX_TOKENS = MAX_TOKENS
GEMINI_DEEP_MAX_TOKENS = DEEP_SEARCH_MAX_TOKENS
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_PRO_MODEL = "mistral-medium-3-5"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_PRO_MODEL = "claude-opus-4-8"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_MODEL = DEEPSEEK_PRO_MODEL
DEFAULT_GROK_MODEL = "grok-4.20-non-reasoning"

OPENAI_FRONTIER_LOW_MODEL = "gpt-5.5-frontier-low"
ANTHROPIC_FRONTIER_LOW_MODEL = "claude-opus-4-8-frontier-low"
GEMINI_FRONTIER_LOW_MODEL = "gemini-3.1-pro-preview-frontier-low"
GROK_FRONTIER_LOW_MODEL = "grok-4.3-frontier-low"

DEFAULT_MODEL_BY_PROVIDER = {
    "openai": DEFAULT_OPENAI_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "grok": DEFAULT_GROK_MODEL,
}

GEMINI_FLASH_MODEL = DEFAULT_GEMINI_MODEL
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
GEMINI_35_FLASH_MODEL = "gemini-3.5-flash"
# Defaults fuer Nutzer OHNE Early-Tag (und ohne Pro): durchweg die guenstigen
# Basis-Modelle. Die teuren Early-Modelle (Frontier-Low + DeepSeek V4 Pro) sind
# nur noch ueber den Early-Tag erreichbar (siehe EARLY_MODELS / is_user_early).
FREE_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": DEFAULT_OPENAI_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "deepseek": DEEPSEEK_FLASH_MODEL,
    "grok": DEFAULT_GROK_MODEL,
}

# Defaults fuer Early-Nutzer: die Frontier-Low-Varianten (frueheres Free-Default).
EARLY_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": OPENAI_FRONTIER_LOW_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": ANTHROPIC_FRONTIER_LOW_MODEL,
    "gemini": GEMINI_FRONTIER_LOW_MODEL,
    "deepseek": DEEPSEEK_PRO_MODEL,
    "grok": GROK_FRONTIER_LOW_MODEL,
}

# Unveraenderliche Basis fuer den Free-Default je Provider. Der Admin kann den
# Free-Default pro Provider in Firestore (Feld "defaults") ueberschreiben; ohne
# Override gilt diese Basis (siehe apply_default_models).
_BASE_FREE_DEFAULTS = dict(FREE_DEFAULT_MODEL_BY_PROVIDER)

# Antwortmodelle fuer geplante Consensus-Watches. Pro Tier kann je Provider
# genau ein Modell aktiv sein; fehlende Legacy-Konfiguration behaelt das
# bisherige Verhalten mit drei guenstigen Modellen bei.
_BASE_WATCH_MODELS_BY_TIER = {
    "free": {
        "openai": DEFAULT_OPENAI_MODEL,
        "mistral": DEFAULT_MISTRAL_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    },
    "pro": {
        "openai": DEFAULT_OPENAI_MODEL,
        "mistral": DEFAULT_MISTRAL_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
    },
}
WATCH_MODELS_BY_TIER = {
    tier: dict(models) for tier, models in _BASE_WATCH_MODELS_BY_TIER.items()
}

# Vom Admin gepflegte Anzeige-Reihenfolge der Modelle je Provider in den normalen
# Pickern. Leere Liste => deterministischer Auto-Sort (model_picker_sort_key).
MODEL_ORDER_BY_PROVIDER: dict[str, list[str]] = {
    provider: [] for provider in DEFAULT_MODEL_BY_PROVIDER
}

# Consensus-Engine, auf die Deep Think die Synthese festkoppelt. Vom Admin
# ueber Firestore (Feld "deep_think_model") umstellbar; ungueltige Werte
# fallen auf die Basis zurueck (siehe apply_deep_think_model).
_BASE_DEEP_THINK_CONSENSUS_MODEL = GEMINI_35_FLASH_MODEL
DEEP_THINK_CONSENSUS_MODEL = _BASE_DEEP_THINK_CONSENSUS_MODEL

# Standard-Judges der Differences-/Resolve-Engine je Provider (Pro-Judges
# loesen weiterhin ueber die "<Familie>-Pro"-Aliasse auf, siehe
# consensus_engine._judge_engine). Vom Admin ueber Firestore (Feld
# "judge_models") umstellbar; ungueltige Werte fallen je Provider auf die
# Basis zurueck (siehe apply_judge_models). WICHTIG: das dict wird in-place
# mutiert, damit Modul-Aliasse (consensus_engine, resolve_engine) live bleiben.
_BASE_DIFFERENCES_JUDGE_BY_PROVIDER = {
    "openai": DEFAULT_OPENAI_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": GEMINI_FLASH_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "grok": DEFAULT_GROK_MODEL,
}
DIFFERENCES_JUDGE_MODEL_BY_PROVIDER = dict(_BASE_DIFFERENCES_JUDGE_BY_PROVIDER)

DEFAULT_CONSENSUS_MODELS = [
    GEMINI_FRONTIER_LOW_MODEL,
    GEMINI_35_FLASH_MODEL,
    "Grok",
    "OpenAI",
    "Anthropic",
    "Mistral",
    "Gemini",
    "DeepSeek",
    "Grok-Pro",
    "OpenAI-Pro",
    "Anthropic-Pro",
    "Mistral-Pro",
    "Gemini-Pro",
    "DeepSeek-Pro",
]

ALLOWED_CONSENSUS_MODELS = list(DEFAULT_CONSENSUS_MODELS)
UNSUPPORTED_GEMINI_MODELS = {
    "gemini-3.1-flash-preview",
    "gemini-3-pro-preview",
}

# Nutzerfreundliche Presets fuer den Consensus-Picker der App: Die primaere
# Picker-Ebene zeigt Eigenschafts-Optionen (Fast/Balanced/Thorough) statt
# roher Modellnamen; "Custom" oeffnet weiterhin die volle Engine-Liste.
# Jedes Preset ist eine geordnete Kandidatenliste ueber Consensus-Werte
# (Engine-Aliase oder direkte Modell-IDs) — der Client waehlt den ersten
# Kandidaten, dessen Picker-Option fuer das Tier des Nutzers freigeschaltet
# ist (Premium-/Early-Optionen sind fuer Free disabled). "balanced" bildet
# bewusst die bisherigen Tier-Defaults ab (Frontier-Low fuer Early/Pro,
# sonst Grok), damit sich das Default-Verhalten nicht aendert.
CONSENSUS_PRESETS = [
    {
        "id": "fast",
        "label": "Fast",
        "hint": "Quick synthesis for everyday questions",
        "candidates": ["Gemini", "Mistral", "Grok"],
    },
    {
        "id": "balanced",
        "label": "Balanced",
        "hint": "Reliable default for most questions",
        "candidates": [GEMINI_FRONTIER_LOW_MODEL, "Grok", "OpenAI", "Gemini"],
    },
    {
        "id": "thorough",
        "label": "Thorough",
        "hint": "Deeper reasoning, takes longer",
        "candidates": ["Gemini-Pro", "Anthropic-Pro", "OpenAI-Pro", "DeepSeek", "Anthropic"],
    },
]
DEFAULT_CONSENSUS_PRESET = "balanced"

PRO_USAGE_LIMIT = LIMITS["pro_usage_limit"]
PRO_DEEP_SEARCH_LIMIT = LIMITS["pro_deep_search_limit"]

VALID_LEADERBOARD_MODELS = {
    "OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok",
    "OpenAI-Pro", "Mistral-Pro", "Anthropic-Pro", "Gemini-Pro", "DeepSeek-Pro", "Grok-Pro",
}

CONSENSUS_ENGINE_ALIASES = {
    "OpenAI": ("openai", DEFAULT_OPENAI_MODEL),
    "OpenAI-Pro": ("openai", "gpt-5.5"),
    "Mistral": ("mistral", DEFAULT_MISTRAL_MODEL),
    "Mistral-Pro": ("mistral", MISTRAL_PRO_MODEL),
    "Anthropic": ("anthropic", DEFAULT_ANTHROPIC_MODEL),
    "Anthropic-Pro": ("anthropic", ANTHROPIC_PRO_MODEL),
    "Gemini": ("gemini", GEMINI_FLASH_MODEL),
    "Gemini-Pro": ("gemini", GEMINI_PRO_MODEL),
    "DeepSeek": ("deepseek", DEFAULT_DEEPSEEK_MODEL),
    "DeepSeek-Pro": ("deepseek", DEEPSEEK_PRO_MODEL),
    "Grok": ("grok", DEFAULT_GROK_MODEL),
    "Grok-Pro": ("grok", "grok-4.3"),
}

# Pro-Judges der Differences-/Resolve-Engine je Provider. Basis sind die
# API-Modelle der "<Familie>-Pro"-Aliasse; vom Admin ueber Firestore (Feld
# "judge_models_pro") umstellbar. Wie DIFFERENCES_JUDGE_MODEL_BY_PROVIDER
# in-place mutiert (Modul-Aliasse bleiben live).
_BASE_PRO_JUDGE_BY_PROVIDER = {
    "openai": CONSENSUS_ENGINE_ALIASES["OpenAI-Pro"][1],
    "mistral": CONSENSUS_ENGINE_ALIASES["Mistral-Pro"][1],
    "anthropic": CONSENSUS_ENGINE_ALIASES["Anthropic-Pro"][1],
    "gemini": CONSENSUS_ENGINE_ALIASES["Gemini-Pro"][1],
    "deepseek": CONSENSUS_ENGINE_ALIASES["DeepSeek-Pro"][1],
    "grok": CONSENSUS_ENGINE_ALIASES["Grok-Pro"][1],
}
PRO_JUDGE_MODEL_BY_PROVIDER = dict(_BASE_PRO_JUDGE_BY_PROVIDER)

# Familien-Prioritaet der Judge-Wahl: primaerer und Fallback-Judge nehmen die
# erste Familie mit verfuegbarem Key, die nicht die der Consensus-Engine ist.
JUDGE_FAMILY_PRIORITY = ["gemini", "openai", "mistral", "deepseek", "grok", "anthropic"]

# Optionales Admin-Mapping Engine-Familie -> bevorzugte Judge-Familie
# (Firestore-Feld "judge_families"). Fehlt ein Eintrag oder ist der Key der
# bevorzugten Familie nicht verfuegbar, greift JUDGE_FAMILY_PRIORITY (Auto).
JUDGE_FAMILY_BY_ENGINE: dict[str, str] = {}
LEADERBOARD_MODEL_ALIASES = {
    "Claude": "Anthropic",
}

ALLOWED_OPENAI_MODELS = {
    "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo",
    "gpt-5", "gpt-5-chat-latest", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.3-chat-latest", "gpt-5.4",
    "gpt-5.5", "gpt-5.4-mini", OPENAI_FRONTIER_LOW_MODEL,
}

ALLOWED_MISTRAL_MODELS = {
    "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", MISTRAL_PRO_MODEL,
    "ministral-3b-latest", "ministral-8b-latest",
}
MISTRAL_REASONING_MODELS = {DEFAULT_MISTRAL_MODEL, MISTRAL_PRO_MODEL}
DEPRECATED_MISTRAL_MODELS = {
    "devstral-small-2507", "devstral-small-latest", "devstral-medium-2507",
    "mistral-large-2411", "pixtral-large-2411", "pixtral-large-latest",
}

ALLOWED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022",
    "claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-opus-4-7", ANTHROPIC_PRO_MODEL, ANTHROPIC_FRONTIER_LOW_MODEL,
}

ALLOWED_GEMINI_MODELS = {
    GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash", "gemini-2.0-flash", GEMINI_35_FLASH_MODEL,
    GEMINI_PRO_MODEL, "gemini-2.5-pro", GEMINI_FRONTIER_LOW_MODEL,
}

ALLOWED_DEEPSEEK_MODELS = {
    DEEPSEEK_FLASH_MODEL, DEEPSEEK_PRO_MODEL,
}

ALLOWED_GROK_MODELS = {
    "grok-4-fast-non-reasoning-latest", "grok-4-1-fast-non-reasoning-latest",
    "grok-4-latest", "grok-3-latest", "grok-4-fast-reasoning-latest", "grok-4.20",
    "grok-4.20-non-reasoning", "grok-4.3", GROK_FRONTIER_LOW_MODEL,
}

FRONTIER_LOW_MODEL_IDS_BY_PROVIDER = {
    "openai": OPENAI_FRONTIER_LOW_MODEL,
    "anthropic": ANTHROPIC_FRONTIER_LOW_MODEL,
    "gemini": GEMINI_FRONTIER_LOW_MODEL,
    "grok": GROK_FRONTIER_LOW_MODEL,
}
FRONTIER_LOW_MODELS = set(FRONTIER_LOW_MODEL_IDS_BY_PROVIDER.values())
# Modelle, die sowohl Early- als auch Pro-Charakter haben (teuer, aber kein
# eigenes Frontier-Low-Mapping). Mistral Small ist hier bewusst NICHT mehr
# enthalten: es ist guenstig und bleibt ein normales Free-Modell.
EARLY_AND_PRO_MODELS = {DEEPSEEK_PRO_MODEL}
# Early-Modelle sind ab sofort tag-gated (nicht mehr gratis fuer alle): nur mit
# Early-Tag (oder Pro, das Early einschliesst) auswaehlbar.
EARLY_MODELS = FRONTIER_LOW_MODELS | EARLY_AND_PRO_MODELS
REQUIRED_PRO_MODELS = {MISTRAL_PRO_MODEL, ANTHROPIC_PRO_MODEL, GEMINI_35_FLASH_MODEL}
DEPRECATED_DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}
REQUIRED_DEEPSEEK_MODELS = {DEEPSEEK_FLASH_MODEL, DEEPSEEK_PRO_MODEL}

def ensure_default_models_allowed():
    ALLOWED_OPENAI_MODELS.add(DEFAULT_OPENAI_MODEL)
    ALLOWED_MISTRAL_MODELS.difference_update(DEPRECATED_MISTRAL_MODELS)
    ALLOWED_MISTRAL_MODELS.add(DEFAULT_MISTRAL_MODEL)
    ALLOWED_MISTRAL_MODELS.add(MISTRAL_PRO_MODEL)
    ALLOWED_ANTHROPIC_MODELS.add(DEFAULT_ANTHROPIC_MODEL)
    ALLOWED_ANTHROPIC_MODELS.add(ANTHROPIC_PRO_MODEL)
    ALLOWED_GEMINI_MODELS.add(DEFAULT_GEMINI_MODEL)
    ALLOWED_GEMINI_MODELS.add(GEMINI_35_FLASH_MODEL)
    ALLOWED_DEEPSEEK_MODELS.difference_update(DEPRECATED_DEEPSEEK_MODELS)
    ALLOWED_DEEPSEEK_MODELS.update(REQUIRED_DEEPSEEK_MODELS)
    ALLOWED_GROK_MODELS.add(DEFAULT_GROK_MODEL)
    ALLOWED_OPENAI_MODELS.add(OPENAI_FRONTIER_LOW_MODEL)
    ALLOWED_ANTHROPIC_MODELS.add(ANTHROPIC_FRONTIER_LOW_MODEL)
    ALLOWED_GEMINI_MODELS.add(GEMINI_FRONTIER_LOW_MODEL)
    ALLOWED_GROK_MODELS.add(GROK_FRONTIER_LOW_MODEL)

ensure_default_models_allowed()

PREMIUM_MODELS = {
    "gpt-5", "gpt-5-chat-latest", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.3-chat-latest", "gpt-5.4",
    "gpt-5.5",
    "claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-opus-4-7", ANTHROPIC_PRO_MODEL,
    "mistral-large-latest", "mistral-medium-latest", MISTRAL_PRO_MODEL,
    GEMINI_PRO_MODEL, "gemini-2.5-pro",
    DEEPSEEK_PRO_MODEL,
    "grok-4-latest", "grok-3-latest", "grok-4-fast-reasoning-latest", "grok-4.20",
    "grok-4.3",
}
PREMIUM_MODELS.difference_update(FRONTIER_LOW_MODELS)
PREMIUM_MODELS.difference_update(DEPRECATED_MISTRAL_MODELS)
PREMIUM_MODELS.difference_update(DEPRECATED_DEEPSEEK_MODELS)
PREMIUM_MODELS.update(EARLY_AND_PRO_MODELS)
PREMIUM_MODELS.update(REQUIRED_PRO_MODELS)

ALL_ALLOWED_MODELS = (
    ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
    ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
)

MODEL_LABEL_OVERRIDES = {
    "gpt-5.5": "GPT-5.5",
    OPENAI_FRONTIER_LOW_MODEL: "GPT-5.5",
    DEFAULT_OPENAI_MODEL: "GPT-5.4 mini",
    DEFAULT_ANTHROPIC_MODEL: "Claude Haiku 4.5",
    DEFAULT_GEMINI_MODEL: "Gemini 3.1 Flash-Lite",
    DEFAULT_GROK_MODEL: "Grok 4.20",
    "mistral-small-latest": "Mistral Small 4",
    MISTRAL_PRO_MODEL: "Mistral Medium 3.5",
    "claude-opus-4-7": "Claude Opus 4.7",
    ANTHROPIC_PRO_MODEL: "Claude Opus 4.8",
    ANTHROPIC_FRONTIER_LOW_MODEL: "Claude Opus 4.8",
    GEMINI_35_FLASH_MODEL: "Gemini 3.5 Flash",
    GEMINI_PRO_MODEL: "Gemini 3.1",
    GEMINI_FRONTIER_LOW_MODEL: "Gemini 3.1",
    "grok-4.3": "Grok 4.3",
    GROK_FRONTIER_LOW_MODEL: "Grok 4.3",
    DEEPSEEK_FLASH_MODEL: "DeepSeek V4 Flash",
    DEEPSEEK_PRO_MODEL: "DeepSeek V4 Pro",
}

MODEL_CONFIGS: dict[str, ModelConfig] = {}


def _fallback_label(model_id: str) -> str:
    return MODEL_LABEL_OVERRIDES.get(model_id, model_id)


def _provider_allowed_sets() -> dict[str, set]:
    return {
        "openai": ALLOWED_OPENAI_MODELS,
        "mistral": ALLOWED_MISTRAL_MODELS,
        "anthropic": ALLOWED_ANTHROPIC_MODELS,
        "gemini": ALLOWED_GEMINI_MODELS,
        "deepseek": ALLOWED_DEEPSEEK_MODELS,
        "grok": ALLOWED_GROK_MODELS,
    }


def rebuild_model_configs():
    MODEL_CONFIGS.clear()
    for provider, models in _provider_allowed_sets().items():
        for model_id in models:
            MODEL_CONFIGS[model_id] = ModelConfig(
                internal_id=model_id,
                provider=provider,
                api_model=model_id,
                label=_fallback_label(model_id),
                is_free=model_id not in PREMIUM_MODELS,
                is_pro=model_id in PREMIUM_MODELS,
            )

    MODEL_CONFIGS.update({
        OPENAI_FRONTIER_LOW_MODEL: ModelConfig(
            internal_id=OPENAI_FRONTIER_LOW_MODEL,
            provider="openai",
            api_model="gpt-5.5",
            label="GPT-5.5",
            is_free=True,
            is_frontier=True,
            is_low_reasoning=True,
            low_config={"reasoning": {"effort": "low"}},
        ),
        ANTHROPIC_FRONTIER_LOW_MODEL: ModelConfig(
            internal_id=ANTHROPIC_FRONTIER_LOW_MODEL,
            provider="anthropic",
            api_model=ANTHROPIC_PRO_MODEL,
            label="Claude Opus 4.8",
            is_free=True,
            is_frontier=True,
            is_low_reasoning=True,
            low_config={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            },
        ),
        GEMINI_FRONTIER_LOW_MODEL: ModelConfig(
            internal_id=GEMINI_FRONTIER_LOW_MODEL,
            provider="gemini",
            api_model=GEMINI_PRO_MODEL,
            label="Gemini 3.1",
            is_free=True,
            is_frontier=True,
            is_low_reasoning=True,
            low_config={"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}},
        ),
        GROK_FRONTIER_LOW_MODEL: ModelConfig(
            internal_id=GROK_FRONTIER_LOW_MODEL,
            provider="grok",
            api_model="grok-4.3",
            label="Grok 4.3",
            is_free=True,
            is_frontier=True,
            is_low_reasoning=True,
            low_config={"reasoning": {"effort": "low"}},
        ),
        DEEPSEEK_PRO_MODEL: ModelConfig(
            internal_id=DEEPSEEK_PRO_MODEL,
            provider="deepseek",
            api_model=DEEPSEEK_PRO_MODEL,
            label=_fallback_label(DEEPSEEK_PRO_MODEL),
            is_free=True,
            is_pro=True,
            is_frontier=True,
        ),
    })


def get_model_config(model_id: str | None, provider: str | None = None) -> ModelConfig | None:
    if not model_id:
        return None
    config = MODEL_CONFIGS.get(model_id)
    if config:
        return config
    return ModelConfig(
        internal_id=model_id,
        provider=provider or "",
        api_model=model_id,
        label=_fallback_label(model_id),
        is_free=model_id not in PREMIUM_MODELS,
        is_pro=model_id in PREMIUM_MODELS,
    )


def resolve_api_model(model_id: str | None, default_model: str, provider: str) -> tuple[str, ModelConfig]:
    selected_model = model_id or default_model
    config = get_model_config(selected_model, provider) or get_model_config(default_model, provider)
    return config.api_model, config


def get_model_label(model_id: str) -> str:
    config = get_model_config(model_id)
    return config.label if config else _fallback_label(model_id)


def get_model_badge(model_id: str) -> str:
    config = get_model_config(model_id)
    badges = []
    if config and config.is_frontier:
        badges.append("Early")
    if (config and config.is_pro) or model_id in PREMIUM_MODELS:
        badges.append("Pro")
    return " · ".join(dict.fromkeys(badges))


def get_consensus_model_config(model_id: str | None) -> ModelConfig | None:
    if not model_id:
        return None
    alias = CONSENSUS_ENGINE_ALIASES.get(model_id)
    if alias:
        provider, api_model = alias
        return ModelConfig(
            internal_id=model_id,
            provider=provider,
            api_model=api_model,
            label=_fallback_label(api_model),
            is_free=not str(model_id).endswith("-Pro"),
            is_pro=str(model_id).endswith("-Pro"),
        )
    return get_model_config(model_id)


def is_premium_consensus_model(model_id: str | None) -> bool:
    config = get_consensus_model_config(model_id)
    return bool(config and config.is_pro and config.internal_id not in EARLY_MODELS)


def is_early_consensus_model(model_id: str | None) -> bool:
    """True, wenn die gewaehlte Consensus-Engine ein tag-gated Early-Modell ist
    (aktuell nur das Gemini-Frontier-Low). Erfordert Early- oder Pro-Zugang."""
    return bool(model_id) and model_id in EARLY_MODELS


def get_consensus_model_label(model_id: str) -> str:
    config = get_consensus_model_config(model_id)
    return config.label if config else _fallback_label(model_id)


def get_consensus_model_badge(model_id: str) -> str:
    if str(model_id or "").endswith("-Pro"):
        return "Pro"
    config = get_consensus_model_config(model_id)
    badges = []
    if config and config.is_frontier:
        badges.append("Early")
    if config and config.is_pro and config.internal_id not in EARLY_MODELS:
        badges.append("Pro")
    return " · ".join(dict.fromkeys(badges))


def get_consensus_presets() -> list[dict]:
    """Presets fuer den App-Consensus-Picker, gefiltert auf die aktuell
    konfigurierte Consensus-Liste (der Admin kann Engines entfernen).
    Presets ohne verbleibende Kandidaten werden ausgelassen; der Client
    faellt dann auf die Custom-Liste zurueck."""
    presets = []
    for preset in CONSENSUS_PRESETS:
        candidates = [
            model for model in preset["candidates"]
            if model in ALLOWED_CONSENSUS_MODELS
        ]
        if not candidates:
            continue
        presets.append({**preset, "candidates": candidates})
    return presets


def normalize_consensus_models(models) -> list[str]:
    incoming = [str(model).strip() for model in (models or []) if str(model or "").strip()]
    if not incoming:
        incoming = list(DEFAULT_CONSENSUS_MODELS)
    allowed = []
    for model in incoming:
        if model in allowed:
            continue
        config = get_consensus_model_config(model)
        if config and config.provider:
            allowed.append(model)
    if GEMINI_FRONTIER_LOW_MODEL not in allowed:
        allowed.insert(0, GEMINI_FRONTIER_LOW_MODEL)
    # Deep Think koppelt die Synthese fest an das konfigurierte Deep-Think-
    # Modell (Basis: Gemini 3.5 Flash). Deshalb muss das Modell auch bei einer
    # Admin-/Firestore-Liste ohne diesen Eintrag als Consensus-Option
    # verfuegbar bleiben.
    if DEEP_THINK_CONSENSUS_MODEL not in allowed:
        allowed.append(DEEP_THINK_CONSENSUS_MODEL)
    return allowed


def get_deep_think_consensus_model() -> str:
    return DEEP_THINK_CONSENSUS_MODEL


def is_valid_deep_think_model(model_id) -> bool:
    """Gueltig ist jeder Consensus-Wert (Alias oder direkte Modell-ID), der
    sich auf einen Provider aufloesen laesst."""
    chosen = str(model_id or "").strip()
    if not chosen:
        return False
    config = get_consensus_model_config(chosen)
    return bool(config and config.provider)


def apply_deep_think_model(model_id) -> None:
    """Setzt die Deep-Think-Consensus-Engine. Ungueltige/leere Werte fallen
    auf die Basis (Gemini 3.5 Flash) zurueck."""
    global DEEP_THINK_CONSENSUS_MODEL
    chosen = str(model_id or "").strip()
    if chosen and is_valid_deep_think_model(chosen):
        DEEP_THINK_CONSENSUS_MODEL = chosen
    else:
        DEEP_THINK_CONSENSUS_MODEL = _BASE_DEEP_THINK_CONSENSUS_MODEL


def is_valid_judge_model(provider: str, model_id) -> bool:
    """Gueltiger Standard-Judge: erlaubtes Modell des Providers, das direkt
    als API-Modell aufrufbar ist. Frontier-Low-IDs sind interne Aliasse
    (api_model != internal_id) und deshalb hier ausgeschlossen."""
    chosen = str(model_id or "").strip()
    if not chosen or chosen in FRONTIER_LOW_MODELS:
        return False
    return chosen in _provider_allowed_sets().get(provider, set())


def apply_judge_models(overrides: dict | None) -> None:
    """Setzt den Standard-Differences-Judge je Provider. Ungueltige/fehlende
    Werte fallen je Provider auf die Basis zurueck. Mutiert das dict in-place
    (Modul-Aliasse in consensus_engine/resolve_engine bleiben live)."""
    data = overrides if isinstance(overrides, dict) else {}
    for provider, base in _BASE_DIFFERENCES_JUDGE_BY_PROVIDER.items():
        chosen = str(data.get(provider) or "").strip()
        if chosen and is_valid_judge_model(provider, chosen):
            DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider] = chosen
        else:
            DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider] = base


def get_judge_models() -> dict:
    return dict(DIFFERENCES_JUDGE_MODEL_BY_PROVIDER)


def apply_pro_judge_models(overrides: dict | None) -> None:
    """Setzt den Pro-Differences-Judge je Provider. Ungueltige/fehlende Werte
    fallen je Provider auf die Basis (API-Modell des "<Familie>-Pro"-Alias)
    zurueck. Mutiert das dict in-place."""
    data = overrides if isinstance(overrides, dict) else {}
    for provider, base in _BASE_PRO_JUDGE_BY_PROVIDER.items():
        chosen = str(data.get(provider) or "").strip()
        if chosen and is_valid_judge_model(provider, chosen):
            PRO_JUDGE_MODEL_BY_PROVIDER[provider] = chosen
        else:
            PRO_JUDGE_MODEL_BY_PROVIDER[provider] = base


def get_pro_judge_models() -> dict:
    return dict(PRO_JUDGE_MODEL_BY_PROVIDER)


def apply_judge_families(overrides: dict | None) -> None:
    """Setzt das Mapping Engine-Familie -> bevorzugte Judge-Familie. Gueltig
    sind nur bekannte Provider, die sich von der Engine-Familie unterscheiden
    (Anti-Self-Judging); alles andere faellt auf Auto (Prioritaetsliste)
    zurueck. Mutiert das dict in-place."""
    data = overrides if isinstance(overrides, dict) else {}
    providers = set(_BASE_DIFFERENCES_JUDGE_BY_PROVIDER)
    JUDGE_FAMILY_BY_ENGINE.clear()
    for engine_provider in providers:
        chosen = str(data.get(engine_provider) or "").strip()
        if chosen in providers and chosen != engine_provider:
            JUDGE_FAMILY_BY_ENGINE[engine_provider] = chosen


def get_judge_families() -> dict:
    return dict(JUDGE_FAMILY_BY_ENGINE)


def get_model_picker_metadata() -> dict[str, dict[str, str]]:
    return {
        model_id: {
            "label": get_model_label(model_id),
            "badge": get_model_badge(model_id),
        }
        for model_id in ALL_ALLOWED_MODELS
    }


def model_picker_sort_key(model_id: str):
    config = get_model_config(model_id)
    label = config.label if config else model_id
    is_premium = model_id in PREMIUM_MODELS and model_id not in EARLY_MODELS
    is_frontier = bool(config and config.is_frontier)
    return (is_premium, label.lower(), not is_frontier, model_id.lower())


def get_ordered_models(provider: str) -> list[str]:
    """Modelle eines Providers in Anzeige-Reihenfolge fuer die normalen Picker.
    Die vom Admin gepflegte Reihenfolge (MODEL_ORDER_BY_PROVIDER) gewinnt; alle
    erlaubten Modelle ohne explizite Position werden deterministisch angehaengt,
    damit neu hinzugefuegte Modelle nie verschwinden."""
    allowed = _provider_allowed_sets().get(provider, set())
    ordered = [model for model in MODEL_ORDER_BY_PROVIDER.get(provider, []) if model in allowed]
    seen = set(ordered)
    rest = sorted((model for model in allowed if model not in seen), key=model_picker_sort_key)
    return ordered + rest


def apply_model_order(order_by_provider: dict | None) -> None:
    """Uebernimmt die Admin-Reihenfolge je Provider (auf erlaubte Modelle gefiltert)."""
    data = order_by_provider or {}
    allowed_sets = _provider_allowed_sets()
    for provider in MODEL_ORDER_BY_PROVIDER:
        incoming = data.get(provider)
        allowed = allowed_sets.get(provider, set())
        if isinstance(incoming, list):
            seen = set()
            ordered = []
            for model in incoming:
                model = str(model)
                if model in allowed and model not in seen:
                    seen.add(model)
                    ordered.append(model)
            MODEL_ORDER_BY_PROVIDER[provider] = ordered
        else:
            MODEL_ORDER_BY_PROVIDER[provider] = []


def apply_default_models(defaults: dict | None) -> None:
    """Setzt den Free-Default je Provider. Ein Override gilt nur, wenn das Modell
    erlaubt und weder Premium noch Early ist (sonst saehe ein eingeloggter
    Free-Nutzer einen gesperrten Default). Sonst greift die Basis."""
    overrides = defaults or {}
    allowed_sets = _provider_allowed_sets()
    for provider, base in _BASE_FREE_DEFAULTS.items():
        chosen = str(overrides.get(provider) or "").strip()
        allowed = allowed_sets.get(provider, set())
        if chosen and chosen in allowed and chosen not in PREMIUM_MODELS and chosen not in EARLY_MODELS:
            FREE_DEFAULT_MODEL_BY_PROVIDER[provider] = chosen
        else:
            FREE_DEFAULT_MODEL_BY_PROVIDER[provider] = base


def apply_watch_models(config: dict | None) -> None:
    """Apply validated per-tier Watch model mappings with legacy fallbacks."""
    incoming = config if isinstance(config, dict) else {}
    allowed_sets = _provider_allowed_sets()
    for tier in ("free", "pro"):
        tier_data = incoming.get(tier)
        tier_data = tier_data if isinstance(tier_data, dict) else {}
        clean = {}
        for provider in DEFAULT_MODEL_BY_PROVIDER:
            model = str(tier_data.get(provider) or "").strip()
            if not model or model not in allowed_sets.get(provider, set()):
                continue
            if tier == "free" and (model in PREMIUM_MODELS or model in EARLY_MODELS):
                continue
            clean[provider] = model
        if len(clean) < 2:
            clean = dict(_BASE_WATCH_MODELS_BY_TIER[tier])
        WATCH_MODELS_BY_TIER[tier].clear()
        WATCH_MODELS_BY_TIER[tier].update(clean)


def get_watch_models(is_pro: bool) -> dict[str, str]:
    return dict(WATCH_MODELS_BY_TIER["pro" if is_pro else "free"])


rebuild_model_configs()


def _coerce_limit(value, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def _sync_limit_constants():
    global FREE_USAGE_LIMIT, PRO_USAGE_LIMIT, PRO_DEEP_SEARCH_LIMIT
    global MAX_WORDS, DEEP_SEARCH_MAX_WORDS, MAX_TOKENS, DEEP_SEARCH_MAX_TOKENS
    global CONSENSUS_MAX_TOKENS, DIFFERENCES_MAX_TOKENS
    global GEMINI_MAX_TOKENS, GEMINI_DEEP_MAX_TOKENS

    FREE_USAGE_LIMIT = LIMITS["free_usage_limit"]
    PRO_USAGE_LIMIT = LIMITS["pro_usage_limit"]
    PRO_DEEP_SEARCH_LIMIT = LIMITS["pro_deep_search_limit"]
    MAX_WORDS = LIMITS["free_max_words"]
    DEEP_SEARCH_MAX_WORDS = LIMITS["pro_deep_search_max_words"]
    MAX_TOKENS = LIMITS["pro_max_tokens"]
    DEEP_SEARCH_MAX_TOKENS = LIMITS["pro_deep_search_max_tokens"]
    CONSENSUS_MAX_TOKENS = LIMITS["consensus_max_tokens"]
    DIFFERENCES_MAX_TOKENS = LIMITS["differences_max_tokens"]
    GEMINI_MAX_TOKENS = MAX_TOKENS
    GEMINI_DEEP_MAX_TOKENS = DEEP_SEARCH_MAX_TOKENS


def apply_limits(limits_data=None):
    incoming = limits_data if isinstance(limits_data, dict) else {}
    normalized = {}
    for key, fallback in DEFAULT_LIMITS.items():
        normalized[key] = _coerce_limit(incoming.get(key, fallback), fallback)

    LIMITS.clear()
    LIMITS.update(normalized)
    _sync_limit_constants()


def get_limits_config() -> dict:
    return dict(LIMITS)


def get_usage_limit(is_pro: bool) -> int:
    return LIMITS["pro_usage_limit"] if is_pro else LIMITS["free_usage_limit"]


def get_deep_search_limit(is_pro: bool) -> int:
    return LIMITS["pro_deep_search_limit"] if is_pro else LIMITS["free_deep_search_limit"]


def get_word_limit(is_pro: bool, deep_search: bool = False) -> int:
    if deep_search:
        key = "pro_deep_search_max_words" if is_pro else "free_deep_search_max_words"
    else:
        key = "pro_max_words" if is_pro else "free_max_words"
    return LIMITS[key]


def get_consensus_answer_char_limit() -> int:
    return LIMITS["consensus_max_answer_chars"]


def get_consensus_question_char_limit() -> int:
    return LIMITS["consensus_max_question_chars"]


def get_followup_question_char_limit() -> int:
    return LIMITS["followup_max_question_chars"]


def get_followup_consensus_char_limit() -> int:
    return LIMITS["followup_max_consensus_chars"]


def get_watch_active_limit(is_pro: bool) -> int:
    key = "watch_pro_active_limit" if is_pro else "watch_free_active_limit"
    return max(0, int(LIMITS[key]))


def get_watch_max_runs_per_day() -> int:
    return max(0, int(LIMITS["watch_max_runs_per_day"]))


def get_output_token_limit(is_pro: bool, deep_search: bool = False) -> int:
    if deep_search:
        key = "pro_deep_search_max_tokens" if is_pro else "free_deep_search_max_tokens"
    else:
        key = "pro_max_tokens" if is_pro else "free_max_tokens"
    return LIMITS[key]


def load_models_from_db():
    global ALL_ALLOWED_MODELS
    import logging
    from app.core.security import db_firestore
    try:
        doc_ref = db_firestore.collection("app_config").document("models")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            
            # Update OpenAI
            if "openai" in data:
                ALLOWED_OPENAI_MODELS.clear()
                ALLOWED_OPENAI_MODELS.update(data["openai"])
            
            # Update Mistral
            if "mistral" in data:
                ALLOWED_MISTRAL_MODELS.clear()
                ALLOWED_MISTRAL_MODELS.update(data["mistral"])
                ALLOWED_MISTRAL_MODELS.difference_update(DEPRECATED_MISTRAL_MODELS)
                ALLOWED_MISTRAL_MODELS.update({DEFAULT_MISTRAL_MODEL, MISTRAL_PRO_MODEL})
            
            # Update Anthropic
            if "anthropic" in data:
                ALLOWED_ANTHROPIC_MODELS.clear()
                ALLOWED_ANTHROPIC_MODELS.update(data["anthropic"])
                ALLOWED_ANTHROPIC_MODELS.update({DEFAULT_ANTHROPIC_MODEL, ANTHROPIC_PRO_MODEL})
            
            # Update Gemini
            if "gemini" in data:
                ALLOWED_GEMINI_MODELS.clear()
                ALLOWED_GEMINI_MODELS.update(data["gemini"])
                ALLOWED_GEMINI_MODELS.difference_update(UNSUPPORTED_GEMINI_MODELS)
                ALLOWED_GEMINI_MODELS.update({GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL})
            
            # Update DeepSeek
            if "deepseek" in data:
                ALLOWED_DEEPSEEK_MODELS.clear()
                ALLOWED_DEEPSEEK_MODELS.update(data["deepseek"])
                ALLOWED_DEEPSEEK_MODELS.difference_update(DEPRECATED_DEEPSEEK_MODELS)
                ALLOWED_DEEPSEEK_MODELS.update(REQUIRED_DEEPSEEK_MODELS)
            
            # Update Grok
            if "grok" in data:
                ALLOWED_GROK_MODELS.clear()
                ALLOWED_GROK_MODELS.update(data["grok"])

            ensure_default_models_allowed()
            
            # Update Premium
            if "premium" in data:
                PREMIUM_MODELS.clear()
                PREMIUM_MODELS.update(data["premium"])
                PREMIUM_MODELS.difference_update(UNSUPPORTED_GEMINI_MODELS)
                PREMIUM_MODELS.difference_update(FRONTIER_LOW_MODELS)
                PREMIUM_MODELS.difference_update(DEPRECATED_MISTRAL_MODELS)
                PREMIUM_MODELS.add(GEMINI_PRO_MODEL)
                PREMIUM_MODELS.difference_update(DEPRECATED_DEEPSEEK_MODELS)
                PREMIUM_MODELS.update(EARLY_AND_PRO_MODELS)
                PREMIUM_MODELS.update(REQUIRED_PRO_MODELS)

            ensure_default_models_allowed()
            ALL_ALLOWED_MODELS = (
                ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
                ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
            )
            rebuild_model_configs()

            # Deep-Think-Modell VOR der Consensus-Normalisierung anwenden,
            # damit normalize_consensus_models das konfigurierte Modell in der
            # Liste sicherstellt.
            apply_deep_think_model(data.get("deep_think_model"))

            # Judges (Differences/Resolve) je Provider; braucht die
            # aktualisierten Provider-Listen fuer die Validierung.
            apply_judge_models(data.get("judge_models"))
            apply_pro_judge_models(data.get("judge_models_pro"))
            apply_judge_families(data.get("judge_families"))

            if "consensus" in data:
                ALLOWED_CONSENSUS_MODELS.clear()
                ALLOWED_CONSENSUS_MODELS.extend(normalize_consensus_models(data["consensus"]))
            else:
                ALLOWED_CONSENSUS_MODELS.clear()
                ALLOWED_CONSENSUS_MODELS.extend(normalize_consensus_models(DEFAULT_CONSENSUS_MODELS))

            # Admin-gepflegte Picker-Reihenfolge (aus den geordneten Provider-Listen)
            # und Free-Default je Provider uebernehmen.
            apply_model_order({provider: data.get(provider) for provider in MODEL_ORDER_BY_PROVIDER})
            apply_default_models(data.get("defaults"))
            apply_watch_models(data.get("watch_models"))

            apply_limits(data.get("limits"))
            ensure_default_models_allowed()
            
            # Update ALL_ALLOWED_MODELS
            ALL_ALLOWED_MODELS = (
                ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
                ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
            )
            rebuild_model_configs()
            logging.info("Models configuration loaded from Firestore successfully.")
        else:
            # If document doesn't exist, create it with default values
            doc_ref.set({
                "openai": list(ALLOWED_OPENAI_MODELS),
                "mistral": list(ALLOWED_MISTRAL_MODELS),
                "anthropic": list(ALLOWED_ANTHROPIC_MODELS),
                "gemini": list(ALLOWED_GEMINI_MODELS),
                "deepseek": list(ALLOWED_DEEPSEEK_MODELS),
                "grok": list(ALLOWED_GROK_MODELS),
                "premium": list(PREMIUM_MODELS),
                "consensus": list(ALLOWED_CONSENSUS_MODELS),
                "deep_think_model": DEEP_THINK_CONSENSUS_MODEL,
                "judge_models": get_judge_models(),
                "judge_models_pro": get_pro_judge_models(),
                "judge_families": get_judge_families(),
                "watch_models": {
                    tier: dict(models) for tier, models in WATCH_MODELS_BY_TIER.items()
                },
                "limits": get_limits_config()
            })
            rebuild_model_configs()
            logging.info("Created default models configuration in Firestore.")
    except Exception as e:
        logging.error(f"Failed to load models from Firestore: {e}")

DEEP_THINK_PROMPT = "Deep Think: Focus as hard as you can! But only on the essentials."
