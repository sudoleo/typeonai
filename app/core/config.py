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
    "differences_max_tokens": 4096,
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
DEFAULT_MISTRAL_MODEL = "mistral-medium-latest"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_GROK_MODEL = "grok-4.20-non-reasoning"

OPENAI_FRONTIER_LOW_MODEL = "gpt-5.5-frontier-low"
ANTHROPIC_FRONTIER_LOW_MODEL = "claude-opus-4-7-frontier-low"
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
FREE_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": OPENAI_FRONTIER_LOW_MODEL,
    "mistral": DEFAULT_MISTRAL_MODEL,
    "anthropic": ANTHROPIC_FRONTIER_LOW_MODEL,
    "gemini": GEMINI_FRONTIER_LOW_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "grok": GROK_FRONTIER_LOW_MODEL,
}
UNSUPPORTED_GEMINI_MODELS = {
    "gemini-3.1-flash-preview",
    "gemini-3-pro-preview",
}

PRO_USAGE_LIMIT = LIMITS["pro_usage_limit"]
PRO_DEEP_SEARCH_LIMIT = LIMITS["pro_deep_search_limit"]

VALID_LEADERBOARD_MODELS = {
    "OpenAI", "Mistral", "Claude", "Gemini", "DeepSeek", "Grok",
    "OpenAI-Pro", "Mistral-Pro", "Anthropic-Pro", "Gemini-Pro", "DeepSeek-Pro", "Grok-Pro",
}

ALLOWED_OPENAI_MODELS = {
    "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo",
    "gpt-5", "gpt-5-chat-latest", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.3-chat-latest", "gpt-5.4",
    "gpt-5.5", "gpt-5.4-mini", OPENAI_FRONTIER_LOW_MODEL,
}

ALLOWED_MISTRAL_MODELS = {
    "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
    "ministral-3b-latest", "ministral-8b-latest", "pixtral-large-latest",
    "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",
}

ALLOWED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022",
    "claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-opus-4-7", ANTHROPIC_FRONTIER_LOW_MODEL,
}

ALLOWED_GEMINI_MODELS = {
    GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash", "gemini-2.0-flash",
    GEMINI_PRO_MODEL, "gemini-2.5-pro", GEMINI_FRONTIER_LOW_MODEL,
}

ALLOWED_DEEPSEEK_MODELS = {
    "deepseek-chat", "deepseek-reasoner",
    "deepseek-v4-flash", "deepseek-v4-pro",
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

def ensure_default_models_allowed():
    ALLOWED_OPENAI_MODELS.add(DEFAULT_OPENAI_MODEL)
    ALLOWED_MISTRAL_MODELS.add(DEFAULT_MISTRAL_MODEL)
    ALLOWED_ANTHROPIC_MODELS.add(DEFAULT_ANTHROPIC_MODEL)
    ALLOWED_GEMINI_MODELS.add(DEFAULT_GEMINI_MODEL)
    ALLOWED_DEEPSEEK_MODELS.add(DEFAULT_DEEPSEEK_MODEL)
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
    "claude-opus-4-7",
    "pixtral-large-latest", "mistral-large-latest",
    GEMINI_PRO_MODEL, "gemini-2.5-pro",
    "deepseek-reasoner", "deepseek-v4-pro",
    "grok-4-latest", "grok-3-latest", "grok-4-fast-reasoning-latest", "grok-4.20",
    "grok-4.3",
}
PREMIUM_MODELS.difference_update(FRONTIER_LOW_MODELS)

ALL_ALLOWED_MODELS = (
    ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
    ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
)

MODEL_LABEL_OVERRIDES = {
    "gpt-5.5": "GPT-5.5",
    OPENAI_FRONTIER_LOW_MODEL: "GPT-5.5",
    "claude-opus-4-7": "Claude Opus 4.7",
    ANTHROPIC_FRONTIER_LOW_MODEL: "Claude Opus 4.7",
    GEMINI_PRO_MODEL: "Gemini 3.1",
    GEMINI_FRONTIER_LOW_MODEL: "Gemini 3.1",
    "grok-4.3": "Grok 4.3",
    GROK_FRONTIER_LOW_MODEL: "Grok 4.3",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
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
            api_model="claude-opus-4-7",
            label="Claude Opus 4.7",
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
    if config and config.is_frontier:
        return "Early"
    if model_id in PREMIUM_MODELS:
        return "Pro"
    return ""


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
    is_premium = model_id in PREMIUM_MODELS
    is_frontier = bool(config and config.is_frontier)
    return (is_premium, label.lower(), not is_frontier, model_id.lower())


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


def get_output_token_limit(is_pro: bool, deep_search: bool = False) -> int:
    if deep_search:
        key = "pro_deep_search_max_tokens" if is_pro else "free_deep_search_max_tokens"
    else:
        key = "pro_max_tokens" if is_pro else "free_max_tokens"
    return LIMITS[key]


def load_models_from_db():
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
            
            # Update Anthropic
            if "anthropic" in data:
                ALLOWED_ANTHROPIC_MODELS.clear()
                ALLOWED_ANTHROPIC_MODELS.update(data["anthropic"])
            
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
                PREMIUM_MODELS.add(GEMINI_PRO_MODEL)

            apply_limits(data.get("limits"))
            ensure_default_models_allowed()
            
            # Update ALL_ALLOWED_MODELS
            global ALL_ALLOWED_MODELS
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
                "limits": get_limits_config()
            })
            rebuild_model_configs()
            logging.info("Created default models configuration in Firestore.")
    except Exception as e:
        logging.error(f"Failed to load models from Firestore: {e}")

DEEP_THINK_PROMPT = "Deep Think: Focus as hard as you can! But only on the essentials."
