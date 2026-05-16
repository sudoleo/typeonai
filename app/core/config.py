import os
from dotenv import load_dotenv

load_dotenv()

FREE_USAGE_LIMIT = 25
MAX_WORDS = 500
DEEP_SEARCH_MAX_WORDS = 1000
MAX_TOKENS = 4096
DEEP_SEARCH_MAX_TOKENS = 8192
CONSENSUS_MAX_TOKENS = 8192
DIFFERENCES_MAX_TOKENS = 4096
REASONING_EFFORT_FOR_DEEP = "low"
GEMINI_MAX_TOKENS = 4096
GEMINI_DEEP_MAX_TOKENS = 8192
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
UNSUPPORTED_GEMINI_MODELS = {
    "gemini-3.1-flash-preview",
    "gemini-3-pro-preview",
}

PRO_USAGE_LIMIT = 500
PRO_DEEP_SEARCH_LIMIT = 50

VALID_LEADERBOARD_MODELS = {
    "OpenAI", "Mistral", "Claude", "Gemini", "DeepSeek", "Grok", 
    "OpenAI-Pro", "Mistral-Pro", "Anthropic-Pro", "Gemini-Pro", "DeepSeek-Pro", "Grok-Pro",
    "Exa"
}

ALLOWED_OPENAI_MODELS = {
    "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo",
    "gpt-5", "gpt-5-chat-latest", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.3-chat-latest", "gpt-5.4",
    "gpt-5.5", "gpt-5.4-mini",
}

ALLOWED_MISTRAL_MODELS = {
    "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
    "ministral-3b-latest", "ministral-8b-latest", "pixtral-large-latest",
    "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",
}

ALLOWED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022",
    "claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6",
    "claude-opus-4-7",
}

ALLOWED_GEMINI_MODELS = {
    GEMINI_FLASH_MODEL, "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash", "gemini-2.0-flash",
    GEMINI_PRO_MODEL, "gemini-2.5-pro",
}

ALLOWED_DEEPSEEK_MODELS = {
    "deepseek-chat", "deepseek-reasoner",
    "deepseek-v4-flash", "deepseek-v4-pro",
}

ALLOWED_GROK_MODELS = {
    "grok-4-fast-non-reasoning-latest", "grok-4-1-fast-non-reasoning-latest",
    "grok-4-latest", "grok-3-latest", "grok-4-fast-reasoning-latest", "grok-4.20",
    "grok-4.20-non-reasoning", "grok-4.3",
}

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

ALL_ALLOWED_MODELS = (
    ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
    ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
)

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
            
            # Update Premium
            if "premium" in data:
                PREMIUM_MODELS.clear()
                PREMIUM_MODELS.update(data["premium"])
                PREMIUM_MODELS.difference_update(UNSUPPORTED_GEMINI_MODELS)
                PREMIUM_MODELS.add(GEMINI_PRO_MODEL)
            
            # Update ALL_ALLOWED_MODELS
            global ALL_ALLOWED_MODELS
            ALL_ALLOWED_MODELS = (
                ALLOWED_OPENAI_MODELS | ALLOWED_MISTRAL_MODELS | ALLOWED_ANTHROPIC_MODELS |
                ALLOWED_GEMINI_MODELS | ALLOWED_DEEPSEEK_MODELS | ALLOWED_GROK_MODELS
            )
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
                "premium": list(PREMIUM_MODELS)
            })
            logging.info("Created default models configuration in Firestore.")
    except Exception as e:
        logging.error(f"Failed to load models from Firestore: {e}")

DEEP_THINK_PROMPT = "Deep Think: Focus as hard as you can! But only on the essentials."
