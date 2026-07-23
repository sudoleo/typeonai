from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
from fastapi import HTTPException
import app.core.config as cfg

def get_system_prompt() -> str:
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    today_str = now.strftime("%A, %Y-%m-%d")
    return (
        f"Today is {today_str}. "
        "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify. No follow-up questions."
    )

FOLLOWUP_CONTEXT_HEADER = "PREVIOUS EXCHANGE (context for a follow-up question):"


def build_followup_system_prompt(base_prompt: str, previous_question: str, previous_consensus: str) -> str:
    """Injiziert genau eine vorherige Frage/Konsens-Ebene vor den System-Prompt.
    Als Kontext geht bewusst nur der Konsens-Text mit, nicht die einzelnen
    Modellantworten (Kostenkontrolle)."""
    return (
        f"{FOLLOWUP_CONTEXT_HEADER}\n"
        f"Previous question: {previous_question}\n"
        f"Consensus answer to the previous question:\n{previous_consensus}\n"
        "END OF PREVIOUS EXCHANGE.\n\n"
        "INSTRUCTIONS:\n"
        "The user's current question is a follow-up to the exchange above. Resolve references "
        "(such as 'it', 'that approach', 'the second option') against that exchange and stay "
        "consistent with it, but answer the current question directly and on its own merits.\n\n"
        f"{base_prompt}"
    )


def count_words(text: Optional[str]) -> int:
    return len((text or "").strip().split())

def validate_model(model: str, allowed: set, provider: str, is_pro: bool = False):
    if model and model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not allowed for {provider}."
        )

    model_config = cfg.get_model_config(model)
    is_pro_only = (
        model in cfg.PREMIUM_MODELS
        and not (model_config and model_config.is_free)
    )
    if is_pro_only and not is_pro:
        raise HTTPException(
            status_code=403,
            detail=f"The model '{model}' is reserved for Premium users. Please upgrade your plan."
        )
