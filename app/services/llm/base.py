from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import HTTPException
from app.core.config import PREMIUM_MODELS

def get_system_prompt() -> str:
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    today_str = now.strftime("%A, %Y-%m-%d")
    return (
        f"Today is {today_str}. "
        "Please respond briefly and precisely, focusing only on the essentials. No follow-up questions."
    )

def count_words(text: str) -> int:
    return len(text.strip().split())

def validate_model(model: str, allowed: set, provider: str, is_pro: bool = False):
    if model and model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not allowed for {provider}."
        )
    
    if model in PREMIUM_MODELS and not is_pro:
        raise HTTPException(
            status_code=403,
            detail=f"The model '{model}' is reserved for Premium users. Please upgrade your plan."
        )
