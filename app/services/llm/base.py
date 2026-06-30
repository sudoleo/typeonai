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

def count_words(text: Optional[str]) -> int:
    return len((text or "").strip().split())

def validate_model(model: str, allowed: set, provider: str, is_pro: bool = False, is_early: bool = False):
    if model and model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not allowed for {provider}."
        )

    # Early-Modelle (Frontier-Low + DeepSeek V4 Pro) sind tag-gated. Pro schliesst
    # Early ein, daher wird is_early an der Aufrufstelle als (pro or early) gesetzt.
    if model in cfg.EARLY_MODELS and not is_early:
        raise HTTPException(
            status_code=403,
            detail=f"The model '{model}' requires Early access. Please contact us to enable it."
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
