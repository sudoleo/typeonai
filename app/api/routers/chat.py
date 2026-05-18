import os
import logging
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.rate_limit import limiter
from app.core.state import usage_counter, deep_search_usage
from app.core.security import verify_user_token, extract_id_token, is_user_pro
import app.core.config as cfg
from app.core.config import (
    ALLOWED_OPENAI_MODELS, ALLOWED_MISTRAL_MODELS, ALLOWED_ANTHROPIC_MODELS,
    ALLOWED_GEMINI_MODELS, ALLOWED_DEEPSEEK_MODELS, ALLOWED_GROK_MODELS,
)
from app.services.llm.base import validate_model, count_words, get_system_prompt
from app.services.llm.engines import (
    query_openai, query_mistral, query_claude, query_gemini, query_deepseek, query_grok
)
from app.services.llm.citations import source_response
from app.services.llm.consensus_engine import query_consensus, query_differences
from tool_heuristics import get_realtime_context, get_intent_from_llm

router = APIRouter()

def get_valid_active_count(data: dict) -> int:
    raw = data.get("active_count", 1)
    if isinstance(raw, bool):
        raise HTTPException(status_code=400, detail="Invalid active_count.")

    try:
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw.isdigit():
                raise ValueError
        active_count = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid active_count.")

    if active_count < 1 or active_count > 6:
        raise HTTPException(status_code=400, detail="active_count must be between 1 and 6.")

    return active_count


def validate_question_word_limit(question: str, is_pro: bool, deep_search: bool):
    max_words_limit = cfg.get_word_limit(is_pro, deep_search)
    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail=f"Input exceeds word limit of {max_words_limit}.")

@router.post("/ask_openai")
@limiter.limit("5/minute")
def ask_openai_post(request: Request, data: dict = Body(...)):
    question = data.get("question")

    # deep_search robust konvertieren
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    # --- 2. Deep Search Check (Strictly Pro Only) ---
    if deep_search and not is_pro_user:
        raise HTTPException(
            status_code=403, 
            detail="Deep Think is exclusively available for Pro users."
        )

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(model, ALLOWED_OPENAI_MODELS, "OpenAI", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        
        # Limits festlegen basierend auf Status
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        # Prüfung Regular Usage
        if current_usage + increment > limit_regular:
            msg = "Pro usage limit reached." if is_pro_user else "Free usage limit reached. Upgrade to Pro."
            return {
                "error": msg,
                "free_usage_remaining": 0,
                "deep_remaining": int(limit_deep - current_deep_usage)
            }

        # Prüfung Deep Search Usage (nur relevant wenn deep_search=True)
        if deep_search:
            if current_deep_usage + increment > limit_deep:
                return {
                    "error": "Your Deep Think quota is exhausted for this period.",
                    "free_usage_remaining": int(limit_regular - current_usage),
                    "deep_remaining": 0
                }

        # Zähler erhöhen
        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        # API Call
        developer_api_key = os.environ.get("DEVELOPER_OPENAI_API_KEY")
        if not developer_api_key:
            raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_openai(
            question, developer_api_key, deep_search=deep_search,
            system_prompt=system_prompt, model_override=model,
            max_output_tokens=cfg.get_output_token_limit(is_pro_user, deep_search)
        )

        # Remaining berechnen
        remaining_regular = int(limit_regular - usage_counter[uid])
        remaining_deep = int(limit_deep - deep_search_usage.get(uid, 0))

        return source_response(
            answer,
            free_usage_remaining=remaining_regular,
            deep_remaining=remaining_deep,
            is_pro_user=is_pro_user,
            key_used="Developer API Key"
        )

    # --- 5. Eigener API Key (Bypass Limits) ---
    elif api_key:
        answer = query_openai(
            question, api_key, deep_search=deep_search,
            system_prompt=system_prompt, model_override=model,
            max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(
            answer,
            free_usage_remaining="Unlimited",
            deep_remaining="Unlimited",
            is_pro_user=False,
            key_used="User API Key"
        )

    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@router.post("/ask_mistral")
@limiter.limit("5/minute")
def ask_mistral_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(model, ALLOWED_MISTRAL_MODELS, "Mistral", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")
        
        answer = query_mistral(
            question, developer_api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(is_pro_user, deep_search)
        )
        
        return source_response(
            answer,
            free_usage_remaining=int(limit_regular - usage_counter[uid]),
            deep_remaining=int(limit_deep - deep_search_usage.get(uid, 0)),
            is_pro_user=is_pro_user,
            key_used="Developer API Key"
        )

    elif api_key:
        answer = query_mistral(
            question, api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(answer, free_usage_remaining="Unlimited", deep_remaining="Unlimited", is_pro_user=False, key_used="User API Key")
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@router.post("/ask_claude")
@limiter.limit("3/minute")
def ask_claude_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(model, ALLOWED_ANTHROPIC_MODELS, "Anthropic", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_claude(
            question, developer_api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(is_pro_user, deep_search)
        )
        
        return source_response(
            answer,
            free_usage_remaining=int(limit_regular - usage_counter[uid]),
            deep_remaining=int(limit_deep - deep_search_usage.get(uid, 0)),
            is_pro_user=is_pro_user,
            key_used="Developer API Key"
        )

    elif api_key:
        answer = query_claude(
            question, api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(answer, free_usage_remaining="Unlimited", deep_remaining="Unlimited", is_pro_user=False, key_used="User API Key")
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@router.post("/ask_gemini")
@limiter.limit("3/minute")
def ask_gemini_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    use_own_keys = str(data.get("useOwnKeys", "false")).lower() == "true"
    id_token = extract_id_token(request, data)
    api_key = (data.get("api_key") or data.get("gemini_key"))
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    max_tokens = cfg.get_output_token_limit(is_pro_user, deep_search)
    validate_model(model, ALLOWED_GEMINI_MODELS, "Gemini", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        # Entscheidung: Key oder Service Account
        if use_own_keys:
            if not (api_key and api_key.strip()):
                raise HTTPException(status_code=400, detail="Missing user API key for Gemini.")
            answer = query_gemini(question, user_api_key=api_key.strip(), deep_search=deep_search, system_prompt=system_prompt, model_override=model, max_output_tokens=max_tokens)
            key_info = "User API Key"
        else:
            # Service Account (SaaS Budget)
            answer = query_gemini(question, user_api_key=None, deep_search=deep_search, system_prompt=system_prompt, model_override=model, max_output_tokens=max_tokens)
            key_info = "Service Account"

        return source_response(
            answer,
            free_usage_remaining=int(limit_regular - usage_counter[uid]),
            deep_remaining=int(limit_deep - deep_search_usage.get(uid, 0)),
            is_pro_user=is_pro_user,
            key_used=key_info
        )

    else:
        # Guest mit eigenem Key
        if not (api_key and api_key.strip()):
            raise HTTPException(status_code=400, detail="No credentials provided.")
        answer = query_gemini(
            question, user_api_key=api_key.strip(), deep_search=deep_search,
            system_prompt=system_prompt, model_override=model,
            max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(answer, key_used="User API Key", is_pro_user=False)


@router.post("/ask_deepseek")
@limiter.limit("3/minute")
def ask_deepseek_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(model, ALLOWED_DEEPSEEK_MODELS, "DeepSeek", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_deepseek(
            question, developer_api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(is_pro_user, deep_search)
        )
        
        return source_response(
            answer,
            free_usage_remaining=int(limit_regular - usage_counter[uid]),
            deep_remaining=int(limit_deep - deep_search_usage.get(uid, 0)),
            is_pro_user=is_pro_user,
            key_used="Developer API Key"
        )

    elif api_key:
        answer = query_deepseek(
            question, api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(answer, free_usage_remaining="Unlimited", deep_remaining="Unlimited", is_pro_user=False, key_used="User API Key")
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")


@router.post("/ask_grok")
@limiter.limit("3/minute")
def ask_grok_post(request: Request, data: dict = Body(...)):
    question = data.get("question")
    deep_search_raw = data.get("deep_search", False)
    deep_search = True if str(deep_search_raw).lower() == "true" else False

    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    api_key = data.get("api_key")
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(model, ALLOWED_GROK_MODELS, "Grok", is_pro=is_pro_user)

    active_count = get_valid_active_count(data)

    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
             return {"error": "Usage limit reached.", "free_usage_remaining": 0, "deep_remaining": int(limit_deep - current_deep_usage)}

        if deep_search and (current_deep_usage + increment > limit_deep):
             return {"error": "Deep Think quota exhausted.", "free_usage_remaining": int(limit_regular - current_usage), "deep_remaining": 0}

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_api_key = os.environ.get("DEVELOPER_GROK_API_KEY")
        if not developer_api_key:
             raise HTTPException(status_code=500, detail="Server error: API key missing")

        answer = query_grok(
            question, developer_api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(is_pro_user, deep_search)
        )
        
        return source_response(
            answer,
            free_usage_remaining=int(limit_regular - usage_counter[uid]),
            deep_remaining=int(limit_deep - deep_search_usage.get(uid, 0)),
            is_pro_user=is_pro_user,
            key_used="Developer API Key"
        )

    elif api_key:
        answer = query_grok(
            question, api_key, system_prompt, deep_search=deep_search,
            model_override=model, max_output_tokens=cfg.get_output_token_limit(False, deep_search)
        )
        return source_response(answer, free_usage_remaining="Unlimited", deep_remaining="Unlimited", is_pro_user=False, key_used="User API Key")
    else:
        raise HTTPException(status_code=400, detail="No auth provided.")

@router.post("/prepare")
async def prepare(request: Request, data: dict = Body(...)):
    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required to analyze intent.")

    try:
        uid = verify_user_token(id_token)
        is_pro = is_user_pro(uid)
        limit = cfg.get_usage_limit(is_pro)
        current_usage = usage_counter.get(uid, 0)
        if current_usage >= limit:
            raise HTTPException(status_code=403, detail="Usage limit exhausted.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Auth failed in /prepare: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")

    raw_system_prompt = data.get("system_prompt")
    if not raw_system_prompt or not str(raw_system_prompt).strip():
        base_system_prompt = get_system_prompt()
    else:
        base_system_prompt = str(raw_system_prompt).strip()

    decision = await run_in_threadpool(get_intent_from_llm, question)
    tool = decision.get("tool")

    realtime_data = None
    if tool in {"weather", "stock", "crypto"}:
        realtime_data = await run_in_threadpool(get_realtime_context, question, decision=decision)

    if realtime_data:
        logging.info("Injecting realtime context.")
        base_system_prompt = (
            f"REAL-TIME DATA:\n{realtime_data}\n\n"
            "INSTRUCTIONS:\n"
            "You can use the real-time data provided above to answer the user's question.\n\n"
            f"{base_system_prompt}"
        )

    return {
        "system_prompt": base_system_prompt,
        "sources": []
    }


@router.post("/consensus")
@limiter.limit("5/minute")
def consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    use_own_keys = str(data.get("useOwnKeys", "false")).lower() == "true"
    consensus_model = data.get("consensus_model")
    uid = None
    is_pro = False
    
    # 1. Auth & Usage Check
    if id_token:
        if not use_own_keys:
            try:
                uid = verify_user_token(id_token)
                is_pro = is_user_pro(uid)  # WICHTIG: Pro-Status prüfen
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")

            if isinstance(consensus_model, str) and consensus_model.endswith("-Pro") and not is_pro:
                raise HTTPException(status_code=403, detail="Premium consensus engines are reserved for Pro users.")

            # Limits basierend auf Status festlegen
            limit_regular = cfg.get_usage_limit(is_pro)
            limit_deep = cfg.get_deep_search_limit(is_pro)

            # Aktuellen Verbrauch abrufen
            current_usage = usage_counter.get(uid, 0)
            current_deep_usage = deep_search_usage.get(uid, 0)

            # Reguläre Usage Prüfung (gilt immer für Consensus Request)
            if current_usage >= limit_regular:
                msg = "Pro usage limit reached." if is_pro else "Your free quota has been used up. Please store your own API keys."
                raise HTTPException(
                    status_code=403,
                    detail=msg
                )
            
            # Reguläre Usage erhöhen
            usage_counter[uid] = current_usage + 1
    else:
        # Gast / ohne Token -> muss eigene Keys nutzen
        use_own_keys = True

    # Parameter extrahieren
    question        = data.get("question")
    answer_openai   = data.get("answer_openai")
    answer_mistral  = data.get("answer_mistral")
    answer_claude   = data.get("answer_claude")
    answer_gemini   = data.get("answer_gemini")
    answer_deepseek = data.get("answer_deepseek")
    answer_grok     = data.get("answer_grok")
    best_model      = data.get("best_model", "")
    excluded_models = data.get("excluded_models", [])

    # API Keys setzen: Bei useOwnKeys werden die vom Nutzer übermittelten Keys genutzt,
    # andernfalls wird für fehlende Keys auf die Developer Keys zurückgegriffen.
    if isinstance(consensus_model, str) and consensus_model.endswith("-Pro"):
        if not id_token:
            raise HTTPException(status_code=403, detail="Premium consensus engines require a Pro account.")
        if uid is None:
            try:
                uid = verify_user_token(id_token)
                is_pro = is_user_pro(uid)
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")
        if not is_pro:
            raise HTTPException(status_code=403, detail="Premium consensus engines are reserved for Pro users.")

    api_keys = {}
    if use_own_keys:
        api_keys["OpenAI"] = data.get("openai_key")
        api_keys["Mistral"] = data.get("mistral_key")
        api_keys["Anthropic"] = data.get("anthropic_key")
        api_keys["Gemini"] = data.get("gemini_key")
        api_keys["DeepSeek"] = data.get("deepseek_key")
        api_keys["Grok"] = data.get("grok_key")

    else:
        api_keys["OpenAI"] = data.get("openai_key") or os.environ.get("DEVELOPER_OPENAI_API_KEY")
        api_keys["Mistral"] = data.get("mistral_key") or os.environ.get("DEVELOPER_MISTRAL_API_KEY")
        api_keys["Anthropic"] = data.get("anthropic_key") or os.environ.get("DEVELOPER_ANTHROPIC_API_KEY")
        api_keys["Gemini"] = data.get("gemini_key") or os.environ.get("DEVELOPER_GEMINI_API_KEY")
        api_keys["DeepSeek"] = data.get("deepseek_key") or os.environ.get("DEVELOPER_DEEPSEEK_API_KEY")
        api_keys["Grok"] = data.get("grok_key") or os.environ.get("DEVELOPER_GROK_API_KEY")

    # Validierung der erforderlichen Parameter (nur für Modelle, die nicht ausgeschlossen wurden)
    missing = []
    if not question:
        missing.append("question")
    if not consensus_model:
        missing.append("consensus_model")

    if "OpenAI" not in excluded_models and not answer_openai:
        missing.append("OpenAI")
    if "Mistral" not in excluded_models and not answer_mistral:
        missing.append("Mistral")
    if "Anthropic" not in excluded_models and not answer_claude:
        missing.append("Anthropic")
    if "Gemini" not in excluded_models and not answer_gemini:
        missing.append("Gemini")
    if "DeepSeek" not in excluded_models and not answer_deepseek:
        missing.append("DeepSeek")
    if "Grok" not in excluded_models and not answer_grok:
        missing.append("Grok")

    if missing:
        raise HTTPException(status_code=400, detail="Missing parameters: " + ", ".join(missing))
    
    # Engine-Key-Check (wichtig, um 401 der Engine zu vermeiden)
    engine = consensus_model
    engine_key_map = {
        "OpenAI": "OpenAI",       "OpenAI-Pro": "OpenAI",
        "Mistral": "Mistral",     "Mistral-Pro": "Mistral",
        "Anthropic": "Anthropic", "Anthropic-Pro": "Anthropic",
        "Gemini": "Gemini",       "Gemini-Pro": "Gemini",
        "DeepSeek": "DeepSeek",   "DeepSeek-Pro": "DeepSeek",
        "Grok": "Grok",           "Grok-Pro": "Grok",
    }
    
    need_key_for = engine_key_map.get(engine)
    if need_key_for:
        # ÄNDERUNG: Prüfe auf "Gemini" ODER "Gemini-Pro"
        if engine in ["Gemini", "Gemini-Pro"]:
            # Erlaube drei Varianten:
            # 1) expliziter Key (User- oder Dev-Key),
            # 2) Dev-Key aus ENV,
            # 3) Service Account via GOOGLE_APPLICATION_CREDENTIALS, aber NUR wenn nicht useOwnKeys
            has_explicit_key = bool(api_keys.get("Gemini"))
            has_dev_key      = bool(os.environ.get("DEVELOPER_GEMINI_API_KEY"))
            using_service_acct = (not use_own_keys) and bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

            if not (has_explicit_key or has_dev_key or using_service_acct):
                raise HTTPException(
                    status_code=400,
                    detail=("Missing credentials for selected consensus engine: Gemini. "
                            "Provide a Gemini API key or configure a Service Account on the server.")
                )
        else:
            if not api_keys.get(need_key_for):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing API key for selected consensus engine: {engine}."
                )

    if best_model and best_model in excluded_models:
        raise HTTPException(status_code=400, detail="The answer marked as best must not be excluded.")

    consensus_answer = query_consensus(
        question,
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        best_model,
        excluded_models,
        consensus_model,
        api_keys,
    )

    differences = query_differences(
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        consensus_answer,
        api_keys,
        differences_model=consensus_model
    )

    return {"consensus_response": consensus_answer, "differences": differences}
