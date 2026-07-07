import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.rate_limit import limiter
from app.core.state import usage_counter, deep_search_usage
from app.core.security import verify_user_token, extract_id_token, is_user_pro, is_user_early
import app.core.config as cfg
from app.services.llm.attachments import parse_attachments
from app.services.llm.base import (
    build_followup_system_prompt,
    count_words,
    get_system_prompt,
    validate_model,
)
from app.services.llm.engines import (
    query_openai, query_mistral, query_claude, query_gemini, query_deepseek, query_grok
)
from app.services.llm.citations import source_response
from app.services.llm.mock_llm import mock_ask_result, mock_ask_stream, mock_llm_enabled
from app.services.llm.streaming import (
    SSE_HEADERS,
    sse_pack,
    streaming_model_response,
    stream_claude_query,
    stream_deepseek_query,
    stream_gemini_query,
    stream_grok_query,
    stream_mistral_query,
    stream_openai_query,
)
from app.services.llm.consensus_engine import (
    DIFFERENCES_SKIPPED_TEXT,
    is_consensus_error_text,
    normalize_model_name,
    query_consensus,
    query_differences,
    resolve_consensus_engine_model,
    stream_consensus,
    stream_differences,
)
from app.services.llm.resolve_engine import (
    InvalidResolvePayload,
    normalize_resolve_positions,
    run_resolve_round,
)
from app.services.share_snapshots import persist_pending_result
from tool_heuristics import get_realtime_context, get_intent_from_llm

router = APIRouter()

OWN_KEYS_LOGIN_REQUIRED = "Please log in to use your own API keys."

def build_usage_limit_detail(message: str, code: str, limit_regular: int, current_usage, limit_deep: int, current_deep_usage):
    return {
        "error": message,
        "error_code": code,
        "free_usage_remaining": max(0, int(limit_regular - current_usage)),
        "deep_remaining": max(0, int(limit_deep - current_deep_usage)),
        "limit": int(limit_regular),
        "deep_limit": int(limit_deep),
    }


def parse_boolean_flag(value) -> bool:
    return str(value).strip().lower() == "true"


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


ENGINE_KEY_FIELDS = {
    "OpenAI": ("openai_key", "DEVELOPER_OPENAI_API_KEY"),
    "Mistral": ("mistral_key", "DEVELOPER_MISTRAL_API_KEY"),
    "Anthropic": ("anthropic_key", "DEVELOPER_ANTHROPIC_API_KEY"),
    "Gemini": ("gemini_key", "DEVELOPER_GEMINI_API_KEY"),
    "DeepSeek": ("deepseek_key", "DEVELOPER_DEEPSEEK_API_KEY"),
    "Grok": ("grok_key", "DEVELOPER_GROK_API_KEY"),
}


def build_engine_api_keys(data: dict, use_own_keys: bool) -> dict:
    """Keys fuer Consensus-/Differences-/Resolve-Engines: bei useOwnKeys nur
    die vom Nutzer uebermittelten Keys, sonst Fallback auf die Developer-Keys."""
    api_keys = {}
    for label, (field, env_name) in ENGINE_KEY_FIELDS.items():
        key = data.get(field)
        if not use_own_keys:
            key = key or os.environ.get(env_name)
        api_keys[label] = key
    return api_keys


def cap_engine_text(value, limit: int):
    """Kappt clientseitig gelieferte Texte (Frage/Modellantworten), bevor sie
    in Consensus-/Differences-Prompts fliessen. Stilles Truncate statt 400:
    legitime Antworten liegen weit unter dem Limit, nur Abuse-Payloads nicht."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit].rstrip()


def normalize_followup_context(raw):
    """Validiert das optionale context-Feld einer Follow-up-Frage
    ({previous_question, previous_consensus}) und kappt beide Texte
    serverseitig, analog zu cap_engine_text bei /consensus. Genau eine
    Kontext-Ebene: ein einzelnes Frage/Konsens-Paar, kein Verlauf.
    Unbrauchbare Payloads werden still ignoriert (kein 400: das Feld ist
    optional und ein kaputter Kontext soll die Frage nicht blockieren)."""
    if not isinstance(raw, dict):
        return None
    previous_question = raw.get("previous_question")
    previous_consensus = raw.get("previous_consensus")
    if not isinstance(previous_question, str) or not previous_question.strip():
        return None
    if not isinstance(previous_consensus, str) or not previous_consensus.strip():
        return None
    return {
        "previous_question": cap_engine_text(
            previous_question.strip(), cfg.get_followup_question_char_limit()
        ),
        "previous_consensus": cap_engine_text(
            previous_consensus.strip(), cfg.get_followup_consensus_char_limit()
        ),
    }


def require_pro_for_followup(followup_context, is_pro: bool):
    """Follow-up-Fragen sind Pro-only; das Gate gilt auch mit eigenen Keys
    (wie Deep Think und Resolve)."""
    if followup_context and not is_pro:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Follow-up questions are a Pro feature.",
                "error_code": "pro_required",
            },
        )


def validate_question_word_limit(question: str, is_pro: bool, deep_search: bool):
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    max_words_limit = cfg.get_word_limit(is_pro, deep_search)
    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail=f"Input exceeds word limit of {max_words_limit}.")

# ---------------------------------------------------------------------------
# /ask_*-Endpoints: ein gemeinsamer Ablauf (handle_ask) fuer alle sechs
# Provider. Alles, was sich zwischen den Provider-APIs unterscheidet, steht
# deklarativ in AskProvider bzw. ASK_PROVIDERS:
#   - Rate-Limits haengen als Literal am jeweiligen Endpoint (slowapi).
#   - Gemini hat keinen Pflicht-Dev-Key (Service-Account/ADC-Fallback im
#     Engine-Layer), nimmt den Key als user_api_key-Kwarg entgegen, kennt
#     das Legacy-Feld "gemini_key" und respektiert das useOwnKeys-Flag.
#   - Die uebrigen Provider erwarten den Key als zweites Positionsargument
#     und brauchen einen DEVELOPER_*_API_KEY aus der Umgebung.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AskProvider:
    label: str                      # kanonisches Provider-Label (Claude -> "Anthropic")
    allowed_models_attr: str        # Set-Name in app.core.config (wird in-place gepflegt)
    query_fn: Callable
    stream_fn: Callable
    developer_key_env: Optional[str]        # None: kein Pflicht-Dev-Key (Gemini)
    developer_key_label: str = "Developer API Key"
    key_kwarg: Optional[str] = None         # Key als Kwarg statt 2. Positionsarg (Gemini)
    alt_key_field: Optional[str] = None     # zusaetzliches Request-Feld fuer den Key
    honors_own_keys_flag: bool = False      # useOwnKeys erzwingt den Own-Key-Pfad
    no_auth_error: tuple = (400, "No auth provided.")


ASK_PROVIDERS = {
    "openai": AskProvider(
        label="OpenAI",
        allowed_models_attr="ALLOWED_OPENAI_MODELS",
        query_fn=query_openai,
        stream_fn=stream_openai_query,
        developer_key_env="DEVELOPER_OPENAI_API_KEY",
    ),
    "mistral": AskProvider(
        label="Mistral",
        allowed_models_attr="ALLOWED_MISTRAL_MODELS",
        query_fn=query_mistral,
        stream_fn=stream_mistral_query,
        developer_key_env="DEVELOPER_MISTRAL_API_KEY",
    ),
    "anthropic": AskProvider(
        label="Anthropic",
        allowed_models_attr="ALLOWED_ANTHROPIC_MODELS",
        query_fn=query_claude,
        stream_fn=stream_claude_query,
        developer_key_env="DEVELOPER_ANTHROPIC_API_KEY",
    ),
    "gemini": AskProvider(
        label="Gemini",
        allowed_models_attr="ALLOWED_GEMINI_MODELS",
        query_fn=query_gemini,
        stream_fn=stream_gemini_query,
        developer_key_env=None,
        developer_key_label="Service Account",
        key_kwarg="user_api_key",
        alt_key_field="gemini_key",
        honors_own_keys_flag=True,
        no_auth_error=(401, "Authentication required"),
    ),
    "deepseek": AskProvider(
        label="DeepSeek",
        allowed_models_attr="ALLOWED_DEEPSEEK_MODELS",
        query_fn=query_deepseek,
        stream_fn=stream_deepseek_query,
        developer_key_env="DEVELOPER_DEEPSEEK_API_KEY",
    ),
    "grok": AskProvider(
        label="Grok",
        allowed_models_attr="ALLOWED_GROK_MODELS",
        query_fn=query_grok,
        stream_fn=stream_grok_query,
        developer_key_env="DEVELOPER_GROK_API_KEY",
    ),
}


def _run_ask(provider: AskProvider, *, stream_requested, question, key,
             system_prompt, deep_search, model, max_tokens, attachments, extras):
    """Fuehrt den Provider-Call aus (streamend oder nicht) und verpackt das
    Ergebnis im bisherigen Response-Format."""
    if mock_llm_enabled():
        # E2E-Suite: deterministischer Fixture-Stream statt Provider-Call.
        # Auth/Limits/Validierung sind zu diesem Zeitpunkt bereits gelaufen.
        if stream_requested:
            return streaming_model_response(
                mock_ask_stream(provider.label, question), provider.label, extras
            )
        return source_response(mock_ask_result(provider.label, question), **extras)

    kwargs = {
        "system_prompt": system_prompt,
        "deep_search": deep_search,
        "model_override": model,
        "max_output_tokens": max_tokens,
        "attachments": attachments,
    }
    if provider.key_kwarg:
        args = (question,)
        kwargs[provider.key_kwarg] = key
    else:
        args = (question, key)

    if stream_requested:
        return streaming_model_response(provider.stream_fn(*args, **kwargs), provider.label, extras)
    return source_response(provider.query_fn(*args, **kwargs), **extras)


def handle_ask(provider: AskProvider, request: Request, data: dict):
    question = data.get("question")
    deep_search = parse_boolean_flag(data.get("deep_search", False))
    stream_requested = parse_boolean_flag(data.get("stream", False))
    system_prompt = data.get("system_prompt")
    id_token = extract_id_token(request, data)
    alt_key = data.get(provider.alt_key_field) if provider.alt_key_field else None
    api_key = str(data.get("api_key") or alt_key or "").strip()
    model = data.get("model")

    is_pro_user = False
    is_early_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
            is_pro_user = is_user_pro(uid)
            is_early_user = is_pro_user or is_user_early(uid)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")

    # Deep Think ist strikt Pro-only.
    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(
        model,
        getattr(cfg, provider.allowed_models_attr),
        provider.label,
        is_pro=is_pro_user,
        is_early=is_early_user,
    )
    attachments = parse_attachments(data, is_pro_user)
    active_count = get_valid_active_count(data)
    max_tokens = cfg.get_output_token_limit(is_pro_user, deep_search)

    # Follow-up-Kontext (Pro): genau eine vorherige Frage/Konsens-Ebene,
    # serverseitig gekappt und hier in den System-Prompt injiziert — nicht in
    # /prepare, damit der Kontext auch dann ankommt, wenn das Frontend nach
    # einem /prepare-Fehler mit dem Basis-Prompt weitermacht.
    followup_context = normalize_followup_context(data.get("context"))
    require_pro_for_followup(followup_context, is_pro_user)
    if followup_context:
        base_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else get_system_prompt()
        )
        system_prompt = build_followup_system_prompt(
            base_prompt,
            followup_context["previous_question"],
            followup_context["previous_consensus"],
        )

    own_keys_requested = bool(api_key) or (
        provider.honors_own_keys_flag and parse_boolean_flag(data.get("useOwnKeys", False))
    )

    # --- Eigener API-Key: eingeloggtes Feature, umgeht die Usage-Zaehlung ---
    if own_keys_requested and uid:
        if not api_key:
            # Nur ueber das useOwnKeys-Flag ohne Key erreichbar (Gemini).
            raise HTTPException(status_code=400, detail=f"Missing user API key for {provider.label}.")
        return _run_ask(
            provider,
            stream_requested=stream_requested,
            question=question,
            key=api_key,
            system_prompt=system_prompt,
            deep_search=deep_search,
            model=model,
            max_tokens=max_tokens,
            attachments=attachments,
            extras={
                "free_usage_remaining": "Unlimited",
                "deep_remaining": "Unlimited",
                "is_pro_user": is_pro_user,
                "key_used": "User API Key",
            },
        )

    # --- Developer-Key/Service-Account: Usage-Zaehlung fuer eingeloggte Nutzer ---
    if uid:
        increment = 1.0 / active_count
        limit_regular = cfg.get_usage_limit(is_pro_user)
        limit_deep = cfg.get_deep_search_limit(is_pro_user)

        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        if current_usage + increment > limit_regular:
            msg = "Pro usage limit reached." if is_pro_user else "Free usage limit reached. Upgrade to Pro."
            return {
                "error": msg,
                "free_usage_remaining": 0,
                "deep_remaining": int(limit_deep - current_deep_usage),
            }

        if deep_search and (current_deep_usage + increment > limit_deep):
            return {
                "error": "Your Deep Think quota is exhausted for this period.",
                "free_usage_remaining": int(limit_regular - current_usage),
                "deep_remaining": 0,
            }

        usage_counter[uid] = current_usage + increment
        if deep_search:
            deep_search_usage[uid] = current_deep_usage + increment

        developer_key = os.environ.get(provider.developer_key_env) if provider.developer_key_env else None
        if provider.developer_key_env and not developer_key:
            raise HTTPException(status_code=500, detail="Server error: API key missing")

        return _run_ask(
            provider,
            stream_requested=stream_requested,
            question=question,
            key=developer_key,
            system_prompt=system_prompt,
            deep_search=deep_search,
            model=model,
            max_tokens=max_tokens,
            attachments=attachments,
            extras={
                "free_usage_remaining": int(limit_regular - usage_counter[uid]),
                "deep_remaining": int(limit_deep - deep_search_usage.get(uid, 0)),
                "is_pro_user": is_pro_user,
                "key_used": provider.developer_key_label,
            },
        )

    # --- Kein Login: eigener Key erfordert Login, sonst Provider-No-Auth-Fehler ---
    if api_key:
        raise HTTPException(status_code=401, detail=OWN_KEYS_LOGIN_REQUIRED)
    status_code, detail = provider.no_auth_error
    raise HTTPException(status_code=status_code, detail=detail)


@router.post("/ask_openai")
@limiter.limit("5/minute")
def ask_openai_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["openai"], request, data)


@router.post("/ask_mistral")
@limiter.limit("5/minute")
def ask_mistral_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["mistral"], request, data)


@router.post("/ask_claude")
@limiter.limit("3/minute")
def ask_claude_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["anthropic"], request, data)


@router.post("/ask_gemini")
@limiter.limit("3/minute")
def ask_gemini_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["gemini"], request, data)


@router.post("/ask_deepseek")
@limiter.limit("3/minute")
def ask_deepseek_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["deepseek"], request, data)


@router.post("/ask_grok")
@limiter.limit("3/minute")
def ask_grok_post(request: Request, data: dict = Body(...)):
    return handle_ask(ASK_PROVIDERS["grok"], request, data)


@router.post("/prepare")
async def prepare(request: Request, data: dict = Body(...)):
    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    use_own_keys = parse_boolean_flag(data.get("useOwnKeys", False))
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required to analyze intent.")

    try:
        uid = verify_user_token(id_token)
        is_pro = is_user_pro(uid)
        limit = cfg.get_usage_limit(is_pro)
        deep_limit = cfg.get_deep_search_limit(is_pro)
        current_usage = usage_counter.get(uid, 0)
        if not use_own_keys and current_usage >= limit:
            current_deep_usage = deep_search_usage.get(uid, 0)
            msg = "Pro usage limit reached." if is_pro else "Free usage limit reached. Upgrade to Pro or use your own API keys."
            raise HTTPException(
                status_code=403,
                detail=build_usage_limit_detail(
                    msg,
                    "usage_limit_exceeded",
                    limit,
                    current_usage,
                    deep_limit,
                    current_deep_usage,
                )
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Auth failed in /prepare: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")

    # Follow-up-Kontext frueh gaten: Free-Nutzer bekommen den Pro-Fehler schon
    # vor dem Fan-out. Injiziert wird der Kontext erst in den /ask_*-Endpoints
    # (handle_ask), sonst stuende er doppelt im System-Prompt.
    require_pro_for_followup(normalize_followup_context(data.get("context")), is_pro)

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
    stream_requested = parse_boolean_flag(data.get("stream", False))
    uid = None
    is_pro = False
    
    # 1. Auth & Usage Check
    if not id_token:
        raise HTTPException(
            status_code=401,
            detail=OWN_KEYS_LOGIN_REQUIRED if use_own_keys else "Authentication required",
        )

    try:
        uid = verify_user_token(id_token)
        is_pro = is_user_pro(uid)  # WICHTIG: Pro-Status prüfen
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if consensus_model not in cfg.ALLOWED_CONSENSUS_MODELS:
        raise HTTPException(status_code=400, detail="Invalid consensus model selected.")

    # Early-Engine (z.B. Gemini Frontier-Low) ist tag-gated. Pro schliesst Early
    # ein; gilt auch bei eigenen Keys, analog zum Premium-Gating weiter unten.
    early_access = is_pro or (uid is not None and is_user_early(uid))
    if cfg.is_early_consensus_model(consensus_model) and not early_access:
        raise HTTPException(status_code=403, detail="Early access consensus engines are reserved for Early or Pro users.")

    if not use_own_keys:
        if cfg.is_premium_consensus_model(consensus_model) and not is_pro:
            raise HTTPException(status_code=403, detail="Premium consensus engines are reserved for Pro users.")

        # Limits basierend auf Status festlegen
        limit_regular = cfg.get_usage_limit(is_pro)
        limit_deep = cfg.get_deep_search_limit(is_pro)

        # Aktuellen Verbrauch abrufen
        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)

        # Reguläre Usage Prüfung (gilt immer für Consensus Request)
        if current_usage >= limit_regular:
            msg = "Pro usage limit reached." if is_pro else "Your free quota has been used up. Please store your own API keys or upgrade to Pro."
            raise HTTPException(
                status_code=403,
                detail=build_usage_limit_detail(
                    msg,
                    "usage_limit_exceeded",
                    limit_regular,
                    current_usage,
                    limit_deep,
                    current_deep_usage,
                )
            )
        
        # Reguläre Usage erhöhen
        usage_counter[uid] = current_usage + 1
    else:
        limit_regular = cfg.get_usage_limit(is_pro)
        limit_deep = cfg.get_deep_search_limit(is_pro)

    # Parameter extrahieren. Frage und Antworten kommen als freier Text vom
    # Client und werden serverseitig gekappt: der Consensus-Prompt enthaelt
    # sonst unbegrenzte Eingaben gegen den Developer-Key (Kostenleck).
    answer_char_limit = cfg.get_consensus_answer_char_limit()
    question        = cap_engine_text(data.get("question"), cfg.get_consensus_question_char_limit())
    answer_openai   = cap_engine_text(data.get("answer_openai"), answer_char_limit)
    answer_mistral  = cap_engine_text(data.get("answer_mistral"), answer_char_limit)
    answer_claude   = cap_engine_text(data.get("answer_claude"), answer_char_limit)
    answer_gemini   = cap_engine_text(data.get("answer_gemini"), answer_char_limit)
    answer_deepseek = cap_engine_text(data.get("answer_deepseek"), answer_char_limit)
    answer_grok     = cap_engine_text(data.get("answer_grok"), answer_char_limit)
    excluded_models = data.get("excluded_models", [])
    model_sources   = data.get("model_sources", {})
    if not isinstance(excluded_models, list):
        excluded_models = []
    excluded_models = list({normalize_model_name(model) for model in excluded_models if model})
    if not isinstance(model_sources, dict):
        model_sources = {}

    if "OpenAI" in excluded_models:
        answer_openai = None
    if "Mistral" in excluded_models:
        answer_mistral = None
    if "Anthropic" in excluded_models:
        answer_claude = None
    if "Gemini" in excluded_models:
        answer_gemini = None
    if "DeepSeek" in excluded_models:
        answer_deepseek = None
    if "Grok" in excluded_models:
        answer_grok = None

    # API Keys setzen: Bei useOwnKeys werden die vom Nutzer übermittelten Keys genutzt,
    # andernfalls wird für fehlende Keys auf die Developer Keys zurückgegriffen.
    if cfg.is_premium_consensus_model(consensus_model):
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

    api_keys = build_engine_api_keys(data, use_own_keys)

    # Validierung der erforderlichen Parameter (nur für Modelle, die nicht ausgeschlossen wurden)
    missing = []
    if not question:
        missing.append("question")
    if not consensus_model:
        missing.append("consensus_model")

    included_answers = {
        model: answer
        for model, answer in {
            "OpenAI": answer_openai,
            "Mistral": answer_mistral,
            "Anthropic": answer_claude,
            "Gemini": answer_gemini,
            "DeepSeek": answer_deepseek,
            "Grok": answer_grok,
        }.items()
        if model not in excluded_models and answer
    }

    if len(included_answers) < 2:
        missing.append("at least two selected model answers")

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
        cfg.GEMINI_FRONTIER_LOW_MODEL: "Gemini",
        "DeepSeek": "DeepSeek",   "DeepSeek-Pro": "DeepSeek",
        "Grok": "Grok",           "Grok-Pro": "Grok",
    }
    
    need_key_for = engine_key_map.get(engine)
    if not need_key_for:
        engine_config = resolve_consensus_engine_model(engine)
        provider_key_map = {
            "openai": "OpenAI",
            "mistral": "Mistral",
            "anthropic": "Anthropic",
            "gemini": "Gemini",
            "deepseek": "DeepSeek",
            "grok": "Grok",
        }
        need_key_for = provider_key_map.get(engine_config.provider if engine_config else "")
    if need_key_for:
        # ÄNDERUNG: Prüfe auf "Gemini" ODER "Gemini-Pro"
        if need_key_for == "Gemini":
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

    # Share-Feature: Ergebnis nur für verifizierte Nutzer persistieren.
    share_uid = uid
    model_labels = data.get("model_labels")

    def persist_share_result(consensus_text, differences_data, differences_text):
        # Mock-Modus (E2E-Tests) darf keine pending_results in das echte
        # Firestore schreiben.
        if not share_uid or mock_llm_enabled():
            return None
        return persist_pending_result(
            uid=share_uid,
            question=question,
            consensus_md=consensus_text,
            differences_data=differences_data,
            differences_text=differences_text,
            model_sources=model_sources,
            included_providers=list(included_answers.keys()),
            model_labels=model_labels,
            consensus_model=consensus_model,
        )

    if stream_requested:
        extra_fields = {}
        if uid:
            extra_fields = {
                "free_usage_remaining": max(0, int(cfg.get_usage_limit(is_pro) - usage_counter.get(uid, 0))),
                "deep_remaining": max(0, int(cfg.get_deep_search_limit(is_pro) - deep_search_usage.get(uid, 0))),
                "is_pro_user": is_pro,
            }

        def consensus_event_source():
            consensus_text = ""
            consensus_failed = False
            differences_text = ""
            differences_data = None
            stream_failed = False
            try:
                for item in stream_consensus(
                    question,
                    answer_openai,
                    answer_mistral,
                    answer_claude,
                    answer_gemini,
                    answer_deepseek,
                    answer_grok,
                    excluded_models,
                    consensus_model,
                    api_keys,
                    model_sources=model_sources,
                ):
                    if item.get("type") == "delta":
                        yield sse_pack("consensus.delta", {"text": item.get("text") or ""})
                    else:
                        consensus_text = item.get("text") or ""
                        consensus_failed = bool(item.get("error")) or is_consensus_error_text(consensus_text)

                if consensus_failed:
                    # Ohne Konsensantwort ist der Vergleich sinnlos: der Judge
                    # würde sonst den Fehlertext "analysieren" und das Ergebnis
                    # würde als Share-Snapshot persistiert.
                    differences_text = DIFFERENCES_SKIPPED_TEXT
                    differences_data = None
                else:
                    for item in stream_differences(
                        answer_openai,
                        answer_mistral,
                        answer_claude,
                        answer_gemini,
                        answer_deepseek,
                        answer_grok,
                        consensus_text,
                        api_keys,
                        differences_model=consensus_model,
                        excluded_models=excluded_models,
                    ):
                        if item.get("type") == "delta":
                            # Das Frontend rendert diese Deltas nicht mehr (die Engine
                            # liefert JSON); sie halten nur die SSE-Verbindung aktiv.
                            yield sse_pack("differences.delta", {"text": item.get("text") or ""})
                        else:
                            differences_text = item.get("text") or ""
                            differences_data = item.get("data")
            except Exception as exc:
                logging.exception("Consensus streaming failed")
                stream_failed = True
                if not consensus_text:
                    consensus_text = f"Consensus error: {exc}"
                if not differences_text:
                    differences_text = ""

            payload = {
                "consensus_response": consensus_text,
                "differences": differences_text,
                "differences_data": differences_data,
            }
            if not stream_failed and not consensus_failed:
                result_id = persist_share_result(consensus_text, differences_data, differences_text)
                if result_id:
                    payload["result_id"] = result_id
            payload.update(extra_fields)
            yield sse_pack("final", payload)

        return StreamingResponse(
            consensus_event_source(),
            media_type="text/event-stream",
            headers=dict(SSE_HEADERS),
        )

    consensus_answer = query_consensus(
        question,
        answer_openai,
        answer_mistral,
        answer_claude,
        answer_gemini,
        answer_deepseek,
        answer_grok,
        excluded_models,
        consensus_model,
        api_keys,
        model_sources=model_sources,
    )

    consensus_failed = is_consensus_error_text(consensus_answer)
    if consensus_failed:
        # Kein Vergleich gegen einen Fehlertext (siehe Streaming-Pfad).
        differences, differences_data = DIFFERENCES_SKIPPED_TEXT, None
    else:
        differences, differences_data = query_differences(
            answer_openai,
            answer_mistral,
            answer_claude,
            answer_gemini,
            answer_deepseek,
            answer_grok,
            consensus_answer,
            api_keys,
            differences_model=consensus_model,
            excluded_models=excluded_models,
        )

    response = {
        "consensus_response": consensus_answer,
        "differences": differences,
        "differences_data": differences_data,
    }
    if not consensus_failed:
        result_id = persist_share_result(consensus_answer, differences_data, differences)
        if result_id:
            response["result_id"] = result_id
    if uid:
        response.update({
            "free_usage_remaining": max(0, int(limit_regular - usage_counter.get(uid, 0))),
            "deep_remaining": max(0, int(limit_deep - deep_search_usage.get(uid, 0))),
            "is_pro_user": is_pro,
        })
    return response


@router.post("/resolve")
@limiter.limit("3/minute")
def resolve(request: Request, data: dict = Body(...)):
    """Resolve-Runde: konfrontiert die dissentierenden Modelle eines
    Widerspruchs (aus differences_data) gezielt mit der Gegenposition.
    Kostet einen regulaeren Usage-Punkt; Ergebnis wird nicht persistiert."""
    id_token = extract_id_token(request, data)
    use_own_keys = parse_boolean_flag(data.get("useOwnKeys", False))

    if not id_token:
        raise HTTPException(
            status_code=401,
            detail=OWN_KEYS_LOGIN_REQUIRED if use_own_keys else "Authentication required",
        )
    try:
        uid = verify_user_token(id_token)
        is_pro = is_user_pro(uid)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Resolve ist ein Pro-Feature; Free-Nutzer sehen den Button nur als Teaser.
    # Serverseitig gilt das Gate auch mit eigenen Keys (wie bei Deep Think).
    if not is_pro:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Resolve rounds are a Pro feature.",
                "error_code": "pro_required",
            },
        )

    question = cap_engine_text(data.get("question"), cfg.get_consensus_question_char_limit())
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    try:
        claim, positions = normalize_resolve_positions(data.get("claim"), data.get("positions"))
    except InvalidResolvePayload as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    limit_regular = cfg.get_usage_limit(is_pro)
    limit_deep = cfg.get_deep_search_limit(is_pro)

    if not use_own_keys:
        current_usage = usage_counter.get(uid, 0)
        current_deep_usage = deep_search_usage.get(uid, 0)
        if current_usage >= limit_regular:
            msg = "Pro usage limit reached." if is_pro else "Your free quota has been used up. Please store your own API keys or upgrade to Pro."
            raise HTTPException(
                status_code=403,
                detail=build_usage_limit_detail(
                    msg,
                    "usage_limit_exceeded",
                    limit_regular,
                    current_usage,
                    limit_deep,
                    current_deep_usage,
                ),
            )
        usage_counter[uid] = current_usage + 1

    api_keys = build_engine_api_keys(data, use_own_keys)

    result = run_resolve_round(question, claim, positions, api_keys)
    result.update({
        "free_usage_remaining": max(0, int(limit_regular - usage_counter.get(uid, 0))),
        "deep_remaining": max(0, int(limit_deep - deep_search_usage.get(uid, 0))),
        "is_pro_user": is_pro,
    })
    return result
