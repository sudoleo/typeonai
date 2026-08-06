from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator

from app.core import config as cfg
from app.core.rate_limit import ApiUidRateLimitExceeded, api_uid_limiter, limiter
from app.core.security import (
    db_firestore,
    extract_id_token,
    is_user_pro,
    verify_user_token,
)
from app.services.chat_store import (
    CHAT_PAGE_SIZE,
    CHAT_PAGE_SIZE_MAX,
    TURN_PAGE_SIZE,
    TURN_PAGE_SIZE_MAX,
    ChatCursorUnavailable,
    ChatIdempotencyConflict,
    ChatNotFound,
    ChatQuotaExceeded,
    ChatStore,
    ChatStoreError,
    InvalidChatCursor,
    TurnStatusConflict,
    normalize_client_request_id,
    normalize_mode,
    normalize_model_name,
    normalize_question,
    normalize_selected_models,
    normalize_title,
)
from app.services.chat_context import (
    ChatContextBuildInProgress,
    ChatContextConflict,
    ChatContextError,
    ChatContextNotFound,
    ChatContextService,
    ChatMemoryCompressor,
    FirestoreChatContextRepository,
)
from app.services.llm.consensus_engine import resolve_consensus_engine_model
from app.services.llm.credentials import (
    enable_gemini_adc,
    gemini_engine_credentials_available,
    resolve_developer_api_keys,
)
from app.services.usage_repository import (
    FirestoreUsageRepository,
    RunStatus,
)


router = APIRouter()

# Every endpoint below is deliberately a sync `def`, so FastAPI runs it in the
# threadpool. The Firestore client is blocking; declaring these `async def`
# would execute those blocking reads and transactions directly ON the event
# loop and stall every concurrent request for their whole duration.


class ChatCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value):
        if value is None:
            return None
        return normalize_title(value) or None


class TurnCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    mode: str
    deep_search: StrictBool
    selected_models: list[str]
    consensus_model: str
    client_request_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        return normalize_question(value)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value):
        return normalize_mode(value)

    @field_validator("selected_models")
    @classmethod
    def validate_selected_models(cls, value):
        return normalize_selected_models(value)

    @field_validator("consensus_model")
    @classmethod
    def validate_consensus_model(cls, value):
        # Dieselbe Allowlist, die /consensus spaeter prueft. Ohne sie landete
        # ein beliebiger Modell-String im Turn-Dokument -- und der Memory-Build
        # liest die Engine genau von dort (_memory_credentials), nicht aus dem
        # Request. Die Familie dieser Engine bestimmt, welcher Key den
        # Kompressions-Call bezahlt.
        model = normalize_model_name(value, field_name="consensus_model")
        if model not in cfg.ALLOWED_CONSENSUS_MODELS:
            raise ValueError("consensus_model is not an allowed consensus engine")
        return model

    @field_validator("client_request_id")
    @classmethod
    def validate_client_request_id(cls, value):
        return normalize_client_request_id(value)


class ContextBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    useOwnKeys: StrictBool = False
    usage_run_key: str | None = None
    memory_api_key: str | None = None

    @field_validator("usage_run_key", "memory_api_key")
    @classmethod
    def validate_optional_key(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("key must be a string")
        value = value.strip()
        if not value:
            return None
        if len(value.encode("utf-8")) > 512:
            raise ValueError("key is too long")
        return value


# Schreibende Chat-Operationen zusaetzlich pro KONTO begrenzen. Der
# slowapi-Limiter am Endpoint zaehlt pro IP; das ist hier die falsche Achse:
# ueber IPv6 (ein /64 liefert praktisch beliebig viele Adressen) laesst sich ein
# IP-Limit umgehen, waehrend sich hinter einem Firmen-NAT fremde Nutzer
# gegenseitig aussperren. Chat- und Turn-Anlage kosten keinen Usage-Slot, also
# ist das Konto die einzige Achse, die den Missbrauch wirklich deckelt.
CHAT_UID_RATE_LIMITS = {
    "create_chat": 20,
    "create_turn": 40,
    "build_context": 30,
    "delete_chat": 30,
}


def _chat_uid(request: Request, operation: str = "") -> str:
    token = extract_id_token(request, {})
    if not token:
        raise HTTPException(status_code=401, detail="Authentication failed")
    try:
        uid = verify_user_token(token)
    except Exception as exc:
        logging.warning("chat history authentication failed")
        raise HTTPException(status_code=401, detail="Authentication failed") from exc

    limit = CHAT_UID_RATE_LIMITS.get(operation)
    if limit is not None:
        try:
            api_uid_limiter.check(uid, f"chat:{operation}", limit)
        except ApiUidRateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too many chat requests. Please slow down.",
                    "error_code": "chat_rate_limited",
                },
            ) from exc
    return uid


def _store() -> ChatStore:
    return ChatStore(db_firestore)


def _context_repository() -> FirestoreChatContextRepository:
    return FirestoreChatContextRepository(db_firestore)


def _memory_credentials(
    uid: str,
    target: dict,
    payload: ContextBuildRequest,
    *,
    chat_id: str,
    turn_id: str,
) -> tuple[ChatMemoryCompressor | None, str, str, str]:
    engine_model = str(target.get("consensus_model") or "").strip()
    config = resolve_consensus_engine_model(engine_model)
    if not config or not config.provider:
        return None, "unsupported_memory_engine", "", engine_model
    provider = {
        "openai": "OpenAI",
        "mistral": "Mistral",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "deepseek": "DeepSeek",
        "grok": "Grok",
    }.get(config.provider, "")
    if not provider:
        return None, "unsupported_memory_engine", "", engine_model

    if payload.useOwnKeys:
        if not payload.memory_api_key:
            return None, "own_key_missing", provider, engine_model
        # This dictionary is intentionally not passed through either developer
        # resolver or Gemini ADC marking. A missing BYOK can never fall back to
        # an operator credential.
        api_keys = {provider: payload.memory_api_key}
    else:
        if not payload.usage_run_key:
            return None, "usage_run_required", provider, engine_model
        try:
            usage_repository = FirestoreUsageRepository(db_firestore)
            usage = usage_repository.get_run(uid, payload.usage_run_key)
            if usage.status is not RunStatus.CONSUMED:
                return None, "usage_run_not_consumed", provider, engine_model
            usage_repository.bind_context_target(
                uid,
                payload.usage_run_key,
                f"chat-context\0{chat_id}\0{turn_id}",
            )
        except Exception:
            return None, "usage_run_unavailable", provider, engine_model
        api_keys = enable_gemini_adc(resolve_developer_api_keys())

    if provider == "Gemini":
        if not gemini_engine_credentials_available(api_keys):
            return None, "memory_credentials_missing", provider, engine_model
    elif not api_keys.get(provider):
        return None, "memory_credentials_missing", provider, engine_model

    # Die Familie bleibt die der Consensus-Engine — nur fuer sie liegt bei
    # Eigenschluesseln ein Key vor. Welches Modell dieser Familie die Memory
    # fortschreibt, entscheidet der Admin (Firestore "chat_memory_models");
    # ohne gueltige Wahl bleibt es bei der Engine des Turns.
    memory_model = cfg.get_chat_memory_model(config.provider) or engine_model
    return ChatMemoryCompressor(memory_model, api_keys), "", provider, memory_model


def _raise_store_error(exc: Exception, *, operation: str, uid: str) -> None:
    if isinstance(exc, ChatNotFound):
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    if isinstance(exc, ChatQuotaExceeded):
        # Kein 500: das Limit ist erreicht, nicht kaputt. Der error_code sagt
        # dem Frontend, ob eine alte Unterhaltung geloescht oder eine neue
        # begonnen werden muss.
        raise HTTPException(
            status_code=403,
            detail={"error": str(exc), "error_code": exc.error_code},
        ) from exc
    if isinstance(exc, InvalidChatCursor):
        raise HTTPException(status_code=400, detail="Invalid or expired chat cursor") from exc
    if isinstance(exc, ChatCursorUnavailable):
        logging.error("%s failed: chat cursor signing is not configured", operation)
        raise HTTPException(status_code=503, detail="Chat pagination unavailable") from exc
    if isinstance(exc, ChatIdempotencyConflict):
        raise HTTPException(status_code=409, detail="Idempotency conflict") from exc
    if isinstance(exc, TurnStatusConflict):
        raise HTTPException(status_code=409, detail="Turn state conflict") from exc
    if isinstance(exc, ChatStoreError):
        logging.error("%s failed for uid=%s: %s", operation, uid, exc)
        raise HTTPException(status_code=500, detail="Chat storage error") from exc
    logging.exception("%s failed for uid=%s", operation, uid)
    raise HTTPException(status_code=500, detail="Chat storage error") from exc


@router.post("/chats", status_code=201)
@limiter.limit("20/minute")
def create_chat(
    request: Request,
    payload: ChatCreateRequest | None = Body(default=None),
):
    uid = _chat_uid(request, "create_chat")
    try:
        chat = _store().create_chat(uid, title=(payload.title if payload else "") or "")
        return {"status": "success", "chat": chat}
    except Exception as exc:
        _raise_store_error(exc, operation="create chat", uid=uid)


@router.get("/chats")
@limiter.limit("30/minute")
def list_chats(
    request: Request,
    cursor: str = Query(default="", max_length=512),
    limit: int = Query(default=CHAT_PAGE_SIZE, ge=1, le=CHAT_PAGE_SIZE_MAX),
):
    uid = _chat_uid(request)
    try:
        result = _store().list_chats(uid, cursor=cursor, limit=limit)
        return {"status": "success", **result}
    except Exception as exc:
        _raise_store_error(exc, operation="list chats", uid=uid)


@router.get("/chats/{chat_id}")
@limiter.limit("60/minute")
def get_chat(request: Request, chat_id: str):
    uid = _chat_uid(request)
    try:
        return {"status": "success", "chat": _store().get_chat(uid, chat_id)}
    except Exception as exc:
        _raise_store_error(exc, operation="get chat", uid=uid)


@router.delete("/chats/{chat_id}")
@limiter.limit("20/minute")
def delete_chat(request: Request, chat_id: str):
    """Delete one owner-bound chat and everything nested beneath it.

    Without this the only handle on a chat is its bookmark, so deleting the
    bookmark stranded the whole transcript with no way to ever reach or remove
    it. Repeating the call on an already deleted chat returns 404, matching
    every other unknown or foreign chat id.
    """
    uid = _chat_uid(request, "delete_chat")
    try:
        _store().delete_chat(uid, chat_id)
        return {"status": "success"}
    except Exception as exc:
        _raise_store_error(exc, operation="delete chat", uid=uid)


@router.post("/chats/{chat_id}/turns", status_code=201)
@limiter.limit("30/minute")
def create_turn(request: Request, chat_id: str, payload: TurnCreateRequest):
    uid = _chat_uid(request, "create_turn")
    # Dasselbe Tier-Gate wie in /consensus, hier gespiegelt. Der Turn haelt die
    # Engine fest, mit der spaeter der Memory-Call laeuft; ohne diese Pruefung
    # koennte ein Free-Konto eine Premium-Engine im Dokument hinterlegen und
    # sie auf dem Developer-Key kompressieren lassen, sobald die
    # Admin-Zuordnung fuer diese Familie einmal leer ist.
    if cfg.is_premium_consensus_model(payload.consensus_model) and not is_user_pro(uid):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Premium consensus engines are reserved for Pro users.",
                "error_code": "pro_required",
            },
        )
    try:
        turn = _store().create_turn(uid, chat_id, **payload.model_dump())
        return {"status": "success", "turn": turn}
    except Exception as exc:
        _raise_store_error(exc, operation="create turn", uid=uid)


@router.get("/chats/{chat_id}/turns/{turn_id}")
@limiter.limit("60/minute")
def get_turn(request: Request, chat_id: str, turn_id: str):
    uid = _chat_uid(request, "build_context")
    try:
        return {
            "status": "success",
            "turn": _store().get_turn(uid, chat_id, turn_id),
        }
    except Exception as exc:
        _raise_store_error(exc, operation="get turn", uid=uid)


@router.post("/chats/{chat_id}/turns/{turn_id}/context")
@limiter.limit("20/minute")
def build_turn_context(
    request: Request,
    chat_id: str,
    turn_id: str,
    payload: ContextBuildRequest,
):
    uid = _chat_uid(request)
    repository = _context_repository()
    service = ChatContextService(repository)
    try:
        target, _predecessors = repository.load_target_and_predecessors(
            uid, chat_id, turn_id
        )
        if target.get("status") == "completed":
            context = service.build_for_turn(
                uid,
                chat_id,
                turn_id,
                compressor=None,
            )
            return {"status": "success", "context": context}
        compressor, degraded_reason, provider, engine_model = _memory_credentials(
            uid,
            target,
            payload,
            chat_id=chat_id,
            turn_id=turn_id,
        )
        context = service.build_for_turn(
            uid,
            chat_id,
            turn_id,
            compressor=compressor,
            degraded_reason=degraded_reason,
            engine_provider=provider,
            engine_model=engine_model,
        )
        return {"status": "success", "context": context}
    except ChatContextBuildInProgress:
        return JSONResponse(
            status_code=202,
            content={"status": "building", "retry_after_seconds": 2},
            headers={"Retry-After": "2"},
        )
    except ChatContextNotFound as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    except ChatContextConflict as exc:
        raise HTTPException(status_code=409, detail="Chat context conflict") from exc
    except ChatContextError as exc:
        logging.error("build chat context failed for uid=%s: %s", uid, exc)
        raise HTTPException(status_code=503, detail="Chat context unavailable") from exc
    except Exception as exc:
        logging.exception("build chat context failed for uid=%s", uid)
        raise HTTPException(status_code=503, detail="Chat context unavailable") from exc


@router.get("/chats/{chat_id}/turns")
@limiter.limit("60/minute")
def list_turns(
    request: Request,
    chat_id: str,
    cursor: str = Query(default="", max_length=512),
    limit: int = Query(default=TURN_PAGE_SIZE, ge=1, le=TURN_PAGE_SIZE_MAX),
):
    uid = _chat_uid(request)
    try:
        result = _store().list_turns(uid, chat_id, cursor=cursor, limit=limit)
        return {"status": "success", **result}
    except Exception as exc:
        _raise_store_error(exc, operation="list turns", uid=uid)
