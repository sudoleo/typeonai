import os
import time
import logging
import re
import hashlib
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import StreamingResponse
from google.api_core.exceptions import Aborted as FirestoreAborted

from app.core.rate_limit import limiter
from app.core.observability import record_metric
from app.core.security import (
    TierStatusUnavailable,
    db_firestore,
    extract_id_token,
    is_user_pro,
    verify_user_token,
)
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
from app.services.llm.citations import coerce_text, source_response
from app.services.llm.credentials import enable_gemini_adc, resolve_developer_api_keys
from app.services.llm.mock_llm import mock_ask_result, mock_ask_stream, mock_llm_enabled
from app.services.llm.streaming import (
    SSE_HEADERS,
    iter_sse_with_keepalive,
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
from app.services.share_snapshots import (
    persist_pending_result,
    sanitize_model_labels,
    sanitize_sources,
)
from app.services.chat_store import (
    ChatNotFound,
    ChatStore,
    ChatStoreError,
    TurnQuestionConflict,
    TurnStatusConflict,
)
from app.services.chat_context import (
    ChatContextConflict,
    ChatContextError,
    ChatContextNotFound,
    ChatContextService,
    FirestoreChatContextRepository,
    ResolvedContextCache,
    build_chat_context_system_prompt,
    resolved_context_cache_key,
)
from app.services.differences_stats import record_differences_stats
from app.services.usage_repository import (
    FirestoreUsageRepository,
    RunKind,
    RunStatus,
    UsageLimitExceeded,
    UsageLimits,
    UsageRunExpired,
    UsageRunConflict,
    UsageRunNotFound,
    UsageTransitionError,
    canonical_request_fingerprint,
)

router = APIRouter()

OWN_KEYS_LOGIN_REQUIRED = "Please log in to use your own API keys."
MAX_QUESTION_CHARS = 8_000
MAX_QUESTION_BYTES = 16_000
MAX_SYSTEM_PROMPT_CHARS = 12_000
MAX_SYSTEM_PROMPT_BYTES = 32_000
run_usage_repository = FirestoreUsageRepository(db_firestore)
chat_store = ChatStore(db_firestore)
chat_context_service = ChatContextService(FirestoreChatContextRepository(db_firestore))
resolved_context_cache = ResolvedContextCache()
_CHAT_DOCUMENT_ID_RE = re.compile(r"[0-9a-f]{32}")


def get_run_usage_limits(is_pro: bool) -> UsageLimits:
    return UsageLimits(
        total=cfg.get_consensus_run_limit(is_pro),
        deep_think=cfg.get_deep_think_run_limit(is_pro),
    )


def get_usage_run_key(data: dict) -> str:
    value = data.get("usage_run_key")
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing usage_run_key. Start the run via /prepare.",
                "error_code": "usage_run_key_required",
            },
        )
    return value.strip()


def usage_run_fingerprint(data: dict, *, question: str, deep_think: bool) -> str:
    """Bind a usage run to the logical fields known before the fan-out."""
    return canonical_request_fingerprint(
        {
            "schema": 1,
            "question": question,
            "deep_think": bool(deep_think),
        }
    )


def usage_response_fields(snapshot, is_pro: bool) -> dict:
    return {
        "free_usage_remaining": snapshot.total.remaining,
        "deep_remaining": snapshot.deep_think.remaining,
        "limit": snapshot.total.limit,
        "deep_limit": snapshot.deep_think.limit,
        "is_pro_user": is_pro,
    }


def reserve_usage_run(uid: str, data: dict, *, is_pro: bool, deep_think: bool):
    key = get_usage_run_key(data)
    kind = RunKind.DEEP_THINK if deep_think else RunKind.REGULAR
    try:
        result = run_usage_repository.reserve(
            uid,
            key,
            kind,
            get_run_usage_limits(is_pro),
            request_fingerprint=usage_run_fingerprint(
                data,
                question=str(data.get("question") or "").strip(),
                deep_think=deep_think,
            ),
        )
    except UsageLimitExceeded as exc:
        detail = usage_response_fields(exc.snapshot, is_pro)
        detail.update(
            {
                "error": (
                    "Your Deep Think quota is exhausted for this UTC day."
                    if exc.limiting_bucket == "deep_think"
                    else "Your run quota is exhausted for this UTC day."
                ),
                "error_code": f"{exc.limiting_bucket}_usage_limit_exceeded",
            }
        )
        raise HTTPException(status_code=403, detail=detail) from None
    except UsageRunConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_run_conflict"},
        ) from None
    except UsageRunExpired as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_run_expired"},
        ) from None
    except FirestoreAborted:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Usage accounting is temporarily busy. Please retry this run.",
                "error_code": "usage_storage_busy",
            },
        ) from None
    if result.status is RunStatus.RELEASED:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "This usage run was already released. Start a new run.",
                "error_code": "usage_run_released",
            },
        )
    return key, result


def consume_usage_run(uid: str, key: str, *, is_pro: bool):
    try:
        return run_usage_repository.consume(uid, key)
    except UsageTransitionError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_run_transition"},
        ) from None
    except FirestoreAborted:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Usage accounting is temporarily busy. Please retry this run.",
                "error_code": "usage_storage_busy",
            },
        ) from None


def claim_usage_operation(uid: str, key: str, operation: str, request_payload):
    """Claim one provider/Judge operation before external work begins."""
    try:
        claim = run_usage_repository.claim_operation(
            uid,
            key,
            operation,
            canonical_request_fingerprint(request_payload),
        )
    except UsageRunNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": str(exc), "error_code": "usage_run_not_found"},
        ) from None
    except UsageRunExpired as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_run_expired"},
        ) from None
    except UsageRunConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_operation_conflict"},
        ) from None
    except UsageTransitionError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "error_code": "usage_run_transition"},
        ) from None
    except FirestoreAborted:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Usage authorization is temporarily busy. Please retry.",
                "error_code": "usage_storage_busy",
            },
        ) from None
    if claim.idempotent:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "This logical operation was already started.",
                "error_code": "usage_operation_already_claimed",
            },
        )
    return claim

def parse_boolean_flag(value) -> bool:
    return str(value).strip().lower() == "true"


ENGINE_KEY_FIELDS = {
    "OpenAI": "openai_key",
    "Mistral": "mistral_key",
    "Anthropic": "anthropic_key",
    "Gemini": "gemini_key",
    "DeepSeek": "deepseek_key",
    "Grok": "grok_key",
}


def build_engine_api_keys(data: dict, use_own_keys: bool) -> dict:
    """Keys fuer Consensus-/Differences-/Resolve-Engines: bei useOwnKeys nur
    die vom Nutzer uebermittelten Keys, sonst die zentral aufgeloesten
    Developer-Keys. Leere/Whitespace-Werte werden einheitlich zu None."""
    if not use_own_keys:
        # Eine gemeinsame Quelle fuer App, API und Benchmark verhindert, dass
        # ein Provider im Answer-Fan-out verfuegbar ist, im Judge-Plan aber
        # wegen abweichender Env-Namen oder Leerwert-Behandlung fehlt.
        return enable_gemini_adc(resolve_developer_api_keys())

    # Im Own-Key-Modus Developer-Keys strikt ignorieren. Sonst koennte der
    # Differences-Fallback unbemerkt einen Server-Key verwenden.
    return {
        label: str(data.get(field) or "").strip() or None
        for label, field in ENGINE_KEY_FIELDS.items()
    }


def cap_engine_text(value, limit: int):
    """Kappt clientseitig gelieferte Texte (Frage/Modellantworten), bevor sie
    in Consensus-/Differences-Prompts fliessen. Stilles Truncate statt 400:
    legitime Antworten liegen weit unter dem Limit, nur Abuse-Payloads nicht."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit].rstrip()


def validate_text_size(
    value,
    *,
    label: str,
    max_chars: int,
    max_bytes: int,
    required: bool = False,
) -> str:
    if not isinstance(value, str):
        if required:
            raise HTTPException(status_code=400, detail=f"{label} must be text.")
        return ""
    normalized = value.strip()
    if required and not normalized:
        raise HTTPException(status_code=400, detail=f"{label} must not be empty.")
    if len(normalized) > max_chars or len(normalized.encode("utf-8")) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} exceeds the limit of {max_chars} characters "
                f"or {max_bytes} UTF-8 bytes."
            ),
        )
    return normalized


def validate_client_system_prompt(value) -> str:
    if value is None or value == "":
        return ""
    return validate_text_size(
        value,
        label="System prompt",
        max_chars=MAX_SYSTEM_PROMPT_CHARS,
        max_bytes=MAX_SYSTEM_PROMPT_BYTES,
    )


def _chat_turn_ids(data: dict) -> Optional[tuple[str, str]]:
    chat_id = data.get("chat_id")
    turn_id = data.get("turn_id")
    if (chat_id is None) != (turn_id is None):
        raise HTTPException(
            status_code=400,
            detail="chat_id and turn_id must be provided together.",
        )
    if chat_id is None:
        return None
    if (
        not isinstance(chat_id, str)
        or not isinstance(turn_id, str)
        or not _CHAT_DOCUMENT_ID_RE.fullmatch(chat_id)
        or not _CHAT_DOCUMENT_ID_RE.fullmatch(turn_id)
    ):
        raise HTTPException(status_code=400, detail="Invalid chat or turn identifier.")
    return chat_id, turn_id


def _context_version_id(
    data: dict, chat_turn_ids: Optional[tuple[str, str]]
) -> Optional[str]:
    version_id = data.get("context_version_id")
    if version_id is None:
        return None
    if chat_turn_ids is None:
        raise HTTPException(
            status_code=400,
            detail="context_version_id requires chat_id and turn_id.",
        )
    if not isinstance(version_id, str) or not _CHAT_DOCUMENT_ID_RE.fullmatch(version_id):
        raise HTTPException(status_code=400, detail="Invalid context version identifier.")
    return version_id


def _chat_turn_disposition(
    ids: tuple[str, str],
    state: str,
    *,
    persisted: Optional[bool] = None,
) -> dict:
    chat_id, turn_id = ids
    result = {
        "chat_id": chat_id,
        "turn_id": turn_id,
        "chat_turn_state": state,
    }
    if persisted is not None:
        result["chat_persisted"] = persisted
    return result


def _chat_turn_error_detail(
    message: str,
    ids: tuple[str, str],
    state: str,
    *,
    persisted: Optional[bool] = None,
) -> dict:
    return {
        "error": message,
        **_chat_turn_disposition(ids, state, persisted=persisted),
    }


def _validate_chat_turn(uid: str, ids: tuple[str, str], question: str) -> dict:
    chat_id, turn_id = ids
    try:
        return chat_store.validate_turn_for_completion(
            uid,
            chat_id,
            turn_id,
            question=question,
        )
    except ChatNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=_chat_turn_error_detail(
                "Chat turn not found.", ids, "failed", persisted=False
            ),
        ) from exc
    except TurnQuestionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=_chat_turn_error_detail(
                "Consensus question does not match the pending turn.",
                ids,
                "failed",
                persisted=False,
            ),
        ) from exc
    except TurnStatusConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=_chat_turn_error_detail(
                "Chat turn is not completable.", ids, "failed", persisted=False
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_chat_turn_error_detail(
                "Invalid chat turn payload.", ids, "pending", persisted=False
            ),
        ) from exc
    except ChatStoreError as exc:
        logging.error(
            "chat turn preflight failed uid=%s chat_id=%s turn_id=%s",
            uid,
            chat_id,
            turn_id,
        )
        raise HTTPException(
            status_code=503,
            detail=_chat_turn_error_detail(
                "Chat persistence unavailable.", ids, "pending", persisted=False
            ),
        ) from exc
    except Exception as exc:
        logging.exception(
            "chat turn preflight failed uid=%s chat_id=%s turn_id=%s",
            uid,
            chat_id,
            turn_id,
        )
        raise HTTPException(
            status_code=503,
            detail=_chat_turn_error_detail(
                "Chat persistence unavailable.", ids, "pending", persisted=False
            ),
        ) from exc


def _fail_chat_turn_best_effort(
    uid: str,
    ids: Optional[tuple[str, str]],
    *,
    error_code: str,
) -> bool:
    if ids is None:
        return False
    chat_id, turn_id = ids
    try:
        failed_turn = chat_store.fail_turn(uid, chat_id, turn_id, error_code=error_code)
        return isinstance(failed_turn, dict) and failed_turn.get("status") == "failed"
    except Exception:
        logging.warning(
            "chat turn failure marker failed uid=%s chat_id=%s turn_id=%s code=%s",
            uid,
            chat_id,
            turn_id,
            error_code,
        )
        return False


def _replay_completed_chat_turn(
    uid: str,
    ids: tuple[str, str],
    *,
    stream_requested: bool,
):
    chat_id, turn_id = ids
    try:
        turn = chat_store.get_turn(uid, chat_id, turn_id)
    except ChatNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=_chat_turn_error_detail(
                "Chat turn not found.", ids, "failed", persisted=False
            ),
        ) from exc
    except Exception as exc:
        logging.exception(
            "completed chat turn replay read failed uid=%s chat_id=%s turn_id=%s",
            uid,
            chat_id,
            turn_id,
        )
        raise HTTPException(
            status_code=503,
            detail=_chat_turn_error_detail(
                "Chat persistence unavailable.", ids, "pending", persisted=False
            ),
        ) from exc

    consensus_text = turn.get("consensus")
    if not isinstance(consensus_text, str) or not consensus_text.strip():
        raise HTTPException(
            status_code=409,
            detail=_chat_turn_error_detail(
                "Completed chat turn has no replayable consensus.",
                ids,
                "failed",
                persisted=False,
            ),
        )

    payload = {
        "consensus_response": consensus_text,
        "differences": turn.get("differences")
        if isinstance(turn.get("differences"), str)
        else "",
        "differences_data": turn.get("differences_data"),
        "sources": turn.get("sources") if isinstance(turn.get("sources"), list) else [],
        "model_answers": (
            turn.get("model_answers")
            if isinstance(turn.get("model_answers"), dict)
            else {}
        ),
        **_chat_turn_disposition(ids, "completed", persisted=True),
        "chat_replayed": True,
    }
    result_id = turn.get("result_id")
    if isinstance(result_id, str) and result_id:
        payload["result_id"] = result_id

    if not stream_requested:
        return payload

    def replay_event_source():
        yield sse_pack("consensus.final", {"text": consensus_text})
        yield sse_pack("final", payload)

    return StreamingResponse(
        replay_event_source(),
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )


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
    previous_question = validate_text_size(
        previous_question,
        label="Previous question",
        max_chars=cfg.get_followup_question_char_limit(),
        max_bytes=cfg.get_followup_question_char_limit() * 4,
        required=True,
    )
    previous_consensus = validate_text_size(
        previous_consensus,
        label="Previous consensus",
        max_chars=cfg.get_followup_consensus_char_limit(),
        max_bytes=cfg.get_followup_consensus_char_limit() * 4,
        required=True,
    )
    return {
        "previous_question": previous_question,
        "previous_consensus": previous_consensus,
    }


def _resolve_authoritative_chat_context(
    uid: Optional[str],
    data: dict,
    question: str,
) -> Optional[str]:
    fields = (
        data.get("chat_id"),
        data.get("turn_id"),
        data.get("context_version_id"),
    )
    if not any(value is not None for value in fields):
        return None
    if uid is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if data.get("context") is not None:
        raise HTTPException(
            status_code=400,
            detail="Legacy context and context_version_id cannot be combined.",
        )
    if not all(
        isinstance(value, str) and _CHAT_DOCUMENT_ID_RE.fullmatch(value)
        for value in fields
    ):
        raise HTTPException(status_code=400, detail="Invalid chat context identifier.")
    chat_id, turn_id, version_id = fields
    try:
        # All six /ask_* calls of one run resolve the identical owner-bound
        # tuple. Resolving stays server-side (the context must never travel
        # through the client), but the cache pays the ~18 reads and six
        # renders exactly once per turn instead of once per provider.
        return resolved_context_cache.get_or_resolve(
            resolved_context_cache_key(uid, chat_id, turn_id, version_id, question),
            lambda: chat_context_service.resolve_for_ask(
                uid,
                chat_id,
                turn_id,
                version_id,
                question=question,
            ),
        )
    except ChatContextNotFound as exc:
        raise HTTPException(status_code=404, detail="Chat context not found") from exc
    except ChatContextConflict as exc:
        raise HTTPException(status_code=409, detail="Chat context conflict") from exc
    except ChatContextError as exc:
        logging.error("chat context resolution failed uid=%s: %s", uid, exc)
        raise HTTPException(status_code=503, detail="Chat context unavailable") from exc
    except Exception as exc:
        logging.exception("chat context resolution failed uid=%s", uid)
        raise HTTPException(status_code=503, detail="Chat context unavailable") from exc


def validate_question_word_limit(question: str, is_pro: bool, deep_search: bool):
    question = validate_text_size(
        question,
        label="Question",
        max_chars=MAX_QUESTION_CHARS,
        max_bytes=MAX_QUESTION_BYTES,
        required=True,
    )

    max_words_limit = cfg.get_word_limit(is_pro, deep_search)
    if count_words(question) > max_words_limit:
        raise HTTPException(status_code=400, detail=f"Input exceeds word limit of {max_words_limit}.")
    return question


def _attachment_claim_payload(attachments: list[dict]) -> list[dict]:
    return [
        {
            "name": str(item.get("name") or ""),
            "mime": str(item.get("mime") or ""),
            "sha256": hashlib.sha256(item.get("raw") or b"").hexdigest(),
        }
        for item in attachments
    ]

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
        source = provider.stream_fn(*args, **kwargs)

        def observed_stream():
            started = time.monotonic()
            outcome = "success"
            try:
                for event in source:
                    result = event.get("result") if isinstance(event, dict) else None
                    if isinstance(result, dict) and result.get("error"):
                        outcome = "failure"
                    yield event
            except TimeoutError:
                outcome = "timeout"
                raise
            except Exception:
                outcome = "failure"
                raise
            finally:
                record_metric(
                    "provider",
                    provider.label,
                    duration_ms=(time.monotonic() - started) * 1000,
                    outcome=outcome,
                )

        return streaming_model_response(observed_stream(), provider.label, extras)
    started = time.monotonic()
    outcome = "success"
    try:
        result = provider.query_fn(*args, **kwargs)
        if isinstance(result, dict) and result.get("error"):
            outcome = "failure"
        return source_response(result, **extras)
    except TimeoutError:
        outcome = "timeout"
        raise
    except Exception:
        outcome = "failure"
        raise
    finally:
        record_metric(
            "provider",
            provider.label,
            duration_ms=(time.monotonic() - started) * 1000,
            outcome=outcome,
        )


def handle_ask(provider: AskProvider, request: Request, data: dict):
    question = data.get("question")
    deep_search = parse_boolean_flag(data.get("deep_search", False))
    stream_requested = parse_boolean_flag(data.get("stream", False))
    system_prompt = validate_client_system_prompt(data.get("system_prompt"))
    id_token = extract_id_token(request, data)
    alt_key = data.get(provider.alt_key_field) if provider.alt_key_field else None
    api_key = str(data.get("api_key") or alt_key or "").strip()
    model = data.get("model")

    is_pro_user = False
    uid = None

    if id_token:
        try:
            uid = verify_user_token(id_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Authentication failed")
        try:
            is_pro_user = is_user_pro(uid)
        except TierStatusUnavailable:
            raise HTTPException(
                status_code=503,
                detail="Account tier is temporarily unavailable. Please retry.",
            ) from None

    # Deep Think ist strikt Pro-only.
    if deep_search and not is_pro_user:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    question = validate_question_word_limit(question, is_pro_user, deep_search)
    validate_model(
        model,
        getattr(cfg, provider.allowed_models_attr),
        provider.label,
        is_pro=is_pro_user,
    )
    attachments = parse_attachments(data, is_pro_user)
    max_tokens = cfg.get_output_token_limit(is_pro_user, deep_search)

    authoritative_context = _resolve_authoritative_chat_context(uid, data, question)
    if authoritative_context:
        base_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else get_system_prompt()
        )
        system_prompt = build_chat_context_system_prompt(
            base_prompt,
            authoritative_context,
        )

    # Legacy-Follow-up-Kontext: genau eine vorherige Frage/Konsens-Ebene,
    # serverseitig hart begrenzt und hier in den System-Prompt injiziert — nicht in
    # /prepare, damit der Kontext auch dann ankommt, wenn das Frontend nach
    # einem /prepare-Fehler mit dem Basis-Prompt weitermacht. Kein Tier-Gate:
    # ein Follow-up ist ein normaler Lauf und zaehlt gegen das Tagesbudget.
    followup_context = normalize_followup_context(data.get("context"))
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

    # --- Developer-Key/Service-Account: genau ein persistenter Slot pro Run ---
    if uid:
        developer_key = os.environ.get(provider.developer_key_env) if provider.developer_key_env else None
        if provider.developer_key_env and not developer_key:
            raise HTTPException(status_code=500, detail="Server error: API key missing")

        usage_key, _ = reserve_usage_run(
            uid, data, is_pro=is_pro_user, deep_think=deep_search
        )
        usage_result = consume_usage_run(uid, usage_key, is_pro=is_pro_user)
        claim_usage_operation(
            uid,
            usage_key,
            f"ask:{provider.label.lower()}",
            {
                "schema": 1,
                "provider": provider.label,
                "question": question,
                "system_prompt": system_prompt or "",
                "deep_search": deep_search,
                "model": model,
                "stream": stream_requested,
                "attachments": _attachment_claim_payload(attachments),
                "chat_id": data.get("chat_id"),
                "turn_id": data.get("turn_id"),
                "context_version_id": data.get("context_version_id"),
            },
        )

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
                **usage_response_fields(usage_result.snapshot, is_pro_user),
                "usage_run_status": usage_result.status.value,
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
def prepare(request: Request, data: dict = Body(...)):
    question = validate_text_size(
        data.get("question"),
        label="Question",
        max_chars=MAX_QUESTION_CHARS,
        max_bytes=MAX_QUESTION_BYTES,
        required=True,
    )

    use_own_keys = parse_boolean_flag(data.get("useOwnKeys", False))
    deep_think = parse_boolean_flag(data.get("deep_search", False))
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required to analyze intent.")

    try:
        uid = verify_user_token(id_token)
        try:
            is_pro = is_user_pro(uid)
        except TierStatusUnavailable:
            raise HTTPException(
                status_code=503,
                detail="Account tier is temporarily unavailable. Please retry.",
            ) from None
        if deep_think and not is_pro:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Deep Think is exclusively available for Pro users.",
                    "error_code": "pro_required",
                },
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Auth failed in /prepare: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed.")

    # Der Follow-up-Kontext wird hier bewusst NICHT verarbeitet: injiziert wird
    # er erst in den /ask_*-Endpoints (handle_ask), sonst stuende er doppelt im
    # System-Prompt. Ein Tier-Gate gibt es nicht mehr.

    validate_question_word_limit(question, is_pro, deep_think)
    raw_system_prompt = validate_client_system_prompt(data.get("system_prompt"))
    if not raw_system_prompt or not str(raw_system_prompt).strip():
        base_system_prompt = get_system_prompt()
    else:
        base_system_prompt = str(raw_system_prompt).strip()

    # Echtzeitdaten holen sich die Modelle selbst ueber die native Web-Suche,
    # die in jedem Provider-Call aktiv ist (siehe engines.py). Ein vorgeschalteter
    # Intent-Router mit Realtime-Injektion waere nur redundante, serielle Latenz.
    response = {
        "system_prompt": base_system_prompt,
        "sources": []
    }
    if not use_own_keys:
        usage_key, _ = reserve_usage_run(
            uid, data, is_pro=is_pro, deep_think=deep_think
        )
        # Den Slot sofort mitverbrauchen. Der anschliessende parallele
        # /ask_*-Fan-out ruft zwar weiterhin reserve+consume, trifft danach aber
        # ausschliesslich die idempotenten No-Write-Pfade (der Run ist bereits
        # consumed). Genau das behebt die "Usage accounting is temporarily busy"-
        # 503er: sechs gleichzeitige Read-Modify-WRITE-Transaktionen auf
        # demselben usage_days-Dokument erzeugten Firestore-Contention (Aborted).
        # Nach dem Vorab-Consume schreibt im Fan-out kein Request mehr; reine
        # Lese-Transaktionen kollidieren nie. Ergo genau ein wirksamer
        # Reserve+Consume pro Lauf, hier und seriell statt 6x parallel. Der
        # Consume im Fan-out bleibt als Selbstheilung erhalten (falls /prepare
        # nur reservieren, aber nicht konsumieren konnte).
        usage_result = consume_usage_run(uid, usage_key, is_pro=is_pro)
        response.update(usage_response_fields(usage_result.snapshot, is_pro))
        response["usage_run_status"] = usage_result.status.value
    return response


@router.post("/consensus")
@limiter.limit("5/minute")
def consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    chat_turn_ids = _chat_turn_ids(data)
    context_version_id = _context_version_id(data, chat_turn_ids)
    use_own_keys = str(data.get("useOwnKeys", "false")).lower() == "true"
    deep_think = parse_boolean_flag(data.get("deep_search", False))
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
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        is_pro = is_user_pro(uid)  # WICHTIG: Pro-Status prüfen
    except TierStatusUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Account tier is temporarily unavailable. Please retry.",
        ) from None

    # A completed turn is owner-bound stored history, not a new engine run.
    # Resolve it before current model/tier/credential checks so a later plan or
    # allowlist change cannot make an already completed answer unreadable.
    question = validate_text_size(
        data.get("question"),
        label="Question",
        max_chars=min(MAX_QUESTION_CHARS, cfg.get_consensus_question_char_limit()),
        max_bytes=MAX_QUESTION_BYTES,
    )
    validated_chat_turn_ids = None
    if question and chat_turn_ids is not None:
        validated_turn = _validate_chat_turn(uid, chat_turn_ids, question)
        validated_chat_turn_ids = chat_turn_ids
        linked_context_version_id = validated_turn.get("context_version_id")
        if linked_context_version_id != context_version_id:
            raise HTTPException(
                status_code=409,
                detail=_chat_turn_error_detail(
                    "Consensus context version does not match the pending turn.",
                    chat_turn_ids,
                    "pending",
                    persisted=False,
                ),
            )
        if validated_turn.get("status") == "completed":
            return _replay_completed_chat_turn(
                uid,
                validated_chat_turn_ids,
                stream_requested=stream_requested,
            )

    # Parameter extrahieren. Frage und Antworten kommen als freier Text vom
    # Client und werden serverseitig hart begrenzt: der Consensus-Prompt enthaelt
    # sonst unbegrenzte Eingaben gegen den Developer-Key (Kostenleck).
    answer_char_limit = cfg.get_consensus_answer_char_limit()
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
        if (
            model not in excluded_models
            and isinstance(answer, str)
            and answer.strip()
        )
    }

    # Ein einzelner ausgefallener Provider (Timeout, 503, leere Reasoning-
    # Antwort) darf den Konsens NICHT blockieren. Fehlt die Antwort eines
    # aktiven Modells, wird es einfach ausgelassen, solange noch mindestens
    # zwei Modelle geantwortet haben – der Consensus-/Differences-Prompt
    # filtert leere Antworten ohnehin heraus. Deshalb hier bewusst KEINE
    # Per-Modell-Pflichtprüfung mehr (die früher den ganzen Lauf mit 400
    # abbrach, sobald ein einzelnes Modell nicht lieferte).
    if len(included_answers) < 2:
        missing.append("at least two selected model answers")

    if missing:
        # An empty question skips the preflight above, so validated_chat_turn_ids
        # stays None — but the pending turn still exists and would otherwise stay
        # pending forever. fail_turn is owner-scoped and only moves pending ->
        # failed, so disposing an unvalidated turn is safe and never leaks state.
        disposable_ids = validated_chat_turn_ids or chat_turn_ids
        failed = _fail_chat_turn_best_effort(
            uid,
            disposable_ids,
            error_code=(
                "insufficient_answers"
                if len(included_answers) < 2
                else "consensus_failed"
            ),
        )
        message = "Missing parameters: " + ", ".join(missing)
        if disposable_ids is not None:
            raise HTTPException(
                status_code=400,
                detail=_chat_turn_error_detail(
                    message,
                    disposable_ids,
                    "failed" if failed else "pending",
                    persisted=False,
                ),
            )
        raise HTTPException(status_code=400, detail=message)

    if consensus_model not in cfg.ALLOWED_CONSENSUS_MODELS:
        raise HTTPException(status_code=400, detail="Invalid consensus model selected.")

    if deep_think and not is_pro:
        raise HTTPException(status_code=403, detail="Deep Think is exclusively available for Pro users.")

    # Premium-Engines bleiben Pro-exklusiv, AUCH im Own-Key-Modus: eigene Keys
    # bezahlen zwar den Call, heben aber das Tier-Gate nicht auf. Bewusst
    # unbedingt (nicht nur unter `not use_own_keys`) — genau diese zweite,
    # unbedingte Pruefung stand vor der Chat-Umsortierung weiter unten.
    if cfg.is_premium_consensus_model(consensus_model) and not is_pro:
        raise HTTPException(status_code=403, detail="Premium consensus engines are reserved for Pro users.")

    # API Keys setzen: Own-Key-Modus nutzt ausschliesslich Nutzer-Keys,
    # andernfalls kommt das vollstaendige Developer-Key-Set aus der gemeinsamen
    # Credential-Quelle. Bei unvollstaendigen Antworten ist der Turn bereits
    # autoritativ disponiert, bevor diese aktuelle Konfiguration relevant wird.
    api_keys = build_engine_api_keys(data, use_own_keys)
    
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
            # Erlaube zwei Varianten:
            # 1) expliziter Key aus dem autoritativen api_keys-Dict,
            # 2) Service Account, aber NUR ausserhalb des Own-Key-Modus.
            has_explicit_key = bool(api_keys.get("Gemini"))
            using_service_acct = (not use_own_keys) and bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

            if not (has_explicit_key or using_service_acct):
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

    usage_result = None
    if not use_own_keys:
        usage_key, _ = reserve_usage_run(
            uid, data, is_pro=is_pro, deep_think=deep_think
        )
        usage_result = consume_usage_run(uid, usage_key, is_pro=is_pro)
        claim_usage_operation(
            uid,
            usage_key,
            "consensus",
            {
                "schema": 1,
                "question": question,
                "consensus_model": consensus_model,
                "deep_think": deep_think,
                "stream": stream_requested,
                "answers": included_answers,
                "excluded_models": sorted(excluded_models),
                "model_sources": model_sources,
                "chat_id": data.get("chat_id"),
                "turn_id": data.get("turn_id"),
                "context_version_id": context_version_id,
            },
        )

    # Share-Feature: Ergebnis nur für verifizierte Nutzer persistieren.
    share_uid = uid
    model_labels = data.get("model_labels")

    allowed_model_labels = sanitize_model_labels(
        model_labels,
        list(included_answers.keys()),
    )
    chat_model_answers = {
        provider: {
            "provider": provider,
            "answer": answer,
            "model_label": allowed_model_labels.get(provider, provider),
            "sources": model_sources.get(provider, []),
        }
        for provider, answer in included_answers.items()
    }
    if "turn_sources" in data:
        raw_turn_sources = data.get("turn_sources")
    else:
        raw_turn_sources = [
            source
            for provider in included_answers
            for source in (
                model_sources.get(provider, [])
                if isinstance(model_sources.get(provider, []), list)
                else []
            )
        ]
    sanitized_turn_sources = sanitize_sources(raw_turn_sources)

    def record_run_stats(differences_data):
        # Anonyme Differences-Telemetrie (keine Texte, keine UID — siehe
        # app/services/differences_stats.py). Mock-Läufe (E2E) schreiben nicht.
        if differences_data is None or mock_llm_enabled():
            return
        record_differences_stats(
            differences_data,
            consensus_model=consensus_model,
            model_labels=model_labels,
            excluded_count=len(excluded_models),
            is_pro_user=bool(is_pro),
            used_own_keys=bool(use_own_keys),
            question_word_count=count_words(question),
        )

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

    def persist_chat_completion(
        consensus_text,
        differences_text,
        differences_data,
        result_id,
    ) -> bool:
        if validated_chat_turn_ids is None:
            return False
        chat_id, turn_id = validated_chat_turn_ids
        try:
            chat_store.complete_turn(
                uid,
                chat_id,
                turn_id,
                question=question,
                model_answers=chat_model_answers,
                consensus=consensus_text,
                differences=differences_text,
                differences_data=differences_data,
                sources=sanitized_turn_sources,
                result_id=result_id,
            )
            return True
        except Exception:
            # The LLM result is authoritative for the UI. Keep the pending turn
            # retryable and report only identifiers/state, never content/keys.
            logging.exception(
                "chat turn completion failed uid=%s chat_id=%s turn_id=%s",
                uid,
                chat_id,
                turn_id,
            )
            return False

    def add_chat_result_fields(
        payload: dict,
        *,
        persisted: bool,
        state: str,
    ) -> None:
        if validated_chat_turn_ids is None:
            return
        payload.update(
            _chat_turn_disposition(
                validated_chat_turn_ids,
                state,
                persisted=persisted,
            )
        )

    if stream_requested:
        extra_fields = {}
        if usage_result is not None:
            extra_fields = usage_response_fields(usage_result.snapshot, is_pro)
            extra_fields["usage_run_status"] = usage_result.status.value
        elif use_own_keys:
            extra_fields = {
                "free_usage_remaining": "Unlimited",
                "deep_remaining": "Unlimited",
                "is_pro_user": is_pro,
            }

        def consensus_event_source():
            consensus_text = ""
            consensus_failed = False
            differences_text = ""
            differences_data = None
            stream_failed = False
            # Reasoning-Marker der Engines gedrosselt weiterleiten (max. alle
            # 2 s, wie im /ask_*-Streaming): hält die Verbindung aktiv und
            # lässt das Frontend "Reasoning" statt eines stummen Spinners zeigen.
            last_reasoning_at = None

            def _reasoning_event(event_name):
                nonlocal last_reasoning_at
                now = time.monotonic()
                if last_reasoning_at is not None and now - last_reasoning_at < 2.0:
                    return None
                last_reasoning_at = now
                return sse_pack(event_name, {"reasoning": True})

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
                        text = coerce_text(item.get("text"))
                        if text:
                            yield sse_pack("consensus.delta", {"text": text})
                    elif item.get("type") == "reasoning":
                        event = _reasoning_event("consensus.delta")
                        if event:
                            yield event
                    else:
                        consensus_text = coerce_text(item.get("text"))
                        consensus_failed = bool(item.get("error")) or is_consensus_error_text(consensus_text)

                if consensus_failed:
                    # Ohne Konsensantwort ist der Vergleich sinnlos: der Judge
                    # würde sonst den Fehlertext "analysieren" und das Ergebnis
                    # würde als Share-Snapshot persistiert.
                    differences_text = DIFFERENCES_SKIPPED_TEXT
                    differences_data = None
                else:
                    # Consensus completion is its own successful phase. Send
                    # the authoritative text before the slower Differences
                    # judge so a later mobile/network interruption cannot
                    # erase an answer the user already received.
                    yield sse_pack("consensus.final", {"text": consensus_text})
                    last_reasoning_at = None
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
                            text = coerce_text(item.get("text"))
                            if text:
                                yield sse_pack("differences.delta", {"text": text})
                        elif item.get("type") == "reasoning":
                            event = _reasoning_event("differences.delta")
                            if event:
                                yield event
                        else:
                            differences_text = coerce_text(item.get("text"))
                            differences_data = item.get("data")
            except GeneratorExit:
                _fail_chat_turn_best_effort(
                    uid,
                    validated_chat_turn_ids,
                    error_code="cancelled",
                )
                raise
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
            result_id = None
            chat_persisted = False
            chat_turn_state = "pending"
            if not stream_failed and not consensus_failed:
                record_run_stats(differences_data)
                result_id = persist_share_result(consensus_text, differences_data, differences_text)
                if result_id:
                    payload["result_id"] = result_id
                chat_persisted = persist_chat_completion(
                    consensus_text,
                    differences_text,
                    differences_data,
                    result_id,
                )
                if chat_persisted:
                    chat_turn_state = "completed"
            else:
                failed = _fail_chat_turn_best_effort(
                    uid,
                    validated_chat_turn_ids,
                    error_code="consensus_failed",
                )
                if failed:
                    chat_turn_state = "failed"
            add_chat_result_fields(
                payload,
                persisted=chat_persisted,
                state=chat_turn_state,
            )
            payload.update(extra_fields)
            yield sse_pack("final", payload)

        # Keepalive-Wrapper: schiebt SSE-Kommentare ein, wenn die Engines
        # (z. B. ein denkender Reasoning-Judge) länger keine Bytes liefern —
        # sonst trennt Cloudflare idle Verbindungen und das final-Event geht
        # verloren (Spinner bliebe für immer stehen).
        return StreamingResponse(
            iter_sse_with_keepalive(consensus_event_source()),
            media_type="text/event-stream",
            headers=dict(SSE_HEADERS),
        )

    try:
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
    except Exception as exc:
        failed = _fail_chat_turn_best_effort(
            uid,
            validated_chat_turn_ids,
            error_code="consensus_failed",
        )
        if validated_chat_turn_ids is not None:
            raise HTTPException(
                status_code=500,
                detail=_chat_turn_error_detail(
                    "Consensus generation failed.",
                    validated_chat_turn_ids,
                    "failed" if failed else "pending",
                    persisted=False,
                ),
            ) from exc
        raise

    response = {
        "consensus_response": consensus_answer,
        "differences": differences,
        "differences_data": differences_data,
    }
    chat_persisted = False
    chat_turn_state = "pending"
    result_id = None
    if not consensus_failed:
        record_run_stats(differences_data)
        result_id = persist_share_result(consensus_answer, differences_data, differences)
        if result_id:
            response["result_id"] = result_id
        chat_persisted = persist_chat_completion(
            consensus_answer,
            differences,
            differences_data,
            result_id,
        )
        if chat_persisted:
            chat_turn_state = "completed"
    else:
        failed = _fail_chat_turn_best_effort(
            uid,
            validated_chat_turn_ids,
            error_code="consensus_failed",
        )
        if failed:
            chat_turn_state = "failed"
    add_chat_result_fields(
        response,
        persisted=chat_persisted,
        state=chat_turn_state,
    )
    if usage_result is not None:
        response.update(usage_response_fields(usage_result.snapshot, is_pro))
        response["usage_run_status"] = usage_result.status.value
    elif use_own_keys:
        response.update({
            "free_usage_remaining": "Unlimited",
            "deep_remaining": "Unlimited",
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
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        is_pro = is_user_pro(uid)
    except TierStatusUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Account tier is temporarily unavailable. Please retry.",
        ) from None

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

    question = validate_text_size(
        data.get("question"),
        label="Question",
        max_chars=min(MAX_QUESTION_CHARS, cfg.get_consensus_question_char_limit()),
        max_bytes=MAX_QUESTION_BYTES,
        required=True,
    )
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    try:
        claim, positions = normalize_resolve_positions(data.get("claim"), data.get("positions"))
    except InvalidResolvePayload as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    usage_result = None
    if not use_own_keys:
        usage_key, _ = reserve_usage_run(
            uid, data, is_pro=is_pro, deep_think=False
        )
        usage_result = consume_usage_run(uid, usage_key, is_pro=is_pro)
        claim_usage_operation(
            uid,
            usage_key,
            "resolve",
            {
                "schema": 1,
                "question": question,
                "claim": claim,
                "positions": positions,
            },
        )

    api_keys = build_engine_api_keys(data, use_own_keys)

    result = run_resolve_round(question, claim, positions, api_keys)
    if usage_result is not None:
        result.update(usage_response_fields(usage_result.snapshot, is_pro))
        result["usage_run_status"] = usage_result.status.value
    else:
        result.update({
            "free_usage_remaining": "Unlimited",
            "deep_remaining": "Unlimited",
            "is_pro_user": is_pro,
        })
    return result
