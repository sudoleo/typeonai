"""Owner-scoped persistence primitives for additive chat history.

This service deliberately does not call any LLM. It owns Firestore paths,
transactions, cursors, lifecycle validation, and the response-field allowlists
for chats, turns, and model answers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import unicodedata
from datetime import datetime, timezone
from typing import Callable, TypeVar

from firebase_admin import firestore
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
from google.cloud.firestore_v1.field_path import FieldPath

import app.core.config as cfg
from app.services.share_snapshots import (
    MAX_CONSENSUS_CHARS,
    MAX_DIFFERENCES_TEXT_CHARS,
    MAX_SOURCES,
    PROVIDER_ORDER,
    SHARE_ID_LENGTH,
    is_valid_share_id,
    sanitize_differences_data,
    sanitize_model_labels,
    sanitize_sources,
)


CHAT_SCHEMA_VERSION = 1
TURN_SCHEMA_VERSION = 1
MODEL_ANSWER_SCHEMA_VERSION = 1
CHAT_STATUS_ACTIVE = "active"
TURN_STATUS_PENDING = "pending"
TURN_STATUS_COMPLETED = "completed"
TURN_STATUS_FAILED = "failed"

PROVIDER_DOCUMENT_IDS = {
    "OpenAI": "openai",
    "Mistral": "mistral",
    "Anthropic": "anthropic",
    "Gemini": "gemini",
    "DeepSeek": "deepseek",
    "Grok": "grok",
}
MAX_MODEL_ANSWERS = len(PROVIDER_DOCUMENT_IDS)
FAILED_TURN_ERROR_CODES = frozenset({
    "consensus_failed",
    "cancelled",
    "insufficient_answers",
    "persistence_interrupted",
})

CHAT_TITLE_MAX_LENGTH = 120
QUESTION_MAX_LENGTH = 20_000
MODE_MAX_LENGTH = 40
MODEL_NAME_MAX_LENGTH = 120
SELECTED_MODELS_MAX_ITEMS = 8
CLIENT_REQUEST_ID_MAX_LENGTH = 128
LATEST_QUESTION_PREVIEW_MAX_LENGTH = 300
CONSENSUS_MAX_LENGTH = MAX_CONSENSUS_CHARS
DIFFERENCES_MAX_LENGTH = MAX_DIFFERENCES_TEXT_CHARS
MODEL_LABEL_MAX_LENGTH = 80
MODEL_ANSWER_MAX_LENGTH_FALLBACK = 40_000
MODEL_SOURCES_MAX_ITEMS = MAX_SOURCES
TURN_SOURCES_MAX_ITEMS = MAX_SOURCES
RESULT_ID_MAX_LENGTH = SHARE_ID_LENGTH
ERROR_CODE_MAX_LENGTH = max(len(code) for code in FAILED_TURN_ERROR_CODES)

CHAT_PAGE_SIZE = 30
CHAT_PAGE_SIZE_MAX = 50
TURN_PAGE_SIZE = 50
TURN_PAGE_SIZE_MAX = 100

_ID_RE = re.compile(r"[0-9a-f]{32}")
_CLIENT_REQUEST_ID_RE = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{CLIENT_REQUEST_ID_MAX_LENGTH - 1}}}"
)
_CURSOR_VERSION = 2
_MAX_FIRESTORE_INTEGER = 9_223_372_036_854_775_807
_CURSOR_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.(?:\d{6}|\d{9})Z"
)
_T = TypeVar("_T")


class _CursorTimestamp(DatetimeWithNanoseconds):
    """Immutable timestamp whose nanoseconds survive Firestore query copying.

    google-cloud-firestore deep-copies dict cursors. The installed SDK's base
    ``DatetimeWithNanoseconds`` loses its private nanosecond remainder during
    that copy, so an immutable self-copy keeps the signed boundary lossless.
    """

    def __deepcopy__(self, memo):
        return self


class ChatStoreError(Exception):
    """Base class for expected chat-store failures."""


class ChatNotFound(ChatStoreError):
    pass


class InvalidChatCursor(ChatStoreError):
    pass


class ChatCursorUnavailable(ChatStoreError):
    pass


class ChatIdempotencyConflict(ChatStoreError):
    pass


class TurnStatusConflict(ChatStoreError):
    pass


class TurnCompletionConflict(TurnStatusConflict):
    pass


class TurnQuestionConflict(TurnStatusConflict):
    pass


def normalize_title(value: object) -> str:
    text = _collapse_whitespace(value)
    if len(text) > CHAT_TITLE_MAX_LENGTH:
        raise ValueError(f"title must be at most {CHAT_TITLE_MAX_LENGTH} characters")
    return text


def derive_title(question: object) -> str:
    return _collapse_whitespace(question)[:CHAT_TITLE_MAX_LENGTH]


def normalize_question(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("question must be a string")
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > QUESTION_MAX_LENGTH:
        raise ValueError(f"question must be at most {QUESTION_MAX_LENGTH} characters")
    return text


def normalize_mode(value: object) -> str:
    text = _collapse_whitespace(value)
    if not text:
        raise ValueError("mode must not be empty")
    if len(text) > MODE_MAX_LENGTH:
        raise ValueError(f"mode must be at most {MODE_MAX_LENGTH} characters")
    return text


def normalize_model_name(value: object, *, field_name: str = "model") -> str:
    text = _collapse_whitespace(value)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > MODEL_NAME_MAX_LENGTH:
        raise ValueError(
            f"{field_name} must be at most {MODEL_NAME_MAX_LENGTH} characters"
        )
    return text


def normalize_selected_models(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("selected_models must be a list")
    if not value:
        raise ValueError("selected_models must not be empty")
    if len(value) > SELECTED_MODELS_MAX_ITEMS:
        raise ValueError(
            f"selected_models must contain at most {SELECTED_MODELS_MAX_ITEMS} items"
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        model = normalize_model_name(item, field_name="selected_models item")
        if model not in seen:
            normalized.append(model)
            seen.add(model)
    return normalized


def normalize_client_request_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("client_request_id must be a string")
    text = unicodedata.normalize("NFKC", value).strip()
    if not _CLIENT_REQUEST_ID_RE.fullmatch(text):
        raise ValueError(
            "client_request_id must be 1-128 ASCII letters, digits, dots, "
            "underscores, colons, or hyphens"
        )
    return text


def latest_question_preview(value: object) -> str:
    return _collapse_whitespace(value)[:LATEST_QUESTION_PREVIEW_MAX_LENGTH]


def normalize_provider(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("provider must be a supported provider")
    candidate = unicodedata.normalize("NFKC", value).strip().lower()
    for provider, document_id in PROVIDER_DOCUMENT_IDS.items():
        if candidate in {provider.lower(), document_id}:
            return provider
    raise ValueError("provider must be a supported provider")


def normalize_result_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("result_id must be a valid result identifier")
    result_id = unicodedata.normalize("NFKC", value).strip()
    if not result_id:
        return None
    if len(result_id) > RESULT_ID_MAX_LENGTH or not is_valid_share_id(result_id):
        raise ValueError("result_id must be a valid result identifier")
    return result_id


def normalize_error_code(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("error_code must be an allowed code")
    error_code = unicodedata.normalize("NFKC", value).strip().lower()
    if (
        len(error_code) > ERROR_CODE_MAX_LENGTH
        or error_code not in FAILED_TURN_ERROR_CODES
    ):
        raise ValueError("error_code must be an allowed code")
    return error_code


def _bounded_text(
    value: object,
    limit: int,
    *,
    field_name: str,
    required: bool = False,
) -> str:
    if not isinstance(value, str):
        if required:
            raise ValueError(f"{field_name} must be a string")
        return ""
    text = unicodedata.normalize("NFKC", value).strip()
    if required and not text:
        raise ValueError(f"{field_name} must not be empty")
    return text[:limit].rstrip()


def _model_answer_char_limit() -> int:
    try:
        value = int(cfg.get_consensus_answer_char_limit())
    except (TypeError, ValueError):
        value = MODEL_ANSWER_MAX_LENGTH_FALLBACK
    return max(1, value)


def normalize_model_answers(value: object) -> dict[str, dict]:
    if isinstance(value, dict):
        entries = [
            ({**item, "provider": provider} if isinstance(item, dict) else {
                "provider": provider,
                "answer": item,
            })
            for provider, item in value.items()
        ]
    elif isinstance(value, list):
        entries = list(value)
    else:
        raise ValueError("model_answers must be a mapping or list")
    if len(entries) > MAX_MODEL_ANSWERS:
        raise ValueError(f"model_answers must contain at most {MAX_MODEL_ANSWERS} items")

    normalized: dict[str, dict] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("model answer entries must be objects")
        provider = normalize_provider(item.get("provider"))
        if provider in normalized:
            raise ValueError("model_answers contains a duplicate provider")
        answer = _bounded_text(
            item.get("answer"),
            _model_answer_char_limit(),
            field_name="answer",
        )
        if not answer:
            continue
        labels = sanitize_model_labels(
            {provider: item.get("model_label")},
            [provider],
        )
        normalized[provider] = {
            "schema_version": MODEL_ANSWER_SCHEMA_VERSION,
            "provider": provider,
            "model_label": labels.get(provider, provider)[:MODEL_LABEL_MAX_LENGTH],
            "answer": answer,
            "sources": sanitize_sources(item.get("sources"))[
                :MODEL_SOURCES_MAX_ITEMS
            ],
        }
    return {
        provider: normalized[provider]
        for provider in PROVIDER_ORDER
        if provider in normalized
    }


def normalize_turn_sources(value: object) -> list[dict]:
    return sanitize_sources(value)[:TURN_SOURCES_MAX_ITEMS]


def normalize_turn_differences_data(value: object) -> dict | None:
    sanitized = sanitize_differences_data(value)
    if sanitized is None:
        return None
    raw_agreement = value.get("agreement") if isinstance(value, dict) else None
    raw_score = raw_agreement.get("score") if isinstance(raw_agreement, dict) else None
    if not _valid_supplied_score(raw_score):
        agreement = sanitized.get("agreement")
        if isinstance(agreement, dict):
            agreement.pop("score", None)
    return sanitized


def _valid_supplied_score(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _agreement_score(differences_data: object) -> int | None:
    if not isinstance(differences_data, dict):
        return None
    agreement = differences_data.get("agreement")
    if not isinstance(agreement, dict):
        return None
    score = agreement.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        return None
    return score if 0 <= score <= 100 else None


def chat_metadata(chat_id: object, data: object, *, compact: bool = False) -> dict:
    source = data if isinstance(data, dict) else {}
    result = {
        "id": str(chat_id),
        "title": str(source.get("title") or ""),
        "status": str(source.get("status") or ""),
        "created_at": source.get("created_at"),
        "updated_at": source.get("updated_at"),
        "turn_count": _safe_non_negative_int(source.get("turn_count")),
        "latest_question": str(source.get("latest_question") or ""),
    }
    if not compact:
        result = {
            "schema_version": _safe_positive_int(
                source.get("schema_version"), CHAT_SCHEMA_VERSION
            ),
            **result,
        }
    return result


def turn_metadata(turn_id: object, data: object) -> dict:
    source = data if isinstance(data, dict) else {}
    result = {
        "schema_version": _safe_positive_int(
            source.get("schema_version"), TURN_SCHEMA_VERSION
        ),
        "id": str(turn_id),
        "position": _safe_positive_int(source.get("position"), 0),
        "status": str(source.get("status") or ""),
        "question": str(source.get("question") or ""),
        "mode": str(source.get("mode") or ""),
        "deep_search": source.get("deep_search") is True,
        "selected_models": [
            str(item) for item in source.get("selected_models", [])
            if isinstance(item, str)
        ],
        "consensus_model": str(source.get("consensus_model") or ""),
        "created_at": source.get("created_at"),
        "updated_at": source.get("updated_at"),
    }
    client_request_id = source.get("client_request_id")
    if isinstance(client_request_id, str) and client_request_id:
        result["client_request_id"] = client_request_id
    context_version_id = source.get("context_version_id")
    if isinstance(context_version_id, str) and _ID_RE.fullmatch(context_version_id):
        result["context_version_id"] = context_version_id
    for field in ("completed_at", "failed_at"):
        if source.get(field) is not None:
            result[field] = source.get(field)
    if "answer_count" in source:
        result["answer_count"] = min(
            MAX_MODEL_ANSWERS,
            _safe_non_negative_int(source.get("answer_count")),
        )
    score = source.get("agreement_score")
    if isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 100:
        result["agreement_score"] = score
    error_code = source.get("error_code")
    if isinstance(error_code, str) and error_code in FAILED_TURN_ERROR_CODES:
        result["error_code"] = error_code
    return result


def model_answer_metadata(data: object) -> dict | None:
    source = data if isinstance(data, dict) else {}
    try:
        provider = normalize_provider(source.get("provider"))
    except ValueError:
        return None
    answer = source.get("answer")
    if not isinstance(answer, str) or not answer:
        return None
    labels = sanitize_model_labels(
        {provider: source.get("model_label")},
        [provider],
    )
    result = {
        "schema_version": _safe_positive_int(
            source.get("schema_version"), MODEL_ANSWER_SCHEMA_VERSION
        ),
        "provider": provider,
        "model_label": labels.get(provider, provider)[:MODEL_LABEL_MAX_LENGTH],
        "answer": answer[:_model_answer_char_limit()],
        "sources": sanitize_sources(source.get("sources"))[:MODEL_SOURCES_MAX_ITEMS],
    }
    for field in ("created_at", "updated_at"):
        if source.get(field) is not None:
            result[field] = source.get(field)
    return result


def turn_detail(turn_id: object, data: object, model_answers: dict[str, dict]) -> dict:
    source = data if isinstance(data, dict) else {}
    result = turn_metadata(turn_id, source)
    if "consensus" in source:
        result["consensus"] = _bounded_text(
            source.get("consensus"), CONSENSUS_MAX_LENGTH, field_name="consensus"
        )
    if "differences" in source:
        result["differences"] = _bounded_text(
            source.get("differences"),
            DIFFERENCES_MAX_LENGTH,
            field_name="differences",
        )
    if "differences_data" in source:
        result["differences_data"] = normalize_turn_differences_data(
            source.get("differences_data")
        )
    if "sources" in source:
        result["sources"] = normalize_turn_sources(source.get("sources"))
    if "included_models" in source:
        included = source.get("included_models")
        result["included_models"] = [
            provider
            for provider in PROVIDER_ORDER
            if isinstance(included, list) and provider in included
        ][:MAX_MODEL_ANSWERS]
    if "result_id" in source:
        try:
            stored_result_id = normalize_result_id(source.get("result_id"))
        except ValueError:
            stored_result_id = None
        if stored_result_id is not None:
            result["result_id"] = stored_result_id
    result["model_answers"] = {
        provider: model_answers[provider]
        for provider in PROVIDER_ORDER
        if provider in model_answers
    }
    return result


class ChatStore:
    def __init__(
        self,
        db,
        *,
        transaction_runner: Callable[[Callable[[object], _T]], _T] | None = None,
    ):
        self.db = db
        self._transaction_runner = transaction_runner

    def create_chat(self, uid: str, *, title: str = "") -> dict:
        title = normalize_title(title)
        chat_id = secrets.token_hex(16)
        ref = self._chats_ref(uid).document(chat_id)
        document = {
            "schema_version": CHAT_SCHEMA_VERSION,
            "title": title,
            "status": CHAT_STATUS_ACTIVE,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "turn_count": 0,
            "latest_question": "",
        }
        ref.set(document)
        snapshot = ref.get()
        return chat_metadata(snapshot.id, snapshot.to_dict() or {})

    def get_chat(self, uid: str, chat_id: str) -> dict:
        ref = self._chat_ref(uid, chat_id)
        snapshot = ref.get()
        if not snapshot.exists:
            raise ChatNotFound("Chat not found")
        return chat_metadata(snapshot.id, snapshot.to_dict() or {})

    def get_turn(self, uid: str, chat_id: str, turn_id: str) -> dict:
        chat_ref = self._chat_ref(uid, chat_id)
        turn_ref = self._turn_ref(uid, chat_id, turn_id)
        if not chat_ref.get().exists:
            raise ChatNotFound("Chat not found")
        turn_snapshot = turn_ref.get()
        if not turn_snapshot.exists:
            raise ChatNotFound("Chat not found")

        answers: dict[str, dict] = {}
        answers_ref = turn_ref.collection("model_answers")
        for provider in PROVIDER_ORDER:
            document_id = PROVIDER_DOCUMENT_IDS[provider]
            snapshot = answers_ref.document(document_id).get()
            if not snapshot.exists:
                continue
            answer = model_answer_metadata(snapshot.to_dict() or {})
            if answer is not None and answer["provider"] == provider:
                answers[provider] = answer
        return turn_detail(turn_snapshot.id, turn_snapshot.to_dict() or {}, answers)

    def validate_turn_for_completion(
        self,
        uid: str,
        chat_id: str,
        turn_id: str,
        *,
        question: str,
    ) -> dict:
        """Read-only owner/path/question check before an expensive consensus.

        Completed turns remain eligible so the consensus endpoint can replay
        their stored full detail without invoking an engine or writing again.
        Failed turns are terminal and cannot be completed.
        """
        question = normalize_question(question)
        chat_ref = self._chat_ref(uid, chat_id)
        turn_ref = self._turn_ref(uid, chat_id, turn_id)
        if not chat_ref.get().exists:
            raise ChatNotFound("Chat not found")
        snapshot = turn_ref.get()
        if not snapshot.exists:
            raise ChatNotFound("Chat not found")
        data = snapshot.to_dict() or {}
        if data.get("question") != question:
            raise TurnQuestionConflict("Turn question does not match")
        if data.get("status") not in {TURN_STATUS_PENDING, TURN_STATUS_COMPLETED}:
            raise TurnStatusConflict("Turn status transition is not allowed")
        return turn_metadata(snapshot.id, data)

    def list_chats(
        self, uid: str, *, limit: int = CHAT_PAGE_SIZE, cursor: str = ""
    ) -> dict:
        chats_ref = self._chats_ref(uid)
        query = chats_ref.order_by(
            "updated_at", direction=firestore.Query.DESCENDING
        ).order_by(
            FieldPath.document_id(), direction=firestore.Query.DESCENDING
        )
        if cursor:
            cursor_timestamp, cursor_id = _decode_chat_cursor(
                cursor, owner_scope=uid
            )
            query = query.start_after({
                "updated_at": cursor_timestamp,
                FieldPath.document_id(): cursor_id,
            })
        snapshots = list(query.limit(limit + 1).stream())
        has_more = len(snapshots) > limit
        page = snapshots[:limit]
        next_cursor = None
        if has_more and page:
            boundary = page[-1]
            next_cursor = _encode_chat_cursor(
                boundary.to_dict().get("updated_at"),
                boundary.id,
                owner_scope=uid,
            )
        return {
            "chats": [
                chat_metadata(snapshot.id, snapshot.to_dict() or {}, compact=True)
                for snapshot in page
            ],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    def create_turn(
        self,
        uid: str,
        chat_id: str,
        *,
        question: str,
        mode: str,
        deep_search: bool,
        selected_models: list[str],
        consensus_model: str,
        client_request_id: str | None = None,
    ) -> dict:
        question = normalize_question(question)
        mode = normalize_mode(mode)
        if not isinstance(deep_search, bool):
            raise ValueError("deep_search must be a boolean")
        selected_models = normalize_selected_models(selected_models)
        consensus_model = normalize_model_name(
            consensus_model, field_name="consensus_model"
        )
        client_request_id = normalize_client_request_id(client_request_id)
        chat_ref = self._chat_ref(uid, chat_id)
        if client_request_id:
            turn_id = _idempotent_turn_id(chat_id, client_request_id)
        else:
            turn_id = secrets.token_hex(16)
        turn_ref = chat_ref.collection("turns").document(turn_id)

        def operation(transaction):
            chat_snapshot = chat_ref.get(transaction=transaction)
            if not chat_snapshot.exists:
                raise ChatNotFound("Chat not found")

            if client_request_id:
                existing = turn_ref.get(transaction=transaction)
                if existing.exists:
                    if not _same_turn_request(
                        existing.to_dict() or {},
                        question=question,
                        mode=mode,
                        deep_search=deep_search,
                        selected_models=selected_models,
                        consensus_model=consensus_model,
                    ):
                        raise ChatIdempotencyConflict(
                            "client_request_id conflicts with an existing turn"
                        )
                    return

            chat_data = chat_snapshot.to_dict() or {}
            position = _safe_non_negative_int(chat_data.get("turn_count")) + 1
            turn_document = {
                "schema_version": TURN_SCHEMA_VERSION,
                "position": position,
                "status": TURN_STATUS_PENDING,
                "question": question,
                "mode": mode,
                "deep_search": deep_search,
                "selected_models": list(selected_models),
                "consensus_model": consensus_model,
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            if client_request_id:
                turn_document["client_request_id"] = client_request_id

            chat_updates = {
                "turn_count": position,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "latest_question": latest_question_preview(question),
            }
            if position == 1 and not str(chat_data.get("title") or "").strip():
                chat_updates["title"] = derive_title(question)

            transaction.set(turn_ref, turn_document)
            transaction.update(chat_ref, chat_updates)

        self._transaction(operation)
        snapshot = turn_ref.get()
        if not snapshot.exists:
            raise ChatStoreError("Turn was not persisted")
        return turn_metadata(snapshot.id, snapshot.to_dict() or {})

    def complete_turn(
        self,
        uid: str,
        chat_id: str,
        turn_id: str,
        *,
        question: str,
        model_answers: object,
        consensus: str,
        differences: str,
        differences_data: object,
        sources: object,
        result_id: object = None,
    ) -> dict:
        question = normalize_question(question)
        normalized_answers = normalize_model_answers(model_answers)
        consensus = _bounded_text(
            consensus,
            CONSENSUS_MAX_LENGTH,
            field_name="consensus",
            required=True,
        )
        differences = _bounded_text(
            differences,
            DIFFERENCES_MAX_LENGTH,
            field_name="differences",
        )
        differences_data = normalize_turn_differences_data(differences_data)
        sources = normalize_turn_sources(sources)
        result_id = normalize_result_id(result_id)
        included_models = list(normalized_answers)
        agreement_score = _agreement_score(differences_data)

        chat_ref = self._chat_ref(uid, chat_id)
        turn_ref = self._turn_ref(uid, chat_id, turn_id)
        fingerprint = _completion_fingerprint(
            chat_id=chat_id,
            turn_id=turn_id,
            question=question,
            model_answers=normalized_answers,
            consensus=consensus,
            differences=differences,
            differences_data=differences_data,
            sources=sources,
            result_id=result_id,
        )

        def operation(transaction):
            chat_snapshot = chat_ref.get(transaction=transaction)
            turn_snapshot = turn_ref.get(transaction=transaction)
            if not chat_snapshot.exists or not turn_snapshot.exists:
                raise ChatNotFound("Chat not found")
            turn_data = turn_snapshot.to_dict() or {}
            if turn_data.get("question") != question:
                raise TurnQuestionConflict("Turn question does not match")

            status = turn_data.get("status")
            if status == TURN_STATUS_COMPLETED:
                stored = turn_data.get("completion_fingerprint")
                if isinstance(stored, str) and hmac.compare_digest(stored, fingerprint):
                    return
                raise TurnCompletionConflict("Turn completion conflicts with stored result")
            if status != TURN_STATUS_PENDING:
                raise TurnStatusConflict("Turn status transition is not allowed")

            answers_ref = turn_ref.collection("model_answers")
            for provider, answer in normalized_answers.items():
                answer_document = {
                    **answer,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                transaction.set(
                    answers_ref.document(PROVIDER_DOCUMENT_IDS[provider]),
                    answer_document,
                )

            turn_updates = {
                "status": TURN_STATUS_COMPLETED,
                "consensus": consensus,
                "differences": differences,
                "differences_data": differences_data,
                "sources": sources,
                "included_models": included_models,
                "answer_count": len(normalized_answers),
                "completion_fingerprint": fingerprint,
                "completed_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            if agreement_score is not None:
                turn_updates["agreement_score"] = agreement_score
            if result_id is not None:
                turn_updates["result_id"] = result_id
            transaction.update(turn_ref, turn_updates)
            transaction.update(chat_ref, {"updated_at": firestore.SERVER_TIMESTAMP})

        self._transaction(operation)
        return self.get_turn(uid, chat_id, turn_id)

    def fail_turn(
        self,
        uid: str,
        chat_id: str,
        turn_id: str,
        *,
        error_code: str,
    ) -> dict:
        error_code = normalize_error_code(error_code)
        chat_ref = self._chat_ref(uid, chat_id)
        turn_ref = self._turn_ref(uid, chat_id, turn_id)

        def operation(transaction):
            chat_snapshot = chat_ref.get(transaction=transaction)
            turn_snapshot = turn_ref.get(transaction=transaction)
            if not chat_snapshot.exists or not turn_snapshot.exists:
                raise ChatNotFound("Chat not found")
            turn_data = turn_snapshot.to_dict() or {}
            status = turn_data.get("status")
            if status == TURN_STATUS_FAILED:
                if turn_data.get("error_code") == error_code:
                    return
                raise TurnStatusConflict("Turn failure conflicts with stored status")
            if status != TURN_STATUS_PENDING:
                raise TurnStatusConflict("Turn status transition is not allowed")
            transaction.update(turn_ref, {
                "status": TURN_STATUS_FAILED,
                "error_code": error_code,
                "failed_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })

        self._transaction(operation)
        snapshot = turn_ref.get()
        if not snapshot.exists:
            raise ChatNotFound("Chat not found")
        return turn_metadata(snapshot.id, snapshot.to_dict() or {})

    def list_turns(
        self,
        uid: str,
        chat_id: str,
        *,
        limit: int = TURN_PAGE_SIZE,
        cursor: str = "",
    ) -> dict:
        chat_ref = self._chat_ref(uid, chat_id)
        if not chat_ref.get().exists:
            raise ChatNotFound("Chat not found")
        turns_ref = chat_ref.collection("turns")
        query = turns_ref.order_by(
            "position", direction=firestore.Query.ASCENDING
        ).order_by(
            FieldPath.document_id(), direction=firestore.Query.ASCENDING
        )
        if cursor:
            cursor_position, cursor_id = _decode_turn_cursor(
                cursor, owner_scope=f"{uid}:{chat_id}"
            )
            query = query.start_after({
                "position": cursor_position,
                FieldPath.document_id(): cursor_id,
            })
        snapshots = list(query.limit(limit + 1).stream())
        has_more = len(snapshots) > limit
        page = snapshots[:limit]
        next_cursor = None
        if has_more and page:
            boundary = page[-1]
            next_cursor = _encode_turn_cursor(
                boundary.to_dict().get("position"),
                boundary.id,
                owner_scope=f"{uid}:{chat_id}",
            )
        return {
            "turns": [
                turn_metadata(snapshot.id, snapshot.to_dict() or {})
                for snapshot in page
            ],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    def _transaction(self, operation: Callable[[object], _T]) -> _T:
        if self._transaction_runner is not None:
            return self._transaction_runner(operation)
        fake_runner = getattr(self.db, "run_transaction", None)
        if callable(fake_runner):
            return fake_runner(operation)
        transaction = self.db.transaction(max_attempts=12)

        @firestore.transactional
        def run(tx):
            return operation(tx)

        return run(transaction)

    def _chats_ref(self, uid: str):
        return self.db.collection("users").document(uid).collection("chats")

    def _chat_ref(self, uid: str, chat_id: str):
        if not _ID_RE.fullmatch(str(chat_id or "")):
            raise ChatNotFound("Chat not found")
        return self._chats_ref(uid).document(chat_id)

    def _turn_ref(self, uid: str, chat_id: str, turn_id: str):
        if not _ID_RE.fullmatch(str(turn_id or "")):
            raise ChatNotFound("Chat not found")
        return self._chat_ref(uid, chat_id).collection("turns").document(turn_id)


def _collapse_whitespace(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _safe_non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _safe_positive_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _idempotent_turn_id(chat_id: str, client_request_id: str) -> str:
    digest = hashlib.sha256(
        f"chat-turn\0{chat_id}\0{client_request_id}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _same_turn_request(
    existing: dict,
    *,
    question: str,
    mode: str,
    deep_search: bool,
    selected_models: list[str],
    consensus_model: str,
) -> bool:
    return (
        existing.get("question") == question
        and existing.get("mode") == mode
        and existing.get("deep_search") is deep_search
        and existing.get("selected_models") == selected_models
        and existing.get("consensus_model") == consensus_model
    )


def _completion_fingerprint(
    *,
    chat_id: str,
    turn_id: str,
    question: str,
    model_answers: dict[str, dict],
    consensus: str,
    differences: str,
    differences_data: dict | None,
    sources: list[dict],
    result_id: str | None,
) -> str:
    payload = {
        "schema_version": TURN_SCHEMA_VERSION,
        "chat_id": chat_id,
        "turn_id": turn_id,
        "question": question,
        "model_answers": {
            PROVIDER_DOCUMENT_IDS[provider]: model_answers[provider]
            for provider in PROVIDER_ORDER
            if provider in model_answers
        },
        "consensus": consensus,
        "differences": differences,
        "differences_data": differences_data,
        "sources": sources,
        "result_id": result_id,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cursor_secret() -> bytes:
    value = (
        os.environ.get("CHAT_CURSOR_SECRET", "").strip()
        or os.environ.get("WATCH_UNSUBSCRIBE_SECRET", "").strip()
    )
    if len(value) < 16:
        raise ChatCursorUnavailable("Chat cursor signing is not configured")
    return value.encode("utf-8")


def _encode_chat_cursor(
    updated_at: object, document_id: str, *, owner_scope: str
) -> str:
    if not _ID_RE.fullmatch(str(document_id or "")):
        raise ChatStoreError("Chat cursor boundary has an invalid document ID")
    return _encode_cursor_payload(
        {
            "v": _CURSOR_VERSION,
            "k": "chats",
            "updated_at": _timestamp_to_cursor(updated_at),
            "id": document_id,
        },
        owner_scope=owner_scope,
    )


def _encode_turn_cursor(
    position: object, document_id: str, *, owner_scope: str
) -> str:
    if not _valid_turn_position(position):
        raise ChatStoreError("Turn cursor boundary has an invalid position")
    if not _ID_RE.fullmatch(str(document_id or "")):
        raise ChatStoreError("Turn cursor boundary has an invalid document ID")
    return _encode_cursor_payload(
        {
            "v": _CURSOR_VERSION,
            "k": "turns",
            "position": position,
            "id": document_id,
        },
        owner_scope=owner_scope,
    )


def _encode_cursor_payload(payload_data: dict, *, owner_scope: str) -> str:
    payload = json.dumps(
        payload_data,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded_payload = _b64encode(payload)
    signature = hmac.new(
        _cursor_secret(),
        f"chat-cursor\0{owner_scope}\0{encoded_payload}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def _decode_chat_cursor(
    cursor: str, *, owner_scope: str
) -> tuple[DatetimeWithNanoseconds, str]:
    payload = _decode_cursor_payload(cursor, kind="chats", owner_scope=owner_scope)
    if set(payload) != {"v", "k", "updated_at", "id"}:
        raise InvalidChatCursor("Invalid chat cursor")
    timestamp_text = payload.get("updated_at")
    if not isinstance(timestamp_text, str) or not _CURSOR_TIMESTAMP_RE.fullmatch(
        timestamp_text
    ):
        raise InvalidChatCursor("Invalid chat cursor")
    try:
        timestamp = _CursorTimestamp.from_rfc3339(timestamp_text)
    except (TypeError, ValueError) as exc:
        raise InvalidChatCursor("Invalid chat cursor") from exc
    if _timestamp_to_cursor(timestamp) != timestamp_text:
        raise InvalidChatCursor("Invalid chat cursor")
    return timestamp, payload["id"]


def _decode_turn_cursor(cursor: str, *, owner_scope: str) -> tuple[int, str]:
    payload = _decode_cursor_payload(cursor, kind="turns", owner_scope=owner_scope)
    if set(payload) != {"v", "k", "position", "id"}:
        raise InvalidChatCursor("Invalid chat cursor")
    position = payload.get("position")
    if not _valid_turn_position(position):
        raise InvalidChatCursor("Invalid chat cursor")
    return position, payload["id"]


def _decode_cursor_payload(cursor: str, *, kind: str, owner_scope: str) -> dict:
    try:
        encoded_payload, encoded_signature = str(cursor).split(".", 1)
        expected = hmac.new(
            _cursor_secret(),
            f"chat-cursor\0{owner_scope}\0{encoded_payload}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        supplied = _b64decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            raise ValueError("invalid signature")
        payload = json.loads(_b64decode(encoded_payload).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid cursor payload")
        document_id = payload.get("id")
        version = payload.get("v")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != _CURSOR_VERSION
            or payload.get("k") != kind
        ):
            raise ValueError("invalid cursor scope")
        if not isinstance(document_id, str) or not _ID_RE.fullmatch(document_id):
            raise ValueError("invalid document id")
        return payload
    except ChatCursorUnavailable:
        raise
    except Exception as exc:
        raise InvalidChatCursor("Invalid chat cursor") from exc


def _timestamp_to_cursor(value: object) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ChatStoreError("Chat cursor boundary has an invalid timestamp")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ChatStoreError("Chat cursor boundary has an invalid timestamp") from exc
    if offset is None:
        raise ChatStoreError("Chat cursor boundary has an invalid timestamp")
    if isinstance(value, DatetimeWithNanoseconds):
        if offset.total_seconds() != 0:
            raise ChatStoreError("Chat cursor boundary timestamp must be UTC")
        return value.rfc3339()
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _valid_turn_position(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 1 <= value <= _MAX_FIRESTORE_INTEGER
    )


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64encode(decoded) != value:
        raise ValueError("non-canonical base64")
    return decoded
