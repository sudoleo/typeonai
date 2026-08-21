from __future__ import annotations

import re
import base64
import json
import logging
import unicodedata
from typing import Literal
from firebase_admin import firestore
from fastapi import APIRouter, Request, Body, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from app.core import config as cfg
from app.core.observability import safe_exception
from app.core.rate_limit import (
    ApiUidRateLimitExceeded,
    api_uid_limiter,
    limiter,
)
from app.core.security import verify_user_token, extract_id_token, db_firestore
from app.services.llm.attachments import ALLOWED_ATTACHMENT_MIMES, MAX_ATTACHMENTS
from app.services import persistence_guard, share_snapshots
from app.services.chat_store import (
    TURN_PAGE_SIZE,
    TURN_PAGE_SIZE_MAX,
    ChatCursorUnavailable,
    ChatNotFound,
    ChatStore,
    InvalidChatCursor,
    derive_title,
)
from app.services.share_snapshots import sanitize_differences_data

router = APIRouter()
BOOKMARK_PAGE_SIZE = 35
BOOKMARK_PAGE_SIZE_MAX = 50
BOOKMARK_ID_RE = re.compile(r"[A-Za-z0-9_]{1,100}")
CHAT_ID_RE = re.compile(r"[0-9a-f]{32}")

# Bookmark-Persistenz ist ein interner Fan-out des Produkt-Flows: Ein Lauf
# schreibt je nach Preset bis zu sechs Modell-Snapshots plus einen
# autoritativen Consensus-Snapshot. Das vorgelagerte slowapi-Limit bleibt als
# grosszuegiger Schutz gegen unautorisierte Request-Fluten bestehen; nach der
# Token-Pruefung begrenzt dieses Budget den tatsaechlichen Kontoinhaber. So
# sperren weder ein gemeinsamer Proxy/NAT noch andere Nutzer einen laufenden
# Daily-, Balanced- oder High-Quality-Save mitten im Snapshot.
BOOKMARK_UID_RATE_LIMITS = {
    "model_save": 120,
    "consensus_save": 60,
}


class BookmarkModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id_token: StrictStr = Field(min_length=1, max_length=20_000)
    question: StrictStr = Field(min_length=1, max_length=8_000)
    response: StrictStr = Field(min_length=1, max_length=40_000)
    modelName: Literal["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]
    mode: Literal["Standard", "Deep Think"]
    bookmarkId: StrictStr | None = Field(default=None, max_length=100)
    previousQuestion: StrictStr = Field(default="", max_length=4_000)
    chatId: StrictStr | None = Field(default=None, max_length=32)
    turnId: StrictStr | None = Field(default=None, max_length=32)
    sources: list[dict] | None = Field(default=None, max_length=24)
    attachments: list[dict] | None = Field(default=None, max_length=10)


class BookmarkConsensusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id_token: StrictStr | None = Field(default=None, max_length=20_000)
    question: StrictStr = Field(min_length=1, max_length=8_000)
    consensusText: StrictStr = Field(max_length=100_000)
    differencesText: StrictStr = Field(max_length=50_000)
    differencesData: dict | None = None
    sources: list[dict] | None = Field(default=None, max_length=24)
    resultId: StrictStr | None = Field(default=None, max_length=64)
    consensusModel: StrictStr | None = Field(default=None, max_length=80)
    modelLabels: dict[str, StrictStr] | None = None
    modelResponses: dict[str, StrictStr] | None = None
    bookmarkId: StrictStr | None = Field(default=None, max_length=100)
    chatId: StrictStr | None = Field(default=None, max_length=32)
    turnId: StrictStr | None = Field(default=None, max_length=32)
    previousQuestion: StrictStr = Field(default="", max_length=4_000)
    previousTurn: dict | None = None


class BookmarkDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id_token: StrictStr = Field(min_length=1, max_length=20_000)
    bookmarkId: StrictStr = Field(min_length=1, max_length=100)


def _clean_previous_question(value):
    return str(value or "").strip()[:cfg.get_followup_question_char_limit()]


def _enforce_bookmark_uid_rate_limit(uid: str, operation: str) -> None:
    limit = BOOKMARK_UID_RATE_LIMITS[operation]
    try:
        api_uid_limiter.check(uid, f"bookmark:{operation}", limit)
    except ApiUidRateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Too many bookmark saves. Please slow down.",
        ) from exc


def _bookmark_display_question(data):
    return str(data.get("query") or "").strip()


def _bookmark_title(data):
    """The name shown in the sidebar: the conversation's opening question.

    `query` is the LATEST question of the bookmark and has to stay that way —
    share, restore and follow-up context all read it. Older bookmarks have no
    stored title yet, so the latest question remains the fallback name.
    """
    return str(data.get("title") or "").strip() or _bookmark_display_question(data)


def _stable_bookmark_title(uid, question, previous_question, chat_binding):
    """Return the title to merge, or "" when the existing one must survive.

    A follow-up must never rename its bookmark, so only the opening turn
    writes the title. A chat-bound follow-up is the one exception: the chat
    document already keeps the first question, which repairs bookmarks that
    were written before titles existed.
    """
    if not previous_question:
        return derive_title(question)
    if not chat_binding:
        return ""
    try:
        chat = _chat_store().get_chat(uid, chat_binding["chat_id"])
    except Exception:
        return ""
    return str(chat.get("title") or "").strip()


def _bookmark_document_id(question, requested_id=""):
    requested_id = str(requested_id or "").strip()
    if requested_id:
        if not BOOKMARK_ID_RE.fullmatch(requested_id):
            raise HTTPException(status_code=400, detail="Invalid bookmark id.")
        return requested_id
    raw_id = base64.b64encode(str(question).encode()).decode()
    return re.sub(r"[^a-zA-Z0-9]", "_", raw_id)[:50]


def _chat_binding(data):
    chat_id = str(data.get("chatId") or "").strip()
    turn_id = str(data.get("turnId") or "").strip()
    if not chat_id and not turn_id:
        return None
    if not CHAT_ID_RE.fullmatch(chat_id) or not CHAT_ID_RE.fullmatch(turn_id):
        raise HTTPException(status_code=400, detail="Invalid chat bookmark binding.")
    return {"chat_id": chat_id, "turn_id": turn_id}


def _sanitize_model_responses(raw):
    if not isinstance(raw, dict):
        return None
    limit = cfg.get_consensus_answer_char_limit()
    return {
        provider: str(raw.get(provider) or "").strip()[:limit]
        for provider in share_snapshots.PROVIDER_ORDER
    }


def _model_provenance(included_models) -> tuple[list[str], dict[str, str]]:
    """Recover canonical providers/labels from a server-owned model list."""
    providers = []
    labels = {}
    for item in included_models if isinstance(included_models, list) else []:
        display, separator, label = str(item).partition(":")
        display = display.strip()
        provider = next(
            (
                name for name in share_snapshots.PROVIDER_ORDER
                if display in {name, share_snapshots.PROVIDER_CITATION_LABELS[name]}
            ),
            None,
        )
        if not provider or provider in providers:
            continue
        providers.append(provider)
        if separator and label.strip():
            labels[provider] = label.strip()
    return providers, labels


def _authoritative_consensus_payload(uid: str, result_id: str, chat_binding: dict | None):
    """Materialize consensus fields from an owner-bound completed server run."""
    if result_id:
        pending = share_snapshots.get_pending_result(uid, result_id, db=db_firestore)
        if pending:
            return {
                "question": str(pending.get("question") or ""),
                "consensus": str(pending.get("consensus_md") or ""),
                "differences": str(pending.get("differences_text") or ""),
                "differences_data": pending.get("differences_data"),
                "sources": pending.get("sources"),
                "included_models": pending.get("included_models"),
                "consensus_model": str(pending.get("consensus_model") or ""),
                "model_responses": pending.get("model_responses"),
                "vote_subject_id": str(pending.get("vote_subject_id") or result_id),
                "result_id": result_id,
            }
    if chat_binding:
        try:
            turn = _chat_store().get_turn(
                uid, chat_binding["chat_id"], chat_binding["turn_id"]
            )
        except ChatNotFound:
            turn = None
        if turn and turn.get("status") == "completed":
            answers = turn.get("model_answers")
            clean_answers = {}
            model_labels = {}
            if isinstance(answers, dict):
                clean_answers = {
                    provider: (
                        str(value.get("answer") or "")
                        if isinstance(value, dict) else str(value or "")
                    )
                    for provider, value in answers.items()
                }
                model_labels = {
                    provider: str(value.get("model_label") or "")
                    for provider, value in answers.items()
                    if isinstance(value, dict) and value.get("model_label")
                }
            return {
                "question": str(turn.get("question") or ""),
                "consensus": str(turn.get("consensus") or ""),
                "differences": str(turn.get("differences") or ""),
                "differences_data": turn.get("differences_data"),
                "sources": turn.get("sources"),
                "model_responses": clean_answers,
                "included_models": turn.get("included_models"),
                "model_labels": model_labels,
                "consensus_model": str(turn.get("consensus_model") or ""),
                "result_id": str(turn.get("result_id") or ""),
            }
    raise HTTPException(
        status_code=409,
        detail="Consensus bookmarks require an owned completed run.",
    )


def _chat_store():
    return ChatStore(db_firestore)


def _same_question(stored, requested) -> bool:
    """Compare the run's question with the one the browser sends to save it.

    A chat turn stores the NFKC-normalized question (``chat_store``), while the
    pending share result keeps the raw text. Comparing the two forms literally
    rejected perfectly valid saves -- one non-breaking space in a follow-up was
    enough to make the answer appear while its bookmark failed.
    """
    def normalized(value):
        return unicodedata.normalize("NFKC", str(value or "")).strip()

    return normalized(stored) == normalized(requested)


def _sanitize_previous_turn(raw):
    if not isinstance(raw, dict):
        return None
    question = _clean_previous_question(raw.get("question"))
    consensus = str(raw.get("consensus") or "").strip()[:cfg.get_followup_consensus_char_limit()]
    if not question or not consensus:
        return None
    turn = {
        "question": question,
        "consensus": consensus,
        "differences": str(raw.get("differences") or "").strip()[:50_000],
        "sources": share_snapshots.sanitize_sources(raw.get("sources")),
    }
    differences_data = share_snapshots.sanitize_differences_data(raw.get("differences_data"))
    if differences_data is not None:
        turn["differences_data"] = differences_data
    return turn


def _bookmark_uid(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication failed")
    try:
        return verify_user_token(auth_header.split(" ", 1)[1])
    except Exception as exc:
        logging.error("bookmark auth failed category=%s", safe_exception(exc))
        raise HTTPException(status_code=401, detail="Authentication failed") from exc


def _bookmark_meta(bookmark_id, data):
    responses = data.get("responses") if isinstance(data.get("responses"), dict) else {}
    return {
        "id": str(bookmark_id),
        "query": _bookmark_display_question(data),
        "title": _bookmark_title(data),
        "mode": str(data.get("mode") or ""),
        "timestamp": data.get("timestamp"),
        "has_consensus": bool(str(responses.get("consensus") or "").strip()),
        "model_count": sum(
            1 for key, value in responses.items()
            if key not in {"consensus", "differences", "differences_data"}
            and str(value or "").strip()
        ),
        "source_count": len(data.get("sources") or []) if isinstance(data.get("sources"), list) else 0,
        "attachment_count": len(data.get("attachments") or []) if isinstance(data.get("attachments"), list) else 0,
    }


def _encode_bookmark_cursor(bookmark_id):
    raw = json.dumps({"id": str(bookmark_id)}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_bookmark_cursor(cursor):
    try:
        raw = base64.urlsafe_b64decode(str(cursor) + "=" * (-len(str(cursor)) % 4))
        bookmark_id = str(json.loads(raw.decode("utf-8")).get("id") or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid bookmark cursor") from exc
    if not BOOKMARK_ID_RE.fullmatch(bookmark_id):
        raise HTTPException(status_code=400, detail="Invalid bookmark cursor")
    return bookmark_id


def sanitize_attachment_meta(raw):
    """Reduziert Attachment-Angaben auf reine Metadaten (Name/Typ/Größe).

    Dateidaten werden bewusst verworfen – in Firestore landen nie Datei-Bytes
    (Dokument-Limit 1 MiB, Kosten). Gibt None zurück, wenn das Feld fehlt,
    damit bestehende Bookmarks beim Merge unangetastet bleiben.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []

    sanitized = []
    for item in raw:
        if len(sanitized) >= MAX_ATTACHMENTS:
            break
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:200]
        mime = str(item.get("mime") or "")
        if not name or mime not in ALLOWED_ATTACHMENT_MIMES:
            continue
        try:
            size = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError):
            size = 0
        sanitized.append({"name": name, "mime": mime, "size": size})
    return sanitized

@router.get("/bookmarks")
@limiter.limit("20/minute")
def load_bookmarks(
    request: Request,
    cursor: str = Query(default="", max_length=256),
    limit: int = Query(default=BOOKMARK_PAGE_SIZE, ge=1, le=BOOKMARK_PAGE_SIZE_MAX),
):
    uid = _bookmark_uid(request)
    try:
        bookmarks_ref = db_firestore.collection("users").document(uid).collection("bookmarks")
        query_ref = bookmarks_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
        if cursor:
            cursor_id = _decode_bookmark_cursor(cursor)
            cursor_snapshot = bookmarks_ref.document(cursor_id).get()
            if not cursor_snapshot.exists:
                raise HTTPException(status_code=400, detail="Bookmark cursor expired")
            query_ref = query_ref.start_after(cursor_snapshot)
        docs = list(query_ref.limit(limit + 1).stream())
        has_more = len(docs) > limit
        page = docs[:limit]
        bookmarks = [_bookmark_meta(doc.id, doc.to_dict() or {}) for doc in page]
        next_cursor = _encode_bookmark_cursor(page[-1].id) if has_more and page else None
        return {
            "status": "success",
            "bookmarks": bookmarks,
            "next_cursor": next_cursor,
            "has_more": has_more,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("Error loading bookmarks category=%s", safe_exception(exc))
        raise HTTPException(status_code=500, detail="Error loading bookmarks")


@router.get("/bookmarks/{bookmark_id}")
@limiter.limit("30/minute")
def load_bookmark_detail(request: Request, bookmark_id: str):
    uid = _bookmark_uid(request)
    if not BOOKMARK_ID_RE.fullmatch(bookmark_id):
        raise HTTPException(status_code=404, detail="Bookmark not found")
    try:
        snap = (
            db_firestore.collection("users").document(uid)
            .collection("bookmarks").document(bookmark_id).get()
        )
    except Exception as exc:
        logging.error(
            "Error loading bookmark detail category=%s", safe_exception(exc)
        )
        raise HTTPException(status_code=500, detail="Error loading bookmark") from exc
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    bookmark = snap.to_dict() or {}
    bookmark["id"] = snap.id
    return {"status": "success", "bookmark": bookmark}


@router.get("/bookmarks/{bookmark_id}/conversation")
@limiter.limit("30/minute")
# Deliberately sync so FastAPI runs it in the threadpool: one page walks the
# chat plus every turn's model answers with blocking Firestore calls, which as
# `async def` would sit ON the event loop and stall every other request.
def load_bookmark_conversation(
    request: Request,
    bookmark_id: str,
    cursor: str = Query(default="", max_length=512),
    limit: int = Query(default=TURN_PAGE_SIZE, ge=1, le=TURN_PAGE_SIZE_MAX),
):
    """Return one owner-bound page of complete turns for a chat bookmark."""
    uid = _bookmark_uid(request)
    if not BOOKMARK_ID_RE.fullmatch(bookmark_id):
        raise HTTPException(status_code=404, detail="Bookmark not found")
    try:
        snap = (
            db_firestore.collection("users").document(uid)
            .collection("bookmarks").document(bookmark_id).get()
        )
        if not snap.exists:
            raise HTTPException(status_code=404, detail="Bookmark not found")
        bookmark = snap.to_dict() or {}
        chat_id = str(bookmark.get("chat_id") or "")
        if not CHAT_ID_RE.fullmatch(chat_id):
            return {
                "status": "success",
                "chat_id": None,
                "turns": [],
                "next_cursor": None,
                "has_more": False,
            }

        store = _chat_store()
        try:
            page = store.list_turn_details(
                uid, chat_id, cursor=cursor, limit=limit, status="completed"
            )
        except (ChatNotFound, InvalidChatCursor, ChatCursorUnavailable):
            raise
        except Exception as exc:
            # The compact path is an optimisation, not a correctness gate. If
            # a Firestore/runtime incompatibility breaks the collection reads,
            # fall back to the older owner-bound detail path so the browser
            # never silently collapses a full chat to two bookmark snapshots.
            logging.warning(
                "Optimized bookmark transcript read failed; using detail fallback "
                "category=%s",
                safe_exception(exc),
            )
            metadata_page = store.list_turns(
                uid, chat_id, cursor=cursor, limit=limit
            )
            page = {
                "turns": [
                    store.get_turn(uid, chat_id, turn["id"])
                    for turn in metadata_page["turns"]
                    if turn.get("status") == "completed"
                ],
                "next_cursor": metadata_page.get("next_cursor"),
                "has_more": metadata_page.get("has_more") is True,
            }
        return {
            "status": "success",
            "chat_id": chat_id,
            "turns": page["turns"],
            "next_cursor": page.get("next_cursor"),
            "has_more": page.get("has_more") is True,
        }
    except HTTPException:
        raise
    except ChatNotFound:
        raise HTTPException(status_code=404, detail="Chat not found")
    except InvalidChatCursor as exc:
        raise HTTPException(status_code=400, detail="Invalid chat cursor") from exc
    except ChatCursorUnavailable as exc:
        # Missing cursor-signing configuration is a server problem. Reporting it
        # as 400 made a deployment error look like a bad client request.
        logging.error("bookmark conversation failed: chat cursor signing is not configured")
        raise HTTPException(
            status_code=503, detail="Conversation pagination unavailable"
        ) from exc
    except Exception as exc:
        logging.error(
            "Error loading bookmark conversation category=%s", safe_exception(exc)
        )
        raise HTTPException(status_code=500, detail="Error loading bookmark conversation") from exc


@router.post("/bookmark")
@limiter.limit("300/minute")
def save_bookmark(request: Request, payload: BookmarkModelRequest):
    data = payload.model_dump()
    id_token     = data.get("id_token")
    question     = data.get("question")
    response_text= data.get("response")
    modelName    = data.get("modelName")
    mode         = data.get("mode")
    sources      = share_snapshots.sanitize_sources(data.get("sources"))
    attachments  = sanitize_attachment_meta(data.get("attachments"))
    previous_question = _clean_previous_question(data.get("previousQuestion"))
    chat_binding = _chat_binding(data)

    if not (id_token and question and response_text and modelName):
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    _enforce_bookmark_uid_rate_limit(uid, "model_save")
    
    doc_id = _bookmark_document_id(question, data.get("bookmarkId"))
    
    dataToMerge = {
        "query": question,
        "previous_question": previous_question,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "mode": mode,
        "responses": { modelName: response_text }
    }
    title = _stable_bookmark_title(uid, question, previous_question, chat_binding)
    if title:
        dataToMerge["title"] = title
    if not previous_question:
        # A regular run reusing the same document id must not inherit a stale
        # archived turn from an older follow-up bookmark.
        dataToMerge["previous_turn"] = {}
    if chat_binding:
        dataToMerge.update(chat_binding)

    # <--- NEU: Quellen hinzufügen, falls vorhanden
    if sources is not None:
        dataToMerge["sources"] = sources

    # Anhänge: nur Metadaten (Name/Typ/Größe), nie Dateidaten
    if attachments is not None:
        dataToMerge["attachments"] = attachments
    
    try:
        # Speichern (merge)
        doc_ref = (
            db_firestore
            .collection("users")
            .document(uid)
            .collection("bookmarks")
            .document(doc_id)
        )
        bm = persistence_guard.write_bookmark(
            uid=uid, doc_ref=doc_ref, patch=dataToMerge, db=db_firestore
        )
        bm["id"] = doc_id

        return {
            "status":  "success",
            "message": f"Bookmark for {modelName} saved.",
            "bookmark": bm
        }
        
    except persistence_guard.PersistenceLimitError as exc:
        status = 413 if exc.code in {"bookmark_too_large", "bookmark_storage_limit"} else 429
        raise HTTPException(status_code=status, detail=exc.message) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving bookmark")


@router.post("/bookmark/consensus")
@limiter.limit("120/minute")
def save_bookmark_consensus(request: Request, payload: BookmarkConsensusRequest):
    data = payload.model_dump()
    id_token = extract_id_token(request, data)
    question = data.get("question")
    result_id = str(data.get("resultId") or "").strip()
    previous_question = _clean_previous_question(data.get("previousQuestion"))
    previous_turn = _sanitize_previous_turn(data.get("previousTurn"))
    chat_binding = _chat_binding(data)
    # Provider answers and provenance are server-owned. Client copies are kept
    # in the request schema only for cached clients and are never persisted.
    model_responses = None

    if not id_token or not question:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    _enforce_bookmark_uid_rate_limit(uid, "consensus_save")

    authoritative = _authoritative_consensus_payload(uid, result_id, chat_binding)
    if not _same_question(authoritative["question"], question):
        raise HTTPException(status_code=409, detail="Bookmark question does not match the completed run.")
    consensusText = authoritative["consensus"][:100_000]
    differencesText = authoritative["differences"][:50_000]
    differencesData = authoritative.get("differences_data")
    sources = share_snapshots.sanitize_sources(authoritative.get("sources"))
    model_responses = _sanitize_model_responses(
        authoritative.get("model_responses") or {}
    )
    included_providers, authoritative_labels = _model_provenance(
        authoritative.get("included_models")
    )
    consensus_model = str(authoritative.get("consensus_model") or "").strip()[:80]
    authoritative_labels.update(authoritative.get("model_labels") or {})
    result_id = authoritative.get("result_id") or result_id

    doc_id = _bookmark_document_id(question, data.get("bookmarkId"))

    dataToMerge = {
        "query": question,
        "previous_question": previous_question,
        "responses": {
            "consensus": consensusText,
            "differences": differencesText
        }
    }
    title = _stable_bookmark_title(uid, question, previous_question, chat_binding)
    if title:
        dataToMerge["title"] = title
    if model_responses is not None:
        # Every provider key is present, including empty strings. That clears
        # stale answers when a later turn in the same chat used fewer models.
        dataToMerge["responses"].update(model_responses)
    if chat_binding:
        dataToMerge.update(chat_binding)
    if previous_turn is not None:
        dataToMerge["previous_turn"] = previous_turn
    else:
        dataToMerge["previous_turn"] = {}

    # Strukturierte Differences whitelisten/kappen (gleiche Validierung wie beim
    # Share-Snapshot) und mitspeichern, damit das Bookmark Verdict, Karten und
    # Modellvergleiche wie eine echte Query rendern kann.
    sanitized_diff_data = sanitize_differences_data(differencesData)
    if sanitized_diff_data is not None:
        dataToMerge["responses"]["differences_data"] = sanitized_diff_data

    if sources is not None:
        dataToMerge["sources"] = sources

    if previous_question:
        # Das Live-Pending-Result des Follow-ups kennt nur die aktuelle Frage.
        # Ein Bookmark-Share muss spaeter serverseitig neu aufgebaut werden und
        # darf keinen alten Result-Verweis wiederverwenden.
        dataToMerge["share_result_id"] = ""
    elif result_id:
        dataToMerge["share_result_id"] = result_id
    dataToMerge["vote_subject_id"] = str(
        authoritative.get("vote_subject_id") or result_id or ""
    )
    dataToMerge["included_providers"] = included_providers
    dataToMerge["consensus_model"] = consensus_model
    clean_labels = share_snapshots.sanitize_model_labels(
        authoritative_labels, included_providers
    )
    dataToMerge["model_labels"] = clean_labels
    
    try:
        doc_ref = (
            db_firestore
            .collection("users")
            .document(uid)
            .collection("bookmarks")
            .document(doc_id)
        )
        bookmark = persistence_guard.write_bookmark(
            uid=uid, doc_ref=doc_ref, patch=dataToMerge, db=db_firestore
        )
        bookmark["id"] = doc_id
        return {
            "status": "success",
            "message": "Consensus and differences saved.",
            "bookmark": bookmark,
        }
    except persistence_guard.PersistenceLimitError as exc:
        status = 413 if exc.code in {"bookmark_too_large", "bookmark_storage_limit"} else 429
        raise HTTPException(status_code=status, detail=exc.message) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving consensus")


@router.post("/bookmark/consensus/share-result")
@limiter.limit("10/minute")
def prepare_bookmark_share_result(request: Request, data: dict = Body(...)):
    """Create or reuse a share/watch pending result for an owned bookmark."""
    id_token = extract_id_token(request, data)
    bookmark_id = str(data.get("bookmarkId") or "").strip()
    if not id_token or not re.fullmatch(r"[A-Za-z0-9_]{1,100}", bookmark_id):
        raise HTTPException(status_code=400, detail="Missing or invalid bookmark id.")
    try:
        uid = verify_user_token(id_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Authentication failed") from exc

    doc_ref = (
        db_firestore.collection("users").document(uid)
        .collection("bookmarks").document(bookmark_id)
    )
    snap = doc_ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Bookmark not found.")
    bookmark = snap.to_dict() or {}
    responses = bookmark.get("responses")
    responses = responses if isinstance(responses, dict) else {}
    consensus_text = str(responses.get("consensus") or "").strip()
    question = _bookmark_display_question(bookmark)
    if not question or not consensus_text:
        raise HTTPException(status_code=400, detail="This bookmark has no consensus result.")

    existing_id = str(bookmark.get("share_result_id") or "").strip()
    if share_snapshots.pending_result_is_available(uid, existing_id, db=db_firestore):
        return {"status": "success", "result_id": existing_id, "created": False}

    compared = responses.get("differences_data")
    compared = compared.get("models_compared") if isinstance(compared, dict) else []
    compared = set(compared) if isinstance(compared, list) else set()
    stored_providers = bookmark.get("included_providers")
    included_providers = (
        [
            provider for provider in share_snapshots.PROVIDER_ORDER
            if isinstance(stored_providers, list) and provider in stored_providers
        ]
        if isinstance(stored_providers, list)
        else [
            provider for provider in share_snapshots.PROVIDER_ORDER
            if str(responses.get(provider) or "").strip() or provider in compared
        ]
    )
    payload = share_snapshots.build_pending_result(
        uid=uid,
        question=question,
        consensus_md=consensus_text,
        differences_data=responses.get("differences_data"),
        differences_text=responses.get("differences") or "",
        model_sources=bookmark.get("sources") or [],
        included_providers=included_providers,
        model_labels=bookmark.get("model_labels"),
        consensus_model=bookmark.get("consensus_model") or "",
    )
    if payload is None:
        raise HTTPException(status_code=400, detail="This bookmark cannot be shared.")
    vote_subject_id = str(
        bookmark.get("vote_subject_id") or existing_id or "bookmark:" + bookmark_id
    )
    payload["vote_subject_id"] = vote_subject_id
    try:
        result_id = share_snapshots.save_pending_result(payload, db=db_firestore)
        persistence_guard.write_bookmark(
            uid=uid,
            doc_ref=doc_ref,
            patch={
                "share_result_id": result_id,
                "vote_subject_id": vote_subject_id,
            },
            db=db_firestore,
        )
    except Exception as exc:
        logging.error(
            "prepare_bookmark_share_result failed category=%s",
            safe_exception(exc),
        )
        raise HTTPException(status_code=500, detail="Could not prepare bookmark for sharing.") from exc
    return {"status": "success", "result_id": result_id, "created": True}


@router.delete("/bookmark")
def delete_bookmark(data: BookmarkDeleteRequest):
    data = data.model_dump()
    id_token = data.get("id_token")
    bookmark_id = data.get("bookmarkId")
    
    if not id_token or not bookmark_id:
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    try:
        ref = (
            db_firestore.collection("users").document(uid)
            .collection("bookmarks").document(bookmark_id)
        )
        # Read the chat binding BEFORE the delete: afterwards the bookmark is
        # the only handle on that chat, and the transcript would be stranded
        # with no way to reach or remove it.
        deleted = persistence_guard.delete_bookmark(
            uid=uid, doc_ref=ref, db=db_firestore
        )
        chat_id = str((deleted or {}).get("chat_id") or "")
    except Exception:
        raise HTTPException(status_code=500, detail="Error deleting bookmark")

    if CHAT_ID_RE.fullmatch(chat_id):
        # Best effort: the bookmark is already gone, so a failing cascade must
        # not turn a successful deletion into an error the user has to retry.
        # A leftover chat stays reachable for the account-level cascade.
        try:
            _chat_store().delete_chat(uid, chat_id)
        except ChatNotFound:
            pass
        except Exception as exc:
            logging.error(
                "Bookmark deleted but chat cascade failed category=%s",
                safe_exception(exc),
            )

    return {"status": "success", "message": "Bookmark deleted."}
