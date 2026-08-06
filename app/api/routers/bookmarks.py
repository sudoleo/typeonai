import re
import base64
import json
import logging
from firebase_admin import firestore
from fastapi import APIRouter, Request, Body, HTTPException, Query

from app.core import config as cfg
from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, db_firestore
from app.services.llm.attachments import ALLOWED_ATTACHMENT_MIMES, MAX_ATTACHMENTS
from app.services import share_snapshots
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


def _clean_previous_question(value):
    return str(value or "").strip()[:cfg.get_followup_question_char_limit()]


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


def _chat_store():
    return ChatStore(db_firestore)


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
        logging.error("bookmark auth failed: %s", exc)
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
async def load_bookmarks(
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
    except Exception as e:
        logging.error(f"Error loading bookmarks for uid={uid}: {e}")
        raise HTTPException(status_code=500, detail="Error loading bookmarks")


@router.get("/bookmarks/{bookmark_id}")
@limiter.limit("30/minute")
async def load_bookmark_detail(request: Request, bookmark_id: str):
    uid = _bookmark_uid(request)
    if not BOOKMARK_ID_RE.fullmatch(bookmark_id):
        raise HTTPException(status_code=404, detail="Bookmark not found")
    try:
        snap = (
            db_firestore.collection("users").document(uid)
            .collection("bookmarks").document(bookmark_id).get()
        )
    except Exception as exc:
        logging.exception("Error loading bookmark detail for uid=%s", uid)
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

        page = _chat_store().list_turn_details(
            uid, chat_id, cursor=cursor, limit=limit, status="completed"
        )
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
        logging.exception("Error loading bookmark conversation for uid=%s", uid)
        raise HTTPException(status_code=500, detail="Error loading bookmark conversation") from exc


@router.post("/bookmark")
@limiter.limit("20/minute")
async def save_bookmark(request: Request, data: dict = Body(...)):
    id_token     = data.get("id_token")
    question     = data.get("question")
    response_text= data.get("response")
    modelName    = data.get("modelName")
    mode         = data.get("mode")
    sources      = data.get("sources") # <--- NEU: Quellen auslesen
    attachments  = sanitize_attachment_meta(data.get("attachments"))
    previous_question = _clean_previous_question(data.get("previousQuestion"))
    chat_binding = _chat_binding(data)

    if not (id_token and question and response_text and modelName):
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
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
        # speichern (merge)
        doc_ref.set(dataToMerge, merge=True)

        # **Neu:** direkt danach auslesen
        snap = doc_ref.get()
        bm = snap.to_dict()
        bm["id"] = snap.id

        return {
            "status":  "success",
            "message": f"Bookmark for {modelName} saved.",
            "bookmark": bm
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving bookmark")


@router.post("/bookmark/consensus")
@limiter.limit("3/minute")
async def save_bookmark_consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    question = data.get("question")
    consensusText = data.get("consensusText")
    differencesText = data.get("differencesText")
    differencesData = data.get("differencesData")
    sources = data.get("sources")
    result_id = str(data.get("resultId") or "").strip()
    consensus_model = str(data.get("consensusModel") or "").strip()[:80]
    model_labels = data.get("modelLabels")
    previous_question = _clean_previous_question(data.get("previousQuestion"))
    previous_turn = _sanitize_previous_turn(data.get("previousTurn"))
    chat_binding = _chat_binding(data)
    model_responses = _sanitize_model_responses(data.get("modelResponses"))

    if not id_token or not question or consensusText is None or differencesText is None:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

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
    elif result_id and share_snapshots.pending_result_is_available(
        uid, result_id, db=db_firestore
    ):
        dataToMerge["share_result_id"] = result_id
    if consensus_model:
        dataToMerge["consensus_model"] = consensus_model
    clean_labels = share_snapshots.sanitize_model_labels(
        model_labels, share_snapshots.PROVIDER_ORDER
    )
    if clean_labels:
        dataToMerge["model_labels"] = clean_labels
    
    try:
        doc_ref = (
            db_firestore
            .collection("users")
            .document(uid)
            .collection("bookmarks")
            .document(doc_id)
        )
        doc_ref.set(dataToMerge, merge=True)
        snap = doc_ref.get()
        bookmark = snap.to_dict()
        bookmark["id"] = snap.id
        return {
            "status": "success",
            "message": "Consensus and differences saved.",
            "bookmark": bookmark,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving consensus")


@router.post("/bookmark/consensus/share-result")
@limiter.limit("10/minute")
async def prepare_bookmark_share_result(request: Request, data: dict = Body(...)):
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
    included_providers = [
        provider for provider in share_snapshots.PROVIDER_ORDER
        if str(responses.get(provider) or "").strip() or provider in compared
    ]
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
    try:
        result_id = share_snapshots.save_pending_result(payload, db=db_firestore)
        doc_ref.set({"share_result_id": result_id}, merge=True)
    except Exception as exc:
        logging.exception("prepare_bookmark_share_result failed")
        raise HTTPException(status_code=500, detail="Could not prepare bookmark for sharing.") from exc
    return {"status": "success", "result_id": result_id, "created": True}


@router.delete("/bookmark")
async def delete_bookmark(data: dict):
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
        snap = ref.get()
        chat_id = str((snap.to_dict() or {}).get("chat_id") or "") if snap.exists else ""
        ref.delete()
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
        except Exception:
            logging.exception(
                "Bookmark deleted but chat cascade failed for uid=%s", uid
            )

    return {"status": "success", "message": "Bookmark deleted."}
