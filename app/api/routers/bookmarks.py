import re
import base64
import json
import logging
from firebase_admin import firestore
from fastapi import APIRouter, Request, Body, HTTPException, Query

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, db_firestore
from app.services.llm.attachments import ALLOWED_ATTACHMENT_MIMES, MAX_ATTACHMENTS
from app.services import share_snapshots
from app.services.share_snapshots import sanitize_differences_data

router = APIRouter()
BOOKMARK_PAGE_SIZE = 30
BOOKMARK_PAGE_SIZE_MAX = 50
BOOKMARK_ID_RE = re.compile(r"[A-Za-z0-9_]{1,100}")


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
        "query": str(data.get("query") or ""),
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

    if not (id_token and question and response_text and modelName):
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    # Berechne die Dokument-ID wie gehabt
    raw_id = base64.b64encode(question.encode()).decode()
    doc_id = re.sub(r'[^a-zA-Z0-9]', '_', raw_id)[:50]
    
    dataToMerge = {
        "query": question,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "mode": mode,
        "responses": { modelName: response_text }
    }

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

    if not id_token or not question or consensusText is None or differencesText is None:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Berechne Dokument-ID (wie oben)
    doc_id = base64.b64encode(question.encode()).decode()
    doc_id = re.sub(r'[^a-zA-Z0-9]', '_', doc_id)[:50]

    dataToMerge = {
        "responses": {
            "consensus": consensusText,
            "differences": differencesText
        }
    }

    # Strukturierte Differences whitelisten/kappen (gleiche Validierung wie beim
    # Share-Snapshot) und mitspeichern, damit das Bookmark Verdict, Karten und
    # Modellvergleiche wie eine echte Query rendern kann.
    sanitized_diff_data = sanitize_differences_data(differencesData)
    if sanitized_diff_data is not None:
        dataToMerge["responses"]["differences_data"] = sanitized_diff_data

    if sources is not None:
        dataToMerge["sources"] = sources

    if result_id and share_snapshots.pending_result_is_available(
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
    question = str(bookmark.get("query") or "").strip()
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
        db_firestore.collection("users").document(uid).collection("bookmarks").document(bookmark_id).delete()
        return {"status": "success", "message": "Bookmark deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting bookmark")
