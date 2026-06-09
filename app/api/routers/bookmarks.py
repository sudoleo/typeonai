import re
import base64
import logging
from firebase_admin import firestore
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, db_firestore
from app.services.llm.attachments import ALLOWED_ATTACHMENT_MIMES, MAX_ATTACHMENTS

router = APIRouter()


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
async def load_bookmarks(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication failed")

    id_token = auth_header.split(" ")[1]
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        logging.error(f"/bookmarks auth failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    try:
        bookmarks_ref = db_firestore.collection("users").document(uid).collection("bookmarks")
        query_ref = bookmarks_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
        docs = query_ref.stream()
        bookmarks = []
        for doc in docs:
            bookmark_data = doc.to_dict()
            bookmark_data["id"] = doc.id
            bookmarks.append(bookmark_data)
        return {"status": "success", "bookmarks": bookmarks}
    except Exception as e:
        logging.error(f"Error loading bookmarks for uid={uid}: {e}")
        raise HTTPException(status_code=500, detail="Error loading bookmarks")


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
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
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
        raise HTTPException(status_code=500, detail="Error saving bookmark: " + str(e))


@router.post("/bookmark/consensus")
@limiter.limit("3/minute")
async def save_bookmark_consensus(request: Request, data: dict = Body(...)):
    id_token = extract_id_token(request, data)
    question = data.get("question")
    consensusText = data.get("consensusText")
    differencesText = data.get("differencesText")
    sources = data.get("sources")
    
    if not id_token or not question or consensusText is None or differencesText is None:
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    # Berechne Dokument-ID (wie oben)
    doc_id = base64.b64encode(question.encode()).decode()
    doc_id = re.sub(r'[^a-zA-Z0-9]', '_', doc_id)[:50]
    
    dataToMerge = {
        "responses": {
            "consensus": consensusText,
            "differences": differencesText
        }
    }

    if sources is not None:
        dataToMerge["sources"] = sources
    
    try:
        db_firestore.collection("users").document(uid).collection("bookmarks").document(doc_id).set(dataToMerge, merge=True)
        return {"status": "success", "message": "Consensus and differences saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving consensus: " + str(e))


@router.delete("/bookmark")
async def delete_bookmark(data: dict):
    id_token = data.get("id_token")
    bookmark_id = data.get("bookmarkId")
    
    if not id_token or not bookmark_id:
        raise HTTPException(status_code=400, detail="Missing required fields.")
    
    try:
        uid = verify_user_token(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed: " + str(e))
    
    try:
        db_firestore.collection("users").document(uid).collection("bookmarks").document(bookmark_id).delete()
        return {"status": "success", "message": "Bookmark deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting bookmark: " + str(e))
