import re
import base64
import logging
from firebase_admin import firestore
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
from app.core.security import verify_user_token, extract_id_token, db_firestore

router = APIRouter()

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
