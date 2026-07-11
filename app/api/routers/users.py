import logging
from firebase_admin import auth, firestore
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
import app.core.config as cfg
from app.core.security import verify_user_token, extract_id_token, is_user_pro, is_user_early, invalidate_tier_cache, db_firestore
from app.core.state import get_usage_snapshot, reset_usage, last_feedback_time

router = APIRouter()

@router.get("/user_status")
@limiter.limit("20/minute")
async def get_user_status(request: Request):
    """
    Prüft den Status des Nutzers (Free vs. Pro) basierend auf dem ID-Token.
    Wird beim Seiten-Load (checkUserStatusOnLoad) aufgerufen.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]

    try:
        # 1. UID verifizieren
        uid = verify_user_token(token)
        
        # 2. Status aus Firestore holen (Pro schliesst Early-Zugang ein)
        pro_status = is_user_pro(uid)
        early_status = pro_status or is_user_early(uid)

        # 3. Limits basierend auf Status setzen
        limit_regular = cfg.get_usage_limit(pro_status)
        limit_deep = cfg.get_deep_search_limit(pro_status)

        return {
            "uid": uid,
            "is_pro": pro_status,
            "is_early": early_status,
            "limit": limit_regular,
            "deep_limit": limit_deep
        }

    except Exception as e:
        logging.error(f"User status check failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.post("/usage")
@limiter.limit("20/minute")
async def get_usage_post(request: Request):
    """
    Liefert die verbleibenden Anfragen dynamisch zurück.
    Rechnet: (Limit_basierend_auf_Tier) - (Bisherige_Nutzung).
    """
    data = await request.json()
    token = data.get("id_token")
    
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
    
    # 1. Status prüfen
    pro_status = is_user_pro(uid)

    # 2. Limits festlegen
    limit_regular = cfg.get_usage_limit(pro_status)
    limit_deep = cfg.get_deep_search_limit(pro_status)

    # 3. Verbrauch abrufen
    current_usage, current_deep_usage = get_usage_snapshot(uid)

    # 4. Verbleibend berechnen (verhindert negative Zahlen in der UI, falls mal überzogen wurde)
    remaining = int(limit_regular - current_usage)
    deep_remaining = int(limit_deep - current_deep_usage)

    return {
        "remaining": remaining,
        "deep_remaining": deep_remaining,
        "is_pro": pro_status,
        "total_limit": limit_regular,
        "deep_total_limit": limit_deep
    }

@router.post("/delete_account")
@limiter.limit("3/minute")
async def delete_account(request: Request, data: dict = Body(default={})):
    """
    Löscht den Account vollständig (DSGVO Art. 17): Auth-Account, users-Dokument
    inkl. Bookmarks, Einträge in pro_waitlist und feedback sowie In-Memory-Zustand.
    allow_unverified=True, damit auch unbestätigte Accounts gelöscht werden können.
    """
    id_token = extract_id_token(request, data)
    if not id_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        uid = verify_user_token(id_token, allow_unverified=True)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    errors = []

    # 1. Bookmarks-Subcollection löschen (Subcollections werden nicht automatisch
    #    mit dem Eltern-Dokument entfernt)
    try:
        bookmarks_ref = db_firestore.collection("users").document(uid).collection("bookmarks")
        for doc in bookmarks_ref.stream():
            doc.reference.delete()
    except Exception as e:
        logging.error(f"delete_account: bookmarks cleanup failed for {uid}: {e}")
        errors.append("bookmarks")

    # 2. users-Dokument löschen
    try:
        db_firestore.collection("users").document(uid).delete()
    except Exception as e:
        logging.error(f"delete_account: user doc cleanup failed for {uid}: {e}")
        errors.append("profile")

    # 3. Waitlist- und Feedback-Einträge des Nutzers löschen
    for collection_name in ("pro_waitlist", "feedback"):
        try:
            docs = db_firestore.collection(collection_name).where("uid", "==", uid).stream()
            for doc in docs:
                doc.reference.delete()
        except Exception as e:
            logging.error(f"delete_account: {collection_name} cleanup failed for {uid}: {e}")
            errors.append(collection_name)

    # 3b. Öffentliche Share-Links und zwischengespeicherte Konsens-Ergebnisse
    #     löschen (DSGVO-Kaskade, Art. 17) – hart, nicht nur revoked
    for collection_name in ("shares", "pending_results"):
        try:
            docs = db_firestore.collection(collection_name).where("owner_uid", "==", uid).stream()
            for doc in docs:
                if collection_name == "shares":
                    for history_doc in doc.reference.collection("watch_history").stream():
                        history_doc.reference.delete()
                doc.reference.delete()
        except Exception as e:
            logging.error(f"delete_account: {collection_name} cleanup failed for {uid}: {e}")
            errors.append(collection_name)

    try:
        docs = db_firestore.collection("watches").where("owner_uid", "==", uid).stream()
        for doc in docs:
            doc.reference.delete()
    except Exception as e:
        logging.error(f"delete_account: watches cleanup failed for {uid}: {e}")
        errors.append("watches")

    # 4. In-Memory-Zustand bereinigen (inkl. Tier-Flag-Cache, sonst wuerde ein
    #    geloeschter Pro-Account bis zu 60s weiter als Pro gecacht)
    reset_usage(uid)
    last_feedback_time.pop(uid, None)
    invalidate_tier_cache(uid)

    # 5. Auth-Account zuletzt löschen, damit der Nutzer bei Teilfehlern
    #    erneut authentifiziert löschen kann
    try:
        auth.delete_user(uid)
    except Exception as e:
        logging.error(f"delete_account: auth deletion failed for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Account deletion failed. Please try again or contact us.")

    if errors:
        logging.warning(f"delete_account: partial cleanup for {uid}, failed: {errors}")

    return {"status": "deleted"}


@router.post("/track-interest")
@limiter.limit("5/minute")
async def track_interest(request: Request, data: dict = Body(...)):
    """
    Speichert das Interesse an der Pro-Version in der DB.
    """
    token = data.get("id_token")
    source = data.get("source", "unknown")
    
    if not token:
         raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        # 1. User verifizieren (deine existierende Funktion nutzen)
        uid = verify_user_token(token)
        user_email = auth.get_user(uid).email
        
        # 2. Daten vorbereiten (Datenminimierung: keine IP / kein User-Agent,
        #    Spam-Schutz übernehmen Rate-Limit und Auth-Pflicht)
        interest_data = {
            "uid": uid,
            "email": user_email,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "source": source
        }
        
        # 3. In "pro_waitlist" Collection schreiben (Backend Admin SDK hat immer Schreibrechte)
        db_firestore.collection("pro_waitlist").add(interest_data)
        
        return {"status": "success", "message": "Interest tracked"}

    except Exception as e:
        logging.error(f"Tracking error for token prefix {token[:10]}...: {e}")
        return {"status": "error", "detail": "Could not track interest. Please try again later."}
