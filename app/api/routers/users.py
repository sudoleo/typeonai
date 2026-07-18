import logging
from firebase_admin import auth, firestore
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
import app.core.config as cfg
from app.core.security import verify_user_token, extract_id_token, is_user_pro, is_user_early, invalidate_tier_cache, db_firestore
from app.core.state import last_feedback_time
from app.services.usage_repository import (
    FirestoreUsageRepository,
    UsageLimits,
    UsageRunNotFound,
    UsageTransitionError,
)

router = APIRouter()
run_usage_repository = FirestoreUsageRepository(db_firestore)


def _run_limits(is_pro: bool) -> UsageLimits:
    return UsageLimits(
        total=cfg.get_consensus_run_limit(is_pro),
        deep_think=cfg.get_deep_think_run_limit(is_pro),
    )

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
        limit_regular = cfg.get_consensus_run_limit(pro_status)
        limit_deep = cfg.get_deep_think_run_limit(pro_status)

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
    Liefert den persistenten Run-Stand des aktuellen UTC-Tags zurück.
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
    limits = _run_limits(pro_status)

    # 3. Persistenten UTC-Tagesstand abrufen. Ein einzelnes Tagesdokument
    #    enthaelt Total- und Deep-Think-Bucket konsistent zusammen.
    snapshot = run_usage_repository.snapshot(uid, limits)

    return {
        "remaining": snapshot.total.remaining,
        "deep_remaining": snapshot.deep_think.remaining,
        "is_pro": pro_status,
        "total_limit": snapshot.total.limit,
        "deep_total_limit": snapshot.deep_think.limit,
        "reserved": snapshot.total.reserved,
        "consumed": snapshot.total.consumed,
        "utc_date": snapshot.utc_date,
    }


@router.post("/usage/run/release")
@limiter.limit("20/minute")
async def release_usage_run(request: Request, data: dict = Body(...)):
    token = extract_id_token(request, data)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        uid = verify_user_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    key = data.get("usage_run_key")
    if not isinstance(key, str) or not key.strip():
        raise HTTPException(status_code=400, detail="Missing usage_run_key")
    try:
        result = run_usage_repository.release(uid, key.strip())
    except UsageRunNotFound:
        raise HTTPException(status_code=404, detail="Usage run not found") from None
    except UsageTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    return {
        "status": result.status.value,
        "remaining": result.snapshot.total.remaining,
        "deep_remaining": result.snapshot.deep_think.remaining,
        "total_limit": result.snapshot.total.limit,
        "deep_total_limit": result.snapshot.deep_think.limit,
        "utc_date": result.utc_date,
    }

@router.post("/delete_account")
@limiter.limit("3/minute")
async def delete_account(request: Request, data: dict = Body(default={})):
    """
    Löscht den Account vollständig (DSGVO Art. 17): Auth-Account, users-Dokument
    inkl. Bookmarks, Usage-Daten, Einträgen in pro_waitlist und feedback.
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

    # 1. Nutzer-Subcollections loeschen (Subcollections werden nicht automatisch
    #    mit dem Eltern-Dokument entfernt). usage_* ist die persistente Grundlage
    #    fuer den kuenftigen run-basierten Consensus-Endpoint.
    user_ref = db_firestore.collection("users").document(uid)
    for subcollection in ("bookmarks", "counters", "usage_days", "usage_runs"):
        try:
            for doc in user_ref.collection(subcollection).stream():
                doc.reference.delete()
        except Exception as e:
            logging.error(
                f"delete_account: {subcollection} cleanup failed for {uid}: {e}"
            )
            errors.append(subcollection)

    # 2. users-Dokument löschen
    try:
        user_ref.delete()
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

    # 4. Feedback-Cooldown bereinigen (inkl. Tier-Flag-Cache, sonst wuerde ein
    #    geloeschter Pro-Account bis zu 60s weiter als Pro gecacht)
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
