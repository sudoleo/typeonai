import logging
from firebase_admin import auth, firestore
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
import app.core.config as cfg
from app.core.security import verify_user_token, is_user_pro, db_firestore
from app.core.state import usage_counter, deep_search_usage

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
        
        # 2. Status aus Firestore holen
        pro_status = is_user_pro(uid)

        # 3. Limits basierend auf Status setzen
        limit_regular = cfg.get_usage_limit(pro_status)
        limit_deep = cfg.get_deep_search_limit(pro_status)

        return {
            "uid": uid,
            "is_pro": pro_status,
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
    current_usage = usage_counter.get(uid, 0)
    current_deep_usage = deep_search_usage.get(uid, 0)

    # 4. Verbleibend berechnen (verhindert negative Zahlen in der UI, falls mal überzogen wurde)
    remaining = int(limit_regular - current_usage)
    deep_remaining = int(limit_deep - current_deep_usage)

    return {
        "remaining": remaining,
        "deep_remaining": deep_remaining,
        "is_pro": pro_status,
        "total_limit": limit_regular
    }

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
        
        # 2. Daten vorbereiten
        interest_data = {
            "uid": uid,
            "email": user_email,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "source": source,
            "ip": request.client.host, # IP auch nützlich für Spamschutz
            "user_agent": request.headers.get("user-agent")
        }
        
        # 3. In "pro_waitlist" Collection schreiben (Backend Admin SDK hat immer Schreibrechte)
        db_firestore.collection("pro_waitlist").add(interest_data)
        
        return {"status": "success", "message": "Interest tracked"}

    except Exception as e:
        logging.error(f"Tracking error for token prefix {token[:10]}...: {e}")
        return {"status": "error", "detail": str(e)}
