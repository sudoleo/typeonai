import logging
import firebase_admin
from firebase_admin import auth
from fastapi import APIRouter, Request, Body, HTTPException

from app.core.rate_limit import limiter
from app.core.state import registered_ips
from app.core.security import verify_user_token

router = APIRouter()

@router.post("/register")
@limiter.limit("3/minute")
async def register_user(request: Request, data: dict = Body(...)):    
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password must be provided.")

    try:
        # Überprüfe, ob die E-Mail bereits existiert
        try:
            existing_user = auth.get_user_by_email(email)
            # Falls kein Fehler auftritt, existiert der Nutzer bereits
            raise HTTPException(status_code=400, detail="This email is already registered.")
        except firebase_admin.auth.UserNotFoundError:
            # Keine Registrierung mit dieser E-Mail gefunden, also weiter
            pass

        user = auth.create_user(email=email, password=password)
        custom_token = auth.create_custom_token(user.uid)
        custom_token_str = custom_token.decode("utf-8")
        return {"uid": user.uid, "email": user.email, "customToken": custom_token_str}

    except HTTPException:
        # bereits bewusst gesetzte Meldungen durchreichen
        raise
    except Exception as e:
        # Keine E-Mail-Adresse in die Server-Logs schreiben (Datenminimierung)
        logging.error(f"/register failed: {e}")
        # generische Meldung an den Client
        raise HTTPException(status_code=400, detail="Registration failed. Please try again later.")
    

@router.post("/confirm-registration")
async def confirm_registration(request: Request, data: dict = Body(...)):
    token = data.get("id_token")
    if not token:
        raise HTTPException(status_code=400, detail="Authentication failed")

    try:
        uid = verify_user_token(token, allow_unverified=True)
        user = auth.get_user(uid)
    except Exception as e:
        logging.error(f"/confirm-registration token error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    if not user.email_verified:
        # Diese Info ist okay, weil sie nichts über Passwort / Existenz aussagt
        raise HTTPException(status_code=400, detail="E-mail address not yet verified.")

    ip_address = request.client.host

    if ip_address in registered_ips and registered_ips[ip_address] != uid:
        raise HTTPException(status_code=400, detail="Only one confirmed account per user/IP is allowed.")

    registered_ips[ip_address] = uid
    return {"status": "registered", "ip": ip_address}
