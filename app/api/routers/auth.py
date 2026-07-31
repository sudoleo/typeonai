import logging
import firebase_admin
from firebase_admin import auth
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.responses import JSONResponse

from app.core.rate_limit import limiter
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
        # Konto-Enumeration vermeiden: eine bereits registrierte Adresse darf
        # NICHT als solche gemeldet werden, sonst kann jeder durchprobieren, wer
        # hier Kunde ist. Beide Faelle liefern dieselbe neutrale Antwort; nur der
        # echte Neuzugang bekommt zusaetzlich ein customToken. Der Client zeigt
        # in beiden Faellen denselben "Check your inbox"-Screen.
        try:
            auth.get_user_by_email(email)
            return {"status": "check_inbox"}
        except firebase_admin.auth.UserNotFoundError:
            # Keine Registrierung mit dieser E-Mail gefunden, also weiter
            pass

        try:
            user = auth.create_user(email=email, password=password)
        except firebase_admin.auth.EmailAlreadyExistsError:
            # Wettlauf zwischen Pruefung und Anlage: ebenfalls neutral bleiben.
            return {"status": "check_inbox"}
        custom_token = auth.create_custom_token(user.uid)
        custom_token_str = custom_token.decode("utf-8")
        return {
            "status": "check_inbox",
            "uid": user.uid,
            "email": user.email,
            "customToken": custom_token_str,
        }

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

    response = JSONResponse({"status": "registered"})
    response.headers["Cache-Control"] = "no-store"
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    response.set_cookie(
        "session",
        token,
        max_age=60 * 60,
        httponly=True,
        secure=request.url.scheme == "https" or forwarded_proto == "https",
        samesite="lax",
        path="/",
    )
    return response


@router.delete("/auth/session")
async def clear_session():
    response = JSONResponse({"status": "signed_out"})
    response.headers["Cache-Control"] = "no-store"
    response.delete_cookie("session", path="/", httponly=True, samesite="lax")
    return response
