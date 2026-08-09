import asyncio
import logging
import time

import firebase_admin
from firebase_admin import auth
from fastapi import APIRouter, BackgroundTasks, Request, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.rate_limit import limiter
from app.core.security import verify_user_token
from app.services.telegram_notifier import send_new_user_registration_notification

router = APIRouter()
_NEW_USER_WINDOW_MS = 10 * 60 * 1000
_REGISTER_RESPONSE_FLOOR_SECONDS = 0.35


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)


async def _neutral_registration_response(started_at: float) -> dict:
    remaining = _REGISTER_RESPONSE_FLOOR_SECONDS - (time.monotonic() - started_at)
    if remaining > 0:
        await asyncio.sleep(remaining)
    return {"status": "check_inbox"}


def _recent_registration_method(user) -> str:
    """Return the provider only while Firebase still identifies a new account."""
    metadata = getattr(user, "user_metadata", None)
    created_ms = getattr(metadata, "creation_timestamp", None)
    try:
        age_ms = (time.time() * 1000) - float(created_ms)
    except (TypeError, ValueError):
        return ""
    if age_ms < -60_000 or age_ms > _NEW_USER_WINDOW_MS:
        return ""

    provider_ids = {
        str(getattr(item, "provider_id", "") or "").strip()
        for item in (getattr(user, "provider_data", None) or [])
    }
    if "google.com" in provider_ids:
        return "google"
    if "password" in provider_ids:
        return "email/password"
    return "firebase"


@router.post("/register")
@limiter.limit("3/minute")
async def register_user(
    request: Request,
    background_tasks: BackgroundTasks,
    data: RegisterRequest,
):
    started_at = time.monotonic()
    email = data.email.strip().lower()
    password = data.password

    try:
        # Konto-Enumeration vermeiden: eine bereits registrierte Adresse darf
        # NICHT als solche gemeldet werden, sonst kann jeder durchprobieren, wer
        # hier Kunde ist. Beide Faelle liefern dieselbe neutrale Antwort; nur der
        # echte Neuzugang bekommt zusaetzlich ein customToken. Der Client zeigt
        # in beiden Faellen denselben "Check your inbox"-Screen.
        try:
            auth.get_user_by_email(email)
            return await _neutral_registration_response(started_at)
        except firebase_admin.auth.UserNotFoundError:
            # Keine Registrierung mit dieser E-Mail gefunden, also weiter
            pass

        try:
            user = auth.create_user(email=email, password=password)
        except firebase_admin.auth.EmailAlreadyExistsError:
            # Wettlauf zwischen Pruefung und Anlage: ebenfalls neutral bleiben.
            return await _neutral_registration_response(started_at)
        background_tasks.add_task(
            send_new_user_registration_notification,
            "email/password",
            user.uid,
        )
        return await _neutral_registration_response(started_at)

    except HTTPException:
        # bereits bewusst gesetzte Meldungen durchreichen
        raise
    except Exception as e:
        # Keine E-Mail-Adresse in die Server-Logs schreiben (Datenminimierung)
        logging.error(f"/register failed: {e}")
        # generische Meldung an den Client
        raise HTTPException(status_code=503, detail="Registration is temporarily unavailable.")
    

@router.post("/confirm-registration")
async def confirm_registration(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
):
    token = data.get("id_token")
    if not token:
        raise HTTPException(status_code=400, detail="Authentication failed")

    try:
        uid = verify_user_token(
            token,
            allow_unverified=True,
            check_revoked=True,
        )
        user = auth.get_user(uid)
    except Exception as e:
        logging.error(f"/confirm-registration token error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    if not user.email_verified:
        # Diese Info ist okay, weil sie nichts über Passwort / Existenz aussagt
        raise HTTPException(status_code=400, detail="E-mail address not yet verified.")

    registration_method = _recent_registration_method(user)
    if registration_method:
        background_tasks.add_task(
            send_new_user_registration_notification,
            registration_method,
            uid,
        )

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
