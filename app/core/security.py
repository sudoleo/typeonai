import logging
from typing import Optional
from fastapi import Request
import firebase_admin
from firebase_admin import credentials, auth, firestore

class CustomSecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                csp = (
                    "default-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.jsdelivr.net https://www.gstatic.com; "
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com https://apis.google.com https://accounts.google.com; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    "img-src 'self' data: https://lh3.googleusercontent.com https:; "
                    "connect-src 'self' "
                    "https://firestore.googleapis.com "
                    "https://*.firebaseio.com "
                    "https://identitytoolkit.googleapis.com "
                    "https://securetoken.googleapis.com "
                    "https://firebaseinstallations.googleapis.com "
                    "https://content-firebaseappcheck.googleapis.com "
                    "https://www.gstatic.com "
                    "https://*.gstatic.com "
                    "https://apis.google.com "
                    "https://accounts.google.com "
                    "https://www.googleapis.com "
                    "https://*.googleapis.com "
                    "https://firebasestorage.googleapis.com "
                    "https://api.openai.com https://api.mistral.ai https://api.anthropic.com "
                    "https://api.x.ai https://api.deepseek.com https://api.exa.ai "
                    "https://cdn.jsdelivr.net; "
                    "frame-src 'self' https://accounts.google.com https://*.google.com https://*.gstatic.com https://*.firebaseapp.com https://*.web.app;"
                )
                headers[b"Content-Security-Policy"] = csp.encode("utf-8")
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"Strict-Transport-Security"] = b"max-age=31536000; includeSubDomains"
                headers[b"Referrer-Policy"] = b"no-referrer-when-downgrade"
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


cred = credentials.Certificate("consensai-firebase-adminsdk-fbsvc-9064a77134.json")
# Prevent initializing app multiple times if reloaded
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db_firestore = firestore.client()

def verify_user_token(token: str, allow_unverified: bool = False) -> str:
    """
    Verifiziert das Firebase-ID-Token. Standardmäßig NUR verifizierte E-Mails zulassen.
    Mit allow_unverified=True kann man Endpoints wie /confirm-registration erlauben.
    """
    try:
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=5)
        if not allow_unverified and not decoded_token.get("email_verified", False):
            raise Exception("Email not verified")
        return decoded_token["uid"]
    except Exception as e:
        logging.error(f"verify_user_token failed: {e}")
        raise Exception("Invalid token")
    

def extract_id_token(request: Request, data: dict) -> Optional[str]:
    raw = data.get("id_token")
    if raw is not None and str(raw).strip().lower() in {"", "null", "undefined"}:
        raw = None
    if raw:
        return raw
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()
        if token:
            return token
    cookie_token = request.cookies.get("session")
    if cookie_token:
        return cookie_token
    return None

def is_user_pro(uid: str) -> bool:
    """
    Liest aus Firestore, ob das Feld 'tier' auf 'premium' (oder 'pro') steht.
    """
    try:
        doc_ref = db_firestore.collection("users").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            tier = data.get("tier", "").lower()
            return tier in ["premium", "pro"]
        return False
    except Exception as e:
        logging.error(f"Pro-Check Fehler für {uid}: {e}")
        return False

def is_valid_session(token: str) -> bool:
    """
    Prüft, ob das übergebene Firebase-ID-Token gültig ist.
    Gibt True zurück, wenn verify_user_token() keinen Fehler wirft.
    """
    try:
        verify_user_token(token)
        return True
    except Exception:
        return False
