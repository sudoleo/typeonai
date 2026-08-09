import logging
import os
import threading
from typing import Optional
from cachetools import TTLCache
from fastapi import Request
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.auth.credentials import AnonymousCredentials

from app.core.e2e_profile import (
    assert_safe_e2e_environment,
    e2e_test_mode_enabled,
    firebase_project_id,
)

# --- E2E-Test-Hook (MOCK_AUTH=1) ------------------------------------------
# Die Playwright-Suite laeuft ohne echten Firebase-Login: verify_user_token
# akzeptiert dann genau das Sentinel-Token und die Tier-Checks antworten fuer
# den Mock-UID ohne Firestore-Roundtrip. In Produktion ist MOCK_AUTH nie
# gesetzt; alle Hooks sind dann No-ops.
E2E_MOCK_TOKEN = "e2e-mock-token"
E2E_MOCK_UID = "e2e-mock-user"

# Test-Flags, die Sicherheitskontrollen abschalten. Ein Kommentar "nie in
# Produktion setzen" ist keine Kontrolle - ein einziges versehentlich in Render
# gesetztes MOCK_AUTH=1 wuerde jedem Besucher Admin-Zugriff geben. Deshalb
# verweigert der Prozess in Produktion den Start, statt unsicher weiterzulaufen.
_UNSAFE_TEST_FLAGS = (
    "MOCK_AUTH",
    "MOCK_ADMIN",
    "MOCK_LLM",
    "DISABLE_RATE_LIMIT",
    "E2E_TEST_MODE",
    "UNIT_TEST_MODE",
)


def _is_production() -> bool:
    """Render setzt RENDER_SERVICE_NAME in jedem Deploy automatisch."""
    if os.environ.get("RENDER_SERVICE_NAME"):
        return True
    return os.environ.get("ENVIRONMENT", "").strip().lower() in {"production", "prod"}


def _assert_no_unsafe_test_flags_in_production() -> None:
    enabled = [flag for flag in _UNSAFE_TEST_FLAGS if os.environ.get(flag) == "1"]
    if enabled and _is_production():
        raise RuntimeError(
            "Refusing to start: the test-only flag(s) "
            + ", ".join(enabled)
            + " are set in a production environment. They disable authentication, "
            "admin checks or rate limiting. Unset them in the Render dashboard."
        )


_assert_no_unsafe_test_flags_in_production()
assert_safe_e2e_environment()


def _mock_auth_enabled() -> bool:
    return os.environ.get("MOCK_AUTH") == "1"


if _mock_auth_enabled():
    logging.warning("MOCK_AUTH=1 aktiv - Firebase-Auth ist fuer das E2E-Sentinel-Token gemockt. NIE in Produktion setzen.")

class CustomSecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        sensitive_api_response = (
            path == "/chats"
            or path.startswith("/chats/")
            or path == "/bookmarks"
            or path.startswith("/bookmarks/")
            or path == "/bookmark"
            or path.startswith("/bookmark/")
            or path.startswith("/api/v1/")
            or path.startswith("/api/admin/")
        )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                csp = (
                    "default-src 'self' https://cdn.jsdelivr.net https://www.gstatic.com; "
                    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com https://apis.google.com https://accounts.google.com https://cloud.umami.is; "
                    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    "img-src 'self' data: https://lh3.googleusercontent.com https:; "
                    "connect-src 'self' "
                    "https://cloud.umami.is "
                    "https://gateway.umami.is "
                    "https://api-gateway.umami.dev "
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
                    "https://api.x.ai https://api.deepseek.com "
                    "https://cdn.jsdelivr.net; "
                    "frame-src 'self' blob: https://accounts.google.com https://*.google.com https://*.gstatic.com https://*.firebaseapp.com https://*.web.app;"
                )
                headers[b"Content-Security-Policy"] = csp.encode("utf-8")
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"Strict-Transport-Security"] = b"max-age=31536000; includeSubDomains"
                # strict-origin-when-cross-origin: an fremde Seiten geht nur noch
                # die Origin, nicht der volle Pfad. Private Watch-Seiten
                # (/s/{id}) sind Secret-URLs - ihr Pfad darf nie im
                # Referer-Header eines ausgehenden Klicks landen.
                headers[b"Referrer-Policy"] = b"strict-origin-when-cross-origin"
                if sensitive_api_response:
                    headers[b"Cache-Control"] = b"private, no-store"
                    headers[b"Pragma"] = b"no-cache"
                    headers[b"Expires"] = b"0"
                    headers[b"Vary"] = b"X-API-Key, Authorization"
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


# Prevent initializing app multiple times if reloaded
class _AnonymousTestCredential(credentials.Base):
    """Credential accepted by firebase-admin without any live ADC lookup."""

    def get_credential(self):
        return AnonymousCredentials()


if not firebase_admin._apps:
    if e2e_test_mode_enabled():
        # Admin SDKs bypass Firestore rules, so the local emulator endpoint and
        # the demo-only project ID are both validated above before this point.
        firebase_admin.initialize_app(
            _AnonymousTestCredential(),
            options={"projectId": firebase_project_id()},
        )
    elif os.environ.get("UNIT_TEST_MODE") == "1":
        # Unit tests patch repositories with in-memory fakes. Initializing the
        # SDK with an unresolvable demo project and a closed loopback endpoint
        # keeps collection/import reproducible without credentials and makes an
        # accidentally unmocked read fail locally instead of reaching Google.
        os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "127.0.0.1:1")
        firebase_admin.initialize_app(
            _AnonymousTestCredential(),
            options={"projectId": "demo-consensio-unit"},
        )
    else:
        cred = credentials.Certificate("consensai-firebase-adminsdk-fbsvc-9064a77134.json")
        firebase_admin.initialize_app(cred)

db_firestore = firestore.client()

def verify_user_token(token: str, allow_unverified: bool = False) -> str:
    """
    Verifiziert das Firebase-ID-Token. Standardmäßig NUR verifizierte E-Mails zulassen.
    Mit allow_unverified=True kann man Endpoints wie /confirm-registration erlauben.
    """
    if _mock_auth_enabled() and token == E2E_MOCK_TOKEN:
        return E2E_MOCK_UID
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

# --- Tier-Flag-Cache -------------------------------------------------------
# is_user_pro/is_user_admin wurden pro Aufruf je ein Firestore-Get;
# ein Frage-Fan-out summierte sich auf 15+ Reads. Ein gemeinsamer Fetch des
# users/{uid}-Dokuments liefert beide Flags, der TTL-Cache haelt sie kurz
# (60s), damit manuell vergebene Pro-/Admin-Tags schnell greifen. Fehler werden
# NICHT gecacht (naechster Aufruf versucht Firestore erneut).
TIER_CACHE_TTL_SECONDS = 60
_tier_cache = TTLCache(maxsize=4096, ttl=TIER_CACHE_TTL_SECONDS)
_tier_cache_lock = threading.Lock()

_TIER_FLAGS_DEFAULT = {"pro": False, "admin": False}


def _compute_tier_flags(data: dict) -> dict:
    tier = str(data.get("tier", "")).lower()
    role = str(data.get("role", "")).lower()
    return {
        "pro": tier in ("premium", "pro"),
        "admin": role == "admin",
    }


def _get_tier_flags(uid: str) -> dict:
    with _tier_cache_lock:
        flags = _tier_cache.get(uid)
    if flags is not None:
        return flags
    try:
        doc = db_firestore.collection("users").document(uid).get()
        data = doc.to_dict() if doc.exists else {}
    except Exception as e:
        logging.error(f"Tier-Lookup Fehler für {uid}: {e}")
        return _TIER_FLAGS_DEFAULT
    flags = _compute_tier_flags(data or {})
    with _tier_cache_lock:
        _tier_cache[uid] = flags
    return flags


def invalidate_tier_cache(uid: str) -> None:
    """Cache-Eintrag verwerfen, z.B. nach /delete_account."""
    with _tier_cache_lock:
        _tier_cache.pop(uid, None)


def is_user_pro(uid: str) -> bool:
    """
    Liest (gecacht) aus Firestore, ob das Feld 'tier' auf 'premium' (oder 'pro') steht.
    """
    if _mock_auth_enabled() and uid == E2E_MOCK_UID:
        return False
    return _get_tier_flags(uid)["pro"]

def is_user_admin(uid: str) -> bool:
    """
    Liest (gecacht) aus Firestore, ob das Feld 'role' auf 'admin' steht.
    """
    if _mock_auth_enabled() and uid == E2E_MOCK_UID:
        # MOCK_ADMIN=1 (nur zusammen mit MOCK_AUTH wirksam) erlaubt E2E-Tests
        # des Admin-Dashboards; ohne das Flag bleibt der Mock-User Non-Admin.
        return os.environ.get("MOCK_ADMIN") == "1"
    return _get_tier_flags(uid)["admin"]

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
