from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
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
from app.core.entitlements import (
    TIER_FREE,
    TIER_PLUS,
    TIER_PRO,
    Entitlements,
    entitlements_for,
    normalize_tier,
    tier_at_least,
)
from app.core.observability import safe_exception

# --- E2E-Test-Hook (MOCK_AUTH=1) ------------------------------------------
# Die Playwright-Suite laeuft ohne echten Firebase-Login: verify_user_token
# akzeptiert dann genau das Sentinel-Token und die Tier-Checks antworten fuer
# den Mock-UID ohne Firestore-Roundtrip. In Produktion ist MOCK_AUTH nie
# gesetzt; alle Hooks sind dann No-ops.
E2E_MOCK_TOKEN = "e2e-mock-token"
E2E_MOCK_UID = "e2e-mock-user"
ACCOUNT_DELETION_JOBS_COLLECTION = "account_deletion_jobs"

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


def _mock_auth_tier() -> str:
    """Stufe des E2E-Mock-Users. Ohne MOCK_TIER bleibt er Free, damit die
    Suite den Free-Pfad testet; MOCK_TIER=plus|pro schaltet gezielt um."""
    return normalize_tier(os.environ.get("MOCK_TIER"))


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
        strict_script_page = (
            path in {"/app", "/app/watches"}
            or path == "/admin"
            or path.startswith("/admin/")
        )
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
                script_src = (
                    "script-src 'self' "
                    + ("" if strict_script_page else "'unsafe-inline' ")
                    + "https://cdn.jsdelivr.net https://www.gstatic.com "
                    "https://apis.google.com https://accounts.google.com https://cloud.umami.is; "
                )
                csp = (
                    "default-src 'self' https://cdn.jsdelivr.net https://www.gstatic.com; "
                    + script_src
                    +
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


class AuthenticationUnavailable(Exception):
    pass


class TierStatusUnavailable(Exception):
    pass


AUTH_TOMBSTONE_CACHE_TTL_SECONDS = 30
_auth_tombstone_cache = TTLCache(maxsize=4096, ttl=AUTH_TOMBSTONE_CACHE_TTL_SECONDS)
_auth_tombstone_cache_lock = threading.Lock()


def invalidate_auth_tombstone_cache(uid: str, *, blocked: bool | None = None) -> None:
    with _auth_tombstone_cache_lock:
        _auth_tombstone_cache.pop(uid, None)
        if blocked is not None:
            _auth_tombstone_cache[uid] = bool(blocked)


def _is_account_tombstoned(uid: str) -> bool:
    with _auth_tombstone_cache_lock:
        cached = _auth_tombstone_cache.get(uid)
    if cached is not None:
        return bool(cached)
    try:
        snap = db_firestore.collection(ACCOUNT_DELETION_JOBS_COLLECTION).document(uid).get()
        data = snap.to_dict() if snap.exists else {}
    except Exception as exc:
        raise AuthenticationUnavailable(
            "Account status is temporarily unavailable"
        ) from exc
    blocked = bool(data and data.get("status") == "pending")
    if data and data.get("status") == "completed":
        expires_at = data.get("tombstone_expires_at")
        blocked = (
            not isinstance(expires_at, datetime)
            or expires_at.tzinfo is None
            or expires_at.astimezone(timezone.utc) > datetime.now(timezone.utc)
        )
    with _auth_tombstone_cache_lock:
        _auth_tombstone_cache[uid] = blocked
    return blocked

def verify_user_token(
    token: str,
    allow_unverified: bool = False,
    *,
    check_revoked: bool = False,
) -> str:
    """
    Verifiziert das Firebase-ID-Token. Standardmäßig NUR verifizierte E-Mails zulassen.
    Mit allow_unverified=True kann man Endpoints wie /confirm-registration erlauben.
    """
    if _mock_auth_enabled() and token == E2E_MOCK_TOKEN:
        return E2E_MOCK_UID
    try:
        kwargs = {"clock_skew_seconds": 5}
        if check_revoked:
            kwargs["check_revoked"] = True
        decoded_token = auth.verify_id_token(token, **kwargs)
        if not allow_unverified and not decoded_token.get("email_verified", False):
            raise Exception("Email not verified")
        uid = decoded_token["uid"]
        if _is_account_tombstoned(uid):
            raise Exception("Account deletion is in progress")
        return uid
    except AuthenticationUnavailable:
        raise
    except Exception as e:
        logging.warning("verify_user_token failed: %s", type(e).__name__)
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

_TIER_FLAGS_DEFAULT = {"tier": TIER_FREE, "pro": False, "admin": False}


def _compute_tier_flags(data: dict) -> dict:
    tier = normalize_tier(data.get("tier"))
    role = str(data.get("role", "")).lower()
    return {
        "tier": tier,
        # "pro" heisst weiterhin "darf teure Modelle und Deep Think" und ist
        # fuer Plus bewusst False (siehe app/core/entitlements.py).
        "pro": tier == TIER_PRO,
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
    except Exception as exc:
        logging.error("Tier lookup failed category=%s", safe_exception(exc))
        raise TierStatusUnavailable("Account tier is temporarily unavailable") from exc
    flags = _compute_tier_flags(data or {})
    with _tier_cache_lock:
        _tier_cache[uid] = flags
    return flags


def invalidate_tier_cache(uid: str) -> None:
    """Cache-Eintrag verwerfen, z.B. nach /delete_account."""
    with _tier_cache_lock:
        _tier_cache.pop(uid, None)


def get_user_tier(uid: str) -> str:
    """Kontostufe aus Firestore (gecacht): "free", "plus" oder "pro".

    Einzige Quelle fuer alles, was Plus zusaetzlich darf. Wer nur wissen will,
    ob teure Modelle erlaubt sind, nimmt weiter is_user_pro().
    """
    if _mock_auth_enabled() and uid == E2E_MOCK_UID:
        return _mock_auth_tier()
    return _get_tier_flags(uid)["tier"]


def get_user_entitlements(uid: str) -> Entitlements:
    return entitlements_for(get_user_tier(uid))


def is_user_pro(uid: str) -> bool:
    """
    Liest (gecacht) aus Firestore, ob das Feld 'tier' auf 'premium' (oder 'pro') steht.
    Plus ist hier absichtlich NICHT enthalten: das Flag steuert den Zugriff auf
    teure Modelle und Deep Think.
    """
    if _mock_auth_enabled() and uid == E2E_MOCK_UID:
        return _mock_auth_tier() == TIER_PRO
    return _get_tier_flags(uid)["pro"]


def is_user_plus(uid: str) -> bool:
    """Plus ODER Pro - die Stufe, ab der Anhaenge und Resolve freigeschaltet sind."""
    return tier_at_least(get_user_tier(uid), TIER_PLUS)

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
