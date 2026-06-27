"""Import-sichere, gemeinsame Credential-Helfer fuer Developer-API-Keys + Gemini-Auth.

Eine **einzige** Quelle fuer die Developer-Key-Env-Namen (``DEVELOPER_*_API_KEY``),
damit der Benchmark dieselben Secrets wie die Produktion nutzt, ohne aus
``app.api.*`` zu importieren (die Router bauen das ``api_keys``-Dict aktuell inline
in ``app/api/routers/chat.py`` – diese Namen sind hier gespiegelt und sollen die
einzige zusaetzliche Quelle bleiben).

Modul-Ebene importiert nur ``os``; google.auth wird lazy in den Funktionen geladen.
Es werden hier **keine** Secrets geloggt.
"""

from __future__ import annotations

import os

# Kanonische Provider-Namen (wie ``api_keys``-Dict + query_consensus sie erwarten)
# -> Developer-Key-Env-Variable.
DEVELOPER_API_KEY_ENV: dict[str, str] = {
    "OpenAI": "DEVELOPER_OPENAI_API_KEY",
    "Mistral": "DEVELOPER_MISTRAL_API_KEY",
    "Anthropic": "DEVELOPER_ANTHROPIC_API_KEY",
    "Gemini": "DEVELOPER_GEMINI_API_KEY",
    "DeepSeek": "DEVELOPER_DEEPSEEK_API_KEY",
    "Grok": "DEVELOPER_GROK_API_KEY",
}

GEMINI_SCOPES = ["https://www.googleapis.com/auth/generative-language"]


def resolve_developer_api_keys(providers: list[str] | None = None) -> dict[str, str | None]:
    """Liest die Developer-Keys aus der Umgebung. Leere Strings -> None."""
    names = providers or list(DEVELOPER_API_KEY_ENV)
    keys: dict[str, str | None] = {}
    for name in names:
        env_name = DEVELOPER_API_KEY_ENV[name]
        keys[name] = (os.environ.get(env_name) or "").strip() or None
    return keys


def gemini_adc_available() -> bool:
    """True, wenn Application Default Credentials auffindbar sind – **ohne**
    Token-Refresh (kein HTTP-Call an einen LLM-Provider). Discovery kann lokal
    erfolgen; in Tests wird diese Funktion gemockt."""
    try:
        import google.auth

        creds, _ = google.auth.default(scopes=GEMINI_SCOPES)
        return creds is not None
    except Exception:
        return False


def gemini_adc_headers() -> dict:
    """ADC-Bearer-Header fuer Gemini – reuse der **produktiven** Logik
    (``app.services.llm.engines._google_adc_headers``), damit der Fallback exakt
    dem der App entspricht. Refresht das Token (Live-Pfad)."""
    from app.services.llm.engines import _google_adc_headers

    return _google_adc_headers()


def missing_credentials(api_keys: dict[str, str | None], required: list[str]) -> list[str]:
    """Liste der Provider ohne nutzbare Credentials. Fuer ``Gemini`` zaehlt ein
    vorhandener API-Key **oder** verfuegbares ADC."""
    missing: list[str] = []
    for name in required:
        if name == "Gemini":
            if not api_keys.get("Gemini") and not gemini_adc_available():
                missing.append(name)
        elif not api_keys.get(name):
            missing.append(name)
    return missing
