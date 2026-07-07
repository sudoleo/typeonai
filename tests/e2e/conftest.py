"""Fixtures der Playwright-Smoke-Suite.

Startet einen eigenen uvicorn-Prozess mit MOCK_LLM/MOCK_AUTH auf einem
dedizierten Testport (Default 8031, NICHT 8021): laeuft parallel ein
normaler Dev-Server, duerfen die Tests nie versehentlich mit echten
LLM-Keys reden.

Voraussetzungen: siehe tests/e2e/README.md (Playwright + Chromium,
Service-Account-JSON im Root, Netz fuer CDN-Skripte).
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_PORT = int(os.environ.get("E2E_PORT", "8031"))
BASE_URL = f"http://127.0.0.1:{E2E_PORT}"

FIREBASE_STUB = (Path(__file__).parent / "firebase_stub.js").read_text(encoding="utf-8")

# Bekannt-harmlose Konsolen-Fehler (extern/umgebungsbedingt), die den
# "App laedt ohne Fehler"-Test nicht brechen sollen.
CONSOLE_ERROR_ALLOWLIST = (
    "favicon",
    "umami",
)


def filter_console_errors(errors):
    return [
        e for e in errors
        if not any(marker in e.lower() for marker in CONSOLE_ERROR_ALLOWLIST)
    ]


@pytest.fixture(scope="session")
def app_server():
    env = os.environ.copy()
    env["MOCK_LLM"] = "1"
    env["MOCK_AUTH"] = "1"
    env["DISABLE_RATE_LIMIT"] = "1"
    # Deltas gedrosselt streamen, damit die Tests den Streaming-
    # Zwischenzustand beobachten koennen (~400ms pro Modellantwort).
    env.setdefault("MOCK_LLM_DELAY_MS", "40")
    # Der Mock faengt alle Provider-Calls ab; Dummy-Keys existieren nur,
    # damit Key-Pruefungen in handle_ask/consensus nicht 500/400 werfen,
    # falls die lokale .env unvollstaendig ist.
    for name in (
        "DEVELOPER_OPENAI_API_KEY",
        "DEVELOPER_MISTRAL_API_KEY",
        "DEVELOPER_ANTHROPIC_API_KEY",
        "DEVELOPER_GEMINI_API_KEY",
        "DEVELOPER_DEEPSEEK_API_KEY",
        "DEVELOPER_GROK_API_KEY",
    ):
        env.setdefault(name, "e2e-dummy-key")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(E2E_PORT), "--log-level", "warning"],
        cwd=str(REPO_ROOT),
        env=env,
    )
    try:
        _wait_until_ready(proc)
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_until_ready(proc, timeout_seconds=90):
    """Pollt /app, bis der Server antwortet (Startup laedt u. a. die
    Modell-Konfiguration aus Firestore und braucht ein paar Sekunden)."""
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"uvicorn hat sich beim Start beendet (Exit-Code {proc.returncode}). "
                "Liegt das Firebase-Service-Account-JSON im Repo-Root?"
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/app", timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Testserver auf Port {E2E_PORT} wurde nicht rechtzeitig bereit: {last_error}")


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def context(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    # Firebase-Modul durch den Auth-Stub ersetzen (eingeloggter Free-User).
    ctx.route(
        "**/static/firebase.js*",
        lambda route: route.fulfill(content_type="application/javascript", body=FIREBASE_STUB),
    )
    # Analytics im Test nicht laden (Netz-Rauschen + Konsolen-Warnungen).
    ctx.route(
        "https://cloud.umami.is/**",
        lambda route: route.fulfill(content_type="application/javascript", body="/* blocked in e2e */"),
    )
    yield ctx
    ctx.close()


@pytest.fixture()
def console_errors():
    return []


@pytest.fixture()
def get_console_errors(console_errors):
    """Liefert die bisher gesammelten, gefilterten Konsolen-Fehler."""
    return lambda: filter_console_errors(console_errors)


@pytest.fixture()
def app_page(app_server, context, console_errors):
    """Geoeffnete /app-Seite mit initialisiertem window.App-Bus."""
    page = context.new_page()
    page.on(
        "console",
        lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
    )
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))
    page.goto(f"{app_server}/app", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.App && typeof window.sendQuestion === 'function'"
        " && typeof window.getConsensus === 'function'",
        timeout=30000,
    )
    return page
