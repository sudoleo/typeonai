"""Fixtures der Playwright-Smoke-Suite.

Startet einen eigenen uvicorn-Prozess mit MOCK_LLM/MOCK_AUTH auf einem
dedizierten Testport (Default 8031, NICHT 8021). Firestore muss als lokaler
Emulator auf Port 8085 laufen; Projekt-ID und Emulatorziel werden vor dem
Serverstart fail-closed validiert.

Voraussetzungen: siehe tests/e2e/README.md (Playwright + Chromium,
lokaler Firestore-Emulator, Netz fuer CDN-Skripte).
"""

import os
import subprocess
import sys
import time
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from app.core.e2e_profile import E2E_PROJECT_ID, assert_safe_e2e_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_PORT = int(os.environ.get("E2E_PORT", "8031"))
BASE_URL = f"http://127.0.0.1:{E2E_PORT}"
FIRESTORE_EMULATOR_HOST = os.environ.get(
    "FIRESTORE_EMULATOR_HOST", "127.0.0.1:8085"
)

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
    env["E2E_TEST_MODE"] = "1"
    env["FIRESTORE_EMULATOR_HOST"] = FIRESTORE_EMULATOR_HOST
    env["GOOGLE_CLOUD_PROJECT"] = E2E_PROJECT_ID
    env["GCLOUD_PROJECT"] = E2E_PROJECT_ID
    env["FIREBASE_PROJECT_ID"] = E2E_PROJECT_ID
    env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    assert_safe_e2e_environment(env)
    _assert_emulator_is_reachable(FIRESTORE_EMULATOR_HOST)
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
                "Pruefe den E2E-Startfehler oberhalb dieser Meldung."
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/app", timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Testserver auf Port {E2E_PORT} wurde nicht rechtzeitig bereit: {last_error}")


def _assert_emulator_is_reachable(emulator_host: str) -> None:
    host, port_text = emulator_host.rsplit(":", 1)
    host = host.strip("[]")
    try:
        with socket.create_connection((host, int(port_text)), timeout=2):
            return
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Der Firestore-Emulator ist nicht erreichbar. Starte ihn wie in "
            "tests/e2e/README.md beschrieben; echte Firebase-Credentials sind "
            "kein erlaubter Fallback."
        ) from exc


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
    # Built und Source-Modus haben unterschiedliche URLs; beide muessen lokal
    # bleiben, sonst redet die angeblich isolierte Suite mit dem echten SDK.
    def fulfill_firebase_stub(route):
        route.fulfill(content_type="application/javascript", body=FIREBASE_STUB)

    ctx.route("**/static/firebase.js*", fulfill_firebase_stub)
    ctx.route("**/static/dist/firebase.*.js", fulfill_firebase_stub)
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
