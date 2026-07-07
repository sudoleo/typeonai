# Playwright-Smoke-Suite (tests/e2e/)

Automatisiert die risikoreichsten Punkte aus `docs/smoke-checklist.md` gegen
einen lokalen Server mit gemockten LLM-Calls und gemocktem Firebase-Login.
Npm-frei: Playwright läuft über das Python-Package im venv.

## Einmaliges Setup

```powershell
# greenlet zuerst pinnen: neuere Versionen liefern kein Wheel mehr fuer das
# Python 3.9 im venv und wuerden einen MSVC-Build verlangen.
venv\Scripts\python.exe -m pip install "greenlet==3.1.1" --only-binary=:all:
venv\Scripts\python.exe -m pip install playwright
venv\Scripts\python.exe -m playwright install chromium   # lädt ~130 MB Browser
```

## Lauf

```powershell
$env:RUN_E2E = "1"
venv\Scripts\python.exe -m pytest tests\e2e -v
Remove-Item Env:RUN_E2E
```

Ohne `RUN_E2E=1` wird `tests/e2e/` von pytest ignoriert — die schnelle
Backend-Baseline (`python -m pytest tests`) bleibt unverändert.

## Wie es funktioniert

- `conftest.py` startet einen **eigenen uvicorn-Prozess auf Port 8031**
  (Override: `E2E_PORT`) — bewusst nicht 8021, damit die Tests nie mit einem
  parallel laufenden echten Dev-Server reden.
- Der Testserver läuft mit:
  - `MOCK_LLM=1` — alle Provider-/Engine-Calls liefern deterministische
    Fixtures (`app/services/llm/mock_llm.py`). Gemockt wird am untersten
    Seam (`_run_ask`, `_call_engine_text`/`_stream_engine_text`,
    `get_intent_from_llm`), d. h. SSE-Protokoll, Differences-Parsing,
    Anchor-/Quote-Verifikation und Agreement-Score laufen **echt**.
    Zusätzlich wird die `pending_results`-Share-Persistenz übersprungen
    (kein Firestore-Schreiben aus Tests).
  - `MOCK_AUTH=1` — `verify_user_token` akzeptiert das Sentinel-Token
    `e2e-mock-token` (uid `e2e-mock-user`, Free-Tier, kein Firestore-Read).
  - `DISABLE_RATE_LIMIT=1` — slowapi aus, sonst laufen die Tests in die
    3-5/minute-Limits der Endpoints.
  - `MOCK_LLM_DELAY_MS=40` — Deltas gedrosselt, damit die Tests den
    Streaming-Zwischenzustand beobachten können.
- Im Browser ersetzt eine Playwright-Route `static/firebase.js` durch
  `firebase_stub.js` (eingeloggter Free-User, No-op-Bookmarks/-Voting).

## Voraussetzungen / Grenzen

- **Nur lokal lauffähig**: Der Server braucht das gitignorte
  Firebase-Service-Account-JSON im Repo-Root (Firestore-Startup:
  `load_models_from_db`). Firestore wird gelesen, aber nicht beschrieben.
- **Netzzugang nötig**: CDN-Skripte (marked, DOMPurify) werden echt geladen.
- **Fixtures**: Eiffelturm-Szenario in `app/services/llm/mock_llm.py` —
  fünf Modelle sagen 1889, Grok sagt 1887 → ergibt deterministisch Claims,
  eine Major-Contradiction-Karte und einen Agreement-Score. Zitate/Anchors
  müssen wörtlich in den Fixture-Texten vorkommen, sonst leert die
  serverseitige Verifikation sie (Tests schlagen dann fehl).

## Bewusst (noch) nicht abgedeckt

Resolve-Runde, Share-Dialog, Attachments (Pro), Follow-up (Pro), Bookmarks,
Agent-Mode-Timer, Demo-Flow, Mobile-Layout, Login-Flow selbst. Mit dem
vorhandenen Auth-Mock sind Resolve/Share später günstig nachrüstbar.
