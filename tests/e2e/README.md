# Playwright-Smoke-Suite (`tests/e2e/`)

Die Suite automatisiert die risikoreichsten Punkte aus
`docs/smoke-checklist.md` gegen einen lokalen Server. LLM-Aufrufe und Login
sind gemockt; sämtliche serverseitigen Datenzugriffe gehen ausschließlich an
den lokalen Firestore-Emulator mit der fest allowgelisteten Demo-Projekt-ID
`demo-consensio-e2e`.

## Einmaliges Setup

Voraussetzungen sind Python, Node.js und Java 21. Danach:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-e2e.txt
venv\Scripts\python.exe -m playwright install chromium
npm install --global firebase-tools@13.35.1
```

## Sicherer lokaler Lauf

Terminal 1 startet nur den Emulator. Es ist absichtlich kein Firebase-
Standardprojekt in `.firebaserc` hinterlegt:

```powershell
firebase emulators:start --only firestore --project demo-consensio-e2e
```

Terminal 2 startet die Browser-Suite:

```powershell
$env:RUN_E2E = "1"
$env:FIRESTORE_EMULATOR_HOST = "127.0.0.1:8085"
venv\Scripts\python.exe -m pytest tests\e2e -v
Remove-Item Env:RUN_E2E
Remove-Item Env:FIRESTORE_EMULATOR_HOST
```

Alternativ kapselt `emulators:exec` Start und Stopp in einem Befehl:

```powershell
$env:RUN_E2E = "1"
firebase emulators:exec --only firestore --project demo-consensio-e2e `
  "venv\Scripts\python.exe -m pytest tests\e2e -v"
Remove-Item Env:RUN_E2E
```

Ohne `RUN_E2E=1` wird `tests/e2e/` von der regulären Suite ignoriert. Ohne
erreichbaren Emulator bricht die E2E-Suite ab. Es gibt keinen Fallback auf ein
Service-Account-JSON oder ein entferntes Firebase-Projekt.

### Phase-4-Browserregressionen ohne Java

`test_phase4_frontend.py` ist absichtlich ein enger, vollständig gemockter
Browserlauf: Er startet nur die App-Shell im writerfreien `E2E_TEST_MODE`,
setzt weiterhin die fest erlaubte Demo-Projekt-ID und einen Loopback-
Emulatorhost, lässt aber keinen Request Firestore erreichen. Auth-, Usage-,
Bookmark-, Share-, Watch-, `/prepare`- und Providerantworten werden im Browser
ersetzt. Dadurch sind die Konto-/Request-/Modal-Races auch ohne Java separat
ausführbar:

```powershell
$env:RUN_E2E = "1"
venv\Scripts\python.exe -m pytest tests\e2e\test_phase4_frontend.py -q
Remove-Item Env:RUN_E2E
```

Diese Ausnahme gilt nur für diese eine Datei. Der vollständige E2E-Lauf und
alle echten Transaktions-/Request-Writer verlangen weiterhin den erreichbaren
Firestore-Emulator wie oben beschrieben.

## Erzwungene Isolation

`tests/e2e/conftest.py` setzt vor dem uvicorn-Start:

- `E2E_TEST_MODE=1`,
- `FIRESTORE_EMULATOR_HOST=127.0.0.1:8085`,
- alle relevanten Projektvariablen auf `demo-consensio-e2e`,
- `MOCK_LLM=1`, `MOCK_AUTH=1` und `DISABLE_RATE_LIMIT=1`.

`app/core/e2e_profile.py` validiert Projekt und Loopback-Host vor der Firebase-
Initialisierung. Unbekannte, entfernte oder produktionsnahe Ziele stoppen den
Prozess. Das E2E-Lifespan-Profil in `main.py` deaktiviert außerdem sämtliche
Startup-/Maintenance-Writer: Modellkonfigurations-Backfill, Pending-/Share-
Cleanup, Account-Cleanup-Retry, Run-Recovery, Publisher-Lineage-Backfill,
Telegram-Startup-Maintenance sowie Watch-, Topic-, SEO-, API-Run- und Account-
Cleanup-Loops.

Request-Writer bleiben absichtlich aktiv, damit echte Datenflüsse geprüft
werden, landen aber nur im kurzlebigen Emulator. Inventar:

- aktuell ausgeführt: Usage-Reservierungen/Run-Metadaten aus `/prepare` sowie
  Chats, Turns, Context-Versionen, Modell-Completions und Turn-Abschluss,
- im Mock-Profil ausdrücklich unterdrückt: `pending_results`, Differences-
  Telemetrie sowie die durch `firebase_stub.js` ersetzten Bookmark-/Vote-Writes,
- für neue Tests erreichbar, aber weiterhin emulatorgebunden: Completions,
  Shares, Votes, Bookmarks, Watches und sonstige App-Endpunkt-Writer.

`test_phase2_transactions.py` spricht den isolierten Emulator zusätzlich direkt
über die Service-Seams an. Die Tests starten je mindestens zwei konkurrierende
Worker für Watch- und Chat-Limits sowie Share-Publikation und prüfen parallele
Share-Reports. Damit werden echte Firestore-Transaktionskonflikte und Retries
getestet; In-Memory-Fakes allein reichen für diese Race-Verträge nicht aus.

## Testprofil

- `MOCK_LLM=1` liefert deterministische Fixtures am untersten Provider-Seam;
  SSE, Differences-Parsing, Anchor-/Quote-Verifikation und Agreement-Score
  laufen echt. `MOCK_LLM_DELAY_MS=40` hält Streaming-Zwischenzustände sichtbar.
- `MOCK_AUTH=1` akzeptiert nur `e2e-mock-token` als Free-User
  `e2e-mock-user`. Im Browser ersetzt eine Playwright-Route `firebase.js` durch
  `firebase_stub.js`.
- Dummy-Eigenkeys passieren lokale Key-Prüfungen, lösen mit `MOCK_LLM=1` aber
  keine Provideraufrufe aus.
- CDN-Skripte wie marked und DOMPurify werden echt geladen; der Lauf braucht
  daher Netzzugang.

Noch nicht automatisiert sind unter anderem echte Firebase-Auth-Flows,
Provideraufrufe, Mail-/Telegram-Zustellung und Admin-Produktionsabläufe. Die
Phase-4-Suite mockt Firebase-Module, wechselt damit aber real durch die
produktive `firebase.js`-Callback-/Generation-Logik. Echte externe Auth bleibt
ausdrücklich außerhalb des E2E-Profils.
