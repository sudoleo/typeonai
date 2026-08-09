# Tests und sichere Ausführung

## Abhängigkeiten

Die Abhängigkeiten sind nach Zweck getrennt:

- `requirements.txt`: produktive Laufzeit,
- `requirements-test.txt`: reguläre Unit-/Integrationstests,
- `requirements-e2e.txt`: zusätzlich Python Playwright und das gepinnte
  `greenlet`.

Installation der regulären Testumgebung:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-test.txt
```

Für E2E zusätzlich:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-e2e.txt
venv\Scripts\python.exe -m playwright install chromium
```

## Reguläre Suite

```powershell
venv\Scripts\python.exe -m pytest tests -q
```

`tests/conftest.py` schließt `tests/e2e/` aus, solange `RUN_E2E` nicht exakt
`1` ist, und aktiviert für die reguläre Suite `UNIT_TEST_MODE=1`. Dadurch wird
Firebase Admin mit anonymen Credentials und dem nicht produktiven Demo-Projekt
`demo-consensio-unit` an einem geschlossenen Loopback-Port initialisiert; ein
vergessener Fake kann dadurch keinen externen Dienst erreichen. Alle fachlichen
Datenzugriffe verwenden weiterhin die jeweiligen In-Memory-Fakes. Unit-/
Integrationstests dürfen daher keine Browserinstallation, Firebase-Credentials
oder externen Dienste voraussetzen.

Reine Quelltextverträge tragen den Marker `source_contract`. Sie laufen in der
regulären Suite mit, gelten aber nicht als Ersatz für Browserverhalten. Eine
gezielte Bestandsaufnahme ist möglich mit:

```powershell
venv\Scripts\python.exe -m pytest -m source_contract -q
```

Kritische Verträge aus `test_agent_mode_ui.py` und
`test_frontend_resilience.py` haben korrespondierende Playwright-Flows für
Disclosure/Agent Mode, Streaming-Degradation und die App-Shell. Der Cache-
Vertrag inventarisiert dagegen bewusst alle aktiven lokalen JS-/CSS-URLs,
prüft ein einheitliches `?v=YYYYMMDD-label` je Asset und vergleicht das Datum
mit dem letzten Git-Commit beziehungsweise einer aktuellen Arbeitsbaumänderung.

## Browser-E2E

Die E2E-Suite ist ein separater Lauf und benötigt den Firestore-Emulator. Sie
darf nie durch beliebige Credentials oder ein Firebase-Standardprojekt ersetzt
werden. Vollständiges Setup, Writer-Inventar und Befehle:
[`tests/e2e/README.md`](../tests/e2e/README.md).

## CI und Artefakte

`.github/workflows/tests.yml` läuft bei Pull Requests, Pushes auf `main` und
manuell:

1. `Unit and integration` installiert `requirements-test.txt`, führt die
   reguläre Suite mit explizit deaktiviertem E2E-Profil aus und blockiert bei
   jedem Fehler.
2. `E2E (Firestore emulator)` startet erst nach erfolgreicher regulärer Suite,
   installiert gepinnte Browser-/Firebase-Werkzeuge und kapselt pytest in
   `firebase emulators:exec --project demo-consensio-e2e`.

Beide Jobs laden ihre JUnit-XML auch im Fehlerfall als GitHub-Actions-Artefakt
hoch (`unit-integration-results` beziehungsweise `e2e-results`). Die E2E-
Umgebung enthält keine produktiven Secrets.

Für manuelle Frontend-QA bleibt [`smoke-checklist.md`](smoke-checklist.md)
verbindlich.
