# Index Refactor Handoff

Stand: 2026-06-26

Branch: `refactor/index-extraction`

Letzter Refactor-Code-Commit vor dieser Handoff-Datei:
`1a0befe Refactor: extract consensus lifecycle`

## Update 2026-06-26: Phase 3 + 4 abgeschlossen

Die JS-Extraktion aus `templates/index.html` ist abgeschlossen. Die Datei
enthält jetzt nur noch den Jinja-`<head>`-Config-Block und Markup plus die
Modul-`<script>`-Tags - **kein App-JS mehr inline** (7149 -> 4254 -> 1147 Zeilen).

Neue Module seit diesem Handoff (verhaltenserhaltend, je per Browser-Smoke +
145 Backend-Tests verifiziert):

- `04b4d84` `static/js/consensus-run.js`: `window.getConsensus` (/consensus-Payload,
  SSE-Stream, Ergebnis-Rendering, Citation/Share-Meta) + `parseBestModel`.
- `5af8171` `static/js/query-send.js`: `window.sendQuestion` (/prepare +
  /ask_*-Fan-out, Streaming, Usage/Tier-UI, Auto-Consensus-Trigger), Query-Run-State,
  `window.cancelCurrentQuery`, Query-Helfer (isDemoQuery/predictSearchIntent/
  getActiveMode/setSendButtonRunning/...). index.html fragt jetzt
  `window.isQueryRequestRunning()` statt das private Flag; `validateInputText`
  bleibt inline und wird via `window.validateInputText` geteilt.
- `a8a0317` `static/js/app-init.js`: das restliche `initApp()` (Theme, Usage/Limits +
  User-Status, Response-Box-UI-Toggles, Sidebar/Layout, Modals, Tooltips,
  Evidence-Rendering, API-Key-Test, alles DOM-Wiring). Deferred Script am
  `</body>`-Ende -> laeuft nach allen anderen Modulen, ruft `initApp()` direkt
  (DOM bereits geparst). Die einzige inline-Jinja-Stelle (`{{ free_limit }}`)
  ist via `window.FREE_LIMIT` im `<head>`-Config-Block gebridged.

### Bewusst noch NICHT angefangen

- **Echter State-Refactor**: `window.App` ist weiterhin ein Uebergangsbus, DOM
  dient vielerorts als State (z. B. `dataset.consensusAnswer`, `.excluded`-Klassen).
  Das Aufloesen kommt erst nach erfolgreicher Extraktion.
- **CSS-Aufraeumen**: noch offen.
- Bekannter toter Code, der mitgewandert ist (nicht angefasst): `predictSearchIntent`
  (nie aufgerufen), `consensusGenerated` (nur geschrieben), `showDisclaimerPopup`
  (in index.html, ungenutzt).

Das Folgende beschreibt den Stand VOR Phase 3/4 und ist historisch.

## Bereits extrahierte Module

- `static/js/app-core.js`: temporaerer `window.App`-Bus, Modell-Konfiguration, gemeinsame UI-/Tracking-Helfer.
- `static/js/model-picker.js`: Modell-Auswahl, Custom-Picker, Default-Modelle.
- `static/js/markdown-stream.js`: Markdown-Rendering und SSE-Streaming-Helfer.
- `static/js/sources.js`: Sources-/Evidence-Mapping, `dataset.consensusAnswer`, `dataset.consensusSources`.
- `static/js/attachments.js`: Attachment UI/Payload fuer Pro-Dateien.
- `static/js/agent-mode.js`: Agent Mode UI, Status, Timer, Auto-Consensus-Lock.
- `static/js/share-dialog.js`: Public share dialog und `window.openShareDialog`.
- `static/js/consensus-actions.js`: Consensus Copy/Citation/Share-Actions.
- `static/js/user-tier.js`: Tier-/Pro-UI und Premium-Modellstatus.
- `static/js/consensus-insights.js`: strukturierte Consensus-Auswertung, Credibility-Frames, Jump-to-answer.
- `static/js/consensus-lifecycle.js`: Consensus-Sichtbarkeit, Gate/Availability, Run-State, Abort/Cancel, Run-ID-Gating, Auto-Consensus-Persistenz.

## Consensus-Lifecycle-Vertraege

Bestehende `window`-Vertraege bleiben erhalten:

- `window.revealConsensusOutput()`
- `window.hideConsensusOutput()`
- `window.canGenerateConsensus()`
- `window.updateConsensusButtonAvailability()`
- `window.cancelCurrentConsensus()`

Neue gezielte Bruecke fuer den noch inline liegenden Run-Code:

- `window.App.consensusLifecycle.initAutoConsensusToggle()`
- `window.App.consensusLifecycle.setGate(disabled)`
- `window.App.consensusLifecycle.startRun()` -> `{ runId, signal }`
- `window.App.consensusLifecycle.isActiveRun(runId)`
- `window.App.consensusLifecycle.finishRun(runId)`
- `window.App.consensusLifecycle.setSynthesizing(isSynthesizing)`
- `window.App.consensusLifecycle.markPendingCanceled()`
- `window.App.consensusLifecycle.isRunning()`

Wichtig: Agent Mode bleibt die einzige Stelle, die den Auto-Consensus-Toggle erzwingt oder sperrt. `consensus-lifecycle.js` verwaltet nur die generische Toggle-Initialisierung, Persistenz und den Change-Listener.

## Verifikation

Syntax:

```powershell
node --check static\js\consensus-lifecycle.js
```

Lifecycle-Harness gegen die echte Datei:

- `startRun()` liefert `{ runId, signal }`.
- `isRunning()` ist nach `startRun()` `true`.
- `isActiveRun(runId)` ist fuer den aktuellen Run `true`.
- `window.cancelCurrentConsensus()` setzt `signal.aborted` auf `true`.
- Nach Cancel ist `isRunning()` `false`.
- Ein alter Run ist nach Start eines neuen Runs nicht mehr aktiv.
- `finishRun(oldRunId)` bereinigt den neuen Run nicht.
- `finishRun(currentRunId)` setzt den Running-State zurueck.

Backend-/Python-Suite:

```powershell
.\venv\Scripts\python.exe -m pytest tests
```

Ergebnis: `145 passed, 15 warnings`.

## Naechster Schritt

1. `static/js/consensus-run.js` verhaltenserhaltend extrahieren.
   - Nur `window.getConsensus()` plus Request-Payload, `/consensus` SSE-/Delta-/Final-Verarbeitung und Ergebnis-Rendering.
   - Keine Aenderung am SSE-Protokoll, Backend, Query-Send oder State-Modell.
   - Die bestehende Lifecycle-Bruecke verwenden: `startRun()`, `isActiveRun()`, `finishRun()`, `setSynthesizing()`.
   - Bestehende `window`-Vertraege wie `window.getConsensus`, `window.consensusCitationMeta` und `window.lastShareResultId` erhalten.

2. Danach erst `query-send.js` analysieren und separat extrahieren.
   - Query-Run-State, Provider-Requests, `/prepare`, Modellantwort-Rendering und Auto-Consensus-Aufruf bleiben bis dahin unveraendert in `templates/index.html`.
