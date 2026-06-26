# Index Refactor Handoff

Stand: 2026-06-26

Branch: `refactor/index-extraction`

Letzter Refactor-Code-Commit vor dieser Handoff-Datei:
`1a0befe Refactor: extract consensus lifecycle`

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
