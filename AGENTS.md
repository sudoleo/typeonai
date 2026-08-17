# Agent-Hinweise

**Erst lesen:** [`docs/codebase-map.md`](docs/codebase-map.md) — kompakte
Architektur-Karte (Stack, Routing, Frontend-Module, Kern-Flows, Backend, Daten,
kritische `window.*`/DOM-Verträge, lokale Befehle).

**Pflicht:** Wenn du **Architektur, Module, API-Endpoints oder Kern-Flows**
änderst, dokumentiere das im selben Commit/PR in `docs/codebase-map.md` mit
(siehe Abschnitt „Bei Änderungen aktualisieren" dort). Die Karte muss zum Code
passen — bei Abweichung gilt der Code, und die Karte wird korrigiert.

**Schnell-Hinweise:**
- Frontend-Module reden über `window.*` / `window.App` (keine ES-Imports). Die
  Script-Ladereihenfolge ist ein Vertrag — für `/app` steht sie in
  `static/js/bundles.json`, nicht mehr in `templates/index.html`.
- Nach Änderungen unter `static/` für `/app`: `npm run build` (Marke kommt aus
  dem Inhalt, es gibt dort nichts mehr von Hand zu bumpen) — Details in
  [`docs/frontend-build.md`](docs/frontend-build.md). Die öffentlichen Seiten
  und `admin.html` hängen weiter am manuellen `?v=`-Buster.
- Tests: `npm test` (JS-Verhalten, Vitest + jsdom) und
  `.\venv\Scripts\python.exe -m pytest tests`. Für alles, was noch keine
  Auto-Tests hat, `docs/smoke-checklist.md` durchgehen.
