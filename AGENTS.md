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
  Script-Ladereihenfolge in `templates/index.html` ist ein Vertrag.
- Nach CSS/JS-Änderungen den `?v=`-Cache-Buster bumpen.
- Frontend hat keine Auto-Tests → `docs/smoke-checklist.md` durchgehen.
  Backend: `.\venv\Scripts\python.exe -m pytest tests`.
