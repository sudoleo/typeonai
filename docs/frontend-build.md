# Frontend-Build & JS-Tests

Betrifft **`/app`** (`templates/index.html`). Die öffentlichen Seiten
(`landing.html`, `share.html`, `topic.html`, …) hängen weiterhin am manuellen
`?v=`-Regime — siehe „Noch offen“ unten.

## Warum

Vorher lud `/app` **36 einzelne JS-Dateien (872 KB)**, deren Reihenfolge in
`index.html` beziehungsweise einem lokalen ESM-Import stand, und jede Datei
trug eine **von Hand
gepflegte `?v=`-Marke**. Vergaß man den Bump, bekamen wiederkehrende Nutzer
altes JS/CSS. Automatische Frontend-Tests gab es keine; die Absicherung waren
Python-Tests, die den JS-Quelltext nach Teilstrings durchsuchten.

Jetzt: **5 gehashte Bundles (419 KB)**, Reihenfolge in einer Datei, Marke aus dem
Inhalt, plus ein echter JS-Test-Runner.

## Befehle

```bash
npm install
```

```bash
npm run build
```

```bash
npm test
```

`npm run build:check` prüft nur, ob `static/dist/` zu den Quellen passt (Exit 1,
wenn nicht). Denselben Abgleich macht `tests/test_frontend_build.py` im
normalen pytest-Lauf, ohne Node.

## Wie es zusammenhängt

```
static/js/bundles.json      Ladereihenfolge (die EINE Quelle der Wahrheit)
        │
        ├── scripts/build_frontend.mjs   →  static/dist/*.js|css + manifest.json
        │
        └── app/core/assets.py           →  die <script>/<link>-Tags für Jinja
```

**`static/js/bundles.json`** listet die Gruppen in Ladereihenfolge. Ein neues
Skript wird **nur hier** eingetragen — nicht mehr in `index.html`. Notizen zu
Reihenfolge-Zwängen stehen als `note` neben der Datei.

**Der Build** hängt die Dateien jeder Klassik-Gruppe in genau dieser Reihenfolge
aneinander und minifiziert das Ergebnis. Bewusst **Verkettung statt
Modul-Bundling**: die Dateien teilen einen globalen Scope und reden über ~85
`window.*`-Verträge miteinander. In Modul-Scopes gewickelt wäre jede implizite
Globale still weg. esbuild läuft deshalb ohne `--bundle`/`--format`, damit
Top-Level-Namen unangetastet bleiben; `tests/test_frontend_build.py` prüft das
nach. Zwischen zwei Dateien steht ein `;` als ASI-Schutz.

`firebase.js` und `demo.js` sind echte ES-Module und werden einzeln gebündelt
(Firebase-SDK bleibt externer CDN-Import).

CSS: `style.css` ist ein `@import`-Aggregator. Der Build zieht die Kette in
Kaskadenreihenfolge in **eine** Datei. `static/dist/` liegt neben `static/css/`,
deshalb zeigen `url(../fonts/…)` und `url(../icons/…)` weiter auf dieselben
Dateien.

**`app/core/assets.py`** rendert daraus die Tags. Zwei Modi:

| | wann | Ergebnis |
|---|---|---|
| **built** | `static/dist/manifest.json` liegt vor | 5 gehashte Bundles |
| **source** | kein Build-Output **oder** `FRONTEND_DEV=1` | 36 Einzeldateien, jede mit Inhalts-Hash |

Der Source-Modus ist der Entwicklungspfad: Datei speichern, neu laden, fertig —
kein Build, kein Bump. Launch-Config dafür: `consensio-mock-dev` (Port 8035).

Auch lokale Abhängigkeiten müssen direkt in `bundles.json` stehen. Deshalb ist
`email-verify.js` Teil der `head`-Gruppe und stellt
`window.App.emailVerification` bereit, statt hinter einem zweiten, unbemerkt
veraltbaren ESM-Import zu liegen. Der Build schreibt außerdem seine vollständige
Input-Liste ins Manifest; der Python-Abgleich hasht diese Liste samt
`bundles.json`, Build-Skript und Lockfile.

## Deploy

`static/dist/` wird **mitcommittet**, damit der Render-Deploy kein Node braucht.
Nach jeder Änderung an `static/` also `npm run build` und das Ergebnis mit
committen. Fehlt es, fällt `assets.py` automatisch auf die Einzeldateien zurück
— die App läuft, nur eben unminifiziert.

Alternative, falls das Diff-Rauschen stört: `static/dist/` ignorieren und im
Render-Build-Command `npm ci && npm run build` ergänzen.

## JS-Tests

`tests/js/*.test.mjs`, Vitest + jsdom. Die Module sind keine ES-Module und
lassen sich nicht importieren, deshalb lädt `tests/js/helpers/appWindow.mjs` sie
als `<script>` in ein frisches jsdom — genau wie der Browser:

```js
const { window } = loadScripts(["static/js/composer-quote.js"], { body: MARKUP });
window.App.quote.set("…");
```

Jeder Aufruf bekommt ein eigenes Fenster, damit ein Modul, das Listener bindet
oder Globals einfriert, nicht ins nächste Testfile leckt.

Abgedeckt: `app-state.js` (Owner-Enforcement), `composer-quote.js`,
`consensus-anchor.js`. Der erste Lauf hat direkt einen echten Mangel gefunden —
`Object.freeze` auf der Owner-Tabelle war flach, jedes Skript konnte einen Owner
umschreiben und danach vorn herein schreiben. Behoben in `app-state.js`.

## Noch offen

- Die öffentlichen Seiten (`landing.html`, `share.html`, `topic.html`,
  `topics.html`, `questions.html`, `benchmark.html`, `consensus-engine.html`,
  `model-pulse.html`, `about/terms/privacy/imprint`) sowie `admin.html` laufen
  weiter mit handgepflegten `?v=`-Marken. Der Vertrag dafür ist unverändert und
  wird von `tests/test_frontend_resilience.py` durchgesetzt.
- Die verbleibenden Python-Quelltext-Verträge (`source_contract` und die
  Teilstring-Prüfungen in `test_*_ui.py`) sind weiter da. Sie sollten Stück für
  Stück nach `tests/js/` wandern, wo sie Verhalten statt Schreibweise prüfen.
