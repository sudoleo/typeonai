# consens.io — Codebase Map

Kompakte Architektur-Übersicht für Coding-Agents. Ziel: in wenigen Minuten
verstehen, wie das Projekt gebaut ist, wo Logik liegt und was bei Änderungen zu
beachten ist. Bewusst kurz gehalten — keine vollständige Datei-/Funktionsliste.

> Nur verifizierte Fakten. Wenn dieses Dokument von der Realität abweicht, gilt
> der Code. Pflege-Regeln siehe **Bei Änderungen aktualisieren** am Ende.

---

## 1. Projektüberblick & Stack

consens.io vergleicht Antworten mehrerer LLM-Provider nebeneinander und
synthetisiert daraus einen **Consensus** plus eine strukturierte
**Differences**-Analyse. Optional: Agent Mode (Auto-Consensus), Datei-Anhänge
(Pro), öffentliche Share-Seiten.

- **Backend**: Python, FastAPI (`fastapi==0.115.8`), via `uvicorn` ausgeliefert.
  SSE-Streaming über `StreamingResponse`. Rate-Limiting via `slowapi`.
- **LLM-Provider**: OpenAI, Mistral, Anthropic, Gemini, DeepSeek, Grok — über die
  jeweiligen SDKs bzw. REST. Provider-Label-Konvention: Claude = `Anthropic`.
- **Auth & Daten**: Firebase Auth (ID-Token) + Firestore (`firebase-admin`).
  Gemini kann zusätzlich über ein Google Service Account laufen.
- **Frontend**: kein Framework. Jinja2-Templates + Vanilla-JS-Module unter
  `static/js/`, geladen als klassische `<script defer>`-Tags. Übergangs-State-Bus
  ist `window.App` plus zahlreiche `window.*`-Globals (siehe §8).
- **Markdown**: `marked` + `DOMPurify` (CDN) clientseitig; serverseitig für
  Share-Seiten in `app/services/public_markdown.py`.
- **Hosting**: Render. Täglicher Render-Restart wird bewusst als Reset für
  In-Memory-State und als Trigger für Cleanup-Jobs genutzt (siehe §7).

---

## 2. Einstiegspunkte / Routing / Templates

**`main.py`** ist der App-Entry: lädt `.env`, setzt `GOOGLE_APPLICATION_CREDENTIALS`,
fügt `CustomSecurityMiddleware` (CSP etc.) + slowapi-Limiter hinzu, mountet
`/static`, registriert globale Exception-Handler und inkludiert alle Router.
Im `lifespan`-Startup: `load_models_from_db()` + Share-Cleanups (siehe §7),
Recovery/Retention noch nicht abgeschlossener Consensus-API-Runs und Retry
blockierter API-Account-Cleanups. Cancellable asyncio-Tasks übernehmen danach
den 60-Sekunden-API-Maintenance-, 5-Minuten-Account-Cleanup- und
30-Minuten-Consensus-Watch-Tick.

Router liegen unter `app/api/routers/` und werden in `main.py` eingebunden:

| Router | Zweck (Auswahl an Pfaden) |
|---|---|
| `pages.py` | HTML-Seiten + SEO: `/` (Landing, auch mit aktiver Session direkt erreichbar), `/app` (Haupt-App), `/app/watches` (gleiche App-Shell; watch.js öffnet anhand des Pfads das Watch-Dashboard), `/admin`, `/admin/benchmark` (Benchmark-Run-Visualisierung), `/about`, `/ai-model-comparison`, `/consensus-engine` (nutzerfreundliche Consensus-Engine-Erklärung), `/privacy` `/imprint` `/terms`, `robots.txt`, `sitemap*.xml`. Außerdem `/feedback`, `/vote`, `/check_keys` (nur verifizierte Logins zum Testen eigener Keys). |
| `chat.py` | Kern-LLM-Flow: `/prepare`, `/ask_openai` `/ask_mistral` `/ask_claude` `/ask_gemini` `/ask_deepseek` `/ask_grok`, `/consensus`, `/resolve`. `/prepare` und die `/ask_*`-Endpoints akzeptieren ein optionales `context`-Feld für Follow-up-Fragen (Pro, siehe §4). Die sechs `/ask_*`-Endpoints sind dünne Wrapper um `handle_ask` + die deklarative Provider-Registry `ASK_PROVIDERS` (Provider-Eigenheiten wie Gemini-Service-Account, `gemini_key`-Legacy-Feld, `useOwnKeys`-Flag und Env-Key-Namen stehen dort, Rate-Limits als Literal am Endpoint). |
| `auth.py` | `/register`, `/confirm-registration` (setzt nach verifiziertem Login zusätzlich eine kurzlebige HttpOnly-Session für private servergerenderte Seiten), `DELETE /auth/session` (Logout-Cleanup). |
| `users.py` | `/user_status`, `/usage`, `/usage/run/release`, `/delete_account`, `/track-interest`. |
| `bookmarks.py` | `/bookmarks` (GET), `/bookmark` (POST/DELETE), `/bookmark/consensus` sowie `POST /bookmark/consensus/share-result` zum sicheren Wiederherstellen eines Share-/Watch-fähigen Pending-Snapshots aus einem eigenen Consensus-Bookmark. Beide Save-Endpunkte liefern den vollständig zusammengeführten Bookmark-Datensatz zurück, damit der Client ihn ohne Reload aktualisiert. |
| `share.py` | `/api/share` (POST), `/api/share/{id}` (DELETE), `/api/my/shares`, `/api/share/{id}/report`, öffentliche Seite `/s/{slug_id}`, `sitemap-shares.xml`. |
| `watch.py` | Consensus Watch: `/api/watch` (POST), `/api/my/watches` (inkl. kompakter History je Watch), `/api/watch/{id}` (PATCH/DELETE), Morning-Brief-Einstellungen `/api/my/watch-brief` (GET/PATCH) sowie öffentliche, HMAC-signierte `/watch/unsubscribe`- und `/watch/brief/unsubscribe`-Links. |
| `api_v1.py` | Nutzergebundene asynchrone Consensus-API: Run-Start/Status/Löschung unter `/api/v1/consensus/runs`, idempotentes Publizieren erfolgreicher Runs per `POST .../{run_id}/share`, eigene Share-Liste/-Details/-Widerruf unter `/api/v1/shares` sowie direkte Admin-Indexfreigabe per `PUT /api/v1/shares/{share_id}/indexing`. Auth über gescopte `X-API-Key`s, Run-Idempotenz über den Pflichtheader `Idempotency-Key`; Pydantic-Modelle bilden den Vertrag in `/openapi.json` ab. |
| `admin.py` | `/api/admin/shares`, `/api/admin/shares/{id}/moderate`, `/api/admin/models` (GET/POST), API-Key-Ausgabe/-Liste/-Widerruf unter `/api/admin/api-keys`, `/api/admin/watches` (Diagnose-Liste), `/api/admin/watches/{id}/run` (fällig stellen + Scheduler sofort wecken), `/api/admin/watches/test-email` (SMTP-Test an die verifizierte Admin-Adresse), `/api/admin/benchmark/runs` (Liste) + `/api/admin/benchmark/runs/{run_id}` (Detail, liest Firestore-publizierte kompakte Benchmark-Reports mit lokalem Disk-Fallback über `benchmark/report_reader.py`). Alle hinter `is_user_admin`. |

**Zentrale Templates** (`templates/`, gerendert mit `Jinja2Templates`):
`landing.html` (Marketing), `index.html` (die App — Haupt-Markup + Script-Tags),
`admin.html`, `admin_benchmark.html` (Admin-Benchmark-Visualisierung, eigenes
Template + Firebase-Auth-Modul wie `admin.html`), `share.html` (öffentliche
Consensus-Seite), `share_unavailable.html`, plus statische Rechts-/SEO-Seiten
und SEO-Erklärseiten wie `ai-model-comparison.html` / `consensus-engine.html`.
Alle öffentlichen HTML-Seiten teilen Navigation und Footer über
`templates/partials/public_nav.html` und `public_footer.html`. Der Landing-Hero
ist seit 2026-07-17 demo-first: Ein klickbares Input-Feld (Look des /app-Inputs,
"Try the demo"-Pill, Provider-Chips darunter) verlinkt auf `/app?demo=1`;
`static/demo.js` erkennt den Parameter und startet die Demo automatisch in der
echten App. Die Consensus-Engine-Seite nutzt weiterhin die Ergebnisdarstellung
aus `partials/product_result_mockup.html`. Die gemeinsamen, an `/app`
ausgerichteten Light-/Dark-Tokens liegen in `static/css/public-tokens.css` und
werden von `landing.css` sowie `public-pages.css` importiert; seitenbezogene
Layouts bleiben in diesen beiden Dateien bzw. in `benchmark.css` und
`consensus-engine.css`.
**`index.html` enthält kein App-JS inline mehr**
— nur den Jinja-Config-Block im `<head>` und die Modul-`<script>`-Tags.

---

## 3. Frontend-Architektur

Geladen werden (Reihenfolge ist Vertrag, siehe §8): zuerst CDN-Libs
(`marked`, `DOMPurify`), dann `firebase.js` + `demo.js` (ES-Module), `app-ui.js`,
dann die Feature-Module unter `static/js/` in fester Reihenfolge, zuletzt —
deferred am `</body>` — `app-init.js`.

**Modul-Verantwortlichkeiten** (alle in `static/js/` außer markiert):

- **`app-core.js`** — MUSS zuerst laden. Definiert `window.App`-Bus, `modelPrefs`
  (zentrales Mapping Provider→DOM-IDs), `deepThinkModelLabels`, gemeinsame Helfer
  (`getModelOptionLabel`, `getSelectedModelCount`, `setAppTitle`, `showPopup`,
  `trackAppEvent`, `exitHeroMode`) sowie den zentralen
  `window.App.renderUsageDisplay`-Renderer. Dieser ignoriert fehlende Usage-Felder
  aus parallelen Antworten und bewahrt den DOM-/Layout-Vertrag (Label links,
  fetter Wert rechts). `window.App.usageRun` hält einen logischen
  Idempotency-Key pro UI-Lauf, geteilt von `/prepare`, allen `/ask_*` und
  `/consensus`. `setAppTitle` setzt den Standardtitel oder
  einen gekürzten, fragebezogenen Browser-Tab-Titel; `exitHeroMode` schaltet
  Antwortbereich und Input vom zentrierten Leerzustand in den Laufzustand.
- **`model-picker.js`** — Modellauswahl/Custom-Picker, Default-Modelle, localStorage-
  Persistenz (`restoreModelSelections`). Der Consensus-Picker hat seit 2026-07-18
  eine Preset-Ebene (Fast/Balanced/High Quality + Custom): Presets kommen aus
  `window.CONSENSUS_PRESETS` und setzen als zusammenhaengendes Model-Set alle
  sechs Antwortmodelle plus Consensus-Engine. High Quality (interne ID
  `thorough`) ist Pro-only und zeigt
  ein Pro-Badge; eine manuelle Antwort- oder Consensus-Modellwahl wechselt zu
  Custom. Die nativen Selects werden dabei OHNE change-Event gesetzt (Muster
  wie Deep Think). Die High-Quality-Basis nutzt OpenAI GPT-5.6 Sol; gespeicherte
  Legacy-Werte mit GPT-5.5 werden bei der Normalisierung migriert. Zustand in localStorage
  `pref_consensus_preset` ("custom" = explizite Modellwahl, ausgeloest durch
  jedes change-Event am Dropdown); `pref_select_consensus` bleibt die
  Custom-Wahl. Bestandsnutzer mit gespeicherter Modellwahl migrieren zu
  "custom"; die volle Modell-Liste bleibt bewusst ohne Beschreibungen.
- **Navigation/Settings-Shell** (`templates/index.html`, `layout.css`,
  `components-modals.css`, `app-init.js`, `firebase.js`) — Models,
  Leaderboard und Bookmarks sind ausschließlich Sidebar-Abschnitte mit
  integrierten Icons. Bei offener Desktop-Sidebar begrenzen symmetrische
  Gutters die Contentbreite und halten den Input in der Viewport-Mitte; mobil
  bleibt außerhalb der Sidebar nur der Burger sichtbar. Gast-Login/-Sign-up
  sitzt oben rechts, während der Sidebar-Footer nur für eingeloggte Accounts
  das Avatar-Menü mit deckender Light-/Dark-Fläche zeigt. Settings sind in
  Experience, Connections, Model behavior und Account gruppiert; die
  bestehenden Control-IDs bleiben der JavaScript-Vertrag.
- **`markdown-stream.js`** — Markdown-Rendering (`injectMarkdown`) + SSE-Helfer
  (`createStreamRenderer`, `streamSSERequest`).
- **`sources.js`** — Quellen/Evidence-Mapping; nutzt DOM-Datasets
  `dataset.consensusAnswer` / `dataset.consensusSources`; `window.currentEvidenceSources`.
- **`attachments.js`** — Attachment-UI/Payload (Pro), inklusive Bild-Paste im
  Fragefeld und Bild-Drag-and-drop auf den Input-Container. Solange ein echter
  Anhang für die nächste Frage bereitliegt, wird DeepSeek temporär mit einem
  sichtbaren Kompatibilitätshinweis deaktiviert, weil dessen Chat-API keine
  Datei-/Bild-Inputs akzeptiert; nach Entfernen wird die vorherige Auswahl
  wiederhergestellt. `window.pendingAttachments`, `getAttachmentsPayload`.
- **`agent-mode.js`** — Agent-Mode-UI/Status/Timer; zeigt pro Modell den
  Query-Abschluss aus `dataset.responseState`; einzige Stelle, die den
  Auto-Consensus-Toggle erzwingt/sperrt. Sobald Antworten vorliegen, bietet das
  Panel einen dezenten, session-lokalen „Show/Hide model answers“-Disclosure;
  jeder neue Lauf startet wieder mit verborgenen Einzelantworten.
- **`consensus-progress.js`** — beobachtende Zwei-Phasen-Anzeige direkt unter
  dem Input für reguläre Läufe: zählt fertige Modellantworten determiniert und
  zeigt die anschließende Consensus-/Differences-Synthese bewusst indeterminiert.
  Im Agent Mode bleibt sie verborgen; Status-Brücke ist
  `window.App.consensusPipeline.*`.
- **`consensus-lifecycle.js`** — Consensus-Sichtbarkeit, Gate/Availability,
  Run-State, Abort/Cancel, Run-ID-Gating, Auto-Consensus-Persistenz. Exponiert die
  `window.App.consensusLifecycle.*`-Brücke (siehe §4/§8).
- **`share-dialog.js`** — `window.openShareDialog`, Share-Liste und die gemeinsame
  `window.App.sharedModal.*`-Steuerung für den Share-/Watch-Dialog (einziger
  innerer Scrollbereich, Background-Scroll-Lock, Escape/Focus-Restore).
- **`consensus-actions.js`** — Copy/Citation/Share-Buttons am Consensus.
- **`watch.js`** — `window.openWatchDialog` (Create-Dialog im Share-Modal) und
  `window.openWatchDashboard` (eigene Seite `/app/watches`: Vollbild-View
  `#watchDashboard` unter dem fixen View-Switch, Styles in
  `static/css/components-watch.css`; URL-Sync via pushState/popstate, Deep-Link
  wartet auf den asynchronen Firebase-Auth-Status): Karten je Watch mit
  Score/Delta/History-Sparkline, letzter Änderung, nächstem Lauf und
  Inline-Settings (Intervall/Uhrzeit/Mailmodus/Condition, Pause/Delete) sowie
  die Morning-Brief-Karte (`/api/my/watch-brief`, Toggle im selben
  `.switch`/`.slider`-Stil wie das Input-Feld). Ohne vorhandene Watch ist der
  Toggle erklärend deaktiviert; das Backend erzwingt dasselbe Gate und schaltet
  den Brief beim Löschen der letzten Watch ab. `openWatchDialog("list")`
  leitet auf die Seite um; Einstieg zusätzlich über den login-gated,
  schwebenden View-Switch `#viewSwitch` (Consensus/Watches; `firebase.js`
  blendet ihn ein/aus, `watch.js` synchronisiert URL und aktiven Zustand). Nach dem
  ersten erfolgreichen, speicherbaren Consensus zeigt `window.App.watch.*`
  einmalig einen dezenten Hinweis am Watch-Button; Schließen oder Öffnen des
  Features persistiert die Bestätigung in `localStorage`. Die
  Browser-IANA-Zeitzone wird zusammen mit `HH:MM` an das Backend gesendet.
  Weekly-Watches senden zusätzlich den gewählten lokalen Wochentag
  (`run_weekday`) und können ihn im Dashboard nachträglich ändern.
- **`user-tier.js`** — Free/Pro-UI, Premium-Modellstatus (`updateUserTierUI`,
  `updatePremiumModelsState`) und Plan-Label im Sidebar-Account-Footer.
- **`consensus-insights.js`** — strukturierte Auswertung: Claim-Badges,
  Difference-Karten, Credibility-Frame-Farben, Jump-to-answer, Spalten-Balancer,
  Resolve-Runde (Button an Widerspruchs-Karten → `POST /resolve`).
- **`consensus-run.js`** — `window.getConsensus`: baut `/consensus`-Payload, fährt
  den SSE-Stream, rendert Ergebnis + Citation/Share-Meta. `parseBestModel`.
- **`query-send.js`** — `window.sendQuestion`: `/prepare` + `/ask_*`-Fan-out,
  Streaming-Rendering, Usage/Tier-UI, Auto-Consensus-Trigger, Query-Run-State
  (`isQueryRequestRunning`, `cancelCurrentQuery`). Ein valider erster Lauf
  beendet über `window.exitHeroMode()` den zentrierten Input-Leerzustand.
- **`app-init.js`** — das gesamte `initApp()`: Theme, Usage/Limits + User-Status,
  Response-Box-Toggles, Sidebar/Layout, Modals, Tooltips, Evidence-Rendering,
  API-Key-Test. Läuft als letztes Script, ruft `initApp()` direkt auf.

**Nicht unter `static/js/`** (älter, eigene Verantwortung):
- **`static/firebase.js`** (ES-Modul) — Firebase-Init, Login/Logout, Token-Handling,
  `window.auth`, Bookmarks-CRUD-Calls, Feedback, Voting, Tier-Sync sowie das
  Nutzericon-Menü im Sidebar-Footer (Avatar, Name/Plan, „Shared links“ und
  direkt darunter „Watched“). Ein geöffnetes Bookmark beendet den Hero-
  Leerzustand sofort. Bookmark-
  Saves aktualisieren `window.bookmarksData` und das DOM direkt aus dem vom
  Server zurückgegebenen Merge-Ergebnis. Nach einer E-Mail-Registrierung zeigt
  das Auth-Modal einen eigenen Verifizierungs-Erfolgszustand statt eines Browser-Alerts.
- **`static/demo.js`** (ES-Modul) — Demo-Flow (`runDemoFlow`) für die „Demo"-Query;
  zeigt Gästen nach Abschluss der Demo am Eingabebereich eine Login-/Registrierungs-
  Aufforderung, ohne die Demo-Frage aus dem deaktivierten Feld zu entfernen, und
  beendet beim Start denselben Hero-Leerzustand wie eine echte Anfrage.
- **`static/app-ui.js`** — System-Prompt-/Help-Modal + App-Width-Resizer.

**Abhängigkeitsrichtung**: `app-core.js` → Feature-Module → `app-init.js`. Module
kommunizieren über `window.*`-Globals und `window.App`, **nicht** über Imports. DOM
dient vielerorts als State (z. B. `.excluded`-Klasse, Datasets) — bewusster
Übergangszustand, noch nicht aufgelöst.

---

## 4. Kern-Flows

### Anfrage an Modelle (Streaming)
1. Frontend `sendQuestion` (`query-send.js`) ruft zuerst **`POST /prepare`**:
   Auth + Follow-up-Gate und transaktionale Usage-Reservierung anhand des vom
   Client erzeugten, kostenfreien `usage_run_key`; Antwort: finaler
   `system_prompt` + persistenter UTC-Tagesstand.
   Echtzeitdaten holen sich die Modelle selbst über die native Web-Suche in jedem
   Provider-Call (`engines.py`), daher kein Intent-Router/Realtime-Injektion mehr.
2. Fan-out an die ausgewählten **`/ask_<provider>`**-Endpoints (parallel), je mit
   `stream:true`. Backend prüft Auth, Pro-Status, Deep-Search-Berechtigung,
   Wortlimit (`validate_question_word_limit`) und Modell (`validate_model`),
   parst Attachments und konsumiert den reservierten Run idempotent. Alle
   parallelen Provider teilen denselben Key: der erste `/ask_*`-Aufruf wechselt
   `reserved→consumed`, alle weiteren sehen `consumed` und kosten nichts
   zusätzlich. Clientseitige Modellanzahl/Kosten werden nicht akzeptiert.
   Eigene Provider-Keys dürfen nur verifizierte Nutzer verwenden; sie umgehen
   die Usage-Zählung, aber nicht Auth/Pro-Gates.
3. **SSE-Protokoll Modellantwort** (`streaming_model_response` in `streaming.py`):
   `event: delta {text}` … dann `event: final {response, sources,
   free_usage_remaining, deep_remaining, is_pro_user, key_used}`. Bei Fehler kommt
   ein `final` mit `error`. Frontend rendert deltas und wertet `final` aus.
4. Ohne Agent Mode begleitet `consensus-progress.js` den Lauf rahmenlos unter
   dem Input: Antwortfortschritt basiert auf `dataset.responseState`; nach dem
   Fan-out wechselt die Anzeige zur nicht prozentual geschätzten Synthesephase
   und verschwindet bei Abschluss, Fehler oder Abbruch.

### Follow-up-Fragen (Pro)
Nach einem erfolgreichen Consensus kann eine Anschlussfrage mit Kontext
gestellt werden. Kontext ist **genau eine Ebene**: das letzte Frage/Konsens-
Paar (`{previous_question, previous_consensus}`) — bewusst NICHT die sechs
Modellantworten (Kostenkontrolle, der Kontext geht in alle `/ask_*`-Prompts).
- Frontend: `window.App.followup` (in `consensus-run.js`) zeigt nach dem
  Consensus-Render eine „Ask a follow-up"-Affordance im Input-Bereich
  (`#followupBar`), Pro-gebadged; Free-Klick öffnet das Pro-Modal. Aktivieren
  erzeugt einen Kontext-Chip mit X; `query-send.js` konsumiert den State beim
  Senden und legt `context` in den `/prepare`- und alle `/ask_*`-Payloads.
  **Follow-ups verketten sich nicht** (Kostenkontrolle): `consume()` markiert
  den Lauf via `followupInFlight`, der Konsens einer Follow-up-Frage bietet
  keine weitere Affordance an — erst eine frische Frage schaltet sie wieder frei.
- Backend: `normalize_followup_context` (`chat.py`) validiert und kappt beide
  Texte serverseitig (`followup_max_question_chars` /
  `followup_max_consensus_chars` in `LIMITS`). `/prepare` gated nur
  (403 `pro_required` für Nicht-Pro, auch mit eigenen Keys); die **Injektion
  passiert ausschließlich in `handle_ask`** via `build_followup_system_prompt`
  (`base.py`), damit der Kontextblock nie doppelt im Prompt steht und auch den
  `/prepare`-Fallback-Pfad des Frontends überlebt.

### Consensus & Differences
- Im App-Layout steht `#consensusOutput` oberhalb der Modellantworten: Das
  synthetisierte Ergebnis ist die Primäransicht, die einzelnen Antworten sind
  darunter die prüfbare Grundlage.
- `getConsensus` (`consensus-run.js`) sammelt die vorhandenen Modellantworten +
  `excluded_models` + `consensus_model` und ruft **`POST /consensus`**
  (`stream:true`) mit demselben `usage_run_key`. Der Endpoint validiert/
  konsumiert den Run idempotent und erzeugt keine zweite Usage-Einheit.
- Deep Think bleibt ein separater Pro-Laufmodus neben High Quality: Das Preset
  waehlt Premium-Modelle, Deep Think ergaenzt Prompt, Provider-Reasoning,
  hoeheres Tokenbudget und eigenes Kontingent. Deep Think wählt im Frontend
  temporär `Gemini 3.5 Flash` als Pro-Consensus-
  Modell. Beim Ausschalten wird die vorherige Consensus-Auswahl wiederhergestellt,
  ohne die gespeicherte Nutzerpräferenz zu überschreiben. Das Modell bleibt in
  der serverseitig normalisierten Consensus-Liste verpflichtend verfügbar.
- Backend (`chat.py::consensus` → `consensus_engine.py`): validiert (mind. **2**
  eingeschlossene Antworten), kappt Frage/Antworten serverseitig
  (`cap_engine_text`, Limits `consensus_max_answer_chars` /
  `consensus_max_question_chars` — Kosten-/Abuse-Schutz, da die Texte vom
  Client kommen), prüft Engine-Keys, dann `stream_consensus` gefolgt von
  `stream_differences`. **SSE-Events**: `consensus.delta`, `differences.delta`
  (Frontend rendert Differences-Deltas nicht), dann `final {consensus_response,
  differences, differences_data, result_id?, …usage}`. Während Reasoning-Phasen
  tragen die Delta-Events gedrosselt `{reasoning: true}`; ein SSE-Wrapper sendet
  zusätzlich Kommentar-Keepalives, wenn eine Engine länger keine Bytes liefert.
  `differences_data` ist
  strukturiertes JSON (Verdict, Karten, `best_model`, `models_compared`).
- Robustheit Differences (`consensus_engine.py`): einheitlicher Engine-Dispatch
  (`_resolve_engine`/`_call_engine_text`/`_stream_engine_text`), Structured
  Output je Provider (json_object / responseMimeType / Anthropic-Prefill).
  Judge-Policy (`_resolve_differences_engine`): die Judge-Familie ist immer
  eine ANDERE als die der gewählten Consensus-Engine (Self-Judging-Bias);
  die Stufe folgt der Engine — Standard-Engine → Standard-Judge
  (`DIFFERENCES_JUDGE_MODEL_BY_PROVIDER`), Pro-Engine → Pro-Judge über die
  Engine-Aliasse (`<Familie>-Pro`). Attempt-Plan: primärer Judge, Retry,
  nächste Fremd-Familie (Pro fail-opent zuletzt auf einen Standard-Judge);
  ohne Fremd-Key fail-open auf den eigenen Standard-Judge. Der tatsächlich
  genutzte Judge steht als `differences_data.judges.differences`
  ({provider, model, tier, attempts, duration_ms}) im Payload/Snapshot und in
  der Telemetrie. Pro-Judges laufen mit niedriger Reasoning-Effort-Kappung;
  die Consensus-Synthese selbst behält die volle Modell-Denktiefe.
  das Frontend zeigt ihn als Fußnote im Verdict-Header. Außerdem:
  JSON-Truncation-Repair, serverseitige Anchor-/Quote-Verifikation gegen
  Konsens- bzw. Modellantworten (nicht belegbare Zitate werden geleert).
  Unparsbares JSON erreicht den Nutzer nie als Rohtext.
- Agreement-Score (`compute_agreement_score`): 0-100 aus Claim-Zustimmungsquoten
  minus severity-gewichteter Widerspruchs-Penalty (major 0.25 / minor 0.10 /
  emphasis 0.05), mit Caps ("very" nur ohne Differenzen; 1 Major → max
  "partially", 2+ Major → max "hardly"; 2 Modelle → max 75). Liegt als
  `differences_data.agreement` im Payload/Snapshot; der Legacy-Credibility-Satz
  wird daraus abgeleitet (nie divergierende Verdicts). Widersprüche tragen
  `severity` ("major"/"minor", Default major); Frontend zeigt Score im
  Verdict-Header und "critical"/"minor detail"-Tags (rote bzw. Bernstein-Stufe),
  alte Bookmarks/Snapshots ohne die Felder degradieren aufs bisherige Rendering.
- Consensus-Fehlerpfad: `query/stream_consensus` versuchen es bei Provider-
  Fehlern (503, Timeout, ...) ein zweites Mal (`CONSENSUS_MAX_ATTEMPTS`);
  gescheiterte Finals tragen `error: true`. `chat.py` erkennt Fehlertexte über
  `is_consensus_error_text` und überspringt dann Differences (Judge darf nie
  den Fehlertext "analysieren") sowie die Share-Persistenz; die Differences-
  Spalte zeigt `DIFFERENCES_SKIPPED_TEXT`.
- Bei erfolgreichem Lauf eines verifizierten Nutzers wird das Ergebnis als
  `pending_result` für das Share-Feature persistiert (→ `result_id`).

### Resolve-Runde
`POST /resolve` (`chat.py` → `resolve_engine.py`) konfrontiert die
dissentierenden Modelle eines Widerspruchs (Karte aus `differences_data`)
gezielt mit der Gegenposition: pro beteiligtem Modell ein paralleler Call auf
dem günstigen Judge-Modell seines Providers
(`DIFFERENCES_JUDGE_MODEL_BY_PROVIDER`), Structured Output
`{decision: maintain|revise, position, reason}`. Aggregiertes Outcome:
`resolved` (≥1 revidiert, ≥1 bleibt) / `standoff` (alle bleiben) /
`mutual_revision` (alle revidieren) / `error`. Verifizierter Login nötig,
kostet 1 eigenen persistenten Run (außer `useOwnKeys`), Eingaben werden wie bei
`/consensus` serverseitig gekappt (`normalize_resolve_positions`), Ergebnis
wird **nicht** persistiert. Frontend: „Resolve with the models"-Button an
Contradiction-Karten in `consensus-insights.js` (nur bei ≥2 beteiligten
Modellen).

### Agent Mode
`agent-mode.js` koppelt Auto-Consensus: nach Abschluss aller Modellantworten löst
`query-send.js` automatisch `getConsensus` aus. Run-State/Gating läuft über
`consensus-lifecycle.js` (`startRun()→{runId,signal}`, `isActiveRun`, `finishRun`,
`setSynthesizing`, `cancelCurrentConsensus`). Agent Mode ist die **einzige** Stelle,
die den Auto-Consensus-Toggle erzwingt/sperrt. Standardmäßig bleiben die sechs
Einzelantwortboxen verborgen; `#agentModeAnswersToggle` setzt ausschließlich die
session-lokale Body-Klasse `.agent-mode-show-answers`, ohne Agent Mode oder dessen
Auto-Consensus-Kopplung zu deaktivieren.

### Attachments (Pro)
Frontend `attachments.js` baut Payload; Backend `app/services/llm/attachments.py`
validiert: max **2** Dateien, je **5 MB**, MIMEs PDF/PNG/JPEG/WebP. Bild-Support:
openai/anthropic/gemini/grok; PDF-Support: openai/anthropic/gemini (sonst
Text-Fallback/PDF-Extraktion). **In Firestore landen nie Datei-Bytes**, nur
Metadaten (Name/Typ/Größe) — siehe `bookmarks.py::sanitize_attachment_meta`.
Bilder können zusätzlich zum Dateiauswahldialog per Paste im `#questionInput`
oder per Drag-and-drop auf `.chat-input-container` angehängt werden; beide Wege
nutzen dasselbe Pro-Gate sowie dieselben Anzahl-/Größenlimits und die bestehende
Whitelist für Bild-MIME-Typen.

### Auth / Usage / Tier
- Firebase-ID-Token wird mit `verify_user_token` geprüft (Standard: nur
  E-Mail-verifizierte Nutzer; `allow_unverified=True` nur für Registrierung/Delete).
- Token-Quelle: `extract_id_token` liest Body `id_token`, sonst `Authorization:
  Bearer`, sonst Cookie `session`.
- Eigene API-Keys sind ein eingeloggtes Feature: `/check_keys`, `/ask_*` mit
  User-Key und `/consensus` mit `useOwnKeys` verlangen ein verifiziertes Token.
- Pro-Status: `is_user_pro` liest Firestore `users/{uid}.tier ∈ {premium, pro}`.
  Admin: `users/{uid}.role == admin`.
- **Tier-Flags sind gecacht**: `is_user_pro`/`is_user_early`/`is_user_admin`
  teilen sich einen TTL-Cache (60s, `security.py::_tier_cache`) über das
  `users/{uid}`-Dokument — ein Firestore-Read statt drei pro Aufrufstelle.
  Fehler werden nicht gecacht; `/delete_account` invalidiert via
  `invalidate_tier_cache(uid)`. Manuell vergebene Pro/Early-Tags greifen
  dadurch erst nach ≤60s.
- **Usage ist persistent und run-basiert:**
  `app/services/usage_repository.py` definiert `UsageRepository` und die
  Firestore-Implementierung `FirestoreUsageRepository`. Ein kompletter
  Consensus-Run reserviert genau **einen Integer-Slot**, unabhängig von
  Modellanzahl oder Provider-Fan-out. Deep Think zählt ebenfalls genau einmal
  gegen dieses Total und zusätzlich gegen ein separates Deep-Think-Kontingent.
  `reserve` führt Idempotenzprüfung, Limitprüfung und Reservierung gemeinsam in
  einer Firestore-Transaktion aus; `consume` und `release` wechseln den Status
  ebenfalls transaktional, `snapshot` liest das einzelne UTC-Tagesaggregat.
  Der neue Free-Default ist 3 reguläre Runs pro UTC-Tag
  (`free_consensus_run_limit`); reguläre und Deep-Think-Run-Limits je Tier sind
  als vier eigene `app_config/models.limits`-Felder konfigurierbar. `/prepare`
  reserviert, der erste serverfinanzierte Provider-Aufruf konsumiert, und
  `/consensus` wiederholt denselben Consume idempotent. `/resolve` erzeugt einen
  eigenen Run. `/usage` und `/user_status` lesen die Firestore-Tagesbasis;
  `/usage/run/release` gibt nur noch nicht konsumierte Reservierungen frei.
  `app/core/state.py` enthält nur noch den kurzlebigen Feedback-Cooldown; die
  alten Float-/Request-Counter sind entfernt.
- Limits/Defaults kommen aus `app/core/config.py` (`get_consensus_run_limit`,
  `get_deep_think_run_limit`, `get_word_limit`, `get_output_token_limit`, …)
  und können per Firestore
  (`app_config/models.limits`) überschrieben werden.
- Die Antwortmodell-Picker wenden bei einem Tier-Wechsel die Free-/Early-/Pro-
  Defaults erneut an, solange der Nutzer für den jeweiligen Provider keine
  explizite Auswahl (`pref_select_*`) gespeichert hat. Explizite Picker-Werte
  haben Vorrang. Watch-Runs lesen den aktuellen Pro-Status des Owners bei jedem
  Lauf neu und wählen danach `WATCH_MODELS_BY_TIER` (ein Upgrade wirkt deshalb
  auch auf bereits bestehende Watches nach Ablauf des Tier-Cache).

### Nutzergebundene Consensus-API (v1)
- Admins stellen über `POST /api/admin/api-keys` einen gescopten Schlüssel für eine
  aktive, E-Mail-verifizierte Firebase-UID aus. Nur die einmalige Antwort enthält den
  Klartextschlüssel (`cns_live_…`); Firestore speichert ausschließlich dessen
  SHA-256-Hash als Dokument-ID. Defaults sind `consensus:run` + `share:write`;
  `share:index` ist nur für Admin-UIDs ausstellbar und wird am Index-Endpoint
  zusammen mit der aktuellen Admin-Rolle erneut geprüft. Legacy-Keys erhalten
  nur die sicheren Defaults. Liste und Widerruf laufen über
  `GET/DELETE /api/admin/api-keys`.
- `POST /api/v1/consensus/runs` akzeptiert ausschließlich `question` und
  `deep_think`; unbekannte Felder werden abgelehnt. Der Server wählt die sechs
  Antwortmodelle aus dem konfigurierten Balanced-Preset, die reguläre
  Consensus-Engine aus demselben Preset und für Deep Think stattdessen
  `DEEP_THINK_CONSENSUS_MODEL`. Kosten, Limits, Modelle oder Modellanzahl sind
  keine Request-Felder.
- API v1 verwendet bewusst immer alle sechs Provider einschließlich DeepSeek;
  für diesen Vertrag gibt es keinen per-Request Opt-out. Privacy/Terms weisen
  auf die verpflichtende Verarbeitung durch DeepSeek in China hin.
- UID + gehashter `Idempotency-Key` zeigen auf genau einen persistenten Run;
  derselbe Key mit anderem Request ergibt 409. Der API-State folgt
  `accepted → reserved → running → succeeded|failed`. Der transaktionale
  `reserved→running`-Claim ist die einzige Berechtigung zum Providerstart.
  Doppelte HTTP-Requests/Worker können deshalb weder doppelt konsumieren noch
  den Provider-Fan-out doppelt starten.
- Die Usage-Reservierung nutzt unverändert `FirestoreUsageRepository`: ein
  vollständiger Run konsumiert beim Übergang zu `running` genau eine Total-
  Einheit; Deep Think zusätzlich genau eine Deep-Think-Einheit. Fehler vor
  Providerstart releasen, Fehler nach Providerstart bleiben konsumiert.
  Provider- und Engine-Aufrufe liegen immer außerhalb aller Transaktionen.
- `api_consensus_runner.py` orchestriert parallel die bestehenden Provider-
  Funktionen und danach unverändert `query_consensus` + `query_differences`;
  es gibt keine zweite Consensus-Engine. `GET /api/v1/consensus/runs/{run_id}`
  liefert nur eigene Runs und nach Erfolg das persistierte Ergebnis. Reservierte
  Runs werden beim Startup sicher neu eingeplant; laufende Runs werden nie
  wiederholt und nach abgelaufenem Lease als `worker_interrupted` beendet. Vor
  dem Usage-Consume wird der Firebase-/Block-Status erneut geprüft. Eine
  deduplizierte Queue lässt höchstens 32 aktive/wartende Run-IDs bei zwei
  parallelen Pipelines zu; der periodische Maintenance-Tick nimmt persistierte
  Restarbeit wieder auf und reconciled pre-provider Usage-Reservierungen.
- Run-Inhalt und Idempotenz-Mapping tragen `expires_at` (30 Tage ab Annahme);
  periodischer Cleanup löscht beide, bestehende v1-Dokumente werden beim
  ersten Maintenance-Lauf nachmigriert. `DELETE /api/v1/consensus/runs/{run_id}`
  löscht eigene terminale Runs früher. Alle v1- und Admin-Key-Antworten sind
  `private, no-store`. Limits greifen vor Auth pro IP/API-Key und danach pro UID.
- Der maschinenlesbare Vertrag kommt aus den typisierten FastAPI-Routen unter
  `/openapi.json` (Security-Scheme `ConsensusApiKey`, Header `X-API-Key` und
  Pflichtheader `Idempotency-Key`).
- Erfolgreiche eigene Runs werden per `POST .../{run_id}/share` direkt (ohne
  24h-Pending-Zwischenschritt) in einen unveränderlichen Public-Snapshot
  überführt. Eine deterministisch aus der privaten Run-ID abgeleitete 95-Bit-
  Share-ID macht Retries idempotent; `GET /api/v1/shares` liefert Resume-/
  Themenhistorie, `DELETE` widerruft. `PUT .../indexing` erzwingt neben
  `share:index` den Quality-Filter sowie Deduplikation gegen bereits indexierte
  gleiche `question_hash`-Seiten und schreibt API-Key-/Review-Auditfelder.
- `scripts/publish_consensus.py` orchestriert Themenwahl (optional OpenAI
  Responses API + Web Search), Run/Poll, Publish und Indexfreigabe ohne externe
  Python-Abhängigkeiten. `.github/workflows/publish-consensus.yml` startet ihn
  wöchentlich oder manuell; Secrets bleiben ausschließlich in GitHub Actions.

### Sharing
- `/consensus` legt ein `pending_results`-Dokument an → `result_id`.
- Die Consensus-API publiziert dagegen direkt aus ihrem 30-Tage-Run-Snapshot;
  `source_api_run_id` bleibt serverintern und wird nie Teil der Public-Payload.
- Consensus-Bookmarks speichern diese ID mit, solange sie gültig ist. Beim
  Teilen/Watchen eines älteren, geöffneten Bookmarks erzeugt
  `POST /bookmark/consensus/share-result` aus dem serverseitigen Bookmark
  best-effort einen neuen sanitisierten Pending-Snapshot; Share/Watch warten
  auf diese Rehydration und können dadurch auch außerhalb der Ursprungssession
  verwendet werden. Ein Versions-Gate verhindert, dass eine verspätete Antwort
  auf ein inzwischen anderes angezeigtes Ergebnis zeigt.
- `POST /api/share` (`share.py` → `share_snapshots.create_share_from_pending`)
  macht daraus einen unveränderlichen Share-Snapshot (`shares`-Collection) mit Slug.
- **`GET /s/{slug_id}`** rendert read-only aus dem Snapshot (keine LLM-Calls).
  Public-Snapshots enthalten JSON-LD, Canonical-Dedup über `question_hash` und
  „verwandte Fragen"; **Indexierung (`index, follow`) nur wenn der Admin `indexed`
  setzt** — nie automatisch; sonst `noindex`. Private Watch-Snapshots werden am
  Endpoint serverseitig auf die Eigentümer-Session geprüft und nie indexiert,
  gecacht, reportet oder als Related/Sitemap-Ziel ausgegeben. Public-Caching via
  `SHARE_CACHE_CONTROL` + In-Process-Cache (`get_share_cached` /
  `invalidate_share_cache`).
- Lösch-/Moderationswege: Owner `DELETE /api/share/{id}`, Besucher-`report`,
  Admin-`moderate`. 30-Tage-Hard-Delete widerrufener Shares via `cleanup_revoked_shares`.

---

## 5. Backend-Struktur

```
main.py                      App-Entry, Middleware, Router-Registrierung, lifespan
app/core/
  config.py                  Modell-Kataloge, Tier-Limits, Firestore-Sync (load_models_from_db)
  security.py                Firebase-Init, Token/Tier/Admin-Checks, CSP-Middleware
  rate_limit.py              slowapi-Limiter (Client-IP hinter Render-Proxy via XFF)
  state.py                   Kurzlebiger In-Memory-Feedback-Cooldown
app/api/routers/             siehe §2
  api_v1.py                  Gescopte Run-, Publish-, Share-Lifecycle- und Indexing-API + OpenAPI-Modelle
app/services/llm/
  base.py                    System-Prompt, Wortzählung, validate_model
  engines.py                 Provider-Requests (build_provider_payload, query_*)
  streaming.py               SSE-Helfer, stream_*_query, streaming_model_response
  consensus_engine.py        query/stream_consensus + query/stream_differences, normalize_model_name
  resolve_engine.py          Resolve-Runde (run_resolve_round, normalize_resolve_positions)
  citations.py               Antwort-Parsing + Quellen (source_response, make_llm_result)
  attachments.py             Attachment-Validierung/Aufbereitung
app/services/
  usage_repository.py        Firestore-Usage fuer logische Runs (reserve/consume/release/snapshot)
  api_account_cleanup.py     Fail-closed Account-Blocks + retrybare API-Datenlöschung
  api_key_repository.py      SHA-256-gehashte, UID-gebundene API-Schluessel
  api_run_repository.py      Idempotenz + persistente API-Run-State-Machine
  api_consensus_runner.py    Asynchroner At-most-once-Orchestrator auf bestehenden Engines
  share_snapshots.py         Snapshot-Lifecycle (pending→share), Quoten, Cleanups, Sitemap-Quellen
  watch_service.py           Watch-CRUD, Tier-/Intervall-/Conditionregeln, Share-Sichtbarkeit, Unsubscribe-Tokens
  opinion_map.py             Datenminimierte, mehrdimensionale Provider-Positionen + Direction-Shift-Berechnung
  watch_brief.py             Morning-Brief-Settings (watch_briefs), transaktionaler Claim, Digest-Aggregation, Brief-Unsubscribe-Tokens
  watch_scheduler.py         Global-Lease, Tagesbudget, sequenzielle tierkonfigurierte Watch-Läufe + run_brief_tick (Morning-Brief-Versand)
  mailer.py                  Multipart-HTML/Plaintext-SMTP-Versand via Thread-Executor
  public_markdown.py         Server-Markdown-Rendering für Share-Seiten
  differences_stats.py       Anonyme Differences-Telemetrie (differences_stats-Collection, §6)
```

Wichtige Verträge im Backend:
- Provider-Label-Set überall identisch: `OpenAI, Mistral, Anthropic, Gemini,
  DeepSeek, Grok` (Claude→Anthropic). `normalize_model_name` vereinheitlicht.
- `/consensus` braucht **mind. 2** nicht-ausgeschlossene Antworten.
- `*-Pro`-Consensus-Engines und Premium-Modelle sind Pro-gated.

---

## 6. Datenhaltung / Firebase / Konfiguration

**Firestore-Collections** (verifiziert über Code):
- `users/{uid}` — `tier`, `role`; Subcollections `bookmarks`, `counters` sowie
  die produktive run-basierte Usage:
  - `usage_days/{YYYY-MM-DD}` — UTC-Tagesaggregat mit Schema-Version und den
    Integer-Zählern `total_reserved`, `total_consumed`,
    `deep_think_reserved`, `deep_think_consumed`. Reservierte und verbrauchte
    Slots zählen gegen das jeweilige Tageslimit; jeder Run belegt das Total,
    Deep Think zusätzlich den Deep-Bucket. Je Bucket gilt `remaining = limit -
    reserved - consumed` (mindestens 0).
  - `usage_runs/{sha256(idempotency_key)}` — idempotenter Run je UID + Key; der
    Klartext-Key wird nicht gespeichert. Enthält `kind=regular|deep_think`, den
    UTC-Tag der Reservierung, beide serverseitigen Limits zum
    Reservierungszeitpunkt und
    `status=reserved|consumed|released`. Erlaubte Übergänge:
    `reserved → consumed` (der erste begonnene Provider-Aufruf kostet genau
    eine Run-Einheit) oder
    `reserved → released` (fehlgeschlagener/abgebrochener Run gibt den Slot
    frei); beide Zielzustände sind terminal, Wiederholungen idempotent. Der Key
    kann nicht für einen anderen Run-Typ wiederverwendet werden. Provider-/LLM-
    Aufrufe finden immer außerhalb der Transaktion zwischen Reservierung und
    Abschluss statt. Beim Account-Löschen werden beide Subcollections entfernt.
  - `api_consensus_idempotency/{sha256(idempotency_key)}` — Mapping von UID +
    gehashtem HTTP-Idempotency-Key auf `run_id` und kanonischen `request_hash`;
    verhindert auch bei parallelen POSTs doppelte Runs. Kein Klartext-Key.
- `api_consensus_keys/{sha256(api_key)}` — admin-ausgegebene API-Schlüssel mit
  `uid`, nicht-geheimem Präfix/Label, `status=active|revoked`, den Scopes
  `consensus:run|share:write|share:index` und Audit-Zeitstempeln. Der
  Klartextschlüssel wird nie gespeichert.
- `api_consensus_account_blocks/{uid}` — temporärer fail-closed Tombstone bei
  Account-Löschung (`blocked`, `cleanup_pending`, Fehler-/Audit-Zeitstempel).
  Er wird vor jeder Löschkaskade geschrieben, von HTTP und Worker geprüft und
  nach erfolgreichem Cleanup plus Firebase-Löschung entfernt; transiente
  Cleanup-Fehler werden periodisch wiederholt.
- `api_consensus_runs/{run_id}` — UID-gebundener v1-API-Run mit serverseitig
  eingefrorenem Request/Modellplan, `idempotency_hash`, Status und Status-
  Zeitstempeln, einstündigem Running-Lease sowie terminal `result` oder
  sanitisiertem `error` und 30-Tage-`expires_at`. Erlaubte Hauptfolge:
  `accepted → reserved → running → succeeded|failed`.
- `app_config/models` — von `load_models_from_db()` gelesen/erzeugt: erlaubte
  Modelle pro Provider, `premium`, `consensus`, `preset_models`, `deep_think_model`,
  `judge_models`, `limits`.
  **Single Source of Truth für Limits/Modelle in Produktion** (überschreibt die
  `config.py`-Defaults beim Startup). `consensus` steuert den App-Consensus-Picker;
  Fehlende Limitfelder werden beim Startup normalisiert und per Merge in das
  Admin-Dokument zurückgeschrieben (Schema-Backfill ohne Verlust vorhandener Werte).
  Werte können historische Engine-Aliase (`Gemini-Pro`) oder direkte Modell-IDs aus
  den Provider-Listen sein. In `/admin` können Provider-Modelle per `Consensus`-
  Checkbox in diese Liste aufgenommen werden. `deep_think_model` ist die
  Consensus-Engine, auf die Deep Think umschaltet (`apply_deep_think_model`,
  Fallback Gemini 3.5 Flash; ans Frontend via `window.DEEP_THINK_CONSENSUS_MODEL`).
  `judge_models`/`judge_models_pro` setzen Standard- bzw. Pro-Differences-/
  Resolve-Judge je Provider (`apply_judge_models`/`apply_pro_judge_models` in
  config.py, in-place — consensus_engine/resolve_engine aliasen dieselben
  dicts; Frontier-Low-IDs sind ausgeschlossen; Fallbacks: günstiges
  Provider-Default-Modell bzw. API-Modell des `<Familie>-Pro`-Alias; Pro-Judges
  laufen unverändert mit effort=low). `judge_families` mappt Engine-Familie →
  bevorzugte Judge-Familie (`apply_judge_families`; nie die eigene Familie,
  ohne Eintrag/Key Auto über `JUDGE_FAMILY_PRIORITY`).
- `pending_results` — kurzlebige Consensus-Ergebnisse fürs Sharing (TTL/Cleanup).
- `shares` — unveränderliche Snapshots (Slug, `visibility=public|private`,
  `indexed`, `status`, `owner_uid`, `question_hash`, optional interne
  `source_api_run_id` und Index-Review-Auditfelder, …). Public-Shares sind per
  Link lesbar; private Watch-Snapshots ausschließlich mit Eigentümer-Session.
- `watches` — owner-gebundene Scheduling-Metadaten (`share_id`, `visibility`,
  Intervall, optionaler `run_weekday` für Weekly sowie lokale `run_time`
  (`HH:MM`) + IANA-`timezone`,
  `email_mode` = `changes_only|condition|every_run`, private
  `condition`, `last_condition_status`, Status, nächste Ausführung,
  Lease/Fehlerzähler); keine IP-/User-Agent-Daten. Conditions werden nie in
  öffentliche Share-Payloads oder History-Punkte kopiert.
  Verlaufspunkte liegen datenminimiert in `shares/{id}/watch_history` und
  verändern den Share-Snapshot nicht. Neben Score/Change-Metadaten können sie
  eine kompakte `opinion_map` tragen: maximal vier aus der strukturierten
  Differences-Analyse abgeleitete Dimensionen, kurze Standpunkte,
  Provider-Gruppen und einen 0–100 `shift_score`; niemals Rohantworten.
- `watch_runtime` — globaler Worker-Lease und datumsgebundener Tageszähler;
  verhindert parallele Scheduler-Worker und begrenzt Watch-Versuche restartfest.
- `watch_briefs/{uid}` — user-level Morning-Brief-Einstellungen (`enabled`,
  `send_time` `HH:MM`, IANA-`timezone`, `mode` = `always|changes_only`,
  `next_send_at`, `last_evaluated_at`, `last_sent_at`, `enabled_at`). Reine
  Aggregation vorhandener Watch-/History-Daten — keine LLM-Calls, daher nicht
  Pro-gated.
- `benchmark_runs` — admin-only Benchmark-Dashboard-Snapshots aus lokalen Runs:
  `manifest`, `results`, `audits`, abgeleitete Fragenmatrix; **keine**
  `calls.jsonl`-Rohantworten, Prompts oder Request-Payloads.
- `differences_stats` — anonyme Differences-Telemetrie (Schema v3): pro erfolgreichem
  Consensus-Lauf ein Dokument mit Zähl-/Strukturdaten (Agreement-Score,
  Widersprüche mit Severity und beteiligten Providern, Modell-Metadaten,
  seit v2 `judges`-Metadaten des tatsächlich genutzten Differences-Judges,
  seit v3 zusätzlich dessen erfolgreiche Attempt-Nummer und Versuchsdauer,
  `schema_version`) — **niemals** Frage-/Antwort-/Claim-Texte, Zitate, UID
  oder IP (anonym i. S. v. ErwGr. 26 DSGVO). Schema + Datenschutz-Regeln in
  `app/services/differences_stats.py`; geschrieben aus `chat.py::consensus`
  (fire-and-forget, Mock-Läufe schreiben nicht).
- `feedback`, `pro_waitlist`, `leaderboard`.

**Service-Account-JSONs** im Root (gitignored): `consensai-firebase-adminsdk-*.json`
(Firebase Admin) und `gen-lang-client-*.json` (Google ADC für Gemini, via
`GOOGLE_APPLICATION_CREDENTIALS`).

**Umgebungsvariablen** (`.env`, Beispiel in `.env.example`):
- Firebase Web-Config: `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`,
  `FIREBASE_PROJECT_ID`, `FIREBASE_STORAGE_BUCKET`, `FIREBASE_MESSAGING_SENDER_ID`,
  `FIREBASE_APP_ID` (ans Frontend durchgereicht via `/app`-Template).
- Developer-LLM-Keys (Fallback wenn Nutzer keine eigenen Keys hat):
  `DEVELOPER_OPENAI_API_KEY`, `DEVELOPER_MISTRAL_API_KEY`,
  `DEVELOPER_ANTHROPIC_API_KEY`, `DEVELOPER_GEMINI_API_KEY`,
  `DEVELOPER_DEEPSEEK_API_KEY`, `DEVELOPER_GROK_API_KEY`.
- Consensus-Watch-Mail/Abmeldung: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
  `SMTP_PASSWORD`, `MAIL_FROM`, `WATCH_UNSUBSCRIBE_SECRET`.

Modell-IDs/Tier-Zuordnung/Labels: ausschließlich in `app/core/config.py` pflegen
(`ALLOWED_*_MODELS`, `PREMIUM_MODELS`, `DEFAULT_MODEL_BY_PROVIDER`,
`FREE_DEFAULT_MODEL_BY_PROVIDER`, `EARLY_DEFAULT_MODEL_BY_PROVIDER`,
Frontier-Low-Mappings, `MODEL_LABEL_OVERRIDES`). Ebenfalls dort: die festen
Produkt-Metadaten `CONSENSUS_PRESET_DEFINITIONS` und die Basis-Model-Sets.
Firestore `preset_models` ueberschreibt pro Fast/Balanced/High Quality (ID:
`thorough`) die sechs
Antwortmodelle plus Consensus-Engine; Fast/Balanced bleiben Free-faehig und
High Quality bleibt unabhaengig von der Konfiguration Pro-only. Grok-Alt-Aliasse
(u. a. 4.1 Fast) werden beim Laden auf explizite interne Grok-4.3-Varianten
migriert: `grok-4.3-no-reasoning` sendet API-Modell `grok-4.3` mit
`reasoning.effort=none` und bleibt Free; `grok-4.3-low-reasoning` nutzt `low` und
bleibt Early, waehrend das Pro-Modell `grok-4.3` explizit `high` nutzt. Nur
aktuell konfigurierte Preset-Modelle werden in Provider-Listen
gesichert, sodass ersetzte Basiswerte im Admin danach entfernt werden koennen.

Early-Gating: `EARLY_MODELS` (Frontier-Low + DeepSeek V4 Pro) sind tag-gated, nicht
mehr gratis. Zugang via `is_user_early(uid)` (Firestore-Feld `early`/`tier=='early'`);
Pro schließt Early ein (Kombination `is_user_pro or is_user_early` an den Aufrufstellen,
`validate_model(..., is_early=...)`, `is_early_consensus_model`). Nicht-Early-Nutzer
defaulten auf die günstigen Basis-Modelle. Mistral Small ist bewusst KEIN Early-Modell.

Admin-Modellkonfig (`/admin`, `app_config/models` in Firestore): Provider-Listen sind
geordnet (Picker-Reihenfolge via `MODEL_ORDER_BY_PROVIDER`/`get_ordered_models`, im Admin
per ↑/↓ sortierbar); Feld `defaults` setzt den Free-Default je Provider (`apply_default_models`,
nur Nicht-Premium/Nicht-Early erlaubt, sonst `_BASE_FREE_DEFAULTS`). Feld `watch_models`
enthält getrennte `free`-/`pro`-Mappings Provider→Modell; je Tier sind mindestens zwei
Provider nötig, Free wird serverseitig auf Nicht-Premium/Nicht-Early begrenzt.
`normalize_models_document` erhält die Reihenfolge (kein `sorted` mehr) und validiert
`defaults`, `preset_models`, `watch_models` + `deep_think_model`.
Das Admin-UI (Tabs: Models / Consensus & Deep Think / Limits / API / Shared Pages /
Consensus Watch) bekommt via
`GET /api/admin/models` ein `meta`-Objekt (Alias-Auflösung, server-erzwungene Modelle je
Provider, Early-Set, Labels), mit dem Required-/Early-Badges gerendert werden — die
ensure/drop-Logik des Servers ist damit im UI sichtbar statt implizit. Der
„API“-Tab gibt Schlüssel für eine bestehende Firebase-UID aus, zeigt den
Klartextschlüssel genau einmal zum Kopieren und listet/widerruft danach nur
Hash-ID, Präfix, Label, UID, Status und Audit-Zeitstempel. Der separate
„Consensus Watch“-Tab zeigt die Free-/Pro-Watch-Modellmatrix, operative Watch-Metadaten,
SMTP-Konfigurationsstatus und admin-only Aktionen für eine echte Testmail sowie den sofortigen Start einer aktiven Watch;
der eigentliche Lauf bleibt im normalen Lease-/Budget-/Scheduler-Pfad. E2E-Zugriff auf
Admin-Endpunkte: `MOCK_ADMIN=1` (wirkt nur zusammen mit `MOCK_AUTH=1`).

---

## 7. Tests, Smoke-Checks & lokale Befehle

- **Backend-Tests** (`tests/`, pytest): `test_attachments`, `test_streaming`,
  `test_share_feature`, `test_differences_schema`, `test_frontier_model_payloads`,
  `test_rate_limit`, `test_seo_basics`. Lauf:
  ```powershell
  .\venv\Scripts\python.exe -m pytest tests
  ```
  Letzte bekannte Baseline: **572 passed** (2026-07-18; inklusive
  run-basierter Usage-, Consensus-API-Publishing-/Scope-/Vertrags- sowie
  Scheduled-Publisher-Tests).
- **Playwright-Smoke-Suite** (`tests/e2e/`, npm-frei via Python-Playwright):
  automatisiert die risikoreichsten Punkte der `docs/smoke-checklist.md`
  (Laden ohne Konsolen-Fehler, Send→Streaming, kompakte Antwort→Consensus-
  Pipeline inkl. Mobile-Clipping/Ergebnis-Reihenfolge,
  Consensus→Differences+Score, Watch-Dialog mit Pflicht-Sichtbarkeit/Condition-
  Feld, Exclude, Theme, Picker-Persistenz). Startet einen eigenen uvicorn auf Port
  8031 mit `MOCK_LLM=1` (deterministische Fixtures in
  `app/services/llm/mock_llm.py`, Seams: `_run_ask`,
  `_call_engine_text`/`_stream_engine_text`),
  `MOCK_AUTH=1` (Sentinel-Token statt Firebase, Browser-Stub ersetzt
  `firebase.js` per Playwright-Route), lokale Dummy-Eigenkeys (kein Einfluss
  des live geladenen Free-Limits; `MOCK_LLM` verhindert echte Calls) und
  `DISABLE_RATE_LIMIT=1`. Lauf:
  ```powershell
  $env:RUN_E2E = "1"; .\venv\Scripts\python.exe -m pytest tests\e2e -v
  ```
  Ohne `RUN_E2E=1` wird `tests/e2e` nicht eingesammelt (Baseline bleibt).
  Nur lokal (braucht Service-Account-JSON + Netz für CDN); Details/Setup in
  `tests/e2e/README.md`.
- **Frontend darüber hinaus ohne automatisierte Tests.** Nach JS-Änderungen
  an nicht abgedeckten Flows (Resolve, Share, Attachments, Follow-up,
  Bookmarks, Agent Mode, Demo, Mobile) die manuelle
  **`docs/smoke-checklist.md`** durchgehen.
- **Benchmark-Runner** (`benchmark/`, kein GUI-Pfad): `python -m benchmark
  --smoke|--pilot|--final`; fertige lokale Runs werden mit
  `python -m benchmark --publish-run <run_id>` oder `--publish-all` als kompakte
  Admin-Dashboard-Snapshots nach Firestore publiziert. `--smoke` ist ein
  dedizierter 1-Frage-MMLU-Pro-Pfad
  mit eigenem Manifest/Run-Kontext; Smoke und Pilot haben separate Live-Gates,
  der finale Run bleibt durch `LIVE_EXECUTION_ENABLED` hart gegatet. Die MC-
  Auswertung akzeptiert nur die letzte `FINAL_ANSWER: X`-Zeile.
- JS-Syntaxcheck einzelner Module:
  ```powershell
  node --check static\js\<modul>.js
  ```
- App lokal starten (Render nutzt eigenes Start-Kommando ohne `--proxy-headers`):
  ```powershell
  .\venv\Scripts\python.exe -m uvicorn main:app --reload
  ```

**Cleanup-Jobs**: Share-Cleanups laufen im `lifespan`-Startup von `main.py`
(zusätzlich durch den täglichen Render-Restart getriggert):
`cleanup_expired_pending`, `cleanup_revoked_shares`. Consensus-API-Retention,
Lease-/Queue-Recovery laufen zusätzlich alle 60 Sekunden; fehlgeschlagene
API-Account-Löschkaskaden alle fünf Minuten.

**Consensus Watch** läuft als eigener asyncio-Lifespan-Task alle 30 Minuten.
Firestore-Transaktionen claimen einen globalen Worker-Lease, den einzelnen
Watch-Lease und das globale Tagesbudget; innerhalb eines Workers laufen Watches
strikt sequenziell. Die Reruns ermitteln den aktuellen Pro-Status des Eigentümers und
nutzen das entsprechende `WATCH_MODELS_BY_TIER`-Mapping aus Firestore `watch_models`;
je konfiguriertem Provider läuft genau ein Modell (mindestens zwei), deren Antwort-Calls
laufen innerhalb des einzelnen Watch-Runs parallel. Keine Attachments/Follow-ups und keine
In-Memory-Usage-Zähler. Jeder erfolgreiche Lauf schreibt genau einen kompakten
History-Punkt; nach drei Fehlern pausiert die Watch.
Der Mailmodus ist pro Watch änderbar: `changes_only` nutzt die bestehende
Major-/Score-Delta-Schwelle, `condition` lässt den bestehenden Change-Judge eine
max. 500 Zeichen lange Nutzerbedingung gegen den neuen Consensus als
`met|not_met|unknown` bewerten und mailt nur beim Übergang zu `met`, `every_run`
sendet nach jedem erfolgreichen Lauf genau eine Multipart-Mail inklusive neuem
Consensus-Text. Bei der Erstellung ist `visibility=private|public` Pflicht im UI:
fehlende Pflichtwerte werden direkt am jeweiligen Feld angezeigt; der mobile
Create-Dialog bleibt innerhalb des dynamischen Viewports und scrollt intern.
private Seiten erfordern die kurzlebige Eigentümer-Session, sind `noindex,nofollow`,
`private,no-store` und erscheinen weder in Sitemap/Related noch im Report-Flow.
Neu angelegte Watches bekommen eine lokale Ausführungszeit; das Backend berechnet
`next_run_at` zeitzonen- und DST-fest und behält die lokale Uhrzeit bei Folge-Runs,
Fehler-Retries und Resume bei. Weekly-Watches können einen lokalen Wochentag wählen;
Legacy-Watches ohne Wochentag bzw. Zeitfelder nutzen weiter die bisherige reine
Intervalladdition.
Watch-Seiten zeigen für aktuelle oder historische Watches eine kompakte Metazeile
mit Intervall sowie letztem und ggf. nächstem Fragenlauf.
Die öffentliche Watch-History rendert zusätzlich eine **Position Map**: statt
einer universellen Ja/Nein-Achse zeigt sie frage-spezifische Standpunkt-
Dimensionen, Provider-Bewegungen über die Läufe und den gemeinsamen
**Direction Shift**. Die Berechnung ist deterministisch aus dem ohnehin
vorhandenen Differences-JSON und verursacht keinen zusätzlichen LLM-Call.
**Morning Brief**: opt-in tägliche Digest-Mail pro Nutzer (nicht pro Watch),
konfiguriert im Watch-Dashboard (`/api/my/watch-brief`), gespeichert in
`watch_briefs/{uid}`; Aktivierung setzt mindestens eine vorhandene Watch voraus.
Der 30-Minuten-Loop ruft nach `run_watch_tick` ein
`run_brief_tick` auf: fällige Briefs werden transaktional geclaimt (Zeitplan
rückt VOR dem Versand vor — at-most-once, nie doppelt), dann wird der Digest
aus `list_watches(include_history=True)` aggregiert (Score/Delta, notable
Changes seit dem letzten Brief = changed-Flag oder Score-Sprung ≥15) und als
Multipart-Mail versendet. Modus `changes_only` überspringt Briefs ohne notable
Changes. Kein LLM-Call, kein Watch-Lease nötig; unverifizierte E-Mail-Adressen
werden übersprungen. `/watch/brief/unsubscribe` (eigener HMAC-Token-Typ,
gleicher `WATCH_UNSUBSCRIBE_SECRET`) deaktiviert nur den Brief.
Im Admin-Dashboard kann eine aktive Watch fällig gestellt und der In-Process-Scheduler
sofort aufgeweckt werden; der HTTP-Request wartet nicht auf die Modellaufrufe.
Der manuelle Lauf verbraucht reale Modellaufrufe, schreibt reguläre History, rückt den Zeitplan vor
und wendet unverändert die konfigurierte Mailregel an. Der unabhängige SMTP-Test führt
keinen Watch-Lauf aus und ändert keinen Zeitplan.

---

## 8. Kritische Verträge & Stolperfallen

- **Script-Ladereihenfolge in `templates/index.html` ist ein Vertrag.**
  `app-core.js` definiert `window.App` und muss vor allen Feature-Modulen laufen;
  `app-init.js` läuft als letztes (deferred am `</body>`) und verdrahtet das DOM.
  Reihenfolge umstellen oder ein Modul rausnehmen ⇒ `ReferenceError` /
  `window.X is not a function`.
- **`window.*` ist die Modul-Schnittstelle.** Es gibt keine ES-Imports zwischen den
  Feature-Modulen. Wer eine Funktion umbenennt/verschiebt, muss alle `window.`-
  Aufrufstellen mitziehen (Grep über `static/js/` + `static/firebase.js` +
  `static/demo.js`). Wichtige Globals u. a.: `window.sendQuestion`,
  `window.getConsensus`, `window.canGenerateConsensus`,
  `window.updateConsensusButtonAvailability`, `window.revealConsensusOutput` /
  `hideConsensusOutput`, `window.cancelCurrentConsensus`, `window.openShareDialog`,
  `window.openWatchDialog`, `window.openWatchDashboard`,
  `window.currentEvidenceSources`, `window.consensusCitationMeta`,
  `window.lastShareResultId`, `window.currentBookmarkShareResultContext`,
  `window.currentBookmarkShareResultPromise`,
  `window.resolveCurrentShareResultId`, `window.clearPreparedBookmarkShareResult`,
  `window.isUserPro`, `window.pendingAttachments`.
- **`window.App.followup`** (definiert in `consensus-run.js`) ist der
  Follow-up-Kontext-State (`offer/arm/discard/consume/reset/render`).
  `query-send.js` (consume beim Senden), `app-init.js` (reset in
  `clearResponseBoxes`) und `user-tier.js` (render bei Tier-Wechsel) hängen
  daran; DOM-Ziel ist `#followupBar` in `index.html`.
- **`window.App.setAppTitle(question?)`** (definiert in `app-core.js`) hält den
  Standard- bzw. fragebezogenen Browser-Tab-Titel bei Query-Send, Bookmark-Open
  und Clear synchron zur aktuellen Ansicht.
- **`window.App.consensusLifecycle.*`** ist die gezielte Run-State-Brücke
  (`startRun/isActiveRun/finishRun/setSynthesizing/isRunning/setGate/
  markPendingCanceled/initAutoConsensusToggle`). Run-ID-Gating nicht umgehen, sonst
  rendern alte Läufe in neue.
- **`window.App.watch.showFeatureNudge()`** wird nach einem erfolgreichen
  Consensus-Final aufgerufen und zeigt den einmaligen, lokal dismissbaren
  Consensus-Watch-Hinweis nur für eingeloggte Nutzer mit `result_id`.
- **`window.App.sharedModal.open(mode)` / `.close()`** koordinieren den gemeinsam
  genutzten `#shareModal` für Share und Watch einschließlich Modusklasse,
  Background-Scroll-Lock und Rückgabe des Fokus an den Auslöser.
- **DOM-als-State**: `dataset.consensusAnswer`, `dataset.consensusSources`,
  `dataset.responseState`, `.excluded`-Klassen und die session-lokale
  `.agent-mode-show-answers`-Body-Klasse u. a. sind echte State-Quellen.
  Vorsicht beim Umbauen von
  Markup — der State-Refactor ist bewusst noch nicht passiert.
- **Jinja↔JS-Brücke**: Config geht nur über den `<head>`-`window.*`-Block
  (`FIREBASE_CONFIG`, `APP_LIMITS`, `FREE_DEFAULT_MODELS`, `PRO_DEFAULT_MODELS`,
  `CONSENSUS_PRESETS` inklusive Antwort-/Consensus-Model-Sets,
  `DEFAULT_CONSENSUS_PRESET`, `FREE_LIMIT`) oder
  serverseitig gerenderte Template-Optionen wie
  `consensus_models` für den Consensus-Picker. `app-init.js` kann kein Jinja
  rendern — neue Server-Werte müssen hier gebridged werden.
- **CSP** (`CustomSecurityMiddleware` in `security.py`): neue externe Hosts (Skripte,
  `connect-src`-Ziele, Frames) müssen explizit in die Policy. Sonst blockt der
  Browser still.
- **Static-Caching / `?v=`**: Nach CSS/JS-Änderungen den `?v=`-Query-String in
  `index.html` (und für CSS in `style.css`/`index.html`) bumpen — sonst wird Stale
  ausgeliefert. (Siehe Memory „CSS cache-busting".)
- **Provider-Label-Konvention**: Frontend nutzt teils `Claude`, Backend kanonisch
  `Anthropic`. Beim Verdrahten neuer Modelle Mapping in `app-core.js::modelPrefs`
  und Backend-`normalize_model_name` synchron halten.
- **Usage-Key ist ein Backend-/Frontend-Vertrag.** Ein frischer logischer Lauf
  nutzt denselben `usage_run_key` in `/prepare`, allen `/ask_*` und
  `/consensus`; Resolve nutzt einen eigenen Key. Run-Typ (`regular` oder
  `deep_think`) und Limits werden ausschließlich serverseitig bestimmt. Niemals
  clientseitige Kosten, Modellanzahl oder Float-Inkremente übernehmen; niemals
  Provider-Aufrufe in die Firestore-Transaktionsfunktion verschieben.
- **Datenminimierung ist Designentscheidung**: keine IP-/User-Agent-Speicherung,
  keine Datei-Bytes in Firestore. Nicht „aus Versehen" mitloggen.

---

## 9. Bei Änderungen aktualisieren

Diese Datei ist die zentrale Architektur-Karte. **Aktualisiere sie im selben
Commit/PR**, wenn sich Folgendes ändert:

- **Architektur/Module**: neues/entferntes/umbenanntes JS-Modul oder Backend-
  Service; geänderte Ladereihenfolge oder Modul-Verantwortlichkeit (§2, §3, §5).
- **API**: neuer/entfernter/umbenannter Endpoint oder geändertes Request/Response-
  bzw. SSE-Format (§2, §4).
- **Flows**: Änderung an Query-Fan-out, Consensus/Differences, Agent Mode,
  Attachments, Auth/Usage oder Sharing (§4).
- **Verträge**: neue/entfernte `window.*`- bzw. `window.App.*`-Schnittstelle, neues
  DOM-Dataset-als-State, neue Jinja↔JS-Brücke, CSP-Erweiterung (§8).
- **Daten/Config**: neue Firestore-Collection/-Feld, neue Umgebungsvariable,
  geänderte Limit-/Modell-Quelle (§6).
- **Cache-Busting (immer, auch bei Kleinständerungen)**: Nach **jeder** Änderung
  an Dateien unter `static/` — egal ob großer Umbau oder Einzeiler — muss der
  `?v=`-Query-String der betroffenen Datei gebumpt werden: für CSS-Module in
  `static/style.css` (@import-Zeilen) **und** den `style.css`-Link in
  `templates/index.html`, für JS die `<script>`-Tags in `templates/index.html`.
  Konvention: `?v=YYYYMMDD-kurzlabel`. Ohne Bump liefern Browser/CDN die alte
  Datei aus und die Änderung ist in Produktion unsichtbar (§8).

Faustregel: Wenn ein neuer Agent durch deine Änderung an einer der obigen Stellen
**überrascht** würde, gehört es hier rein. Kurz halten — verifizierte Fakten statt
Implementierungsdetails. Bei Detailtiefe lieber auf den Code verweisen.
