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
Im `lifespan`-Startup: `load_models_from_db()` + Share-Cleanups (siehe §7).

Router liegen unter `app/api/routers/` und werden in `main.py` eingebunden:

| Router | Zweck (Auswahl an Pfaden) |
|---|---|
| `pages.py` | HTML-Seiten + SEO: `/` (Landing, redirect→`/app` bei Session), `/app` (Haupt-App), `/admin`, `/admin/benchmark` (Benchmark-Run-Visualisierung), `/about`, `/ai-model-comparison`, `/privacy` `/imprint` `/terms`, `robots.txt`, `sitemap*.xml`. Außerdem `/feedback`, `/vote`, `/check_keys` (nur verifizierte Logins zum Testen eigener Keys). |
| `chat.py` | Kern-LLM-Flow: `/prepare`, `/ask_openai` `/ask_mistral` `/ask_claude` `/ask_gemini` `/ask_deepseek` `/ask_grok`, `/consensus`, `/resolve`. Die sechs `/ask_*`-Endpoints sind dünne Wrapper um `handle_ask` + die deklarative Provider-Registry `ASK_PROVIDERS` (Provider-Eigenheiten wie Gemini-Service-Account, `gemini_key`-Legacy-Feld, `useOwnKeys`-Flag und Env-Key-Namen stehen dort, Rate-Limits als Literal am Endpoint). |
| `auth.py` | `/register`, `/confirm-registration`. |
| `users.py` | `/user_status`, `/usage`, `/delete_account`, `/track-interest`. |
| `bookmarks.py` | `/bookmarks` (GET), `/bookmark` (POST/DELETE), `/bookmark/consensus`. |
| `share.py` | `/api/share` (POST), `/api/share/{id}` (DELETE), `/api/my/shares`, `/api/share/{id}/report`, öffentliche Seite `/s/{slug_id}`, `sitemap-shares.xml`. |
| `admin.py` | `/api/admin/shares`, `/api/admin/shares/{id}/moderate`, `/api/admin/models` (GET/POST), `/api/admin/benchmark/runs` (Liste) + `/api/admin/benchmark/runs/{run_id}` (Detail, liest Firestore-publizierte kompakte Benchmark-Reports mit lokalem Disk-Fallback über `benchmark/report_reader.py`). Alle hinter `is_user_admin`. |

**Zentrale Templates** (`templates/`, gerendert mit `Jinja2Templates`):
`landing.html` (Marketing), `index.html` (die App — Haupt-Markup + Script-Tags),
`admin.html`, `admin_benchmark.html` (Admin-Benchmark-Visualisierung, eigenes
Template + Firebase-Auth-Modul wie `admin.html`), `share.html` (öffentliche
Consensus-Seite), `share_unavailable.html`, plus statische Rechts-/SEO-Seiten. **`index.html` enthält kein App-JS inline mehr**
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
  (`getModelOptionLabel`, `getSelectedModelCount`, `showPopup`, `trackAppEvent`).
- **`model-picker.js`** — Modellauswahl/Custom-Picker, Default-Modelle, localStorage-
  Persistenz (`restoreModelSelections`).
- **`markdown-stream.js`** — Markdown-Rendering (`injectMarkdown`) + SSE-Helfer
  (`createStreamRenderer`, `streamSSERequest`).
- **`sources.js`** — Quellen/Evidence-Mapping; nutzt DOM-Datasets
  `dataset.consensusAnswer` / `dataset.consensusSources`; `window.currentEvidenceSources`.
- **`attachments.js`** — Attachment-UI/Payload (Pro); `window.pendingAttachments`,
  `getAttachmentsPayload`.
- **`agent-mode.js`** — Agent-Mode-UI/Status/Timer; zeigt pro Modell den
  Query-Abschluss aus `dataset.responseState`; einzige Stelle, die den
  Auto-Consensus-Toggle erzwingt/sperrt.
- **`consensus-lifecycle.js`** — Consensus-Sichtbarkeit, Gate/Availability,
  Run-State, Abort/Cancel, Run-ID-Gating, Auto-Consensus-Persistenz. Exponiert die
  `window.App.consensusLifecycle.*`-Brücke (siehe §4/§8).
- **`share-dialog.js`** — `window.openShareDialog` und Share-Liste.
- **`consensus-actions.js`** — Copy/Citation/Share-Buttons am Consensus.
- **`user-tier.js`** — Free/Pro-UI, Premium-Modellstatus (`updateUserTierUI`,
  `updatePremiumModelsState`).
- **`consensus-insights.js`** — strukturierte Auswertung: Claim-Badges,
  Difference-Karten, Credibility-Frame-Farben, Jump-to-answer, Spalten-Balancer,
  Resolve-Runde (Button an Widerspruchs-Karten → `POST /resolve`).
- **`consensus-run.js`** — `window.getConsensus`: baut `/consensus`-Payload, fährt
  den SSE-Stream, rendert Ergebnis + Citation/Share-Meta. `parseBestModel`.
- **`query-send.js`** — `window.sendQuestion`: `/prepare` + `/ask_*`-Fan-out,
  Streaming-Rendering, Usage/Tier-UI, Auto-Consensus-Trigger, Query-Run-State
  (`isQueryRequestRunning`, `cancelCurrentQuery`).
- **`app-init.js`** — das gesamte `initApp()`: Theme, Usage/Limits + User-Status,
  Response-Box-Toggles, Sidebar/Layout, Modals, Tooltips, Evidence-Rendering,
  API-Key-Test. Läuft als letztes Script, ruft `initApp()` direkt auf.

**Nicht unter `static/js/`** (älter, eigene Verantwortung):
- **`static/firebase.js`** (ES-Modul) — Firebase-Init, Login/Logout, Token-Handling,
  `window.auth`, Bookmarks-CRUD-Calls, Feedback, Voting, Tier-Sync.
- **`static/demo.js`** (ES-Modul) — Demo-Flow (`runDemoFlow`) für die „Demo"-Query.
- **`static/app-ui.js`** — System-Prompt-/Help-Modal + App-Width-Resizer.

**Abhängigkeitsrichtung**: `app-core.js` → Feature-Module → `app-init.js`. Module
kommunizieren über `window.*`-Globals und `window.App`, **nicht** über Imports. DOM
dient vielerorts als State (z. B. `.excluded`-Klasse, Datasets) — bewusster
Übergangszustand, noch nicht aufgelöst.

---

## 4. Kern-Flows

### Anfrage an Modelle (Streaming)
1. Frontend `sendQuestion` (`query-send.js`) ruft zuerst **`POST /prepare`**:
   Auth + Usage-Pre-Check; `get_intent_from_llm` (aus `tool_heuristics.py`) erkennt
   Intent; bei `weather/stock/crypto` wird via `get_realtime_context` Echtzeitdaten
   in den System-Prompt injiziert. Antwort: finaler `system_prompt`.
2. Fan-out an die ausgewählten **`/ask_<provider>`**-Endpoints (parallel), je mit
   `stream:true`. Backend prüft Auth, Pro-Status, Deep-Search-Berechtigung,
   Wortlimit (`validate_question_word_limit`) und Modell (`validate_model`),
   parst Attachments, zählt Usage hoch (`active_count` teilt den Increment:
   `1/active_count`). Eigene Provider-Keys dürfen nur verifizierte Nutzer
   verwenden; sie umgehen die Free-Usage-Zählung, aber nicht Auth/Pro-Gates.
3. **SSE-Protokoll Modellantwort** (`streaming_model_response` in `streaming.py`):
   `event: delta {text}` … dann `event: final {response, sources,
   free_usage_remaining, deep_remaining, is_pro_user, key_used}`. Bei Fehler kommt
   ein `final` mit `error`. Frontend rendert deltas und wertet `final` aus.

### Consensus & Differences
- `getConsensus` (`consensus-run.js`) sammelt die vorhandenen Modellantworten +
  `excluded_models` + `consensus_model` und ruft **`POST /consensus`**
  (`stream:true`).
- Backend (`chat.py::consensus` → `consensus_engine.py`): validiert (mind. **2**
  eingeschlossene Antworten), kappt Frage/Antworten serverseitig
  (`cap_engine_text`, Limits `consensus_max_answer_chars` /
  `consensus_max_question_chars` — Kosten-/Abuse-Schutz, da die Texte vom
  Client kommen), prüft Engine-Keys, dann `stream_consensus` gefolgt von
  `stream_differences`. **SSE-Events**: `consensus.delta`, `differences.delta`
  (nur Keep-Alive, Frontend rendert sie nicht), dann `final {consensus_response,
  differences, differences_data, result_id?, …usage}`. `differences_data` ist
  strukturiertes JSON (Verdict, Karten, `best_model`, `models_compared`).
- Robustheit Differences (`consensus_engine.py`): einheitlicher Engine-Dispatch
  (`_resolve_engine`/`_call_engine_text`/`_stream_engine_text`), Structured
  Output je Provider (json_object / responseMimeType / Anthropic-Prefill),
  Judge läuft immer auf dem günstigen Default-Modell des gewählten Providers
  (`DIFFERENCES_JUDGE_MODEL_BY_PROVIDER`), JSON-Truncation-Repair, 1 Retry +
  Fallback-Judge auf anderem Provider mit Key, serverseitige Anchor-/Quote-
  Verifikation gegen Konsens- bzw. Modellantworten (nicht belegbare Zitate
  werden geleert). Unparsbares JSON erreicht den Nutzer nie als Rohtext.
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
kostet 1 regulären Usage-Punkt (außer `useOwnKeys`), Eingaben werden wie bei
`/consensus` serverseitig gekappt (`normalize_resolve_positions`), Ergebnis
wird **nicht** persistiert. Frontend: „Resolve with the models"-Button an
Contradiction-Karten in `consensus-insights.js` (nur bei ≥2 beteiligten
Modellen).

### Agent Mode
`agent-mode.js` koppelt Auto-Consensus: nach Abschluss aller Modellantworten löst
`query-send.js` automatisch `getConsensus` aus. Run-State/Gating läuft über
`consensus-lifecycle.js` (`startRun()→{runId,signal}`, `isActiveRun`, `finishRun`,
`setSynthesizing`, `cancelCurrentConsensus`). Agent Mode ist die **einzige** Stelle,
die den Auto-Consensus-Toggle erzwingt/sperrt.

### Attachments (Pro)
Frontend `attachments.js` baut Payload; Backend `app/services/llm/attachments.py`
validiert: max **2** Dateien, je **5 MB**, MIMEs PDF/PNG/JPEG/WebP. Bild-Support:
openai/anthropic/gemini/grok; PDF-Support: openai/anthropic/gemini (sonst
Text-Fallback/PDF-Extraktion). **In Firestore landen nie Datei-Bytes**, nur
Metadaten (Name/Typ/Größe) — siehe `bookmarks.py::sanitize_attachment_meta`.

### Auth / Usage / Tier
- Firebase-ID-Token wird mit `verify_user_token` geprüft (Standard: nur
  E-Mail-verifizierte Nutzer; `allow_unverified=True` nur für Registrierung/Delete).
- Token-Quelle: `extract_id_token` liest Body `id_token`, sonst `Authorization:
  Bearer`, sonst Cookie `session`.
- Eigene API-Keys sind ein eingeloggtes Feature: `/check_keys`, `/ask_*` mit
  User-Key und `/consensus` mit `useOwnKeys` verlangen ein verifiziertes Token.
- Pro-Status: `is_user_pro` liest Firestore `users/{uid}.tier ∈ {premium, pro}`.
  Admin: `users/{uid}.role == admin`.
- **Usage-Zähler liegen In-Memory** (`app/core/state.py`: `usage_counter`,
  `deep_search_usage`, `registered_ips`, `last_feedback_time`) — kein Persistieren,
  Reset beim täglichen Render-Restart ist gewollt.
- Limits/Defaults kommen aus `app/core/config.py` (`get_usage_limit`,
  `get_word_limit`, `get_output_token_limit`, …) und können per Firestore
  (`app_config/models.limits`) überschrieben werden.

### Sharing
- `/consensus` legt ein `pending_results`-Dokument an → `result_id`.
- `POST /api/share` (`share.py` → `share_snapshots.create_share_from_pending`)
  macht daraus einen unveränderlichen Share-Snapshot (`shares`-Collection) mit Slug.
- Öffentliche Seite **`GET /s/{slug_id}`** rendert read-only aus dem Snapshot
  (keine LLM-Calls), inkl. JSON-LD, Canonical-Dedup über `question_hash`,
  „verwandte Fragen". **Indexierung (`index, follow`) nur wenn der Admin `indexed`
  setzt** — nie automatisch; sonst `noindex`. Caching via `SHARE_CACHE_CONTROL`
  + In-Process-Cache (`get_share_cached` / `invalidate_share_cache`).
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
  state.py                   In-Memory-Dicts (Usage, IPs, Feedback-Zeit)
app/api/routers/             siehe §2
app/services/llm/
  base.py                    System-Prompt, Wortzählung, validate_model
  engines.py                 Provider-Requests (build_provider_payload, query_*)
  streaming.py               SSE-Helfer, stream_*_query, streaming_model_response
  consensus_engine.py        query/stream_consensus + query/stream_differences, normalize_model_name
  resolve_engine.py          Resolve-Runde (run_resolve_round, normalize_resolve_positions)
  citations.py               Antwort-Parsing + Quellen (source_response, make_llm_result)
  attachments.py             Attachment-Validierung/Aufbereitung
app/services/
  share_snapshots.py         Snapshot-Lifecycle (pending→share), Quoten, Cleanups, Sitemap-Quellen
  public_markdown.py         Server-Markdown-Rendering für Share-Seiten
tool_heuristics.py           Intent-Erkennung + Realtime-Kontext (weather/stock/crypto)
```

Wichtige Verträge im Backend:
- Provider-Label-Set überall identisch: `OpenAI, Mistral, Anthropic, Gemini,
  DeepSeek, Grok` (Claude→Anthropic). `normalize_model_name` vereinheitlicht.
- `/consensus` braucht **mind. 2** nicht-ausgeschlossene Antworten.
- `*-Pro`-Consensus-Engines und Premium-Modelle sind Pro-gated.

---

## 6. Datenhaltung / Firebase / Konfiguration

**Firestore-Collections** (verifiziert über Code):
- `users/{uid}` — `tier`, `role`; Subcollections `bookmarks`, `counters`.
- `app_config/models` — von `load_models_from_db()` gelesen/erzeugt: erlaubte
  Modelle pro Provider, `premium`, `consensus`, `limits`. **Single Source of
  Truth für Limits/Modelle in Produktion** (überschreibt die `config.py`-Defaults
  beim Startup). `consensus` steuert den App-Consensus-Picker; Werte können
  historische Engine-Aliase (`Gemini-Pro`) oder direkte Modell-IDs aus den
  Provider-Listen sein. In `/admin` können Provider-Modelle per `Consensus`-
  Checkbox in diese Liste aufgenommen werden.
- `pending_results` — kurzlebige Consensus-Ergebnisse fürs Sharing (TTL/Cleanup).
- `shares` — veröffentlichte Snapshots (Slug, `indexed`, `status`, `owner_uid`,
  `question_hash`, …).
- `benchmark_runs` — admin-only Benchmark-Dashboard-Snapshots aus lokalen Runs:
  `manifest`, `results`, `audits`, abgeleitete Fragenmatrix; **keine**
  `calls.jsonl`-Rohantworten, Prompts oder Request-Payloads.
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

Modell-IDs/Tier-Zuordnung/Labels: ausschließlich in `app/core/config.py` pflegen
(`ALLOWED_*_MODELS`, `PREMIUM_MODELS`, `DEFAULT_MODEL_BY_PROVIDER`,
`FREE_DEFAULT_MODEL_BY_PROVIDER`, `EARLY_DEFAULT_MODEL_BY_PROVIDER`,
Frontier-Low-Mappings, `MODEL_LABEL_OVERRIDES`).

Early-Gating: `EARLY_MODELS` (Frontier-Low + DeepSeek V4 Pro) sind tag-gated, nicht
mehr gratis. Zugang via `is_user_early(uid)` (Firestore-Feld `early`/`tier=='early'`);
Pro schließt Early ein (Kombination `is_user_pro or is_user_early` an den Aufrufstellen,
`validate_model(..., is_early=...)`, `is_early_consensus_model`). Nicht-Early-Nutzer
defaulten auf die günstigen Basis-Modelle. Mistral Small ist bewusst KEIN Early-Modell.

Admin-Modellkonfig (`/admin`, `app_config/models` in Firestore): Provider-Listen sind
geordnet (Picker-Reihenfolge via `MODEL_ORDER_BY_PROVIDER`/`get_ordered_models`, im Admin
per ^/v sortierbar); Feld `defaults` setzt den Free-Default je Provider (`apply_default_models`,
nur Nicht-Premium/Nicht-Early erlaubt, sonst `_BASE_FREE_DEFAULTS`). `normalize_models_document`
erhält die Reihenfolge (kein `sorted` mehr) und validiert `defaults`.

---

## 7. Tests, Smoke-Checks & lokale Befehle

- **Backend-Tests** (`tests/`, pytest): `test_attachments`, `test_streaming`,
  `test_share_feature`, `test_differences_schema`, `test_frontier_model_payloads`,
  `test_rate_limit`, `test_seo_basics`. Lauf:
  ```powershell
  .\venv\Scripts\python.exe -m pytest tests
  ```
  Letzte bekannte Baseline: **145 passed**.
- **Frontend hat keine automatisierten Tests.** Nach JS-Änderungen die manuelle
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

**Cleanup-Jobs** laufen ohne eigenen Scheduler im `lifespan`-Startup von `main.py`
(getriggert durch den täglichen Render-Restart): `cleanup_expired_pending`,
`cleanup_revoked_shares`.

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
  `window.currentEvidenceSources`, `window.consensusCitationMeta`,
  `window.lastShareResultId`, `window.isUserPro`, `window.pendingAttachments`.
- **`window.App.consensusLifecycle.*`** ist die gezielte Run-State-Brücke
  (`startRun/isActiveRun/finishRun/setSynthesizing/isRunning/setGate/
  markPendingCanceled/initAutoConsensusToggle`). Run-ID-Gating nicht umgehen, sonst
  rendern alte Läufe in neue.
- **DOM-als-State**: `dataset.consensusAnswer`, `dataset.consensusSources`,
  `dataset.responseState`, `.excluded`-Klassen u. a. sind echte State-Quellen.
  Vorsicht beim Umbauen von
  Markup — der State-Refactor ist bewusst noch nicht passiert.
- **Jinja↔JS-Brücke**: Config geht nur über den `<head>`-`window.*`-Block
  (`FIREBASE_CONFIG`, `APP_LIMITS`, `FREE_DEFAULT_MODELS`, `PRO_DEFAULT_MODELS`,
  `FREE_LIMIT`) oder serverseitig gerenderte Template-Optionen wie
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
- **Usage ist In-Memory & nicht atomar** (`active_count`-Increment `1/n`).
  Beim Ändern der Limit-/Zähl-Logik alle `/ask_*` + `/consensus` + `/usage` +
  `/user_status` konsistent halten.
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

Faustregel: Wenn ein neuer Agent durch deine Änderung an einer der obigen Stellen
**überrascht** würde, gehört es hier rein. Kurz halten — verifizierte Fakten statt
Implementierungsdetails. Bei Detailtiefe lieber auf den Code verweisen.
