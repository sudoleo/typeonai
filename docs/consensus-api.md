# Consensus API v1

Der vollständige maschinenlesbare Vertrag wird von FastAPI unter
`/openapi.json` ausgeliefert. Die v1-Routen verwenden das OpenAPI-
Security-Scheme `ConsensusApiKey` (`X-API-Key`).

API-v1-Antworten werden mit `Cache-Control: private, no-store` ausgeliefert.
IP- und API-Key-bezogene Rate-Limits schützen Authentifizierung, Polling und
Worker-Kapazität; ein `429` bzw. `503` soll mit Backoff erneut versucht werden.

## Schlüssel ausgeben

Im Admin-Dashboard unter `/admin#api` können Schlüssel für bestehende Firebase-
UIDs ausgegeben, gefiltert und widerrufen werden. Nur ein Firebase-Admin kann
einen Schlüssel ausgeben; der zugrunde liegende HTTP-Aufruf lautet:

```http
POST /api/admin/api-keys
Authorization: Bearer <firebase-id-token>
Content-Type: application/json

{"uid":"firebase-uid","label":"production"}
```

Die Antwort enthält `api_key` genau einmal. Firestore speichert nur den
SHA-256-Hash als Dokument-ID in `api_consensus_keys`. Admins können Schlüssel
mit `GET /api/admin/api-keys?uid=…` auflisten und mit
`DELETE /api/admin/api-keys/{key_id}` widerrufen.
Schlüssel werden nur für aktive, E-Mail-verifizierte Firebase-Nutzer
ausgegeben. Gelöschte, deaktivierte oder lokal zur Löschung gesperrte Accounts
können keinen API-Schlüssel mehr verwenden.

Jeder Schlüssel trägt explizite Scopes:

- `consensus:run`: Runs starten, lesen und löschen.
- `share:write`: eigene erfolgreiche Runs publizieren sowie eigene Shares
  auflisten, lesen und widerrufen.
- `share:index`: eigene geeignete Shares direkt indexierbar bzw. wieder auf
  `noindex` setzen. Dieser Scope kann ausschließlich für eine Admin-UID
  ausgegeben werden und der Endpoint prüft die Admin-Rolle erneut.

Neue Schlüssel erhalten standardmäßig `consensus:run` und `share:write`.
Legacy-Schlüssel ohne gespeichertes Scope-Feld erhalten dieselben sicheren
Defaults, aber niemals rückwirkend `share:index`.
Für den Scheduled Publisher muss im Admin-Dashboard „Direct indexing“ aktiviert
werden; äquivalent kann die Ausgabe mit
`{"uid":"<admin-uid>","label":"scheduled-publisher","scopes":["consensus:run","share:write","share:index"]}`
erfolgen.

## Run starten

```http
POST /api/v1/consensus/runs
X-API-Key: cns_live_…
Idempotency-Key: 019f78b5-unique-per-logical-run
Content-Type: application/json

{"question":"Welche Evidenz spricht für und gegen diese These?","deep_think":false}
```

Antwort: HTTP `202`, `Location: /api/v1/consensus/runs/{run_id}` und ein
Run-Objekt. Der Request akzeptiert nur `question` und optional `deep_think`;
Modelle, Modellanzahl, Kosten und Limits werden ausschließlich serverseitig
bestimmt. Derselbe Idempotency-Key mit identischem Request liefert denselben
Run. Mit anderem Request folgt HTTP `409`.

Consensus API v1 verwendet für jeden Run die feste serverseitige
Sechs-Provider-Auswahl OpenAI, Mistral, Anthropic, Gemini, DeepSeek und Grok.
DeepSeek ist in v1 verpflichtend und verarbeitet den Prompt in China; es gibt
für diesen API-Vertrag keinen per-Request Opt-out.

## Status/Ergebnis lesen

```http
GET /api/v1/consensus/runs/{run_id}
X-API-Key: cns_live_…
```

Mögliche Statuswerte sind `accepted`, `reserved`, `running`, `succeeded` und
`failed`. Nur `succeeded` enthält `result`; nur `failed` enthält `error`.
Ein Schlüssel kann ausschließlich Runs seiner zugeordneten UID lesen.

## Aufbewahrung und vorzeitige Löschung

Jeder Run und sein Idempotenz-Mapping erhalten bei Annahme ein `expires_at` und
werden spätestens 30 Tage danach gelöscht. Zusätzlich zur Firestore-TTL-
Eignung räumt ein periodischer serverseitiger Fallback abgelaufene Datensätze
auf. Die Antwort enthält `expires_at`.

Ein bereits terminaler Run (`succeeded` oder `failed`) kann früher gelöscht
werden:

```http
DELETE /api/v1/consensus/runs/{run_id}
X-API-Key: cns_live_…
```

Erfolg liefert HTTP `204`. Noch laufende bzw. reservierte Runs liefern `409`,
damit Usage- und Provider-Lifecycle nicht durch eine Lösch-Race entkoppelt
werden. Nach der Löschung kann derselbe Idempotency-Key wieder für einen neuen
logischen Run verwendet werden.

## Erfolgreichen Run publizieren

```http
POST /api/v1/consensus/runs/{run_id}/share
X-API-Key: cns_live_…
```

Der Run muss `succeeded` sein und derselben UID wie der Schlüssel gehören.
Der Endpoint erzeugt direkt einen unveränderlichen, öffentlichen Share-Snapshot
und liefert `share_id`, kanonische absolute `url`, `index_eligible`,
`indexing_status`, `robots` und `in_sitemap`. Wiederholungen für denselben Run
liefern denselben Link (`200` statt initial `201`) und verbrauchen keine weitere
Share-Quote. API-Publikationen verwenden nicht das kurzlebige Browser-
`pending_results`-Zwischenformat; sie bleiben dadurch während der Run-Retention
publizierbar.

Status und Lifecycle:

```http
GET    /api/v1/shares?limit=20
GET    /api/v1/shares/{share_id}
DELETE /api/v1/shares/{share_id}
```

`DELETE` widerruft die Seite wie der bestehende Browser-Flow, setzt sie sofort
auf `noindex` und liefert `204`.

## Seite indexierbar schalten

```http
PUT /api/v1/shares/{share_id}/indexing
X-API-Key: cns_live_…
Content-Type: application/json

{"indexed":true}
```

Hierfür sind Scope `share:index` **und** eine aktuell aktive Admin-Rolle nötig.
Automatische Freigabe ist nur möglich, wenn der bestehende Share-Quality-Filter
erfüllt ist. Existiert bereits eine indexierte Seite mit demselben normalisierten
Frage-Hash, antwortet die API mit `409 duplicate` samt Canonical-Ziel. Nach
Erfolg liefert die Seite `index, follow` und ist in `sitemap-shares.xml` enthalten.
Das macht die URL indexierbar; die tatsächliche Aufnahme in einen externen
Suchindex bleibt Sache der jeweiligen Suchmaschine/Search Console.

## Geplanter Publisher via GitHub Actions

`scripts/publish_consensus.py` bildet den kompletten Ablauf ab:

1. letzte eigene Share-Fragen laden,
2. optional per OpenAI Responses API + Web Search eine neue, nicht redundante
   Frage wählen,
3. Consensus-Run starten und pollen,
4. Run publizieren,
5. geeigneten Share direkt indexierbar schalten.

Der Workflow `.github/workflows/publish-consensus.yml` läuft standardmäßig
dienstags um 07:15 UTC und kann manuell mit einer festen Frage gestartet werden.
Im GitHub-Repository werden folgende Actions-Secrets benötigt:

- `CONSENSUS_API_KEY`: Key einer Admin-UID mit allen drei Scopes.
- `OPENAI_API_KEY`: nur für die automatische Themenwahl; bei manueller Frage
  wird kein OpenAI-Call ausgeführt.

Optionale Repository-Variablen: `CONSENSUS_API_BASE_URL`,
`OPENAI_TOPIC_MODEL` (Default `gpt-5.6-luna`) und `CONSENSUS_TOPIC_BRIEF`.
Der Workflow verwendet eine Run-stabile Idempotency-Key-ID; ein Retry kann
deshalb keinen zweiten Consensus-Run für denselben Workflow-Lauf starten.
