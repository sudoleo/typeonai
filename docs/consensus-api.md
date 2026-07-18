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
