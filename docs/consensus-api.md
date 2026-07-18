# Consensus API v1

Der vollständige maschinenlesbare Vertrag wird von FastAPI unter
`/openapi.json` ausgeliefert. Die v1-Routen verwenden das OpenAPI-
Security-Scheme `ConsensusApiKey` (`X-API-Key`).

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

## Status/Ergebnis lesen

```http
GET /api/v1/consensus/runs/{run_id}
X-API-Key: cns_live_…
```

Mögliche Statuswerte sind `accepted`, `reserved`, `running`, `succeeded` und
`failed`. Nur `succeeded` enthält `result`; nur `failed` enthält `error`.
Ein Schlüssel kann ausschließlich Runs seiner zugeordneten UID lesen.
