# Briefing: Admin-SEO-Tab neu ordnen (UX, keine Funktionsänderung)

Stand 2026-08-24. Auftraggeber: Max (Betreiber consens.io).

## 1. Warum

Der SEO-Tab im Admin (`templates/admin.html`, Abschnitt `#tab-seo`) ist über
mehrere Iterationen gewachsen und zeigt heute **alles gleichzeitig und gleich
laut**. Konkret:

- Die Seitentabelle listet **jede einzelne Seite flach** — aktuell 55, Tendenz
  steigend, weil jeder veröffentlichte Share eine neue Zeile erzeugt. Bei 200
  Seiten ist die Ansicht unbenutzbar.
- Das Portfolio als Ganzes ist nirgends ablesbar. Man sieht 55 Zeilen, aber
  nicht „wie steht es um die Sammlung insgesamt und was hat sich verändert".
- Der weit überwiegende Teil der Zeilen ist zu jedem Zeitpunkt **irrelevant**
  (Status `emerging`, Empfehlung `monitor`/`wait`, nichts zu tun) und verdeckt
  die wenigen Zeilen, die eine Entscheidung brauchen.
- Die redaktionellen Entscheidungen im Wochenreview sind Zeile für Zeile zu
  bestätigen: pro Seite ein Dropdown, ein Notizfeld, ein Button. Bei zehn
  offenen Entscheidungen sind das dreißig Interaktionen.
- Review-Panel und Tabelle zeigen teils dieselbe Information zweimal
  (Empfehlung, Status, Watch-Zustand), in unterschiedlicher Form.

**Ziel:** dieselbe Funktionalität, deutlich bessere Übersicht. Der Betreiber
soll in zehn Sekunden sehen, wie das Portfolio steht und was zu tun ist — und
erst danach, auf Wunsch, in die Einzelseiten absteigen.

Dies ist der erste von zwei Schritten. Der zweite Schritt (mehr Automatisierung:
Query-Auswertung, Titel-Vorschläge, Änderungs-Journal) kommt später und ist
**nicht Teil dieser Aufgabe**. Die Oberfläche sollte allerdings so gebaut sein,
dass später ohne Umbau Platz für weitere Befunde ist.

## 2. Nicht-Ziele (hart)

- **Keine Backend-Semantik ändern.** Schwellenwerte, Gruppenlogik, Safeguards,
  Judge-Prompts, Endpunkt-Verträge bleiben, wie sie sind.
- **Keine Funktion entfernen.** Jedes heute vorhandene Bedienelement muss
  danach erreichbar sein — es darf umsortiert, gruppiert, hinter eine
  Aufklappebene oder einen Filter gelegt werden, aber nicht verschwinden.
- **Keine neuen Aktionen erfinden**, die serverseitig etwas auslösen könnten.
  Bulk-Bedienung derselben bestehenden Endpunkte ist erlaubt und erwünscht.
- Kein neues Framework, keine neue Build-Stufe. Der Admin ist bewusst
  Vanilla-JS mit `admin.js` als ES-Modul.

## 3. Vollständige Bestandsaufnahme (Abnahmeliste)

Alles Folgende existiert heute und muss nachher erreichbar sein.

### 3.1 Kopfbereich „Search Console performance"

| Element | ID | Funktion |
| --- | --- | --- |
| Check Search Console connection | `checkSeoConnectionBtn` | `POST /api/admin/seo/check` |
| Collect Search Console data now | `collectSeoBtn` | `POST /api/admin/seo/collect` |
| Reload | `reloadSeoBtn` | `GET /api/admin/seo` |
| Statuszeile | `seoStatus` | Rückmeldung aller SEO-Aktionen |
| Karte Configuration | `seoConfigState`, `seoConfigMessage` | Credential-Prüfung ohne Google-Request |
| Karte Connection | `seoConnectionState`, `seoConnectionMessage` | Live-Property-Check |
| Karte Captured / eligible URLs | `seoCapturedCount` | Abdeckung |
| Karte Latest collection run | `seoLastRun`, `seoLastRunMessage` | Status, Zeit, geschriebene Zeilen |
| Latest finalized date | `seoFinalDate` | Ende des Datenfensters |
| Optional content judge | `seoContentJudgeState` | Konfigurationszustand |
| Disclaimer | `seoDisclaimer` | Pflichthinweis zur Datenqualität |

### 3.2 Panel „Weekly portfolio review"

Zeitplan: `seoReviewEnabled`, `seoReviewInterval`, `seoReviewTime`,
`seoReviewTimezone`, `saveSeoReviewConfigBtn` (`PUT /api/admin/seo/review/config`),
`runSeoReviewBtn` (`POST /api/admin/seo/review/run`).

Karten: `seoReviewLast`/`seoReviewNext`, `seoPortfolioJudge`,
`seoPublisherWatchCount`, `seoReviewSummary`, `seoReviewJudgeState` +
`seoReviewJudgeError`, `seoReviewCollection` + `seoReviewCollectionMessage`,
`seoReviewDelta`.

Befunde: `seoReviewFindings` mit `seoReviewFindingsPositive` /
`seoReviewFindingsNegative` (die Prosa des Portfolio-Judge).

Fortschritt: `seoReviewProgress` (offene sichere Aktionen, offene redaktionelle
Entscheidungen, erledigte Seiten, Telegram-Status).

Gruppen: `seoReviewGroups` — je Gruppe Überschrift mit Anzahl und ein
Aktionsbutton (außer `manual_improvement`), je Seite eine Checkbox plus
Metazeile. Gruppen und ihre Bedeutung stehen in
`app/services/seo_weekly_review.py` (`GROUPS`, `GROUP_ACTIONS`,
`SAFE_APPLY_ALL_GROUPS`). `applyAllSeoReviewBtn` wendet alle sicheren Gruppen
an. Vor jedem Anwenden läuft ein Preview-Dialog
(`POST …/preview`), dann `POST …/apply`; Löschen verlangt zusätzlich
`confirm_delete`.

Redaktionelle Entscheidungen: in Gruppe `manual_improvement` je Zeile ein
Auswahlfeld (Optionen kommen aus `editorial_decision_template.options` des
Servers), ein optionales Notizfeld (max. 500 Zeichen, der graue Text ist der
Platzhalter aus `template.explanation`) und ein Bestätigen-Button
(`POST …/editorial-decision`).

Erledigtes: `seoReviewCompleted` / `seoReviewCompletedList`.

Topic Brief: `seoTopicBriefPanel` mit `seoTopicBriefStrength`,
`seoTopicBriefReason`, `seoTopicBriefEvidence`, `seoCurrentTopicBrief`,
`seoProposedTopicBrief`, `acceptSeoTopicBriefBtn`, `rejectSeoTopicBriefBtn`,
`seoTopicBriefDecision`. Bei `topic_brief_decision === 'insufficient_evidence'`
bleibt der Vorschlag sichtbar, ist aber nicht anwendbar.

Historie: `seoReviewHistoryPanel` / `seoReviewHistory`, lädt beim Aufklappen
`GET /api/admin/seo/reviews?limit=12`.

Regeln: `seoSearchRules` (fest verdrahtete Search-Opportunity-Regeln,
read-only).

### 3.3 Seitentabelle

Spalten heute: Page (URL, Origin + Share-ID, Dossier-Titel, bis zu drei Top-
Queries) · 7 Tage (Clicks, Impressions, CTR, Position) · 28 Tage (dieselben
vier) · Status (+ Zeile „x/28 daily rows · visibility n") · Recommendation
(Empfehlung, Confidence, aufklappbare Historie der letzten drei Judgements,
Button „Ask content judge" bei Status `opportunity`/`declining`, sofern der
optionale Judge konfiguriert ist) · Read-only analysis (Button
`POST /api/admin/seo/pages/{page_id}/recommendation` bzw. `…/content-judge`).

Dazu `seoEmptyState` und `seoRules` (Erklärung der Statusklassen).

## 4. Gewünschte Richtung (Vorschlag, nicht Vorschrift)

Die konkrete Umsetzung ist Sache der bearbeitenden Session; das hier ist die
Absicht dahinter.

1. **Drei Ebenen statt einer Fläche.**
   - *Portfolio*: wie steht die Sammlung insgesamt da, was hat sich seit dem
     letzten Lauf verändert, was sagt der Judge. Das gehört nach oben und darf
     Platz beanspruchen — Statusverteilung, Delta, Befunde.
   - *Arbeitsliste*: nur, was jetzt eine Entscheidung braucht. Kurz, endlich,
     abarbeitbar. Wenn sie leer ist, soll das sichtbar und beruhigend sein.
   - *Bestand*: die vollständige Seitenliste, standardmäßig eingeklappt oder
     gefiltert, mit Suche.
2. **Die Tabelle muss gruppieren oder filtern können.** Nach Status, nach
   Empfehlung, nach Herkunft (`static_page` / `share` / `topic`). Voreinstellung
   sollte „nur Auffälliges" sein, nicht „alles". Eine Volltextsuche über URL und
   Titel ist bei 200 Seiten Pflicht.
3. **Doppelungen zusammenführen.** Wenn eine Seite im Review-Panel als
   Entscheidungskandidat auftaucht, sollte die Tabelle das nicht ein zweites
   Mal in anderer Form erzählen — oder die Tabelle wird zur einzigen Liste und
   das Panel verlinkt hinein.
4. **Redaktionelle Entscheidungen in Serie bedienbar machen.** Etwa: der
   Server schlägt ohnehin eine Entscheidung vor (`suggested_decision`) — eine
   Möglichkeit, mehrere Zeilen mit dem Vorschlag auf einmal zu bestätigen, spart
   den Großteil der Klicks. Die Einzelentscheidung inklusive Notiz muss weiter
   möglich bleiben. Der Endpunkt ist pro Seite; mehrere Aufrufe sind in Ordnung,
   müssen aber einzeln quittiert und bei Teilfehlern verständlich berichtet
   werden.
5. **Zahlen lesbar machen.** Acht Metrikspalten nebeneinander sind schwer zu
   erfassen. Sichtbarkeit (`visibility`, ein Klick zählt wie 20 Impressionen)
   ist bereits die sinnvolle Sortiergröße; 7-Tage- und 28-Tage-Werte könnten als
   Wert plus Veränderung statt als zwei getrennte Blöcke erscheinen.
6. **Leere und Fehlzustände ernst nehmen.** „Keine Daten gesammelt",
   „Collection fehlgeschlagen", „Judge nicht gelaufen" sind reale Dauerzustände
   dieses Systems gewesen und müssen im neuen Layout sofort auffallen, nicht in
   einer Karte unter fünf anderen stehen.

## 5. Harte Randbedingungen

- **Keine Inline-Skripte, keine Inline-Styles, keine `on*`-Attribute** in
  `templates/admin.html`. `tests/test_phase6_architecture.py` prüft das.
- **`templates/admin.html` muss unter 700 Zeilen bleiben** (aktuell 508). Der
  Test prüft es. Wenn das Markup wächst, gehört Struktur nach `admin.js`.
- Derselbe Test pinnt die Versionsmarken
  `/static/css/admin.css?v=…` und `/static/js/admin.js?v=…`. **Wer CSS oder JS
  ändert, muss die Marke im Template hochziehen *und* den Test anpassen** —
  sonst wird veraltetes CSS ausgeliefert.
- **Kein `innerHTML` mit Serverdaten.** `admin.js` baut alles über
  `createElement` + `textContent` (siehe `appendSeoText`). Seitentitel und
  Suchanfragen sind Fremdtext und werden als Daten behandelt, nie als Markup.
- Styles gehören nach `static/css/admin.css`; bestehende Klassen
  (`admin-btn`, `admin-section`, `section-hint`, `seo-summary-card`,
  `seo-review-group`, `seo-status`, `seo-table`) weiterverwenden statt
  parallele Sprachen aufzumachen. Kein neues Farbkonzept: der Admin bleibt
  bei der vorhandenen, zurückhaltenden Palette, Ampelfarben nur dort, wo sie
  heute schon Bedeutung tragen.
- Dark Mode muss weiter funktionieren (der Admin wird dunkel betrieben).
- Der SEO-Tab lädt über `loadSeoOverview()`; `renderSeoOverview(data)` ist der
  eine Einstiegspunkt für die Daten. Diese Struktur darf umgebaut werden, aber
  bewusst und in einem Stück.

## 6. Abnahme

- `venv\Scripts\python.exe -m pytest tests -q --ignore=tests/e2e` ist grün
  (aktuell 1342 Tests).
- Jedes Element aus Abschnitt 3 ist erreichbar und funktioniert; am besten
  einmal die Liste durchgehen und quittieren.
- Bei 55 Seiten passt der Einstieg ohne Scrollen auf einen Bildschirm, und die
  Frage „was muss ich jetzt tun" ist ohne Scrollen beantwortet.
- Die Seitenliste bleibt bei 200+ Zeilen bedienbar (Filter, Suche, Gruppierung
  greifen clientseitig; die API liefert weiterhin alles auf einmal).
- Kein neuer Netzwerkaufruf beim Tab-Wechsel, der vorher nicht nötig war.

## 7. Kontext, der hilft

- Die SEO-Pipeline selbst ist in `docs/codebase-map.md` und in
  `app/services/seo_data.py`, `seo_recommendation.py`, `seo_weekly_review.py`
  beschrieben. Für diese Aufgabe reicht die Datenform, nicht die Logik.
- Wichtig für die Textgestaltung: `insufficient_data` bedeutet **zu wenige
  gespeicherte Tageszeilen, nicht zu wenig Traffic** — 28 Tage mit Nullen
  ergeben `invisible`. Diese Verwechslung hat den Betreiber real fehlgeleitet;
  die Oberfläche sollte sie ausschließen.
- Ein Share ist ein unveränderlicher Schnappschuss: er kann nicht bearbeitet,
  nur durch eine Nachfolgeseite ersetzt werden. Statische Seiten liegen als
  Templates im Repo. Das erklärt die unterschiedlichen Entscheidungsoptionen.
- Commits gehen in diesem Projekt **direkt auf `main`**, kein Feature-Branch.
  Gepusht wird nur auf ausdrückliche Aufforderung.
