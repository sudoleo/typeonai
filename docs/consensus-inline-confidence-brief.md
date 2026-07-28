# Auftrag: Consensus-Bereich auf Inline-Confidence umbauen

**Erstellt:** 2026-07-24
**Status:** Umsetzungsauftrag für eine neue Session
**Voraussetzung:** [`docs/codebase-map.md`](codebase-map.md) zuerst lesen (AGENTS.md-Pflicht).

---

## 0. Worum es geht (Kontext, nicht überspringen)

Heute rendert die App zwei konkurrierende Dokumente nebeneinander: links die
Consensus-Antwort (80 %), rechts eine Differences-Spalte (20 %). Der Nutzer muss
zweimal lesen, um zu wissen, welchen Passagen er trauen kann.

**Ziel:** Es gibt weiterhin **eine** Antwort, die man von oben nach unten liest.
Die Uneinigkeit steckt *in* dieser Antwort — an der Textstelle, an der sie
auftritt — statt in einer zweiten Spalte daneben.

Empirischer Hintergrund: `data/benchmark/runs/pooled_v1/results.json` zeigt, dass
die synthetisierte Antwort gegenüber dem Synthesizer-Modell allein keinen
Genauigkeitsvorteil hat (286 vs. 286 von 314). Der Wert des Produkts liegt in der
Uneinigkeitsstruktur, nicht im Antworttext. Die UI stellt das derzeit umgekehrt dar.

**Nicht-Ziel:** Kein Framework-Wechsel, kein Umbau der Judge-Pipeline, keine
Änderung am Differences-JSON-Schema, keine neuen LLM-Calls.

---

## 1. Vorentscheidungen (bereits vom Owner getroffen — nicht neu aufrollen)

### 1.1 Watch-Alarmwelle: bewusst akzeptiert

Der Consensus-Prompt (`app/services/llm/consensus_engine.py`, ca. Z. 570–585,
`user_facing_instruction`) enthält heute:

> `"Resolve disagreements silently where possible."`

Diese Zeile muss weg — sie löscht genau das Signal, das die neue UI markieren soll.

Nebenwirkung: Watch-Läufe vergleichen jeden neuen Consensus mit dem vorherigen
(`watch_scheduler.py`, `opinion_map.py`, Change-Judge). Ein geänderter Prompt
ändert den Textstil aller künftigen Läufe gegenüber ihrer Baseline, sodass beim
nächsten Lauf viele Watches einmalig „changed" melden können.

**Entscheidung des Owners (2026-07-24): akzeptiert.** Es gibt derzeit wenige
aktive Watches; die Entwicklungsgeschwindigkeit hat Vorrang.

Konkret heißt das:
- **Kein Feature-Flag, kein zweiter Prompt-Pfad.** Der neue Prompt gilt überall
  gleich — App, Watch-Runner, Topic-Runner, API v1.
- Keine Baseline-Migration, kein Pausieren von Watches, keine Nutzerankündigung.
- Der Prompt bekommt trotzdem einen **eigenen Commit** — nicht als Gate, sondern
  damit er einzeln revertierbar bleibt, falls der neue Antwortstil nicht gefällt.

### 1.2 Prompt-Änderung bricht die Benchmark-Vergleichbarkeit (V1-Freeze)

`docs/benchmark-plan.md` und die Benchmark-Historie führen den Prompt-Stand als
Versionsmarke (V0 → V1 bei der Anonymisierung). Eine Änderung am
Consensus-Prompt erzeugt **V2**.

Pflicht:
- In `docs/benchmark-plan.md` als V2 dokumentieren, mit Datum und Diff.
- Deutlich vermerken: **neue Runs dürfen nicht mit `pooled_v1` gepoolt werden.**
- `data/benchmark/runs/*` und die publizierte `/benchmark`-Seite bleiben
  unverändert (sie beschreiben V1).

---

## 2. Ist-Zustand: die Verträge, die nicht brechen dürfen

Vor jeder Änderung verifizieren — die Angaben unten stammen aus einer
Code-Durchsicht am 2026-07-24 und sind gegen den Code zu prüfen.

### 2.1 DOM (`templates/index.html`, ca. Z. 530–560)

```
#consensusOutput
  #consensusVerdict          .consensus-verdict     (Verdict-Header)
  #consensusResponse         .consensus-box
    .consensus-main          (80 %)  > h2 + p + #consensusClaimsFallback
    .consensus-differences   (20 %)  > h2 + #differencesCards + p
  .consensus-divider
#claimSheetBackdrop, #claimPopover                  (nach <body> verschoben)
```

**Kritisch:** Die gesamte Markdown-Antwort wird per `injectMarkdown` in das
**eine** `<p>` unter `.consensus-main` geschrieben. Der Selektor
`.consensus-main p` wird an mindestens sechs Stellen in `consensus-run.js`
verwendet (Streaming-Ziel Z. 481, Spinner Z. 367, Reset Z. 347, Final-Render
Z. 574, zwei Fehlerpfade Z. 644/661) sowie in `consensus-insights.js:536`
(`renderClaimBadges`) und in `firebase.js` / `demo.js`.

### 2.2 JS-Verträge

| Vertrag | Wo definiert | Wer ruft |
|---|---|---|
| `window.renderConsensusInsights(differences_data, includedCount) -> bool` | `consensus-insights.js` | `consensus-run.js:585`, `firebase.js:1252`, `demo.js:446` |
| `window.applyCredibilityFrame` / `window.colorizeCredibility` | `consensus-insights.js` | dieselben drei (Freitext-Fallback, wenn `false`) |
| `createStreamRenderer(el, isActive)` | `markdown-stream.js` | streamt live in `.consensus-main p` |
| `injectMarkdown(el, md)` | `markdown-stream.js` | rendert Markdown **und** KaTeX |
| `dataset.consensusAnswer`, `dataset.consensusSources` | `sources.js` | Evidence-Mapping |

**Reihenfolge ist ein Vertrag:** erst `injectMarkdown` (inkl. KaTeX), **danach**
`renderConsensusInsights` — sonst zerstört der Math-Renderer die eingefügten
Marker oder die Anker-Suche trifft KaTeX-Knoten.

### 2.3 Anker-Mechanik (existiert bereits, wird wiederverwendet)

- Backend verlangt vom Judge: `"anchor": "verbatim excerpt of 5-12 consecutive
  words copied exactly from the consensus answer"` (`consensus_engine.py` ~Z. 751)
  und verifiziert ihn serverseitig gegen den Consensus-Text (~Z. 1173 ff.);
  nicht auffindbare Anker werden geleert.
- Frontend: `findAnchorTarget()` / `findRangeInTextNode()` /
  `searchVariants()` in `consensus-insights.js` (Z. 124–180) lösen den Anker in
  einen Textknoten-Range auf. **Diese Funktionen sind gut und bleiben.**

### 2.4 Bekannte Stolpersteine (aus früheren Sessions)

- Ein globaler `button:not(...)`-Selektor in `static/css/components-input.css`
  übersteuert neu eingeführte Buttons. Neue Marker-Buttons dort ausnehmen.
- Nach **jeder** CSS-/JS-Änderung den `?v=`-Cache-Buster in
  `templates/index.html` (und ggf. `style.css`) bumpen — sonst wird stale CSS
  ausgeliefert.
- Die Script-Ladereihenfolge in `templates/index.html` ist ein Vertrag.
- Module kommunizieren ausschließlich über `window.*` / `window.App`,
  **nicht** über ES-Imports.

---

## 3. Umsetzung — in dieser Reihenfolge, jede Phase einzeln lauffähig

> **Reihenfolge-Hinweis:** Der Prompt kommt bewusst **zuerst**. Er ist der größte
> Hebel pro Zeile Code, er ist seit der Entscheidung in §1.1 nicht mehr blockiert,
> und er verändert den Antworttext, den alle folgenden Phasen annotieren. Erst mit
> dem neuen Prompt markiert die UI Sätze, die die Spannung tatsächlich tragen —
> vorher würde man Marker an geglättete Prosa heften und das Ergebnis falsch
> beurteilen.

### Phase 1 — Prompt (eigener Commit)

In `consensus_engine.py`, `user_facing_instruction`:
- `"Resolve disagreements silently where possible."` **entfernen**.
- Ersetzen durch sinngemäß: *Wo die Quellen bei etwas auseinandergehen, das für
  die Entscheidung des Lesers relevant ist, benenne es in einem Nebensatz im
  Textfluss — mit dem sachlichen Grund, nicht mit Zählungen, nicht mit
  Modellnamen. Alles Übrige weiterhin glätten.*
- Die Regeln zu Quellen-Tags (`[S1]`), „keine Modellnamen nennen" und „keine
  Rückfragen stellen" **bleiben unverändert**.
- `CONSENSUS_ERROR_PREFIXES` / `is_consensus_error_text` nicht anfassen.
- Gilt einheitlich für alle Pfade (App, Watch, Topic, API v1) — siehe §1.1.

**Abnahme dieser Phase, bevor es weitergeht:** drei bis fünf echte Fragen mit
bekanntem Widerspruch laufen lassen und die Antworttexte lesen. Prüfen:
- Wird die Uneinigkeit im Fluss benannt, ohne dass der Text zur Aufzählung wird?
- Tauchen Modellnamen oder Consensus-Mechanik auf? (Dürfen sie nicht.)
- Lassen sich die Judge-Anker weiterhin im Text auflösen? (Sollte eher besser
  werden, weil die Stelle nun tatsächlich existiert.)

Erst wenn der Text stimmt, lohnt die UI-Arbeit.

### Phase 2 — Stabile Container einziehen (reines Refactoring, keine Optik)

Ziel: das fragile `<p>` als Renderziel loswerden, ohne Verhalten zu ändern.

1. In `templates/index.html` innerhalb `.consensus-main` ein
   `<div id="consensusAnswerBody" class="consensus-answer-body"></div>`
   einführen und als Render-/Streamziel verwenden.
2. Alle Fundstellen von `.consensus-main p` auf einen **einzigen Helper**
   umstellen, z. B. `window.App.consensusBodyEl()` in `app-core.js`.
   Betroffen: `consensus-run.js` (6 Stellen), `consensus-insights.js`,
   `firebase.js`, `demo.js`.
3. Kompatibilität: Solange irgendwo noch `.consensus-main p` erwartet wird,
   darf der Helper darauf zurückfallen. Nach dem Umbau entfernen.

**Abnahme:** Live-Lauf, Bookmark-Öffnen und Landing-Demo rendern exakt wie
vorher. Keine Optikänderung.

### Phase 3 — Marker-Rendering (das visuelle Kernstück)

In `consensus-insights.js`, aufbauend auf `findAnchorTarget`:

1. **Markierungseinheit auf Satzgrenze erweitern.** Der verifizierte Anker
   bleibt intern präzise; für die Darstellung auf den umgebenden Satz ausdehnen
   (Grenzen: `.!?` + Zeilenumbruch, Abkürzungen tolerieren). Ein unterstrichenes
   7-Wort-Fragment mitten im Satz wirkt zufällig.
2. Den Range in ein `<span class="cx-claim">` wrappen, mit Zustandsklasse:
   - `is-unanimous` — alle Modelle stützen die Aussage
   - `is-minor` — Difference ohne `severity: major`
   - `is-split` — Claim mit Dissens (z. B. 2/4)
   - `is-major` — Widerspruch mit `severity: major`
3. **Visuelles Vokabular** (CSS, Rechtschreibprüfungs-Metapher):
   - `is-unanimous`: keine Textdekoration; nur die Marke daneben — eine ruhige
     Mikro-Quote wie „6/6" mit transparenter Flaeche und feiner Kontur. Die
     tabellarischen Ziffern in einer kleinen Kapsel bleiben scanbar und sind
     zugleich klar von hochgestellten Quellenzahlen unterschieden.
   - `is-minor`: feine durchgezogene 1px-Unterlinie, neutral
   - `is-split`: durchgezogene 1px-Unterlinie in Bernstein — dieselbe Farbe wie
     das `has-dissent`-Badge daneben (seit 2026-07-28; die graue Linie unter
     einer gelben Quote war ein Widerspruch in sich)
   - `is-major`: durchgezogene 2px-Unterlinie in Bernstein
     (seit 2026-07-27 `underline solid` statt `underline wavy` — die
     Wellenlinie las sich als Rechtschreibfehler, User-Vorgabe)
   - **Kein** reduzierter Textkontrast und **keine** Hintergrundfarbe —
     WCAG-Kontrast bleibt unangetastet.
4. **Widersprüche (`differences[]`) ebenfalls inline verankern.** Bisher sind
   nur `claims[]` verankert; die Contradictions leben ausschließlich in den
   Karten. Sie tragen `positions[].quote` (wörtlich, serverseitig verifiziert) —
   dieselbe Anker-Suche darauf anwenden. **Das ist die wichtigste inhaltliche
   Änderung dieser Phase.**
5. Klick auf einen Marker öffnet **das bestehende** `openClaimPopover` bzw. die
   Contradiction-Karte. Popover-Logik, Bottom-Sheet-Verhalten, „View answer"-
   Sprung und die Resolve-Runde **unverändert lassen**.
6. Barrierefreiheit: Marker sind `<button>` (Tastatur erreichbar),
   `aria-haspopup="dialog"`, mit sprechendem `aria-label`
   („4 von 6 Modellen stützen das — Details öffnen"). Trefferfläche ≥ 44 px
   über unsichtbares Padding, ohne die Zeilenhöhe zu verändern.

**Abnahme:** Anker, die nicht auflösbar sind, landen weiterhin sichtbar in
`#consensusClaimsFallback` — es darf keine Aussage stillschweigend verschwinden.

### Phase 4 — Layout: Differences aus der Primäransicht lösen

1. `.consensus-main` bekommt die volle Breite; die 80/20-Spaltenteilung in
   `components-*.css` entfällt.
2. `.consensus-differences` wird zu einem aufklappbaren `<details>`-Überblick
   **unter** der Antwort („Alle Unterschiede ansehen"), Standard: zu.
   `#differencesCards` und das Karten-Rendering bleiben inhaltlich unverändert.
3. Legendenzeile direkt unter der Antwort, dezent:
   *„Markiert sind geprüfte Kernaussagen. Unmarkierter Text wurde nicht einzeln
   abgeglichen."* — verhindert, dass fehlende Markierung als Bestätigung
   gelesen wird.
4. Der Freitext-Fallback (`renderConsensusInsights` liefert `false`) muss
   weiterhin funktionieren: dann erscheint der alte Differences-Block sichtbar
   und aufgeklappt.

### Phase 5 — Verdict-Header inhaltlich statt zählend

Heute: „The models contradict each other on 1 point" (eine Zählung).
Neu: **worüber** sie sich einig bzw. uneinig sind — aus `differences[].claim`
generiert, ohne zusätzlichen LLM-Call, z. B.
*„Einig beim Kern. Uneinig bei: Dosierung und Zeitpunkt."*
Agreement-Score, Judge-Fußnote („Analysis by …, independent of the consensus
engine") und Severity-Tags bleiben.

### Phase 6 — Redundanz-Check (klein, aber nicht überspringen)

Nach Phase 1 benennt der Antworttext die Uneinigkeit bereits in Prosa. Dazu
kommen jetzt Wellenlinie, Badge, Verdict-Zeile und Karte. Das kann sich vierfach
wiederholen und den Text schwatzhaft wirken lassen.

An echten Läufen gegenprüfen und **eine** Ebene zurücknehmen, falls es doppelt:
- Wenn der Satz die Uneinigkeit schon klar benennt, reicht dort der dezente
  Marker ohne zusätzliche Severity-Auszeichnung.
- Die Verdict-Zeile darf zusammenfassen, was im Text steht — sie soll es nicht
  wortgleich wiederholen.

Das ist eine Urteilsfrage am fertigen Ergebnis, keine Spezifikation im Voraus.

---

## 4. Ausdrücklich außerhalb des Scope

- **Öffentliche Share-Seiten** (`templates/share.html`) rendern serverseitig über
  eine eigene Jinja-Implementierung mit Toggle zwischen Consensus- und
  Differences-Ansicht. Claims werden zwar in
  `share_snapshots.py` (~Z. 419–470, max. 30) persistiert, aber dort **nicht**
  inline gerendert. Das ist ein eigener, zweiter Umbau — **nicht** in dieser
  Session mitmachen.
- `templates/topic.html` (eigenes Rendering) — ebenfalls später.
- Differences-JSON-Schema, Judge-Kaskade, Agreement-Score-Berechnung,
  `opinion_map.py` — unangetastet.
- Watch-Dashboard, Bookmarks-Pagination, Pricing/Plan-Katalog — anderes Thema.

---

## 5. Abnahme

**Backend**
```
.\venv\Scripts\python.exe -m pytest tests
```
Muss grün bleiben (Referenz: 691 passed). Wird der Prompt geändert, sind
Prompt-Assertions in den Tests entsprechend anzupassen — nicht zu löschen.

**Frontend** (keine Auto-Tests → `docs/smoke-checklist.md` durchgehen), zusätzlich:

1. Live-Lauf mit Widerspruch: Marker erscheinen inline, Popover öffnet,
   „View answer" springt korrekt.
2. Live-Lauf ohne Widerspruch: nur Unanimous-Badges, keine Wellenlinien.
3. **Copy-/Zitat-Buttons** (`consensus-actions.js`): Der kopierte Text darf
   **keine** Marker-Beschriftungen wie „4/6" enthalten. Falls doch → beim
   Kopieren aus einer bereinigten Quelle lesen. **Explizit prüfen.**
4. Bookmark öffnen (`firebase.js`-Pfad): identisches Rendering wie live.
5. Landing-Demo (`demo.js`): rendert weiterhin, inklusive Contradiction.
6. Freitext-Fallback: `differences_data` künstlich auf `null` setzen → alter
   Block erscheint sichtbar.
7. Anker, der im Text nicht auffindbar ist → erscheint in
   `#consensusClaimsFallback`, verschwindet nicht.
8. KaTeX: Antwort mit LaTeX → Formeln intakt, Marker zerstören sie nicht.
9. Mobil (≤ 768 px): Bottom-Sheet statt Popover, Trefferflächen ≥ 44 px,
   kein horizontales Scrollen.
10. Light **und** Dark: Wellenlinie in beiden Themes erkennbar.
11. Streaming: während des Streams keine Marker; sie erscheinen erst mit dem
    `final`-Event.
12. Abbruch mitten im Consensus (Cancel) → kein halb markierter Zustand.

**Pflicht vor Abschluss**
- `?v=`-Cache-Buster bumpen.
- `docs/codebase-map.md` aktualisieren (§3 Frontend-Module, §4 Kern-Flows) —
  AGENTS.md-Pflicht bei Änderungen an Architektur/Kern-Flows.
- Bei Prompt-Änderung zusätzlich `docs/benchmark-plan.md` (V2, siehe §1.2).

---

## 6. Wenn der Umfang zu groß wird

Priorisierung, falls gekürzt werden muss — die ersten beiden Punkte tragen den
größten Teil des Nutzens und sind zusammen etwa ein Nachmittag:

1. **Phase 1** (Prompt) — größter Effekt pro Zeile Code
2. **Phase 5** (Verdict inhaltlich statt zählend)
3. **Phase 3.4** (Widersprüche inline verankern — die wichtigste UI-Änderung)
4. **Phase 2** (Container-Refactoring) — nur nötig, sobald Phase 3 kommt
5. Rest (Wellenlinien-Feinschliff, Layout, Redundanz-Check)

Phase 1 und 5 sind unabhängig von Phase 2 umsetzbar: Der Prompt ist Backend,
der Verdict-Header rendert in `#consensusVerdict` und nicht in den Antworttext.
Man kann also den größten Teil des Nutzens holen, ohne das fragile `<p>`
anzufassen.
