# Product- und UX-Briefing: consens.io

**Stand:** 22. Juli 2026  
**Scope:** Live-Produkt auf Desktop und Mobile, Landingpage, App, Demo, Consensus-Ergebnis, öffentliche Share-/Watch-Seiten, Questions-Hub, Watch-Dashboard, Watch-Erstellung, Benchmark, About-Seite sowie Abgleich mit Architektur und Code.  
**Technische Verifikation:** `691 passed`, 33 Deprecation-Warnungen; keine Produktcode-Änderungen im Rahmen dieses Audits.

## Executive Summary

consens.io ist deutlich weiter als ein typisches Side Project. Das Produkt besitzt bereits drei wertvolle Ebenen:

1. **Compare:** Eine Frage wird parallel von mehreren Modellfamilien beantwortet.
2. **Decide:** Konsens, Widersprüche, Agreement Score, Quellen und unabhängiger Judge reduzieren die kognitive Last.
3. **Monitor:** Dieselbe Frage wird wiederholt geprüft; Änderungen werden als Watch, Verlauf und Benachrichtigung produktisiert.

Die dritte Ebene ist der strategische Vorteil. Reiner Multi-Model-Vergleich ist inzwischen ein dicht besetzter Markt. Das kontinuierliche Beobachten einer Antwort, inklusive Drift, Bedingungen, Historie, E-Mail und Telegram, ist dagegen ein deutlich schärferes Produktversprechen.

Meine Kernempfehlung lautet deshalb:

> consens.io sollte nicht primär als Multi-LLM-Chat positioniert werden, sondern als unabhängige Plattform für **AI Answer Intelligence**: vergleichen, Unsicherheit sichtbar machen und relevante Antworten über Zeit überwachen.

Die Landingpage bewegt sich bereits in diese Richtung: „The independent AI answer observatory“ und „We track when they move“ sind eigenständig und merkfähig. Im restlichen Produkt wird dieses Versprechen jedoch noch nicht konsequent genug eingelöst. Die größten Risiken liegen aktuell nicht im Design, sondern in **Vertrauen, Konsistenz, Monetarisierung und Skalierung**.

Vor weiterem Wachstum sollten vier Dinge behoben werden:

- Watch-Seiten dürfen niemals ursprüngliche Antwort, aktuellen Score und Drift aus verschiedenen Versionen vermischen.
- Plan-, Limit- und Pricing-Kommunikation muss aus einer einzigen dynamischen Quelle kommen.
- Bookmarks benötigen eine paginierte List-/Detail-Architektur statt eines vollständigen Initial-Downloads.
- „Upgrade“ darf nicht auf eine Pricing-Vorschau und einen JavaScript-Alert führen; entweder ehrlicher Beta-Zugang oder echtes Self-Service-Billing.

## Gesamturteil

| Bereich | Bewertung | Kurzurteil |
|---|---:|---|
| Differenzierung | 8/10 | Watch + Consensus + öffentliche Historie ist ein starkes, seltenes Paket. |
| Landingpage und Marke | 8/10 | Hochwertig, ruhig, eigenständig; Hero-Versprechen und primäre Aktion sind noch nicht ganz synchron. |
| Core-App | 8/10 | Sehr gute Reduktion eines komplexen Multi-Model-Flows auf eine Eingabe und einen klaren Ergebnisraum. |
| Watch-Dashboard | 8/10 | Professionell, datenreich und bereits nahe an einem eigenständigen SaaS-Produkt. |
| Vertrauen und Datenkonsistenz | 5/10 | Aktuelle/originale Versionen und Scores können sich widersprechen; dies trifft den Kernnutzen. |
| Aktivierung | 7/10 | Die Demo ist überzeugend; der direkte App-Einstieg könnte stärker geführt werden. |
| Monetarisierungsreife | 3/10 | Pro existiert funktional, aber nicht als belastbarer, selbst bedienbarer Kaufprozess. |
| Mobile UX und Accessibility | 7/10 | Responsive und visuell gut; mehrere Touch-Ziele und Semantiken sind zu klein oder missverständlich. |
| Engineering-Fundament | 8/10 | Starkes Backend, Idempotenz, Datenschutz und Tests; Frontend und History-Laden nähern sich einer Skalierungsgrenze. |

## Was bereits außergewöhnlich gut funktioniert

### 1. Die Kategorie ist besser als die Feature-Liste

„AI answer observatory“ ist stärker als „compare six AI models“. Es beschreibt einen dauerhaften Nutzen und lässt Raum für Watches, öffentliche Intelligence, API und Team-Produkte. Der Satz „AI answers shape what the world believes. We track when they move.“ ist mutig, hochwertig und merkfähig.

### 2. Die Demo liefert schnell ein echtes Aha-Erlebnis

Die interaktive Demo benötigt kein Konto, simuliert einen vollständigen Lauf und endet nach rund 18 Sekunden in einer überzeugenden Ergebnisansicht. Das ist erheblich besser als ein Video oder ein statischer Screenshot: Nutzer verstehen Agent Mode, Konsens und Differences durch Beobachtung.

### 3. Das Ergebnisdesign löst das richtige Problem

Die Reihenfolge ist produktstrategisch richtig:

- Consensus zuerst
- Agreement und Widersprüche sichtbar
- Einzelantworten als überprüfbare Grundlage
- Quellen und Zitationen
- Resolve und Watch als nächste Aktionen

Damit wird nicht nur Output aggregiert, sondern Unsicherheit strukturiert. Besonders gut ist, dass der Judge aus einer anderen Modellfamilie stammen soll und im UI offengelegt wird.

### 4. Watch ist bereits ein eigenständiger Retention-Loop

Das Dashboard bietet:

- aktive/pausierte Watches und Kapazität
- Checks und Changes der letzten sieben Tage
- nächsten Lauf
- Recent Movement
- Filter für Changed, Stable und Paused
- Agreement-Verlauf und Movement Score
- E-Mail, Telegram und Morning Brief
- Bedingungen, Zeitplan, Sichtbarkeit und Alert-Regeln

Der zweistufige Create-Flow ist besonders gelungen. Schritt 1 konzentriert sich ausschließlich auf die Frage; Schritt 2 zeigt sinnvolle Defaults und versteckt Komplexität in „Customize schedule and alerts“. „No model run starts until the Watch reaches its scheduled check“ schafft Kostentransparenz.

### 5. Datenschutz und Backend-Robustheit sind überdurchschnittlich

Der Code zeigt bewusste Datenminimierung: keine IP-/User-Agent-Speicherung in zentralen Produktdaten, keine Attachment-Bytes in Firestore, gehashte API- und Idempotency-Keys, sanitisiertes Analytics-Tracking und klar getrennte private/public Shares. Usage, API-Runs und Watch-Scheduler besitzen transaktionale Zustände, Leases und Idempotenz.

Die vollständige Backend-Suite bestand mit 691 Tests. Das ist ein relevantes Vertrauenssignal für ein Solo-/Independent-Produkt.

## Kritische Befunde

### P0: Öffentliche Watch-Seiten können widersprüchliche Wahrheiten zeigen

Der gravierendste Befund betrifft eine reale indexierte Watch:

- Der Questions-Hub zeigte für „Does GPT-5.6 Sol delete files without permission in Codex?“ **100/100 Agreement**.
- Die Watch-Detailseite zeigte **48/100 Agreement**.
- Dieselbe Seite kennzeichnete den Text als **„Original consensus 2026-07-20“**.
- Der Drift-Header sagte **„Stable since last check“**, beschrieb im Text aber einen erkennbaren inhaltlichen Shift.
- Direction Shift und Agreement Change zeigten „—“.
- Das interne Dashboard klassifizierte dieselbe Watch als **Changed**.

Die Ursache ist im Code nachvollziehbar:

- `list_hub_shares()` liest den Agreement Score aus dem unveränderlichen Share-Baseline-Dokument.
- Die Share-Seite legt für ihr Scoreboard den neuesten History-Score darüber.
- Wenn die neueste Vollversion nicht geladen werden kann, fällt der Antworttext auf den ursprünglichen Snapshot zurück.
- Dadurch kann die Seite aktuellen Score und aktuellen Drift mit ursprünglichem Antworttext und ursprünglichen Differences kombinieren.

Das ist kein kosmetischer Fehler. Die Plattform verkauft epistemische Klarheit; widersprüchliche Versionswahrheit beschädigt exakt dieses Versprechen.

**Empfehlung:**

1. Eine autoritative `display_version` bestimmen, bevor irgendein Score, Modell, Source Count, Differences- oder Driftwert gerendert wird.
2. Alle sichtbaren Daten müssen aus derselben Version stammen.
3. Kann die aktuelle Vollversion nicht geladen werden, klarer Fail-closed-Zustand: „Latest check metadata is available, but the current full answer could not be loaded. Showing the original baseline.“ Keine Überlagerung des aktuellen Scores.
4. Questions-Hub für Watches aus der neuesten vollständigen Version speisen; alternativ beide Werte explizit zeigen: „Current 48 · Original 100“.
5. „Stable since previous check“ und „Changed since original baseline“ als zwei getrennte Konzepte behandeln.
6. Regressionstests für fehlende Vollversion, Legacy-History, erste Watch-Ausführung und Hub/Detail-Konsistenz ergänzen.

### P0: Pricing, Limits und Pro-Kommunikation widersprechen der Live-Konfiguration

Im Live-Account waren 50 Pro-Runs, 5 Deep-Think-Runs und 10 aktive Watch-Slots sichtbar. Gleichzeitig enthält der Pro-Dialog harte Werte wie:

- Free: 25 Fast Queries pro Tag
- Pro: 100 Fast Queries pro Tag
- Pro: 25 Deep Think pro Tag
- Preis: 10 Euro pro Monat
- „Pricing preview, not active yet“

Der Code-Default ist dagegen 3 Free-Runs, 500 Pro-Runs, 1 Free-Watch und 5 Pro-Watches; Firestore überschreibt diese Werte im Live-Betrieb erneut. `watch.js` behauptet hart „Pro includes 5 active Watches“, während das Dashboard 10 Slots zeigt. Auch die FAQ nennt weiterhin 25 Standardanfragen pro Tag.

**Folge:** Nutzer können weder Leistung noch Preis verlässlich verstehen. Zudem führt „Upgrade“ nicht in einen Kaufprozess, sondern in eine Interessensregistrierung per Alert.

**Empfehlung:**

- Alle Plantexte aus einem serverseitigen `PLAN_CATALOG` rendern: Limits, Features, Frequenzen, Preis, Verfügbarkeit und CTA.
- Firestore darf Werte überschreiben, aber nicht die Semantik des Plans fragmentieren.
- Solange Billing nicht aktiv ist: Button „Request Pro access“ oder „Join Pro beta“, klare Erwartung und Bestätigungszustand im Modal statt Browser-Alert.
- Bei aktivem Billing: Checkout, Rechnung, Kündigung, Planstatus, Upgrade/Downgrade und fehlgeschlagene Zahlung als vollständiger Lifecycle.
- Keine Tageskontingente oder Preise versprechen, bevor p50/p95-Kosten pro vollständigem Run und pro Watch bekannt sind. Ein Run umfasst mehrere Provider sowie Consensus/Judge-Schritte; 10 Euro bei sehr hohen Tageslimits kann wirtschaftlich gefährlich sein.

### P0/P1: Bookmark-Architektur skaliert nicht

Beim Login lädt `GET /bookmarks` jedes Firestore-Dokument vollständig, einschließlich gespeicherter Modellantworten, Consensus, Differences und Quellen. Danach werden alle Einträge sofort in den Sidebar-DOM gerendert.

Im untersuchten Account:

- 337 Bookmarks
- rund 10.873 Pixel Bookmark-Listenhöhe
- 2.234 DOM-Elemente direkt nach App-Start

Das verursacht unnötige Firestore-Reads, große JSON-Payloads, Speicherverbrauch und langsamer werdende Suche/Rendering. Es wird bei Power-Usern oder Teams schnell problematisch.

**Empfehlung:**

- API in Liste und Detail trennen.
- `GET /bookmarks?limit=30&cursor=...` liefert nur ID, Frage, Datum, Typ, Modelllabels, Score und Watch/Share-Status.
- Vollständige Antwort erst beim Öffnen laden.
- Infinite Scroll oder Pagination; DOM-Virtualisierung ab etwa 100 Zeilen.
- Suche serverseitig oder über einen kleinen lokalen Metadatenindex.
- Optional Ordner/Tags statt einer einzigen chronologischen Masse.

### P1: Die Marke verspricht Monitoring, der Hero aktiviert primär Comparison

Headline und Eyebrow positionieren Monitoring als Kern. Die primäre Hero-Aktion startet jedoch eine einmalige Compare-Demo; „See Watches“ erscheint schwächer und tiefer. Dadurch entsteht eine kleine strategische Lücke zwischen Versprechen und erstem Produktkontakt.

**Empfehlung:** Zwei klar benannte Einstiege direkt im Hero:

- Primary: **Run the live demo**
- Secondary: **Track a question**

Unter der Demo sollte unmittelbar eine Mini-Timeline erscheinen: Baseline → changed result → alert. So wird „track when they move“ bereits im ersten View bewiesen.

### P1: Die App ist für Wiederkehrer stark, aber für Direktbesucher zu leer

Die App reduziert Komplexität vorbildlich, zeigt beim direkten Einstieg aber hauptsächlich eine leere Eingabe. Nutzer aus Search, Link oder Empfehlung sehen nicht sofort:

- welche Fragen sich besonders eignen,
- warum ein Lauf Zeit benötigt,
- was „Fast“, „Balanced“, „High Quality“ und „Agent Mode“ praktisch verändern,
- dass ein Run sechs Antworten plus Consensus umfasst.

**Empfehlung:** Ein kompakter First-Run-State unter oder über der Eingabe:

- „Ask a question worth cross-checking“
- drei Beispielchips: aktuelle Faktenlage, Entscheidung, kontroverse Behauptung
- „6 models · one consensus · sources and disagreements included“
- nach dem ersten Run vollständig verschwinden

Die Sidebar-Usage sollte nicht nur „Runs: 48/50“ zeigen, sondern „48 runs left today“ oder „2 of 50 used“, damit die Richtung eindeutig ist.

### P1: Benchmark ist ein starkes Asset, aber noch nicht vollständig reproduzierbar

Die Benchmark-Seite ist ungewöhnlich ehrlich: Sie nennt überlappende Confidence Intervals, bezeichnet die Spitze als statistischen Gleichstand und trennt Gesamtmenge von Disagreement-Subset. Das schafft Vertrauen.

Gleichzeitig sagt die Seite „nothing hidden“, bietet aber keine öffentliche Rohdatei, kein Manifest, keinen fixierten Prompt, keine Kategorienmatrix, keine Per-Question-Ergebnisse und keinen Code-/Run-Fingerprint an.

**Empfehlung:** Eine „Reproduce / inspect the run“-Sektion mit:

- Run-Datum und Modell-IDs
- System-/Consensus-Prompt
- Sampling-Manifest
- anonymisierte Per-Question-Ergebnisse
- Majority-Vote-Regel
- Konfidenzintervall-Methode
- Git-Commit oder Benchmark-Version
- CSV/JSON-Download

Das würde aus einer Marketingseite ein belastbares Trust Asset machen.

### P1: Öffentliche Ergebnis- und Quellenansicht ist zu lang und teilweise zu technisch

Die untersuchte mobile Watch-Seite war rund 5.800 Pixel hoch. Die Struktur ist grundsätzlich gut, aber:

- Der Drift-Block dominiert den ersten View, obwohl seine Metriken leer waren.
- Inline-Zitationen erscheinen teils als wiederholtes „openai“ statt als klare Nummer/Quelle.
- Die Vollzitation enthält sehr lange Redirect-URLs und eine kaum scanbare Quellenliste.
- Quellen unterschiedlicher Qualität stehen visuell nahezu gleichberechtigt nebeneinander.

**Empfehlung:**

- Primärquellen, Fachmedien und Community-Quellen markieren.
- Redirect-URLs normalisieren und deduplizieren.
- Inline-Zitate als `[1]`, `[2]` mit Titel/Domain im Hover bzw. Popover.
- Zitation standardmäßig kompakt; Export als BibTeX/Markdown/APA statt langer Textwand.
- Drift-Block bei fehlenden Vergleichsmetriken auf eine kompakte Statuszeile reduzieren.

### P2: Questions-Hub ist eine Liste, noch kein Intelligence-Produkt

Der Hub ist für SEO und interne Verlinkung sinnvoll, aber 41 Fragen in einer langen, gemischten Liste erzeugen wenig Orientierung. Legacy-Einträge haben keinen Score, aktive Watches zeigen nicht zuverlässig den aktuellen Score, Themen und Movement fehlen als Filter.

**Empfehlung:**

- Tabs: Latest movement, Most disputed, Newly tracked, All
- Topic-Filter: AI products, policy, science, health, markets, other
- Watch-Zeile: aktueller Score, Veränderung seit Baseline, letzte Prüfung, Frequenz
- redaktionelle „What moved this week?“-Zusammenfassung
- gespeicherte Suche bzw. Follow-CTA direkt aus dem Hub

Damit wird der Hub vom SEO-Verzeichnis zum öffentlichen Beweis des Produkts.

### P2: About-Seite ist ehrlich, aber unterverkauft das Produkt

„Small independent tool“, „public side project“, „no marketing team“ und „iterated as time and budget allow“ sind sympathisch, können aber professionellen Nutzern und Käufern mangelnde Verlässlichkeit signalisieren. Das kollidiert mit der anspruchsvollen Observatory-Positionierung.

**Empfehlung:** Nicht größer wirken als man ist, aber souveräner formulieren:

- „Independent product built and maintained by [Name]“
- klare Mission und redaktionelle Prinzipien
- Statuspage, Changelog und Kontakt-/Support-Erwartung
- Security- und Data-Principles
- öffentliche Methodik und Roadmap

„Independent“ ist ein Vorteil; „side project“ sollte nicht die zentrale Risikobotschaft sein.

### P2: Mobile Bedienung ist schön, aber mehrere Touch-Ziele sind zu klein

Die mobile Darstellung funktioniert ohne horizontales Scrollen und ist visuell sauber. Mehrere zentrale Controls liegen jedoch bei ungefähr 25 bis 36 Pixeln Höhe/Breite: View-Switch, Attachment, Preset, Delete und Send. Für zuverlässige Touch-Bedienung sollten interaktive Zielbereiche ungefähr 44 × 44 Pixel besitzen, auch wenn das sichtbare Icon kleiner bleibt.

Zusätzlich:

- Der Watch-Tab wird mobil auf ein unlabeled Auge reduziert; das ist elegant, aber wenig selbsterklärend.
- Das Agent-Mode-Checkbox-Label wechselt zu „Disable Agent Mode“, obwohl ein Checkbox-/Switch-Name den Zustand und nicht die Aktion beschreiben sollte.
- Die Actions liegen innerhalb der Consensus-Überschrift; dadurch lautet der Screenreader-Heading sinngemäß „Consensus Answer Share… Watch… Copy…“.

**Empfehlung:** größere unsichtbare Hit-Areas, sichtbarer Tooltip/Coachmark beim Watch-Icon, Switch mit Name „Agent Mode“ plus `aria-checked`, Controls aus dem `<h2>` herauslösen.

## Produktstrategie: Wo consens.io gewinnen kann

### Der Compare-Markt ist voll

Mehrere aktuelle Produkte bieten bereits „ask once, compare models“:

- [Multii Chat](https://multii.chat/) bündelt bis zu sechs Modelle und verkauft eine zentrale Subscription.
- [League of LLMs](https://leagueofllm.com/) kombiniert Side-by-Side-Antworten, AI Judge und Sharing.
- [ConsensusAI](https://consensusai.cloud/) verwendet nahezu dieselbe Sprache aus Multi-Model-Consensus, Fact-Checking und Confidence Score.
- [Definitive](https://definitive.sh/) verspricht vier Modelle, Synthese und einen finalen Verdict.

Auch der Name hat Verwechslungsrisiko: [Consensus.app](https://help.consensus.app/en/articles/9922673-how-consensus-works) ist eine etablierte wissenschaftliche Suchplattform mit eigenem „Consensus Meter“.

Ein generischer Vergleichsclaim wird deshalb langfristig teuer und austauschbar.

### Monitoring ist die bessere Wedge

Der Markt für wiederkehrendes AI-Answer-Monitoring entsteht gerade, wird aber häufig nur aus GEO-/Brand-Sicht besetzt. [Orbilo](https://orbilo.co/features/prompt-tracking) und [Tracking LLM](https://trackingllm.com/tools/answer-drift-monitor/) sprechen beispielsweise Marketing- und Visibility-Teams an. consens.io besitzt eine breitere und nutzerfreundlichere Interpretation: Nicht nur „Wird meine Marke erwähnt?“, sondern „Wie verändert sich die inhaltliche Antwort mehrerer Modelle auf eine relevante Frage?“

Das öffnet drei Segmente:

1. **Individual Pro:** Researcher, Analysten, Journalisten, Studierende, Entscheider.
2. **Teams:** Policy, Kommunikation, Competitive Intelligence, Trust & Safety, Produktmanagement.
3. **API/Publisher:** wiederkehrende öffentliche oder interne Consensus-Reports, Webhooks und eingebettete Monitoring-Flows.

### Empfohlenes Zielbild

**Category:** AI Answer Intelligence  
**Promise:** Compare what leading AI models say, understand where they disagree, and know when the answer changes.  
**Primary object:** die überwachte Frage, nicht der Chat-Thread.  
**Primary retention mechanism:** Watch + Brief + Alert.  
**Acquisition loop:** öffentliche Frage → Quellen/Verlauf → Follow → eigene Frage → Watch → Alert → erneuter Besuch/Share.

Comparison bleibt wichtig, aber vor allem als Aktivierungs- und Vertrauensmechanismus für Monitoring.

## Empfohlene Informationsarchitektur

### Public

- Product
- Live Questions
- How it works
- Benchmark
- Pricing oder Pro beta
- About

### App

- Ask
- Watches
- Library
- Modelle und technische Einstellungen nachrangig

Die aktuelle Sidebar trennt Models, Leaderboard und Bookmarks. Für das Zielbild sollte „Library“ der primäre gespeicherte Arbeitsraum sein; Model Controls gehören eher in Query-Kontext oder Settings. Leaderboard ist interessant, aber nicht zentral für die Nutzeraufgabe.

### Watch-Detail

Oben immer ein expliziter Version Switch:

`Current` · `Original` · `History`

Darunter:

1. aktueller Status und geprüfter Zeitpunkt
2. Änderung seit vorherigem Check
3. kumulative Änderung seit Original
4. aktueller Consensus
5. Differences und Quellen
6. Verlauf/Position Map

Damit werden „previous“ und „baseline“ nicht mehr vermischt.

## Copy-Vorschlag für den Hero

**Eyebrow**  
The independent AI answer observatory

**Headline**  
Ask once. See where AI models disagree — and know when the answer changes.

**Subline**  
consens.io compares answers from leading AI models, synthesizes the common ground, and alerts you when the consensus moves.

**Primary CTA**  
Run the live demo

**Secondary CTA**  
Track a question

Die aktuelle emotionale Headline kann alternativ bestehen bleiben, wenn direkt darunter eine sehr konkrete Subline ergänzt wird. Poetry erzeugt Interesse; die Subline muss das Produkt in einem Satz erklären.

## Monetarisierung

Die Preisfrage sollte nicht zuerst über Wettbewerber, sondern über tatsächliche COGS entschieden werden. Ein vollständiger Run kann sechs Provider-Antworten, Consensus, Differences-Judge und bei Watches zusätzliche Change-Auswertung enthalten. Deshalb sind hohe Tageslimits bei niedrigem Fixpreis riskant.

Vor Pricing-Freigabe messen:

- p50/p95 Providerkosten pro Fast/Balanced/High-Quality-Run
- p50/p95 Watch-Kosten je Tier
- Anteil eigener API-Keys
- durchschnittliche Runs und aktive Watches je Nutzer
- fehlgeschlagene/abgebrochene Runs mit Kosten
- Gross Margin pro Kohorte

Mögliche Verpackung:

- **Free:** Demo, wenige tägliche Runs, eine wöchentliche Watch, öffentliche Library.
- **Pro Individual:** Frontier-Modelle, Follow-ups, Resolve, Files, mehrere/daily Watches, Morning Brief.
- **Team/Monitor:** mehr Watches, Folder, Rollen, Slack/Webhooks, CSV/API, Audit Trail, gemeinsame Dashboards.

Bis Self-Service-Billing existiert, sollte das Produkt explizit als Pro Beta auftreten und keine aktive Subscription vortäuschen.

## 90-Tage-Roadmap

### Phase 1: Trust Repair, Woche 1–2

- Version/Score/Drift-Invariante für Watches
- Hub zeigt aktuelle Watch-Daten oder explizit Original + Current
- Plan-Katalog als Single Source of Truth
- veraltete FAQ-/Pricing-Texte entfernen
- Upgrade zu Pro-Beta-Flow umbenennen
- Touch-Ziele und Heading-/Switch-Semantik korrigieren
- Benchmark-Manifest und Download veröffentlichen

### Phase 2: Activation und Public Loop, Woche 3–6

- App-First-Run-State mit Beispiel-Fragen
- Hero mit Track-CTA und Mini-Timeline
- Questions-Hub zu „Latest movement“ erweitern
- Current/Original/History Switch
- Quellenqualität und kompakte Zitierform
- Follow-Conversion auf öffentlichen Watches messen und verbessern

### Phase 3: Retention und Revenue, Woche 7–12

- Bookmark List-/Detail-Pagination
- Ordner/Tags für Library und Watches
- saubere Pro-Beta- oder Billing-Einführung
- Team-Wedge testen: Policy/Competitive Intelligence/Communications
- Slack/Webhook als erster Team-Kanal
- COGS-basierte Limits und Preisgestaltung

## Messplan

### North-Star-Metrik

**Weekly active monitored questions**: Anzahl unterschiedlicher Watches, die in einer Woche erfolgreich geprüft und von ihrem Owner oder Follower konsumiert wurden.

Diese Metrik verbindet wiederkehrenden Produktwert mit dem differenzierenden Kern.

### Activation

- Landing → Demo Start
- Demo Start → Demo Complete
- Demo Complete → erste eigene Frage
- erste Frage → erfolgreicher Consensus
- erster Consensus → Watch erstellt
- Time to First Consensus
- Time to First Watch

### Trust

- Source Click Rate
- Differences Open Rate
- Citation Copy Rate
- Original/Current/History Switch Rate
- Share-Reports
- technische Invariante: 0 Seiten mit gemischten Versionen

### Retention

- D1/D7/D30 Wiederkehr
- Anteil Nutzer mit mindestens einer aktiven Watch
- Watch-Alert Open/Click Rate
- Morning-Brief-Aktivierung und Unsubscribe
- Anteil Stable/Changed/Condition Alerts
- zweiter und vierter erfolgreicher Watch-Lauf

### Revenue und Guardrails

- Locked Feature → Pro Interest/Checkout
- Checkout Completion
- COGS pro aktiver Person und Watch
- Gross Margin
- Provider Error Rate
- Consensus/Watch Failure Rate
- Alert False-Positive Feedback
- öffentliche Report-Rate

Umami ist bereits datensparsam integriert und viele App-Events existieren. Der nächste Schritt ist kein neues Tracking-Tool, sondern ein konsistentes Funnel-Schema und ein kleines wöchentliches Product Dashboard.

## Technisches Briefing

### Stärken

- klare FastAPI-Router und Services
- transaktionale Usage-Reservierung
- API-Idempotenz und persistente Run-State-Machine
- Watch-Leases, Budgets und Retry-Pfade
- sanitisiertes Markdown und CSP
- private/public Snapshot-Trennung
- datensparsame Telemetrie
- 691 bestandene Backend-Tests

### Risiken

- Frontend-Kommunikation über viele `window.*`- und DOM-State-Verträge
- große Module: `watch.js` und `app-init.js` jeweils rund 75 KB, `consensus-insights.js` rund 58 KB
- alle App-Module werden auf jeder App-Ansicht geladen
- `index.html` rund 80 KB, Admin-Template rund 148 KB
- Bookmarks ohne Pagination und List-/Detail-Trennung
- Frontend-E2E deckt nur ausgewählte Kernpfade ab
- 33 Deprecation-Warnungen, hauptsächlich Starlette `TemplateResponse`-Signatur
- CSP enthält weiterhin `unsafe-inline`; langfristig Nonces/Hashes und ausgelagerte Inline-Skripte anstreben

Ein Framework-Rewrite ist nicht die erste Priorität. Zuerst sollten Datenkonsistenz, Billing, History-Skalierung und klare Produktobjekte stabilisiert werden. Danach kann schrittweise auf ES-Module, typisierte API-Verträge und komponentisierte UI migriert werden.

## Konkrete Entscheidungsvorlage

Wenn nur drei Initiativen finanziert werden, sollten es diese sein:

1. **Trust Layer reparieren:** jede Watch-Seite zeigt eine einzige, konsistente Version; Benchmark und Quellen werden nachvollziehbarer.
2. **Watch als Hauptprodukt führen:** Hero, App und Library orientieren sich an überwachten Fragen statt an Chat-Historie und Modellauswahl.
3. **Kommerziellen Vertrag klären:** ehrliche Pro Beta oder vollständiges Billing, dynamischer Plan-Katalog und COGS-basierte Limits.

Das visuelle Design benötigt keinen großen Neustart. Es ist bereits hochwertig. Der größere Hebel liegt in der Produktwahrheit: Was ist aktuell? Was hat sich verändert? Was kostet es? Für wen ist es unverzichtbar?

consens.io besitzt bereits die Bausteine eines starken SaaS-Produkts. Jetzt muss aus einer beeindruckenden Feature-Sammlung ein konsistenter, vertrauenswürdiger und kaufbarer Answer-Intelligence-Workflow werden.
