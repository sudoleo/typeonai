# Smoke-Checkliste — Frontend (index.html Refactor)

Teilweise automatisiert: die Playwright-Suite `tests/e2e/` deckt Konsolen-
Fehler beim Laden, Send→Streaming, Consensus→Differences+Agreement-Score,
Modell-Ausschluss, Theme-Toggle und Picker-Persistenz ab (Lauf: siehe
`tests/e2e/README.md`). Die übrigen Punkte weiterhin manuell durchgehen
(oder zumindest die vom Cluster betroffenen), bevor committet wird. Backend
bleibt durch `venv/Scripts/python -m pytest tests/` abgesichert
(Baseline: 718 passed, Stand 2026-07-24).

## Browser-Konsole
- [ ] Beim Laden **keine** JS-Fehler in der Konsole (besonders: keine
      `ReferenceError: X is not defined`, keine `window.X is not a function`).

## Öffentliche Seiten
- [ ] `/`, `/about`, `/ai-model-comparison`, `/consensus-engine`, `/benchmark`,
      `/privacy`, `/terms`, `/imprint` und öffentliche Share-/Unavailable-Seiten
      verwenden dieselbe Navigation, denselben Footer und die an `/app`
      angelehnten Tokens. Light/Dark folgen der gespeicherten App-Einstellung
      bzw. ohne Einstellung dem System-Theme.
- [ ] Desktop und Mobile haben keinen horizontalen Overflow; Focus States sind
      auf Links, Buttons und Formularfeldern klar sichtbar. Landingpage und
      Consensus-Engine-Seite zeigen dieselbe aktuelle Consensus-/Differences-
      Darstellung. Der Landingpage-Walkthrough verwendet die aktuellen
      Modellnamen und hält Einzelantworten im Agent Mode standardmäßig verborgen.

## Kern-Flow
- [ ] Frischer `/app`-Load passt ohne vertikales Scrollen in den Desktop-
      Viewport; der Consensus-Picker hat keinen horizontalen Scrollbalken.
- [ ] Frischer `/app`-Load: keine Topbar; Brand + Collapse im Sidebar-Kopf,
      eingeloggter Account (Name/Plan + Avatar) und Settings im Sidebar-Footer.
      Ausgeloggt stehen Login/Sign-up nur oben rechts; die Sidebar zeigt kein
      zweites Login-Feld. Das Account-Popup hat in Light und Dark einen
      vollständig deckenden, gut lesbaren Hintergrund.
      Das Eingabefeld steht mit Begrüßung mittig; nach dem ersten Senden gleitet
      es nach oben und die unveränderten Modell-Antwortboxen blenden ein.
- [ ] Sidebar-Navigation: Models ist eine einzelne kompakte Zeile mit
      Providerzahl und öffnet den Run-Picker am Composer; sie klappt keine
      sechs Providerzeilen auf. Der Custom-Picker nutzt Checkboxen statt
      Toggle-Switches und bleibt in Light/Dark vollständig deckend und lesbar.
      Bei offener Desktop-Sidebar bleibt das Eingabefeld in der Viewport-Mitte;
      mobil verschwindet die schwebende Brand vollständig.
- [ ] Die Landingpage verlinkt direkt im Hero mit einer schmalen Live-Zeile auf
      `/model-pulse`. Die eigene Seite erklärt „Best answer“ als anonymisierte
      Judge-Auswahl (kein Benchmark/User-Vote/Accuracy-Score), führt
      Anthropic/Claude nicht doppelt und verlinkt den kontrollierten Benchmark;
      `/benchmark` verlinkt seinerseits sichtbar zurück auf den Model pulse.
- [ ] Settings: Experience, Connections, Model behavior und Account sind als
      klar getrennte Kategorien erkennbar; alle Schalter, API-Key-Felder,
      System Prompt und Account-Löschung funktionieren weiterhin.
- [ ] Frage eingeben + senden → alle ausgewählten Modelle streamen Antworten.
- [ ] Bei null oder einem ausgewählten Modell ist Senden deaktiviert; Sidebar-
      Zähler und Custom-Picker nennen „choose at least 2“. Ab zwei Modellen startet
      der Lauf normal und endet in Consensus + Differences.
- [ ] Ohne Agent Mode erscheint direkt unter dem Input die kompakte Pipeline:
      Zähler folgt den fertigen Modellantworten, danach wird „Consensus &
      differences“ ohne falsche Prozent-/Zeitprognose aktiv; Abschluss, Fehler
      und Stop blenden die Zeile wieder aus. Light/Dark und Mobile ohne Clipping.
- [ ] Senden während Lauf abbrechen (Stop) funktioniert.
- [ ] Modell per Checkbox ein-/ausschließen blendet die Antwortbox korrekt ein/aus.
- [ ] Echter Bild-/PDF-Anhang pausiert DeepSeek mit sichtbarer Erklärung; nach
      Entfernen aller Anhänge wird die vorherige DeepSeek-Auswahl wiederhergestellt.
- [ ] Quellen-Chips / Evidence-Links erscheinen und sind klickbar.

## Consensus (höchstes Risiko)
- [ ] Presets: Fast/Balanced setzen sichtbar alle sechs Antwortmodelle und die
      konfigurierte Consensus-Engine; eine manuelle Modellwahl wechselt zu Custom.
- [ ] High Quality zeigt ein Pro-Badge, hat beim Hover/Fokus eine dezente
      Power-Animation, oeffnet fuer Free den Kosten-Erklaerdialog (kein Kauf-,
      kein Zugangs-Request-Button) und setzt fuer Pro das vollstaendige
      Premium-Model-Set. Deep Think bleibt separat.
- [ ] Consensus und Differences erscheinen oberhalb der Modellantworten; der
      Reveal scrollt nur dann sanft zum Ergebnis, wenn es außerhalb des
      relevanten Viewports liegt.
- [ ] Consensus manuell generieren → Antwort + Differences erscheinen.
- [ ] Auto-Consensus (Toggle an) triggert automatisch nach Abschluss.
- [ ] Credibility-Frame-Farbe (cred-very … cred-not) wird gesetzt.
- [ ] Consensus-Insights: Claim-Badges, Difference-Karten, Klick öffnet Popover,
      „Jump to model answer" highlightet die Originalantwort.
- [ ] Verdict-Semantik: Score 85+/65+/40+/20+/<20 zeigt High/Strong/Partial/
      Low/Very low agreement; Grün beginnt erst bei 65. „No contradictions"
      bzw. disputed/critical/minor bleiben als getrennte Detailaussage sichtbar.
- [ ] Resolve-Runde: „Resolve with the models"-Button an Widerspruchs-Karten
      (nur Contradictions mit ≥2 beteiligten Modellen), Klick zeigt Outcome-Badge
      + Modell-Zeilen, Usage-Counter aktualisiert sich, Fehlerfall reaktiviert
      den Button.
- [ ] Spalten-Balancer: Differences-Spalte passt Breite an.
- [ ] Share-Dialog: Link erstellen, Liste anzeigen, Link kopieren.

## Agent Mode
- [ ] Agent-Mode an/aus, Timer läuft, Status-Text korrekt, Auto-Consensus-Kopplung.
- [ ] Nach der ersten fertigen Modellantwort erscheint dezent „Compare answers“
      (auch im eingeklappten Mobile-Panel); der Toggle zeigt/versteckt die
      einzelnen Antwortboxen, ohne Agent Mode auszuschalten, und startet bei
      einer neuen Frage wieder in der cleanen, verborgenen Ansicht.
- [ ] Mobile Consensus: Tipp auf eine unterstrichene Passage öffnet denselben
      Agreement-Dialog wie die Quote; Fokus bleibt im Dialog und kehrt zurück.
- [ ] Mobile Footer: Score, Aktionen und die drei Detail-Tabs bleiben kompakt,
      gleichmäßig ausgerichtet und erzeugen keinen horizontalen Scroll.
- [ ] Mobile Composer: Plus, Modell-Picker und Send-Pfeil liegen auf derselben
      horizontalen Achse; der Composer bleibt am unteren Viewport-Rand fixiert.
      Das Fragefeld wächst beim Tippen bis 180 px, scrollt danach intern und
      schrumpft beim Löschen wieder auf seine Ausgangshöhe.
      Am vollständigen Scrollende liegen die geschlossenen Detail-Tabs direkt
      darüber, ohne Leerraum oder verdeckten Inhalt.
- [ ] Quellen-Fussnoten im Consensus stehen hinter Punkt, Frage- oder
      Ausrufezeichen; dasselbe gilt auf öffentlichen Share-Seiten.
- [ ] „Run again“ kehrt zum normalen Composer zurück, übernimmt die vorige
      Frage, startet aber erst nach einem bewussten Klick auf Senden. Der Knopf
      beziffert vorher den Preis („Run again · uses 1 run“, Tooltip mit Rest-
      Kontingent); bei unbegrenztem Plan entfällt der Zusatz. Nach dem Klick
      steht über dem Eingabefeld, dass Senden einen vollständigen neuen Lauf
      startet — der Hinweis verschwindet mit dem Absenden oder mit
      „New comparison“.

## Consensus Watch
- [ ] Nach erfolgreichem Consensus erscheint „Watch“ neben Share; Aktivierung
      verlangt die explizite Wahl zwischen privater Eigentümer-Seite und öffentlicher,
      nicht indexierter Link-Seite und bietet Weekly/Monthly. Ein Klick auf „Start
      watching“ markiert fehlende Pflichtangaben direkt am jeweiligen Feld und scrollt
      zum ersten Fehler. Der Dialog bleibt auf iPhone-Größen vollständig im sichtbaren
      Bereich. Private Seiten sind in einem fremden oder ausgeloggten Browser nicht lesbar.
- [ ] „Ready with smart defaults“ trägt rechts einen „Edit“-Schalter, die drei
      Werte-Chips öffnen selbst ihr Feld (Fokus liegt danach darin), und
      „Customize schedule and alerts“ steht direkt darunter — über den
      Zustellkanälen, nicht am Dialogende.
- [ ] Lokale Run-Uhrzeit ist bei Erstellung wählbar und zeigt die erkannte Zeitzone;
      Weekly bietet auch Free-Nutzern einen Wochentag-Picker und startet standardmäßig
      am morgigen Wochentag statt erst nach einer vollen Woche. „Watched“ erlaubt eine
      spätere Änderung von Tag und Uhrzeit. `next_run_at` entspricht dem gewählten
      lokalen Wochentag und der Uhrzeit (mit bis zu 30 Minuten Scheduler-Toleranz),
      auch über einen Sommer-/Winterzeitwechsel hinweg.
- [ ] Free: Daily ist als Pro markiert/gesperrt und das aktive Limit öffnet den
      bestehenden Pro-Teaser. Dashboard und Create-Dialog zeigen vorher den
      autoritativen Plan, „aktiv von Limit“ und freie Plätze; pausierte Watches
      sind ausdrücklich als nicht mitgezählt erklärt. Bei 1/1 (Free) bzw. 5/5
      (Pro) ist die Erstellung bereits vor dem Request gesperrt. Pro: Daily und
      bis zu fünf aktive Watches funktionieren.
- [ ] Das Watch-Dashboard ist eine eigene Seite `/app/watches`: erreichbar über
      den schwebenden View-Switch „Consensus | Watches“ (nur eingeloggt, Watches
      auf Mobile icon-only) und „Watched“ im Nutzericon-Menü; aktiver Pill-Zustand,
      Browser-Back/Forward
      und Deep-Link/Reload auf `/app/watches` funktionieren (vor dem Login
      erscheint ein Hinweis statt Daten). Ohne Watch zeigt die Seite statt Null-
      KPIs einen Ask→Check→Alert-Empty-State mit optionalen Beispielfragen; der
      Query-first-Dialog nutzt private/wöchentliche/Changes-only-Defaults, hält
      E-Mail und Telegram sichtbar und legt Zeitplan/Sichtbarkeit/Condition unter
      „Customize schedule and alerts“. Die Vorschlags-Chips bleiben optisch sekundär;
      Empty-State und Überschrift haben auf Desktop und Mobile ausreichend Luft. Ein
      Klick auf den Dialog-Backdrop schließt den Watch-Dialog nicht, X/Cancel/Back
      und Escape weiterhin schon. KPI-Karten zeigen sonst aktive Watches,
      Checks/Änderungen der letzten sieben Tage und den nächsten Lauf; ein
      „Recent movement“-Feed bündelt die neuesten Changes. All/Changed/Stable/Paused
      filtern die Karten. Pro Watch zeigt die Karte Driftstatus/-Summary,
      Direction Shift, Agreement-Score + Delta, History-Sparkline und nächsten Lauf.
      Der Notifications-Bereich klappt Telegram und Morning Brief gemeinsam ein und
      aus; sein Zustand bleibt beim erneuten Öffnen des Dashboards erhalten.
      „Settings“ klappt
      Intervall/Uhrzeit/Alert-Regel/Condition sowie E-Mail-/Telegram-Kanäle auf;
      Pause/Resume und Delete
      funktionieren. Delete lässt bereits vorhandene Share-History bestehen.
      „← Back to app“ und ESC führen zurück. Light/Dark und Mobile ohne Overflow.
- [ ] Der Watches-Schalter pulsiert als neuer, unbestätigter Einstieg nur zweimal
      dezent, stoppt nach dem ersten Öffnen dauerhaft und ist bei reduzierter Bewegung
      still. Er konkurriert nicht mit dem resultatspezifischen Watch-Hinweis.
- [ ] Morning Brief (Karte im Dashboard): Toggle aktiviert die tägliche
      Digest-Mail mit Uhrzeit (Browser-Zeitzone) und Modus „Every morning“ /
      „Only when something changed“; Einstellungen überleben ein erneutes
      Öffnen. Mit Test-SMTP: Brief-Mail listet alle Watches mit Score/Delta und
      Änderungs-Summaries; der Abmelde-Link deaktiviert nur den Brief, nicht
      die Watch-Mails.
- [ ] Ohne Watch ist der Morning-Brief-Toggle deaktiviert und erklärt „Create a
      watch first“; ein direkter Aktivierungs-Request wird abgelehnt. Nach dem
      Löschen der letzten Watch ist ein zuvor aktiver Brief ausgeschaltet.
- [ ] Aktive Watch-Seite erklärt vor dem ersten Vergleich verständlich, dass
      erst eine Baseline vorliegt; Status bleibt sichtbar, Zeitplan/letzter/
      nächster Lauf sind über „Schedule and check dates“ erreichbar. Mit History rendert
      sie den neuesten gespeicherten Consensus statt des ursprünglichen Texts,
      einen Stable/Changed-Drift-Header; Direction Shift und Agreement Change
      stehen erst in „advanced change metrics“. Die komplette Timeline mit
      SVG-Linie/Punkten und Change-Liste startet eingeklappt und funktioniert in
      Light/Dark ohne Mobile-Overflow. Jede neue Vollversion ist über „Browse saved
      consensus versions“ erreichbar; `?version=original` zeigt unverändert die
      Ausgangsversion. Eine normale Shared Page ohne Watch bleibt unverändert.
      Neue History zeigt davor die mehrdimensionale Position Map mit Provider-
      Trajektorien, aktuellen Standpunkt-Gruppen und Direction Shift; alte
      Punkte ohne `opinion_map` degradieren auf den Agreement-Chart.
- [ ] Fehlende SMTP-Konfiguration blockiert Watch-Läufe nicht. Mit Test-SMTP:
      Major Change bzw. Score-Delta ≥15 sendet genau eine Multipart-Mail; Minor
      Change darunter sendet im Modus „changes only“ keine. „Every new consensus“
      sendet bei jedem erfolgreichen Lauf genau eine Mail mit Consensus-Inhalt.
      Eine Condition sendet nur bei `not met -> met` (bzw. beim ersten `met`), nicht
      erneut bei weiter bestehendem `met`; `unknown` löst nicht aus. Die Mail enthält
      Condition, Begründung und neuen Consensus.
      Abmelde-Link pausiert ohne Login.
- [ ] Zwei aufeinanderfolgende Watch-Runs vergleichen Previous → Current für
      Benachrichtigungen; Original → Current bleibt als langfristiger Baseline-
      Drift erhalten. Persistierte Ereignisse bleiben grob und eindeutig:
      `watch.checked`, `watch.changed`, `watch.condition_met`, `watch.run_failed`.
- [ ] Mit gesetztem `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` und
      `TELEGRAM_WEBHOOK_SECRET`: „Connect Telegram“ öffnet den Bot, `/start`
      verbindet ausschließlich den eingeloggten Account und das Dashboard zeigt
      danach Identität, „Send test“ und „Disconnect“. Abgelaufene oder erneut
      verwendete Deep-Links werden abgelehnt.
- [ ] Telegram lässt sich beim Erstellen und je Watch an-/abschalten; mindestens
      E-Mail oder Telegram bleibt aktiv. Ein materieller Change erzeugt genau
      eine Telegram-Nachricht mit Score/Änderung und Buttons. „Mute 24h“
      unterdrückt weitere Telegram-Alerts, „Pause“ verlangt eine zweite
      Bestätigung und pausiert nur die eigene Watch. Ein erneuter Scheduler-
      Versuch für dieselbe Run-ID verschickt kein Duplikat.

## Curated Topics
- [ ] `/topics` zeigt nur Topics mit mindestens einem veröffentlichten Snapshot;
      Suche und Kategorie-Filter funktionieren ohne Reload. Navigation, Footer,
      Light/Dark, Focus States und Mobile-Layout bleiben ohne horizontalen
      Overflow.
- [ ] `/topics/{slug}` zeigt das vollständige Dossier und die Timeline beim
      ersten Laden bereits geöffnet. Die fünf bestgereihten aktuellen Quellen
      sind sichtbar, alle weiteren starten in einem eigenen Detail eingeklappt;
      ältere Evidence bleibt separat eingeklappt. Quellen tragen die Rollen
      Primary source, Research paper, Documentation, Reporting, Community oder
      Rumor und sind nach Qualität sortiert. Alle Links öffnen sicher in einem
      neuen Tab.
- [ ] Die visuelle Timeline zeigt alle versionierten Stände und ihre Agreement-
      Entwicklung. Ein historischer `?version=<run_id>`-Link rendert den
      unveränderlichen alten Consensus samt damaligen Modellen/Evidence,
      kennzeichnet die historische Ansicht und verlinkt zur aktuellen Version.
- [ ] Topic-Follow sendet mit Test-SMTP eine Double-Opt-in-Mail; erst der
      Bestätigungslink persistiert das Abo. Ein neuer Minor-/Major-Snapshot
      versendet genau ein Update, Stable nicht; der Abmelde-Link entfernt nur
      das Topic-Abo und verändert weder Nutzer-Watches noch Share-Follower.
- [ ] `/admin#topics`: Create/Edit, Slug/Kategorie/Intervall, Status
      Active/Paused/Archived, konkrete Modelle je Provider, Quellenpräferenzen
      und SEO lassen sich verständlich speichern. `/admin/topics` leitet auf
      diesen Tab um.
- [ ] „Run now“ benötigt keinen manuell eingetragenen Consensus, Score,
      Evidence-Link oder Opinion Change. Der Lauf recherchiert aktuelle Quellen,
      führt nur die ausgewählten Modelle aus und legt eine neue unveränderliche
      Version mit Agreement, Change-Summary, Opinion Map und Evidence an. Ein
      zweiter Run vergleicht gegen den ersten Stand.
- [ ] Daily/Weekly/Biweekly/Monthly setzt einen sichtbaren `next_run_at`; der
      Topic-Scheduler führt fällige aktive Topics aus. Manual setzt keinen
      Termin, Paused/Archived laufen weder automatisch noch über „Run now“.
      Bei fehlendem SMTP wird ein erfolgreicher Material-Change-Run gespeichert
      und der Admin transparent auf nicht versendete Updates hingewiesen.

## Modelle / Picker
- [ ] Custom Model Picker öffnet/wählt, sichtbarer Name aktualisiert.
- [ ] Tier-Defaults (Free vs. Pro) werden beim Tier-Wechsel angewandt; eine
      zuvor explizit im Picker gewählte Provider-Auswahl bleibt erhalten.
- [ ] Modell-Auswahl bleibt nach Reload erhalten (localStorage).

## Attachments (Pro)
- [ ] Datei anhängen → Chip erscheint, Vorschau öffnet, Entfernen funktioniert.
- [ ] PNG/JPG/WebP mit Strg+V im Fragefeld einfügen → Bild-Chip erscheint;
      normaler Text-Paste bleibt unverändert möglich.
- [ ] PDF/DOCX/TXT/MD/CSV/PNG/JPG/WebP auf den Input ziehen → Drop-Hinweis
      erscheint und nach dem Ablegen wird der passende Datei-Chip angelegt.
- [ ] Mit einem echten Anhang zeigt der Modell-Picker 5 statt 6 Modelle und
      beim Senden entsteht kein `/ask_deepseek`-Request; nach Entfernen wird
      die vorherige DeepSeek-Auswahl wiederhergestellt.
- [ ] Bookmark-Attachments werden angezeigt.

## Auth / Usage / Tier
- [ ] Login (E-Mail + Google), Logout.
- [ ] Nach Logout verschwinden Account-Label, Kontingent-Ring/-Panel, Usage-
      Zahlen, Watch-Kontingent und Bookmark-Inhalte sofort; Bookmarks und Suche
      sind als Gast nicht klick- bzw. fokussierbar. Auch eine vor dem Logout
      gestartete langsame Usage-/Bookmark-Antwort darf nichts wieder einblenden.
- [ ] Free-User: Usage-Counter + Limit-Anzeige korrekt, Limit-Fehler greift.
- [ ] Pro-User: Premium-Modelle freigeschaltet, UI-Status korrekt.

## Bookmarks / Sidebar
- [ ] Models und Bookmarks beginnen auf derselben Icon-/Textachse und verwenden
      dieselbe Titelgröße/-stärke; im Gastzustand bleibt Bookmarks deaktiviert.
- [ ] Bookmarks laden/aufklappen, Chat-Suche filtert.
- [ ] Bookmark aus dem frischen Leerzustand öffnen: Input dockt ohne Hero-Sprung
      oben an und die gespeicherten Antworten sind direkt sichtbar.
- [ ] Einen gespeicherten Consensus nach Reload öffnen: Share-Link und Watch
      lassen sich ohne erneuten Consensus-Lauf erstellen (während der kurzen
      Vorbereitung zeigt der Dialog einen deaktivierten Ladezustand).
- [ ] Model insights über Help → FAQ öffnen/schließen; im Thread bleibt es verborgen.

## Demo & Sonstiges
- [ ] „Demo"-Query startet den Demo-Flow (demo.js Integration intakt).
- [ ] Demo zeigt 83/100, genau einen kleinen Omega-3-Widerspruch und jede
      Quellenmarke öffnet den fachlich passenden Eintrag (inkl. Creatine S8).
- [ ] Demo erzeugt weder einen Best-answer-Vote noch einen Eintrag in der
      Differences-Telemetrie oder einen Bookmark-Persistenzaufruf.
- [ ] Die Demo tippt erst die vollständige Frage, leert beim simulierten
      Absenden das Eingabefeld und startet danach die Modell-Ladeanimation.
      Nach Abschluss sieht ein ausgeloggter Nutzer eine
      Login-/Registrierungs-Aufforderung; deren Button öffnet das Login-Modal.
      Nach erfolgreichem Login verschwindet die Aufforderung.
- [ ] Dark/Light-Toggle in Settings (Desktop und Mobile).
- [ ] Mobile-Layout (< 768px): Overlay-Sidebar, Info-Popups.
- [ ] System-Prompt-Modal + Help-Modal (app-ui.js) öffnen/speichern.
