# Smoke-Checkliste — Frontend (index.html Refactor)

Teilweise automatisiert: die Playwright-Suite `tests/e2e/` deckt Konsolen-
Fehler beim Laden, Send→Streaming, Consensus→Differences+Agreement-Score,
Modell-Ausschluss, Theme-Toggle und Picker-Persistenz ab (Lauf: siehe
`tests/e2e/README.md`). Die übrigen Punkte weiterhin manuell durchgehen
(oder zumindest die vom Cluster betroffenen), bevor committet wird. Backend
bleibt durch `venv/Scripts/python -m pytest tests/` abgesichert
(Baseline: 498 passed, Stand 2026-07-18).

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
- [ ] Sidebar-Navigation: Models, Leaderboard und Bookmarks tragen ihre Icons
      direkt in der jeweiligen Überschrift; außerhalb der Sidebar gibt es keine
      zweite Icon-Leiste. Bei offener Desktop-Sidebar bleibt das Eingabefeld in
      der Viewport-Mitte; mobil verschwindet die schwebende Brand vollständig.
- [ ] Settings: Experience, Connections, Model behavior und Account sind als
      klar getrennte Kategorien erkennbar; alle Schalter, API-Key-Felder,
      System Prompt und Account-Löschung funktionieren weiterhin.
- [ ] Frage eingeben + senden → alle ausgewählten Modelle streamen Antworten.
- [ ] Ohne Agent Mode erscheint direkt unter dem Input die kompakte Pipeline:
      Zähler folgt den fertigen Modellantworten, danach wird „Consensus &
      differences“ ohne falsche Prozent-/Zeitprognose aktiv; Abschluss, Fehler
      und Stop blenden die Zeile wieder aus. Light/Dark und Mobile ohne Clipping.
- [ ] Senden während Lauf abbrechen (Stop) funktioniert.
- [ ] Modell ein-/ausschließen (Checkbox/Toggle) blendet Antwortbox korrekt ein/aus.
- [ ] Quellen-Chips / Evidence-Links erscheinen und sind klickbar.

## Consensus (höchstes Risiko)
- [ ] Presets: Fast/Balanced setzen sichtbar alle sechs Antwortmodelle und die
      konfigurierte Consensus-Engine; eine manuelle Modellwahl wechselt zu Custom.
- [ ] High Quality zeigt ein Pro-Badge, oeffnet fuer Free das Upgrade-Modal und setzt
      fuer Pro das vollstaendige Premium-Model-Set. Deep Think bleibt separat.
- [ ] Consensus und Differences erscheinen oberhalb der Modellantworten; der
      Reveal scrollt nur dann sanft zum Ergebnis, wenn es außerhalb des
      relevanten Viewports liegt.
- [ ] Consensus manuell generieren → Antwort + Differences erscheinen.
- [ ] Auto-Consensus (Toggle an) triggert automatisch nach Abschluss.
- [ ] Credibility-Frame-Farbe (cred-very … cred-not) wird gesetzt.
- [ ] Consensus-Insights: Claim-Badges, Difference-Karten, Klick öffnet Popover,
      „Jump to model answer" highlightet die Originalantwort.
- [ ] Resolve-Runde: „Resolve with the models"-Button an Widerspruchs-Karten
      (nur Contradictions mit ≥2 beteiligten Modellen), Klick zeigt Outcome-Badge
      + Modell-Zeilen, Usage-Counter aktualisiert sich, Fehlerfall reaktiviert
      den Button.
- [ ] Spalten-Balancer: Differences-Spalte passt Breite an.
- [ ] Share-Dialog: Link erstellen, Liste anzeigen, Link kopieren.

## Agent Mode
- [ ] Agent-Mode an/aus, Timer läuft, Status-Text korrekt, Auto-Consensus-Kopplung.
- [ ] Nach der ersten fertigen Modellantwort erscheint dezent „Show model answers“
      (auch im eingeklappten Mobile-Panel); der Toggle zeigt/versteckt die
      einzelnen Antwortboxen, ohne Agent Mode auszuschalten, und startet bei
      einer neuen Frage wieder in der cleanen, verborgenen Ansicht.

## Consensus Watch
- [ ] Nach erfolgreichem Consensus erscheint „Watch“ neben Share; Aktivierung
      verlangt die explizite Wahl zwischen privater Eigentümer-Seite und öffentlicher,
      nicht indexierter Link-Seite und bietet Weekly/Monthly. Ein Klick auf „Start
      watching“ markiert fehlende Pflichtangaben direkt am jeweiligen Feld und scrollt
      zum ersten Fehler. Der Dialog bleibt auf iPhone-Größen vollständig im sichtbaren
      Bereich. Private Seiten sind in einem fremden oder ausgeloggten Browser nicht lesbar.
- [ ] Lokale Run-Uhrzeit ist bei Erstellung wählbar und zeigt die erkannte Zeitzone;
      Weekly bietet auch Free-Nutzern einen Wochentag-Picker. „Watched“ erlaubt eine
      spätere Änderung von Tag und Uhrzeit. `next_run_at` entspricht dem gewählten
      lokalen Wochentag und der Uhrzeit (mit bis zu 30 Minuten Scheduler-Toleranz),
      auch über einen Sommer-/Winterzeitwechsel hinweg.
- [ ] Free: Daily ist als Pro markiert/gesperrt und das aktive Limit öffnet den
      bestehenden Pro-Teaser. Pro: Daily und bis zu fünf aktive Watches funktionieren.
- [ ] Das Watch-Dashboard ist eine eigene Seite `/app/watches`: erreichbar über
      den schwebenden View-Switch „Consensus | Watches“ (nur eingeloggt, Watches
      auf Mobile icon-only) und „Watched“ im Nutzericon-Menü; aktiver Pill-Zustand,
      Browser-Back/Forward
      und Deep-Link/Reload auf `/app/watches` funktionieren (vor dem Login
      erscheint ein Hinweis statt Daten). Kopfzeile mit aktiv/pausiert-Zählung,
      nächstem Lauf und Änderungen der letzten 7 Tage; pro Watch eine Karte mit
      Frage, Status-/Sichtbarkeits-Chip, Agreement-Score + Delta,
      History-Sparkline, letzter Änderung und nächstem Lauf. „Settings“ klappt
      Intervall/Uhrzeit/Mailmodus/Condition auf; Pause/Resume und Delete
      funktionieren. Delete lässt bereits vorhandene Share-History bestehen.
      „← Back to app“ und ESC führen zurück. Light/Dark und Mobile ohne Overflow.
- [ ] Morning Brief (Karte im Dashboard): Toggle aktiviert die tägliche
      Digest-Mail mit Uhrzeit (Browser-Zeitzone) und Modus „Every morning“ /
      „Only when something changed“; Einstellungen überleben ein erneutes
      Öffnen. Mit Test-SMTP: Brief-Mail listet alle Watches mit Score/Delta und
      Änderungs-Summaries; der Abmelde-Link deaktiviert nur den Brief, nicht
      die Watch-Mails.
- [ ] Ohne Watch ist der Morning-Brief-Toggle deaktiviert und erklärt „Create a
      watch first“; ein direkter Aktivierungs-Request wird abgelehnt. Nach dem
      Löschen der letzten Watch ist ein zuvor aktiver Brief ausgeschaltet.
- [ ] Aktive Watch-Seite zeigt bereits vor dem ersten History-Punkt in einer
      kompakten Metazeile Status, Intervall, letzten und nächsten Lauf. Mit History rendert
      sie zusätzlich SVG-Linie/Punkte und Change-Liste in Light/Dark ohne Mobile-Overflow.
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

## Modelle / Picker
- [ ] Custom Model Picker öffnet/wählt, sichtbarer Name aktualisiert.
- [ ] Tier-Defaults (Free vs. Pro) werden beim Tier-Wechsel angewandt; eine
      zuvor explizit im Picker gewählte Provider-Auswahl bleibt erhalten.
- [ ] Modell-Auswahl bleibt nach Reload erhalten (localStorage).

## Attachments (Pro)
- [ ] Datei anhängen → Chip erscheint, Vorschau öffnet, Entfernen funktioniert.
- [ ] PNG/JPG/WebP mit Strg+V im Fragefeld einfügen → Bild-Chip erscheint;
      normaler Text-Paste bleibt unverändert möglich.
- [ ] PNG/JPG/WebP auf den Input ziehen → Drop-Hinweis erscheint und nach dem
      Ablegen wird ein Bild-Chip angelegt.
- [ ] Bookmark-Attachments werden angezeigt.

## Auth / Usage / Tier
- [ ] Login (E-Mail + Google), Logout.
- [ ] Free-User: Usage-Counter + Limit-Anzeige korrekt, Limit-Fehler greift.
- [ ] Pro-User: Premium-Modelle freigeschaltet, UI-Status korrekt.

## Bookmarks / Sidebar
- [ ] Bookmarks laden/aufklappen, Chat-Suche filtert.
- [ ] Bookmark aus dem frischen Leerzustand öffnen: Input dockt ohne Hero-Sprung
      oben an und die gespeicherten Antworten sind direkt sichtbar.
- [ ] Einen gespeicherten Consensus nach Reload öffnen: Share-Link und Watch
      lassen sich ohne erneuten Consensus-Lauf erstellen (während der kurzen
      Vorbereitung zeigt der Dialog einen deaktivierten Ladezustand).
- [ ] Leaderboard auf/zu.

## Demo & Sonstiges
- [ ] „Demo"-Query startet den Demo-Flow (demo.js Integration intakt).
- [ ] Nach Abschluss der Demo sieht ein ausgeloggter Nutzer unter dem gefüllten
      Eingabefeld eine Login-/Registrierungs-Aufforderung; deren Button öffnet
      das Login-Modal. Nach erfolgreichem Login verschwindet die Aufforderung.
- [ ] Dark/Light-Toggle in Settings (Desktop und Mobile).
- [ ] Mobile-Layout (< 768px): Overlay-Sidebar, Info-Popups.
- [ ] System-Prompt-Modal + Help-Modal (app-ui.js) öffnen/speichern.
