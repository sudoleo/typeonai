# Smoke-Checkliste — Frontend (index.html Refactor)

Teilweise automatisiert: die Playwright-Suite `tests/e2e/` deckt Konsolen-
Fehler beim Laden, Send→Streaming, Consensus→Differences+Agreement-Score,
Modell-Ausschluss, Theme-Toggle und Picker-Persistenz ab (Lauf: siehe
`tests/e2e/README.md`). Die übrigen Punkte weiterhin manuell durchgehen
(oder zumindest die vom Cluster betroffenen), bevor committet wird. Backend
bleibt durch `venv/Scripts/python -m pytest tests/` abgesichert
(Baseline: 409 passed, Stand 2026-07-12).

## Browser-Konsole
- [ ] Beim Laden **keine** JS-Fehler in der Konsole (besonders: keine
      `ReferenceError: X is not defined`, keine `window.X is not a function`).

## Kern-Flow
- [ ] Frage eingeben + senden → alle ausgewählten Modelle streamen Antworten.
- [ ] Senden während Lauf abbrechen (Stop) funktioniert.
- [ ] Modell ein-/ausschließen (Checkbox/Toggle) blendet Antwortbox korrekt ein/aus.
- [ ] Quellen-Chips / Evidence-Links erscheinen und sind klickbar.

## Consensus (höchstes Risiko)
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

## Consensus Watch
- [ ] Nach erfolgreichem Consensus erscheint „Watch“ neben Share; Aktivierung
      verlangt die explizite Wahl zwischen privater Eigentümer-Seite und öffentlicher,
      nicht indexierter Link-Seite und bietet Weekly/Monthly. Private Seiten sind in
      einem fremden oder ausgeloggten Browser nicht lesbar.
- [ ] Lokale Run-Uhrzeit ist bei Erstellung wählbar und zeigt die erkannte Zeitzone;
      „Watched“ erlaubt eine spätere Änderung. `next_run_at` entspricht der gewählten
      lokalen Uhrzeit (mit bis zu 30 Minuten Scheduler-Toleranz), auch über einen
      Sommer-/Winterzeitwechsel hinweg.
- [ ] Free: Daily ist als Pro markiert/gesperrt und das aktive Limit öffnet den
      bestehenden Pro-Teaser. Pro: Daily und bis zu fünf aktive Watches funktionieren.
- [ ] „Watched“ listet Status, Intervall und Sichtbarkeit; Intervall ändern,
      Mailmodus bzw. Condition ändern, Pause/Resume und Delete funktionieren. „Watched“ steht
      außerdem im Nutzericon-Menü direkt unter „Shared links“. Delete lässt
      bereits vorhandene Share-History bestehen.
- [ ] Aktive Watch-Seite zeigt bereits vor dem ersten History-Punkt in einer
      kompakten Metazeile Status, Intervall, letzten und nächsten Lauf. Mit History rendert
      sie zusätzlich SVG-Linie/Punkte und Change-Liste in Light/Dark ohne Mobile-Overflow.
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
- [ ] Tier-Defaults (Free vs. Pro) werden angewandt.
- [ ] Modell-Auswahl bleibt nach Reload erhalten (localStorage).

## Attachments (Pro)
- [ ] Datei anhängen → Chip erscheint, Vorschau öffnet, Entfernen funktioniert.
- [ ] Bookmark-Attachments werden angezeigt.

## Auth / Usage / Tier
- [ ] Login (E-Mail + Google), Logout.
- [ ] Free-User: Usage-Counter + Limit-Anzeige korrekt, Limit-Fehler greift.
- [ ] Pro-User: Premium-Modelle freigeschaltet, UI-Status korrekt.

## Bookmarks / Sidebar
- [ ] Bookmarks laden/aufklappen, Chat-Suche filtert.
- [ ] Leaderboard auf/zu.

## Demo & Sonstiges
- [ ] „Demo"-Query startet den Demo-Flow (demo.js Integration intakt).
- [ ] Nach Abschluss der Demo sieht ein ausgeloggter Nutzer unter dem gefüllten
      Eingabefeld eine Login-/Registrierungs-Aufforderung; deren Button öffnet
      das Login-Modal. Nach erfolgreichem Login verschwindet die Aufforderung.
- [ ] Dark/Light-Toggle.
- [ ] Mobile-Layout (< 768px): Overlay-Sidebar, Info-Popups.
- [ ] System-Prompt-Modal + Help-Modal (app-ui.js) öffnen/speichern.
