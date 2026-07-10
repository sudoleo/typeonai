# Smoke-Checkliste — Frontend (index.html Refactor)

Teilweise automatisiert: die Playwright-Suite `tests/e2e/` deckt Konsolen-
Fehler beim Laden, Send→Streaming, Consensus→Differences+Agreement-Score,
Modell-Ausschluss, Theme-Toggle und Picker-Persistenz ab (Lauf: siehe
`tests/e2e/README.md`). Die übrigen Punkte weiterhin manuell durchgehen
(oder zumindest die vom Cluster betroffenen), bevor committet wird. Backend
bleibt durch `venv/Scripts/python -m pytest tests/` abgesichert
(Baseline: 326 passed, Stand 2026-07-06).

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
