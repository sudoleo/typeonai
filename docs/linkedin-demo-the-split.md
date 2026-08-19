# THE SPLIT — consens.io Launch-Film

Der aktuelle Launch-Cut, gerendert aus `recording/lib/storyboard.cjs`.
Die dauerhaften Regeln stehen in `docs/linkedin-content-guide.md`.

**Gemessen:** 54,30 s · 1080 × 1350 (4:5) · 30 fps · dark · 1.629 Frames · 8,92 MB

```bash
node recording/record-linkedin-demo.cjs --out recordings/launch-v5
```

| Datei | Zweck |
| --- | --- |
| `consensio-linkedin-4x5.mp4` | Veröffentlichungs-Master mit Ton |
| `consensio-linkedin-4x5-silent.mp4` | Identische Bildspur ohne Ton |
| `consensio-linkedin-4x5-poster.jpg` | LinkedIn-Poster aus dem 3:3-Evidenz-Hold |
| `score-4x5.wav` | Unkomprimierte Tonspur dieses Schnitts |
| `cut.json` | Gemessene Szenenzeiten, Posterframe und Audiopegel |

---

## 1. Die These

> Eine Antwort ist nicht genug. consens.io zeigt, welche Aussagen sechs
> Modelle gemeinsam tragen, wo sie sich widersprechen und behält den Kontext
> für die nächste Frage.

Der Film bleibt bei einer konkreten Entscheidung: Soll eine Nachricht an eine
Kundin so abgeschickt werden, nachdem sich ein Launch um zwei Wochen verschoben
hat? Das Produkt liefert vier einstimmig getragene Korrekturen, markiert eine
kritische 3:3-Spaltung und zeigt einen Agreement Score von 52/100. Anschließend
bleibt das Ergebnis sichtbar, während eine Folgefrage im selben Kontext
geschrieben wird.

Das frühere Mechanikdiagramm ist bewusst nicht mehr Teil des Launch-Cuts. Es
kam vor dem Produktbeweis, wechselte die Designsprache und verlangte zu viel
Interpretation in zu kurzer Zeit.

---

## 2. Gemessener Ablauf

Zeiten aus `recordings/launch-v5/cut.json`.

| t | Szene | Bild und Aussage |
| --- | --- | --- |
| **0,00** | `intro` | Wordmark, **„Would you send this?“**, darunter **„A client. A missed deadline. One message.“** Kein Typewriter-Effekt; 2,8 Sekunden bis zum Produkt-Cut. |
| **2,80** | `draft` | Harter, scharfer Cut auf den bereits eingefügten Entwurf. Kein Blur. Der Zeiger fährt bewusst zum Senden, hält vor dem Klick und lässt das Klickfeedback vollständig stehen. |
| **7,43** | `run` | Das echte Run-Panel füllt den Frame. Alle sechs Modellbalken laufen sichtbar bis mindestens 95 %, danach folgen **„Writing the consensus“**, **„Checking for contradictions“** und **„Done“**. Caption: **„Six models read it independently.“** |
| **22,70** | `answer` | Erst nach dem abgeschlossenen Run harter Cut auf den fertigen Consensus. Keine Caption, 3,1 Sekunden ruhiger Leseraum. |
| **26,47** | `claims` | Vier Claim-Badges werden gemeinsam hervorgehoben. Caption: **„Agreement, claim by claim.“** |
| **29,70** | `split` | Licht schließt auf die markierte kritische Passage; Hover öffnet die echte Evidenzkarte. Caption: **„Then the line that splits them.“** |
| **35,43** | `evidence` | Fahrt auf die 3:3-Evidenzkarte. Zeiger verschwindet vor dem Hold. Hero: **„Three keep it. Three cut it.“** Posterframe bei 37,07 s. |
| **39,60** | `verdict` | Fahrt auf den echten Score und dessen Evidenz-Aktionen. Caption: **„52/100 agreement. No false certainty.“** |
| **45,13** | `memory` | Ergebnis und Follow-up-Composer stehen gleichzeitig im Bild. **„Make it warmer. Keep the date firm.“** wird sichtbar getippt. Caption: **„Ask the next question. The context stays.“** |
| **51,10** | `end-card` | `DEMO SCENARIO · REAL INTERFACE`, **„Know what agrees. See what doesn’t.“**, `consens.io`. |

Der Score und die 3:3-Verteilung werden aus dem DOM gelesen beziehungsweise
gegen die Produktdaten geprüft. Wenn die Demo-Daten nicht mehr dazu passen,
bricht der Render ab.

---

## 3. Dramaturgie und Kamera

- Die erste Zeile stellt eine menschliche Entscheidung, keine Produktfunktion.
- Das Eingabefeld ist durch diese Frage motiviert und erscheint in einem
  scharfen Cut bei lesbarer Größe.
- Der fertige Consensus erscheint bei 22,70 s erst nach dem vollständig
  nachvollziehbaren Modell- und Syntheselauf, nicht mitten in einer Animation.
- Der eigentliche 3:3-Produktbeweis erscheint bei 35,43 s statt nach einer vorgelagerten
  Architektur-Erklärung.
- Der 3:3-Split ist der erste WOW-Moment; der sichtbare Follow-up-Kontext ist
  der zweite, ruhigere Produktmoment.
- Die Kamera fährt nur auf Motive, deren Text vorher durch Licht isoliert wurde.
- Der Posterframe enthält keinen zufällig schwebenden Mauszeiger.
- Farbe bleibt auf die echten Agreement-/Contradiction-Signale begrenzt.

---

## 4. Chat Memory

Der Memory-Beat verwendet den echten `window.App.followup`-State und den echten
Composer des Produkts. Nach dem öffentlichen Gast-Demo-Lauf wird der
Login-Hinweis für diese Aufnahme geschlossen, weil er den Composer semantisch
ersetzt. Danach wird der vorhandene Consensus als fortsetzbarer Austausch
angeboten und der Composer zeigt den echten Follow-up-Zustand.

Vor jeder Kamerafahrt prüft der Recorder, dass der Composer eine reale Breite
und Höhe besitzt. Ein unsichtbares 0×0-Feld bricht den Render ab, statt die
Kamera still in die linke obere Ecke fahren zu lassen.

---

## 5. Ton

Die Sounddesign-Fassung aus Tonflächen und Einzelgeräuschen wurde vollständig
verworfen. Der Master trägt jetzt ein zusammenhängendes Instrumentalstück.

Der neue Score besteht aus:

- **88 BPM** und einer durchgehenden Taktstruktur;
- der Progression **Dm9 → Bbmaj7 → Fmaj9 → Cadd9**;
- gehaltenen Akkorden, hörbarer Bassbewegung und Felt-Piano-Arpeggio;
- einer dichteren musikalischen Bewegung bei Run, Split und Memory;
- keinem Noise-Bett, keinem Sweep und keinem separaten UI-Soundeffekt.

Gemessen im finalen Master: **21 musikalische Takte/Cues · −17,0 dB Mean ·
−2,5 dB Peak**. Vor der Veröffentlichung bleibt die im Content Guide
vorgeschriebene Hörkontrolle auf Kopfhörern und Telefon erforderlich.

---

## 6. QA

- Finaler Master: H.264 High, 1080 × 1350, 30 fps.
- Audio: AAC LC, 48 kHz, Stereo.
- Silent Master, WAV-Score und Poster wurden zusammen mit demselben Cut erzeugt.
- Keine Framing- oder Subject-Violation im finalen Recorderlauf.
- Intro, Evidenz, Verdict, Memory und Endcard wurden im Kontaktblatt geprüft.
