"""Parsing & Bewertung (Plan §7).

- ``extract_letter(text, options=None)``: finaler FINAL_ANSWER-Marker -> Buchstabe oder None.
- ``majority_vote(letters)``: Mehrheitsbuchstabe oder ``"no_majority"`` (E2).
- ``grade(letter, ground_truth)``: bool.
"""

from __future__ import annotations

import re
from collections import Counter

NO_MAJORITY = "no_majority"

# Toleranter FINAL_ANSWER-Marker auf der **letzten nicht-leeren Zeile**:
# case-insensitive, Unterstrich oder Leerzeichen ("FINAL_ANSWER"/"FINAL ANSWER"),
# optionale Klammern um den Buchstaben, optionaler Schluss-Punkt. Markdown-Fettung
# (``*``) wird vor dem Matchen entfernt, damit ``**FINAL_ANSWER: B**`` matcht.
# Bewusst **kein** semantischer Fallback ("the answer is …", "Option B", lose
# Buchstabenzeilen) – der Marker muss kommen; die Toleranz faengt nur
# Formatierungs-Kleinkram ab (Plan §7 + Pilot-Befund: Modelle haengen oft Punkt
# oder Klammern an).
_FINAL_LINE_RE = re.compile(
    r"^final[_ ]?answer\s*:\s*\(?\s*([A-J])\s*\)?\s*\.?$",
    re.IGNORECASE,
)


def extract_letter(text: str, options: list[str] | None = None) -> str | None:
    """Extrahiert den gewaehlten Options-Buchstaben aus einer Modellantwort.

    Benchmark-Contract: Die letzte nicht-leere Zeile muss der
    ``FINAL_ANSWER: X``-Marker sein (formattolerant, s. ``_FINAL_LINE_RE``).
    Begruendungstext davor wird nicht interpretiert. ``options`` bleibt als
    kompatibler Parameter erhalten, wird aber absichtlich **nicht** fuer
    semantische Fallbacks genutzt.
    """
    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return None
    final_line = lines[-1].replace("*", "").strip()
    match = _FINAL_LINE_RE.match(final_line)
    if not match:
        return None
    return match.group(1).upper()


def majority_vote(letters: list[str | None]) -> str:
    """Mehrheitsentscheid ueber die Modell-Buchstaben.

    ``None``/abstain-Stimmen zaehlen nicht mit. Bei Gleichstand der fuehrenden
    Buchstaben (oder ohne gueltige Stimmen) -> ``no_majority`` (E2, kein Tie-Break).
    """
    valid = [letter for letter in letters if letter]
    if not valid:
        return NO_MAJORITY

    counts = Counter(valid)
    ordered = counts.most_common()
    top_count = ordered[0][1]
    leaders = [letter for letter, count in ordered if count == top_count]
    if len(leaders) != 1:
        return NO_MAJORITY
    return leaders[0]


def grade(letter: str | None, ground_truth: str) -> bool:
    """True genau dann, wenn ``letter`` mit der Ground Truth uebereinstimmt.
    ``no_majority``/None/abstain gelten nie als korrekt."""
    if not letter or letter == NO_MAJORITY:
        return False
    return str(letter).strip().upper() == str(ground_truth or "").strip().upper()
