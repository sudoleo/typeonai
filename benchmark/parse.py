"""Parsing & Bewertung (Plan §7).

- ``extract_letter(text, options=None)``: finaler FINAL_ANSWER-Marker -> Buchstabe oder None.
- ``majority_vote(letters)``: Mehrheitsbuchstabe oder ``"no_majority"`` (E2).
- ``grade(letter, ground_truth)``: bool.
"""

from __future__ import annotations

from collections import Counter

from benchmark.prompt import LETTERS

NO_MAJORITY = "no_majority"

_FINAL_MARKER = "FINAL_ANSWER: "


def extract_letter(text: str, options: list[str] | None = None) -> str | None:
    """Extrahiert den gewaehlten Options-Buchstaben aus einer Modellantwort.

    Benchmark-Contract: Die letzte nicht-leere Zeile muss exakt
    ``FINAL_ANSWER: X`` enthalten. Begruendungstext davor wird nicht
    interpretiert. ``options`` bleibt als kompatibler Parameter erhalten, wird
    aber absichtlich nicht fuer semantische Fallbacks genutzt.
    """
    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return None
    final_line = lines[-1]
    if not final_line.startswith(_FINAL_MARKER):
        return None
    letter = final_line[len(_FINAL_MARKER):]
    if len(letter) != 1 or letter not in LETTERS:
        return None
    return letter


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
