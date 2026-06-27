"""Parsing & Bewertung (Plan §7).

- ``extract_letter(text, options=None)``: Regex-Kaskade -> Buchstabe oder None.
- ``majority_vote(letters)``: Mehrheitsbuchstabe oder ``"no_majority"`` (E2).
- ``grade(letter, ground_truth)``: bool.
"""

from __future__ import annotations

import re
from collections import Counter

from benchmark.prompt import LETTERS

NO_MAJORITY = "no_majority"

# (1) "the answer is (C)" / "the answer is C" – case-insensitive, letzter Treffer.
_ANSWER_IS_RE = re.compile(r"answer\s*(?:is|:)\s*\(?([A-J])\)?", re.IGNORECASE)
# (2) letzte nicht-leere Zeile beginnt mit "C)" / "C." / "(C)".
_LINE_LETTER_RE = re.compile(r"^\(?([A-J])[).]", re.IGNORECASE)
# Fallback: ein isoliert stehender Buchstabe als ganze Zeile, evtl. mit **fett**.
_BARE_LETTER_RE = re.compile(r"^\**\(?([A-J])\)?\**$", re.IGNORECASE)


def extract_letter(text: str, options: list[str] | None = None) -> str | None:
    """Extrahiert den gewaehlten Options-Buchstaben aus einer Modellantwort.

    Kaskade: (1) "the answer is (X)", (2) letzte Zeile "X)"/"X.", (3) Match auf
    Options-Text (falls ``options`` uebergeben), sonst None (-> abstain).
    """
    if not text:
        return None
    raw = str(text)

    # (1) "the answer is (X)" – letzter Treffer gewinnt (Modelle wiederholen oft).
    matches = _ANSWER_IS_RE.findall(raw)
    if matches:
        return matches[-1].upper()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # (2) letzte/erste markante Zeile, die mit einem Options-Buchstaben beginnt.
    for line in reversed(lines):
        m = _LINE_LETTER_RE.match(line) or _BARE_LETTER_RE.match(line)
        if m:
            return m.group(1).upper()

    # (3) Match auf Options-Text.
    if options:
        letter = _match_option_text(raw, options)
        if letter:
            return letter

    return None


def _match_option_text(text: str, options: list[str]) -> str | None:
    """Findet den Buchstaben, dessen Options-Text als ganzer (genug langer)
    String im Antworttext vorkommt. Konservativ: nur eindeutige Treffer."""
    lowered = text.lower()
    hits = []
    for idx, option in enumerate(options):
        if idx >= len(LETTERS):
            break
        candidate = str(option or "").strip().lower()
        if len(candidate) >= 3 and candidate in lowered:
            hits.append(LETTERS[idx])
    return hits[0] if len(hits) == 1 else None


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
