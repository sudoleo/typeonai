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

# Klare Antwortsignale. Der Buchstabe muss am Ende einer Antwortphrase stehen
# (Punkt/Zeilenende), damit "Option I is discussed" nicht matcht.
_ANSWER_SIGNAL_RES = (
    re.compile(r"\b(?:the\s+)?answer\s+is\s*\(?([A-J])\)?(?=\s*(?:[.!?]|$))", re.IGNORECASE),
    re.compile(r"\bfinal\s+answer\s*:\s*\(?([A-J])\)?(?=\s*(?:[.!?]|$))", re.IGNORECASE),
    re.compile(r"\banswer\s*:\s*\(?([A-J])\)?(?=\s*(?:[.!?]|$))", re.IGNORECASE),
    re.compile(r"\boption\s+\(?([A-J])\)?(?=\s*(?:[.!?]|$))", re.IGNORECASE),
)
# Eigenstaendige finale Zeile: "I.", "(I)", "**I**".
_STANDALONE_LETTER_LINE_RE = re.compile(r"^\**\(?([A-J])\)?(?:[.)])?\**$", re.IGNORECASE)
# Klassische MC-Zeile am Ende: "C) Option text". Kein "I. think ...".
_LETTER_OPTION_LINE_RE = re.compile(r"^\(?([A-J])\)\s+\S", re.IGNORECASE)
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def extract_letter(text: str, options: list[str] | None = None) -> str | None:
    """Extrahiert den gewaehlten Options-Buchstaben aus einer Modellantwort.

    Kaskade: (1) klare finale Antwortsignale, (2) Match auf Options-Text
    (falls ``options`` uebergeben), sonst None (-> abstain).
    """
    if not text:
        return None
    raw = str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Finale Antwortsignale. Bei widerspruechlichen Signalen nicht raten.
    found_signal, letter = _extract_final_signal(lines or [raw])
    if found_signal:
        return letter

    # Match auf Options-Text.
    if options:
        letter = _match_option_text(raw, options)
        if letter:
            return letter

    return None


def _extract_final_signal(lines: list[str]) -> tuple[bool, str | None]:
    """Sucht klare Antwortsignale bevorzugt am Ende der Antwort.

    Gibt ``(True, None)`` bei widerspruechlichen finalen Signalen zurueck, damit
    der Caller nicht auf einen loseren Fallback ausweicht.
    """
    candidates: list[str] = []
    for line in reversed(lines):
        found = _signals_in_line(line)
        if found:
            candidates.extend(found)
            continue
        if candidates:
            break

    if not candidates:
        return False, None
    unique = {candidate.upper() for candidate in candidates}
    if len(unique) != 1:
        return True, None
    return True, next(iter(unique))


def _signals_in_line(line: str) -> list[str]:
    normalized = line.strip()
    signals: list[str] = []

    m = _STANDALONE_LETTER_LINE_RE.fullmatch(normalized)
    if m:
        signals.append(m.group(1).upper())

    m = _LETTER_OPTION_LINE_RE.match(normalized)
    if m:
        signals.append(m.group(1).upper())

    for pattern in _ANSWER_SIGNAL_RES:
        signals.extend(match.upper() for match in pattern.findall(normalized))
    return signals


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
    if len(hits) == 1:
        return hits[0]
    if hits:
        return None

    return _match_option_components(text, options)


def _match_option_components(text: str, options: list[str]) -> str | None:
    """Strikter Fallback fuer Fill-in-the-blank-Optionen.

    Eine Option zaehlt nur, wenn alle comma-getrennten, normalisierten Komponenten
    exakt im Antworttext vorkommen und genau eine Option diese Bedingung erfuellt.
    """
    normalized_text = f" {_normalize_for_component_match(text)} "
    hits = []
    for idx, option in enumerate(options):
        if idx >= len(LETTERS):
            break
        components = [
            _normalize_for_component_match(part)
            for part in str(option or "").split(",")
        ]
        components = [part for part in components if len(part) >= 3]
        if len(components) < 3:
            continue
        if all(f" {component} " in normalized_text for component in components):
            hits.append(LETTERS[idx])
    return hits[0] if len(hits) == 1 else None


def _normalize_for_component_match(value: str) -> str:
    return _NORMALIZE_RE.sub(" ", str(value or "").lower()).strip()


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
