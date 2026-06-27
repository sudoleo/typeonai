"""MC-Frage-Template (Optionen A-J + "The answer is (X)"-Instruktion).

Der closed-book System-Prompt liegt in ``config.SYSTEM_PROMPT``; dieses Modul baut
nur den User-Prompt (die eigentliche Frage mit Optionen).
"""

from __future__ import annotations

LETTERS = "ABCDEFGHIJ"


def letters_for(num_options: int) -> str:
    """Buchstaben A.. fuer ``num_options`` Optionen (max 10 -> A-J)."""
    if num_options < 1 or num_options > len(LETTERS):
        raise ValueError(f"Unsupported option count: {num_options}")
    return LETTERS[:num_options]


def build_mc_question(question: str, options: list[str]) -> str:
    """Baut den MC-Frage-Text: Fragetext, nummerierte Optionen (A), (B), ...,
    plus eine abschliessende Format-Instruktion."""
    opts = list(options or [])
    if not opts:
        raise ValueError("MC question requires at least one option")
    if len(opts) > len(LETTERS):
        raise ValueError(f"Too many options ({len(opts)}); max {len(LETTERS)}")

    lines = [str(question or "").strip(), ""]
    for idx, option in enumerate(opts):
        lines.append(f"({LETTERS[idx]}) {str(option).strip()}")
    lines.append("")
    lines.append("Choose the single correct option. End with: The answer is (X).")
    return "\n".join(lines)
