"""Audits (Plan E4).

1. ``assert_no_web_tools(payload)`` – Sicherheits-Audit ueber **jeden** Payload
   (Dry-Run + vor jedem realen Call). Bricht ab, sobald irgendwo ein Web-Tool
   auftaucht.
2. ``permute_options(...)`` – Basis fuer den Optionen-Permutations-Audit
   (Positions-Bias der Einzelmodelle).
3. ``consensus_order_variants(...)`` / ``run_consensus_order_audit(...)`` –
   Consensus-Reihenfolge-Audit (Stabilitaet der Synthese ueber normal / umgekehrt
   / deterministisch gemischt; nur Consensus wird neu berechnet, keine erneuten
   Kandidaten-Calls).
"""

from __future__ import annotations

import random
from typing import Callable

from benchmark.prompt import LETTERS

# Marker, die in keinem Benchmark-Payload vorkommen duerfen.
WEB_TOOL_MARKERS = ("web_search", "google_search", "web_search_call")
# Top-Level-/verschachtelte Keys, die ein injiziertes Tool signalisieren.
FORBIDDEN_KEYS = ("tools", "tool_choice")


def find_web_tool_violations(value, path: str = "payload") -> list[str]:
    """Sammelt rekursiv alle Pfade, an denen ein Web-Tool-Marker auftaucht."""
    violations: list[str] = []

    if isinstance(value, dict):
        for key, sub in value.items():
            key_str = str(key)
            if key_str in FORBIDDEN_KEYS:
                violations.append(f"{path}.{key_str}")
            if _has_marker(key_str):
                violations.append(f"{path}.<key:{key_str}>")
            violations.extend(find_web_tool_violations(sub, f"{path}.{key_str}"))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            violations.extend(find_web_tool_violations(item, f"{path}[{idx}]"))
    elif isinstance(value, str):
        if _has_marker(value):
            violations.append(f"{path}=={value!r}")

    return violations


def _has_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in WEB_TOOL_MARKERS)


def assert_no_web_tools(payload: dict, context: str = "") -> None:
    """Bricht mit AssertionError ab, sobald der Payload ein Web-Tool enthaelt."""
    violations = find_web_tool_violations(payload)
    if violations:
        where = f" [{context}]" if context else ""
        raise AssertionError(
            f"Web tool leaked into closed-book payload{where}: {', '.join(violations)}"
        )


# --- Optionen-Permutations-Audit -------------------------------------------


def permute_options(
    options: list[str], answer_index: int, rng: random.Random
) -> tuple[list[str], int, str, list[int]]:
    """Mischt die Optionen deterministisch (via ``rng``) und gibt
    (neue_optionen, neuer_answer_index, neuer_answer_letter, permutation) zurueck.

    ``permutation[i]`` ist der urspruengliche Index der Option an neuer Position i.
    """
    order = list(range(len(options)))
    rng.shuffle(order)
    new_options = [options[i] for i in order]
    new_answer_index = order.index(answer_index)
    return new_options, new_answer_index, LETTERS[new_answer_index], order


# --- Consensus-Reihenfolge-Audit -------------------------------------------


def consensus_order_variants(
    providers: list[str], rng: random.Random
) -> dict[str, list[str]]:
    """Liefert drei Provider-Reihenfolgen fuer den Stabilitaets-Audit:
    normal, umgekehrt und deterministisch gemischt."""
    shuffled = list(providers)
    rng.shuffle(shuffled)
    return {
        "normal": list(providers),
        "reversed": list(reversed(providers)),
        "shuffled": shuffled,
    }


def run_consensus_order_audit(
    providers: list[str],
    recompute_letter: Callable[[list[str]], str | None],
    rng: random.Random,
) -> dict:
    """Berechnet den Consensus-Letter ueber die drei Reihenfolgen neu (via
    injizierter ``recompute_letter``) und protokolliert die Stabilitaet.

    ``recompute_letter`` bekommt die Provider-Reihenfolge und liefert den
    extrahierten Consensus-Buchstaben – **keine** Kandidaten-Calls hier.
    """
    variants = consensus_order_variants(providers, rng)
    letters = {name: recompute_letter(order) for name, order in variants.items()}
    distinct = {letter for letter in letters.values()}
    return {
        "orders": variants,
        "letters": letters,
        "stable": len(distinct) == 1,
    }
