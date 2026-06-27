"""Kostenrechnung: Usage -> USD aus der Pricing-Tabelle (Plan §10).

Pricing sind manuell gepflegte Schaetzungen (``config.PRICING_USD_PER_1M``).
Fehlt ein ``api_model`` in der Tabelle, ist die Kostenschaetzung 0.0 und der
Aufrufer wird per ``has_pricing`` gewarnt.
"""

from __future__ import annotations

from benchmark import config


def has_pricing(api_model: str) -> bool:
    return api_model in config.PRICING_USD_PER_1M


def est_cost_usd(api_model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Schaetzt die Kosten eines Calls in USD aus Prompt-/Completion-Tokens."""
    rates = config.PRICING_USD_PER_1M.get(api_model)
    if not rates:
        return 0.0
    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    return (prompt * rates["input"] + completion * rates["output"]) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Grobe Input-Token-Schaetzung ohne tiktoken (Plan §2/§8): len/4-Heuristik."""
    return max(1, len(str(text or "")) // 4)
