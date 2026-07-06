"""Resolve-Runde: gezielte Konfrontation der dissentierenden Modelle.

Nimmt einen Widerspruch aus der Differences-Analyse (Claim + Positionen mit
Modellen/Zitaten) und fragt jedes beteiligte Modell erneut - mit seiner
eigenen Position und der Gegenposition. Jedes Modell antwortet strukturiert
mit "maintain" oder "revise". Die Calls laufen wie bei der Differences-Engine
auf dem guenstigen Judge-Modell des jeweiligen Providers und parallel.

Ergebnis-Outcomes:
- "resolved":        mindestens ein Modell revidiert, mindestens eines bleibt
                     (der Dissens konvergiert Richtung der bestaetigten Position)
- "standoff":        alle Modelle bleiben bei ihrer Position (echter Dissens)
- "mutual_revision": alle Modelle revidieren (Ergebnis unklar, Nutzer prueft)
- "error":           kein Modell hat ein auswertbares Ergebnis geliefert
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from app.services.llm.consensus_engine import (
    DIFFERENCES_JUDGE_MODEL_BY_PROVIDER,
    _call_engine_text,
    _clip,
    _extract_json_object,
    normalize_model_name,
)

PROVIDER_BY_LABEL = {
    "OpenAI": "openai",
    "Mistral": "mistral",
    "Anthropic": "anthropic",
    "Gemini": "gemini",
    "DeepSeek": "deepseek",
    "Grok": "grok",
}

MAX_RESOLVE_POSITIONS = 4
MAX_RESOLVE_MODELS = 4
MAX_RESOLVE_TEXT_CHARS = 400
RESOLVE_MAX_TOKENS = 1000
RESOLVE_TEMPERATURE = 0.2

RESOLVE_SYSTEM_PROMPT = (
    "You re-examine one disputed point from an earlier answer. "
    "Respond with ONLY one JSON object, no prose, no markdown fences."
)


class InvalidResolvePayload(ValueError):
    """Client-Payload ist strukturell unbrauchbar (fuehrt zu HTTP 400)."""


def normalize_resolve_positions(claim, positions):
    """Validiert und kappt den Client-Payload der Resolve-Runde.

    Liefert (claim, positions) mit kanonischen Modellnamen oder wirft
    InvalidResolvePayload. Positionen ohne bekannte Modelle fallen raus;
    es muessen mindestens zwei Positionen mit insgesamt mindestens zwei
    verschiedenen Modellen uebrig bleiben.
    """
    claim = _clip(claim, MAX_RESOLVE_TEXT_CHARS)
    if not claim:
        raise InvalidResolvePayload("Missing disputed claim.")

    normalized = []
    seen_models = set()
    for entry in positions if isinstance(positions, list) else []:
        if not isinstance(entry, dict):
            continue
        models = []
        for label in entry.get("models") if isinstance(entry.get("models"), list) else []:
            canonical = normalize_model_name(label)
            if canonical in PROVIDER_BY_LABEL and canonical not in seen_models:
                models.append(canonical)
                seen_models.add(canonical)
        if not models:
            continue
        normalized.append({
            "stance": _clip(entry.get("stance"), MAX_RESOLVE_TEXT_CHARS),
            "models": models[:MAX_RESOLVE_MODELS],
            "quote": _clip(entry.get("quote"), MAX_RESOLVE_TEXT_CHARS),
        })
        if len(normalized) >= MAX_RESOLVE_POSITIONS:
            break

    if len(normalized) < 2 or len(seen_models) < 2:
        raise InvalidResolvePayload(
            "A resolve round needs at least two positions held by different models."
        )
    return claim, normalized


def _describe_position(position) -> str:
    text = position.get("stance") or "(no stance given)"
    quote = position.get("quote")
    if quote:
        text += f' (supporting quote: "{quote}")'
    return text


def _build_resolve_prompt(question: str, claim: str, own_position: dict, opposing: list) -> str:
    opposing_lines = "\n".join(
        f"- {_describe_position(pos)}" for pos in opposing
    )
    return (
        "In an earlier round you answered a user question. Your answer conflicts "
        "with at least one other AI assistant's answer on one specific point.\n\n"
        f"User question:\n{question}\n\n"
        f"Disputed point:\n{claim}\n\n"
        f"Your position:\n{_describe_position(own_position)}\n\n"
        f"Opposing position(s):\n{opposing_lines}\n\n"
        "Task: Re-examine ONLY the disputed point. Decide honestly:\n"
        '- "maintain" if, after reconsideration, you are confident your position is correct.\n'
        '- "revise" if the opposing position is correct or your statement was wrong or imprecise.\n\n'
        "JSON schema:\n"
        '{"decision": "maintain", "position": "your current position in one short sentence", '
        '"reason": "one short sentence explaining the decision"}\n\n'
        'Rules:\n'
        '- "decision" must be exactly "maintain" or "revise".\n'
        '- Write "position" and "reason" in the same language as the disputed point.\n'
        "- Do not mention other assistants, models, or this comparison process in "
        '"position" - it must read as a standalone factual statement.\n'
    )


def _query_resolve_model(model_label: str, question: str, claim: str,
                         own_position: dict, opposing: list, api_keys: dict) -> dict:
    provider = PROVIDER_BY_LABEL[model_label]
    judge_model = DIFFERENCES_JUDGE_MODEL_BY_PROVIDER[provider]
    result = {"model": model_label, "decision": "error", "position": "", "reason": ""}

    # Gemini kann ohne expliziten Key ueber Dev-Key/ADC laufen (siehe
    # _call_engine_text); alle anderen Provider brauchen einen Key.
    if provider != "gemini" and not api_keys.get(model_label):
        result["reason"] = "missing API key"
        return result

    prompt = _build_resolve_prompt(question, claim, own_position, opposing)
    # Transparenz: der tatsaechlich gestellte Prompt geht mit ins Ergebnis,
    # damit das Frontend zeigen kann, was die Modelle gefragt wurden. Wird
    # bewusst NICHT persistiert (Bookmark/Share strippen das Feld).
    result["prompt"] = prompt
    try:
        raw = _call_engine_text(
            provider, judge_model, judge_model, api_keys,
            system=RESOLVE_SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=RESOLVE_MAX_TOKENS,
            temperature=RESOLVE_TEMPERATURE,
            json_mode=True,
        )
    except Exception as e:
        logging.warning(f"Resolve call failed for {model_label} ({provider}/{judge_model}): {e}")
        result["reason"] = "provider error"
        return result

    parsed = _extract_json_object(raw)
    decision = str((parsed or {}).get("decision") or "").strip().lower()
    if decision not in ("maintain", "revise"):
        logging.warning(f"Resolve output unparsable for {model_label}: {raw!r:.200}")
        result["reason"] = "unparsable output"
        return result

    result["decision"] = decision
    result["position"] = _clip(parsed.get("position"), MAX_RESOLVE_TEXT_CHARS)
    result["reason"] = _clip(parsed.get("reason"), MAX_RESOLVE_TEXT_CHARS)
    return result


def _aggregate_outcome(results: list) -> str:
    decisions = [r["decision"] for r in results if r["decision"] in ("maintain", "revise")]
    if not decisions:
        return "error"
    if all(d == "maintain" for d in decisions):
        return "standoff"
    if all(d == "revise" for d in decisions):
        return "mutual_revision"
    return "resolved"


def run_resolve_round(question: str, claim: str, positions: list, api_keys: dict) -> dict:
    """Fuehrt die Resolve-Runde aus. positions muss bereits durch
    normalize_resolve_positions gelaufen sein (kanonische Modellnamen,
    gekappte Texte). Die Modell-Calls laufen parallel."""
    jobs = []
    for idx, position in enumerate(positions):
        opposing = [p for i, p in enumerate(positions) if i != idx]
        for model_label in position["models"]:
            jobs.append((model_label, position, opposing))

    with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_RESOLVE_MODELS)) as pool:
        futures = [
            pool.submit(
                _query_resolve_model,
                model_label, question, claim, position, opposing, api_keys,
            )
            for model_label, position, opposing in jobs
        ]
        results = [future.result() for future in futures]

    return {
        "claim": claim,
        "outcome": _aggregate_outcome(results),
        "results": results,
    }
