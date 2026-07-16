"""Compact, text-minimised position maps for Consensus Watch history.

The map is derived from the already structured Differences result.  It keeps
short stance labels and provider membership, never complete model answers.
"""

from __future__ import annotations

import re


SCHEMA_VERSION = 1
MAX_DIMENSIONS = 4
MAX_POSITIONS = 3
PROVIDERS = ("OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok")
_PROVIDER_ALIASES = {
    "openai": "OpenAI",
    "chatgpt": "OpenAI",
    "mistral": "Mistral",
    "anthropic": "Anthropic",
    "claude": "Anthropic",
    "gemini": "Gemini",
    "google": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "xai": "Grok",
}
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _clip(value, limit: int) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _provider(value) -> str:
    raw = _clip(value, 40)
    return _PROVIDER_ALIASES.get(raw.lower(), raw if raw in PROVIDERS else "")


def _tokens(value) -> set[str]:
    return {
        token for token in _WORD_RE.findall(str(value or "").lower())
        if len(token) > 2
    }


def _similarity(left, right) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dimensions(differences_data: dict, *, allow_single=False) -> list[dict]:
    raw_differences = differences_data.get("differences")
    if not isinstance(raw_differences, list):
        return []
    dimensions = []
    for raw in raw_differences[:MAX_DIMENSIONS]:
        if not isinstance(raw, dict):
            continue
        label = _clip(raw.get("claim"), 140)
        raw_positions = raw.get("positions")
        if not label or not isinstance(raw_positions, list):
            continue
        positions = []
        for position in raw_positions[:MAX_POSITIONS]:
            if not isinstance(position, dict):
                continue
            stance = _clip(position.get("stance"), 180)
            models = []
            for model in position.get("models") or []:
                name = _provider(model)
                if name and name not in models:
                    models.append(name)
            if stance and models:
                positions.append({"stance": stance, "models": models})
        if len(positions) >= (1 if allow_single else 2):
            dimensions.append({
                "label": label,
                "type": raw.get("type") if raw.get("type") in {"contradiction", "claim"} else "emphasis",
                "positions": positions,
            })
    return dimensions


def _claim_dimensions(differences_data: dict) -> list[dict]:
    """Fallback map for unanimous runs using structured claim support."""
    raw_claims = differences_data.get("claims")
    if not isinstance(raw_claims, list):
        return []
    dimensions = []
    for raw in raw_claims[:MAX_DIMENSIONS]:
        if not isinstance(raw, dict):
            continue
        anchor = _clip(raw.get("anchor"), 180)
        if not anchor:
            continue
        agree = []
        for model in raw.get("agree") or []:
            provider = _provider(model)
            if provider and provider not in agree:
                agree.append(provider)
        positions = []
        if agree:
            positions.append({"stance": anchor, "models": agree})
        dissent_groups = {}
        for item in raw.get("dissent") or []:
            if not isinstance(item, dict):
                continue
            provider = _provider(item.get("model"))
            if not provider:
                continue
            stance = _clip(item.get("quote"), 180) or "Does not support this claim"
            dissent_groups.setdefault(stance, []).append(provider)
        positions.extend(
            {"stance": stance, "models": models}
            for stance, models in list(dissent_groups.items())[:MAX_POSITIONS - len(positions)]
        )
        if positions:
            dimensions.append({"label": anchor, "type": "claim", "positions": positions})
    return dimensions


def sanitize_opinion_map(value) -> dict | None:
    """Whitelist a persisted map before it reaches a public response."""
    if not isinstance(value, dict):
        return None
    dimensions = []
    raw_dimensions = value.get("dimensions")
    # _dimensions expects the same label/positions shape, with claim instead of
    # label. Convert explicitly to keep one validation path.
    if isinstance(raw_dimensions, list):
        converted = []
        for dimension in raw_dimensions[:MAX_DIMENSIONS]:
            if isinstance(dimension, dict):
                converted.append({
                    "claim": dimension.get("label"),
                    "type": dimension.get("type"),
                    "positions": dimension.get("positions"),
                })
        dimensions = _dimensions({"differences": converted}, allow_single=True)
    models = []
    raw_models = value.get("models")
    for item in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(item, dict):
            continue
        provider = _provider(item.get("provider"))
        if not provider:
            continue
        try:
            movement_score = int(item.get("movement_score") or 0)
        except (TypeError, ValueError):
            movement_score = 0
        models.append({
            "provider": provider,
            "movement_score": max(0, min(100, movement_score)),
            "moved": bool(item.get("moved")),
            "summary": _clip(item.get("summary"), 240),
        })
    raw_shift = value.get("shift_score")
    shift_score = None
    if isinstance(raw_shift, (int, float)):
        shift_score = max(0, min(100, int(round(raw_shift))))
    if not dimensions and shift_score is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "dimensions": dimensions,
        "models": models,
        "shift_score": shift_score,
        "shift_label": _clip(value.get("shift_label"), 24),
        "center": [
            _clip(item, 180) for item in (
                value.get("center")[:MAX_DIMENSIONS]
                if isinstance(value.get("center"), list) else []
            ) if _clip(item, 180)
        ],
    }


def _position_for(dimension: dict, provider: str):
    for position in dimension.get("positions") or []:
        if provider in (position.get("models") or []):
            return position
    return None


def _match_dimensions(current: list[dict], previous: list[dict]) -> list[tuple[dict, dict]]:
    matches = []
    used = set()
    for dimension in current:
        best_index, best_score = None, 0.0
        for index, candidate in enumerate(previous):
            if index in used:
                continue
            score = _similarity(dimension.get("label"), candidate.get("label"))
            if score > best_score:
                best_index, best_score = index, score
        if best_index is not None and best_score >= 0.34:
            used.add(best_index)
            matches.append((dimension, previous[best_index]))
    return matches


def build_opinion_map(differences_data: dict, previous=None) -> dict | None:
    """Build a multidimensional map and compare it with the previous map.

    Movement uses a conservative lexical comparison of each provider's stance
    on matched dimensions. This avoids counting a provider as moved merely
    because another provider joined or left its cluster, and avoids inventing a
    universal yes/no axis while still yielding a shared 0-100 Direction Shift
    score.
    """
    if not isinstance(differences_data, dict):
        return None
    dimensions = _dimensions(differences_data)
    if not dimensions:
        dimensions = _claim_dimensions(differences_data)
    if not dimensions:
        return None

    providers = [
        provider for provider in PROVIDERS
        if any(_position_for(dimension, provider) for dimension in dimensions)
    ]
    center = []
    for dimension in dimensions:
        dominant = max(dimension["positions"], key=lambda item: len(item["models"]))
        center.append(dominant["stance"])

    previous = sanitize_opinion_map(previous)
    previous_dimensions = previous.get("dimensions") if previous else []
    matches = _match_dimensions(dimensions, previous_dimensions)
    model_views = []
    all_movements = []
    for provider in providers:
        moved_count = 0
        comparable = 0
        summaries = []
        for current_dimension, previous_dimension in matches:
            current_position = _position_for(current_dimension, provider)
            previous_position = _position_for(previous_dimension, provider)
            if not current_position or not previous_position:
                continue
            comparable += 1
            stance_changed = _similarity(
                current_position.get("stance"), previous_position.get("stance")
            ) < 0.24
            moved = stance_changed
            if moved:
                moved_count += 1
                summaries.append(
                    f"{current_dimension['label']}: {current_position['stance']}"
                )
        movement_score = round(100 * moved_count / comparable) if comparable else 0
        if comparable:
            all_movements.append(movement_score)
        model_views.append({
            "provider": provider,
            "movement_score": movement_score,
            "moved": moved_count > 0,
            "summary": _clip(" · ".join(summaries), 240),
        })

    shift_score = None
    if previous:
        if matches and all_movements:
            shift_score = round(sum(all_movements) / len(all_movements))
        elif previous_dimensions:
            # The set of answer dimensions was completely reframed.
            shift_score = 100
            for item in model_views:
                item["movement_score"] = 100
                item["moved"] = True
    if shift_score is None:
        shift_label = "New baseline"
    elif shift_score <= 15:
        shift_label = "Stable"
    elif shift_score <= 45:
        shift_label = "Evolving"
    else:
        shift_label = "Turning"
    return {
        "schema_version": SCHEMA_VERSION,
        "dimensions": dimensions,
        "models": model_views,
        "shift_score": shift_score,
        "shift_label": shift_label,
        "center": center,
    }
