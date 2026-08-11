"""Pure agreement scoring for normalized Differences data."""

from __future__ import annotations


AGREEMENT_LEVEL_THRESHOLDS = [
    (85, "very"),
    (65, "largely"),
    (40, "partially"),
    (20, "hardly"),
]


def compute_agreement_score(data: dict) -> dict:
    claims = data.get("claims") or []
    differences = data.get("differences") or []
    model_count = len(data.get("models_compared") or [])
    ratios = []
    for claim in claims:
        agree = len(claim.get("agree") or [])
        dissent = len(claim.get("dissent") or [])
        if agree + dissent:
            ratios.append(agree / (agree + dissent))
    base = sum(ratios) / len(ratios) if ratios else 1.0
    contradictions = [item for item in differences if item.get("type") == "contradiction"]
    major = sum(1 for item in contradictions if item.get("severity") != "minor")
    minor = len(contradictions) - major
    emphases = len(differences) - len(contradictions)
    score = base - 0.25 * major - 0.10 * minor - 0.05 * emphases
    caps = [1.0]
    if differences:
        caps.append(0.84)
    if major >= 2:
        caps.append(0.39)
    elif major == 1:
        caps.append(0.64)
    if model_count == 3:
        caps.append(0.90)
    elif model_count == 2:
        caps.append(0.75)
    elif model_count <= 1:
        caps.append(0.50)
    score_pct = int(round(max(0.0, min(score, *caps)) * 100))
    level = "not"
    for threshold, name in AGREEMENT_LEVEL_THRESHOLDS:
        if score_pct >= threshold:
            level = name
            break
    return {
        "score": score_pct,
        "level": level,
        "model_count": model_count,
        "major_contradictions": major,
        "minor_contradictions": minor,
        "emphases": emphases,
    }

