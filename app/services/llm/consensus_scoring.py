"""Pure agreement scoring for normalized Differences data."""

from __future__ import annotations


AGREEMENT_LEVEL_THRESHOLDS = [
    (85, "very"),
    (65, "largely"),
    (40, "partially"),
    (20, "hardly"),
]

# Eine einzelne Stimme belegt nichts: "1/1" liest sich wie eine Bestaetigung,
# ist aber nur ein Modell. Solche Claims sind seit dem Coverage-Judge sichtbar
# (grau, "zu wenige Modelle haben das behandelt") - in den Score gehen sie
# weiterhin NICHT ein, sonst hoebe fehlende Abdeckung die Zahl.
MIN_SCORED_CLAIM_SUPPORT = 2


def compute_agreement_score(data: dict) -> dict:
    claims = data.get("claims") or []
    differences = data.get("differences") or []
    model_count = len(data.get("models_compared") or [])
    ratios = []
    thin = 0
    for claim in claims:
        agree = len(claim.get("agree") or [])
        dissent = len(claim.get("dissent") or [])
        if agree + dissent < MIN_SCORED_CLAIM_SUPPORT:
            thin += 1
            continue
        ratios.append(agree / (agree + dissent))
    base = sum(ratios) / len(ratios) if ratios else 1.0
    contradictions = [item for item in differences if item.get("type") == "contradiction"]
    major = sum(1 for item in contradictions if item.get("severity") != "minor")
    minor = len(contradictions) - major
    emphases = len(differences) - len(contradictions)
    score = base - 0.25 * major - 0.10 * minor - 0.05 * emphases
    caps = [1.0]
    if not ratios:
        # Kein einziger belegter Satz - entweder hat die Antwort gar keine
        # pruefbaren Saetze, oder der Coverage-Judge ist ausgefallen. Ohne
        # diese Kappe stuende dort "very credible", weil kein Claim etwas
        # abgezogen hat: volle Zuversicht aus null Messungen. Genau die
        # Behauptung darf das Produkt nicht aufstellen.
        caps.append(0.64)
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
        # Wie breit die Zahl ueberhaupt getragen ist: gewertete Aussagen und
        # die, die zu wenige Modelle behandelt haben. Ohne diese beiden Zahlen
        # sieht ein Score aus 2 Claims genauso aus wie einer aus 40.
        "scored_claims": len(ratios),
        "thin_claims": thin,
    }

