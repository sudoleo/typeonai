from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_verdict_color_and_label_follow_the_agreement_score():
    script = (ROOT / "static" / "js" / "consensus-insights.js").read_text(
        encoding="utf-8"
    )
    verdict = script.split("function renderVerdictHeader", 1)[1].split(
        "// --- Agreement-Badges", 1
    )[0]

    assert 'boundedScore >= 65 ? "is-calm"' in verdict
    assert 'boundedScore >= 40 ? "is-warn" : "is-alert"' in verdict
    assert "boundedScore >= 85" in verdict
    assert 'headline.textContent = "High agreement"' in verdict
    assert 'headline.textContent = "Strong agreement"' in verdict
    assert 'headline.textContent = "Partial agreement"' in verdict
    assert 'headline.textContent = "Low agreement"' in verdict
    assert 'headline.textContent = "Very low agreement"' in verdict


def test_public_mockups_use_the_same_score_semantics():
    landing = (ROOT / "templates" / "landing.html").read_text(encoding="utf-8")
    engine = (ROOT / "templates" / "consensus-engine.html").read_text(
        encoding="utf-8"
    )
    product = (
        ROOT / "templates" / "partials" / "product_result_mockup.html"
    ).read_text(encoding="utf-8")

    assert "Agreement on the core" not in landing
    assert "Agreement on the core" not in engine
    assert "Agreement on the core" not in product
    assert '<div class="consensus-verdict is-calm">' in product
    assert '<span class="verdict-headline">Strong agreement</span>' in product
    assert (
        'style="--val:52" title="Agreement score 52/100"' in landing
        and '<span class="verdict-headline">Partial agreement</span>' in landing
    )
    # Der Landing-Walkthrough zeigt denselben Lauf wie die Demo in /app: unter
    # 65 faerbt der Verdict-Balken amber, sonst behauptet das Mockup eine Ruhe,
    # die der Score nicht deckt.
    assert '<div class="consensus-verdict is-warn">' in landing
    assert (
        'style="--val:64" title="Agreement score 64/100"' in engine
        and '<span class="verdict-headline">Partial agreement</span>' in engine
    )
    assert (
        'style="--val:82" title="Agreement score 82/100"' in engine
        and '<span class="verdict-headline">Strong agreement</span>' in engine
    )
