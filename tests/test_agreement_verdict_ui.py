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


def test_settings_offer_three_agreement_display_levels_persistently():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    init = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")
    shell = (ROOT / "static" / "css" / "shell.css").read_text(encoding="utf-8")

    assert 'id="agreementDisplaySelect"' in template
    for value in ("full", "summary", "off"):
        assert f'<option value="{value}">' in template
    assert "consensio.agreementDisplay.v1" in init
    assert 'classList.toggle("agreement-score-hidden", mode === "summary")' in init
    assert 'classList.toggle("agreement-verdict-hidden", mode === "off")' in init
    # Nur die Zahl weg: die Worte des Urteils bleiben.
    assert "body.agreement-score-hidden .verdict-gauge" in shell
    # Alles weg: unter der Antwort bleiben nur die Schubladen.
    assert "body.agreement-verdict-hidden .consensus-verdict" in shell


def test_the_hidden_verdict_lets_drawers_share_the_line_with_the_actions():
    shell = (ROOT / "static" / "css" / "shell.css").read_text(encoding="utf-8")

    merged = shell.split("body.agreement-verdict-hidden .run-provenance::before", 1)[1]
    merged = merged.split("/* The headline is a headline", 1)[0]

    # Ohne Urteil stand Share/Watch/Cite allein in einer halbleeren Zeile ueber
    # den Schubladen. Wo die Breite reicht, teilen sie sich eine Grundlinie.
    assert "@media (min-width: 641px)" in merged
    assert "flex-wrap: wrap" in merged
    for area, order in (("tabs", 1), ("facts", 2), ("actions", 3)):
        block = merged.split(f"consensus-footer-{area} {{", 1)[1].split("}", 1)[0]
        assert f"order: {order}" in block
    # Die Fakten duerfen die Zeile weder aufsaugen (flex-grow) noch als leere
    # Huelle eine Luecke kosten — beides trieb die Aktionen in den Umbruch.
    assert "flex: 0 1 auto" in merged
    assert ":has(.run-provenance-facts:not(:empty))" in merged
    assert ":has(.run-replay-btn:not([hidden]))" in merged


def test_the_old_agreement_score_switch_choice_still_applies():
    init = (ROOT / "static" / "js" / "app-init.js").read_text(encoding="utf-8")

    assert "consensio.showAgreementScore.v1" in init
    assert 'localStorage.getItem(AGREEMENT_SCORE_STORAGE_KEY) === "false"' in init


def test_sentence_checks_have_a_discreet_persistent_visibility_control():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    insights = (ROOT / "static" / "js" / "consensus-insights.js").read_text(
        encoding="utf-8"
    )
    shell = (ROOT / "static" / "css" / "shell.css").read_text(encoding="utf-8")
    inputs = (ROOT / "static" / "css" / "components-input.css").read_text(
        encoding="utf-8"
    )

    assert 'id="consensusMarkerToggle"' in template
    assert '>Hide checks</button>' in template
    assert ":not(.consensus-marker-toggle)" in inputs
    assert "consensio.showConsensusMarkers.v1" in insights
    assert 'classList.toggle(MARKERS_HIDDEN_CLASS, !show)' in insights
    assert 'show ? "Hide checks" : "Show checks"' in insights
    assert "body.consensus-markers-hidden .consensus-marker-legend-copy" in shell
    assert "body.consensus-markers-hidden .consensus-answer-body .claim-badge" in shell


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
