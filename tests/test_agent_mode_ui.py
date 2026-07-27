from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_answer_disclosure_contract_is_mode_independent():
    """Die Einzelantworten liegen in BEIDEN Modi hinter "Show model answers"
    (seit 2026-07-27). Der Schalter ist eine der drei Aufklapp-Flaechen der
    Provenance-Fusszeile und war, an den Agent Mode gebunden, in zwei von drei
    Faellen unsichtbar — obwohl er das Wichtigste dahinter oeffnet: worauf die
    Antwort beruht. Ausgenommen bleibt nur der Hero, wo die leeren Boxen die
    Vorschau auf den kommenden Lauf sind."""
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "css" / "components-consensus.css").read_text(encoding="utf-8")

    assert 'id="agentModeAnswersToggle"' in template
    assert "Show model answers" in template
    assert '"Hide model answers"' in script
    assert '"agent-mode-show-answers"' in script
    assert "answersRow.hidden = false" in script
    assert "showModelAnswers()" in script
    assert "body:not(.is-hero):not(.agent-mode-show-answers)" in css
    # Der Agent Mode darf die Sichtbarkeit nicht mehr mitentscheiden.
    assert ".agent-mode-enabled:not(.agent-mode-show-answers)" not in css
    assert ".agent-mode-answers-row[hidden]" in css


def test_consensus_jumps_reveal_answers_without_disabling_agent_mode():
    """Claim- und Difference-Buttons teilen denselben Jump-Pfad. Dieser muss
    die Disclosure öffnen, statt die inzwischen unabhängige Agent-Mode-
    Einstellung zu verändern."""
    script = (ROOT / "static" / "js" / "consensus-insights.js").read_text(encoding="utf-8")

    jump = script.split("function jumpToModelAnswer", 1)[1].split(
        "// --- Verdict", 1
    )[0]
    assert "window.App?.agentMode?.showModelAnswers?.();" in jump
    assert "box.getBoundingClientRect()" in jump
    assert "window.requestAnimationFrame" not in jump
    assert "window.setAgentMode" not in jump


def test_answer_disclosure_is_restored_for_manual_consensus_and_bookmarks():
    """Der Footer darf nicht davon abhaengen, ob noch eine Query-Pipeline lebt."""
    progress = (ROOT / "static" / "js" / "consensus-progress.js").read_text(encoding="utf-8")
    firebase = (ROOT / "static" / "firebase.js").read_text(encoding="utf-8")

    assert 'if (stage !== "consensus" && stage !== "differences") {' in progress
    assert "renderProvenance();" in progress
    assert "consensusPipeline?.renderProvenance?.()" in firebase
