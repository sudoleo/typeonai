from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.source_contract


def test_answer_disclosure_is_agent_mode_only():
    """Im Direktvergleich sind die sechs Antworten selbst das Ergebnis.
    Nur Agent Mode darf sie nach dem Consensus hinter dem Disclosure sammeln."""
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "css" / "components-consensus.css").read_text(encoding="utf-8")
    shell = (ROOT / "static" / "css" / "shell.css").read_text(encoding="utf-8")

    assert 'id="agentModeAnswersToggle"' in template
    assert "Compare answers" in template
    assert '"Hide answers"' in script
    assert '"agent-mode-show-answers"' in script
    assert "answersRow.hidden = false" in script
    assert "showModelAnswers()" in script
    assert "body:not(.is-hero).agent-mode-enabled:not(.agent-mode-show-answers)" in css
    assert "body:not(.is-hero):not(.agent-mode-show-answers)" not in css
    assert "body:not(.is-hero).agent-mode-enabled:not(.agent-mode-show-answers)" in shell
    assert ".agent-mode-answers-row[hidden]" in css


def test_disabled_agent_mode_is_a_direct_six_answer_flow():
    query = (ROOT / "static" / "js" / "query-send.js").read_text(encoding="utf-8")
    agent = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")
    core = (ROOT / "static" / "js" / "app-core.js").read_text(encoding="utf-8")
    input_css = (ROOT / "static" / "css" / "components-input.css").read_text(encoding="utf-8")

    assert 'const agentModeAtStart = isAgentModeEnabled?.() === true;' in query
    assert 'document.body.classList.add("is-hero", "direct-comparison-active")' in query
    assert 'window.App.followup?.reset?.();' in query
    assert 'window.App.chatSession?.reset?.();' in query
    assert 'window.App?.consensusPipeline?.dismiss?.();' in query
    assert 'const autoConsensusOn = agentModeAtStart' in query
    assert '&& isAgentModeEnabled?.() === true' in query
    assert 'if (!autoConsensusOn) return;' in query
    assert 'autoToggle.checked = !!enabled;' in agent
    assert 'autoToggle.disabled = true;' in agent
    assert 'if (isAgentModeEnabled()) {' in agent
    assert 'directComparisonActive' in core
    assert 'body.is-hero.direct-comparison-active .response-section' in input_css
    assert 'body.is-hero.direct-comparison-active .consensus-section' in input_css


def test_plus_menu_agent_mode_switch_is_free_and_synchronized():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")

    assert 'for="agentModeMenuSwitch"' in template
    assert 'id="agentModeMenuSwitch"' in template
    assert 'class="attach-menu-toggle"' in template
    assert 'class="switch deep-switch attach-menu-switch"' in template
    agent_row = template.split('for="agentModeMenuSwitch"', 1)[1].split("</label>", 1)[0]
    assert "Agent Mode" in agent_row
    assert "pro-badge" not in agent_row
    assert 'const menuSwitchEl = document.getElementById("agentModeMenuSwitch");' in script
    assert 'if (menuSwitchEl) menuSwitchEl.checked = enabled;' in script
    assert 'agentModeMenuSwitch.addEventListener("change"' in script
    assert 'setAgentMode(this.checked, { persist: true });' in script


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


def test_consensus_actions_are_explicit_and_hover_preview_has_no_native_duplicate():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    insights = (ROOT / "static" / "js" / "consensus-insights.js").read_text(
        encoding="utf-8"
    )

    assert "Review differences" in template
    assert "Compare answers" in template
    assert "Verify sources" in template
    assert "badge.title" not in insights
    assert "marker.title" not in insights
    assert "span.title" not in insights


def test_answer_disclosure_is_restored_for_manual_consensus_and_bookmarks():
    """Der Footer darf nicht davon abhaengen, ob noch eine Query-Pipeline lebt."""
    progress = (ROOT / "static" / "js" / "consensus-progress.js").read_text(encoding="utf-8")
    firebase = (ROOT / "static" / "firebase.js").read_text(encoding="utf-8")

    assert 'if (stage !== "consensus" && stage !== "differences") {' in progress
    assert "renderProvenance();" in progress
    assert "consensusPipeline?.renderProvenance?.()" in firebase
