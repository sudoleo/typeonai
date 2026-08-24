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
    view = (ROOT / "static" / "js" / "run-view.js").read_text(encoding="utf-8")
    agent = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")
    core = (ROOT / "static" / "js" / "app-core.js").read_text(encoding="utf-8")
    input_css = (ROOT / "static" / "css" / "components-input.css").read_text(encoding="utf-8")

    assert 'const agentMode = window.isAgentModeEnabled?.() === true;' in query
    assert "const config = {\n      agentMode," in query
    # Die Vergleichsansicht wird an EINER Stelle aufgebaut, damit ein frisch
    # gesendeter und ein aus einem Bookmark geladener Direktvergleich nicht
    # auseinanderlaufen.
    assert 'if (context.config?.agentMode === false)' in view
    assert 'window.enterDirectComparisonView?.();' in view
    assert 'document.body.classList.add("is-hero", "direct-comparison-active")' in core
    assert 'followup?.reset?.();' in view
    assert 'window.App.chatSession?.reset?.();' in view
    assert 'pipeline.dismiss?.();' in view
    assert 'if (context.config.agentMode && context.config.autoConsensus)' in query
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


def test_composer_and_its_plus_menu_stay_above_the_answer_boxes():
    """Waehrend des Hero-Slides ist die .input-section selbst ein
    Stacking-Context — der z-index des offenen (+)-Menues bleibt darin
    gefangen. Ohne eigenen z-index gewannen die spaeter im DOM stehenden
    .response-box (will-change/isolation) und ihre Modellnamen lagen fuer die
    Dauer der Umschalt-Animation vor dem Menue."""
    input_css = (ROOT / "static" / "css" / "components-input.css").read_text(encoding="utf-8")

    base = input_css.split(".input-section {", 1)[1].split("}", 1)[0]
    assert "position: relative;" in base
    assert "z-index: 5;" in base
    # Der Direktvergleich ist kein Leerzustand: keine Begruessung ueber dem
    # nach oben gerueckten Composer.
    assert "body.is-hero.agent-mode-enabled:not(.direct-comparison-active) .hero-greeting" in input_css


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
