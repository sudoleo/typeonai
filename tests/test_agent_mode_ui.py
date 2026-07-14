from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_mode_answer_disclosure_contract():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "js" / "agent-mode.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "css" / "components-consensus.css").read_text(encoding="utf-8")

    assert 'id="agentModeAnswersToggle"' in template
    assert "Show model answers" in template
    assert '"Hide model answers"' in script
    assert '"agent-mode-show-answers"' in script
    assert ".agent-mode-enabled:not(.agent-mode-show-answers)" in css
    assert ".agent-mode-answers-row[hidden]" in css
