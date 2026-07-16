from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_consensus_result_precedes_model_answers_and_pipeline_is_loaded():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert template.index('class="consensus-section"') < template.index(
        'class="response-section"'
    )
    assert 'id="consensusPipeline"' in template
    assert 'id="consensusPipelineStatus"' in template
    assert 'src="/static/js/consensus-progress.js?' in template
    assert template.index("/static/js/agent-mode.js?") < template.index(
        "/static/js/consensus-progress.js?"
    ) < template.index("/static/js/consensus-lifecycle.js?")


def test_pipeline_contract_covers_both_phases_and_terminal_states():
    progress = (ROOT / "static" / "js" / "consensus-progress.js").read_text(
        encoding="utf-8"
    )
    lifecycle = (ROOT / "static" / "js" / "consensus-lifecycle.js").read_text(
        encoding="utf-8"
    )
    query = (ROOT / "static" / "js" / "query-send.js").read_text(encoding="utf-8")
    demo = (ROOT / "static" / "demo.js").read_text(encoding="utf-8")

    assert 'responseState === "complete"' in progress
    assert 'stage = "consensus"' in progress
    assert 'stage = "answers-done"' in progress
    assert "onConsensusStart" in lifecycle
    assert "onConsensusEnd" in lifecycle
    assert 'setAgentModeStatus("canceled")' in query
    assert "consensusPipeline?.onConsensusEnd" in demo


def test_pipeline_visual_is_compact_and_synthesis_is_indeterminate():
    css = (ROOT / "static" / "css" / "components-input.css").read_text(
        encoding="utf-8"
    )

    assert ".consensus-pipeline.is-visible" in css
    assert "max-height: 34px" in css
    assert "height: 2px" in css
    assert "consensusPipelineSweep" in css
    assert '@media (prefers-reduced-motion: reduce)' in css


def test_consensus_loader_matches_the_compact_pipeline_visual_language():
    consensus_css = (ROOT / "static" / "css" / "components-consensus.css").read_text(
        encoding="utf-8"
    )
    feedback_css = (ROOT / "static" / "css" / "components-feedback.css").read_text(
        encoding="utf-8"
    )

    assert ".consensus-box.is-synthesizing::after" in consensus_css
    assert "height: 2px" in consensus_css
    assert "animation: consensusLoadingLine" in consensus_css
    assert "background: transparent" in consensus_css
    assert "width: 4px" in feedback_css
    assert "box-shadow: none" in feedback_css
    assert "consensusSynthesisSweep" not in feedback_css
