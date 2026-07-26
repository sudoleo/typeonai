from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_consensus_result_precedes_model_answers_and_run_block_is_loaded():
    template = read("templates/index.html")

    assert template.index('class="consensus-section"') < template.index(
        'class="response-section"'
    )
    assert 'id="consensusRun"' in template
    assert 'id="runStatus"' in template
    assert 'src="/static/js/consensus-progress.js?' in template
    assert template.index("/static/js/agent-mode.js?") < template.index(
        "/static/js/consensus-progress.js?"
    ) < template.index("/static/js/consensus-lifecycle.js?")


def test_run_block_shows_one_step_at_a_time():
    """The guided run's whole point is that exactly one step is active. The
    markup therefore has ONE label, ONE count, ONE bar and ONE next line —
    if a second set ever appears here, the reduction has been undone."""
    template = read("templates/index.html")

    for single in ('id="runLabel"', 'id="runCount"', 'id="runBar"', 'id="runNext"'):
        assert template.count(single) == 1, f"expected exactly one {single}"

    # Finished steps collapse into the past line; per-model rows live in the
    # detail container and are only shown while the models answer.
    assert 'id="runPast"' in template
    assert 'id="runDetail"' in template


def test_run_covers_every_phase_and_terminal_state():
    progress = read("static/js/consensus-progress.js")
    lifecycle = read("static/js/consensus-lifecycle.js")
    query = read("static/js/query-send.js")
    demo = read("static/demo.js")

    # Four phases, in order, each with its own step.
    for stage in ("prepare", "answers", "consensus", "differences"):
        assert f'{stage}: {{' in progress or f'"{stage}"' in progress, stage

    assert 'responseState === "complete"' in progress
    assert "onConsensusStart" in lifecycle
    assert "onConsensusEnd" in lifecycle
    assert 'setAgentModeStatus("canceled")' in query
    assert "consensusPipeline?.onConsensusEnd" in demo

    # The run starts at /prepare, not at the fan-out: the gap between click
    # and first answer is exactly where a user needs to see something happen.
    assert "consensusPipeline?.onPrepare" in query
    # The differences judge gets its own step, signalled from the stream.
    assert "onDifferencesStart" in read("static/js/consensus-run.js")


def test_run_is_compact_and_unknowable_phases_stay_indeterminate():
    css = read("static/css/shell.css")

    assert ".run.is-visible" in css
    # Collapses away instead of vanishing, so the eye follows the answer up.
    assert ".run.is-gone" in css
    assert "height: 2px" in css
    # A phase that cannot report a share of itself sweeps, it does not fake
    # a percentage.
    assert ".run-track.is-indeterminate" in css
    assert "runSweep" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_run_hands_over_to_a_provenance_line():
    template = read("templates/index.html")
    progress = read("static/js/consensus-progress.js")
    css = read("static/css/shell.css")

    assert 'id="runProvenance"' in template
    assert 'id="runProvenanceFacts"' in template
    assert 'id="runReplayButton"' in template
    assert ".run-provenance" in css
    assert "renderProvenance" in progress


def test_followup_offer_sits_with_the_answer_and_the_chip_with_the_input():
    """The offer belongs to the answer it would send as context; the armed
    context chip belongs to the field it will be sent from."""
    template = read("templates/index.html")
    run = read("static/js/consensus-run.js")

    # Offer slot inside the provenance line, chip slot inside the composer.
    provenance_at = template.index('id="runProvenance"')
    offer_at = template.index('id="followupBar"')
    chip_at = template.index('id="followupChipBar"')
    input_at = template.index('id="questionInput"')

    assert provenance_at < offer_at
    assert chip_at < input_at
    assert "followupChipBar" in run


def test_composer_row_is_reduced_to_attach_run_switch_and_send():
    """Agent Mode, the mode explainer and the clear button left the composer.
    Their functions live in Settings, the (+) menu and 'New comparison'."""
    template = read("templates/index.html")

    assert 'id="attachTrigger"' in template
    assert 'id="consensusModelDropdown"' in template
    assert 'id="sendButton"' in template

    assert 'id="toggleAllButton"' not in template
    assert 'id="modeExplainerTrigger"' not in template
    assert 'id="clearButton"' not in template

    # ...but the functions are still reachable.
    assert 'id="agentModeSwitch"' in template
    assert 'id="autoConsensusToggle"' in template
    assert 'id="deepSearchToggle"' in template
    assert 'id="newRunButton"' in template


def test_sidebar_top_row_is_toggle_plus_new_comparison():
    template = read("templates/index.html")

    top = template.index('class="sidebar-top"')
    toggle = template.index('id="sidebarToggleInner"')
    new_run = template.index('id="newRunButton"')
    search = template.index('id="chatSearch"')

    assert top < toggle < new_run < search
    # Exactly one sidebar toggle in the markup besides the floating one.
    assert template.count('id="sidebarToggleInner"') == 1
    assert template.count('id="toggleSidebarButton"') == 1


def test_consensus_loader_matches_the_run_visual_language():
    consensus_css = read("static/css/components-consensus.css")
    feedback_css = read("static/css/components-feedback.css")

    assert ".consensus-box.is-synthesizing::after" in consensus_css
    assert "height: 2px" in consensus_css
    assert "animation: consensusLoadingLine" in consensus_css
    assert "background: transparent" in consensus_css
    assert "width: 4px" in feedback_css
    assert "box-shadow: none" in feedback_css
    assert "consensusSynthesisSweep" not in feedback_css
