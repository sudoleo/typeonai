"""Source-level contracts for the browser-session multi-run boundary.

Behavioral isolation is covered by tests/js/run-registry.test.mjs and
tests/js/multi-run-view.test.mjs. These checks protect the script order and the
explicit context hand-offs that are easy to accidentally bypass later.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_registry_is_the_single_browser_session_owner_and_loads_before_consumers():
    bundles = read("static/js/bundles.json")
    registry = read("static/js/run-registry.js")

    assert bundles.index("static/js/run-registry.js") < bundles.index("static/firebase.js")
    assert bundles.index("static/js/run-view.js") < bundles.index("static/js/query-send.js")
    assert "const MAX_ACTIVE_RUNS = 2" in registry
    assert "const runs = new Map()" in registry
    assert "let visibleRunId = null" in registry
    assert "let selectedConversationBasis = null" in registry
    assert "conversationLocks = new Map()" in registry
    assert '"parallel_limit"' in registry
    assert '"conversation_busy"' in registry
    assert '"conversation_uncertain"' in registry
    assert "deepFreeze(cloneValue(spec.config" in registry
    assert "cancelAll(reason)" in registry
    assert "clearAll(reason" in registry


def test_query_and_consensus_callbacks_use_their_bound_context_not_visible_dom():
    query = read("static/js/query-send.js")
    consensus = read("static/js/consensus-run.js")
    sources = read("static/js/sources.js")

    assert "async function runProvider(context, providerConfig" in query
    assert "registry.isExecuting(context.runId)" in query
    assert "registry.isVisible(context.runId)" in query
    assert "context.config.providers.map(provider => runProvider(context" in query
    assert "persistenceOptions(context)" in query
    assert 'document.getElementById("openaiResponse")' not in query
    assert "window.currentEvidenceSources =" not in query

    assert "window.App.executeConsensusRun = async function (context" in consensus
    context_path = consensus.split(
        "window.App.executeConsensusRun = async function (context", 1
    )[1].split("window.getConsensus =", 1)[0]
    assert "runAnswer(context" in context_path
    assert "context.consensus.bookmarkPayload" in context_path
    assert "window.saveBookmarkConsensus?.(" in context_path
    assert 'disposition?.chat_turn_state === "failed"' in context_path
    assert 'disposition?.chat_turn_state === "pending"' in context_path
    assert "failedChatTurn" in context_path
    assert "pendingChatTurn" in context_path
    assert 'document.getElementById("openaiResponse")' not in context_path
    manual_bridge = consensus.split("window.getConsensus =", 1)[1]
    assert "if (window.App.runRegistry)" in manual_bridge
    assert manual_bridge.index("if (window.App.runRegistry)") < manual_bridge.index(
        "consensusLifecycle.startRun()"
    )
    assert "prepareResponseSourcesForEvidence" in sources


def test_bookmark_view_and_logout_keep_run_ownership_explicit():
    firebase = read("static/firebase.js")
    view = read("static/js/run-view.js")
    init = read("static/js/app-init.js")
    insights = read("static/js/consensus-insights.js")

    assert "runOptions?.runId" in firebase
    assert "conversation?.runId" in firebase
    assert "conversation?.bookmarkId" in firebase
    assert "window.App?.runRegistry?.showSavedView?.(" in firebase
    assert 'window.App?.runRegistry?.clearAll?.("logout")' in firebase
    assert "restoreRegistryRunRows()" in firebase
    assert "clearPreparedBookmarkShareResult();" in firebase.split(
        'window.addEventListener("consensio:run-registry-change"', 1
    )[1].split("function publishAuthState", 1)[0]
    assert "expectedVersion" in firebase
    assert "versionParts" in firebase

    assert "registry.setProjector(project)" in view
    # The guided-run block is part of the projection, not a free-floating
    # display: a saved bookmark opened while a run keeps going in the
    # background must not inherit its progress bar or its facts.
    assert "window.App.consensusPipeline?.detach?.()" in view
    assert "pipeline.setRunFacts?.(runFacts(context))" in view
    assert 'row.dataset.runId = context.runId' in view
    assert 'row.addEventListener("click", () => registry.show(context.runId))' in view
    assert "if (registry.isVisible(context.runId))" not in view  # projector is already selected-only

    new_comparison = init.split('getElementById("newRunButton")', 1)[1].split(
        "// Fenstergröße", 1
    )[0]
    assert "cancelCurrentQuery" not in new_comparison
    assert "cancelCurrentConsensus" not in new_comparison
    assert "clearResponseBoxes" in new_comparison

    assert "const boundContext = window.App.runRegistry?.visible?.()" in insights
    assert "resolutionBindingIsWritable(binding)" in insights
    assert "window.App.runRegistry?.beginAction?.(" in insights
    assert "expectedBookmarkVersion" in insights
    assert "signal: actionController.signal" in insights
    assert "sources: context.evidenceSources || []" in view
