"""Phase-4 frontend correctness contracts.

The browser suite exercises the race-sensitive paths. These source contracts
keep the architectural guardrails visible when the classic-script modules are
edited without a frontend unit-test runner.
"""

from pathlib import Path

import pytest

from tests.frontend_order import loads_before


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.source_contract


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_usage_countdown_tracks_utc_midnight_and_refreshes_server_state():
    app_init = read("static/js/app-init.js")
    countdown = section(app_init, "function utcDayKey", 'document.getElementById("logoLink")')

    assert "getUTCFullYear()" in countdown
    assert "Date.UTC(" in countdown
    assert "window.refreshUsageData?.()" in countdown
    assert "usageRefreshTargetUtcDay" in countdown
    assert "refreshed === true" in countdown
    assert "location.reload()" not in countdown


def test_bookmark_writes_and_views_are_bound_to_auth_and_intent_generations():
    firebase = read("static/firebase.js")
    model_save = section(firebase, "async function saveBookmark(", "window.saveBookmark =")
    consensus_save = section(
        firebase, "async function saveBookmarkConsensus(", "window.saveBookmarkConsensus ="
    )
    open_bookmark = section(firebase, "window.openBookmark =", "window.loadAllBookmarkMetadata")

    for save in (model_save, consensus_save):
        assert "requestUid" in save and "requestGeneration" in save
        assert save.count("isCurrentAuthenticatedUser(requestUid, requestGeneration)") >= 3
    assert "const requestEpoch = ++bookmarkViewEpoch" in open_bookmark
    assert open_bookmark.count("viewIsCurrent()") >= 4


def test_bookmark_save_failures_are_visible_and_deduplicated():
    firebase = read("static/firebase.js")
    notice = section(
        firebase,
        "function showBookmarkSaveError(",
        "async function saveBookmark(",
    )
    model_save = section(firebase, "async function saveBookmark(", "window.saveBookmark =")
    consensus_save = section(
        firebase, "async function saveBookmarkConsensus(", "window.saveBookmarkConsensus ="
    )

    assert "Bookmark limit reached" in notice
    assert "Bookmark storage is full" in notice
    assert "600_000" in notice
    assert "window.App?.showPopup?.(message)" in notice
    assert 'showBookmarkSaveError(res.status, data.detail, `${bookmarkId}:${question}`)' in model_save
    assert 'showBookmarkSaveError(res.status, data.detail, `${bookmarkId}:${question}`)' in consensus_save


def test_bookmark_model_and_consensus_writes_are_serialized_per_saved_run():
    firebase = read("static/firebase.js")
    queue = section(
        firebase,
        "function enqueueBookmarkWrite(",
        "function showBookmarkSaveError(",
    )
    model_save = section(firebase, "async function saveBookmark(", "window.saveBookmark =")
    consensus_save = section(
        firebase, "async function saveBookmarkConsensus(", "window.saveBookmarkConsensus ="
    )

    assert "bookmarkWriteChains.get(queueKey) || Promise.resolve()" in queue
    assert ".catch(() => undefined)" in queue
    assert "bookmarkWriteChains.delete(queueKey)" in queue
    assert "return enqueueBookmarkWrite(" in model_save
    assert "return enqueueBookmarkWrite(" in consensus_save


def test_share_requests_use_auth_snapshots_abort_controllers_and_view_epochs():
    share = read("static/js/share-dialog.js")

    assert "sharedModalEpoch" in share
    assert "activeShareControllers" in share
    assert "controller.abort()" in share
    assert "authSnapshotIsCurrent" in share
    assert "shareViewIsCurrent" in share
    assert "if (err?.stale) return" in share


def test_complete_model_failure_is_an_error_and_marker_legend_is_not_cleared():
    query = read("static/js/query-send.js")

    assert 'finishFailed(context, "All selected model requests failed.")' in query
    assert 'type: "run_failed"' in query
    # Query callbacks mutate only their RunContext. The selected-view adapter
    # owns all response/consensus DOM writes.
    assert 'document.getElementById("consensusAnswerBody")' not in query
    assert 'document.getElementById("consensusResponse").querySelector("p")' not in query


def test_minimum_model_count_is_rechecked_after_attachment_filter_before_usage():
    query = read("static/js/query-send.js")
    filtered_check = query.index("if (providers.length < 2)")
    usage_start = query.index("const usage = createUsage", filtered_check)

    assert filtered_check < usage_start
    assert "const providers = selectedProviders(attachments.length, deepSearch);" in query
    assert 'provider === "DeepSeek" && attachmentCount > 0' in query


def test_usage_snapshot_can_recover_pro_tier_after_status_failure():
    firebase = read("static/firebase.js")
    usage = section(firebase, "async function fetchUsageData", "window.refreshUsageData")

    assert 'window.App.state.set("isUserPro", isPro, "userTier")' in usage
    assert "window.updateUserTierUI(isPro, true)" in usage
    assert "window.setCurrentUsageLimits(isPro, data)" in usage


def test_auth_bootstrap_watchdog_precedes_firebase_and_clears_stale_skeletons():
    template = read("templates/index.html")
    watchdog = read("static/js/auth-bootstrap.js")
    bootstrap = read("static/js/app-bootstrap.js")

    assert loads_before("auth-bootstrap.js", "firebase.js")
    assert 'document.getElementById("authTopActions")' in bootstrap
    assert "authTopActions.hidden = false" in bootstrap
    assert "Login is temporarily unavailable" in watchdog
    assert 'window.dispatchEvent(new CustomEvent("consensio:auth-unavailable"))' in watchdog


def test_watch_modal_route_and_brief_state_have_deterministic_rollback_contracts():
    watch = read("static/js/watch.js")
    list_branch = section(watch, 'if (view === "list")', "const { modal } = els()")
    reset = section(watch, "function resetAfterLogout()", "window.openWatchDialog")

    assert "closeDialog();" in list_branch
    assert "renderWatchLoginHint" in reset
    assert 'replaceState(null, "", APP_PATH)' not in reset
    assert 'window.addEventListener("consensio:auth-state"' in watch
    assert "persistedSendTime" in watch and "persistedMode" in watch
    assert "timeInput.value = persistedSendTime" in watch
    assert "modeSelect.value = persistedMode" in watch
    assert "watchModalIntentIsCurrent" in watch
    assert "window.App.sharedModal.isCurrent(intent)" in watch


def test_bookmark_restore_uses_owned_run_state_and_token_wait_is_fenced():
    firebase = read("static/firebase.js")
    query = read("static/js/query-send.js")

    assert 'window.App.state.set("lastQuestion", displayQuestion, "run")' in firebase
    assert "lastQuestion = displayQuestion" not in firebase
    token_wait = section(
        query,
        "idToken = await context.auth.user?.getIdToken?.()",
        "if (!idToken)",
    )
    assert "runIsCurrent(context)" in token_wait
    assert "registry.isAuthCurrent(context)" in query


def test_bookmark_load_failure_has_a_visible_retry_state():
    firebase = read("static/firebase.js")

    assert "function renderBookmarksLoadError" in firebase
    assert 'message.textContent = "Bookmarks could not be loaded."' in firebase
    assert 'retry.textContent = "Try again"' in firebase
    assert "renderBookmarksLoadError(container)" in firebase


def test_account_popup_and_settings_controls_have_exactly_one_binder():
    firebase = read("static/firebase.js")
    app_init = read("static/js/app-init.js")
    app_ui = read("static/app-ui.js")

    assert "removeEventListener(\"click\", accountMenuDocumentClickHandler)" in firebase
    assert "accountMenuDocumentClickHandler = e =>" in firebase
    for control in (
        "editSystemPromptBtn", "closeSystemPromptModal", "saveSystemPromptBtn",
        "helpButton", "closeHelpModal",
    ):
        assert control not in app_init
        assert control in app_ui


def test_login_dialog_accessibility_and_current_landing_marker_vocabulary():
    template = read("templates/index.html")
    firebase = read("static/firebase.js")
    landing = read("templates/landing.html")

    assert 'id="loginModal" class="modal" role="dialog" aria-modal="true"' in template
    assert 'aria-labelledby="authTitle"' in template
    assert 'id="closeLoginModal" aria-label="Close login dialog"' in template
    assert 'if (event.key === "Escape")' in firebase
    assert "authModalReturnFocus.focus" in firebase
    assert 'event.key !== "Tab"' in firebase
    # Seit 2026-08-15 sind die Marken farbige Textmarker statt Linien unter
    # dem Satz (User-Vorgabe: eine Unterstreichung liest sich wie ein Link).
    # Die Legende muss von Farben sprechen, nicht mehr von Linienstaerken.
    assert "green &middot; all of them agreed" in landing or "green · all of them agreed" in landing
    assert "red · they contradict each other" in landing
    assert "fine rule" not in landing
    assert "heavier amber rule" not in landing
    assert "fine dotted line" not in landing
    assert "amber wavy line" not in landing
