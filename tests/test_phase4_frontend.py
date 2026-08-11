"""Phase-4 frontend correctness contracts.

The browser suite exercises the race-sensitive paths. These source contracts
keep the architectural guardrails visible when the classic-script modules are
edited without a frontend unit-test runner.
"""

from pathlib import Path

import pytest


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

    assert 'markQueryBlockingError("All selected model requests failed.")' in query
    assert 'status: "error"' in query
    assert 'document.getElementById("consensusAnswerBody")' in query
    assert 'document.getElementById("consensusResponse").querySelector("p")' not in query


def test_minimum_model_count_is_rechecked_after_attachment_filter_before_usage():
    query = read("static/js/query-send.js")
    filtered_check = query.index("selectedProviderConfigsForRun.length < 2")
    usage_start = query.index("window.App.usageRun.start", filtered_check)

    assert filtered_check < usage_start
    assert 'reason: "minimum_models_after_filters"' in query


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

    assert template.index("/static/js/auth-bootstrap.js?") < template.index("/static/firebase.js?")
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
    assert "fine rule" in landing
    assert "heavier amber rule" in landing
    assert "fine dotted line" not in landing
    assert "amber wavy line" not in landing
