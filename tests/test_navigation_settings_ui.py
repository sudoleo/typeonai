from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_sidebar_navigation_is_self_contained_and_guest_login_is_top_only():
    template = read("templates/index.html")
    landing = read("templates/landing.html")
    app_init = read("static/js/app-init.js")
    model_picker = read("static/js/model-picker.js")
    shell = read("static/css/shell.css")
    input_css = read("static/css/components-input.css")

    assert 'id="sidebarRail"' not in template
    assert "sidebar-rail-btn" not in template
    assert "sidebar-rail-btn" not in app_init
    assert template.count('class="sidebar-heading-label"') == 1
    assert "<span>Bookmarks</span>" in template
    assert 'id="sidebarModelPicker"' in template
    assert '<strong>Models</strong>' in template
    assert '<span>Leaderboard</span>' not in template
    assert 'id="modelInsights"' not in template
    assert 'class="faq-answer faq-model-insights"' not in template
    assert 'id="leaderboardContent"' not in template
    assert 'class="lp-hero-pulse" href="/model-pulse"' in landing
    assert 'id="modelLeaderboard"' not in landing
    assert ".faq-item > p, .faq-item > .faq-answer" in app_init
    assert 'id="authTopActions"' in template
    assert 'id="loginContainer" class="login-text" hidden></div>' in template
    assert 'class="sidebar-bookmarks-head"' in template
    assert 'class="sidebar-section bookmarks-section is-locked"' in template
    assert 'id="bookmarksToggle"' in template and 'aria-disabled="true" disabled' in template
    assert 'id="bookmarkSearchTrigger"' in template
    assert 'aria-label="Search bookmarks"' in template
    assert 'aria-disabled="true" disabled' in template
    assert ".sidebar-bookmarks-head.is-searching .sidebar-search-box" in shell
    assert ".sidebar-bookmarks-head:hover .sidebar-search-box" not in shell
    assert "setBookmarkSearchOpen(true, { focus: true })" in app_init
    assert 'event.key !== "Escape"' in app_init
    assert ".sidebar-bookmarks-head .sidebar-toggle-btn" in shell
    assert ".sidebar-bookmarks-head .sidebar-heading-label span" in shell
    assert "justify-self: start" in shell
    assert ".models-section {\n  margin-bottom: 8px;\n}" in shell
    search_rule = shell.split("\n.sidebar-search-box {\n", 1)[1].split("}", 1)[0]
    assert "inset: -4px" in search_rule
    assert "24px" not in search_rule
    assert ".sidebar-bookmarks-head .sidebar-toggle-arrow" in shell
    arrow_rule = shell.split(".sidebar-bookmarks-head .sidebar-toggle-arrow {", 1)[1].split("}", 1)[0]
    assert "justify-self: end" in arrow_rule
    assert "left: 0.25px" in arrow_rule
    assert "padding: 7px 32px 7px 32px" in shell
    assert "grid-template-columns: 18px minmax(0, 1fr)" in shell
    assert ".bookmarks-section.is-locked .sidebar-search-box" in shell
    assert 'id="sidebarModelCount"' in template
    assert 'class="model-selection-check"' not in template
    assert 'id="modelSelectionArea" hidden' in template
    assert "window.App.openModelPicker(consensusSelect)" in app_init
    assert "function syncSidebarModelCount()" in model_picker
    assert ":not(.sidebar-model-entry)" in input_css
    assert ":not(.sidebar-search-trigger)" in input_css


def test_public_navigation_is_compact_and_learning_links_live_in_footer():
    nav = read("templates/partials/public_nav.html")
    footer = read("templates/partials/public_footer.html")
    assert ">Models</a>" not in nav
    assert ">About</a>" not in nav
    assert 'href="/ai-model-comparison">Model guide</a>' in footer
    assert 'href="/model-pulse">Model pulse</a>' in footer
    assert 'href="/about">About</a>' in footer
    assert '{% set public_nav_cta_href = "/app" %}' in read("templates/share.html")
    assert '<a href="/app" class="button-primary">Ask your own question</a>' in read("templates/share.html")


def test_demo_watch_nudge_and_dedicated_model_pulse_match_the_product_contract():
    demo = read("static/demo.js")
    misc_css = read("static/css/components-misc.css")
    watch = read("static/js/watch.js")
    leaderboard = read("static/js/model-pulse.js")
    pulse_page = read("templates/model-pulse.html")
    assert "score: 83" in demo
    assert 'btn.textContent = "Watch demo"' in demo
    assert "inputActions.prepend(btn)" in demo
    demo_rule = misc_css.split(".demo-chip {", 1)[1].split("}", 1)[0]
    assert "position: static" in demo_rule
    assert 'severity: "minor"' in demo
    assert '[S8].</li>' in demo
    assert 'class="watch-feature-nudge-close" aria-label="Dismiss new feature tip"></button>' in watch
    assert '[/anthropic|claude/i, "/static/icons/chat_icons/claude.png"]' in leaderboard
    assert 'id="modelLeaderboard"' in pulse_page
    assert "not a popularity vote" in pulse_page.lower()
    assert 'href="/benchmark"' in pulse_page


def test_consensus_run_requires_two_selected_models_before_starting():
    app_init = read("static/js/app-init.js")
    query_send = read("static/js/query-send.js")
    model_picker = read("static/js/model-picker.js")

    assert "const hasMinimumModels = selectedModelCount >= 2" in app_init
    assert "sendButton.disabled = !canStartRun" in app_init
    assert "if (selectedModelCount < 2)" in query_send
    assert 'reason: "minimum_models"' in query_send
    assert "choose at least 2" in model_picker


def test_cross_check_greeting_uses_the_same_light_typography_in_app_and_landing():
    app_css = read("static/css/components-input.css")
    landing_css = read("static/css/landing.css")

    app_rule = app_css.split(".hero-greeting {", 1)[1].split("}", 1)[0]
    landing_rule = landing_css.split(".lp-app-greeting {", 1)[1].split("}", 1)[0]
    assert "font-weight: var(--font-weight-regular)" in app_rule
    assert "font-weight: var(--font-weight-regular)" in landing_rule


def test_demo_never_writes_product_usage_signals():
    demo = read("static/demo.js")

    assert "recordModelVote" not in demo
    assert "saveBookmarkConsensus" not in demo
    assert 'fetch("/consensus"' not in demo


def test_mobile_brand_and_desktop_input_centering_contract():
    layout = read("static/css/layout.css")

    assert "calc(100vw - 520px)" in layout
    assert "translateX(130px)" not in layout
    assert "@media (max-width: 1099px)" in layout
    assert ".brand-float," in layout
    assert "body:has(.sidebar.collapsed) .brand-float" in layout


def test_fixed_navigation_yields_while_consensus_or_watch_content_is_read():
    app_init = read("static/js/app-init.js")
    layout = read("static/css/layout.css")
    watch_css = read("static/css/components-watch.css")

    assert "function initReadingChrome()" in app_init
    assert 'window.addEventListener("scroll", () => handleScroll(window, mainState)' in app_init
    assert 'watchPage?.addEventListener("scroll", () => handleScroll(watchPage, watchState)' in app_init
    assert 'body.classList.toggle("is-reading-chrome-hidden"' in app_init
    assert '["Tab", "Home", "PageUp", "ArrowUp"]' in app_init
    assert "body.is-reading-chrome-hidden .app-nav-float" in layout
    assert "body.is-reading-chrome-hidden .view-switch" in layout
    assert "prefers-reduced-motion: reduce" in layout
    assert "transition: transform 0.22s ease" in watch_css


def test_disclaimer_stays_attached_below_the_moving_input_section():
    template = read("templates/index.html")
    input_css = read("static/css/components-input.css")

    input_start = template.index('<div class="input-section">')
    footer = template.index('<footer class="app-footer">')
    consensus = template.index('<div class="consensus-section">')

    assert input_start < footer < consensus
    assert template.count('<footer class="app-footer">') == 1
    assert "body.is-hero .app-footer" not in input_css


def test_light_input_is_white_and_account_popup_uses_opaque_surfaces():
    input_css = read("static/css/components-input.css")
    layout = read("static/css/layout.css")
    firebase = read("static/firebase.js")

    assert "body:not(.dark-mode) .chat-input-container" in input_css
    assert "background: #fff" in input_css
    assert ".email-popup" in layout
    assert "background: #fff" in layout
    assert ".dark-mode .email-popup" in layout
    assert "background: #282828" in layout
    assert 'class="email-popup" role="menu" hidden' in firebase
    assert "emailPopup.hidden = !isOpen" in firebase


def test_chat_textarea_does_not_keep_the_generic_inset_frame():
    input_css = read("static/css/components-input.css")
    chat_rule = input_css.split(".chat-input-container .input-field {", 1)[1].split("}", 1)[0]

    assert "border: none;" in chat_rule
    assert "box-shadow: none;" in chat_rule


def test_chat_textarea_grows_until_responsive_height_limit():
    template = read("templates/index.html")
    input_css = read("static/css/components-input.css")
    app_init = read("static/js/app-init.js")

    assert 'id="questionInput" rows="1"' in template
    assert "resize: none;" in input_css
    assert "overflow-y: hidden;" in input_css
    assert "max-height: 220px;" in input_css
    assert "@media (max-width: 1099px)" in input_css
    assert "max-height: 180px;" in input_css
    assert "function resizeQuestionInput()" in app_init
    assert 'questionInput?.addEventListener("input", resizeQuestionInput)' in app_init
    assert 'contentHeight > maxHeight + 1 ? "auto" : "hidden"' in app_init


def test_hero_greeting_requires_agent_mode_and_available_space():
    template = read("templates/index.html")
    input_css = read("static/css/components-input.css")

    assert "What should the models cross-check?" in template
    assert "body.is-hero.agent-mode-enabled .hero-greeting" in input_css
    assert "@media (min-width: 680px) and (min-height: 620px)" in input_css
    assert "body.is-hero .hero-greeting" not in input_css
    assert "body.is-hero:not(.agent-mode-enabled) .hero-greeting" not in input_css


def test_settings_are_grouped_without_changing_control_ids():
    template = read("templates/index.html")

    for category in ("Experience", "Connections", "Model behavior", "Account"):
        assert f">{category}<" in template

    for control_id in (
        "mobileModeToggle",
        "agentModeSwitch",
        "autoConsensusToggle",
        "useOwnKeysSwitch",
        "apiSettingsArea",
        "systemPromptInput",
        "accountSettingsSection",
    ):
        assert f'id="{control_id}"' in template

    assert template.count('class="settings-category"') >= 3
    assert template.count('class="settings-group"') >= 4


def test_logout_clears_the_loaded_run_and_aborts_active_streams():
    firebase = read("static/firebase.js")
    app_init = read("static/js/app-init.js")

    assert "function resetLoadedRunAfterLogout()" in firebase
    assert "window.cancelCurrentQuery?.();" in firebase
    assert "window.cancelCurrentConsensus?.();" in firebase
    assert "window.clearResponseBoxes?.({ silent: true });" in firebase
    assert "window.App?.watch?.resetAfterLogout?.();" in firebase
    assert "await signOut(auth);" in firebase
    assert "function clearAuthenticatedUiState()" in firebase
    assert '["freeUsageDisplay", "deepUsageDisplay", "watchUsageDisplay", "countdownDisplay"]' in firebase
    assert "window.App?.sidebarQuota?.setOpen?.(false);" in firebase
    assert "window.App?.sharedModal?.close?.();" in firebase
    assert "function clearLocalProviderKeys()" in firebase
    assert '["openaiKey", "mistralKey", "anthropicKey", "geminiKey", "deepseekKey", "grokKey"]' in firebase
    assert "if (previousAuthUid) clearLocalProviderKeys();" in firebase
    assert "function isCurrentAuthenticatedUser(uid, generation)" in firebase
    assert "setBookmarksAccess(false);" in firebase
    assert 'searchHead?.classList.remove("is-searching");' in firebase
    assert 'document.body.classList.add("is-hero")' in firebase
    assert "window.clearResponseBoxes = function (options = {})" in app_init
    assert "window.consensusCitationMeta = null" in app_init


def test_watch_change_surfaces_use_tint_without_a_left_rail():
    shell = read("static/css/shell.css")
    drift_rule = shell.split(".watch-dash-stat.is-drift,", 1)[1].split("}", 1)[0]
    limit_rule = shell.split(".watch-dash-heading-row .watch-limit-summary {", 1)[1].split("}", 1)[0]

    assert "border-left" not in drift_rule
    assert "border-radius: var(--radius-md)" in drift_rule
    assert "border-radius: var(--radius-sm)" in limit_rule
    assert "padding: 8px 10px" in limit_rule


def test_watch_requests_cannot_repopulate_account_state_after_logout():
    watch = read("static/js/watch.js")

    assert "let watchSessionEpoch = 0;" in watch
    assert "requestEpoch !== watchSessionEpoch" in watch
    assert "window.auth?.currentUser?.uid !== userUid" in watch
    assert "watchSessionEpoch += 1;" in watch
    assert "telegramState = null;" in watch
