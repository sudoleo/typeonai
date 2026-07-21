from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_sidebar_navigation_is_self_contained_and_guest_login_is_top_only():
    template = read("templates/index.html")
    app_init = read("static/js/app-init.js")

    assert 'id="sidebarRail"' not in template
    assert "sidebar-rail-btn" not in template
    assert "sidebar-rail-btn" not in app_init
    assert template.count('class="sidebar-heading-label"') == 3
    for label in ("Models", "Leaderboard", "Bookmarks"):
        assert f'<span class="sidebar-heading-label">' in template
        assert f"<span>{label}</span>" in template
    assert 'id="authTopActions"' in template
    assert 'id="loginContainer" class="login-text" hidden></div>' in template


def test_mobile_brand_and_desktop_input_centering_contract():
    layout = read("static/css/layout.css")

    assert "calc(100vw - 520px)" in layout
    assert "translateX(130px)" not in layout
    assert "@media (max-width: 1099px)" in layout
    assert ".brand-float," in layout
    assert "body:has(.sidebar.collapsed) .brand-float" in layout


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
    assert 'document.body.classList.add("is-hero")' in firebase
    assert "window.clearResponseBoxes = function (options = {})" in app_init
    assert "window.consensusCitationMeta = null" in app_init
