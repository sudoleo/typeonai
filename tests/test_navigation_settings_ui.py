from pathlib import Path

from tests.frontend_order import position


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
    # Die Modellwahl kennt keinen Follow-up-Choice-State mehr, der sie sperren
    # koennte: eine Folgefrage ist der Default, kein Zwischenschritt.
    assert "isAwaitingChoice" not in app_init
    assert "before changing models" not in app_init
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
    assert "score: 52" in demo
    # Zwei Beschriftungen, eine sichtbar: in der Knopfzeile des Composers ist
    # auf dem Handy kein Platz fuer den ganzen Satz, sonst faellt der
    # Senden-Knopf in eine zweite Zeile.
    assert '<span class="demo-chip-label demo-chip-label-full">Watch demo</span>' in demo
    assert '<span class="demo-chip-label demo-chip-label-short">Demo</span>' in demo
    assert 'btn.setAttribute("aria-label", "Start interactive demo")' in demo
    assert "inputActions.prepend(btn)" in demo
    demo_rule = misc_css.split(".demo-chip {", 1)[1].split("}", 1)[0]
    assert "position: static" in demo_rule
    narrow_rule = misc_css.split("@media (max-width: 640px) {", 1)[1]
    assert ".demo-chip-label-full" in narrow_rule
    assert ".demo-chip-label-short" in narrow_rule
    # Die Demo zeigt beide Schweregrade: ein kritischer Widerspruch traegt die
    # kraeftige Markierung, ein kleiner die feine. Genau daran haengt, dass die
    # Demo das Produkt zeigt und nicht nur eine Randnotiz.
    assert 'severity: "major"' in demo
    assert 'severity: "minor"' in demo
    assert 'type: "emphasis"' in demo
    # Das Kreuz ist ein echtes Glyph (wie bei den Modal-Close-Buttons), nicht
    # mehr aus zwei gedrehten Pseudo-Elementen gebaut.
    assert (
        'class="watch-feature-nudge-close" aria-label="Dismiss new feature tip">&#10005;</button>'
        in watch
    )
    assert "rows.slice(0, 8)" in leaderboard
    assert "row.available_since" in leaderboard
    assert 'data-model-pulse-period="since-2026-08-31"' in pulse_page
    assert "Kimi and GLM joined consens.io on 31 August 2026" in pulse_page
    assert 'id="modelLeaderboard"' in pulse_page
    assert "not a popularity vote" in pulse_page.lower()
    assert 'href="/benchmark"' in pulse_page


def test_kimi_and_glm_are_present_across_public_provider_surfaces():
    landing = read("templates/landing.html")
    about = read("templates/about.html")
    comparison = read("templates/ai-model-comparison.html")
    engine = read("templates/consensus-engine.html")
    app = read("templates/index.html")
    share = read("templates/share.html")

    for page in (landing, about, comparison):
        assert "kimi.svg" in page
        assert "zai.svg" in page
        assert "Kimi" in page
        assert "GLM" in page
    assert "Kimi, and GLM" in engine
    assert "Moonshot AI/Kimi" in app
    assert "Z.ai/GLM" in app
    assert "'Kimi', 'GLM'" in share


def test_consensus_run_requires_two_selected_models_before_starting():
    app_init = read("static/js/app-init.js")
    query_send = read("static/js/query-send.js")
    model_picker = read("static/js/model-picker.js")

    assert "const hasMinimumModels = selectedModelCount >= 2" in app_init
    assert "sendButton.disabled = !canStartRun" in app_init
    assert "if (selectedCount < 2)" in query_send
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
    # Der sichtbare Direktvergleich ist ausgenommen: dort steht der Composer
    # oben, die Begruessung schwebte darueber hinaus in die Kopfzeile.
    assert (
        "body.is-hero.agent-mode-enabled:not(.direct-comparison-active) .hero-greeting"
        in input_css
    )
    assert "@media (min-width: 680px) and (min-height: 620px)" in input_css
    assert "body.is-hero .hero-greeting" not in input_css
    assert "body.is-hero:not(.agent-mode-enabled) .hero-greeting" not in input_css


def test_settings_are_grouped_without_changing_control_ids():
    template = read("templates/index.html")

    for category in (
        "Memory",
        "Model behavior",
        "Runs",
        "Display",
        "Connections",
        "Account",
    ):
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

    assert template.count('class="settings-group"') >= 4


def test_every_settings_category_is_a_tab_panel_with_a_nav_item():
    """Sechs Kategorien untereinander waren beim Oeffnen eine Wand. Genau ein
    Panel ist sichtbar, die Leiste links ist das Inhaltsverzeichnis."""
    template = read("templates/index.html")

    panels = (
        "memorySettingsSection",
        "behaviorSettingsSection",
        "runsSettingsSection",
        "displaySettingsSection",
        "connectionsSettingsSection",
        "accountSettingsSection",
    )
    for panel_id in panels:
        assert f'id="{panel_id}"' in template
        assert f'data-settings-tab="{panel_id}"' in template
        assert f'aria-controls="{panel_id}"' in template

    assert template.count('role="tabpanel"') >= len(panels)
    assert 'role="tablist"' in template

    # Der Account-Reiter startet versteckt; firebase.js gibt ihn frei. Das
    # Panel selbst traegt kein u-display-none mehr — sonst kaempften der
    # Tab-Controller und der Auth-Code um dieselbe Sichtbarkeit.
    account = template.index('id="settingsTabAccount"')
    assert "hidden>Account<" in template[account:account + 400]
    assert 'id="accountSettingsSection" class="settings-category u-display-none"' not in template


def test_settings_tabs_read_as_navigation_not_as_buttons():
    """Die Reiter muessen in der `button:not(...)`-Kette stehen.

    Sonst gewinnt der globale Button-Stil und die sechs Kategorien rendern als
    sechs grosse gefuellte Aktionsknoepfe — eine Navigation, die aussieht wie
    ein Formular. Dieselbe Falle wie beim Watch-Nudge-Schliessknopf.
    """
    input_css = read("static/css/components-input.css")
    modals_css = read("static/css/components-modals.css")

    assert ":not(.settings-nav-item)" in input_css
    assert ":not(.settings-inline-btn)" in input_css

    nav_item = modals_css[modals_css.index(".settings-nav-item {"):]
    nav_item = nav_item[:nav_item.index(".settings-body {")]
    assert "background: transparent;" in nav_item
    assert "border: 0;" in nav_item
    assert "box-shadow: none;" in nav_item

    # Die Tints werden aus der Textfarbe gemischt: im Dark Mode ist `--raise`
    # exakt der Modalhintergrund, ein Hover darauf waere unsichtbar.
    assert "color-mix(in srgb, var(--text-color) 6%, transparent)" in nav_item
    assert "color-mix(in srgb, var(--text-color) 11%, transparent)" in nav_item
    assert "var(--raise)" not in nav_item


def test_settings_visibility_is_owned_by_the_tab_controller():
    firebase = read("static/firebase.js")
    app_ui = read("static/app-ui.js")

    # Ein inline gesetztes display haette den Controller uebersteuert.
    assert 'getElementById("accountSettingsSection")' not in firebase
    assert 'setTabAvailable?.("accountSettingsSection", true)' in firebase
    assert 'setTabAvailable?.("accountSettingsSection", false)' in firebase

    assert "window.App.settingsTabs = settingsTabs;" in app_ui
    # Beim Oeffnen immer der erste Reiter: Einstellungen findet man ueber einen
    # festen Ort, nicht ueber den zuletzt benutzten.
    assert "settingsTabs.reset();" in app_ui


def test_memory_is_the_first_settings_category():
    """Die Reihenfolge ist die Aussage: was die Modelle ueber dich wissen zuerst,
    Konto zuletzt. Wandert das Gedaechtnis nach unten, ist es wieder eine
    Nebeneinstellung statt der erklaerten Hauptfunktion."""
    template = read("templates/index.html")

    # Ueber die Heading-IDs, nicht ueber die Beschriftung: ">Runs<" steht auch
    # im Kontingent-Panel der Sidebar, lange vor den Einstellungen.
    positions = {
        heading: template.index(f'id="{heading}"')
        for heading in (
            "settingsMemoryTitle",
            "settingsBehaviorTitle",
            "settingsRunsTitle",
            "settingsDisplayTitle",
            "settingsConnectionsTitle",
            "settingsAccountTitle",
        )
    }
    assert list(positions) == sorted(positions, key=positions.get)

    body = template.index('class="settings-body"')
    assert template.index('id="memorySettingsSection"') > body

    for control_id in (
        "memoryEnabledSwitch",
        "memoryRoleInput",
        "memoryFocusInput",
        "memoryStyleInput",
        "memoryConstraintsInput",
        "memoryNotesInput",
        "saveMemoryBtn",
        "clearMemoryBtn",
        "memoryStatus",
    ):
        assert f'id="{control_id}"' in template

    # Die Erklaerung gehoert in die UI, nicht nur in die Doku: ohne sie ist ein
    # unsichtbarer Vorspann vor jeder Frage genau der Vertrauensbruch, den das
    # Produkt sonst vermeidet. Sie steht aber ZUGEKLAPPT unter dem Panel --
    # sichtbar war sie eine Textwand vor vier Eingabefeldern.
    memory_panel = template[
        template.index('id="memorySettingsSection"') : template.index(
            'id="behaviorSettingsSection"'
        )
    ]
    assert '<details class="settings-note">' in memory_panel
    assert "<summary>How it works</summary>" in memory_panel
    assert "never <strong>what</strong> is true" in memory_panel
    assert "Nothing is collected" in memory_panel

    # Vier kompakte About-you-Felder plus die grosse, nutzerkontrollierte Notebox.
    # Automatische Ableitung bleibt ausgeschlossen; die expliziten Add-/Correct-
    # Aktionen werden direkt am Feld erklaert.
    assert "<ul" not in memory_panel
    assert memory_panel.count("<textarea") == 5
    assert 'id="memoryNotesInput" rows="12" maxlength="12000"' in memory_panel
    assert 'data-always-visible="true"' in memory_panel
    assert "explicitly submit a Remember or Correct memory action" in memory_panel
    assert "this note is sent with your question" in memory_panel


def test_the_memory_profile_is_only_fetched_when_the_settings_open():
    """`consensio:auth-state` feuert bei JEDEM Seitenaufruf eines eingeloggten
    Kontos. Wird dort geladen, haengt an jedem Aufruf ein Firestore-Read fuer
    ein Panel, das die meisten Nutzer nie oeffnen -- genau der Read, den der
    Modal-Oeffner bewusst aufschiebt. Das Ereignis verwirft nur den Stand."""
    memory = read("static/js/user-memory.js")

    listener = memory.split('window.addEventListener("consensio:auth-state"', 1)[1]
    listener = listener.split("\n    });", 1)[0]
    # Geladen wird nur bei offenem Fenster oder beim Logout (dann ohne Netz).
    assert "if (settingsModalIsOpen() || !uid) {" in listener
    assert "state.saved = null;" in listener
    # Der einzige Auslöser fuer einen Read im Normalfall.
    assert 'getElementById("editSystemPromptBtn")?.addEventListener("click", () => load())' in memory
    # Und er bleibt einmalig: ein zweites Oeffnen liest den gemerkten Stand.
    assert "if (state.loaded && state.uid === user.uid && !force) {" in memory


def test_memory_selection_has_explicit_add_and_correct_flows():
    memory_edit = read("static/js/memory-edit.js")
    index = read("templates/index.html")

    assert 'data-memory-intent="add"' in memory_edit
    assert 'data-memory-intent="correct"' in memory_edit
    assert "Remember this" in memory_edit
    assert "Correct memory" in memory_edit
    assert "updates one clearly matching saved passage" in memory_edit
    assert "intent: state.intent" in memory_edit
    assert 'state.intent === "add" ? state.selection.text.slice(0, 500) : ""' in memory_edit
    assert "Firma" not in memory_edit
    assert "For example: I live in Hanover." in memory_edit
    # Der Composer-Shortcut ("Save to Memory instead of asking") ist entfallen:
    # Memory-Aktionen brauchen eine markierte Aussage, sonst nichts.
    assert "rememberDraftButton" not in memory_edit
    assert "rememberDraftButton" not in index
    assert "memory-draft-action" not in index
    assert "memory-draft-action" not in read("static/css/components-memory-edit.css")


def test_selecting_answer_text_offers_asking_about_it():
    """ChatGPTs "Ask ChatGPT": der markierte Abschnitt wandert als Zitat in den
    Composer, die naechste Frage geht MIT ihm raus. Zitiert wird nur, was eine
    Antwort gesagt hat -- die eigene Frage zu zitieren waere ein Kreis."""
    memory_edit = read("static/js/memory-edit.js")
    quote = read("static/js/composer-quote.js")
    query = read("static/js/query-send.js")
    index = read("templates/index.html")

    assert 'data-selection-action="ask"' in memory_edit
    assert "Ask about this" in memory_edit
    assert 'selection?.kind === "consensus" || selection?.kind === "model_answer"' in memory_edit
    assert "window.App.quote.set(state.selection.text)" in memory_edit
    # Ohne Konto bleiben die Memory-Aktionen weg, das Zitieren nicht.
    assert "if (!askable && !rememberable) return hideMenu();" in memory_edit

    assert 'id="composerQuote"' in index
    assert position("composer-quote.js") >= 0

    # Das Zitat ist beim Senden Teil der Frage -- genau ein Text fuer Thread,
    # Bookmark, Chat-Kontext und die sechs Modelle. Die getippte Frage steht
    # vorn: Thread-Kopf, Seitentitel und Bookmark-Name zeigen den Anfang
    # dieser Zeichenkette als reinen Text.
    assert "`${typed}\\n\\nQuoted from the previous answer:\\n${passage}`" in quote
    assert "> ${" not in quote
    assert "window.App.quote?.compose?.(draftQuestion) ?? draftQuestion" in query
    # Ein geplatzter Lauf gibt Entwurf UND Zitat unveraendert zurueck.
    assert "metadata: { draftQuestion, quotedContext }" in query
    assert "input.value = context.metadata.draftQuestion" in query
    assert "window.App.quote?.set?.(context.metadata.quotedContext)" in query


def test_logout_clears_the_loaded_run_and_aborts_active_streams():
    firebase = read("static/firebase.js")
    app_init = read("static/js/app-init.js")

    assert "function resetLoadedRunAfterLogout()" in firebase
    assert 'window.App?.runRegistry?.clearAll?.("logout");' in firebase
    assert "window.clearResponseBoxes?.({ silent: true });" in firebase
    assert "window.App?.watch?.resetAfterLogout?.();" in firebase
    assert "await signOut(auth);" in firebase
    logout = firebase.split("async function performLogout()", 1)[1].split(
        "function openLogoutConfirm", 1
    )[0]
    assert logout.index('cancelAll?.("logout")') < logout.index("await signOut(auth)")
    assert "function clearAuthenticatedUiState()" in firebase
    assert '["freeUsageDisplay", "deepUsageDisplay", "watchUsageDisplay", "countdownDisplay"]' in firebase
    assert "window.App?.sidebarQuota?.setOpen?.(false);" in firebase
    assert "window.App?.sharedModal?.close?.();" in firebase
    assert "function clearLegacyProviderKeys()" in firebase
    assert "clearLegacyProviderKeys();" in firebase
    assert "function clearLocalProviderKeys()" in firebase
    for key in (
        "openrouterKey",
        "openaiKey",
        "mistralKey",
        "anthropicKey",
        "geminiKey",
        "deepseekKey",
        "grokKey",
    ):
        assert f'"{key}"' in firebase
    assert "if (previousAuthUid) clearLocalProviderKeys();" in firebase
    assert "function isCurrentAuthenticatedUser(uid, generation)" in firebase
    assert "setBookmarksAccess(false);" in firebase
    assert 'searchHead?.classList.remove("is-searching");' in firebase
    assert 'document.body.classList.add("is-hero")' in firebase
    assert "window.clearResponseBoxes = function (options = {})" in app_init
    assert 'window.App.state.set("consensusCitationMeta", null, "consensus")' in app_init


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
    owner = read("static/js/watch-state.js")

    assert "sessionEpoch: 0" in owner
    assert "requestEpoch !== watchState.sessionEpoch" in watch
    assert "window.auth?.currentUser?.uid !== userUid" in watch
    assert "watchState.resetSession();" in watch
    assert "values.sessionEpoch += 1;" in owner
    assert "values.telegram = null;" in owner
    assert "values.limits = null;" in owner
