"""Playwright-Smoke-Suite: automatisiert die risikoreichsten Punkte aus
docs/smoke-checklist.md gegen den Mock-Server (MOCK_LLM/MOCK_AUTH).

Bewusst NICHT abgedeckt (Stand der ersten Iteration): Resolve-Runde,
Share-Dialog-CRUD, Attachments, Follow-up, Bookmarks, Agent-Mode-Timer,
Demo-Flow und das vollständige Mobile-Layout - siehe tests/e2e/README.md.
"""

from playwright.sync_api import expect

QUESTION = "When was the Eiffel Tower completed?"

# Provider -> (Checkbox-ID, Response-Box-ID, eindeutiger Fixture-Marker aus
# app/services/llm/mock_llm.py).
PROVIDERS = {
    "OpenAI": ("selectOpenAI", "openaiResponse", "OpenAI mock answer"),
    "Mistral": ("selectMistral", "mistralResponse", "Mistral mock answer"),
    "Anthropic": ("selectClaude", "claudeResponse", "Claude mock answer"),
    "Gemini": ("selectGemini", "geminiResponse", "Gemini mock answer"),
    "DeepSeek": ("selectDeepSeek", "deepseekResponse", "DeepSeek mock answer"),
    "Grok": ("selectGrok", "grokResponse", "Grok mock answer"),
}

ALL_CHECKBOX_IDS = [check_id for check_id, _, _ in PROVIDERS.values()]


def _select_all_models(page):
    """Alle sechs Modelle einschalten (klickt die Checkbox, damit die
    model-picker-Handler inkl. .excluded-Sync und Persistenz laufen)."""
    page.evaluate(
        """(ids) => {
          for (const id of ids) {
            const cb = document.getElementById(id);
            if (cb && !cb.checked) cb.click();
          }
        }""",
        ALL_CHECKBOX_IDS,
    )


def _send_question(page, question=QUESTION):
    # Die Suite prüft Frontend/SSE, nicht die Quotenlogik. Dummy-Eigenkeys
    # halten die Runs unabhängig vom aktuell aus Firestore geladenen Free-Limit;
    # MOCK_LLM verhindert weiterhin jeden echten Provider-Aufruf.
    page.evaluate(
        """() => {
          const keys = [
            'openaiKey', 'mistralKey', 'anthropicKey',
            'geminiKey', 'deepseekKey', 'grokKey',
          ];
          for (const key of keys) localStorage.setItem(key, 'e2e-dummy-key');
          const ownKeys = document.getElementById('useOwnKeysSwitch');
          if (ownKeys) ownKeys.checked = true;
        }"""
    )
    _select_all_models(page)
    page.fill("#questionInput", question)
    page.click("#sendButton")


def _wait_for_all_final_answers(page):
    for _, (_, response_id, marker) in PROVIDERS.items():
        expect(page.locator(f"#{response_id}")).to_contain_text(marker, timeout=15000)
    # Letztes Fixture-Fragment jeder Antwort abwarten (Stream beendet).
    expect(page.locator("#openaiResponse")).to_contain_text("330 metres tall", timeout=15000)


def test_app_loads_without_console_errors(app_page, get_console_errors):
    """Smoke-Checkliste 'Browser-Konsole' + Script-Ladereihenfolge (§8):
    die zentralen window.*-Vertraege muessen nach dem Laden existieren."""
    missing = app_page.evaluate(
        """() => [
          "sendQuestion", "getConsensus", "canGenerateConsensus",
          "updateConsensusButtonAvailability", "openShareDialog",
          "injectMarkdown", "updateUserTierUI",
        ].filter((name) => typeof window[name] !== "function")"""
    )
    assert missing == [], f"Fehlende window-Funktionen (Ladereihenfolge?): {missing}"
    assert app_page.evaluate("() => typeof window.App.consensusLifecycle") == "object"
    expect(app_page).to_have_title("Compare AI Answers | consens.io")

    # Kurze Nachlaufzeit fuer asynchrone Init-Fehler (Tooltips, Usage-Fetch).
    app_page.wait_for_timeout(1500)
    errors = get_console_errors()
    assert errors == [], f"Konsolen-Fehler beim Laden: {errors}"


def test_latex_is_typeset_after_markdown_rendering(app_page):
    result = app_page.evaluate(
        r"""() => {
          const host = document.createElement("div");
          document.body.appendChild(host);
          window.injectMarkdown(
            host,
            String.raw`\[ \det(DF)\equiv -2 \] and \(F:\mathbb C^3\to\mathbb C^3\)`
          );
          const result = {
            displays: host.querySelectorAll(".katex-display").length,
            expressions: host.querySelectorAll(".katex").length,
            errors: host.querySelectorAll(".katex-error").length,
            fontFamily: getComputedStyle(host.querySelector(".katex")).fontFamily,
          };
          host.remove();
          return result;
        }"""
    )
    assert result["displays"] == 1
    assert result["expressions"] == 2
    assert result["errors"] == 0
    assert "KaTeX" in result["fontFamily"]


def test_usage_display_is_stable_and_keeps_value_layout(app_page):
    """Parallele Antworten ohne Usage-Metadaten duerfen weder auf 0
    zurueckfallen noch den rechtsbuendigen, fetten Wert-Wrapper entfernen."""
    metrics = app_page.evaluate(
        """() => {
          window.App.renderUsageDisplay({
            remaining: 2,
            deepRemaining: 0,
            totalLimit: 3,
            deepLimit: 0,
          });
          window.App.renderUsageDisplay({});

          const line = document.getElementById('freeUsageDisplay');
          const value = line.querySelector('strong');
          const lineRect = line.getBoundingClientRect();
          const valueRect = value.getBoundingClientRect();
          return {
            text: line.textContent,
            deepText: document.getElementById('deepUsageDisplay').textContent,
            valueTag: value.tagName,
            valueWeight: Number.parseInt(getComputedStyle(value).fontWeight, 10),
            rightGap: Math.abs(lineRect.right - valueRect.right),
          };
        }"""
    )

    assert metrics["text"] == "Runs: 2 / 3"
    assert metrics["deepText"] == "Deep Think: 0 / 0"
    assert metrics["valueTag"] == "STRONG"
    assert metrics["valueWeight"] >= 600
    assert metrics["rightGap"] < 1


def test_empty_app_and_consensus_picker_do_not_scroll_unnecessarily(app_page):
    page_metrics = app_page.evaluate(
        """() => ({
          scrollHeight: document.documentElement.scrollHeight,
          clientHeight: document.documentElement.clientHeight,
        })"""
    )
    assert page_metrics["scrollHeight"] <= page_metrics["clientHeight"]

    app_page.locator(".consensus-model .model-picker-display").click()
    menu_metrics = app_page.locator(".consensus-model .model-picker-menu").evaluate(
        """element => ({
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
          overflowX: getComputedStyle(element).overflowX,
        })"""
    )
    assert menu_metrics["scrollWidth"] <= menu_metrics["clientWidth"]
    assert menu_metrics["overflowX"] == "hidden"
    app_page.locator(".consensus-model .model-picker-display").click()


def test_send_question_streams_all_models(app_page):
    """Kern-Flow: Frage senden -> alle Modelle streamen (Zwischenzustand
    sichtbar) und rendern die finale Mock-Antwort."""
    _send_question(app_page)
    expect(app_page).to_have_title(f"{QUESTION} | consens.io")

    # Streaming-Zwischenzustand: Anfang der Antwort sichtbar, Ende noch nicht
    # (MOCK_LLM_DELAY_MS drosselt die Deltas auf ~400ms pro Antwort).
    app_page.wait_for_function(
        """() => {
          const el = document.getElementById("openaiResponse");
          const text = el ? el.innerText : "";
          return text.includes("OpenAI mock answer") && !text.includes("330 metres tall");
        }""",
        timeout=15000,
    )

    _wait_for_all_final_answers(app_page)


def test_consensus_renders_differences_and_agreement_score(app_page, get_console_errors):
    """Hoechstes Risiko laut Smoke-Checkliste: Auto-Consensus (per Default an)
    triggert nach Abschluss aller Antworten und rendert Consensus-Text,
    Claim-Badges, Widerspruchs-Karte und Agreement-Score. Einen manuellen
    Consensus-Button gibt es im aktuellen UI nicht mehr."""
    app_page.set_viewport_size({"width": 390, "height": 844})
    app_page.evaluate("() => window.setAgentMode(false, { persist: true })")
    _send_question(app_page)

    # Regulärer Modus: determinate Antwortphase -> indeterminate Synthese.
    # Die rahmenlose Zeile bleibt mobil kompakt und clippt ihren Text nicht.
    pipeline = app_page.locator("#consensusPipeline")
    expect(pipeline).to_be_visible(timeout=10000)
    expect(pipeline).to_have_attribute("data-stage", "answers")
    metrics = pipeline.evaluate(
        """(el) => {
          const steps = el.querySelector('.consensus-pipeline-steps');
          return {
            height: el.getBoundingClientRect().height,
            clipped: steps.scrollWidth > steps.clientWidth,
          };
        }"""
    )
    assert metrics["height"] <= 34
    assert metrics["clipped"] is False

    _wait_for_all_final_answers(app_page)
    expect(pipeline).to_have_attribute("data-stage", "consensus", timeout=20000)

    expect(app_page.locator("#consensusResponse")).to_contain_text("Mock consensus", timeout=30000)

    verdict = app_page.locator("#consensusVerdict")
    expect(verdict).to_be_visible(timeout=15000)
    expect(verdict).to_contain_text("/100")

    expect(app_page.locator(".claim-badge").first).to_be_visible(timeout=15000)
    expect(app_page.locator(".diff-card.is-contradiction").first).to_be_visible(timeout=15000)

    consensus_box = app_page.locator("#consensusOutput").bounding_box()
    first_answer_box = app_page.locator("#openaiResponse").bounding_box()
    assert consensus_box is not None
    assert first_answer_box is not None
    assert consensus_box["y"] < first_answer_box["y"]
    expect(pipeline).to_be_hidden(timeout=10000)

    errors = get_console_errors()
    assert errors == [], f"Konsolen-Fehler im Consensus-Flow: {errors}"


def test_agent_mode_can_reveal_hidden_model_answers_on_mobile(app_page):
    """The compact mobile Agent Mode panel explains and toggles hidden answers."""
    app_page.set_viewport_size({"width": 390, "height": 844})
    app_page.evaluate(
        """() => {
          localStorage.setItem("agentModePanelCollapsed", "true");
          window.setAgentMode(true, { persist: true });
        }"""
    )
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)

    toggle = app_page.locator("#agentModeAnswersToggle")
    expect(toggle).to_be_visible(timeout=15000)
    expect(toggle).to_have_text("Show model answers")
    expect(app_page.locator("#openaiResponse")).to_be_hidden()

    toggle.click()
    expect(app_page.locator("body.agent-mode-enabled.agent-mode-show-answers")).to_have_count(1)
    expect(toggle).to_have_text("Hide model answers")
    expect(app_page.locator("#openaiResponse")).to_be_visible()

    toggle.click()
    expect(toggle).to_have_text("Show model answers")
    expect(app_page.locator("#openaiResponse")).to_be_hidden()


def test_watch_dialog_uses_safe_defaults_keeps_telegram_visible_and_reveals_condition(app_page):
    """Watch-Erstellung startet kompakt mit sicheren Defaults, hält Telegram
    sichtbar und blendet erweiterte Felder nur bei Bedarf ein."""
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    expect(app_page.locator("#consensusResponse")).to_contain_text("Mock consensus", timeout=30000)

    # Dieser Test prüft nur Client-Validierung/Layout und braucht keinen echten
    # Firestore-persistierten pending_result.
    app_page.evaluate("() => { window.lastShareResultId = 'e2e-watch-validation'; }")
    app_page.set_viewport_size({"width": 390, "height": 844})
    app_page.click("#consensusWatchButton")
    app_page.locator("#shareModal").click(position={"x": 2, "y": 2})
    expect(app_page.locator("#watchConfirmBtn")).to_be_visible()
    expect(app_page.locator(".watch-delivery-field")).to_be_visible()
    expect(app_page.locator("#watchTelegramEnabled")).to_be_visible()
    expect(app_page.locator("#watchTelegramConnect")).to_be_visible()
    expect(app_page.locator("#watchAdvancedSettings")).not_to_have_attribute("open", "")
    expect(app_page.locator("#watchVisibility")).to_be_hidden()
    dialog_box = app_page.locator("#shareModal .share-modal-content").bounding_box()
    assert dialog_box is not None
    assert dialog_box["y"] >= 0
    assert dialog_box["y"] + dialog_box["height"] <= 844.5
    assert app_page.locator("#watchVisibility").input_value() == "private"
    expect(app_page.locator("#watchVisibilitySummary")).to_have_text("Private page")
    app_page.click("#watchAdvancedSettings > summary")
    expect(app_page.locator("#watchVisibility")).to_be_visible()
    expect(app_page.locator("#watchRunTime")).to_have_value("09:00")
    expect(app_page.locator("#watchWeekdayWrap")).to_be_visible()
    tomorrow_weekday = app_page.evaluate("""() => {
      const days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      return days[tomorrow.getDay()];
    }""")
    expect(app_page.locator("#watchWeekday")).to_have_value(tomorrow_weekday)
    app_page.select_option("#watchWeekday", "friday")
    expect(app_page.locator("#watchWeekday")).to_have_value("friday")
    app_page.select_option("#watchInterval", "monthly")
    expect(app_page.locator("#watchWeekdayWrap")).to_be_hidden()
    app_page.select_option("#watchInterval", "weekly")
    expect(app_page.locator("#watchWeekdayWrap")).to_be_visible()
    expect(app_page.locator("#watchWeekday")).to_have_value("friday")
    assert app_page.locator("#watchTimezoneLabel").text_content()
    expect(app_page.locator("#watchConditionWrap")).to_be_hidden()

    app_page.select_option("#watchVisibility", "public")
    expect(app_page.locator("#watchVisibilitySummary")).to_have_text("Public page")

    app_page.fill("#watchRunTime", "")
    app_page.click("#watchConfirmBtn")
    expect(app_page.locator("#watchRunTimeError")).to_have_text(
        "Choose a run time for the automatic check."
    )
    expect(app_page.locator("#watchRunTime")).to_have_attribute("aria-invalid", "true")
    app_page.fill("#watchRunTime", "09:00")

    app_page.select_option("#watchEmailMode", "condition")
    expect(app_page.locator("#watchConditionWrap")).to_be_visible()
    expect(app_page.locator("#watchCondition")).to_have_attribute("maxlength", "500")
    app_page.click("#watchConfirmBtn")
    expect(app_page.locator("#watchConditionError")).to_have_text(
        "Enter the condition you want to monitor."
    )
    expect(app_page.locator("#watchCondition")).to_have_attribute("aria-invalid", "true")


def test_query_first_watch_guides_question_then_configuration(app_page):
    """Das Watch-Dashboard startet einen Query-first-Flow, ohne vorherigen
    Consensus, und bewahrt die Frage beim Zuruecknavigieren."""
    app_page.route(
        "**/api/my/watches",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","watches":[],"limits":{"plan":"free","active_count":0,"active_limit":1,"remaining":1,"paused_count":0,"daily_available":false}}',
        ),
    )
    app_page.route(
        "**/api/my/watch-brief",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","brief":{}}',
        ),
    )
    app_page.route(
        "**/api/my/telegram",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","telegram":{"configured":false,"connected":false}}',
        ),
    )

    app_page.click("#viewSwitchWatches")
    expect(app_page.locator("#watchDashCreate")).to_be_visible()
    expect(app_page.locator("#watchDashLimit")).to_contain_text("Free plan")
    expect(app_page.locator("#watchDashLimit")).to_contain_text("0 of 1 active")
    expect(app_page.locator("#watchDashLimit")).to_contain_text("Paused Watches do not count")

    empty_box = app_page.locator(".watch-dash-empty").bounding_box()
    empty_copy_box = app_page.locator(".watch-dash-empty > p").bounding_box()
    empty_cta_box = app_page.locator(".watch-empty-actions > .share-primary-btn").bounding_box()
    assert empty_box is not None and empty_copy_box is not None and empty_cta_box is not None
    assert empty_cta_box["y"] >= empty_copy_box["y"] + empty_copy_box["height"] + 8
    assert abs(
        (empty_cta_box["x"] + empty_cta_box["width"] / 2)
        - (empty_box["x"] + empty_box["width"] / 2)
    ) < 2
    example_chips = app_page.locator(".watch-example-chip")
    assert example_chips.count() == 3
    first_example_box = example_chips.first.bounding_box()
    assert first_example_box is not None and first_example_box["height"] <= 27

    app_page.set_viewport_size({"width": 390, "height": 844})
    title_box = app_page.locator("#watchDashTitle").bounding_box()
    header_cta_box = app_page.locator("#watchDashCreate").bounding_box()
    mobile_subtitle_box = app_page.locator(".watch-dash-subtitle").bounding_box()
    mobile_empty_box = app_page.locator(".watch-dash-empty").bounding_box()
    assert title_box is not None and header_cta_box is not None
    assert mobile_subtitle_box is not None and mobile_empty_box is not None
    assert header_cta_box["y"] > title_box["y"]
    assert header_cta_box["x"] + header_cta_box["width"] <= 378.5
    assert mobile_empty_box["y"] >= mobile_subtitle_box["y"] + mobile_subtitle_box["height"] + 20

    app_page.click("#watchDashCreate")
    expect(app_page.locator("#watchQuestion")).to_be_visible()
    expect(app_page.locator("#watchDialogLimit")).to_contain_text("1 slot available")
    expect(app_page.locator("#watchQuestionNext")).to_have_text("Continue to schedule")
    expect(app_page.locator("#shareModalBody")).to_contain_text(
        "No model run starts until the Watch reaches its scheduled check."
    )

    app_page.fill("#watchQuestion", "Short")
    app_page.click("#watchQuestionNext")
    expect(app_page.locator("#watchQuestionError")).to_be_visible()

    question = "Has the EU guidance for general-purpose AI models changed?"
    app_page.fill("#watchQuestion", question)
    app_page.click("#watchQuestionNext")
    expect(app_page.locator(".watch-question-preview strong")).to_have_text(question)
    expect(app_page.locator(".watch-setup-summary")).to_be_visible()
    expect(app_page.locator(".watch-delivery-field")).to_be_visible()
    expect(app_page.locator("#watchTelegramEnabled")).to_be_visible()
    expect(app_page.locator("#watchVisibility")).to_be_hidden()
    expect(app_page.locator("#watchVisibility")).to_have_value("private")
    expect(app_page.locator("#watchCancelBtn")).to_have_text("Back")

    app_page.click("#watchCancelBtn")
    expect(app_page.locator("#watchQuestion")).to_have_value(question)


def test_watch_limit_is_explained_before_creation(app_page):
    app_page.route(
        "**/api/my/watches",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","watches":[],"limits":{"plan":"free","active_count":1,"active_limit":1,"remaining":0,"paused_count":0,"daily_available":false}}',
        ),
    )
    app_page.route(
        "**/api/my/watch-brief",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","brief":{}}',
        ),
    )
    app_page.route(
        "**/api/my/telegram",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body='{"status":"success","telegram":{"configured":false,"connected":false}}',
        ),
    )

    app_page.click("#viewSwitchWatches")
    expect(app_page.locator("#watchDashLimit")).to_contain_text("1 of 1 active")
    expect(app_page.locator("#watchDashLimit")).to_contain_text("Limit reached")

    app_page.click("#watchDashCreate")
    expect(app_page.locator("#watchDialogLimit")).to_contain_text(
        "Pro beta offers a larger Watch allowance and more frequent checks"
    )
    expect(app_page.locator("#watchQuestionNext")).to_be_disabled()
    expect(app_page.locator("#watchQuestionNext")).to_have_text("Watch limit reached")


def test_exclude_model_toggles_excluded_class(app_page):
    """Modell ausschliessen: Checkbox aus -> Response-Box bekommt .excluded,
    wieder an -> Klasse verschwindet."""
    _select_all_models(app_page)

    app_page.evaluate("() => document.getElementById('selectGrok').click()")
    app_page.wait_for_function(
        "() => document.getElementById('grokResponse').classList.contains('excluded')",
        timeout=5000,
    )

    app_page.evaluate("() => document.getElementById('selectGrok').click()")
    app_page.wait_for_function(
        "() => !document.getElementById('grokResponse').classList.contains('excluded')",
        timeout=5000,
    )


def test_theme_toggle(app_page):
    """Dark/Light-Toggle: body.dark-mode kippt und wird in localStorage
    persistiert."""
    initially_dark = app_page.evaluate("() => document.body.classList.contains('dark-mode')")

    # Ohne Topbar lebt der Theme-Toggle in den Settings; das Zahnrad sitzt in
    # der Sidebar-Fußzeile und braucht die geöffnete Sidebar. Auf Desktop-
    # Breiten (Push-Modus ab 1100px) ist sie standardmäßig schon offen —
    # nur öffnen, wenn sie tatsächlich zu ist.
    app_page.evaluate(
        """() => {
          const sidebar = document.querySelector(".sidebar");
          if (sidebar && sidebar.classList.contains("collapsed")) {
            document.getElementById("toggleSidebarButton").click();
          }
        }"""
    )
    app_page.click("#editSystemPromptBtn")
    app_page.click("#mobileModeToggle")
    app_page.wait_for_function(
        "(wasDark) => document.body.classList.contains('dark-mode') !== wasDark",
        arg=initially_dark,
        timeout=5000,
    )
    stored = app_page.evaluate("() => localStorage.getItem('theme')")
    assert stored in ("dark", "light")

    app_page.click("#mobileModeToggle")
    app_page.wait_for_function(
        "(wasDark) => document.body.classList.contains('dark-mode') === wasDark",
        arg=initially_dark,
        timeout=5000,
    )
    # Modal wieder schliessen, damit Folgetests keine ueberdeckte Seite sehen.
    app_page.click("#closeSystemPromptModal")
    expect(app_page.locator("#systemPromptModal")).to_be_hidden()


def test_deep_think_temporarily_selects_configured_engine(app_page):
    """Deep Think nutzt die Admin-konfigurierte Engine, ohne die zuvor
    gespeicherte Consensus-Auswahl des Pro-Nutzers dauerhaft zu ersetzen."""
    initial_model = app_page.evaluate(
        """() => {
          window.isUserPro = true;
          window.updatePremiumModelsState(true);
          const select = document.getElementById("consensusModelDropdown");
          const initial = Array.from(select.options).find(option =>
            !option.disabled && option.value !== window.DEEP_THINK_CONSENSUS_MODEL
          ).value;
          select.value = initial;
          select.dispatchEvent(new Event("change", { bubbles: true }));
          return initial;
        }"""
    )
    assert app_page.evaluate("() => localStorage.getItem('pref_select_consensus')") == initial_model

    app_page.evaluate("() => document.getElementById('deepSearchToggle').click()")
    app_page.wait_for_function(
        "() => document.getElementById('consensusModelDropdown').value === window.DEEP_THINK_CONSENSUS_MODEL",
        timeout=5000,
    )
    assert app_page.evaluate("() => localStorage.getItem('pref_select_consensus')") == initial_model

    app_page.evaluate("() => document.getElementById('deepSearchToggle').click()")
    app_page.wait_for_function(
        "(initial) => document.getElementById('consensusModelDropdown').value === initial",
        arg=initial_model,
        timeout=5000,
    )


def test_consensus_presets_apply_full_model_sets_and_gate_thorough(app_page):
    """Fast/Balanced sind vollstaendige Model-Sets; High Quality bleibt fuer Free
    sichtbar, aber oeffnet mit Pro-Badge das Upgrade-Modal."""
    result = app_page.evaluate(
        """() => {
          window.isUserPro = false;
          window.updatePremiumModelsState(false);
          localStorage.setItem("pref_consensus_preset", "balanced");
          window.restoreModelSelections();
          const consensus = document.getElementById("consensusModelDropdown");
          consensus._customModelPicker.displayButton.click();
          const fast = consensus._customModelPicker.menu.querySelector('[data-preset="fast"]');
          fast.click();
          const configured = window.CONSENSUS_PRESETS.find(preset => preset.id === "fast");
          const actual = Object.fromEntries(window.App.modelPrefs.map(pref => [
            pref.provider,
            document.getElementById(pref.selectId).value,
          ]));

          consensus._customModelPicker.displayButton.click();
          const thorough = consensus._customModelPicker.menu.querySelector('[data-preset="thorough"]');
          const understandableLabel = thorough.textContent.includes('High Quality');
          const hasProBadge = !!thorough.querySelector('.model-picker-pro-badge');
          thorough.click();
          return {
            actual,
            expected: configured.models,
            consensus: consensus.value,
            expectedConsensus: configured.consensus_model,
            storedPreset: localStorage.getItem("pref_consensus_preset"),
            understandableLabel,
            hasProBadge,
            proModalDisplay: document.getElementById("proFeatureModal").style.display,
          };
        }"""
    )
    assert result["actual"] == result["expected"]
    assert result["consensus"] == result["expectedConsensus"]
    assert result["storedPreset"] == "fast"
    assert result["understandableLabel"]
    assert result["hasProBadge"]
    assert result["proModalDisplay"] == "block"

    app_page.evaluate(
        """() => {
          document.getElementById("keepFreeBtn").click();
          const pref = window.App.modelPrefs[0];
          const select = document.getElementById(pref.selectId);
          const alternative = Array.from(select.options).find(option =>
            !option.disabled && option.value !== select.value
          );
          if (alternative) {
            select.value = alternative.value;
            select.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }"""
    )
    assert app_page.evaluate("() => localStorage.getItem('pref_consensus_preset')") == "custom"
    app_page.evaluate(
        """() => {
          for (const pref of window.App.modelPrefs) {
            localStorage.removeItem("pref_select_" + pref.key);
          }
          localStorage.removeItem("pref_select_consensus");
          localStorage.removeItem("pref_consensus_preset");
          window.updatePremiumModelsState(false, false);
        }"""
    )


def test_attachment_pauses_deepseek_and_restores_previous_selection(app_page):
    """Echte Anhaenge nehmen DeepSeek temporaer aus dem Fan-out; reine
    Bookmark-Metadaten tun das nicht und die Nutzerwahl bleibt erhalten."""
    result = app_page.evaluate(
        """() => {
          const checkbox = document.getElementById("selectDeepSeek");
          window.App.setModelSelectionState("deepseekResponse", true, {
            persist: false,
            syncCheckbox: true,
            animate: false,
          });
          const persistedBefore = localStorage.getItem("pref_check_DeepSeek");

          window.pendingAttachments = [{
            name: "brief.pdf",
            mime: "application/pdf",
            size: 128,
            data: "JVBERi0xLjcK",
          }];
          window.renderAttachmentChips();
          const whileAttached = {
            checked: checkbox.checked,
            disabled: checkbox.disabled,
            notice: document.getElementById("attachmentProviderNotice")?.textContent || "",
            responseExcluded: document.getElementById("deepseekResponse").classList.contains("excluded"),
          };

          window.clearPendingAttachments();
          window.pendingAttachments = [{
            name: "saved-image.png",
            mime: "image/png",
            size: 64,
            data: null,
            previewOnly: true,
          }];
          window.renderAttachmentChips();
          const withBookmarkPreview = {
            checked: checkbox.checked,
            disabled: checkbox.disabled,
            noticeExists: !!document.getElementById("attachmentProviderNotice"),
          };
          window.clearPendingAttachments();
          return {
            persistedBefore,
            persistedAfter: localStorage.getItem("pref_check_DeepSeek"),
            whileAttached,
            withBookmarkPreview,
            afterRemoval: {
              checked: checkbox.checked,
              disabled: checkbox.disabled,
              noticeExists: !!document.getElementById("attachmentProviderNotice"),
              responseExcluded: document.getElementById("deepseekResponse").classList.contains("excluded"),
            },
          };
        }"""
    )

    assert result["whileAttached"]["checked"] is False
    assert result["whileAttached"]["disabled"] is True
    assert "cannot read attachments" in result["whileAttached"]["notice"]
    assert result["whileAttached"]["responseExcluded"] is True
    assert result["withBookmarkPreview"] == {
        "checked": True,
        "disabled": False,
        "noticeExists": False,
    }
    assert result["afterRemoval"] == {
        "checked": True,
        "disabled": False,
        "noticeExists": False,
        "responseExcluded": False,
    }
    assert result["persistedAfter"] == result["persistedBefore"]


def test_tier_upgrade_applies_pro_defaults_but_keeps_explicit_picker_choice(app_page):
    """Im Custom-Modus aendert Free -> Pro nur nicht explizit gewaehlte Defaults.
    Aktive Presets haben absichtlich Vorrang vor diesen Tier-Defaults."""
    result = app_page.evaluate(
        """() => {
          localStorage.setItem("pref_consensus_preset", "custom");
          const pref = window.App.modelPrefs.find(item =>
            window.FREE_DEFAULT_MODELS[item.provider] !== window.PRO_DEFAULT_MODELS[item.provider]
          );
          if (!pref) return { skipped: true };
          const select = document.getElementById(pref.selectId);
          const key = "pref_select_" + pref.key;
          localStorage.removeItem(key);

          window.updatePremiumModelsState(false, false);
          const freeValue = select.value;
          window.updatePremiumModelsState(true, true);
          const proValue = select.value;

          select.value = freeValue;
          select.dispatchEvent(new Event("change", { bubbles: true }));
          window.updatePremiumModelsState(false, false);
          window.updatePremiumModelsState(true, true);
          return {
            skipped: false,
            freeValue,
            proValue,
            expectedFree: window.FREE_DEFAULT_MODELS[pref.provider],
            expectedPro: window.PRO_DEFAULT_MODELS[pref.provider],
            explicitValue: select.value,
            storedValue: localStorage.getItem(key),
          };
        }"""
    )
    assert not result["skipped"]
    assert result["freeValue"] == result["expectedFree"]
    assert result["proValue"] == result["expectedPro"]
    assert result["explicitValue"] == result["freeValue"]
    assert result["storedValue"] == result["freeValue"]


def test_model_selection_persists_across_reload(app_page):
    """Model-Picker-Persistenz: abgewaehltes Modell bleibt nach Reload
    abgewaehlt (localStorage pref_check_*)."""
    _select_all_models(app_page)

    app_page.evaluate("() => document.getElementById('selectGrok').click()")
    app_page.wait_for_function(
        "() => localStorage.getItem('pref_check_Grok') === 'false'",
        timeout=5000,
    )

    app_page.reload(wait_until="domcontentloaded")
    app_page.wait_for_function(
        "() => window.App && typeof window.sendQuestion === 'function'",
        timeout=30000,
    )
    app_page.wait_for_function(
        """() => {
          const cb = document.getElementById("selectGrok");
          const box = document.getElementById("grokResponse");
          return cb && !cb.checked && box && box.classList.contains("excluded");
        }""",
        timeout=10000,
    )
