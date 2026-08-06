"""Playwright-Smoke-Suite: automatisiert die risikoreichsten Punkte aus
docs/smoke-checklist.md gegen den Mock-Server (MOCK_LLM/MOCK_AUTH).

Bewusst NICHT abgedeckt (Stand der ersten Iteration): Resolve-Runde,
Share-Dialog-CRUD, Attachments, Follow-up, Bookmarks, Agent-Mode-Timer,
Demo-Flow und das vollständige Mobile-Layout - siehe tests/e2e/README.md.
"""

import re

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


def test_sidebar_header_hover_search_and_provider_picker(app_page):
    sidebar = app_page.locator(".sidebar")
    if sidebar.get_attribute("aria-hidden") == "true":
        app_page.locator(".app-nav-float .sidebar-toggle").click()

    metrics = app_page.evaluate(
        """() => {
          const sidebar = document.querySelector('.sidebar');
          const brand = document.querySelector('.sidebar-brand-link').getBoundingClientRect();
          const toggle = document.querySelector('#sidebarToggleInner').getBoundingClientRect();
          const newRun = document.querySelector('#newRunButton').getBoundingClientRect();
          return {
            brandCenter: brand.top + brand.height / 2,
            toggleCenter: toggle.top + toggle.height / 2,
            headerBottom: Math.max(brand.bottom, toggle.bottom),
            newRunTop: newRun.top,
            newRunWidth: newRun.width,
            sidebarInnerWidth: sidebar.clientWidth - 24,
          };
        }"""
    )
    assert abs(metrics["brandCenter"] - metrics["toggleCenter"]) <= 1
    assert metrics["newRunTop"] >= metrics["headerBottom"]
    assert abs(metrics["newRunWidth"] - metrics["sidebarInnerWidth"]) <= 1

    search = app_page.locator(".sidebar-search-box")
    search_trigger = app_page.locator("#bookmarkSearchTrigger")
    expect(search).to_have_css("opacity", "0")
    app_page.locator(".sidebar-bookmarks-head").hover()
    expect(search).to_have_css("opacity", "0")
    expect(search_trigger).to_have_css("opacity", "1")
    trigger_metrics = search_trigger.evaluate(
        """element => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return {
            width: rect.width,
            height: rect.height,
            paddingLeft: style.paddingLeft,
            paddingRight: style.paddingRight,
          };
        }"""
    )
    assert trigger_metrics == {
        "width": 28,
        "height": 28,
        "paddingLeft": "0px",
        "paddingRight": "0px",
    }
    search_trigger.click()
    expect(search).to_have_css("opacity", "1")
    expect(search).to_have_css("pointer-events", "auto")

    app_page.locator("#sidebarModelPicker").click()
    picker = app_page.locator("#consensusModelDropdown").locator("xpath=..")
    expect(picker.locator(".model-picker-menu")).to_have_class(re.compile(r"\bis-open\b"))
    assert sidebar.evaluate("element => element.scrollTop") == 0
    expect(app_page.locator("#sidebarModelCount")).to_have_text("6 providers selected")
    picker.locator(".model-picker-custom-option").click()
    picker.locator(".model-picker-row-toggle").nth(5).click()
    expect(app_page.locator("#sidebarModelCount")).to_have_text("5 providers selected")


def test_question_input_grows_and_caps_on_desktop_and_mobile(app_page):
    input_box = app_page.locator("#questionInput")

    input_box.fill("")
    base_height = input_box.bounding_box()["height"]

    input_box.fill("\n".join(f"Line {index}" for index in range(6)))
    grown_height = input_box.bounding_box()["height"]
    assert grown_height > base_height
    assert grown_height < 220

    input_box.fill("\n".join(f"Line {index}" for index in range(40)))
    desktop = input_box.evaluate(
        """el => ({
          height: el.getBoundingClientRect().height,
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
          overflowY: getComputedStyle(el).overflowY,
        })"""
    )
    assert desktop["height"] == 220
    assert desktop["scrollHeight"] > desktop["clientHeight"]
    assert desktop["overflowY"] == "auto"

    app_page.set_viewport_size({"width": 390, "height": 844})
    mobile = input_box.evaluate(
        """el => ({
          height: el.getBoundingClientRect().height,
          overflowY: getComputedStyle(el).overflowY,
        })"""
    )
    assert mobile["height"] == 180
    assert mobile["overflowY"] == "auto"

    input_box.fill("")
    reset = input_box.evaluate(
        """el => ({
          height: el.getBoundingClientRect().height,
          overflowY: getComputedStyle(el).overflowY,
        })"""
    )
    assert reset["height"] == base_height
    assert reset["overflowY"] == "hidden"


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


def test_consensus_citations_follow_terminal_punctuation(app_page):
    result = app_page.evaluate(
        """() => {
          const host = document.getElementById("consensusAnswerBody");
          window.currentEvidenceSources = [{
            id: "S1",
            title: "Example source",
            url: "https://example.org/source",
          }];
          window.injectMarkdown(
            host,
            'Fact [S1]. Question [S1]? "Quote [S1]."'
          );
          return {
            text: host.textContent,
            previous: Array.from(host.querySelectorAll(".src-ref"))
              .map(ref => ref.previousSibling?.textContent || ""),
          };
        }"""
    )
    assert result["text"].strip() == 'Fact.1 Question?1 "Quote."1'
    assert result["previous"] == [".", "?", '."']


def test_usage_display_is_stable_and_updates_visible_quota_panel(app_page):
    """Leere Usage-Updates behalten den letzten Wert; das sichtbare
    Sidebar-Panel spiegelt die versteckte Kompatibilitaetsquelle."""
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
          return {
            text: line.textContent,
            deepText: document.getElementById('deepUsageDisplay').textContent,
            valueTag: value.tagName,
            valueWeight: Number.parseInt(getComputedStyle(value).fontWeight, 10),
          };
        }"""
    )
    app_page.wait_for_function(
        "() => document.getElementById('quotaRunsValue').textContent.trim() === '2 / 3'"
    )

    assert metrics["text"] == "Runs: 2 / 3"
    assert metrics["deepText"] == "Deep Think: 0 / 0"
    assert metrics["valueTag"] == "STRONG"
    assert metrics["valueWeight"] >= 600
    assert app_page.locator("#quotaRowRuns").evaluate("element => element.hidden") is False
    expect(app_page.locator("#quotaRunsValue")).to_have_text("2 / 3")


def test_empty_app_and_consensus_picker_do_not_scroll_unnecessarily(app_page):
    app_page.set_viewport_size({"width": 390, "height": 844})
    page_metrics = app_page.evaluate(
        """() => ({
          scrollHeight: document.documentElement.scrollHeight,
          clientHeight: document.documentElement.clientHeight,
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
        })"""
    )
    assert page_metrics["scrollHeight"] <= page_metrics["clientHeight"]
    assert page_metrics["scrollWidth"] <= page_metrics["clientWidth"]

    control_metrics = app_page.evaluate(
        """() => {
          const selectors = [
            "#attachTrigger",
            ".consensus-model .model-picker-display",
            "#sendButton",
          ];
          return selectors.map(selector => {
            const rect = document.querySelector(selector).getBoundingClientRect();
            return { selector, top: rect.top, height: rect.height, center: rect.top + rect.height / 2 };
          });
        }"""
    )
    centers = [control["center"] for control in control_metrics]
    assert max(centers) - min(centers) <= 1
    assert all(control["height"] == 36 for control in control_metrics)

    sidebar = app_page.locator(".sidebar")
    expect(sidebar).to_have_attribute("aria-hidden", "true")
    assert sidebar.evaluate("element => element.inert") is True
    app_page.locator(".app-nav-float .sidebar-toggle").click()
    expect(sidebar).not_to_have_attribute("aria-hidden", "true")
    assert sidebar.evaluate("element => element.inert") is False
    app_page.locator("#sidebarToggleInner").click()
    expect(sidebar).to_have_attribute("aria-hidden", "true")
    assert sidebar.evaluate("element => element.inert") is True

    app_page.locator(".consensus-model .model-picker-display").click()
    menu_metrics = app_page.locator(".consensus-model .model-picker-menu").evaluate(
        """element => ({
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
          overflowX: getComputedStyle(element).overflowX,
          left: element.getBoundingClientRect().left,
          right: element.getBoundingClientRect().right,
          viewportWidth: document.documentElement.clientWidth,
          documentWidth: document.documentElement.scrollWidth,
        })"""
    )
    assert menu_metrics["scrollWidth"] <= menu_metrics["clientWidth"]
    assert menu_metrics["overflowX"] == "hidden"
    assert menu_metrics["left"] >= 0
    assert menu_metrics["right"] <= menu_metrics["viewportWidth"]
    assert menu_metrics["documentWidth"] <= menu_metrics["viewportWidth"]
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
    # Passage-Interaktion explizit unter Touch-Bedingungen pruefen. Die alte
    # Implementierung brach bei genau diesem Media Query vor dem Binding ab.
    app_page.evaluate(
        """() => {
          const nativeMatchMedia = window.matchMedia.bind(window);
          window.matchMedia = query => {
            if (query.includes("(hover: hover)") || query.includes("(pointer: fine)")) {
              return {
                matches: false, media: query, onchange: null,
                addListener() {}, removeListener() {},
                addEventListener() {}, removeEventListener() {}, dispatchEvent() { return false; },
              };
            }
            return nativeMatchMedia(query);
          };
        }"""
    )
    _send_question(app_page)

    # Regulärer Modus: determinate Antwortphase -> indeterminate Synthese.
    # Die rahmenlose Zeile bleibt mobil kompakt und clippt ihren Text nicht.
    pipeline = app_page.locator("#consensusRun")
    expect(pipeline).to_be_visible(timeout=10000)
    expect(pipeline).to_have_attribute("data-stage", "answers")
    metrics = pipeline.evaluate(
        """(el) => {
          const current = el.querySelector('.run-now');
          return {
            height: current.getBoundingClientRect().height,
            clipped: current.scrollWidth > current.clientWidth,
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

    claim_badges = app_page.locator("#consensusAnswerBody .claim-badge")
    expect(claim_badges).to_have_count(1)
    expect(claim_badges.first).to_be_visible(timeout=15000)
    expect(claim_badges.first).to_have_text(
        re.compile(r"^\d+/\d+$")
    )

    # Der fertige Footer zeigt die Einzelantworten-Disclosure auch ohne
    # Agent Mode. Sie ist ein fester Teil jeder Consensus-Antwort.
    model_answers_toggle = app_page.locator("#agentModeAnswersToggle")
    expect(model_answers_toggle).to_be_visible(timeout=15000)
    expect(model_answers_toggle.locator(".consensus-tab-label")).to_have_text("Compare answers")
    # Wie die beiden anderen Schubladen sagt der Chip, wie viel dahinter liegt.
    expect(model_answers_toggle.locator(".consensus-tab-count")).to_have_text("6")

    footer_metrics = app_page.locator("#consensusFooterTabs").evaluate(
        """element => ({
          display: getComputedStyle(element).display,
          columns: getComputedStyle(element).gridTemplateColumns.split(" ").length,
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
          buttonHeights: Array.from(element.querySelectorAll(".consensus-tab:not([hidden])"))
            .map(button => button.getBoundingClientRect().height),
        })"""
    )
    assert footer_metrics["display"] == "grid"
    assert footer_metrics["columns"] == 3
    assert footer_metrics["scrollWidth"] <= footer_metrics["clientWidth"]
    assert all(height <= 40 for height in footer_metrics["buttonHeights"])

    # Mobile: Aktionen und Lauf-Fakten leben in getrennten Zeilen. Zuvor
    # teilten sie sich eine Grid-Zeile und Watch/Cite kollidierten bei langen
    # Laufdaten sichtbar mit "6 models" und "Run again".
    footer_layout = app_page.locator("#runProvenance").evaluate(
        """element => {
          const actions = element.querySelector("#consensusFooterActions").getBoundingClientRect();
          const facts = element.querySelector(".consensus-footer-facts").getBoundingClientRect();
          const verdict = element.querySelector("#consensusVerdict").getBoundingClientRect();
          const tabs = element.querySelector("#consensusFooterTabs").getBoundingClientRect();
          const overlaps = (a, b) =>
            a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
          return {
            scrollWidth: element.scrollWidth,
            clientWidth: element.clientWidth,
            actionsFactsOverlap: overlaps(actions, facts),
            ordered:
              verdict.bottom <= tabs.top + 1
              && tabs.bottom <= actions.top + 1
              && actions.bottom <= facts.top + 1,
          };
        }"""
    )
    assert footer_layout["scrollWidth"] <= footer_layout["clientWidth"]
    assert footer_layout["actionsFactsOverlap"] is False
    assert footer_layout["ordered"] is True

    # Desktop/tablet: Share/Watch/Cite sitzen an der Oberkante der gemeinsamen
    # Summary-Flaeche und nicht mehr vertikal in der Mitte des hohen Verdicts.
    app_page.set_viewport_size({"width": 1000, "height": 844})
    desktop_footer = app_page.locator("#runProvenance").evaluate(
        """element => {
          const actions = element.querySelector("#consensusFooterActions").getBoundingClientRect();
          const verdict = element.querySelector("#consensusVerdict").getBoundingClientRect();
          const surface = getComputedStyle(element, "::before");
          return {
            topDelta: actions.top - verdict.top,
            hasSummarySurface: surface.backgroundImage !== "none",
          };
        }"""
    )
    assert 0 <= desktop_footer["topDelta"] <= 12
    assert desktop_footer["hasSummarySurface"] is True
    app_page.set_viewport_size({"width": 390, "height": 844})

    app_page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    app_page.wait_for_timeout(100)
    result_gap = app_page.evaluate(
        """() => {
          const input = document.querySelector(".input-section");
          const footer = document.getElementById("runProvenance").getBoundingClientRect();
          const inputRect = input.getBoundingClientRect();
          const responses = document.querySelector(".response-section");
          const container = document.querySelector(".container");
          return {
            responsesDisplay: getComputedStyle(responses).display,
            inputMarginTop: getComputedStyle(input).marginTop,
            inputPosition: getComputedStyle(input).position,
            gap: inputRect.top - footer.bottom,
            bottomGap: window.innerHeight - inputRect.bottom,
            reserve: parseFloat(getComputedStyle(container).paddingBottom),
            inputHeight: inputRect.height,
            scrollable:
              document.documentElement.scrollHeight - window.innerHeight > 2,
          };
        }"""
    )
    assert result_gap["responsesDisplay"] == "none"
    assert result_gap["inputMarginTop"] == "0px"
    assert result_gap["inputPosition"] == "fixed"
    assert abs(result_gap["bottomGap"]) <= 1
    assert abs(result_gap["reserve"] - result_gap["inputHeight"]) <= 1
    # Der Composer darf den Fuss nie ueberdecken. Dass er unmittelbar an ihm
    # anschliesst, ist erst pruefbar, wenn die Seite ueberhaupt scrollt: seit
    # der Fuss zwei statt drei Zeilen hat (2026-07-28), fuellt eine kurze
    # Mock-Antwort den Viewport nicht mehr aus, und die verbleibende Luft ist
    # dann keine scrollbare Strecke, sondern schlicht eine kurze Seite.
    assert result_gap["gap"] >= -2
    if result_gap["scrollable"]:
        assert abs(result_gap["gap"]) <= 2

    # Nach der fertigen Antwort schrumpft der Composer zur Entscheidung:
    # kein leeres/deaktiviertes Fragefeld, nur die zwei ehrlichen Wege weiter.
    expect(app_page.locator("#composerGate")).to_be_visible(timeout=15000)
    expect(app_page.locator("#questionInput")).to_be_hidden()
    expect(app_page.locator(".chat-input-container .consensus-switch-container")).to_be_hidden()
    expect(app_page.locator(".composer-gate-new")).to_be_visible()
    expect(app_page.locator(".composer-gate-followup")).to_be_visible()
    collapsed_composer = app_page.locator(".chat-input-container").bounding_box()
    assert collapsed_composer is not None
    assert collapsed_composer["height"] <= 64

    # Der kompakte Choice-State gilt fuer alle Tiers, und die Follow-up-Wahl
    # steht auch Free offen (kein Pro-Gate mehr): erst die bewusste Wahl
    # öffnet das Feld wieder und hängt den Kontext-Chip an.
    expect(app_page.locator(".composer-gate-followup .pro-badge")).to_have_count(0)
    expect(app_page.locator("#questionInput")).to_be_hidden()
    app_page.locator(".composer-gate-followup").click()
    expect(app_page.locator("#questionInput")).to_be_visible()
    expect(app_page.locator("#questionInput")).to_be_enabled()
    expect(app_page.locator("#questionInput")).to_have_attribute(
        "placeholder", "Ask a follow-up question"
    )
    expect(app_page.locator("#followupChipBar")).to_be_visible()

    # Inline-Confidence: der Widerspruch wird im Antworttext selbst markiert
    # (Linie + Quote), nicht nur in einer Karte daneben.
    marked = app_page.locator("#consensusAnswerBody .cx-claim.is-major").first
    expect(marked).to_be_visible(timeout=15000)
    # Claim und Difference treffen im Fixture denselben Satz. Dort bleibt nur
    # die aussagekraeftigere Quote sichtbar; der doppelte Punkt ist weg.
    overlap_marker = app_page.locator("#consensusAnswerBody .cx-marker").first
    expect(overlap_marker).to_be_hidden()
    assert (
        overlap_marker.get_attribute("aria-label")
    ), "Marker braucht ein sprechendes aria-label"

    # Linie und Quote sagen dasselbe: wo das Badge gelb ist (Dissens), darf die
    # Unterlinie nicht neutral grau bleiben.
    assert app_page.evaluate(
        """() => Array.from(
             document.querySelectorAll("#consensusAnswerBody .claim-badge.has-dissent")
           ).every(badge => {
             const span = badge.previousElementSibling;
             return span && span.classList.contains("cx-claim")
               && (span.classList.contains("is-split") || span.classList.contains("is-major"));
           })"""
    ), "Gelbe Quote braucht eine Bernstein-Linie, keine graue"

    # Touch: die markierte Passage selbst oeffnet dasselbe Agreement-Sheet wie
    # der sichtbare Badge. Der Hintergrund ist inert und der Fokus bleibt im
    # modalen Dialog.
    # Die Erklaerung liefert die formatierte Hover-Vorschau und der Dialog,
    # NICHT zusaetzlich ein nativer Browser-Tooltip (bewusste Entscheidung an
    # Badge und Marker, siehe makeBadge in consensus-insights.js). Geprueft
    # wird deshalb, dass die Passage bedienbar ist - nicht, dass sie ein
    # title-Attribut traegt.
    assert "is-interactive" in (marked.get_attribute("class") or "")
    marked.click()
    popover = app_page.locator("#claimPopover")
    expect(popover).to_be_visible()
    expect(popover).to_have_attribute("aria-modal", "true")
    expect(popover).to_have_attribute("aria-labelledby", "claimPopoverTitle")
    expect(app_page.locator("#claimPopover .claim-popover-close")).to_be_focused()
    assert app_page.evaluate(
        """() => Array.from(document.body.children)
          .filter(el => !["claimPopover", "claimSheetBackdrop"].includes(el.id))
          .every(el => el.inert)"""
    )
    app_page.keyboard.press("Escape")
    expect(popover).to_be_hidden()
    expect(claim_badges.first).to_be_focused()

    # Der vollstaendige Differences-Ueberblick liegt zugeklappt UNTER der
    # Antwort; die Karten bleiben erreichbar, sind aber nicht mehr die zweite
    # Spalte der Primaeransicht.
    panel = app_page.locator("#consensusDifferencesPanel")
    expect(panel).to_be_hidden(timeout=15000)
    assert panel.evaluate("el => el.open") is False, "Differences starten zugeklappt"
    expect(app_page.locator(".diff-card.is-contradiction").first).to_be_hidden()

    # Alle drei Disclosures geben denselben sanften Scroll-Hinweis auf den
    # neu sichtbaren Inhalt. Sources zielt auf den ersten Beleg, Compare
    # answers auf die erste eingeschlossene Antwort (jeweils nach dem Layout).
    app_page.evaluate(
        """() => {
          window.currentEvidenceSources = [{
            id: "S1",
            title: "Disclosure scroll fixture",
            url: "https://example.org/source",
            snippet: "A source rendered specifically for the disclosure behavior test.",
          }];
          window.renderEvidenceSources(window.currentEvidenceSources);
          window.__disclosureScrollTargets = [];
          window.__originalScrollIntoView = Element.prototype.scrollIntoView;
          Element.prototype.scrollIntoView = function (options) {
            window.__disclosureScrollTargets.push({
              id: this.id || "",
              tag: this.tagName,
              options,
              visible: this.getBoundingClientRect().height > 0,
            });
            return window.__originalScrollIntoView.call(this, options);
          };
        }"""
    )

    sources_tab = app_page.locator("#consensusSourcesTab")
    sources_tab.click()
    expect(app_page.locator("#consensusSourcesPanel li").first).to_be_visible()
    app_page.wait_for_timeout(100)
    source_scroll = app_page.evaluate(
        "() => window.__disclosureScrollTargets.find(item => item.tag === 'LI')"
    )
    assert source_scroll["visible"] is True
    assert source_scroll["options"]["block"] == "nearest"
    sources_tab.click()

    model_answers_toggle.click()
    expect(app_page.locator("#openaiResponse")).to_be_visible()
    app_page.wait_for_timeout(100)
    answer_scroll = app_page.evaluate(
        "() => window.__disclosureScrollTargets.find(item => item.id === 'openaiResponse')"
    )
    assert answer_scroll["visible"] is True
    assert answer_scroll["options"]["block"] == "nearest"
    model_answers_toggle.click()

    app_page.evaluate(
        """() => {
          Element.prototype.scrollIntoView = window.__originalScrollIntoView;
          delete window.__originalScrollIntoView;
          delete window.__disclosureScrollTargets;
        }"""
    )

    app_page.locator("#consensusDifferencesTab").click()
    expect(app_page.locator(".diff-card.is-contradiction").first).to_be_visible(timeout=5000)

    # Beide Sprungpfade öffnen die standardmäßig verborgenen Modellantworten,
    # fahren die Fundstelle an und markieren das Originalzitat. Der Modus darf
    # sich dabei nicht als Seiteneffekt ändern.
    diff_jump = app_page.locator(".diff-card.is-contradiction .diff-jump-link").first
    expect(diff_jump).to_be_visible()
    diff_jump.click()
    expect(app_page.locator("body.agent-mode-show-answers")).to_have_count(1)
    expect(model_answers_toggle.locator(".consensus-tab-label")).to_have_text("Hide answers")
    expect(app_page.locator(".response-section mark.quote-flash, .response-section .quote-flash-block").first).to_be_visible()
    assert app_page.evaluate("() => window.isAgentModeEnabled()") is False

    model_answers_toggle.click()
    expect(app_page.locator("#openaiResponse")).to_be_hidden()

    claim_badges.first.click()
    claim_jump = app_page.locator("#claimPopover .claim-model-row.is-dissent .claim-jump-link")
    expect(claim_jump).to_be_visible()
    claim_jump.click()
    expect(app_page.locator("body.agent-mode-show-answers")).to_have_count(1)
    expect(app_page.locator("#grokResponse mark.quote-flash, #grokResponse .quote-flash-block").first).to_be_visible()
    assert app_page.evaluate("() => window.isAgentModeEnabled()") is False

    # Ausgangszustand für die bestehende Disclosure-/Reihenfolge-Prüfung.
    model_answers_toggle.click()
    expect(model_answers_toggle.locator(".consensus-tab-label")).to_have_text("Compare answers")

    # Kopierter Text darf keine Marker-/Badge-Beschriftung enthalten.
    copied = app_page.evaluate(
        """() => {
          const body = window.App.consensusBodyEl();
          const clone = body.cloneNode(true);
          clone.querySelectorAll('.claim-badge, .cx-marker, .copy-btn, .response-code-copy')
            .forEach(el => el.remove());
          clone.style.position = 'absolute';
          clone.style.left = '-99999px';
          document.body.appendChild(clone);
          const text = clone.innerText.trim();
          clone.remove();
          return text;
        }"""
    )
    assert not re.search(r"\d+\s*/\s*\d+", copied), f"Badge-Zaehlung im Copy-Text: {copied!r}"

    model_answers_toggle.click()
    expect(app_page.locator("#openaiResponse")).to_be_visible()
    consensus_box = app_page.locator("#consensusOutput").bounding_box()
    first_answer_box = app_page.locator("#openaiResponse").bounding_box()
    assert consensus_box is not None
    assert first_answer_box is not None
    assert consensus_box["y"] < first_answer_box["y"]
    expect(pipeline).to_be_hidden(timeout=10000)

    run_again = app_page.locator("#runReplayButton")
    expect(run_again).to_be_visible()
    expect(run_again.locator(".run-replay-label")).to_have_text("Run again")
    # Ein Wiederholen ist ein voller zweiter Lauf. Am Knopf muss stehen, was er
    # kostet, und am Eingabefeld muss es stehen bleiben, bis abgeschickt wird.
    # Der Mock-Nutzer laeuft ohne Limit; der Preis gehoert an den Zaehler, also
    # bekommt er hier einen.
    app_page.evaluate(
        "() => window.App.renderUsageDisplay({remaining: 2, totalLimit: 3, deepRemaining: 0, deepLimit: 0})"
    )
    expect(run_again).to_contain_text("uses 1 run")
    run_again.click()
    expect(app_page.locator("body.is-hero")).to_have_count(1)
    expect(app_page.locator("#questionInput")).to_be_visible()
    expect(app_page.locator("#questionInput")).to_have_value(QUESTION)
    expect(app_page.locator("#questionInput")).to_be_focused()
    expect(app_page.locator("#composerRunNotice")).to_be_visible()
    expect(app_page.locator("#composerRunNotice")).to_contain_text("complete new run")

    errors = get_console_errors()
    assert errors == [], f"Konsolen-Fehler im Consensus-Flow: {errors}"


def test_followup_keeps_the_previous_answer_and_appends_the_new_question(app_page):
    """Ein Follow-up darf den fertigen Turn nicht visuell recyceln: die alte
    Frage/Antwort bleibt stehen, die neue User-Frage beginnt darunter."""
    app_page.evaluate("() => window.setAgentMode(false, { persist: true })")
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    expect(app_page.locator("#composerGate")).to_be_visible(timeout=20000)

    original_answer = app_page.locator("#consensusAnswerBody").text_content().strip()
    assert original_answer
    original_agreement = app_page.locator("#consensusVerdict .verdict-score").text_content().strip()
    assert original_agreement

    app_page.locator(".composer-gate-followup").click()
    followup_question = "Which consideration should I prioritize first?"
    app_page.locator("#questionInput").fill(followup_question)
    app_page.locator("#sendButton").click()

    archived_turn = app_page.locator("#threadHistory .thread-history-turn").last
    expect(archived_turn).to_be_visible(timeout=15000)
    expect(archived_turn.locator(".thread-history-question-text")).to_have_text(QUESTION)
    archived_answer = archived_turn.locator(".thread-history-answer-body")
    expect(archived_answer).to_be_visible()
    assert original_answer in archived_answer.text_content()
    archived_agreement = archived_turn.locator(".thread-history-verdict .verdict-score")
    expect(archived_agreement).to_have_text(original_agreement)
    assert "…" not in archived_turn.locator(".thread-history-verdict .verdict-detail").text_content()
    expect(app_page.locator("#threadAskText")).to_have_text(followup_question)

    layout = app_page.evaluate(
        """() => {
          const previous = document.querySelector('.thread-history-answer');
          const current = document.getElementById('threadAsk');
          return {
            previousBottom: previous.getBoundingClientRect().bottom,
            currentTop: current.getBoundingClientRect().top,
            verdictRight: document.querySelector('.thread-history-verdict')
              .getBoundingClientRect().right,
            verdictMainRight: document.querySelector('.thread-history-verdict .verdict-main')
              .getBoundingClientRect().right,
            currentAlign: getComputedStyle(current).textAlign,
            archivedAlign: getComputedStyle(
              document.querySelector('.thread-history-question')
            ).textAlign,
          };
        }"""
    )
    assert layout["previousBottom"] <= layout["currentTop"]
    assert abs(layout["verdictRight"] - layout["verdictMainRight"]) <= 2
    assert layout["currentAlign"] == "right"
    assert layout["archivedAlign"] == "right"

    # Turn 3 is the regression boundary: the second exchange used to vanish
    # when the live render tree was recycled for the next answer.
    expect(app_page.locator("#composerGate")).to_be_visible(timeout=20000)
    app_page.locator(".composer-gate-followup").click()
    third_question = "What should I do immediately after that?"
    app_page.locator("#questionInput").fill(third_question)
    app_page.locator("#sendButton").click()

    archived_turns = app_page.locator("#threadHistory .thread-history-turn")
    expect(archived_turns).to_have_count(2, timeout=15000)
    expect(archived_turns.nth(0).locator(".thread-history-question-text")).to_have_text(QUESTION)
    expect(archived_turns.nth(1).locator(".thread-history-question-text")).to_have_text(followup_question)
    expect(app_page.locator("#threadAskText")).to_have_text(third_question)


def test_split_claim_underlines_in_the_colour_of_its_badge(app_page):
    """Geteilte Zustimmung ohne Widerspruchs-Karte: Linie und Quote muessen
    dieselbe Bernstein-Note tragen. Vorher lief die Linie hier neutral grau,
    waehrend die 2/4-Quote daneben schon gelb war (User-Befund 2026-07-28).
    Das Mock-Fixture deckt nur den Ueberlappungsfall (is-major) ab, deshalb
    wird der Renderer hier direkt mit einem geteilten Claim gefuettert."""
    app_page.evaluate(
        """() => {
          document.getElementById("consensusAnswerBody").innerHTML =
            "<p>The tower stands 330 metres tall today.</p>";
          window.renderConsensusInsights({
            claims: [{
              anchor: "stands 330 metres tall",
              agree: [{model: "openai"}, {model: "gemini"}],
              dissent: [{model: "grok", quote: "324 metres"},
                        {model: "claude", quote: "300 metres"}]
            }],
            differences: [],
            models_compared: ["openai", "gemini", "grok", "claude"]
          }, 4);
        }"""
    )
    marked = app_page.locator("#consensusAnswerBody .cx-claim").first
    assert "is-split" in (marked.get_attribute("class") or "")
    expect(app_page.locator("#consensusAnswerBody .claim-ratio")).to_have_text("2/4")
    # Bernstein statt neutral: Rot deutlich ueber Blau, in derselben Richtung
    # wie die Schriftfarbe des Badges.
    tones = app_page.evaluate(
        """() => {
          const channels = (value) => (value.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
          const span = document.querySelector("#consensusAnswerBody .cx-claim");
          const badge = document.querySelector("#consensusAnswerBody .claim-badge");
          return {
            line: channels(getComputedStyle(span).textDecorationColor),
            badge: channels(getComputedStyle(badge).color),
          };
        }"""
    )
    for name, (red, _green, blue) in tones.items():
        assert red > blue * 1.5, f"{name} ist nicht bernsteinfarben: {tones[name]}"


def test_claim_anchor_with_markdown_syntax_marks_the_rendered_sentence(app_page):
    """Anker sind woertliche Kopien aus dem MARKDOWN-Quelltext ("1. **World
    class:** about 1,300 watts"), markiert und angezeigt wird aber der
    GERENDERTE Text. Ohne Entfernen der Auszeichnung fand der Anker seine
    Stelle nie und landete mitsamt sichtbarer Sternchen in der Fallback-Liste
    "Key claims" (User-Befund 2026-07-29)."""
    app_page.evaluate(
        """() => {
          document.getElementById("consensusAnswerBody").innerHTML =
            "<ol><li><strong>World class:</strong> about 1,300 to 1,500 watts.</li></ol>";
          window.renderConsensusInsights({
            claims: [{
              anchor: "1. **World class:** about 1,300 to 1,500 watts.",
              agree: [{model: "openai"}, {model: "gemini"}, {model: "grok"}],
              dissent: [{model: "claude", quote: "**about 1,200 watts**"}]
            }],
            differences: [],
            models_compared: ["openai", "gemini", "grok", "claude"]
          }, 4);
        }"""
    )
    # Der Anker haengt jetzt am Satz - die Fallback-Liste bleibt leer. Die
    # Markierung laeuft ueber die <strong>-Grenze und damit ueber mehrere
    # Textknoten, ist also bewusst nicht ein einzelner Span.
    marked = app_page.locator("#consensusAnswerBody .cx-claim")
    assert marked.count() >= 1
    assert "World class" in "".join(marked.all_inner_texts())
    expect(app_page.locator("#consensusClaimsFallback")).to_be_hidden()
    expect(app_page.locator("#consensusAnswerBody .claim-ratio")).to_have_text("3/4")

    # Und wo der Anker doch als Text erscheint, als sicheres Inline-Markdown
    # statt mit sichtbaren Sternchen. dispatch_event ist hier bewusst: Das
    # isolierte Renderer-Fixture haelt den aeusseren Ergebniscontainer hidden.
    app_page.locator("#consensusAnswerBody .claim-badge").dispatch_event("click")
    claim_text = app_page.locator(".claim-popover-claim")
    expect(claim_text).to_be_visible()
    assert "*" not in claim_text.inner_text()
    expect(claim_text.locator("strong")).to_have_text("World class:")
    expect(app_page.locator(".claim-model-quote")).to_have_text("about 1,200 watts")
    expect(app_page.locator(".claim-model-quote strong")).to_have_text("about 1,200 watts")

    # Ein wirklich nicht auffindbarer Anker bleibt als Key-Claims-Fallback
    # sichtbar, rendert die Auszeichnung dort aber ebenfalls korrekt.
    app_page.evaluate(
        """() => {
          window.renderConsensusInsights({
            claims: [{
              anchor: "2. **Unmatched claim:** still readable.",
              agree: [{model: "openai"}],
              dissent: []
            }],
            differences: [],
            models_compared: ["openai"]
          }, 1);
        }"""
    )
    fallback = app_page.locator("#consensusClaimsFallback")
    assert fallback.get_attribute("hidden") is None
    expect(fallback.locator("strong")).to_have_text("Unmatched claim:")
    assert "*" not in fallback.inner_text()


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
    # Der Chip traegt neben dem Label auch die Zahl der Antworten dahinter,
    # deshalb wird gezielt das Label geprueft.
    toggle_label = toggle.locator(".consensus-tab-label")
    expect(toggle_label).to_have_text("Compare answers")
    expect(toggle.locator(".consensus-tab-count")).to_have_text("6")
    expect(app_page.locator("#openaiResponse")).to_be_hidden()

    toggle.click()
    expect(app_page.locator("body.agent-mode-enabled.agent-mode-show-answers")).to_have_count(1)
    expect(toggle_label).to_have_text("Hide answers")
    expect(app_page.locator("#openaiResponse")).to_be_visible()

    toggle.click()
    expect(toggle_label).to_have_text("Compare answers")
    expect(app_page.locator("#openaiResponse")).to_be_hidden()


def test_watch_dialog_uses_safe_defaults_keeps_telegram_visible_and_reveals_condition(app_page):
    """Watch-Erstellung startet kompakt mit sicheren Defaults, hält Telegram
    sichtbar und blendet erweiterte Felder nur bei Bedarf ein."""
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)
    expect(app_page.locator("#consensusResponse")).to_contain_text("Mock consensus", timeout=30000)

    # Share/Watch/Cite haengen an der FERTIGEN Antwort: sie stehen in der
    # Provenance-Fusszeile, die erst erscheint, wenn der Lauf uebergeben hat.
    # Erst danach den Ergebnis-Kontext faelschen — das Final-Event des Streams
    # ueberschreibt window.lastShareResultId sonst wieder.
    expect(app_page.locator("#consensusWatchButton")).to_be_visible(timeout=30000)

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

    # Die Defaults sahen aus wie feste Fakten. Der Weg zum Verstellen muss
    # deshalb IN der Zusammenfassung stehen und ueber der Zustellzeile liegen —
    # nicht als letzte Zeile des Dialogs, wo ihn niemand gesehen hat.
    edit_box = app_page.locator("#watchEditDefaults").bounding_box()
    summary_label_box = app_page.locator(".watch-setup-summary-label").bounding_box()
    advanced_box = app_page.locator("#watchAdvancedSettings").bounding_box()
    delivery_box = app_page.locator(".watch-delivery-field").bounding_box()
    assert edit_box is not None and summary_label_box is not None
    assert advanced_box is not None and delivery_box is not None
    assert abs(edit_box["y"] - summary_label_box["y"]) < 20
    assert advanced_box["y"] < delivery_box["y"]

    # Jeder Chip ist selbst der Weg zu seinem Feld.
    app_page.click("#watchScheduleSummary")
    expect(app_page.locator("#watchAdvancedSettings")).to_have_attribute("open", "")
    expect(app_page.locator("#watchInterval")).to_be_focused()
    expect(app_page.locator("#watchEditDefaults")).to_have_text("Done")
    app_page.click("#watchEditDefaults")
    expect(app_page.locator("#watchAdvancedSettings")).not_to_have_attribute("open", "")
    expect(app_page.locator("#watchEditDefaults")).to_have_text("Edit")

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
    expect(app_page.locator("#watchDashLimit")).to_contain_text("Standard access")
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
        "costs me money each time"
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
    sichtbar, oeffnet mit Pro-Badge aber den Kosten-Hinweis statt eines Kaufdialogs."""
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
            pickerText: document.querySelector(
              ".consensus-model .model-picker-display-text"
            )?.textContent || "",
          };

          // /prepare liefert den autoritativen Tier-Status und loest genau
          // diesen Restore aus. Der gespeicherte DeepSeek-Wert darf die
          // Attachment-Sperre dabei nicht ueberschreiben.
          window.updatePremiumModelsState(window.isUserPro === true);
          const afterTierRefresh = {
            checked: checkbox.checked,
            disabled: checkbox.disabled,
            responseExcluded: document.getElementById("deepseekResponse").classList.contains("excluded"),
            includedProgressModels: document.querySelectorAll(
              ".response-section > .response-box:not(.excluded)"
            ).length,
            canGenerateWithTwoAnswers: (() => {
              for (const id of ["openaiResponse", "mistralResponse"]) {
                const box = document.getElementById(id);
                box.dataset.responseState = "complete";
                box.dataset.consensusAnswer = id + " answer";
                box.querySelector(".collapsible-content").textContent = id + " answer";
              }
              return window.canGenerateConsensus();
            })(),
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
            afterTierRefresh,
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
    assert result["whileAttached"]["pickerText"].startswith("5 models")
    assert result["afterTierRefresh"] == {
        "checked": False,
        "disabled": True,
        "responseExcluded": True,
        "includedProgressModels": 5,
        "canGenerateWithTwoAnswers": True,
    }
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


def test_pdf_drop_uses_full_attachment_whitelist(app_page):
    """Drag-and-drop akzeptiert dieselben Dokumenttypen wie der Dateidialog."""
    dialogs = []

    def dismiss_dialog(dialog):
        dialogs.append(dialog.message)
        dialog.dismiss()

    app_page.on("dialog", dismiss_dialog)
    app_page.evaluate(
        """() => {
          window.isUserPro = true;
          const input = document.querySelector(".chat-input-container");
          const transfer = new DataTransfer();
          transfer.items.add(new File(
            [new Uint8Array([37, 80, 68, 70, 45, 49, 46, 55, 10])],
            "brief.pdf",
            { type: "application/pdf" }
          ));
          input.dispatchEvent(new DragEvent("dragenter", {
            bubbles: true,
            cancelable: true,
            dataTransfer: transfer,
          }));
          window.__attachmentDropOverlayText =
            getComputedStyle(input, "::after").content;
          input.dispatchEvent(new DragEvent("drop", {
            bubbles: true,
            cancelable: true,
            dataTransfer: transfer,
          }));
        }"""
    )
    app_page.wait_for_function(
        "() => window.pendingAttachments?.some(att => att.name === 'brief.pdf')"
    )

    result = app_page.evaluate(
        """() => ({
          attachment: window.pendingAttachments.find(att => att.name === "brief.pdf"),
          chipName: document.querySelector(".attachment-chip-name")?.textContent || "",
          deepSeekChecked: document.getElementById("selectDeepSeek").checked,
          deepSeekDisabled: document.getElementById("selectDeepSeek").disabled,
          dropOverlayText: window.__attachmentDropOverlayText,
        })"""
    )
    assert dialogs == []
    assert result["attachment"]["mime"] == "application/pdf"
    assert result["chipName"] == "brief.pdf"
    assert result["dropOverlayText"] == '"Drop file to attach"'
    assert result["deepSeekChecked"] is False
    assert result["deepSeekDisabled"] is True


def test_attachment_send_hard_blocks_stale_deepseek_selection(app_page):
    """Auch veralteter Checkbox-State darf keinen DeepSeek-Request erzeugen."""
    ask_requests = []
    app_page.on(
        "request",
        lambda request: ask_requests.append(request.url)
        if "/ask_" in request.url else None,
    )
    app_page.evaluate(
        """() => {
          window.isUserPro = true;
          for (const pref of window.App.modelPrefs) {
            window.App.setModelSelectionState(pref, true, {
              persist: false,
              syncCheckbox: true,
              animate: false,
            });
          }
          for (const key of [
            "openaiKey", "mistralKey", "anthropicKey",
            "geminiKey", "deepseekKey", "grokKey",
          ]) {
            localStorage.setItem(key, "e2e-dummy-key");
          }
          document.getElementById("useOwnKeysSwitch").checked = true;
          window.pendingAttachments = [{
            name: "brief.pdf",
            mime: "application/pdf",
            size: 9,
            data: "JVBERi0xLjcK",
          }];
          window.renderAttachmentChips();

          // Simuliert einen veralteten/extern wieder gesetzten UI-State.
          document.getElementById("selectDeepSeek").checked = true;
          document.getElementById("deepseekResponse").classList.remove("excluded");
        }"""
    )
    app_page.fill("#questionInput", QUESTION)
    app_page.click("#sendButton")
    app_page.wait_for_timeout(1500)

    paths = [url.split("?", 1)[0].rsplit("/", 1)[-1] for url in ask_requests]
    assert sorted(paths) == sorted([
        "ask_openai",
        "ask_mistral",
        "ask_claude",
        "ask_gemini",
        "ask_grok",
    ])
    assert "ask_deepseek" not in paths
    assert app_page.locator("#selectDeepSeek").is_checked() is False
    expect(app_page.locator("#deepseekResponse")).to_have_class(re.compile(r"\bexcluded\b"))


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
