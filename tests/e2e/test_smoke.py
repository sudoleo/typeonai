"""Playwright-Smoke-Suite: automatisiert die risikoreichsten Punkte aus
docs/smoke-checklist.md gegen den Mock-Server (MOCK_LLM/MOCK_AUTH).

Bewusst NICHT abgedeckt (Stand der ersten Iteration): Resolve-Runde,
Share-Dialog, Attachments, Follow-up, Bookmarks, Agent-Mode-Timer,
Demo-Flow, Mobile-Layout - siehe tests/e2e/README.md.
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

    # Kurze Nachlaufzeit fuer asynchrone Init-Fehler (Tooltips, Usage-Fetch).
    app_page.wait_for_timeout(1500)
    errors = get_console_errors()
    assert errors == [], f"Konsolen-Fehler beim Laden: {errors}"


def test_send_question_streams_all_models(app_page):
    """Kern-Flow: Frage senden -> alle Modelle streamen (Zwischenzustand
    sichtbar) und rendern die finale Mock-Antwort."""
    _send_question(app_page)

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
    _send_question(app_page)
    _wait_for_all_final_answers(app_page)

    expect(app_page.locator("#consensusResponse")).to_contain_text("Mock consensus", timeout=30000)

    verdict = app_page.locator("#consensusVerdict")
    expect(verdict).to_be_visible(timeout=15000)
    expect(verdict).to_contain_text("/100")

    expect(app_page.locator(".claim-badge").first).to_be_visible(timeout=15000)
    expect(app_page.locator(".diff-card.is-contradiction").first).to_be_visible(timeout=15000)

    errors = get_console_errors()
    assert errors == [], f"Konsolen-Fehler im Consensus-Flow: {errors}"


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

    app_page.click("#modeToggle")
    app_page.wait_for_function(
        "(wasDark) => document.body.classList.contains('dark-mode') !== wasDark",
        arg=initially_dark,
        timeout=5000,
    )
    stored = app_page.evaluate("() => localStorage.getItem('theme')")
    assert stored in ("dark", "light")

    app_page.click("#modeToggle")
    app_page.wait_for_function(
        "(wasDark) => document.body.classList.contains('dark-mode') === wasDark",
        arg=initially_dark,
        timeout=5000,
    )


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
