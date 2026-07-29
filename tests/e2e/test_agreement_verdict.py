from playwright.sync_api import expect


def test_low_score_without_contradictions_is_not_green_or_high(app_page):
    app_page.evaluate(
        """() => window.renderConsensusInsights({
          models_compared: ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"],
          agreement: {score: 30, level: "hardly", model_count: 6},
          claims: [],
          differences: [{
            claim: "the main emphasis",
            type: "emphasis",
            severity: "minor",
            positions: [
              {models: ["OpenAI"], stance: "Focus A"},
              {models: ["Gemini"], stance: "Focus B"}
            ]
          }]
        }, 6)"""
    )

    verdict = app_page.locator("#consensusVerdict")
    expect(verdict).to_have_class("consensus-verdict is-alert")
    expect(verdict.locator(".verdict-headline")).to_have_text("Low agreement")
    expect(verdict.locator(".verdict-detail")).to_contain_text("no contradictions")
    assert verdict.evaluate(
        "(element) => getComputedStyle(element).getPropertyValue('--verdict-ring').trim()"
    ) == "#cf5a4a"
