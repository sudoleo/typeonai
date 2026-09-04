import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const STORAGE_KEY = "consensio.showConsensusMarkers.v1";
const CLAIM = "The launch date is September 3.";

function boot(stored = null) {
  return loadScripts([
    "static/js/consensus-anchor.js",
    "static/js/consensus-insights.js"
  ], {
    body: `
      <div id="consensusAnswerBody" class="consensus-answer-body"><p>${CLAIM}</p></div>
      <div id="consensusClaimsFallback" class="consensus-claims-fallback" hidden></div>
      <p id="consensusMarkerLegend" class="consensus-marker-legend" hidden>
        <span class="consensus-marker-legend-copy">Every sentence was checked.</span>
        <button id="consensusMarkerToggle" type="button">Hide checks</button>
      </p>
      <div id="differencesCards"></div>
      <div id="claimPopover" hidden></div>
      <div id="claimSheetBackdrop" hidden></div>
    `,
    before(window) {
      if (stored !== null) window.localStorage.setItem(STORAGE_KEY, stored);
      window.App = window.App || {};
      window.App.consensusBodyEl = () => window.document.getElementById("consensusAnswerBody");
      window.matchMedia = query => ({
        matches: false, media: query, addEventListener() {}, removeEventListener() {}
      });
    }
  });
}

function renderContradiction(window) {
  window.renderConsensusInsights({
    models_compared: ["OpenAI", "Gemini"],
    claims: [],
    differences: [{
      claim: "the launch date",
      consensus_anchor: CLAIM,
      type: "contradiction",
      severity: "major",
      positions: [
        { models: ["OpenAI"], stance: "September 3", quote: "September 3" },
        { models: ["Gemini"], stance: "September 5", quote: "September 5" }
      ]
    }]
  }, 2);
}

describe("consensus sentence-check visibility", () => {
  it("uses an unobtrusive persistent toggle without removing the analysis", () => {
    const { window, document } = boot();
    renderContradiction(window);

    const toggle = document.getElementById("consensusMarkerToggle");
    const mark = document.querySelector("#consensusAnswerBody .cx-claim");
    expect(mark.getAttribute("role")).toBe("button");
    expect(toggle.textContent).toBe("Hide checks");

    toggle.click();
    expect(document.body.classList.contains("consensus-markers-hidden")).toBe(true);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("false");
    expect(toggle.textContent).toBe("Show checks");
    expect(mark.hasAttribute("role")).toBe(false);
    expect(document.querySelectorAll("#differencesCards .diff-card").length).toBe(1);

    toggle.click();
    expect(document.body.classList.contains("consensus-markers-hidden")).toBe(false);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("true");
    expect(mark.getAttribute("role")).toBe("button");
  });

  it("restores the hidden preference before a new answer is rendered", () => {
    const { window, document } = boot("false");
    expect(document.body.classList.contains("consensus-markers-hidden")).toBe(true);

    renderContradiction(window);
    const mark = document.querySelector("#consensusAnswerBody .cx-claim");
    expect(mark).not.toBe(null);
    expect(mark.hasAttribute("role")).toBe(false);
    expect(document.getElementById("consensusMarkerToggle").textContent).toBe("Show checks");
  });
});
