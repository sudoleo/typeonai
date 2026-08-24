import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

// The panel is a projection of the selected run. When nothing is selected --
// a saved bookmark was opened while a run keeps going -- it has to fall back
// to the controls instead of reading a run that is not there.
const BODY = `
  <div id="agentModePanel">
    <span id="agentModeTitle"></span>
    <span id="agentModeCount"></span>
    <span id="agentModeStatus"></span>
    <span id="agentModeTimer"></span>
    <div id="agentModeModels"></div>
  </div>
  <input type="checkbox" id="openaiCheck" checked>
  <select id="openaiModelSelect"><option value="gpt" data-model-label="GPT">GPT</option></select>
  <span id="openaiModelText">GPT</span>
  <div id="openaiResponse" class="response-box"><div class="collapsible-content"></div></div>
`;

function boot() {
  return loadScripts(["static/js/agent-mode.js"], {
    body: BODY,
    before(window) {
      window.App = {
        modelPrefs: [{
          key: "OpenAI",
          label: "OpenAI",
          checkId: "openaiCheck",
          selectId: "openaiModelSelect",
          textId: "openaiModelText",
          responseId: "openaiResponse"
        }],
        deepThinkModelLabels: {},
        getModelOptionLabel: option => option?.textContent || "",
        getSelectedModelCount: () => 1,
        initCustomModelPicker: vi.fn(),
        trackAppEvent: vi.fn()
      };
      window.localStorage.setItem("agentMode", "true");
    }
  });
}

describe("agent mode panel projection", () => {
  it("falls back to the controls when no run is selected", () => {
    const { window, document, dom } = boot();

    window.projectAgentModeRun({
      runId: "run-1",
      status: "running",
      phase: "answers",
      startedAt: Date.now(),
      config: { agentMode: true, providers: [{ provider: "OpenAI", modelLabel: "Frozen model" }] },
      modelResults: { OpenAI: { status: "streaming", streamText: "half" } }
    });
    expect(document.getElementById("agentModeModels").textContent).toContain("Frozen model");
    expect(document.body.classList.contains("agent-mode-running")).toBe(true);

    // Deselecting the run must not throw: everything after this call in
    // run-view's projection (the guided-run block, the send button) would
    // otherwise be skipped, and the bookmark restore that triggered it would
    // abort halfway through.
    expect(() => window.projectAgentModeRun(null)).not.toThrow();
    expect(document.getElementById("agentModeModels").textContent).not.toContain("Frozen model");
    expect(document.getElementById("agentModeModels").textContent).toContain("OpenAI");
    expect(document.body.classList.contains("agent-mode-running")).toBe(false);
    dom.window.close();
  });
});
