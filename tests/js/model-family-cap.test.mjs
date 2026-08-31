import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

// Vier Familien in der Konfiguration, aber ein Lauf vergleicht hoechstens drei:
// genau die Lage, die es mit Kimi/GLM geben wird.
const FAMILIES = [
  { key: "OpenAI", label: "ChatGPT", dom: "openai" },
  { key: "Mistral", label: "Mistral", dom: "mistral" },
  { key: "Anthropic", label: "Claude", dom: "claude" },
  { key: "Grok", label: "Grok", dom: "grok" }
];

const BODY = FAMILIES.map(family => `
  <label for="select${family.key}"><input type="checkbox" id="select${family.key}"></label>
  <div class="response-box" id="${family.dom}Response"><div class="collapsible-content"></div></div>
  <span id="${family.dom}ModelText"></span>
  <select id="${family.dom}ModelSelect"><option value="m-${family.dom}">M</option></select>
`).join("");

function boot({ cap = 3, checked = [] } = {}) {
  const popups = [];
  const harness = loadScripts(["static/js/model-picker.js"], {
    body: BODY,
    before(window) {
      window.App = {
        maxRunFamilies: cap,
        modelPrefs: FAMILIES.map(family => ({
          key: family.key,
          label: family.label,
          checkId: `select${family.key}`,
          selectId: `${family.dom}ModelSelect`,
          responseId: `${family.dom}Response`,
          textId: `${family.dom}ModelText`
        })),
        getModelOptionLabel: option => option?.textContent || "",
        getSelectedModelCount: () => 0,
        trackAppEvent: vi.fn(),
        showPopup: message => popups.push(message)
      };
      window.updateAgentModeUI = vi.fn();
      window.updateConsensusButtonAvailability = vi.fn();
    }
  });
  const { window } = harness;
  checked.forEach(key => {
    window.document.getElementById(`select${key}`).checked = true;
  });
  return { window, popups };
}

function isChecked(window, key) {
  return window.document.getElementById(`select${key}`).checked;
}

describe("run family cap", () => {
  it("includes a family while the run still has room", () => {
    const { window, popups } = boot({ checked: ["OpenAI", "Mistral"] });

    window.App.setModelSelectionState("claudeResponse", true, { persist: true });

    expect(isChecked(window, "Anthropic")).toBe(true);
    expect(popups).toEqual([]);
  });

  it("refuses the family beyond the cap instead of dropping another one", () => {
    const { window, popups } = boot({ checked: ["OpenAI", "Mistral", "Anthropic"] });

    window.App.setModelSelectionState("grokResponse", true, { persist: true });

    expect(isChecked(window, "Grok")).toBe(false);
    // Die bestehende Auswahl bleibt unangetastet -- kein stiller Tausch.
    expect(["OpenAI", "Mistral", "Anthropic"].every(key => isChecked(window, key))).toBe(true);
    expect(popups.join(" ")).toContain("up to 3 models");
  });

  it("applies the cap to restored selections without shouting at the user", () => {
    const { window, popups } = boot({ checked: ["OpenAI", "Mistral", "Anthropic"] });

    window.App.setModelSelectionState("grokResponse", true, { persist: false });

    expect(isChecked(window, "Grok")).toBe(false);
    expect(popups).toEqual([]);
  });

  it("frees a slot again when a family is left out", () => {
    const { window } = boot({ checked: ["OpenAI", "Mistral", "Anthropic"] });

    window.App.setModelSelectionState("mistralResponse", false, { persist: true });
    window.App.setModelSelectionState("grokResponse", true, { persist: true });

    expect(isChecked(window, "Mistral")).toBe(false);
    expect(isChecked(window, "Grok")).toBe(true);
  });
});
