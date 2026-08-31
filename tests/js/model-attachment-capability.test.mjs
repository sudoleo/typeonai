import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

function boot(family) {
  return loadScripts(["static/js/app-core.js"], {
    before(window) {
      window.matchMedia = () => ({ matches: false });
      window.MODEL_FAMILIES = [family];
      window.MAX_RUN_FAMILIES = 6;
    },
  }).window;
}

describe("model-level attachment capability", () => {
  it("distinguishes GLM Flash from the text-only Pro model", () => {
    const window = boot({
      provider: "glm",
      key: "GLM",
      attachmentModels: ["glm-5.3-flash"],
      deepThinkModel: "glm-5.3",
    });
    const pref = window.App.modelPrefs[0];

    expect(window.App.modelAcceptsAttachments(pref, "glm-5.3-flash", false)).toBe(true);
    expect(window.App.modelAcceptsAttachments(pref, "glm-5.3", false)).toBe(false);
    expect(window.App.modelAcceptsAttachments(pref, "glm-5.3-flash", true)).toBe(false);
  });

  it("keeps the legacy provider boolean as a cached-client fallback", () => {
    const window = boot({ provider: "deepseek", key: "DeepSeek", handlesAttachments: false });
    const pref = window.App.modelPrefs[0];

    expect(window.App.modelAcceptsAttachments(pref, "deepseek-v4", false)).toBe(false);
  });
});
