/**
 * app-state.js -- owner-enforced cross-module state.
 *
 * tests/test_phase6_architecture.py greps every static/*.js for direct writes
 * to these keys. That catches the syntax; it cannot show that the enforcement
 * actually holds. These do.
 */

import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

function boot(before) {
  return loadScripts(["static/js/app-state.js"], { before });
}

describe("App.state", () => {
  it("exposes every declared key with its initial value", () => {
    const { window } = boot();

    expect(window.App.state.get("lastQuestion")).toBe("");
    expect(window.App.state.get("isUserPro")).toBe(false);
    expect(window.App.state.get("currentEvidenceSources")).toEqual([]);
    expect(window.App.state.get("consensusCitationMeta")).toBeNull();
  });

  it("accepts a write from the owning module", () => {
    const { window } = boot();

    window.App.state.set("lastQuestion", "Why is the sky blue?", "run");

    expect(window.App.state.get("lastQuestion")).toBe("Why is the sky blue?");
    expect(window.lastQuestion).toBe("Why is the sky blue?");
  });

  it("rejects a write from any other module and leaves the value alone", () => {
    const { window } = boot();
    window.App.state.set("lastQuestion", "original", "run");

    expect(() =>
      window.App.state.set("lastQuestion", "hijacked", "consensus")
    ).toThrow(/belongs to run/);
    expect(window.App.state.get("lastQuestion")).toBe("original");
  });

  it("rejects unknown keys on both read and write", () => {
    const { window } = boot();

    expect(() => window.App.state.get("nope")).toThrow(/Unknown app state key/);
    expect(() => window.App.state.set("nope", 1, "run")).toThrow(
      /Unknown app state key/
    );
  });

  it("makes the window view read-only", () => {
    const { window } = boot();

    expect(() => {
      window.lastQuestion = "written directly";
    }).toThrow(/Direct write to window.lastQuestion is forbidden/);
    expect(window.lastQuestion).toBe("");
  });

  it("announces each change with key, owner and value", () => {
    const { window } = boot();
    const seen = [];
    window.document.addEventListener("app:state-change", (event) =>
      seen.push(event.detail)
    );

    window.App.state.set("isUserPro", true, "userTier");

    expect(seen).toEqual([{ key: "isUserPro", owner: "userTier", value: true }]);
  });

  it("adopts a value a previous script already put on window", () => {
    // app-bootstrap.js runs first and may have seeded a global before the
    // state module defines its accessor.
    const { window } = boot((w) => {
      w.isUserPro = true;
    });

    expect(window.App.state.get("isUserPro")).toBe(true);
  });

  it("keeps the definitions frozen against tampering", () => {
    const { window } = boot();

    expect(Object.isFrozen(window.App.state)).toBe(true);
    const before = window.App.state.definitions.lastQuestion.owner;
    try {
      window.App.state.definitions.lastQuestion.owner = "anyone";
    } catch {
      /* strict mode in the module */
    }
    expect(window.App.state.definitions.lastQuestion.owner).toBe(before);
  });
});
