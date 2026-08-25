import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

/**
 * Das Dashboard hat "Changed" aus `point.changed` gelesen. Dieses Flag setzt
 * der Change-Judge auch fuer eine umformulierte Einschraenkung, also stand auf
 * jeder Karte nach jedem Lauf "Changed" — und damit auf keiner mehr etwas.
 * Die Bewertung faellt jetzt einmal auf dem Server (drift_signal) und kommt
 * als `trigger` an; die Karte liest nur noch die.
 */
function boot() {
  const harness = loadScripts(
    ["static/js/watch-state.js", "static/js/watch.js"],
    {
      before(window) {
        window.fetch = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
        window.auth = { currentUser: { uid: "drift-user" } };
        window.App = { watch: {} };
      }
    }
  );
  return harness.window.App.watch.driftState;
}

function watchWith(point) {
  return { history: [{ ts: "2026-08-20T08:00:00+00:00", ...point }] };
}

describe("driftState", () => {
  it("reads the server verdict instead of the raw changed flag", () => {
    const driftState = boot();

    const restated = driftState(watchWith({
      agreement_score: 82,
      changed: true,
      severity: "minor",
      trigger: "stable",
      restated: true,
      change_summary: "A qualification was rephrased."
    }));

    expect(restated.key).toBe("stable");
    expect(restated.label).toBe("Stable");
    expect(restated.summary).toContain("Restated, not moved");
  });

  it("still announces a check the judge graded as material", () => {
    const driftState = boot();

    const moved = driftState(watchWith({
      agreement_score: 64,
      changed: true,
      severity: "major",
      trigger: "changed",
      change_summary: "The recommendation flipped."
    }));

    expect(moved.key).toBe("changed");
    expect(moved.summary).toBe("The recommendation flipped.");
  });
});
