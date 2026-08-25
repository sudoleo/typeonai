import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

/**
 * Der Hinweis "Keep this answer current" haengt am Watch-Knopf im Fuss der
 * Antwort. Sein z-index endet an jedem Stacking-Context darueber, deshalb
 * markiert watch.js die Boxen, die mit hoch muessen. Genau diese Markierung
 * war still kaputt: sie suchte ein <h2>, seit dem Umzug der Aktionen in den
 * Fuss (#consensusFooterActions) gibt es das ueber dem Anker nicht mehr — und
 * der halbe Hinweis lag unter dem Composer.
 */
const BODY = `
  <div class="container">
    <div class="consensus-section">
      <div id="consensusOutput" class="consensus-output">
        <div class="consensus-box" id="consensusResponse">
          <div class="consensus-main">
            <h2>Consensus Answer</h2>
            <div id="consensusAnswerBody" class="consensus-answer-body"></div>
            <div id="runProvenance" class="run-provenance consensus-footer">
              <span class="consensus-footer-actions" id="consensusFooterActions">
                <span class="consensus-copy-inline"></span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="input-section"></div>
  </div>
`;

function boot() {
  let queued = null;
  const harness = loadScripts(
    ["static/js/watch-state.js", "static/js/watch.js"],
    {
      body: BODY,
      before(window) {
        // Der Hinweis kommt mit 650 ms Verzoegerung; hier wird er von Hand
        // ausgeloest, damit der Test nicht auf die Uhr wartet.
        window.setTimeout = fn => {
          queued = fn;
          return 7;
        };
        window.clearTimeout = () => {
          queued = null;
        };
        window.fetch = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
        window.auth = { currentUser: { uid: "nudge-user" } };
        window.lastShareResultId = "share-1";
        window.App = { watch: {} };
      }
    }
  );
  return { ...harness, show: () => {
    harness.window.App.watch.showFeatureNudge();
    queued?.();
  } };
}

describe("watch feature nudge", () => {
  it("hebt Fuss und Konsens-Sektion, solange der Hinweis offen ist", () => {
    const { document, show } = boot();
    show();

    const nudge = document.getElementById("watchFeatureNudge");
    expect(nudge).not.toBeNull();
    expect(nudge.closest(".watch-feature-anchor")).not.toBeNull();
    // Ohne diese beiden liegt der Hinweis unter der eigenen Antwort
    // (.run-provenance) bzw. unter dem Composer (.consensus-section).
    expect(
      document.getElementById("runProvenance").classList.contains("has-watch-feature-nudge")
    ).toBe(true);
    expect(
      document.querySelector(".consensus-section").classList.contains("has-watch-feature-nudge")
    ).toBe(true);
  });

  it("nimmt die Markierung beim Schliessen wieder zurueck", () => {
    const { document, show } = boot();
    show();

    document.querySelector(".watch-feature-nudge-close").click();

    expect(document.getElementById("watchFeatureNudge")).toBeNull();
    expect(document.querySelectorAll(".has-watch-feature-nudge").length).toBe(0);
  });
});
