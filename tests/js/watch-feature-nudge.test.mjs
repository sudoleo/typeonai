import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

/**
 * Der Hinweis "Keep this answer current" haengt am Watch-Knopf im Fuss der
 * Antwort. Er lebt als eigener Layer direkt unter <body>: Wenn stattdessen
 * die Consensus-Sektion ueber den fixierten Composer gehoben wird, malt auch
 * der Antworttext ueber das Eingabefeld.
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
  // Der Hinweis erscheint erst ab der dritten abgeschlossenen Frage: jeder
  // Lauf meldet sich, die ersten beiden zaehlen nur.
  const run = harness => {
    harness.window.App.watch.showFeatureNudge();
    queued?.();
  };
  return {
    ...harness,
    run: () => run(harness),
    show: () => {
      run(harness);
      run(harness);
      run(harness);
    }
  };
}

describe("watch feature nudge", () => {
  it("haelt sich bis zur dritten Frage zurueck", () => {
    const { document, run } = boot();

    run();
    expect(document.getElementById("watchFeatureNudge")).toBeNull();
    run();
    expect(document.getElementById("watchFeatureNudge")).toBeNull();
    run();
    expect(document.getElementById("watchFeatureNudge")).not.toBeNull();
  });

  it("portaliert nur den Hinweis ueber den Composer, nie die Antwort", () => {
    const { document, show } = boot();
    show();

    const nudge = document.getElementById("watchFeatureNudge");
    expect(nudge).not.toBeNull();
    expect(nudge.parentElement).toBe(document.body);
    expect(document.querySelector(".watch-feature-anchor").classList.contains("has-feature-nudge"))
      .toBe(true);
    expect(document.querySelectorAll(".has-watch-feature-nudge").length).toBe(0);
    expect(nudge.style.top).not.toBe("");
    expect(nudge.style.left).not.toBe("");
  });

  it("nimmt die Markierung beim Schliessen wieder zurueck", () => {
    const { document, show } = boot();
    show();

    document.querySelector(".watch-feature-nudge-close").click();

    expect(document.getElementById("watchFeatureNudge")).toBeNull();
    expect(document.querySelector(".watch-feature-anchor").classList.contains("has-feature-nudge"))
      .toBe(false);
  });
});
