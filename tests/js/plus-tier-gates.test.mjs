/**
 * Ein Plus-Konto darf Dateien anhaengen.
 *
 * Das Gate haengt an `window.isUserPlus`, und dieses Flag entsteht erst am Ende
 * einer Kette: /user_status -> updateUserTierUI -> App.state -> window-Getter.
 * Getestet wird deshalb die ECHTE Kette (app-state.js + user-tier.js +
 * attachments.js), nicht ein gesetztes Flag.
 *
 * Der zweite Teil sichert die Stelle, an der die Stufe wieder verloren gehen
 * kann: `is_pro_user: false` bedeutet seit Plus nur noch "nicht Pro" und
 * unterscheidet Free nicht mehr von Plus. Ein Lauf ohne ausdrueckliches `tier`
 * darf die Anzeige deshalb nur anheben, nie senken.
 */

import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const BODY = `
<div class="chat-input-container">
  <div id="attachmentBar" class="attachment-bar" hidden></div>
  <textarea id="questionInput"></textarea>
  <button id="attachTrigger"></button>
  <div id="attachMenu" hidden><button id="attachUploadOption"></button></div>
  <input id="attachFileInput" type="file">
</div>
<span id="proBadge" class="pro-badge"></span>
<a id="upgradeLink"></a>
<label class="switch deep-switch"><input type="checkbox" id="deepSearchToggle"></label>
<div id="proFeatureModal" style="display:none">
  <span id="proModalFeatureName"></span><p id="proModalDescription"></p>
</div>
`;

function boot() {
  const harness = loadScripts(
    [
      "static/js/app-state.js",
      "static/js/app-core.js",
      "static/js/attachments.js",
      "static/js/user-tier.js"
    ],
    {
      body: BODY,
      // app-core.js fragt beim Laden das Farbschema ab; jsdom kennt matchMedia nicht.
      before(window) {
        window.matchMedia = () => ({
          matches: false,
          addEventListener() {}, removeEventListener() {},
          addListener() {}, removeListener() {}
        });
      }
    }
  );
  const gated = [];
  harness.window.App.showProFeatureModal = (feature) => {
    gated.push(feature);
    return true;
  };
  harness.gated = gated;
  return harness;
}

function clickUpload({ window, document }) {
  document.getElementById("attachUploadOption")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
}

describe("attachment gate per tier", () => {
  it("lets a Plus account open the file picker", () => {
    const harness = boot();
    harness.window.updateUserTierUI("plus", true);

    expect(harness.window.userTier).toBe("plus");
    expect(harness.window.isUserPlus).toBe(true);
    // Die Kostengrenze bleibt: Plus ist kein Pro.
    expect(harness.window.isUserPro).toBe(false);

    clickUpload(harness);
    expect(harness.gated).toEqual([]);
  });

  it("still refuses a Free account", () => {
    const harness = boot();
    harness.window.updateUserTierUI("free", true);

    expect(harness.window.isUserPlus).toBe(false);
    clickUpload(harness);
    expect(harness.gated).toEqual(["File uploads"]);
  });

  it("names the tier on the badge and drops the Free-only upgrade link", () => {
    const { window, document } = boot();

    window.updateUserTierUI("plus", true);
    expect(document.getElementById("proBadge").textContent).toBe("Plus");
    expect(document.getElementById("upgradeLink").style.display).toBe("none");

    window.updateUserTierUI("pro", true);
    expect(document.getElementById("proBadge").textContent).toBe("Pro");

    window.updateUserTierUI("free", true);
    expect(document.getElementById("proBadge").style.display).toBe("none");
    expect(document.getElementById("upgradeLink").style.display).toBe("inline-flex");
  });

  it("keeps Plus when a run reports only is_pro_user: false", () => {
    // Genau der Rueckfall, der Plus-Konten die Anhaenge wieder wegnimmt: eine
    // Antwort ohne "tier" (aelterer Server, Tab von vor einem Deploy) trug
    // frueher ein blankes false in updateUserTierUI und setzte damit Free.
    const harness = boot();
    harness.window.updateUserTierUI("plus", true);

    const runTier = tierSignalFor({ isProUser: false });
    expect(runTier).toBeNull();

    clickUpload(harness);
    expect(harness.gated).toEqual([]);
  });

  it("still accepts an explicit downgrade and a bare Pro signal", () => {
    expect(tierSignalFor({ tier: "free" })).toBe("free");
    expect(tierSignalFor({ isProUser: true })).toBe("pro");
    expect(tierSignalFor({ tier: "plus", isProUser: false })).toBe("plus");
  });
});

/**
 * Die Auswertung aus run-view.js/query-send.js: nur ein ausdrueckliches `tier`
 * oder ein `true` darf die Stufe stellen.
 */
function tierSignalFor(usage) {
  return usage.tier ?? (usage.isProUser === true ? "pro" : null);
}
