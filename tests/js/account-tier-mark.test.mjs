/**
 * Die Kontostufe am Konto-Kuerzel.
 *
 * Pro und Plus tragen keine eigene Flaeche im Shell: der Kreis mit dem
 * Kuerzel wechselt die Farbe, ausgeschrieben steht die Stufe im Popup. Zwei
 * Dinge muessen dabei stimmen, und beide sind schon einmal schiefgegangen:
 *
 *   1. Plus darf nicht als Free erscheinen. "is_pro" ist fuer Plus false, ein
 *      Aufrufer, der nur das Flag durchreicht, stuft Plus herunter.
 *   2. Die Marke gehoert dem KONTO, nicht dem sichtbaren Lauf. run-view.js
 *      ruft updateUserTierUI mit der Stufe des geoeffneten Laufs auf -- ein
 *      alter Free-Lauf darf dem Pro-Konto seine Marke nicht abnehmen.
 */

import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const BODY = `
<div id="loginContainer">
  <span id="emailIcon" class="email-icon">M</span>
  <div id="emailPopup" hidden>
    <span class="user-email">
      <span class="user-email-address">max@example.com</span>
      <span id="accountTierLabel" class="pro-badge is-subtle account-tier-label" hidden></span>
    </span>
  </div>
</div>
<span id="proBadge" class="pro-badge"></span>
<a id="upgradeLink"></a>
<label class="switch deep-switch"><input type="checkbox" id="deepSearchToggle"></label>
`;

function boot() {
  return loadScripts(["static/js/app-state.js", "static/js/user-tier.js"], {
    body: BODY
  });
}

function mark({ document }) {
  const icon = document.getElementById("emailIcon");
  const label = document.getElementById("accountTierLabel");
  return {
    pro: icon.classList.contains("is-pro"),
    plus: icon.classList.contains("is-plus"),
    labelHidden: label.hidden,
    labelText: label.textContent
  };
}

describe("account tier mark", () => {
  it("marks Pro in gold and names it in the popup", () => {
    const harness = boot();
    harness.window.App.accountTier.set("pro");
    expect(mark(harness)).toEqual({
      pro: true,
      plus: false,
      labelHidden: false,
      labelText: "Pro"
    });
  });

  it("marks Plus without gold and names it too", () => {
    const harness = boot();
    harness.window.App.accountTier.set("plus");
    const state = mark(harness);
    expect(state.pro).toBe(false);
    expect(state.plus).toBe(true);
    expect(state.labelHidden).toBe(false);
    expect(state.labelText).toBe("Plus");
    expect(
      harness.document.getElementById("accountTierLabel").classList.contains("is-plus")
    ).toBe(true);
  });

  it("leaves Free unmarked", () => {
    const harness = boot();
    harness.window.App.accountTier.set("free");
    expect(mark(harness)).toEqual({
      pro: false,
      plus: false,
      labelHidden: true,
      labelText: ""
    });
  });

  it("keeps the mark when a Free run is opened", () => {
    const harness = boot();
    harness.window.App.accountTier.set("pro");
    // Das macht run-view.js beim Projizieren eines gespeicherten Laufs.
    harness.window.updateUserTierUI("free", true);
    expect(harness.window.userTier).toBe("free");
    expect(harness.window.accountTier).toBe("pro");
    expect(mark(harness).pro).toBe(true);
  });

  it("re-applies the mark after the account footer is re-rendered", () => {
    const harness = boot();
    harness.window.App.accountTier.set("pro");
    // firebase.js schreibt #loginContainer bei jedem Login neu.
    harness.document.getElementById("loginContainer").innerHTML =
      '<span id="emailIcon" class="email-icon">M</span>';
    expect(harness.document.getElementById("emailIcon").classList.contains("is-pro")).toBe(false);
    harness.window.App.accountTier.render();
    expect(harness.document.getElementById("emailIcon").classList.contains("is-pro")).toBe(true);
  });
});
