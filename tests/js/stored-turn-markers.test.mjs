/**
 * Inline-Marken eines archivierten Turns.
 *
 * Sobald eine Follow-up-Frage gestellt wird, rutscht die fertige Antwort in
 * #threadHistory und wird aus differences_data neu verankert. Der Widerspruch
 * muss diesen Umzug ueberleben: bis 2026-08-22 bekam der strittige Satz dort
 * nur noch sein Claim-Badge ("1 of 6 models support this", bernstein) und las
 * sich damit als blosse Stuetzungsquote statt als roter Widerspruch.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const ANSWER = "The tower is 330 m tall. A ticket costs 29 euros.";
const DISPUTED = "A ticket costs 29 euros.";

const DIFFERENCES_DATA = {
  models_compared: ["OpenAI", "Gemini", "Anthropic"],
  claims: [
    {
      anchor: DISPUTED,
      agree: [{ model: "OpenAI", quote: "29 euros" }],
      dissent: [
        { model: "Gemini", quote: "35 euros" },
        { model: "Anthropic", quote: "22 euros" }
      ]
    }
  ],
  differences: [
    {
      claim: "the ticket price",
      type: "contradiction",
      severity: "major",
      consensus_anchor: DISPUTED,
      positions: [
        { models: ["OpenAI"], stance: "29 euros", quote: "29 euros" },
        { models: ["Gemini", "Anthropic"], stance: "more than 29 euros", quote: "35 euros" }
      ]
    }
  ]
};

function boot() {
  const { window, document } = loadScripts([
    "static/js/consensus-anchor.js",
    "static/js/consensus-insights.js"
  ], {
    body: `
      <article class="thread-history-turn" data-turn-id="t1">
        <div class="consensus-answer-body thread-history-answer-body"><p>${ANSWER}</p></div>
        <div class="consensus-claims-fallback" hidden></div>
        <div class="thread-history-panels"></div>
      </article>
      <div id="differencesCards"></div>
    `
  });
  // jsdom kennt kein Scrollen; die Marke ruft es beim Oeffnen der Karte auf.
  window.Element.prototype.scrollIntoView = function () {};
  const body = document.querySelector(".thread-history-answer-body");
  const fallback = document.querySelector(".consensus-claims-fallback");
  return { window, document, body, fallback };
}

// Die Karten baut consensus-run.js erst NACH dem Verankern in die Schublade
// des Turns -- genau diese Reihenfolge bildet der Helfer nach.
function appendStoredCards(window, document, data) {
  const turn = document.querySelector(".thread-history-turn");
  const panel = document.createElement("div");
  panel.className = "thread-history-panel";
  panel.id = "threadHistoryPanel-1";
  panel.hidden = true;
  const tab = document.createElement("button");
  tab.className = "consensus-tab";
  tab.setAttribute("aria-expanded", "false");
  tab.setAttribute("aria-controls", panel.id);
  const cards = document.createElement("div");
  cards.className = "differences-cards thread-history-differences";
  window.renderStoredDifferenceCards(cards, data);
  panel.appendChild(cards);
  turn.querySelector(".thread-history-panels").append(tab, panel);
  return { tab, panel, cards };
}

describe("renderStoredConsensusClaims", () => {
  let ctx;
  beforeEach(() => {
    ctx = boot();
    ctx.window.renderStoredConsensusClaims(ctx.body, DIFFERENCES_DATA, ctx.fallback, []);
  });

  it("keeps the contradiction line on the disputed sentence", () => {
    const marks = ctx.body.querySelectorAll(".cx-claim.is-major");

    expect(marks.length).toBeGreaterThan(0);
    expect(marks[0].textContent).toContain("ticket costs 29 euros");
  });

  it("does not downgrade the contradiction to a support ratio", () => {
    expect(ctx.body.querySelector(".claim-badge")).toBe(null);
    expect(ctx.body.textContent).not.toContain("support this");
    expect(ctx.body.querySelector("[role='button']").getAttribute("aria-label"))
      .toContain("contradict");
  });

  it("opens the difference card of its own turn, not the live footer", () => {
    const { tab, panel, cards } = appendStoredCards(ctx.window, ctx.document, DIFFERENCES_DATA);
    const liveCards = ctx.document.getElementById("differencesCards");

    ctx.body.querySelector(".cx-claim.is-major")
      .dispatchEvent(new ctx.window.MouseEvent("click", { bubbles: true }));

    expect(panel.hidden).toBe(false);
    expect(tab.getAttribute("aria-expanded")).toBe("true");
    expect(cards.querySelector(".diff-card").classList.contains("is-focused")).toBe(true);
    expect(liveCards.querySelector(".is-focused")).toBe(null);
  });
});

describe("renderStoredConsensusClaims without differences", () => {
  it("still shows the support ratio for a merely split claim", () => {
    const ctx = boot();

    ctx.window.renderStoredConsensusClaims(
      ctx.body,
      { ...DIFFERENCES_DATA, differences: [] },
      ctx.fallback,
      []
    );

    const badge = ctx.body.querySelector(".claim-badge");
    expect(badge).not.toBe(null);
    expect(badge.getAttribute("aria-label")).toContain("1 of 3 models support this");
  });
});
