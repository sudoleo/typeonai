/**
 * Die vier Anzeigezustaende einer belegten Aussage.
 *
 * Seit der Coverage-Judge JEDEN pruefbaren Satz beurteilt (2026-08-31), darf
 * eine duenn belegte Aussage nicht mehr verschwinden: unmarkiert saehe sie aus
 * wie ungepruefter Fliesstext. Sie bekommt deshalb eine eigene, leiseste Stufe
 * - und ausdruecklich KEINE Quote, weil "1/1" wie Einstimmigkeit liest.
 */

import { describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const SUPPORTED = "The tower is 330 m tall.";
const SPLIT = "A ticket costs 29 euros.";
const THIN = "The queue is shortest on Tuesday.";
const ANSWER = `${SUPPORTED} ${SPLIT} ${THIN}`;

const MODELS = ["OpenAI", "Gemini", "Anthropic"];

function data(claims, differences = []) {
  return { models_compared: MODELS, claims, differences };
}

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
      <div id="claimPopover" hidden></div>
      <div id="claimSheetBackdrop" hidden></div>
    `
  });
  window.Element.prototype.scrollIntoView = function () {};
  // jsdom kennt weder Scrollen noch Media Queries; beide fragt die Karte beim
  // Oeffnen ab.
  window.matchMedia = query => ({
    matches: false, media: query, addEventListener() {}, removeEventListener() {}
  });
  return {
    window,
    document,
    body: document.querySelector(".thread-history-answer-body"),
    fallback: document.querySelector(".consensus-claims-fallback")
  };
}

function render(ctx, payload) {
  ctx.window.renderStoredConsensusClaims(ctx.body, payload, ctx.fallback, []);
}

function badgeFor(ctx, text) {
  const mark = Array.from(ctx.body.querySelectorAll(".cx-claim"))
    .find(span => text.includes(span.textContent.trim()));
  return mark ? mark.nextElementSibling : null;
}

describe("claim coverage states", () => {
  it("marks a supported, a split and a thin sentence differently", () => {
    const ctx = boot();
    render(ctx, data([
      { anchor: SUPPORTED, agree: MODELS, dissent: [], coverage: "supported" },
      {
        anchor: SPLIT,
        agree: ["OpenAI", "Gemini"],
        dissent: [{ model: "Anthropic", quote: "22 euros" }],
        coverage: "split"
      },
      { anchor: THIN, agree: ["OpenAI"], dissent: [], coverage: "thin" }
    ]));

    expect(ctx.body.querySelectorAll(".cx-claim.is-unanimous").length).toBe(1);
    expect(ctx.body.querySelectorAll(".cx-claim.is-split").length).toBe(1);
    expect(ctx.body.querySelectorAll(".cx-claim.is-thin").length).toBe(1);
  });

  it("shows a dash instead of a ratio when too few models addressed it", () => {
    const ctx = boot();
    render(ctx, data([
      { anchor: THIN, agree: ["OpenAI"], dissent: [], coverage: "thin" }
    ]));

    const badge = ctx.body.querySelector(".claim-badge");
    expect(badge.classList.contains("is-thin")).toBe(true);
    // "1/1" laese sich wie einstimmige Zustimmung - genau das Gegenteil.
    expect(badge.textContent).not.toMatch(/\d\/\d/);
    expect(badge.textContent).toBe("–");
    expect(badge.getAttribute("aria-label")).toContain("Too few models addressed this");
  });

  it("keeps the ratio on a supported sentence", () => {
    const ctx = boot();
    render(ctx, data([
      { anchor: SUPPORTED, agree: MODELS, dissent: [], coverage: "supported" }
    ]));

    const badge = ctx.body.querySelector(".claim-badge");
    expect(badge.classList.contains("is-thin")).toBe(false);
    expect(badge.textContent).toBe("3/3");
  });

  it("derives the state from the counts for snapshots without the field", () => {
    const ctx = boot();
    render(ctx, data([
      { anchor: SUPPORTED, agree: ["OpenAI", "Gemini"], dissent: [] },
      { anchor: SPLIT, agree: ["OpenAI"], dissent: [{ model: "Gemini", quote: "35" }] }
    ]));

    expect(ctx.body.querySelectorAll(".cx-claim.is-unanimous").length).toBe(1);
    expect(ctx.body.querySelectorAll(".cx-claim.is-split").length).toBe(1);
    expect(ctx.body.querySelectorAll(".cx-claim.is-thin").length).toBe(0);
  });

  it("lets a contradiction override the thin mark on the same sentence", () => {
    const ctx = boot();
    render(ctx, data(
      [{ anchor: SPLIT, agree: ["OpenAI"], dissent: [], coverage: "thin" }],
      [{
        claim: "the ticket price",
        type: "contradiction",
        severity: "major",
        consensus_anchor: SPLIT,
        positions: [
          { models: ["OpenAI"], stance: "29 euros", quote: "29 euros" },
          { models: ["Gemini"], stance: "35 euros", quote: "35 euros" }
        ]
      }]
    ));

    expect(ctx.body.querySelector(".cx-claim.is-major")).not.toBe(null);
    expect(ctx.body.querySelector(".cx-claim.is-thin")).toBe(null);
    expect(ctx.body.querySelector(".claim-badge")).toBe(null);
  });

  it("says in the detail card that agreement is not proof", () => {
    const ctx = boot();
    render(ctx, data([
      { anchor: SUPPORTED, agree: MODELS, dissent: [], coverage: "supported" }
    ]));

    ctx.body.querySelector(".claim-badge")
      .dispatchEvent(new ctx.window.MouseEvent("click", { bubbles: true }));

    const note = ctx.document.querySelector("#claimPopover .claim-popover-note");
    expect(note).not.toBe(null);
    expect(note.textContent).toContain("not proof");
  });
});
