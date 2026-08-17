/**
 * composer-quote.js -- "Ask about this".
 *
 * The Python contracts assert that the strings exist in the source. What
 * actually matters is the shape of the text that leaves the composer: the
 * typed question first, the passage below it, and a hard cap so a quote cannot
 * eat the run's word budget.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

const COMPOSER = `
  <div id="composerQuote" hidden>
    <span id="composerQuoteText"></span>
    <button id="composerQuoteRemove" type="button">remove</button>
  </div>
  <textarea id="questionInput"></textarea>
`;

function boot() {
  const { window, document } = loadScripts(["static/js/composer-quote.js"], {
    body: COMPOSER,
  });
  return { window, document, quote: window.App.quote };
}

describe("App.quote", () => {
  let ctx;
  beforeEach(() => {
    ctx = boot();
  });

  it("starts empty and hidden", () => {
    expect(ctx.quote.has()).toBe(false);
    expect(ctx.quote.text()).toBe("");
    expect(ctx.document.getElementById("composerQuote").hidden).toBe(true);
  });

  it("shows the passage in the composer once set", () => {
    ctx.quote.set("The models disagree about the timeline.");

    expect(ctx.quote.has()).toBe(true);
    expect(ctx.document.getElementById("composerQuote").hidden).toBe(false);
    expect(ctx.document.getElementById("composerQuoteText").textContent).toBe(
      "The models disagree about the timeline."
    );
  });

  it("collapses runs of spaces and caps blank lines at one", () => {
    ctx.quote.set("  one   two \r\n\r\n\r\n\r\n three  ");

    // Spaces adjacent to a line break survive: the normalizer collapses
    // horizontal whitespace and vertical whitespace separately, and the box
    // renders with textContent, where a trailing space is invisible.
    expect(ctx.quote.text()).toBe("one two \n\n three");
  });

  it("normalizes CRLF so a pasted passage does not carry \\r into the run", () => {
    ctx.quote.set("first\r\nsecond");

    expect(ctx.quote.text()).toBe("first\nsecond");
  });

  it("truncates beyond the cap with an ellipsis", () => {
    ctx.quote.set("x".repeat(2000));

    const text = ctx.quote.text();
    expect(text.length).toBe(1201); // 1200 chars + the ellipsis
    expect(text.endsWith("…")).toBe(true);
  });

  it("puts the typed question first and the passage below it", () => {
    ctx.quote.set("Only 3 of 6 models cited a source.");

    expect(ctx.quote.compose("Which ones?")).toBe(
      "Which ones?\n\nQuoted from the previous answer:\n“Only 3 of 6 models cited a source.”"
    );
  });

  it("asks for a comment when the user typed nothing", () => {
    ctx.quote.set("A claim without a source.");

    expect(ctx.quote.compose("")).toBe(
      "Please comment on this passage from the previous answer:\n“A claim without a source.”"
    );
  });

  it("returns the question untouched when there is no quote", () => {
    expect(ctx.quote.compose("Plain question")).toBe("Plain question");
    expect(ctx.quote.compose(undefined)).toBe("");
  });

  it("hides the box again on clear", () => {
    ctx.quote.set("something");
    ctx.quote.clear();

    expect(ctx.quote.has()).toBe(false);
    expect(ctx.document.getElementById("composerQuote").hidden).toBe(true);
  });

  it("clears through the remove button and hands focus back to the field", () => {
    ctx.quote.set("something");

    ctx.document.getElementById("composerQuoteRemove").click();

    expect(ctx.quote.has()).toBe(false);
    expect(ctx.document.activeElement.id).toBe("questionInput");
  });

  it("survives a page that has no composer markup at all", () => {
    const bare = loadScripts(["static/js/composer-quote.js"]);

    expect(() => bare.window.App.quote.set("text")).not.toThrow();
    expect(bare.window.App.quote.text()).toBe("text");
  });
});
