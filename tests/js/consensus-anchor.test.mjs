/**
 * consensus-anchor.js -- deterministic text anchoring for the claim markers.
 *
 * This is the module the claim ledger stands on: an anchor that misses its
 * passage means a highlighted sentence lands in the wrong place, or nowhere.
 * It is pure text/DOM work with no network, so it can be exercised directly.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

// The real caller passes the markdown stripper; the anchoring logic under test
// only needs it to be a function that returns plain text.
const stripMarkdown = (value) =>
  String(value || "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/[*_`]/g, "");

function boot(extraScripts = []) {
  const { window, document } = loadScripts([
    "static/js/consensus-anchor.js",
    ...extraScripts
  ]);
  return { window, document, anchor: window.App.consensusAnchor.create(stripMarkdown) };
}

describe("consensusAnchor.normalizeForSearch", () => {
  let anchor;
  beforeEach(() => {
    anchor = boot().anchor;
  });

  it("folds case, curly quotes and runs of whitespace", () => {
    expect(anchor.normalizeForSearch('  The  “Tower”\n is\t330 m  ')).toBe(
      'the "tower" is 330 m'
    );
  });

  it("treats every quote flavour as the same character", () => {
    const variants = ['„x"', "‘x’", "«x»", '"x"'];
    const normalized = variants.map((v) => anchor.normalizeForSearch(v));

    expect(new Set(normalized).size).toBe(1);
  });

  it("survives null and undefined", () => {
    expect(anchor.normalizeForSearch(null)).toBe("");
    expect(anchor.normalizeForSearch(undefined)).toBe("");
  });
});

describe("consensusAnchor.findRangesInText", () => {
  let anchor;
  beforeEach(() => {
    anchor = boot().anchor;
  });

  it("maps a normalized needle back onto the raw offsets", () => {
    const raw = "The  tower   is 330 m tall.";
    const needle = anchor.normalizeForSearch("tower is 330 m");

    const range = anchor.findRangeInText(raw, needle);

    expect(raw.slice(range.start, range.end)).toBe("tower   is 330 m");
  });

  it("finds every non-overlapping occurrence", () => {
    const raw = "yes and yes and yes";

    const ranges = anchor.findRangesInText(raw, "yes");

    expect(ranges).toHaveLength(3);
    expect(ranges.map((r) => raw.slice(r.start, r.end))).toEqual([
      "yes",
      "yes",
      "yes",
    ]);
  });

  it("returns nothing for an empty needle instead of matching everywhere", () => {
    expect(anchor.findRangesInText("any text", "")).toEqual([]);
    expect(anchor.findRangeInText("any text", "")).toBeNull();
  });

  it("returns null when the passage is not there", () => {
    expect(anchor.findRangeInText("one two three", "four")).toBeNull();
  });

  it("matches across a line break, because the source wrapped mid-sentence", () => {
    const raw = "the tower is\n330 m tall";

    const range = anchor.findRangeInText(raw, "tower is 330 m");

    expect(range).not.toBeNull();
    expect(raw.slice(range.start, range.end)).toBe("tower is\n330 m");
  });
});

describe("consensusAnchor.searchVariants", () => {
  let anchor;
  beforeEach(() => {
    anchor = boot().anchor;
  });

  it("offers a tag-free variant, because the rendered DOM has no [S1]", () => {
    const variants = anchor.searchVariants("The tower is 330 m tall.[S1]");

    expect(variants.some((v) => v.includes("[s1]"))).toBe(true);
    expect(variants).toContain("the tower is 330 m tall.");
  });

  it("pulls the punctuation back to the word when the tag is dropped", () => {
    // Old bookmarks render as "statement[S1]." -- the space left behind by the
    // removed tag must not survive, or the anchor misses again.
    const variants = anchor.searchVariants("A statement [S1].");

    expect(variants).toContain("a statement.");
  });

  it("ranks full passages ahead of the eight-word fallbacks", () => {
    const long = "one two three four five six seven eight nine ten";

    const variants = anchor.searchVariants(long);
    const full = variants.indexOf(long);
    const short = variants.indexOf("one two three four five six seven eight");

    expect(full).toBeGreaterThanOrEqual(0);
    expect(short).toBeGreaterThan(full);
  });

  it("does not shorten a passage that is already short", () => {
    expect(anchor.searchVariants("one two three")).toEqual(["one two three"]);
  });

  it("strips leading and trailing ellipses from a clipped quote", () => {
    expect(anchor.searchVariants("…the middle of a sentence…")).toContain(
      "the middle of a sentence"
    );
  });

  it("emits no duplicates when markup makes the variants identical", () => {
    const variants = anchor.searchVariants("plain sentence");

    expect(new Set(variants).size).toBe(variants.length);
  });
});

describe("consensusAnchor.locateAnchor", () => {
  it("finds the passage inside a rendered answer and skips the source chips", () => {
    const { window, document, anchor } = boot();
    const root = document.createElement("div");
    root.innerHTML =
      "<p>The tower is 330 m tall<sup class=\"src-ref\">1</sup> today.</p>" +
      "<p>Unrelated paragraph.</p>";
    document.body.appendChild(root);

    const hit = anchor.locateAnchor(root, "The tower is 330 m tall today.");

    expect(hit).not.toBeNull();
    // The "1" of the source chip is not part of the flat view, so the anchor
    // matches straight across it -- that is the whole point of MARK_SKIP.
    expect(hit.flat).toBe("The tower is 330 m tall today.");
    expect(hit.flat.slice(hit.start, hit.end)).toBe(
      "The tower is 330 m tall today."
    );
    expect(hit.block.tagName).toBe("P");
    expect(window.App.consensusAnchor).toBeDefined();
  });

  it("finds a sentence whose formula is already rendered by KaTeX", () => {
    // The stored anchor carries the LaTeX SOURCE, the answer shows a .katex
    // block whose text the flat view skips. Without a formula-free variant
    // such a claim can never be marked -- it lands in the Key claims list
    // with its raw LaTeX on display.
    const { document, anchor } = boot(["static/js/math-render.js"]);
    const root = document.createElement("div");
    root.innerHTML =
      "<p>Der Zuwachs betraegt <span class=\"katex\">17,5 %</span> gegenueber Q1.</p>";
    document.body.appendChild(root);

    const hit = anchor.locateAnchor(
      root,
      String.raw`Der Zuwachs betraegt \(17{,}5\%\) gegenueber Q1.`
    );

    expect(hit).not.toBeNull();
    expect(hit.flat.slice(hit.start, hit.end)).toBe(
      "Der Zuwachs betraegt  gegenueber Q1."
    );
  });

  it("returns null rather than guessing when the passage is absent", () => {
    const { document, anchor } = boot();
    const root = document.createElement("div");
    root.innerHTML = "<p>Nothing relevant here.</p>";
    document.body.appendChild(root);

    expect(anchor.locateAnchor(root, "A claim that was never made.")).toBeNull();
  });
});

describe("consensusAnchor.sentenceBounds", () => {
  it("does not treat currency abbreviations as sentence endings", () => {
    const { anchor } = boot();
    const text =
      "Der Umsatz stieg von 5,7 Mrd. $ in Q1 auf 6,7 Mrd. $ in Q2. Danach blieb er stabil.";
    const start = text.indexOf("$ in Q1");
    const end = text.indexOf("Mrd. $ in Q2") + "Mrd.".length;

    const bounds = anchor.sentenceBounds(text, start, end);

    expect(text.slice(bounds.start, bounds.end)).toBe(
      "Der Umsatz stieg von 5,7 Mrd. $ in Q1 auf 6,7 Mrd. $ in Q2."
    );
  });

  it("still recognizes a quantity abbreviation at a real sentence end", () => {
    const { anchor } = boot();
    const text = "Der Umsatz liegt bei 40 Mio. Danach steigt die Prognose weiter.";
    const start = text.indexOf("Umsatz");
    const end = text.indexOf("Mio.") + "Mio.".length;

    const bounds = anchor.sentenceBounds(text, start, end);

    expect(text.slice(bounds.start, bounds.end)).toBe(
      "Der Umsatz liegt bei 40 Mio."
    );
  });
});
