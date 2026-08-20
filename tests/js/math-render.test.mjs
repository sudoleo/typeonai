/**
 * math-render.js -- der gemeinsame LaTeX-Pfad von Konsens, Key claims und
 * Share-Seiten.
 *
 * Zwei Dinge stehen hier auf dem Spiel, und beide sind still: Markdown frisst
 * die Escapes INNERHALB einer Formel ("17{,}5\%" -> "17{,}5%", und "%" ist in
 * TeX ein Kommentar, der den Rest der Zeile verschluckt), und dasselbe
 * Dollarzeichen traegt in denselben Antworten Betraege ("6,7 Mrd. $ in Q1").
 * Beides faellt visuell kaum auf und veraendert trotzdem, was dasteht.
 */

import { beforeEach, describe, expect, it } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

// Das macht marked mit dem vorbereiteten Text: ein Backslash vor einem
// ASCII-Satzzeichen ist ein Escape und verschwindet (CommonMark).
const MARKDOWN_ESCAPE_RE = /\\([!-/:-@[-`{-~])/g;
const throughMarkdown = (value) => value.replace(MARKDOWN_ESCAPE_RE, "$1");

function boot() {
  const { window, document } = loadScripts(["static/js/math-render.js"]);
  return { window, document, math: window.ConsensusMath };
}

describe("ConsensusMath.prepareMarkdown", () => {
  let math;
  beforeEach(() => {
    math = boot().math;
  });

  it("laesst die Escapes einer Formel den Markdown-Pass ueberleben", () => {
    const source = String.raw`Also \(6{,}7 / 5{,}7 - 1 \approx 17{,}5\%\).`;
    expect(throughMarkdown(math.prepareMarkdown(source))).toBe(source);
  });

  it("macht aus $...$ die von KaTeX erkannte Form", () => {
    const source = String.raw`Also $6{,}7 / 5{,}7 - 1 \approx 17{,}5\%$.`;
    expect(throughMarkdown(math.prepareMarkdown(source))).toBe(
      String.raw`Also \(6{,}7 / 5{,}7 - 1 \approx 17{,}5\%\).`
    );
  });

  it("laesst Betraege in Ruhe, auch wenn zwei davon nebeneinander stehen", () => {
    const source = "Anthropic erzielte 5,7 Mrd. $ in Q1 gegenueber 6,7 Mrd. $ in Q2.";
    expect(math.prepareMarkdown(source)).toBe(source);
  });

  it("haelt einen Preisvergleich fuer Text, keine Formel", () => {
    const source = "Der Preis fiel von $100 auf $80.";
    expect(math.prepareMarkdown(source)).toBe(source);
  });

  it("schuetzt Indizes vor der Kursivschrift", () => {
    const source = String.raw`Es gilt $x_1 + x_2 = y^2$.`;
    // Vor dem Markdown-Pass sind die Unterstriche escapt, danach stehen sie
    // wieder als Notation da - kursiv wird nichts.
    expect(math.prepareMarkdown(source)).toContain("x\\_1");
    expect(throughMarkdown(math.prepareMarkdown(source))).toBe(
      String.raw`Es gilt \(x_1 + x_2 = y^2\).`
    );
  });

  it("fasst Code nicht an", () => {
    const source = "Nutze `const cost = $total * 2` im Skript.";
    expect(math.prepareMarkdown(source)).toBe(source);
  });

  it("laesst $$-Bloecke unveraendert durch", () => {
    const source = "$$\\frac{a}{b}$$";
    expect(throughMarkdown(math.prepareMarkdown(source))).toBe(source);
  });

  it("behaelt ein einzelnes \\( ausserhalb einer Formel sichtbar", () => {
    expect(throughMarkdown(math.prepareMarkdown(String.raw`Rest \( ohne Ende`))).toBe(
      String.raw`Rest \( ohne Ende`
    );
  });
});

describe("ConsensusMath.stripMath", () => {
  let math;
  beforeEach(() => {
    math = boot().math;
  });

  it("liefert den Satz ohne Formel - so, wie ihn das DOM zeigt", () => {
    expect(
      math.stripMath(String.raw`Der Zuwachs betraegt \(17{,}5\%\) gegenueber Q1.`)
    ).toBe("Der Zuwachs betraegt   gegenueber Q1.");
  });

  it("erkennt auch die Dollar-Schreibweise", () => {
    expect(math.stripMath(String.raw`Der Zuwachs betraegt $17{,}5\%$.`)).toBe(
      "Der Zuwachs betraegt  ."
    );
  });

  it("meldet mit \"\", dass gar keine Formel drin war", () => {
    expect(math.stripMath("6,7 Mrd. $ in Q1 gegenueber 6,7 Mrd. $ in Q2.")).toBe("");
  });
});

describe("ConsensusMath.wrapBareLatex", () => {
  let math;
  beforeEach(() => {
    math = boot().math;
  });

  it("erkennt den Anker einer abgesetzten Formel aus einem alten Lauf", () => {
    expect(math.wrapBareLatex(String.raw`6{,}7 / 5{,}7 - 1 \approx 17{,}5\%`)).toBe(
      String.raw`\(6{,}7 / 5{,}7 - 1 \approx 17{,}5\%\)`
    );
  });

  it("laesst jeden Anker mit gewoehnlichem Text in Ruhe", () => {
    const claims = [
      "$ ARR = annualisiertes aktuelles Umsatztempo im spaeteren Zeitraum",
      "Die haeufig genannten mehr als 40 Mrd.",
      "Der Zuwachs betraegt 17,5 % gegenueber Q1.",
      String.raw`Also \(17{,}5\%\) mehr.`,
      "2026-08-20"
    ];
    claims.forEach((claim) => expect(math.wrapBareLatex(claim)).toBe(""));
  });
});

describe("ConsensusMath.render", () => {
  it("uebergibt auch $...$ als echten KaTeX-Ausdruck", () => {
    const { window, document, math } = boot();
    const seen = [];
    window.renderMathInElement = (root) => seen.push(root.textContent);
    const box = document.createElement("div");
    box.textContent = String.raw`Also $6{,}7 / 5{,}7 - 1 \approx 17{,}5\%$.`;
    document.body.appendChild(box);

    math.render(box);

    expect(seen).toEqual([
      String.raw`Also \(6{,}7 / 5{,}7 - 1 \approx 17{,}5\%\).`
    ]);
  });

  it("fasst Betraege im fertigen DOM nicht an", () => {
    const { window, document, math } = boot();
    window.renderMathInElement = () => {};
    const box = document.createElement("div");
    const text = "6,7 Mrd. $ in Q1 gegenueber 6,7 Mrd. $ in Q2.";
    box.textContent = text;
    document.body.appendChild(box);

    math.render(box);

    expect(box.textContent).toBe(text);
  });
});
