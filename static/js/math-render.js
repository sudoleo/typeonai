// =====================================================================
// math-render.js
// Gemeinsame LaTeX-Darstellung fuer App- und Share-Seiten.
// KaTeX + auto-render werden davor aus dem CDN geladen.
// Export: window.ConsensusMath.{prepareMarkdown,render}.
// =====================================================================

(function initConsensusMath() {
  "use strict";

  const delimiters = [
    { left: "$$", right: "$$", display: true },
    { left: "\\(", right: "\\)", display: false },
    { left: "\\begin{equation}", right: "\\end{equation}", display: true },
    { left: "\\begin{align}", right: "\\end{align}", display: true },
    { left: "\\begin{alignat}", right: "\\end{alignat}", display: true },
    { left: "\\begin{gather}", right: "\\end{gather}", display: true },
    { left: "\\begin{CD}", right: "\\end{CD}", display: true },
    { left: "\\[", right: "\\]", display: true }
  ];

  // Zusammenhaengende Formelabschnitte in den von KaTeX erkannten Formen.
  const MATH_SPAN_RE = new RegExp(
    "\\$\\$[\\s\\S]*?\\$\\$"
    + "|\\\\\\[[\\s\\S]*?\\\\\\]"
    + "|\\\\\\([\\s\\S]*?\\\\\\)"
    + "|\\\\begin\\{(equation|align|alignat|gather|CD)\\*?\\}"
    + "[\\s\\S]*?\\\\end\\{\\1\\*?\\}",
    "g"
  );

  // Inline-Math als "$...$" ist die Schreibweise, die die Modelle am
  // haeufigsten liefern - KaTeX kennt sie ab Werk NICHT, und genau dasselbe
  // Zeichen steht in denselben Antworten als Waehrung ("6,7 Mrd. $ in Q1
  // gegenueber 6,7 Mrd. $ in Q2"). Wer "$" einfach als Trennzeichen
  // registriert, macht aus dem Text zwischen zwei Betraegen eine Formel.
  // Erkannt wird deshalb nur ein Paar, das eindeutig Formel ist: der Inhalt
  // liegt ohne Leerzeichen an beiden Dollarzeichen an, bleibt in einer Zeile,
  // ist kurz und enthaelt mindestens ein LaTeX-Signal (\befehl, ^, _, {}).
  const DOLLAR_INLINE_RE =
    /(^|[^\\$\w])\$(?!\s)((?:[^$\n\\]|\\[^\n])+?)(?<![\s\\])\$(?![\w$])/g;
  const LATEX_SIGNAL_RE = /\\[A-Za-z]|[\^_{}]/;
  const MAX_INLINE_MATH_CHARS = 200;

  function dollarInlineToParens(text) {
    return String(text).replace(
      DOLLAR_INLINE_RE,
      function (match, before, body) {
        if (body.length > MAX_INLINE_MATH_CHARS) return match;
        if (!LATEX_SIGNAL_RE.test(body)) return match;
        return before + "\\(" + body + "\\)";
      }
    );
  }

  // Innerhalb einer Formel ist Markdown kein Markup, sondern Notation.
  // Markdown frisst aber jeden Backslash vor einem Satzzeichen: aus
  // "17{,}5\%" wird "17{,}5%" - und das Prozentzeichen leitet in TeX einen
  // Kommentar ein. Die Formel verlor also stillschweigend ihr Ergebnis.
  // Ein verdoppelter Backslash ueberlebt den Markdown-Pass als einer;
  // *, _, ` und ~ bekommen einen, damit aus "x_1" kein Kursivtext wird.
  function escapeMathSegment(segment) {
    return segment.replace(/\\/g, "\\\\").replace(/([*_`~])/g, "\\$1");
  }

  function prepareMarkdown(markdown) {
    // Code bleibt unangetastet: dort ist "$" ein Zeichen und kein Trenner.
    const codeSegments = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;
    return String(markdown || "")
      .split(codeSegments)
      .map(function (part, index) {
        if (index % 2) return part;
        // Erst die Dollar-Schreibweise auf die KaTeX-Form bringen, dann alle
        // Formelabschnitte gegen den Markdown-Pass schuetzen. Was ausserhalb
        // einer Formel steht, behaelt die alte, konservative Behandlung:
        // ein einzelnes "\(" bleibt sichtbar, statt zu "(" zu verkuemmern.
        return withMathSegments(
          dollarInlineToParens(part),
          escapeMathSegment,
          function (plain) { return plain.replace(/\\([\[\]()])/g, "\\\\$1"); }
        );
      })
      .join("");
  }

  function withMathSegments(text, onMath, onPlain) {
    let out = "";
    let cursor = 0;
    let match;
    MATH_SPAN_RE.lastIndex = 0;
    while ((match = MATH_SPAN_RE.exec(text)) !== null) {
      out += onPlain(text.slice(cursor, match.index)) + onMath(match[0]);
      cursor = match.index + match[0].length;
    }
    return out + onPlain(text.slice(cursor));
  }

  // Eine gerenderte Formel ist im DOM ein .katex-Block und damit fuer jede
  // Textsuche unsichtbar. Der gespeicherte Anker traegt dagegen den
  // LaTeX-QUELLTEXT - ein Satz mit Formel fand seine Stelle im Konsens
  // deshalb nie und landete samt Rohtext in der Key-claims-Liste. Diese
  // Fassung laesst die Formel weg und passt damit wieder auf den sichtbaren
  // Text. "" heisst: keine Formel enthalten, es gibt nichts zu ergaenzen.
  function stripMath(value) {
    const source = String(value || "");
    const stripped = withMathSegments(
      dollarInlineToParens(source),
      function () { return " "; },
      function (plain) { return plain; }
    );
    return stripped === source ? "" : stripped;
  }

  // Ein blanker LaTeX-Ausdruck OHNE Trennzeichen. So stehen abgesetzte
  // Formeln in Ankern aus alten Laeufen: gespeichert wurde die Zeile
  // zwischen den "$$", nicht der Block - "6{,}7 / 5{,}7 - 1 \approx 17{,}5\%"
  // steht seitdem als Quelltext in der Key-claims-Liste. Nachtraeglich als
  // Formel gelesen wird das nur, wenn NICHTS ausser LaTeX darin steht: ein
  // einziges Wort gewoehnlichen Textes laesst den Anker Text bleiben.
  const LATEX_LEFTOVER_RE = /\\[A-Za-z]+|[\s\d{}[\]()^_+\-*/=<>.,:;|&~!%'"\\]/g;

  function wrapBareLatex(value) {
    const source = String(value || "").trim();
    if (!source || source.length > MAX_INLINE_MATH_CHARS) return "";
    if (source.indexOf("$") !== -1 || /\\[([]/.test(source)) return "";
    if (!LATEX_SIGNAL_RE.test(source)) return "";
    if (source.replace(LATEX_LEFTOVER_RE, "")) return "";
    return "\\(" + source + "\\)";
  }

  // Nicht jeder Weg fuehrt ueber prepareMarkdown: Share- und Topic-Seiten
  // liefern fertiges HTML vom Server, und ohne marked/DOMPurify zeigt die App
  // den Rohtext. Damit "$...$" auch dort eine Formel wird, laeuft dieselbe
  // konservative Erkennung noch einmal ueber die fertigen Textknoten.
  const MATH_SKIP_SELECTOR = "script, style, code, pre, textarea, .katex";

  function normalizeDollarMath(root) {
    const doc = root.ownerDocument || document;
    const walker = doc.createTreeWalker(root, 4 /* SHOW_TEXT */);
    const pending = [];
    let node;
    while ((node = walker.nextNode())) {
      const value = node.nodeValue;
      if (!value || value.indexOf("$") === -1) continue;
      if (node.parentElement && node.parentElement.closest(MATH_SKIP_SELECTOR)) continue;
      const converted = dollarInlineToParens(value);
      if (converted !== value) pending.push([node, converted]);
    }
    pending.forEach(function (entry) { entry[0].nodeValue = entry[1]; });
  }

  function render(root) {
    if (!root) return;
    normalizeDollarMath(root);
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(root, {
      delimiters: delimiters,
      throwOnError: false,
      trust: false
    });
  }

  window.ConsensusMath = {
    prepareMarkdown: prepareMarkdown,
    stripMath: stripMath,
    wrapBareLatex: wrapBareLatex,
    render: render
  };

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-math-render]").forEach(render);
  });
})();
