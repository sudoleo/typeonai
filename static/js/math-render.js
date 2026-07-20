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

  function prepareMarkdown(markdown) {
    // Markdown behandelt \[, \], \( und \) als Escapes und entfernt den
    // Backslash. Ein zusaetzlicher Backslash laesst das echte LaTeX-
    // Trennzeichen den Markdown-Pass ueberleben. Befehle wie \frac oder
    // \mathbb werden von Markdown ohnehin nicht veraendert.
    const codeSegments = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;
    return String(markdown || "")
      .split(codeSegments)
      .map(function (part, index) {
        return index % 2 ? part : part.replace(/\\([\[\]()])/g, "\\\\$1");
      })
      .join("");
  }

  function render(root) {
    if (!root || typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(root, {
      delimiters: delimiters,
      throwOnError: false,
      trust: false
    });
  }

  window.ConsensusMath = { prepareMarkdown: prepareMarkdown, render: render };

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-math-render]").forEach(render);
  });
})();
