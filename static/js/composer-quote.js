// =====================================================================
// composer-quote.js
// "Ask about this": ein markierter Abschnitt aus einer Antwort wandert als
// sichtbares Zitat in den Composer und geht beim Senden als Markdown-
// Blockzitat VOR der getippten Frage raus.
//
// Warum eine eigene Flaeche statt Text im Feld: das Zitat ist Kontext, keine
// Frage. Als "> ..."-Text im Textfeld haette man es beim Weitertippen im Weg,
// muesste es von Hand wieder loeschen und saehe nicht, wo das Zitat endet und
// die eigene Frage anfaengt. Hier steht es ueber dem Feld, ist mit einem Klick
// weg, und der Composer bleibt beim Tippen die eine Zeile, die er sein soll.
//
// Der Lauf selbst kennt kein Zitat: compose() setzt es der Frage voran, und ab
// da ist es Teil der Frage — Thread-Kopf, Bookmark, Chat-Kontext und die sechs
// Modelle sehen genau einen Text. Deshalb braucht keine andere Stelle als
// query-send.js (Senden) und app-init.js (Aufraeumen) davon zu wissen.
// Exporte: window.App.quote.{set,clear,text,has,compose,element}
// =====================================================================

(function () {
  "use strict";

  window.App = window.App || {};

  // Ein Zitat ist ein Beleg, kein Anhang: laenger als ein paar Absaetze waere
  // es die Frage selbst und wuerde das Wortlimit des Laufs auffressen.
  const MAX_QUOTE_CHARS = 1200;

  const state = { text: "" };

  function els() {
    return {
      box: document.getElementById("composerQuote"),
      text: document.getElementById("composerQuoteText"),
      remove: document.getElementById("composerQuoteRemove"),
      input: document.getElementById("questionInput")
    };
  }

  function normalize(raw) {
    const text = String(raw || "")
      .replace(/\r\n?/g, "\n")
      .replace(/[ \t]+/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (text.length <= MAX_QUOTE_CHARS) return text;
    return `${text.slice(0, MAX_QUOTE_CHARS).trimEnd()}…`;
  }

  function render() {
    const { box, text } = els();
    if (!box || !text) return;
    text.textContent = state.text;
    box.hidden = !state.text;
  }

  function set(raw) {
    state.text = normalize(raw);
    render();
    return state.text;
  }

  function clear() {
    if (!state.text) return;
    state.text = "";
    render();
  }

  // Der Composer waechst gerade um eine Zeile: erst danach steht fest, wohin
  // gescrollt werden muss, damit Zitat UND Feld im Bild sind.
  function focusComposer() {
    const { input, box } = els();
    window.App.composer?.expand?.();
    if (input) {
      input.focus({ preventScroll: true });
      const end = input.value.length;
      try { input.setSelectionRange(end, end); } catch (_) { /* type=textarea only */ }
      window.App.resizeQuestionInput?.();
    }
    requestAnimationFrame(() => {
      (box && !box.hidden ? box : input)?.scrollIntoView({ block: "nearest" });
    });
  }

  // Die getippte Frage steht VORN, das Zitat darunter. Der Thread-Kopf, der
  // Seitentitel und der Bookmark-Name sind reiner Text und zeigen den Anfang
  // dieser einen Zeichenkette — sie muessen mit der Frage anfangen, nicht mit
  // einem fremden Absatz. Aus demselben Grund Anfuehrungszeichen statt
  // Markdown-"> ": ein Blockzitat-Zeichen, das nirgends gerendert wird, ist
  // ueberall dort sichtbarer Muell.
  function compose(question) {
    const typed = String(question ?? "").trim();
    if (!state.text) return String(question ?? "");
    const passage = `“${state.text}”`;
    return typed
      ? `${typed}\n\nQuoted from the previous answer:\n${passage}`
      : `Please comment on this passage from the previous answer:\n${passage}`;
  }

  function bind() {
    const { remove, input } = els();
    remove?.addEventListener("click", () => {
      clear();
      input?.focus({ preventScroll: true });
    });
    render();
  }

  window.App.quote = {
    set,
    clear,
    text: () => state.text,
    has: () => !!state.text,
    compose,
    focusComposer,
    // Fuer composer-collapse.js: ein stehendes Zitat ist Angefangenes und darf
    // nicht hinter dem Ruecken des Nutzers weggeklappt werden.
    element: () => els().box
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
