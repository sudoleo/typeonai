// =====================================================================
// composer-collapse.js
// Der Composer schrumpft auf dem Handy zu EINER Zeile, sobald er nicht
// gebraucht wird: nach dem Absenden einer Frage und beim Scrollen nach
// unten (also genau dann, wenn man liest). Im Thread ist er unten fixiert
// und hat mit Anhangleiste, Lauf-Schalter und Disclaimer-Fuss knapp ein
// Drittel des Bildschirms belegt — Flaeche, die der Antwort fehlte.
//
// Aufgeklappt wird er wieder, sobald man ihn anfasst (Fokus/Tap) oder nach
// oben scrollt. Zustand ist eine einzige Klasse an <body>, die Optik liegt
// in shell.css (dort auch die Breakpoint-Grenze).
//
// Desktop bleibt bewusst unangetastet: dort steht der Composer neben, nicht
// vor der Antwort. COLLAPSE_QUERY ist die einzige Stelle, an der das haengt.
// Exporte: window.App.composer.{collapse,expand}
// =====================================================================

(function () {
  "use strict";

  window.App = window.App || {};

  // Muss zur Breakpoint-Grenze der eingeklappten Optik in shell.css passen
  // (dieselbe, ab der der Composer fixiert am unteren Bildrand sitzt).
  const COLLAPSE_QUERY = "(max-width: 1099px)";
  const COLLAPSED_CLASS = "composer-collapsed";
  // Kleine Scroll-Bewegungen (Momentum, Adressleiste, Layoutspruenge) duerfen
  // den Composer nicht flackern lassen.
  const SCROLL_THRESHOLD = 24;

  const mediaQuery = window.matchMedia(COLLAPSE_QUERY);
  let lastScrollY = window.scrollY;

  function isCollapsible() {
    return mediaQuery.matches && !document.body.classList.contains("is-hero");
  }

  function isCollapsed() {
    return document.body.classList.contains(COLLAPSED_CLASS);
  }

  // Der Autosizer in app-init.js schreibt bei jedem Tastendruck eine INLINE-
  // Hoehe ins Feld — die schlaegt jede Regel aus shell.css. Nach jedem
  // Umschalten muss er deshalb neu rechnen: sonst behaelt das Feld die Hoehe
  // des anderen Zustands, der Text scrollt in einem 34-px-Kasten weg und der
  // Cursor steht sichtbar ausserhalb des Feldes, in das man gerade tippt.
  function syncFieldHeight() {
    window.App.resizeQuestionInput?.();
  }

  function collapse() {
    if (!isCollapsible() || isCollapsed()) return;
    const input = document.getElementById("questionInput");
    // Wer gerade tippt oder einen Entwurf stehen hat, bekommt sein Feld nicht
    // unter den Fingern auf eine Zeile zusammengeschoben.
    if (input && (document.activeElement === input || input.value.trim())) return;
    document.body.classList.add(COLLAPSED_CLASS);
    syncFieldHeight();
  }

  function expand() {
    if (!isCollapsed()) return;
    document.body.classList.remove(COLLAPSED_CLASS);
    syncFieldHeight();
  }

  // Ein Tap irgendwo auf den Composer klappt ihn auf — auch auf den (+)- oder
  // Modell-Knopf, der im eingeklappten Zustand gar nicht sichtbar ist. Deshalb
  // in der Capture-Phase: erst aufklappen, dann den Klick wirken lassen.
  document.addEventListener("pointerdown", function (event) {
    if (!isCollapsed()) return;
    if (event.target.closest?.(".input-section")) expand();
  }, true);

  document.addEventListener("focusin", function (event) {
    if (event.target.closest?.(".input-section")) expand();
  });

  window.addEventListener("scroll", function () {
    const current = window.scrollY;
    const delta = current - lastScrollY;
    if (Math.abs(delta) < SCROLL_THRESHOLD) return;
    lastScrollY = current;
    if (delta > 0) collapse();
    else expand();
  }, { passive: true });

  mediaQuery.addEventListener?.("change", function (event) {
    if (!event.matches) expand();
  });

  window.App.composer = {
    collapse: collapse,
    expand: expand,
    isCollapsed: isCollapsed
  };

  // "New comparison", Logout und der Hero-Zustand starten immer offen.
  window.addEventListener("pageshow", expand);
  document.getElementById("newRunButton")?.addEventListener("click", expand);
})();
