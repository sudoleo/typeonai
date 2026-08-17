// =====================================================================
// composer-collapse.js
// Der Composer auf dem Handy verhaelt sich wie in einem Chat-Client: nach
// einer abgeschickten Frage ist er EINE Zeile, und er oeffnet sich erst
// wieder, wenn man ihn anfasst, um die naechste Frage zu tippen. Verlaesst
// der Fokus ein leeres Feld, faellt er von selbst wieder zusammen.
//
// Bewusst NICHT am Scrollen aufgehaengt: das Aufklappen beim Hochscrollen
// hat den Composer beim Zurueckblaettern in die Antwort geschoben. Nach
// unten scrollen klappt weiterhin zu — dort ist das Zuklappen genau das,
// was man will (mehr Antwort sehen), und es kostet keine Geste.
//
// Zustand ist eine einzige Klasse an <body>, die Optik liegt in shell.css
// (dort auch die Breakpoint-Grenze). Die Bewegung dazwischen misst dieses
// Modul: Start- und Zielhoehe der Box werden als Inline-Hoehe geschrieben,
// die Kurve steht in shell.css unter .composer-animating.
//
// Der Desktop hat KEINEN Zustand: dort ist die eine Zeile ab der ersten Frage
// einfach die Form des Composers, aufklappen gibt es nicht (User-Vorgabe
// 2026-08-14). Das ist reines CSS in shell.css (Abschnitt A) — dieses Modul
// ruehrt oberhalb von COLLAPSE_QUERY nichts an ausser der Feldhoehe beim
// Verlassen des Hero-Zustands.
// Exporte: window.App.composer.{collapse,expand,isCollapsed}
// =====================================================================

(function () {
  "use strict";

  window.App = window.App || {};

  // Muss zur Breakpoint-Grenze der eingeklappten Optik in shell.css passen
  // (dieselbe, ab der der Composer fixiert am unteren Bildrand sitzt).
  const COLLAPSE_QUERY = "(max-width: 1099px)";
  const COLLAPSED_CLASS = "composer-collapsed";
  const ANIMATING_CLASS = "composer-animating";
  const COLLAPSING_CLASS = "composer-collapsing";
  // Kleine Scroll-Bewegungen (Momentum, Adressleiste, Layoutspruenge) duerfen
  // den Composer nicht flackern lassen.
  const SCROLL_THRESHOLD = 24;
  // Muss zur Transition-Dauer in shell.css passen (Notbremse, falls
  // transitionend ausbleibt, weil die Bewegung unterbrochen wurde).
  const ANIMATION_MS = 240;

  const mediaQuery = window.matchMedia(COLLAPSE_QUERY);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let lastScrollY = window.scrollY;
  let animationTimer = null;

  // Animiert wird der ganze Composer-Block, nicht nur das Eingabefeld: zum
  // eingeklappten Zustand gehoert auch der Disclaimer-Fuss, und der ist auf
  // einem 375er Schirm hoeher als das Feld selbst. Wuerde nur die Box
  // wachsen, klappte der Fuss trotzdem sprunghaft darunter auf.
  function composerBox() {
    return document.querySelector(".input-section");
  }

  function questionField() {
    return document.getElementById("questionInput");
  }

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

  // Alles, was im eingeklappten Zustand unsichtbar waere und deshalb nicht
  // hinter dem Ruecken des Nutzers verschwinden darf.
  function hasUnfinishedBusiness() {
    const input = questionField();
    if (input && input.value.trim()) return true;

    const attachments = document.getElementById("attachmentBar");
    if (attachments && !attachments.hidden) return true;

    const quote = document.getElementById("composerQuote");
    if (quote && !quote.hidden) return true;

    const notice = document.getElementById("composerRunNotice");
    if (notice && !notice.hidden) return true;

    const inputBox = document.querySelector(".input-section .chat-input-container");
    if (inputBox && (inputBox.classList.contains("attach-menu-open") ||
                     inputBox.querySelector(".model-picker-menu.is-open"))) {
      return true;
    }

    return false;
  }

  // ---- Die Bewegung -------------------------------------------------------
  // Zwischen den beiden Zustaenden liegen ein display:none, ein Umbruch und
  // zwei Paddings — nichts davon ist fuer sich animierbar. Animierbar ist die
  // HOEHE des Blocks: Startwert messen, Zustand umschalten, Zielwert messen,
  // und die Strecke dazwischen als Transition laufen lassen. Der Block clippt
  // waehrenddessen (overflow: hidden), also faltet sich auf, was dazukommt,
  // statt aufzupoppen.
  //
  // Die beiden Richtungen verankern verschieden, damit in beiden Faellen das
  // Eingabefeld stillsteht und nur der Ueberschuss wandert: aufklappen von
  // oben (der Block ist unten fixiert, das Feld steigt aus seiner
  // eingeklappten Position auf), zuklappen von unten (die eine Zeile steht
  // schon dort, wo sie hingehoert, darueber faltet sich der Rest weg).
  function endAnimation() {
    window.clearTimeout(animationTimer);
    animationTimer = null;
    const box = composerBox();
    if (box) box.style.height = "";
    document.body.classList.remove(ANIMATING_CLASS);
    document.body.classList.remove(COLLAPSING_CLASS);
  }

  function animateToggle(applyState, collapsing) {
    const box = composerBox();
    if (!box || reducedMotion.matches) {
      applyState();
      syncFieldHeight();
      return;
    }

    // getBoundingClientRect liefert auch mitten in einer laufenden Transition
    // den aktuellen Wert — ein Umschalten waehrend der Bewegung setzt also
    // dort an, wo der Block gerade steht, statt zurueckzuspringen.
    const from = box.getBoundingClientRect().height;

    applyState();
    // Erst ohne Inline-Hoehe, ohne die Animationsklassen und mit frisch
    // gerechnetem Feld steht die natuerliche Zielhoehe fest.
    box.style.height = "";
    document.body.classList.remove(ANIMATING_CLASS);
    document.body.classList.remove(COLLAPSING_CLASS);
    syncFieldHeight();
    const to = box.getBoundingClientRect().height;

    if (!from || !to || Math.abs(to - from) < 1) {
      endAnimation();
      return;
    }

    box.style.height = `${from}px`;
    document.body.classList.add(ANIMATING_CLASS);
    if (collapsing) document.body.classList.add(COLLAPSING_CLASS);
    // Reflow, damit der Startwert zusammen mit der aktiven Transition als
    // "before-change style" feststeht — sonst springt der Block direkt aufs Ziel.
    void box.offsetHeight;
    box.style.height = `${to}px`;

    window.clearTimeout(animationTimer);
    animationTimer = window.setTimeout(endAnimation, ANIMATION_MS + 80);
  }

  function collapse(options) {
    const force = Boolean(options && options.force);
    if (!isCollapsible() || isCollapsed()) return;
    if (!force) {
      // Wer gerade tippt oder einen Entwurf stehen hat, bekommt sein Feld
      // nicht unter den Fingern auf eine Zeile zusammengeschoben.
      if (document.activeElement === questionField()) return;
      if (hasUnfinishedBusiness()) return;
    }
    animateToggle(function () {
      document.body.classList.add(COLLAPSED_CLASS);
    }, true);
  }

  function expand() {
    if (!isCollapsed()) return;
    animateToggle(function () {
      document.body.classList.remove(COLLAPSED_CLASS);
    }, false);
  }

  // ---- Der Tap auf den eingeklappten Composer ------------------------------
  // Aufgeklappt ist der Composer mehr als doppelt so hoch, und die Zeile mit
  // (+), Modellwahl und Senden liegt danach genau dort, wo der Finger
  // aufgesetzt hat. Der Browser loest den Klick auf der NEUEN Position auf:
  // man tippt aufs Feld und trifft den Modellwaehler. Deshalb nimmt dieser
  // Handler den Tap selbst an — aufklappen, Fokus noch in derselben Geste ins
  // Feld setzen (nur so geht die Tastatur auf und der Cursor bleibt drin) und
  // den nachfolgenden Klick verwerfen.
  let clickGuard = null;

  function releaseClickGuard() {
    if (!clickGuard) return;
    document.removeEventListener("click", clickGuard.handler, true);
    window.clearTimeout(clickGuard.timer);
    clickGuard = null;
  }

  function swallowNextClick() {
    releaseClickGuard();
    const handler = function (event) {
      releaseClickGuard();
      event.preventDefault();
      event.stopPropagation();
      // Zweiter Versuch fuer den Fokus: pointerdown reicht auf iOS nicht
      // immer, um die Tastatur zu oeffnen — der Klick ist ebenfalls eine
      // Nutzergeste und damit die letzte Gelegenheit dafuer.
      focusField();
    };
    clickGuard = {
      handler: handler,
      timer: window.setTimeout(releaseClickGuard, 400)
    };
    document.addEventListener("click", handler, true);
  }

  function focusField() {
    const input = questionField();
    if (!input || input.disabled || document.activeElement === input) return;
    input.focus({ preventScroll: true });
  }

  document.addEventListener("pointerdown", function (event) {
    const inComposer = event.target.closest?.(".input-section");
    if (!inComposer) {
      releaseClickGuard();
      return;
    }
    if (!isCollapsed()) return;

    // Waehrend eines Laufs ist der Senden-Knopf der Abbrechen-Knopf: der
    // einzige Griff, der eingeklappt sichtbar bleibt und seine eigene Wirkung
    // behalten muss.
    if (event.target.closest(".input-actions-container")) {
      expand();
      return;
    }

    const input = questionField();
    if (!input || input.disabled) {
      expand();
      return;
    }

    event.preventDefault();
    swallowNextClick();
    expand();
    focusField();
  }, true);

  // Fokus kann auch ohne Tap ankommen (Tastatur, Sprachassistent, ein Skript,
  // das eine Frage vorbefuellt).
  document.addEventListener("focusin", function (event) {
    if (event.target.closest?.(".input-section")) expand();
  });

  // "Oeffnet sich, wenn man reintippt": auch dann, wenn der Fokus schon im
  // eingeklappten Feld sitzt (direkt nach dem Absenden bleibt er dort, wenn
  // die Tastatur offen bleibt). Capture-Phase, damit die Klasse weg ist,
  // bevor der Autosizer die neue Hoehe misst.
  document.addEventListener("input", function (event) {
    if (event.target === questionField()) expand();
  }, true);

  // Fokus weg und nichts Angefangenes im Feld: zurueck auf eine Zeile. Ein
  // Tick warten, weil der Fokus beim Oeffnen eines Menues kurz nirgends liegt.
  document.addEventListener("focusout", function (event) {
    if (!event.target.closest?.(".input-section")) return;
    window.setTimeout(function () {
      const active = document.activeElement;
      if (active && active.closest?.(".input-section")) return;
      collapse();
    }, 0);
  });

  window.addEventListener("scroll", function () {
    const current = window.scrollY;
    const delta = current - lastScrollY;
    if (Math.abs(delta) < SCROLL_THRESHOLD) return;
    lastScrollY = current;
    // Nur nach unten. Hochscrollen ist Zurueckblaettern in der Antwort und
    // darf den Composer nicht wieder davorschieben.
    if (delta > 0) collapse();
  }, { passive: true });

  // Ueber der Breakpoint-Grenze gibt es keinen eingeklappten Zustand. Nicht nur
  // am change-Event haengen: bei einem Wechsel ohne Event (Emulation, manche
  // Orientierungswechsel) bliebe der Composer sonst als Klasse haengen und
  // waere beim Zurueckdrehen im falschen Zustand.
  function syncBreakpoint() {
    if (mediaQuery.matches || !isCollapsed()) return;
    endAnimation();
    document.body.classList.remove(COLLAPSED_CLASS);
    syncFieldHeight();
  }

  mediaQuery.addEventListener?.("change", syncBreakpoint);
  window.addEventListener("resize", syncBreakpoint, { passive: true });

  // Ein geladenes Bookmark und der Demo-Lauf verlassen den Hero-Zustand, ohne
  // durch sendQuestion() zu gehen. Auch dann steht ab jetzt eine Antwort auf
  // dem Schirm, und der Composer ist die kleine Zeile darunter.
  let wasHero = document.body.classList.contains("is-hero");
  new MutationObserver(function () {
    const nowHero = document.body.classList.contains("is-hero");
    if (nowHero === wasHero) return;
    wasHero = nowHero;
    if (nowHero) expand();
    else collapse();
    // Die Hero-Grenze ist zugleich eine Feldhoehen-Grenze: auf dem Desktop
    // faellt die min-height des Feldes hier von 52 auf 34 px, ohne dass eine
    // der beiden Bewegungen oben laeuft. Der Autosizer haelt sonst die
    // Inline-Hoehe des Hero-Zustands fest und das Feld bliebe zu hoch.
    // (Nach collapse()/expand(), damit deren Messung nicht schon die neue
    // Hoehe sieht und die Bewegung ins Leere laeuft.)
    syncFieldHeight();
  }).observe(document.body, { attributes: true, attributeFilter: ["class"] });

  document.addEventListener("transitionend", function (event) {
    if (event.propertyName !== "height") return;
    if (event.target !== composerBox()) return;
    endAnimation();
  });

  window.App.composer = {
    collapse: collapse,
    expand: expand,
    isCollapsed: isCollapsed
  };

  // "New comparison" und der Hero-Zustand starten immer offen.
  window.addEventListener("pageshow", function () {
    if (document.body.classList.contains("is-hero")) expand();
  });
  document.getElementById("newRunButton")?.addEventListener("click", expand);
})();
