const DEFAULT_SYSTEM_PROMPT = "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify. Do not ask any follow-up or clarifying questions; answer directly with the information available.";

// Einmalige Migration: Bestandsnutzer, die noch auf dem alten "respond briefly"-Default
// stehen, werden auf den neuen Default gehoben. Individuell angepasste Prompts bleiben
// unberührt, da nur der exakte alte Default-String ersetzt wird.
(function migrateLegacySystemPrompt() {
  const LEGACY_DEFAULTS = [
    "Please respond briefly and precisely, focusing only on the essentials.",
    "Please respond briefly and precisely, focusing only on the essentials. No follow-up questions.",
    "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify.",
  ];
  try {
    const stored = localStorage.getItem("systemPrompt");
    if (stored !== null && LEGACY_DEFAULTS.includes(stored.trim())) {
      localStorage.setItem("systemPrompt", DEFAULT_SYSTEM_PROMPT);
    }
  } catch (e) {
    // localStorage nicht verfügbar (z. B. Private Mode) — kein Abbruch nötig.
  }
})();

function getStoredSystemPrompt() {
  const stored = localStorage.getItem("systemPrompt");
  if (stored !== null) {
    return stored;
  }
  localStorage.setItem("systemPrompt", DEFAULT_SYSTEM_PROMPT);
  return DEFAULT_SYSTEM_PROMPT;
}

/**
 * Die Einstellungen als Reiter statt als eine lange Bahn.
 *
 * Sechs Kategorien untereinander waren beim Oeffnen eine Wand: der Nutzer musste
 * scrollen, um zu wissen, was es ueberhaupt gibt. Jetzt ist genau EINE Kategorie
 * sichtbar und die Liste links ist das Inhaltsverzeichnis.
 *
 * Die Sichtbarkeit der Panels gehoert ausschliesslich hierher. firebase.js hat
 * frueher `style.display` direkt auf dem Account-Panel gesetzt; ein Inline-Style
 * haette diesen Controller uebersteuert, deshalb schaltet firebase.js jetzt
 * ueber `setTabAvailable` nur noch den REITER frei.
 */
const settingsTabs = (function () {
  function tabs() {
    return Array.from(document.querySelectorAll("[data-settings-tab]"));
  }

  function panelOf(tab) {
    return document.getElementById(tab.dataset.settingsTab);
  }

  function available(tab) {
    return !tab.hidden;
  }

  function activate(panelId, options) {
    const all = tabs();
    if (!all.length) return;
    let target = all.find(tab => tab.dataset.settingsTab === panelId && available(tab));
    if (!target) target = all.find(available);
    if (!target) return;

    all.forEach(tab => {
      const isActive = tab === target;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
      // Genau ein Reiter liegt in der Tab-Reihenfolge; zwischen den Reitern
      // navigieren die Pfeiltasten (WAI-ARIA Tabs-Muster).
      tab.tabIndex = isActive ? 0 : -1;
      const panel = panelOf(tab);
      if (panel) panel.hidden = !isActive;
    });

    if (options?.focus) target.focus();
    const body = document.querySelector(".settings-body");
    if (body) body.scrollTop = 0;
  }

  function activeTab() {
    return tabs().find(tab => tab.classList.contains("is-active")) || null;
  }

  function setTabAvailable(panelId, isAvailable) {
    const tab = tabs().find(item => item.dataset.settingsTab === panelId);
    if (!tab) return;
    tab.hidden = !isAvailable;
    // Der aktive Reiter darf nicht verschwinden, waehrend sein Panel offen ist
    // (Logout mit geoeffnetem Account-Tab): dann faellt die Auswahl zurueck.
    if (!isAvailable && tab.classList.contains("is-active")) activate(null);
  }

  function step(delta) {
    const usable = tabs().filter(available);
    const index = usable.indexOf(activeTab());
    if (index < 0) return;
    const next = usable[(index + delta + usable.length) % usable.length];
    activate(next.dataset.settingsTab, { focus: true });
  }

  function bind() {
    const nav = document.querySelector(".settings-nav");
    if (!nav || nav.dataset.bound === "true") return;
    nav.dataset.bound = "true";

    nav.addEventListener("click", event => {
      const tab = event.target.closest("[data-settings-tab]");
      if (tab) activate(tab.dataset.settingsTab);
    });

    nav.addEventListener("keydown", event => {
      const usable = tabs().filter(available);
      if (event.key === "ArrowDown" || event.key === "ArrowRight") step(1);
      else if (event.key === "ArrowUp" || event.key === "ArrowLeft") step(-1);
      else if (event.key === "Home") activate(usable[0]?.dataset.settingsTab, { focus: true });
      else if (event.key === "End") activate(usable[usable.length - 1]?.dataset.settingsTab, { focus: true });
      else return;
      event.preventDefault();
    });

    activate(null);
  }

  return { bind, activate, setTabAvailable, reset: () => activate(null) };
})();

// Sofort veroeffentlichen, nicht erst beim Binden: firebase.js kann den
// Account-Reiter freigeben, bevor DOMContentLoaded gelaufen ist. Waere der Bus
// dann noch leer, bliebe der Reiter fuer eingeloggte Konten unsichtbar.
window.App = window.App || {};
window.App.settingsTabs = settingsTabs;

function openSettingsModal() {
  const modal = document.getElementById("systemPromptModal");
  const textarea = document.getElementById("systemPromptInput");
  if (!modal || !textarea) return;

  textarea.value = getStoredSystemPrompt();
  // Immer auf dem ersten Reiter oeffnen. Der zuletzt benutzte waere clever,
  // aber unvorhersehbar — man findet Einstellungen ueber einen festen Ort.
  settingsTabs.reset();
  modal.style.display = "block";
  window.App?.trackAppEvent?.("app_settings_open");
}

function closeSettingsModal() {
  const modal = document.getElementById("systemPromptModal");
  if (modal) {
    modal.style.display = "none";
  }
}

function saveSystemPrompt() {
  const textarea = document.getElementById("systemPromptInput");
  if (textarea) {
    localStorage.setItem("systemPrompt", textarea.value.trim());
  }
  closeSettingsModal();
  window.App?.trackAppEvent?.("app_settings_saved");
}

function openHelpModal() {
  const helpModal = document.getElementById("helpModal");
  if (helpModal) {
    helpModal.style.display = "block";
    window.App?.trackAppEvent?.("app_help_open");
  }
}

function closeHelpModal() {
  const helpModal = document.getElementById("helpModal");
  if (helpModal) {
    helpModal.style.display = "none";
  }
}

function bindSettingsModalControls() {
  if (window.__settingsModalControlsBound) return;
  window.__settingsModalControlsBound = true;

  settingsTabs.bind();

  document.getElementById("editSystemPromptBtn")?.addEventListener("click", openSettingsModal);
  document.getElementById("closeSystemPromptModal")?.addEventListener("click", closeSettingsModal);
  document.getElementById("saveSystemPromptBtn")?.addEventListener("click", saveSystemPrompt);
  document.getElementById("helpButton")?.addEventListener("click", openHelpModal);
  document.getElementById("closeHelpModal")?.addEventListener("click", closeHelpModal);

  window.addEventListener("click", (event) => {
    if (event.target === document.getElementById("systemPromptModal")) {
      closeSettingsModal();
    }
    if (event.target === document.getElementById("helpModal")) {
      closeHelpModal();
    }
  });
}

function initAppWidthResizer() {
  if (window.__appWidthResizerBound) return;

  const container = document.querySelector(".container");
  if (!container) return;

  window.__appWidthResizerBound = true;

  const storageKey = "consens_app_container_width";
  const defaultWidth = 900;
  const minWidth = 760;
  const desktopQuery = window.matchMedia("(min-width: 1024px)");
  let activeDrag = null;
  let pendingWidth = null;

  const leftHandle = document.createElement("div");
  leftHandle.className = "app-width-resize-handle left";
  leftHandle.setAttribute("aria-hidden", "true");

  const rightHandle = document.createElement("div");
  rightHandle.className = "app-width-resize-handle right";
  rightHandle.setAttribute("aria-hidden", "true");

  container.append(leftHandle, rightHandle);

  function getStoredWidth() {
    const stored = Number(localStorage.getItem(storageKey));
    return Number.isFinite(stored) && stored > 0 ? stored : defaultWidth;
  }

  function getMaxWidth() {
    const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
    const sidebar = document.querySelector(".sidebar");
    const sidebarVisible = sidebar
      && !sidebar.classList.contains("collapsed")
      && window.matchMedia("(min-width: 1550px)").matches;
    const leftClearance = sidebarVisible ? Math.ceil(sidebar.getBoundingClientRect().right + 24) : 20;
    const symmetricMax = viewportWidth - (leftClearance * 2);
    return Math.max(minWidth, Math.min(viewportWidth - 20, symmetricMax));
  }

  function clampWidth(width) {
    return Math.round(Math.min(Math.max(width, minWidth), getMaxWidth()));
  }

  function applyWidth(width) {
    if (!desktopQuery.matches) {
      container.style.removeProperty("--app-container-width");
      container.classList.remove("is-width-resizable");
      return defaultWidth;
    }

    const clamped = clampWidth(width);
    container.style.setProperty("--app-container-width", `${clamped}px`);
    container.classList.add("is-width-resizable");
    return clamped;
  }

  function syncWidth() {
    if (!desktopQuery.matches) {
      applyWidth(defaultWidth);
      return;
    }

    pendingWidth = applyWidth(getStoredWidth());
  }

  function beginDrag(event, side) {
    if (!desktopQuery.matches || event.button !== 0) return;

    event.preventDefault();
    activeDrag = {
      side,
      startX: event.clientX,
      startWidth: container.getBoundingClientRect().width
    };
    document.body.classList.add("app-width-resizing");
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function updateDrag(event) {
    if (!activeDrag) return;

    const delta = event.clientX - activeDrag.startX;
    const nextWidth = activeDrag.side === "right"
      ? activeDrag.startWidth + (delta * 2)
      : activeDrag.startWidth - (delta * 2);
    pendingWidth = applyWidth(nextWidth);
  }

  function endDrag() {
    if (!activeDrag) return;

    activeDrag = null;
    document.body.classList.remove("app-width-resizing");
    if (desktopQuery.matches && pendingWidth) {
      localStorage.setItem(storageKey, String(pendingWidth));
    }
  }

  function resetWidth() {
    localStorage.removeItem(storageKey);
    pendingWidth = applyWidth(defaultWidth);
  }

  leftHandle.addEventListener("pointerdown", (event) => beginDrag(event, "left"));
  rightHandle.addEventListener("pointerdown", (event) => beginDrag(event, "right"));
  leftHandle.addEventListener("dblclick", resetWidth);
  rightHandle.addEventListener("dblclick", resetWidth);
  window.addEventListener("pointermove", updateDrag);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", syncWidth);

  const sidebar = document.querySelector(".sidebar");
  if (sidebar) {
    new MutationObserver(syncWidth).observe(sidebar, {
      attributes: true,
      attributeFilter: ["class"]
    });
  }

  if (desktopQuery.addEventListener) {
    desktopQuery.addEventListener("change", syncWidth);
  } else {
    desktopQuery.addListener(syncWidth);
  }

  syncWidth();
}

function bindAppUiControls() {
  bindSettingsModalControls();
  // App-width resize handles were removed with the framed container; the app
  // now uses a fixed, fluid canvas width. initAppWidthResizer is left defined
  // but no longer invoked.
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bindAppUiControls);
} else {
  bindAppUiControls();
}
