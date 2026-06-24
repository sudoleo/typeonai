const DEFAULT_SYSTEM_PROMPT = "Please respond briefly and precisely, focusing only on the essentials.";

function getStoredSystemPrompt() {
  const stored = localStorage.getItem("systemPrompt");
  if (stored !== null) {
    return stored;
  }
  localStorage.setItem("systemPrompt", DEFAULT_SYSTEM_PROMPT);
  return DEFAULT_SYSTEM_PROMPT;
}

function openSettingsModal() {
  const modal = document.getElementById("systemPromptModal");
  const textarea = document.getElementById("systemPromptInput");
  if (!modal || !textarea) return;

  textarea.value = getStoredSystemPrompt();
  modal.style.display = "block";
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
}

function openHelpModal() {
  const helpModal = document.getElementById("helpModal");
  if (helpModal) {
    helpModal.style.display = "block";
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
      && window.matchMedia("(min-width: 1466px)").matches;
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
