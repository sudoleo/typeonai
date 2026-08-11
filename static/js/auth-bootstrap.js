// Auth bootstrap watchdog. firebase.js is an ES module with CDN imports; if
// those imports fail, none of that module executes. This small same-origin
// classic script keeps the signed-out UI usable instead of trusting a stale
// localStorage token forever.
(function () {
  const timeout = Number(window.AUTH_BOOTSTRAP_TIMEOUT_MS) || 8000;
  let fallbackActive = false;
  let returnFocus = null;

  function modalEls() {
    return {
      modal: document.getElementById("loginModal"),
      close: document.getElementById("closeLoginModal"),
      error: document.getElementById("loginError")
    };
  }

  function closeUnavailableDialog() {
    const { modal } = modalEls();
    if (!modal || modal.style.display !== "block") return;
    modal.style.display = "none";
    if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    returnFocus = null;
  }

  function openUnavailableDialog(trigger) {
    const { modal, close, error } = modalEls();
    if (!modal) return;
    returnFocus = trigger || document.activeElement;
    if (error) {
      error.textContent = "Login is temporarily unavailable. Check your connection and try again.";
    }
    modal.style.display = "block";
    close?.focus({ preventScroll: true });
  }

  function activateFallback() {
    if (window.auth || fallbackActive) return;
    fallbackActive = true;
    document.documentElement.dataset.authUnavailable = "true";

    const actions = document.getElementById("authTopActions");
    if (actions) actions.hidden = false;
    const loginContainer = document.getElementById("loginContainer");
    if (loginContainer) {
      loginContainer.innerHTML = "";
      loginContainer.hidden = true;
    }
    for (const id of ["freeUsageDisplay", "deepUsageDisplay", "watchUsageDisplay"]) {
      const node = document.getElementById(id);
      if (node) node.textContent = "";
    }
    const bookmarks = document.getElementById("bookmarksContainer");
    if (bookmarks?.querySelector(".skeleton")) bookmarks.innerHTML = "";

    actions?.addEventListener("click", event => {
      const trigger = event.target.closest("button");
      if (!trigger || window.auth) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      openUnavailableDialog(trigger);
    }, true);
    modalEls().close?.addEventListener("click", closeUnavailableDialog);
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") closeUnavailableDialog();
    });
    window.dispatchEvent(new CustomEvent("consensio:auth-unavailable"));
  }

  const timer = window.setTimeout(activateFallback, timeout);
  window.addEventListener("consensio:auth-state", () => {
    window.clearTimeout(timer);
    delete document.documentElement.dataset.authUnavailable;
  }, { once: true });
})();
