(function () {
  const config = document.getElementById("appBootstrapConfig");
  const parse = (name, fallback) => {
    try { return JSON.parse(config?.dataset?.[name] || ""); }
    catch (_) { return fallback; }
  };
  window.FIREBASE_CONFIG = {
    apiKey: config?.dataset.firebaseApiKey || "",
    authDomain: config?.dataset.firebaseAuthDomain || "",
    projectId: config?.dataset.firebaseProjectId || "",
    storageBucket: config?.dataset.firebaseStorageBucket || "",
    messagingSenderId: config?.dataset.firebaseMessagingSenderId || "",
    appId: config?.dataset.firebaseAppId || ""
  };
  window.APP_LIMITS = parse("limits", {});
  window.FREE_DEFAULT_MODELS = parse("freeModels", {});
  window.PRO_DEFAULT_MODELS = parse("proModels", {});
  window.DEEP_THINK_CONSENSUS_MODEL = parse("deepThinkModel", "");
  window.CONSENSUS_PRESETS = parse("consensusPresets", []);
  window.DEFAULT_CONSENSUS_PRESET = parse("defaultConsensusPreset", "");
  // Die eine Familienliste der App: Antwortboxen, Picker, Sendepfad und
  // Fortschritt lesen ausschliesslich hieraus (Server = cfg.PROVIDERS).
  window.MODEL_FAMILIES = parse("modelFamilies", []);
  window.MAX_RUN_FAMILIES = Number(config?.dataset.maxRunFamilies || 6);
  window.FREE_LIMIT = Number(config?.dataset.freeLimit || 0);

  window.trackUmamiEvent = function (eventName, eventData = {}) {
    if (!eventName || !window.umami || typeof window.umami.track !== "function") return;
    const blocked = /(email|mail|token|key|prompt|question|message|response|answer|password)/i;
    const safeData = {};
    Object.entries(eventData || {}).forEach(([key, value]) => {
      if (!key || blocked.test(key) || value == null) return;
      if (typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value))) {
        safeData[key] = value;
      } else if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed && trimmed.length <= 120 && !trimmed.includes("@")) safeData[key] = trimmed;
      }
    });
    window.umami.track(eventName, safeData);
  };

  try {
    if (localStorage.getItem("agentMode") === null) localStorage.setItem("agentMode", "true");
    if (localStorage.getItem("agentModePanelCollapsed") === null) {
      localStorage.setItem("agentModePanelCollapsed", "false");
    }
  } catch (_) { /* storage unavailable */ }

  document.addEventListener("DOMContentLoaded", () => {
    try {
      if (localStorage.getItem("agentMode") === "true") {
        document.body.classList.add("agent-mode-enabled");
      }
    } catch (_) { /* storage unavailable */ }
    const authTopActions = document.getElementById("authTopActions");
    const authState = window.__consensioAuthState;
    if (authTopActions && !(authState?.known && authState.uid)) {
      authTopActions.hidden = false;
    }
    try {
      if (document.documentElement.dataset.authUnavailable === "true") return;
      if (!localStorage.getItem("id_token")) return;
      const line = '<i class="skeleton skeleton-line skeleton-line-usage"></i>';
      const free = document.getElementById("freeUsageDisplay");
      const deep = document.getElementById("deepUsageDisplay");
      if (free) free.innerHTML = "Runs: " + line;
      if (deep) deep.innerHTML = "Deep Think: " + line;
      const bookmarks = document.getElementById("bookmarksContainer");
      if (bookmarks) {
        bookmarks.innerHTML = '<div class="skeleton skeleton-bookmark" aria-hidden="true"></div>'.repeat(4);
      }
      const login = document.getElementById("loginContainer");
      if (login) {
        login.hidden = false;
        login.innerHTML = '<span class="skeleton login-skeleton" aria-hidden="true" role="status" aria-label="Loading account"></span>';
      }
    } catch (_) { /* best-effort first paint */ }
  });
})();
