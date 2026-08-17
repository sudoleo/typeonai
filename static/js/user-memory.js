/**
 * Das nutzereigene Gedaechtnis in den Einstellungen.
 *
 * Der Server ist die einzige Quelle: das Profil liegt am Konto, nicht im
 * Browser. Deshalb gibt es hier bewusst KEINEN localStorage-Spiegel -- ein
 * zweiter Stand, der nach einem Geraetewechsel still danebenliegt, waere genau
 * die Verwirrung, die dieses Feature aufloesen soll.
 *
 * Der Schalter speichert immer den zuletzt GESPEICHERTEN Textstand, nie den
 * Entwurf in den Feldern: sonst committet ein Klick auf "Use my memory"
 * nebenbei einen halb getippten Satz.
 */
(function () {
  "use strict";

  const FIELDS = ["role", "focus", "style", "constraints", "notes"];
  const FIELD_INPUT_IDS = {
    role: "memoryRoleInput",
    focus: "memoryFocusInput",
    style: "memoryStyleInput",
    constraints: "memoryConstraintsInput",
    notes: "memoryNotesInput"
  };

  const state = {
    saved: null,      // letzter vom Server bestaetigter Stand
    loaded: false,
    loading: false,
    saving: false,
    uid: null
  };

  function emptyProfile() {
    const profile = { enabled: true };
    FIELDS.forEach(field => { profile[field] = ""; });
    return profile;
  }

  function els() {
    const inputs = {};
    FIELDS.forEach(field => {
      inputs[field] = document.getElementById(FIELD_INPUT_IDS[field]);
    });
    return {
      section: document.getElementById("memorySettingsSection"),
      enabled: document.getElementById("memoryEnabledSwitch"),
      saveBtn: document.getElementById("saveMemoryBtn"),
      clearBtn: document.getElementById("clearMemoryBtn"),
      status: document.getElementById("memoryStatus"),
      inputs
    };
  }

  function currentUser() {
    const user = window.auth?.currentUser;
    return user && user.uid ? user : null;
  }

  function setStatus(message, tone) {
    const { status } = els();
    if (!status) return;
    status.textContent = message || "";
    status.dataset.tone = tone || "";
  }

  async function api(method, body) {
    const user = currentUser();
    if (!user) throw new Error("Please log in first.");
    const uid = user.uid;
    const token = await user.getIdToken();
    if (window.auth?.currentUser?.uid !== uid) throw new Error("Authentication changed.");
    const response = await fetch("/api/my/memory", {
      method,
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + token },
      body: body ? JSON.stringify(body) : undefined
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (window.auth?.currentUser?.uid !== uid) throw new Error("Authentication changed.");
    if (!response.ok) {
      const error = new Error(data.detail || data.error || ("HTTP " + response.status));
      error.status = response.status;
      throw error;
    }
    return data.memory || emptyProfile();
  }

  function readForm() {
    const { enabled, inputs } = els();
    const profile = { enabled: enabled ? !!enabled.checked : true };
    FIELDS.forEach(field => {
      profile[field] = (inputs[field]?.value || "").trim();
    });
    return profile;
  }

  function writeForm(profile) {
    const { enabled, inputs } = els();
    if (enabled) enabled.checked = profile.enabled !== false;
    FIELDS.forEach(field => {
      if (inputs[field]) inputs[field].value = profile[field] || "";
    });
    updateCounts();
  }

  // Die kurzen Felder zeigen den Zaehler erst nahe ihrer Grenze. Bei der grossen
  // Notebox ist die Kapazitaet selbst relevante Information und bleibt sichtbar.
  const COUNT_VISIBLE_RATIO = 0.8;

  function updateCounts() {
    document.querySelectorAll("[data-memory-count-for]").forEach(node => {
      const input = document.getElementById(node.dataset.memoryCountFor);
      if (!input) return;
      const max = Number(input.getAttribute("maxlength")) || 0;
      const used = (input.value || "").length;
      const always = node.dataset.alwaysVisible === "true";
      const near = always || (max > 0 && used >= max * COUNT_VISIBLE_RATIO);
      node.textContent = near ? `${used.toLocaleString()}/${max.toLocaleString()}` : "";
      node.dataset.near = near ? "true" : "";
      node.dataset.full = max && used >= max ? "true" : "";
    });
  }

  function isDirty() {
    if (!state.saved) return false;
    const form = readForm();
    return FIELDS.some(field => (form[field] || "") !== (state.saved[field] || ""));
  }

  function syncControls() {
    const { section, enabled, saveBtn, clearBtn, inputs } = els();
    if (!section) return;
    const signedIn = !!currentUser();
    section.dataset.signedIn = signedIn ? "true" : "false";

    const disabled = !signedIn || state.loading || state.saving;
    if (enabled) enabled.disabled = disabled;
    FIELDS.forEach(field => {
      if (inputs[field]) inputs[field].disabled = disabled;
    });
    if (clearBtn) clearBtn.disabled = disabled;
    if (saveBtn) {
      saveBtn.disabled = disabled || (state.loaded && !isDirty());
      saveBtn.textContent = state.saving ? "Saving…" : "Save memory";
    }
  }

  async function load(force) {
    const user = currentUser();
    if (!user) {
      state.loaded = false;
      state.saved = null;
      state.uid = null;
      writeForm(emptyProfile());
      setStatus("Log in to set up your memory — it lives on your account, not in this browser.", "muted");
      syncControls();
      return;
    }
    if (state.loaded && state.uid === user.uid && !force) {
      syncControls();
      return;
    }

    state.loading = true;
    setStatus("Loading…", "muted");
    syncControls();
    try {
      const profile = await api("GET");
      state.saved = profile;
      state.uid = user.uid;
      state.loaded = true;
      writeForm(profile);
      setStatus("", "");
    } catch (error) {
      setStatus(error.message || "Memory could not be loaded.", "error");
    } finally {
      state.loading = false;
      syncControls();
    }
  }

  async function persist(profile, successMessage, options) {
    const rewriteFields = options?.rewriteFields !== false;
    state.saving = true;
    syncControls();
    try {
      const saved = await api("PUT", profile);
      state.saved = saved;
      state.loaded = true;
      if (rewriteFields) {
        // Der Server normalisiert (Whitespace, Laenge, Rahmenmarken). Zurueck-
        // schreiben, damit das Feld zeigt, was tatsaechlich gespeichert ist.
        writeForm(saved);
      } else {
        // Nur der Schalter wurde geschrieben. Die Textfelder bleiben, wie der
        // Nutzer sie gerade hat -- ein Klick auf "Use my memory" darf einen
        // halb getippten Satz weder speichern noch wegwerfen.
        const { enabled } = els();
        if (enabled) enabled.checked = saved.enabled !== false;
      }
      setStatus(successMessage, "ok");
      window.App?.trackAppEvent?.("app_memory_saved");
      return true;
    } catch (error) {
      setStatus(error.message || "Memory could not be saved.", "error");
      return false;
    } finally {
      state.saving = false;
      syncControls();
    }
  }

  async function save() {
    if (!currentUser()) {
      setStatus("Please log in first.", "error");
      return;
    }
    const profile = readForm();
    const empty = FIELDS.every(field => !profile[field]);
    await persist(profile, empty ? "Memory cleared." : "Saved. Your next run starts with this.");
  }

  async function toggleEnabled() {
    const { enabled } = els();
    if (!enabled) return;
    if (!currentUser()) {
      enabled.checked = !enabled.checked;
      setStatus("Please log in first.", "error");
      return;
    }
    const next = { ...(state.saved || emptyProfile()), enabled: enabled.checked };
    const ok = await persist(
      next,
      enabled.checked ? "Memory is on again." : "Memory paused. Runs go out without it.",
      { rewriteFields: false }
    );
    if (!ok) enabled.checked = !enabled.checked;
  }

  function clearFields() {
    const { inputs } = els();
    FIELDS.forEach(field => {
      if (inputs[field]) inputs[field].value = "";
    });
    updateCounts();
    // Bewusst nicht sofort speichern: Loeschen soll dieselbe bestaetigte
    // Handlung sein wie jede andere Aenderung.
    setStatus("Cleared. Press Save to apply.", "muted");
    syncControls();
  }

  function settingsModalIsOpen() {
    const modal = document.getElementById("systemPromptModal");
    return !!modal && getComputedStyle(modal).display !== "none";
  }

  function bind() {
    if (window.__userMemoryBound) return;
    const { section } = els();
    if (!section) return;
    window.__userMemoryBound = true;

    const { enabled, saveBtn, clearBtn, inputs } = els();
    saveBtn?.addEventListener("click", save);
    clearBtn?.addEventListener("click", clearFields);
    enabled?.addEventListener("change", toggleEnabled);
    FIELDS.forEach(field => {
      inputs[field]?.addEventListener("input", () => {
        updateCounts();
        syncControls();
      });
    });

    window.addEventListener("consensio:auth-state", event => {
      const uid = event.detail?.uid || null;
      if (uid === state.uid) return;
      state.loaded = false;
      state.saved = null;
      state.uid = null;
      // Nur den Stand verwerfen, NICHT nachladen: dieses Ereignis feuert bei
      // jedem Seitenaufruf eines eingeloggten Kontos. Ein Fetch hier haette den
      // Read, den der Modal-Oeffner bewusst aufschiebt, an jeden Aufruf
      // gehaengt. Steht das Fenster gerade offen, wird sofort nachgezogen --
      // sonst zeigte es weiter das Profil des vorigen Kontos.
      if (settingsModalIsOpen() || !uid) {
        load();
        return;
      }
      writeForm(emptyProfile());
      setStatus("", "");
      syncControls();
    });

    // Die Einstellungen sind ein Modal: laden, wenn es tatsaechlich geoeffnet
    // wird, statt bei jedem Seitenaufruf einen Firestore-Read zu bezahlen.
    document.getElementById("editSystemPromptBtn")?.addEventListener("click", () => load());

    writeForm(emptyProfile());
    syncControls();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }

  window.App = window.App || {};
  window.App.userMemory = { load, save, isDirty };
})();
