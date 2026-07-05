// =====================================================================
// agent-mode.js
// Agent Mode: gruppierter Modell-Lauf mit Timer, Status und erzwungenem
// Auto-Consensus. In eigene IIFE gekapselt. State (Status/Timer) ist
// modul-privat; agentModeStatus wird extern via window.isAgentModeRunning()
// gelesen.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.setAgentModeStatus, window.updateAgentModeUI,
// window.isAgentModeEnabled, window.setAgentMode, window.isAgentModeRunning.
// Abhaengigkeiten: window.App.{modelPrefs,deepThinkModelLabels,
// getModelOptionLabel,getSelectedModelCount,trackAppEvent,initCustomModelPicker},
// window.updateConsensusButtonAvailability.
// =====================================================================

(function () {
  const AGENT_MODE_STORAGE_KEY = "agentMode";
  let agentModeStatus = "idle";
  let agentModeStatusMessage = "";
  let agentModeTimerStartedAt = null;
  let agentModeTimerElapsedMs = 0;
  let agentModeTimerInterval = null;

  function isAgentModeEnabled() {
    return localStorage.getItem(AGENT_MODE_STORAGE_KEY) === "true";
  }

  function formatAgentElapsed(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function updateAgentModeTimerDisplay() {
    const timerEl = document.getElementById("agentModeTimer");
    if (!timerEl) return;
    const elapsed = agentModeTimerStartedAt
      ? Date.now() - agentModeTimerStartedAt
      : agentModeTimerElapsedMs;
    const isVisible = !!agentModeTimerStartedAt || agentModeTimerElapsedMs > 0;
    timerEl.classList.toggle("is-visible", isVisible);
    timerEl.textContent = `Elapsed ${formatAgentElapsed(elapsed)}`;
  }

  function startAgentModeTimer() {
    if (agentModeTimerStartedAt) return;
    agentModeTimerStartedAt = Date.now();
    agentModeTimerElapsedMs = 0;
    window.clearInterval(agentModeTimerInterval);
    updateAgentModeTimerDisplay();
    agentModeTimerInterval = window.setInterval(updateAgentModeTimerDisplay, 1000);
  }

  function stopAgentModeTimer() {
    if (agentModeTimerStartedAt) {
      agentModeTimerElapsedMs = Date.now() - agentModeTimerStartedAt;
    }
    agentModeTimerStartedAt = null;
    window.clearInterval(agentModeTimerInterval);
    agentModeTimerInterval = null;
    updateAgentModeTimerDisplay();
  }

  function resetAgentModeTimer() {
    agentModeTimerStartedAt = null;
    agentModeTimerElapsedMs = 0;
    window.clearInterval(agentModeTimerInterval);
    agentModeTimerInterval = null;
    updateAgentModeTimerDisplay();
  }

  function getActiveAgentModels() {
    const deepSearchActive = !!document.getElementById("deepSearchToggle")?.checked;
    return window.App.modelPrefs
      .filter(pref => document.getElementById(pref.checkId)?.checked)
      .map(pref => {
        const select = document.getElementById(pref.selectId);
        const responseBox = document.getElementById(pref.responseId);
        const displayedText = document.getElementById(pref.textId)?.textContent || "";
        const selectedText = deepSearchActive
          ? (window.App.deepThinkModelLabels[pref.key] || displayedText)
          : (window.App.getModelOptionLabel(select?.options[select.selectedIndex]) || select?.value || displayedText);
        const modelText = selectedText.trim();
        return {
          pref,
          label: pref.label,
          model: modelText,
          responseState: responseBox?.dataset?.responseState || "",
          usesDeepThinkModel: deepSearchActive
        };
      });
  }

  function syncAgentModePicker(agentSelect, pref) {
    const sourceSelect = document.getElementById(pref.selectId);
    const labelText = document.getElementById(pref.textId);
    if (!sourceSelect || !agentSelect) return;

    sourceSelect.value = agentSelect.value;
    localStorage.setItem("pref_select_" + pref.key, agentSelect.value);
    if (labelText) {
      const selectedLabel = window.App.getModelOptionLabel(agentSelect.options[agentSelect.selectedIndex]) || agentSelect.value;
      labelText.textContent = selectedLabel;
      labelText.title = `Choose model: ${selectedLabel}`;
    }
    sourceSelect.dispatchEvent(new Event("change", { bubbles: true }));
    updateAgentModeUI();
  }

  function getAgentModeStatusText(activeModels) {
    const count = activeModels.length;
    if (count === 0) return "No models selected.";
    if (agentModeStatus === "running") return "Querying selected models in parallel.";
    if (agentModeStatus === "complete") return "Model responses are ready for consensus.";
    if (agentModeStatus === "canceled") return "Request canceled.";
    if (agentModeStatus === "error") return agentModeStatusMessage || "The request could not be completed.";
    return "Ready for a grouped model run.";
  }

  function setAutoConsensusForAgentMode(enabled) {
    // Auto Consensus lebt jetzt in den Einstellungen. In Agent Mode bleibt
    // es zwingend an und wird ausgegraut, damit der Konsens garantiert läuft.
    const autoToggle = document.getElementById("autoConsensusToggle");
    if (!autoToggle) return;
    const autoWrap = autoToggle.closest(".settings-section");

    if (enabled) {
      autoToggle.checked = true;
      localStorage.setItem("autoConsensus", "true");
    }

    autoToggle.disabled = !!enabled;
    autoWrap?.classList.toggle("is-agent-locked", !!enabled);
    autoToggle.title = enabled
      ? "Auto Consensus is always on in Agent Mode"
      : "";
  }

  function updateAgentModeUI() {
    const enabled = isAgentModeEnabled();
    const panel = document.getElementById("agentModePanel");
    const switchEl = document.getElementById("agentModeSwitch");
    const toggleBtn = document.getElementById("toggleAllButton");
    const toggleSwitch = document.querySelector(".agent-mode-switch");
    const modelsEl = document.getElementById("agentModeModels");
    const statusEl = document.getElementById("agentModeStatus");
    const countEl = document.getElementById("agentModeCount");
    const titleEl = document.getElementById("agentModeTitle");
    const activeModels = getActiveAgentModels();

    document.body.classList.toggle("agent-mode-enabled", enabled);
    document.body.classList.toggle("agent-mode-running", enabled && agentModeStatus === "running");

    if (switchEl) switchEl.checked = enabled;
    if (toggleBtn) {
      toggleBtn.checked = enabled;
      toggleBtn.classList.toggle("is-agent-active", enabled);
      toggleBtn.setAttribute("aria-checked", String(enabled));
      toggleBtn.title = enabled ? "Disable Agent Mode" : "Enable Agent Mode";
      toggleBtn.setAttribute("aria-label", toggleBtn.title);
    }
    if (toggleSwitch) {
      toggleSwitch.title = enabled ? "Disable Agent Mode" : "Enable Agent Mode";
      toggleSwitch.setAttribute("aria-label", toggleSwitch.title);
    }
    setAutoConsensusForAgentMode(enabled);
    if (panel) panel.setAttribute("aria-hidden", String(!enabled));

    if (titleEl) {
      titleEl.textContent = agentModeStatus === "running" ? "Models are working" : "Selected models";
    }
    if (countEl) {
      countEl.textContent = `${activeModels.length} ${activeModels.length === 1 ? "model" : "models"}`;
    }
    if (statusEl) {
      statusEl.textContent = getAgentModeStatusText(activeModels);
    }
    if (modelsEl) {
      modelsEl.innerHTML = "";
      activeModels.forEach(modelInfo => {
        const chip = document.createElement("span");
        chip.className = "agent-mode-chip";
        chip.textContent = modelInfo.model
          ? `${modelInfo.label} · ${modelInfo.model}`
          : modelInfo.label;
        chip.setAttribute("role", "group");
        chip.setAttribute("aria-label", `Choose ${modelInfo.label} model`);
        chip.textContent = "";
        if (modelInfo.responseState) {
          chip.dataset.responseState = modelInfo.responseState;
        }

        const chipLabel = document.createElement("span");
        chipLabel.className = "agent-mode-chip-label";
        chipLabel.textContent = modelInfo.label;
        chip.appendChild(chipLabel);

        const sourceSelect = document.getElementById(modelInfo.pref.selectId);
        if (sourceSelect && !modelInfo.usesDeepThinkModel) {
          const picker = document.createElement("select");
          picker.className = "agent-mode-picker";
          picker.setAttribute("aria-label", `Choose ${modelInfo.label} model`);
          Array.from(sourceSelect.options).forEach(option => {
            picker.appendChild(option.cloneNode(true));
          });
          picker.value = sourceSelect.value;
          picker.addEventListener("change", function () {
            syncAgentModePicker(this, modelInfo.pref);
          });
          chip.appendChild(picker);
          window.App.initCustomModelPicker(picker);
        } else if (modelInfo.model) {
          const chipModel = document.createElement("span");
          chipModel.className = "agent-mode-chip-model";
          chipModel.textContent = modelInfo.model;
          chip.appendChild(chipModel);
        }

        if (modelInfo.responseState === "complete") {
          const done = document.createElement("span");
          done.className = "agent-mode-chip-done";
          done.setAttribute("aria-hidden", "true");
          done.title = `${modelInfo.label} response complete`;
          chip.appendChild(done);
          chip.setAttribute("aria-label", `${modelInfo.label} response complete`);
        }
        modelsEl.appendChild(chip);
      });
    }
  }

  function setAgentMode(enabled, options = {}) {
    const { persist = false } = options;
    const nextEnabled = !!enabled;
    const wasEnabled = isAgentModeEnabled();
    if (persist) {
      localStorage.setItem(AGENT_MODE_STORAGE_KEY, String(nextEnabled));
    }
    if (wasEnabled !== nextEnabled) {
      document.body.classList.add("agent-mode-transitioning");
      window.setTimeout(() => {
        document.body.classList.remove("agent-mode-transitioning");
      }, 340);
      if (persist) {
        window.App.trackAppEvent("app_agent_mode_changed", {
          enabled: nextEnabled,
          selected_models: window.App.getSelectedModelCount()
        });
      }
    }
    updateAgentModeUI();
  }

  function setAgentModeStatus(status, message = "") {
    if (status === "running") {
      agentModeStatusMessage = "";
      startAgentModeTimer();
    } else if (status === "complete" || status === "canceled" || status === "error") {
      stopAgentModeTimer();
    } else if (status === "idle") {
      agentModeStatusMessage = "";
      resetAgentModeTimer();
    }
    if (message) {
      agentModeStatusMessage = message;
    } else if (status !== "error") {
      agentModeStatusMessage = "";
    }
    agentModeStatus = status;
    updateAgentModeUI();
    if (typeof window.updateConsensusButtonAvailability === "function") {
      window.updateConsensusButtonAvailability();
    }
  }

  window.setAgentModeStatus = setAgentModeStatus;
  window.updateAgentModeUI = updateAgentModeUI;
  window.isAgentModeEnabled = isAgentModeEnabled;
  window.setAgentMode = setAgentMode;

  // Getter fuer den (modul-privaten) Status, damit Query-/Consensus-Code
  // weiterhin auf den "running"-Zustand pruefen kann.
  window.isAgentModeRunning = function () {
    return agentModeStatus === "running";
  };
})();
