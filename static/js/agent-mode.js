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
  const AGENT_PANEL_COLLAPSED_KEY = "agentModePanelCollapsed";

  // Mobile-Default: Agent Mode aktiv und Panel AUSGEKLAPPT (die Modellnamen
  // sind sofort sichtbar) — aber nur, wenn der Nutzer noch nie selbst gewählt
  // hat (localStorage-Keys fehlen). Eine explizite Entscheidung (an/aus,
  // auf/zu) bleibt auf allen Geräten erhalten.
  try {
    const isMobileViewport = window.matchMedia("(max-width: 640px)").matches;
    if (isMobileViewport) {
      if (localStorage.getItem(AGENT_MODE_STORAGE_KEY) === null) {
        localStorage.setItem(AGENT_MODE_STORAGE_KEY, "true");
      }
      if (localStorage.getItem(AGENT_PANEL_COLLAPSED_KEY) === null) {
        localStorage.setItem(AGENT_PANEL_COLLAPSED_KEY, "false");
      }
    }
  } catch (e) { /* localStorage gesperrt: Default bleibt aus */ }

  let agentModeStatus = "idle";
  let agentModeStatusMessage = "";
  let agentModeTimerStartedAt = null;
  let agentModeTimerElapsedMs = 0;
  let agentModeTimerInterval = null;
  // Session-only disclosure: every new grouped run starts in the clean view.
  let modelAnswersVisible = false;

  // Stream-Fortschritt pro Modell (0..1), monoton steigend innerhalb eines
  // Runs. Treibt den „Ladebalken“ in jedem Chip (CSS-Variable --stream-progress).
  const modelProgress = new Map();        // pref.key -> 0..1
  const modelStreamStartedAt = new Map(); // pref.key -> ts (sanfter Anlauf)
  let agentProgressTicker = null;

  // Schätzt den Fortschritt eines Modells aus dem tatsächlichen Stream:
  // fertig ⇒ voll, während des Streams asymptotisch aus der Textlänge, davor
  // ein langsamer zeitbasierter Anlauf, damit der Balken sichtbar „lebt“.
  function computeModelProgress(pref) {
    const box = document.getElementById(pref.responseId);
    if (!box) return 0;
    const state = box.dataset.responseState || "";
    if (state === "complete" || state === "error") return 1;
    const contentEl = box.querySelector(".collapsible-content");
    const streaming = !!contentEl && contentEl.classList.contains("is-streaming");
    if (streaming) {
      const chars = (contentEl.textContent || "").trim().length;
      // Asymptotisch: viel Text ⇒ Balken fast voll, aber nie ganz — die vollen
      // 100 % kommen erst mit dem „complete“-Status.
      const eased = 1 - Math.exp(-chars / 420);
      return Math.min(0.92, 0.12 + eased * 0.8);
    }
    // Noch kein Token: langsamer Anlauf über die Zeit (bis ~10 %).
    const startedAt = modelStreamStartedAt.get(pref.key) || Date.now();
    modelStreamStartedAt.set(pref.key, startedAt);
    return Math.min(0.1, (Date.now() - startedAt) / 26000);
  }

  function applyModelProgress() {
    (window.App?.modelPrefs || []).forEach(pref => {
      const next = computeModelProgress(pref);
      const value = Math.max(modelProgress.get(pref.key) || 0, next); // monoton
      modelProgress.set(pref.key, value);
      const chip = document.querySelector(
        `.agent-mode-chip[data-model-key="${pref.key}"]`
      );
      if (chip) chip.style.setProperty("--stream-progress", value.toFixed(3));
    });
  }

  function startAgentProgressTicker() {
    window.clearInterval(agentProgressTicker);
    applyModelProgress();
    agentProgressTicker = window.setInterval(applyModelProgress, 120);
  }

  function stopAgentProgressTicker() {
    window.clearInterval(agentProgressTicker);
    agentProgressTicker = null;
  }

  function resetModelProgress() {
    modelProgress.clear();
    modelStreamStartedAt.clear();
  }

  function isAgentModeEnabled() {
    return localStorage.getItem(AGENT_MODE_STORAGE_KEY) === "true";
  }

  function isAgentPanelCollapsed() {
    return localStorage.getItem(AGENT_PANEL_COLLAPSED_KEY) === "true";
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
          hasAnswer: Boolean(responseBox?.querySelector(".collapsible-content")?.textContent?.trim()),
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
    const answersRow = document.getElementById("agentModeAnswersRow");
    const answersToggle = document.getElementById("agentModeAnswersToggle");
    const activeModels = getActiveAgentModels();
    const hasModelAnswers = activeModels.some(model => model.hasAnswer || model.responseState === "complete");

    if (!enabled || !hasModelAnswers) modelAnswersVisible = false;

    document.body.classList.toggle("agent-mode-enabled", enabled);
    document.body.classList.toggle("agent-mode-running", enabled && agentModeStatus === "running");
    // Hero-Desktop zeigt die Response-Boxen nur ohne Agent Mode; inert/
    // aria-hidden muessen der CSS-Sichtbarkeit folgen (app-core.js).
    if (typeof window.syncHeroResponseAccess === "function") {
      window.syncHeroResponseAccess();
    }
    document.body.classList.toggle(
      "agent-mode-show-answers",
      enabled && hasModelAnswers && modelAnswersVisible
    );

    if (answersRow) answersRow.hidden = !enabled || !hasModelAnswers;
    if (answersToggle) {
      const label = modelAnswersVisible ? "Hide model answers" : "Show model answers";
      answersToggle.setAttribute("aria-expanded", String(modelAnswersVisible));
      answersToggle.title = label;
      answersToggle.setAttribute("aria-label", label);
      const labelEl = answersToggle.querySelector("span");
      if (labelEl) labelEl.textContent = label;
    }

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

    // Eingeklappter Zustand: Panel wird zur Kompaktzeile (Titel, beantwortete
    // Modelle, Laufzeit); Chips/Status sind per CSS ausgeblendet.
    const collapsed = isAgentPanelCollapsed();
    if (panel) panel.classList.toggle("is-collapsed", collapsed);
    const collapseBtn = document.getElementById("agentModeCollapseBtn");
    if (collapseBtn) {
      collapseBtn.setAttribute("aria-expanded", String(!collapsed));
      collapseBtn.title = collapsed ? "Expand to configure models" : "Collapse Agent Mode panel";
      collapseBtn.setAttribute("aria-label", collapseBtn.title);
    }
    const answeredEl = document.getElementById("agentModeAnswered");
    if (answeredEl) {
      const answeredCount = activeModels.filter(m => m.responseState === "complete").length;
      answeredEl.textContent = `${answeredCount}/${activeModels.length} answered`;
      answeredEl.hidden = !collapsed;
    }
    const collapsedHintEl = document.getElementById("agentModeCollapsedHint");
    if (collapsedHintEl) collapsedHintEl.hidden = !collapsed;

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
        chip.dataset.modelKey = modelInfo.pref.key;
        // Gespeicherten Fortschritt sofort anlegen, damit der Balken beim
        // Neuaufbau der Chips (z. B. wenn ein Modell fertig wird) nicht auf 0
        // zurückspringt.
        chip.style.setProperty(
          "--stream-progress",
          (modelProgress.get(modelInfo.pref.key) || 0).toFixed(3)
        );
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
      modelAnswersVisible = false;
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
      if (agentModeStatus !== "running") {
        modelAnswersVisible = false;
        resetModelProgress();
      }
      agentModeStatusMessage = "";
      startAgentModeTimer();
      if (isAgentModeEnabled()) startAgentProgressTicker();
    } else if (status === "complete" || status === "canceled" || status === "error") {
      stopAgentModeTimer();
      applyModelProgress(); // fertige Modelle auf 100 % schnappen lassen
      stopAgentProgressTicker();
    } else if (status === "idle") {
      modelAnswersVisible = false;
      agentModeStatusMessage = "";
      resetAgentModeTimer();
      resetModelProgress();
      stopAgentProgressTicker();
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
    // Zentraler Status-Hub für JEDEN Lauf (auch ohne Agent Mode): die
    // Fortschritts-Pipeline unter dem Input hört hier mit (consensus-progress.js).
    window.App?.consensusPipeline?.onQueryStatus?.(status);
  }

  // Einklapp-Pfeil oben rechts im Panel (Zustand wird gemerkt).
  const agentCollapseBtn = document.getElementById("agentModeCollapseBtn");
  if (agentCollapseBtn) {
    agentCollapseBtn.addEventListener("click", function () {
      const next = !isAgentPanelCollapsed();
      localStorage.setItem(AGENT_PANEL_COLLAPSED_KEY, String(next));
      if (window.App && typeof window.App.trackAppEvent === "function") {
        window.App.trackAppEvent("app_agent_mode_panel_toggled", { collapsed: next });
      }
      updateAgentModeUI();
    });
  }

  const agentAnswersToggle = document.getElementById("agentModeAnswersToggle");
  if (agentAnswersToggle) {
    agentAnswersToggle.addEventListener("click", function () {
      modelAnswersVisible = !modelAnswersVisible;
      window.App?.trackAppEvent?.("app_agent_mode_answers_toggled", {
        visible: modelAnswersVisible
      });
      updateAgentModeUI();
    });
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
