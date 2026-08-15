// =====================================================================
// agent-mode.js
// Agent Mode: gruppierter Modell-Lauf mit Timer, Status und erzwungenem
// Auto-Consensus. Ausgeschaltet bleibt die App im direkten Sechs-Antworten-
// Vergleich und startet keine Consensus-Pipeline. In eigene IIFE gekapselt.
// State (Status/Timer) ist
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

  // Default fuer neue Nutzer (seit 2026-07-27 auf ALLEN Geraeten, vorher nur
  // mobil): Agent Mode aktiv und Panel AUSGEKLAPPT — die Modellnamen sind
  // sofort sichtbar, die einzelnen Antwort-Boxen bleiben zu. Greift nur,
  // solange der Nutzer nie selbst gewaehlt hat (localStorage-Keys fehlen);
  // eine explizite Entscheidung (an/aus, auf/zu) bleibt erhalten.
  try {
    if (localStorage.getItem(AGENT_MODE_STORAGE_KEY) === null) {
      localStorage.setItem(AGENT_MODE_STORAGE_KEY, "true");
    }
    if (localStorage.getItem(AGENT_PANEL_COLLAPSED_KEY) === null) {
      localStorage.setItem(AGENT_PANEL_COLLAPSED_KEY, "false");
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

  // Beim zweiten Lauf stehen die Antwort-Boxen beim Start noch auf dem
  // Ergebnis des vorherigen Laufs: setAgentModeStatus("running") kommt vor
  // dem Zuruecksetzen der Boxen auf "pending" (query-send.js). Ohne diese
  // Sperre liest der erste Tick "complete", der monoton steigende Balken
  // rastet sofort auf 100 % ein und haengt dort den ganzen Lauf fest.
  const staleModelKeys = new Set();

  function isTerminalResponseState(state) {
    return state === "complete" || state === "error";
  }

  // Schätzt den Fortschritt eines Modells aus dem tatsächlichen Stream:
  // fertig ⇒ voll, während des Streams asymptotisch aus der Textlänge, davor
  // ein langsamer zeitbasierter Anlauf, damit der Balken sichtbar „lebt“.
  function computeModelProgress(pref) {
    const box = document.getElementById(pref.responseId);
    if (!box) return 0;
    const state = box.dataset.responseState || "";
    // Noch das alte Ergebnis in der Box: erst wenn der neue Lauf sie auf
    // "pending" zurueckgesetzt hat, zaehlt der Fortschritt wieder.
    if (staleModelKeys.has(pref.key)) {
      if (isTerminalResponseState(state)) return 0;
      staleModelKeys.delete(pref.key);
    }
    if (isTerminalResponseState(state)) return 1;
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
    staleModelKeys.clear();
    (window.App?.modelPrefs || []).forEach(pref => {
      const box = document.getElementById(pref.responseId);
      if (box && isTerminalResponseState(box.dataset.responseState || "")) {
        staleModelKeys.add(pref.key);
      }
    });
  }

  // ---- Einzelantworten als Vorschau statt als Scroll-Schacht --------------
  // Hinter "Compare answers" hatte jede Box ihren eigenen Innen-Scroll
  // (max-height + overflow-y in components-misc.css). Sechs private
  // Scroll-Bereiche in einer scrollenden Seite heissen: das Mausrad tut je
  // nach Zeigerposition etwas anderes, die Spalten enden auf verschiedenen
  // Hoehen, und lange Antworten sind abgeschnitten, ohne dass es jemand
  // ansagt - ausgerechnet in der Ansicht, deren einziger Zweck der Vergleich
  // ist. Stattdessen: gleich hohe Vorschauen mit Ausblendkante, und ein Knopf
  // oeffnet genau die eine Antwort ganz.
  //
  // Geklappt wird nur, was wirklich ueberlaeuft, und nie waehrend des
  // Streams: waehrend die Antwort noch waechst, will man sie wachsen sehen.
  const ANSWER_PREVIEW_HEIGHT = 300;
  const ANSWER_PREVIEW_SLACK = 48;

  function isAnswerSettled(box) {
    const content = box.querySelector(".collapsible-content");
    if (!content) return false;
    if (content.classList.contains("is-streaming")) return false;
    if (content.querySelector(".thinking-wrap")) return false;
    return isTerminalResponseState(box.dataset.responseState || "");
  }

  function answerToggleLabel(open) {
    return open ? "Show less" : "Show full answer";
  }

  function ensureAnswerToggle(box) {
    let btn = box.querySelector(".response-answer-more");
    if (btn) return btn;
    btn = document.createElement("button");
    btn.type = "button";
    btn.className = "response-answer-more";
    btn.addEventListener("click", function () {
      const open = box.dataset.answerOpen !== "1";
      box.dataset.answerOpen = open ? "1" : "0";
      box.classList.toggle("is-clamped", !open);
      btn.textContent = answerToggleLabel(open);
      btn.setAttribute("aria-expanded", String(open));
      window.App?.trackAppEvent?.("app_model_answer_expanded", {
        model: box.dataset.model || box.id,
        open: open
      });
    });
    box.appendChild(btn);
    return btn;
  }

  function syncAnswerPreviews() {
    document.querySelectorAll(".response-section > .response-box").forEach(box => {
      const content = box.querySelector(".collapsible-content");
      const existing = box.querySelector(".response-answer-more");
      const eligible = modelAnswersVisible
        && !box.classList.contains("excluded")
        && isAnswerSettled(box);
      if (!content || !eligible) {
        box.classList.remove("is-clamped");
        if (existing) existing.hidden = true;
        return;
      }
      // scrollHeight bleibt auch im geklappten Zustand die volle Hoehe
      // (overflow: hidden), der Test kippt also nicht hin und her.
      const overflows = content.scrollHeight
        > ANSWER_PREVIEW_HEIGHT + ANSWER_PREVIEW_SLACK;
      if (!overflows) {
        box.classList.remove("is-clamped");
        if (existing) existing.hidden = true;
        return;
      }
      const btn = ensureAnswerToggle(box);
      const open = box.dataset.answerOpen === "1";
      btn.hidden = false;
      btn.textContent = answerToggleLabel(open);
      btn.setAttribute("aria-expanded", String(open));
      box.classList.toggle("is-clamped", !open);
    });
  }

  let answerPreviewFrame = 0;
  function scheduleAnswerPreviewSync() {
    if (answerPreviewFrame) return;
    answerPreviewFrame = window.requestAnimationFrame(() => {
      answerPreviewFrame = 0;
      syncAnswerPreviews();
    });
  }

  let answerPreviewResizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(answerPreviewResizeTimer);
    answerPreviewResizeTimer = window.setTimeout(syncAnswerPreviews, 150);
  });

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
    // Auto Consensus ist eine Eigenschaft des Agent Mode, kein zweiter Modus.
    // Der gekoppelte Settings-Schalter zeigt den Zustand read-only:
    // an bedeutet Consensus, aus bedeutet ausschliesslich Modellantworten.
    const autoToggle = document.getElementById("autoConsensusToggle");
    if (!autoToggle) return;
    const autoWrap = autoToggle.closest(".settings-section");

    autoToggle.checked = !!enabled;
    localStorage.setItem("autoConsensus", String(!!enabled));
    autoToggle.disabled = true;
    autoWrap?.classList.add("is-agent-locked");
    autoToggle.title = enabled
      ? "Auto Consensus is always on in Agent Mode"
      : "Auto Consensus is available only in Agent Mode";
  }

  function updateAgentModeUI() {
    const enabled = isAgentModeEnabled();
    const panel = document.getElementById("agentModePanel");
    const switchEl = document.getElementById("agentModeSwitch");
    const menuSwitchEl = document.getElementById("agentModeMenuSwitch");
    const toggleSwitch = document.querySelector(".agent-mode-switch");
    const modelsEl = document.getElementById("agentModeModels");
    const statusEl = document.getElementById("agentModeStatus");
    const countEl = document.getElementById("agentModeCount");
    const titleEl = document.getElementById("agentModeTitle");
    const answersRow = document.getElementById("agentModeAnswersRow");
    const answersToggle = document.getElementById("agentModeAnswersToggle");
    const activeModels = getActiveAgentModels();

    // Seit 2026-07-27 haengt die Einzelantworten-Disclosure NICHT mehr am
    // Agent Mode. Sie ist eine der drei Aufklapp-Flaechen in der Fusszeile
    // ("Review differences · Compare answers · Verify sources") und war dort
    // in zwei von
    // drei Faellen unsichtbar, obwohl sie das Wichtigste dahinter oeffnet:
    // worauf die Antwort beruht. Der Footer selbst wird erst zusammen mit
    // einer fertigen Consensus-Antwort sichtbar; innerhalb dieses Footers ist
    // der Schalter deshalb immer da. Das vermeidet Sonderfaelle fuer manuell
    // gestartete Consensuses, Bookmarks und nachtraeglich ausgeschlossene
    // Modelle.

    document.body.classList.toggle("agent-mode-enabled", enabled);
    document.body.classList.toggle("agent-mode-running", enabled && agentModeStatus === "running");
    // "direct-comparison-active" beschreibt, was GERADE AUF DEM SCHIRM steht,
    // der Agent-Mode-Schalter dagegen, was der NAECHSTE Lauf tut. Nur das
    // Umlegen des Schalters raeumt den sichtbaren Direktvergleich weg (siehe
    // setAgentMode) — nicht jeder beilaeufige updateAgentModeUI-Aufruf. Sonst
    // riss die naechstbeste /ask-Antwort oder ein Bookmark-Restore die
    // wiederhergestellte Vergleichsansicht wieder ein und liess nur die Frage
    // stehen.
    // Hero-Desktop zeigt die Response-Boxen nur ohne Agent Mode; inert/
    // aria-hidden muessen der CSS-Sichtbarkeit folgen (app-core.js).
    if (typeof window.syncHeroResponseAccess === "function") {
      window.syncHeroResponseAccess();
    }
    document.body.classList.toggle(
      "agent-mode-show-answers",
      modelAnswersVisible
    );

    if (answersRow) answersRow.hidden = false;
    if (answersToggle) {
      const label = modelAnswersVisible ? "Hide answers" : "Compare answers";
      answersToggle.setAttribute("aria-expanded", String(modelAnswersVisible));
      answersToggle.title = label;
      answersToggle.setAttribute("aria-label", label);
      // Gezielt das Label, nicht "das erste span": der Chip traegt seit
      // 2026-07-28 auch einen Chevron und eine Zahl.
      const labelEl = answersToggle.querySelector(".consensus-tab-label");
      if (labelEl) {
        labelEl.textContent = label;
        // Kurzform fuer die Telefon-Leiste (siehe shell.css): dort steht das
        // Substantiv allein, damit die drei Knoepfe einzeilig bleiben.
        labelEl.dataset.short = modelAnswersVisible ? "Hide" : "Answers";
      }
    }

    if (switchEl) switchEl.checked = enabled;
    if (menuSwitchEl) menuSwitchEl.checked = enabled;
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

    // Nach dem Layout messen: die Boxen sind in genau diesem Aufruf sichtbar
    // geworden (agent-mode-show-answers), vorher ist ihre Hoehe 0.
    scheduleAnswerPreviewSync();
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
      // Der Schalter wechselt den Modus: ein sichtbarer Direktvergleich (oder
      // ein aus einem Bookmark wiederhergestellter) gehoert zum alten Modus.
      // (updateAgentModeUI unten synchronisiert inert/aria-hidden danach.)
      if (nextEnabled) {
        document.body.classList.remove("direct-comparison-active");
      }
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
        // Ein neuer Lauf bringt neue Antworten: die Entscheidung, welche
        // davon ganz aufgeklappt war, gilt fuer die alten.
        document.querySelectorAll(".response-box[data-answer-open]")
          .forEach(box => { delete box.dataset.answerOpen; });
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
    // Der Modellstatus bleibt fuer jeden Lauf zentral. Nur Agent Mode reicht
    // ihn an die gefuehrte Pipeline weiter; der Direktvergleich raeumt sie ab.
    if (isAgentModeEnabled()) {
      window.App?.consensusPipeline?.onQueryStatus?.(status);
    } else {
      window.App?.consensusPipeline?.dismiss?.();
    }
  }

  function setModelAnswersVisible(visible, options = {}) {
    const nextVisible = !!visible;
    const changed = modelAnswersVisible !== nextVisible;
    modelAnswersVisible = nextVisible;
    if (changed && options.track) {
      window.App?.trackAppEvent?.("app_agent_mode_answers_toggled", {
        visible: modelAnswersVisible
      });
    }
    updateAgentModeUI();
    if (changed && nextVisible) {
      // Derselbe dezente Reveal wie bei Differences/Sources: erst nachdem die
      // CSS-Klasse die Boxen sichtbar gemacht hat, den Anfang der ersten
      // Antwort mit "nearest" ins Bild holen.
      requestAnimationFrame(() => {
        const firstAnswer = document.querySelector(
          ".response-section > .response-box:not(.excluded)"
        );
        firstAnswer?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    }
    return changed;
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
      setModelAnswersVisible(!modelAnswersVisible, { track: true });
    });
  }

  const agentModeMenuSwitch = document.getElementById("agentModeMenuSwitch");
  if (agentModeMenuSwitch) {
    agentModeMenuSwitch.addEventListener("change", function () {
      setAgentMode(this.checked, { persist: true });
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

  // Der gefuehrte Lauf (consensus-progress.js) zeigt dieselben Modell-Balken
  // ausserhalb des Agent Mode. Statt die Schaetzung zu duplizieren, liest er
  // hier den bereits berechneten, monoton steigenden Fortschritt — indiziert
  // nach Response-Box-ID, weil das die ID ist, die er ohnehin in der Hand hat.
  window.App = window.App || {};
  window.App.agentMode = {
    streamProgressByResponseId() {
      const out = {};
      (window.App?.modelPrefs || []).forEach(pref => {
        const next = computeModelProgress(pref);
        const value = Math.max(modelProgress.get(pref.key) || 0, next);
        modelProgress.set(pref.key, value);
        out[pref.responseId] = value;
      });
      return out;
    },
    resetStreamProgress: resetModelProgress,
    // Claim- und Difference-Spruenge muessen ein verborgenes Ziel zuerst
    // idempotent aufdecken, ohne dafuer den Agent Mode umzuschalten.
    showModelAnswers() {
      return setModelAnswersVisible(true);
    }
  };
})();
