// =====================================================================
// model-picker.js
// Modell-Auswahl + Custom-Picker-UI. In eigene IIFE gekapselt.
// Drei Teile:
//   1) Model-Selection-State: Wiederherstellen/Persistieren der pro-Provider
//      Auswahl (Dropdown + Ein/Ausschluss-Checkbox), Tier-Defaults, Reorder-
//      Animation beim Ein-/Ausschliessen.
//   2) Custom-Model-Picker: das eigene, getunte Listbox-Dropdown, das die
//      nativen <select> ueberlagert (expandedModelPicker ist modul-privat).
//   3) Consensus-Presets: der Consensus-Picker zeigt primaer die Presets
//      Daily/Balanced/High Quality (window.CONSENSUS_PRESETS) plus "Custom" fuer
//      die volle Modell-Liste. Ein Preset setzt die sechs Antwortmodelle und
//      die Consensus-Engine gemeinsam. High Quality bleibt Pro-only.
// Extrahiert aus templates/index.html (initApp-Closure), verhaltenserhaltend.
// Exporte:
//   window.restoreModelSelections, window.syncCustomModelPickers,
//   window.App.{applyTierDefaultModels, setModelSelectionState,
//   openModelPicker, collapseExpandedModelPicker, initCustomModelPicker}.
// Abhaengigkeiten: window.App.{modelPrefs, getModelOptionLabel,
//   getSelectedModelCount, trackAppEvent}, window.updateAgentModeUI,
//   window.updateConsensusButtonAvailability,
//   window.PRO_DEFAULT_MODELS, window.FREE_DEFAULT_MODELS.
// Das Wiring (Event-Listener, Init-Aufrufe) lebt weiterhin in initApp und
// nutzt diese Exporte ueber lokale Aliase (Uebergangsbus, siehe app-core.js).
// =====================================================================

(function () {
  // --- MODEL PREFERENCES WIEDERHERSTELLUNG ---

  function setPickerToValue(select, labelText, value) {
    if (!select || !value) return false;
    const option = select.querySelector(`option[value="${value}"]`);
    if (!option || option.disabled) return false;
    select.value = value;
    if (labelText) {
      const label = window.App.getModelOptionLabel(option);
      labelText.textContent = label;
      labelText.title = `Choose model: ${label}`;
    }
    return true;
  }

  function applyTierDefaultModels(isPro) {
    const defaults = isPro
      ? (window.PRO_DEFAULT_MODELS || {})
      : (window.FREE_DEFAULT_MODELS || {});
    window.App.modelPrefs.forEach(pref => {
      if (localStorage.getItem("pref_select_" + pref.key) !== null) return;
      const select = document.getElementById(pref.selectId);
      const labelText = document.getElementById(pref.textId);
      setPickerToValue(select, labelText, defaults[pref.provider]);
    });
  }

  function getModelPrefByResponseId(responseId) {
    return window.App.modelPrefs.find(pref => pref.responseId === responseId) || null;
  }

  // Ein Lauf vergleicht hoechstens so viele Familien, wie der Server erlaubt
  // (cfg.MAX_RUN_FAMILIES). Mehr Familien duerfen konfiguriert sein: die
  // Auswahl bleibt eine Auswahl, sie waechst nur nicht ueber das Limit.
  function runFamilyCap() {
    return Number(window.App.maxRunFamilies) > 0 ? Number(window.App.maxRunFamilies) : 6;
  }

  function selectedFamilyCount(exceptPref) {
    return window.App.modelPrefs.filter(pref => (
      pref !== exceptPref && document.getElementById(pref.checkId)?.checked
    )).length;
  }

  function capBlocksInclusion(pref) {
    return selectedFamilyCount(pref) >= runFamilyCap();
  }

  window.App.runFamilyCap = runFamilyCap;
  window.App.capBlocksInclusion = capBlocksInclusion;

  // Provider, die der LAUFENDE Lauf bewusst ausgelassen hat (heute: DeepSeek,
  // solange Anhaenge mitgehen). Der Block gehoert dem Lauf, nicht dem Composer:
  // beim Senden wandern die Anhaenge an die Nachricht, der Composer ist wieder
  // leer — und jede /ask-Antwort loest einen Tier-Refresh und damit
  // restoreModelSelections() aus. Ohne diesen Block schaltet der gespeicherte
  // Wert das Modell mitten im Lauf wieder ein, obwohl fuer es nie eine Anfrage
  // rausging: der Konsens wartet dann auf eine Antwort, die nie kommt, und
  // bricht mit "at least two completed model answers" ab.
  const runBlockedResponseIds = new Set();

  function setRunModelBlock(responseId, blocked) {
    const pref = getModelPrefByResponseId(responseId);
    if (!pref) return;

    if (blocked) {
      runBlockedResponseIds.add(responseId);
      setModelSelectionState(pref, false, {
        persist: false,
        syncCheckbox: true,
        animate: false
      });
      return;
    }

    if (!runBlockedResponseIds.delete(responseId)) return;
    // Der Block faellt: die gespeicherte Nutzerwahl gilt wieder. Ohne Eintrag
    // bleibt die aktuelle Auswahl stehen (der Composer stellt seinen gemerkten
    // Wert selbst wieder her, wenn die Dateien entfernt werden).
    const saved = localStorage.getItem("pref_check_" + pref.key);
    if (saved === null) return;
    setModelSelectionState(pref, saved === "true", {
      persist: false,
      syncCheckbox: true,
      animate: false
    });
  }

  function animateResponseReorder(box, applyStateChange) {
    if (!box || typeof box.animate !== "function") {
      applyStateChange();
      return;
    }

    const first = box.getBoundingClientRect();
    applyStateChange();
    const last = box.getBoundingClientRect();
    const deltaX = first.left - last.left;
    const deltaY = first.top - last.top;

    if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1) return;

    box.animate(
      [
        { transform: `translate(${deltaX}px, ${deltaY}px)`, opacity: 0.86 },
        { transform: "translate(0, 0)", opacity: box.classList.contains("excluded") ? 0.72 : 1 }
      ],
      {
        duration: 180,
        easing: "cubic-bezier(0.2, 0, 0, 1)"
      }
    );
  }

  function syncSidebarModelCount() {
    const count = window.App.modelPrefs.reduce((total, pref) => {
      return total + (document.getElementById(pref.checkId)?.checked ? 1 : 0);
    }, 0);
    const counter = document.getElementById("sidebarModelCount");
    if (counter) {
      const providerLabel = count === 1 ? "provider" : "providers";
      counter.textContent = count >= 2
        ? `${count} ${providerLabel} selected`
        : `${count} ${providerLabel} · choose at least 2`;
      counter.classList.toggle("is-invalid", count < 2);
    }
    document.getElementById("sidebarModelPicker")?.classList.toggle("is-invalid", count < 2);
  }

  function setModelSelectionState(prefOrResponseId, isChecked, options = {}) {
    const pref = typeof prefOrResponseId === "string"
      ? getModelPrefByResponseId(prefOrResponseId)
      : prefOrResponseId;
    if (!pref) return;

    const { persist = false, syncCheckbox = true, animate = persist } = options;
    const checkbox = document.getElementById(pref.checkId);
    const box = document.getElementById(pref.responseId);
    const label = document.querySelector(`label[for='${pref.checkId}']`);
    // Eine ausdrueckliche Nutzeraktion (persist) ist staerker als der Block des
    // laufenden Laufs: wer das Modell selbst wieder anhakt, bekommt es zurueck.
    if (persist) runBlockedResponseIds.delete(pref.responseId);
    // Tier refreshes restore persisted selections after /prepare. A provider
    // disabled by the attachment compatibility gate must stay excluded during
    // that restore; otherwise progress and consensus wait for an answer whose
    // request was intentionally never started.
    const attachmentBlocked = runBlockedResponseIds.has(pref.responseId)
      || (checkbox?.disabled
        && checkbox.getAttribute("aria-describedby") === "attachmentProviderNotice");
    let checked = attachmentBlocked ? false : !!isChecked;

    // Die siebte Familie wird abgelehnt statt eine andere still abzuwaehlen.
    // Gilt auch fuer die Wiederherstellung gespeicherter Auswahlen, damit ein
    // alter localStorage-Stand das Limit nicht umgeht.
    if (checked && capBlocksInclusion(pref)) {
      checked = false;
      if (persist) {
        window.App.showPopup?.(
          `A run compares up to ${runFamilyCap()} models. Leave one out to add ${pref.label}.`
        );
      }
    }

    if (checkbox && syncCheckbox) {
      checkbox.checked = checked;
    }

    if (label) {
      label.classList.toggle("is-unselected", !checked);
      label.title = checked ? pref.key : `${pref.key} is excluded. Check to include it again.`;
    }

    if (box) {
      const applyBoxState = () => {
        box.classList.toggle("excluded", !checked);
        box.title = checked ? "" : `${pref.key} is excluded. Click the checkmark to include it again.`;
        const excludeBtn = box.querySelector(".exclude-btn");
        if (excludeBtn) {
          excludeBtn.textContent = checked ? "×" : "✓";
          excludeBtn.title = checked ? "Exclude answer" : "Include answer";
          excludeBtn.setAttribute("aria-label", checked ? "Exclude answer" : "Include answer");
        }
        if (window.App.runRegistry?.visible?.()) {
          window.App.runRegistry.renderVisible();
        }
      };

      if (animate) {
        animateResponseReorder(box, applyBoxState);
      } else {
        applyBoxState();
      }
    }

    if (persist) {
      localStorage.setItem("pref_check_" + pref.key, String(checked));
      window.App.trackAppEvent("app_model_selection_changed", {
        provider: pref.key,
        enabled: checked,
        selected_models: window.App.getSelectedModelCount()
      });
    }

    syncSidebarModelCount();

    if (typeof window.updateConsensusButtonAvailability === "function") {
      window.updateConsensusButtonAvailability();
    }
    if (typeof window.updateQuestionInputAccess === "function") {
      window.updateQuestionInputAccess();
    }

    // The composer trigger includes the selected provider count. Temporary
    // compatibility exclusions must update it just like a manual toggle.
    window.syncCustomModelPickers?.();
    window.updateAgentModeUI();
  }

  window.restoreModelSelections = function () {
    // 1. Bestehende Logik für die Chat-Boxen (OpenAI, Mistral etc.)
    window.App.modelPrefs.forEach(pref => {
      const checkbox = document.getElementById(pref.checkId);
      const select = document.getElementById(pref.selectId);
      const labelText = document.getElementById(pref.textId);

      // Checkboxen wiederherstellen
      const savedCheck = localStorage.getItem("pref_check_" + pref.key);
      if (checkbox) {
        const isChecked = savedCheck === null ? checkbox.checked : savedCheck === "true";
        setModelSelectionState(pref, isChecked, { persist: false, syncCheckbox: true });
      }

      // Dropdowns wiederherstellen
      const savedSelect = localStorage.getItem("pref_select_" + pref.key);
      if (savedSelect !== null && select) {
        setPickerToValue(select, labelText, savedSelect);
      }
    });

    // --- Consensus wiederherstellen: Preset aufloesen oder explizite Wahl ---
    const consensusSelect = document.getElementById("consensusModelDropdown");
    if (consensusSelect) {
      const presetId = getActiveConsensusPresetId();
      if (presetId === "custom") {
        const savedConsensus = localStorage.getItem("pref_select_consensus"); // Eigener Key
        if (savedConsensus) {
          // Prüfen, ob die Option existiert und für den User freigeschaltet ist
          const option = consensusSelect.querySelector(`option[value="${savedConsensus}"]`);
          if (option && !option.disabled) {
            consensusSelect.value = savedConsensus;
          }
        }
      } else {
        applyConsensusPreset(consensusSelect, presetId);
      }
      syncCustomModelPicker(consensusSelect);
    }
  };

  // --- CONSENSUS-PRESETS (Daily/Balanced/High Quality + Custom) ---

  const CONSENSUS_PRESET_STORAGE_KEY = "pref_consensus_preset";

  // Zuletzt angezeigtes Preset-/Modell-Label OHNE die Modellanzahl davor.
  // Der gefuehrte Lauf nennt es in seiner "Question prepared"-Zeile; die Zahl
  // steht dort schon separat, sie wuerde sich sonst doppeln.
  let lastPresetDisplayLabel = "";

  function getConsensusPresets() {
    return Array.isArray(window.CONSENSUS_PRESETS) ? window.CONSENSUS_PRESETS : [];
  }

  function getDefaultConsensusPresetId() {
    const presets = getConsensusPresets();
    if (!presets.length) return "custom";
    const configured = window.DEFAULT_CONSENSUS_PRESET;
    return presets.some(preset => preset.id === configured) ? configured : presets[0].id;
  }

  function getActiveConsensusPresetId() {
    const presets = getConsensusPresets();
    if (!presets.length) return "custom";
    const stored = localStorage.getItem(CONSENSUS_PRESET_STORAGE_KEY);
    if (stored === "custom") return "custom";
    if (stored && presets.some(preset => preset.id === stored)) {
      const preset = presets.find(entry => entry.id === stored);
      if (!preset.pro_only || window.isUserPro === true) return stored;
      return getDefaultConsensusPresetId();
    }
    // Migration: eine bereits gespeicherte explizite Modellwahl (Bestand vor
    // den Presets) bleibt als Custom-Auswahl erhalten.
    if (localStorage.getItem("pref_select_consensus") !== null) return "custom";
    return getDefaultConsensusPresetId();
  }

  function resolveConsensusPresetValue(select, presetId) {
    const preset = getConsensusPresets().find(entry => entry.id === presetId);
    if (!preset || !select) return null;
    if (preset.pro_only && window.isUserPro !== true) return null;

    const consensusValue = preset.consensus_model
      || (Array.isArray(preset.candidates) ? preset.candidates[0] : null);
    const consensusOption = Array.from(select.options).find(opt =>
      opt.value === consensusValue && !opt.disabled
    );
    if (!consensusOption) return null;

    for (const pref of window.App.modelPrefs) {
      const model = preset.models?.[pref.provider];
      const providerSelect = document.getElementById(pref.selectId);
      const option = Array.from(providerSelect?.options || []).find(opt =>
        opt.value === model && !opt.disabled
      );
      if (!option) return null;
    }
    return consensusValue;
  }

  function applyConsensusPreset(select, presetId) {
    // Kein change-Event: die Preset-Aufloesung darf pref_select_consensus
    // nicht ueberschreiben (gleiches Muster wie die temporaere
    // Deep-Think-Auswahl in app-init.js).
    const value = resolveConsensusPresetValue(select, presetId);
    if (!value) return null;
    const preset = getConsensusPresets().find(entry => entry.id === presetId);
    window.App.modelPrefs.forEach(pref => {
      const providerSelect = document.getElementById(pref.selectId);
      const labelText = document.getElementById(pref.textId);
      setPickerToValue(providerSelect, labelText, preset.models?.[pref.provider]);
    });
    select.value = value;
    window.syncCustomModelPickers?.();
    window.updateAgentModeUI?.();
    return value;
  }

  function selectConsensusPreset(select, presetId) {
    const preset = getConsensusPresets().find(entry => entry.id === presetId);
    if (preset?.pro_only && window.isUserPro !== true) {
      window.App.showProFeatureModal?.(`${preset.label || "High Quality"} mode`);
      return;
    }
    localStorage.setItem(CONSENSUS_PRESET_STORAGE_KEY, presetId);
    applyConsensusPreset(select, presetId);
    window.App.trackAppEvent("app_consensus_preset_changed", { preset: presetId });
    collapseExpandedModelPicker(select);
    syncCustomModelPicker(select);
  }

  function markConsensusPresetCustom() {
    localStorage.setItem(CONSENSUS_PRESET_STORAGE_KEY, "custom");
    window.App.modelPrefs.forEach(pref => {
      const select = document.getElementById(pref.selectId);
      if (select?.value) localStorage.setItem("pref_select_" + pref.key, select.value);
    });
    const consensus = document.getElementById("consensusModelDropdown");
    if (consensus?.value) localStorage.setItem("pref_select_consensus", consensus.value);
    window.syncCustomModelPickers?.();
  }

  // --- CUSTOM MODEL PICKER (eigene Listbox ueber den nativen <select>) ---

  let expandedModelPicker = null;

  function getModelPickerState(select) {
    return select?._customModelPicker || null;
  }

  function syncCustomModelPicker(select) {
    const state = getModelPickerState(select);
    if (!state) return;

    const selectedOption = select.options[select.selectedIndex] || select.options[0];
    const selectedLabel = window.App.getModelOptionLabel(selectedOption);

    // Mit Preset-Ebene zeigt der Trigger den Preset-Namen, solange der
    // Select-Wert der Preset-Aufloesung entspricht. Weicht der Wert ab
    // (z. B. temporaere Deep-Think-Auswahl), bleibt der echte Modellname.
    let displayLabel = selectedLabel;
    let displayTitle = selectedLabel;
    if (state.presets) {
      const presetId = getActiveConsensusPresetId();
      if (presetId !== "custom") {
        const preset = getConsensusPresets().find(entry => entry.id === presetId);
        const presetValue = resolveConsensusPresetValue(select, presetId);
        if (preset && presetValue && select.value === presetValue) {
          displayLabel = preset.label;
          displayTitle = `${preset.label} · ${selectedLabel}`;
        }
      }
      // Der Consensus-Trigger ist im rahmenlosen Composer der EINZIGE
      // Lauf-Schalter. Er sagt deshalb beides: wie viele Modelle antworten
      // und welche Stufe den Konsens schreibt ("6 models · Balanced").
      const count = window.App.getSelectedModelCount?.() || 0;
      const countLabel = `${count} ${count === 1 ? "model" : "models"}`;
      lastPresetDisplayLabel = displayLabel;
      displayLabel = `${countLabel} · ${displayLabel}`;
      displayTitle = count < 2
        ? `${countLabel} selected · Choose at least 2 models to run consensus`
        : `${countLabel} · ${displayTitle}`;
    }

    if (state.displayButton) {
      state.displayButton.querySelector(".model-picker-display-text").textContent = displayLabel;
      state.displayButton.title = displayTitle;
    }

    state.host.setAttribute("aria-label", `Choose model: ${displayLabel}`);
    state.menu.querySelectorAll(".model-picker-option[data-value]").forEach(item => {
      const isSelected = item.dataset.value === select.value;
      item.classList.toggle("is-selected", isSelected);
      item.setAttribute("aria-selected", String(isSelected));
    });
  }

  function renderConsensusPresetMenu(select, state) {
    const activePresetId = getActiveConsensusPresetId();

    getConsensusPresets().forEach(preset => {
      const resolved = resolveConsensusPresetValue(select, preset.id);
      const item = document.createElement("button");
      item.type = "button";
      item.className = "model-picker-option model-picker-preset-option";
      item.dataset.preset = preset.id;
      item.setAttribute("role", "option");
      const isProLocked = !!preset.pro_only && window.isUserPro !== true;
      item.disabled = !resolved && !isProLocked;
      item.classList.toggle("is-locked", isProLocked);
      item.setAttribute("aria-disabled", String(isProLocked || !resolved));
      const isSelected = preset.id === activePresetId;
      item.classList.toggle("is-selected", isSelected);
      item.setAttribute("aria-selected", String(isSelected));

      const label = document.createElement("span");
      label.className = "model-picker-option-label model-picker-preset-label";
      const name = document.createElement("span");
      name.className = "model-picker-preset-name";
      name.textContent = preset.label;
      label.appendChild(name);
      if (preset.hint) {
        const hint = document.createElement("span");
        hint.className = "model-picker-preset-hint";
        hint.textContent = preset.hint;
        label.appendChild(hint);
      }
      item.appendChild(label);

      if (preset.pro_only) {
        const badge = document.createElement("span");
        badge.className = "pro-badge model-picker-pro-badge";
        badge.textContent = "Pro";
        item.appendChild(badge);
      }

      item.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        if (isProLocked) {
          window.App.showProFeatureModal?.(`${preset.label || "High Quality"} mode`);
          return;
        }
        if (item.disabled) return;
        selectConsensusPreset(select, preset.id);
      });

      state.menu.appendChild(item);
    });

    // "Custom" oeffnet die Aufstellung des Laufs: antwortende Provider samt
    // Modellwahl plus die Consensus-Engine (bewusst ohne Beschreibungen —
    // wer hier waehlt, kennt die Modelle).
    const customItem = document.createElement("button");
    customItem.type = "button";
    customItem.className = "model-picker-option model-picker-preset-option model-picker-custom-option";
    customItem.setAttribute("role", "option");
    const customActive = activePresetId === "custom";
    customItem.classList.toggle("is-selected", customActive);
    customItem.setAttribute("aria-selected", String(customActive));

    const customLabel = document.createElement("span");
    customLabel.className = "model-picker-option-label model-picker-preset-label";
    const customName = document.createElement("span");
    customName.className = "model-picker-preset-name";
    customName.textContent = "Custom";
    customLabel.appendChild(customName);
    const customHint = document.createElement("span");
    customHint.className = "model-picker-preset-hint";
    const selectedOption = select.options[select.selectedIndex];
    const modelCount = window.App.getSelectedModelCount?.() || 0;
    customHint.textContent = customActive
      ? `${modelCount} ${modelCount === 1 ? "model" : "models"} · ${window.App.getModelOptionLabel(selectedOption)}`
      : "Choose each model yourself";
    customLabel.appendChild(customHint);
    customItem.appendChild(customLabel);

    const chevron = document.createElement("span");
    chevron.className = "model-picker-option-chevron";
    chevron.setAttribute("aria-hidden", "true");
    customItem.appendChild(chevron);

    customItem.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      state.view = "custom";
      renderCustomModelPicker(select);
    });

    state.menu.appendChild(customItem);
  }

  function renderBackRow(select, state, label, targetView) {
    const back = document.createElement("button");
    back.type = "button";
    back.className = "model-picker-option model-picker-back-option";
    const chevron = document.createElement("span");
    chevron.className = "model-picker-option-chevron is-back";
    chevron.setAttribute("aria-hidden", "true");
    back.appendChild(chevron);
    const labelEl = document.createElement("span");
    labelEl.className = "model-picker-option-label";
    labelEl.textContent = label;
    back.appendChild(labelEl);

    back.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      state.view = targetView;
      renderCustomModelPicker(select);
    });

    state.menu.appendChild(back);
  }

  function appendSectionLabel(menu, text) {
    const label = document.createElement("div");
    label.className = "model-picker-section-label";
    label.textContent = text;
    menu.appendChild(label);
  }

  // --- Custom: die ganze Aufstellung eines Laufs -------------------------
  // "Custom" hiess bisher nur "nimm ein anderes Consensus-Modell". Die sechs
  // antwortenden Modelle — also der eigentliche Lauf — waren im Composer gar
  // nicht erreichbar; man musste die Antwortboxen finden, die im Agent Mode
  // verborgen sind. Custom zeigt jetzt beides: wer antwortet (mit Ein-/
  // Ausschluss und Modellwahl je Provider) und wer daraus den Konsens
  // schreibt. Jede Zeile fuehrt in ihre eigene Modell-Liste und zurueck.

  function providerViewId(pref) {
    return "provider:" + pref.key;
  }

  // Views, die eine Modell-Liste zeigen: "engine" und "provider:<key>".
  function isModelListView(view) {
    return view === "engine" || String(view || "").startsWith("provider:");
  }

  function providerForView(view) {
    if (!String(view || "").startsWith("provider:")) return null;
    const key = String(view).slice("provider:".length);
    return window.App.modelPrefs.find(pref => pref.key === key) || null;
  }

  function currentOptionLabel(targetSelect) {
    const option = targetSelect?.options[targetSelect.selectedIndex];
    return option ? window.App.getModelOptionLabel(option) : "";
  }

  function renderProviderRow(select, state, pref) {
    const providerSelect = document.getElementById(pref.selectId);
    if (!providerSelect) return;
    const checkbox = document.getElementById(pref.checkId);
    const included = checkbox ? checkbox.checked : true;

    const row = document.createElement("div");
    row.className = "model-picker-row";
    row.classList.toggle("is-excluded", !included);

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "model-picker-row-toggle";
    toggle.setAttribute("role", "checkbox");
    toggle.setAttribute("aria-checked", String(included));
    toggle.setAttribute(
      "aria-label",
      (included ? "Exclude " : "Include ") + pref.label
    );
    const capped = !included && capBlocksInclusion(pref);
    toggle.title = included
      ? pref.label + " answers this run — click to leave it out"
      : (capped
        ? `A run compares up to ${runFamilyCap()} models — leave one out to add ${pref.label}`
        : pref.label + " is left out — click to include it");
    toggle.disabled = !!checkbox?.disabled || capped;
    toggle.innerHTML = '<span class="model-picker-row-check" aria-hidden="true"></span>';
    toggle.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      if (toggle.disabled) return;
      setModelSelectionState(pref, !included, { persist: true });
      renderCustomModelPicker(select);
    });

    const open = document.createElement("button");
    open.type = "button";
    open.className = "model-picker-option model-picker-row-open";
    const label = document.createElement("span");
    label.className = "model-picker-option-label model-picker-preset-label";
    const name = document.createElement("span");
    name.className = "model-picker-preset-name";
    name.textContent = pref.label;
    const hint = document.createElement("span");
    hint.className = "model-picker-preset-hint";
    hint.textContent = currentOptionLabel(providerSelect);
    label.append(name, hint);
    open.appendChild(label);
    const chevron = document.createElement("span");
    chevron.className = "model-picker-option-chevron";
    chevron.setAttribute("aria-hidden", "true");
    open.appendChild(chevron);
    open.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      state.view = providerViewId(pref);
      renderCustomModelPicker(select);
    });

    row.append(toggle, open);
    state.menu.appendChild(row);
  }

  function renderCustomOverview(select, state) {
    renderBackRow(select, state, "Presets", "presets");

    appendSectionLabel(state.menu, "Answering models");
    const selectedCount = window.App.getSelectedModelCount?.() || 0;
    if (selectedCount < 2) {
      const requirement = document.createElement("p");
      requirement.className = "model-picker-requirement";
      requirement.setAttribute("role", "status");
      requirement.textContent = `Select at least 2 models to run consensus · ${selectedCount} selected`;
      state.menu.appendChild(requirement);
    }
    window.App.modelPrefs.forEach(pref => renderProviderRow(select, state, pref));

    appendSectionLabel(state.menu, "Consensus engine");
    const engine = document.createElement("button");
    engine.type = "button";
    engine.className = "model-picker-option model-picker-row-open";
    const label = document.createElement("span");
    label.className = "model-picker-option-label model-picker-preset-label";
    const name = document.createElement("span");
    name.className = "model-picker-preset-name";
    name.textContent = "Writes the consensus";
    const hint = document.createElement("span");
    hint.className = "model-picker-preset-hint";
    hint.textContent = currentOptionLabel(select);
    label.append(name, hint);
    engine.appendChild(label);
    const chevron = document.createElement("span");
    chevron.className = "model-picker-option-chevron";
    chevron.setAttribute("aria-hidden", "true");
    engine.appendChild(chevron);
    engine.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      state.view = "engine";
      renderCustomModelPicker(select);
    });
    state.menu.appendChild(engine);
  }

  function renderCustomModelPicker(select) {
    const state = getModelPickerState(select);
    if (!state) return;

    state.menu.innerHTML = "";

    // Preset-Ebene: Daily/Balanced/High Quality + Custom statt der Modell-Liste.
    if (state.presets && state.view !== "custom" && !isModelListView(state.view)) {
      renderConsensusPresetMenu(select, state);
      syncCustomModelPicker(select);
      return;
    }

    if (state.presets && state.view === "custom") {
      renderCustomOverview(select, state);
      syncCustomModelPicker(select);
      return;
    }

    // Modell-Liste: entweder die eines Providers oder die der Engine. Ohne
    // Preset-Ebene (die Picker in den Antwortboxen) ist es unveraendert die
    // Liste des eigenen Selects.
    const pref = state.presets ? providerForView(state.view) : null;
    const targetSelect = pref ? document.getElementById(pref.selectId) : select;
    if (!targetSelect) {
      state.view = "custom";
      renderCustomOverview(select, state);
      syncCustomModelPicker(select);
      return;
    }

    if (state.presets) {
      renderBackRow(select, state, pref ? pref.label : "Consensus engine", "custom");
    }

    Array.from(targetSelect.options).forEach(option => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "model-picker-option";
      // data-value ist der Sync-Haken fuer den EIGENEN Select (siehe
      // syncCustomModelPicker). In der Liste eines fremden Provider-Selects
      // waere er eine falsche Aussage ueber die Consensus-Auswahl.
      if (targetSelect === select) item.dataset.value = option.value;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.selected));
      item.disabled = option.disabled;
      item.classList.toggle("is-selected", option.selected);
      const badges = (option.dataset.modelBadge || "")
        .split(/\s*(?:\u00c2\u00b7|\u00b7)\s*/)
        .map(badge => badge.trim())
        .filter(Boolean);
      if (!badges.length && option.classList.contains("premium-option")) {
        badges.push("Pro");
      }
      const hasProBadge = badges.some(badge => badge.toLowerCase() === "pro");
      item.classList.toggle("is-premium", option.classList.contains("premium-option") || hasProBadge);

      const label = document.createElement("span");
      label.className = "model-picker-option-label";
      label.textContent = window.App.getModelOptionLabel(option);
      item.appendChild(label);

      badges.forEach(badgeText => {
        const badge = document.createElement("span");
        badge.className = "pro-badge model-picker-pro-badge";
        badge.textContent = badgeText;
        item.appendChild(badge);
      });

      let pointerStart = null;
      let didCommitFromPointer = false;

      function commitSelection(event) {
        event.preventDefault();
        event.stopPropagation();
        if (option.disabled) return;

        // Das change-Event traegt die ganze Persistenz (app-init.js speichert
        // pref_select_* und schaltet auf "custom" um) — deshalb wird es auch
        // fuer die Provider-Selects hier gefeuert, nicht nur fuer die Engine.
        targetSelect.selectedIndex = option.index;
        targetSelect.value = option.value;
        targetSelect.dispatchEvent(new Event("input", { bubbles: true }));
        targetSelect.dispatchEvent(new Event("change", { bubbles: true }));

        // Ein Provider-Modell ist eine Zeile in der Aufstellung, kein Abschluss:
        // zurueck in die Uebersicht statt das Menue zu schliessen.
        if (pref) {
          state.view = "custom";
          renderCustomModelPicker(select);
          return;
        }
        collapseExpandedModelPicker(select);
      }

      item.addEventListener("pointerdown", event => {
        pointerStart = { x: event.clientX, y: event.clientY };
        didCommitFromPointer = false;
      });

      item.addEventListener("pointerup", event => {
        if (!pointerStart) return;
        const dx = Math.abs(event.clientX - pointerStart.x);
        const dy = Math.abs(event.clientY - pointerStart.y);
        pointerStart = null;
        if (dx > 8 || dy > 8) return;

        didCommitFromPointer = true;
        commitSelection(event);
      });

      item.addEventListener("click", event => {
        if (didCommitFromPointer) {
          event.preventDefault();
          event.stopPropagation();
          didCommitFromPointer = false;
          return;
        }
        commitSelection(event);
      });

      state.menu.appendChild(item);
    });

    syncCustomModelPicker(select);
  }

  function collapseExpandedModelPicker(select = expandedModelPicker) {
    const state = getModelPickerState(select);
    if (!state) return;

    state.menu.classList.remove("is-open");
    state.host.classList.remove("is-expanded", "is-open");
    state.host.setAttribute("aria-expanded", "false");

    if (state.displayButton) {
      state.displayButton.setAttribute("aria-expanded", "false");
    }

    if (expandedModelPicker === select) {
      expandedModelPicker = null;
    }
  }

  function openModelPicker(select) {
    const state = getModelPickerState(select);
    if (!select || select.disabled || !state) return;

    if (expandedModelPicker && expandedModelPicker !== select) {
      collapseExpandedModelPicker(expandedModelPicker);
    }

    // Einstiegs-View: aktive Preset-Nutzer sehen die Presets, Custom-Nutzer
    // landen ohne Umweg in ihrer Aufstellung. Nie in einer Modell-Liste —
    // die ist immer nur eine Ebene tiefer, die man selbst geoeffnet hat.
    if (state.presets) {
      state.view = getActiveConsensusPresetId() === "custom" ? "custom" : "presets";
    }

    renderCustomModelPicker(select);
    state.host.classList.add("is-expanded", "is-open");
    state.host.setAttribute("aria-expanded", "true");
    state.menu.classList.add("is-open");

    if (state.displayButton) {
      state.displayButton.setAttribute("aria-expanded", "true");
    }

    expandedModelPicker = select;
  }

  function initCustomModelPicker(select, options = {}) {
    if (!select || getModelPickerState(select)) return;

    const host = select.closest(".model-picker-wrapper") || select.closest(".select-wrapper") || select.closest(".agent-mode-chip");
    if (!host) return;

    host.classList.add("custom-model-picker", "is-enhanced");
    host.setAttribute("role", "button");
    host.setAttribute("aria-haspopup", "listbox");
    host.setAttribute("aria-expanded", "false");
    host.tabIndex = options.externalTrigger ? 0 : -1;
    select.classList.add("native-model-picker");

    const menu = document.createElement("div");
    menu.className = "model-picker-menu";
    menu.setAttribute("role", "listbox");

    let displayButton = null;
    if (!options.externalTrigger) {
      displayButton = document.createElement("button");
      displayButton.type = "button";
      displayButton.className = "model-picker-display";
      displayButton.setAttribute("aria-haspopup", "listbox");
      displayButton.setAttribute("aria-expanded", "false");
      displayButton.innerHTML = '<span class="model-picker-display-text"></span>';
      host.appendChild(displayButton);
      displayButton.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        openModelPicker(select);
      });
    }

    host.appendChild(menu);
    select._customModelPicker = {
      host,
      menu,
      displayButton,
      // Preset-Ebene nur fuer den Consensus-Picker (options.presets) und nur,
      // wenn der Server Presets liefert — sonst unveraendert die Modell-Liste.
      presets: !!options.presets && getConsensusPresets().length > 0,
      view: "presets"
    };

    host.addEventListener("click", event => {
      if (menu.contains(event.target) || event.target === displayButton) return;
      event.preventDefault();
      event.stopPropagation();
      openModelPicker(select);
    });

    host.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " " || event.key === "ArrowDown") {
        event.preventDefault();
        openModelPicker(select);
      } else if (event.key === "Escape") {
        collapseExpandedModelPicker(select);
      }
    });

    select.addEventListener("change", () => syncCustomModelPicker(select));
    renderCustomModelPicker(select);
  }

  window.syncCustomModelPickers = function () {
    document.querySelectorAll(".native-model-picker").forEach(syncCustomModelPicker);
  };

  document.addEventListener("click", function (event) {
    if (!expandedModelPicker) return;
    const state = getModelPickerState(expandedModelPicker);
    const title = expandedModelPicker.closest(".title");
    const isInsideTitle = title && title.contains(event.target);
    const isInsidePicker = state && state.host.contains(event.target);
    if (!isInsideTitle && !isInsidePicker) {
      collapseExpandedModelPicker();
    }
  });

  // --- Exporte fuer das in initApp verbliebene Wiring + andere Module ---
  window.App.applyTierDefaultModels = applyTierDefaultModels;
  window.App.setModelSelectionState = setModelSelectionState;
  window.App.setRunModelBlock = setRunModelBlock;
  window.App.openModelPicker = openModelPicker;
  window.App.collapseExpandedModelPicker = collapseExpandedModelPicker;
  window.App.initCustomModelPicker = initCustomModelPicker;
  window.App.markConsensusPresetCustom = markConsensusPresetCustom;
  window.App.currentPresetLabel = () => lastPresetDisplayLabel;
})();
