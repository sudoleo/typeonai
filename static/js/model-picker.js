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
//      Fast/Balanced/Thorough (window.CONSENSUS_PRESETS, tier-bewusst
//      aufgeloest ueber disabled-Optionen) plus "Custom" fuer die volle
//      Modell-Liste. Preset-Zustand in localStorage "pref_consensus_preset";
//      die Preset-Aufloesung setzt den nativen Select OHNE change-Event,
//      damit pref_select_consensus (explizite Modellwahl) erhalten bleibt.
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

  function applyTierDefaultModels(isPro, isEarly) {
    // Pro -> Pro-Defaults; nur Early-Tag -> Early-Defaults (Frontier-Low);
    // sonst die guenstigen Free-Defaults. Pro schliesst Early-Zugang ein, hat
    // aber eigene Defaults, daher die Pro-Abfrage zuerst.
    let defaults;
    if (isPro) {
      defaults = window.PRO_DEFAULT_MODELS || {};
    } else if (isEarly) {
      defaults = window.EARLY_DEFAULT_MODELS || {};
    } else {
      defaults = window.FREE_DEFAULT_MODELS || {};
    }
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

  function setModelSelectionState(prefOrResponseId, isChecked, options = {}) {
    const pref = typeof prefOrResponseId === "string"
      ? getModelPrefByResponseId(prefOrResponseId)
      : prefOrResponseId;
    if (!pref) return;

    const checked = !!isChecked;
    const checkbox = document.getElementById(pref.checkId);
    const box = document.getElementById(pref.responseId);
    const label = document.querySelector(`label[for='${pref.checkId}']`);
    const { persist = false, syncCheckbox = true, animate = persist } = options;

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

    if (typeof window.updateConsensusButtonAvailability === "function") {
      window.updateConsensusButtonAvailability();
    }

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

  // --- CONSENSUS-PRESETS (Fast/Balanced/Thorough + Custom) ---

  const CONSENSUS_PRESET_STORAGE_KEY = "pref_consensus_preset";

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
    if (stored && presets.some(preset => preset.id === stored)) return stored;
    // Migration: eine bereits gespeicherte explizite Modellwahl (Bestand vor
    // den Presets) bleibt als Custom-Auswahl erhalten.
    if (localStorage.getItem("pref_select_consensus") !== null) return "custom";
    return getDefaultConsensusPresetId();
  }

  function resolveConsensusPresetValue(select, presetId) {
    // Erster Kandidat, dessen Option fuer das aktuelle Tier freigeschaltet
    // ist (Premium/Early sind via option.disabled gegatet, siehe user-tier.js).
    const preset = getConsensusPresets().find(entry => entry.id === presetId);
    if (!preset || !select) return null;
    for (const candidate of preset.candidates || []) {
      const option = Array.from(select.options).find(opt => opt.value === candidate);
      if (option && !option.disabled) return candidate;
    }
    return null;
  }

  function applyConsensusPreset(select, presetId) {
    // Kein change-Event: die Preset-Aufloesung darf pref_select_consensus
    // nicht ueberschreiben (gleiches Muster wie die temporaere
    // Deep-Think-Auswahl in app-init.js).
    const value = resolveConsensusPresetValue(select, presetId);
    if (value && select.value !== value) {
      select.value = value;
    }
    return value;
  }

  function selectConsensusPreset(select, presetId) {
    localStorage.setItem(CONSENSUS_PRESET_STORAGE_KEY, presetId);
    applyConsensusPreset(select, presetId);
    window.App.trackAppEvent("app_consensus_preset_changed", { preset: presetId });
    collapseExpandedModelPicker(select);
    syncCustomModelPicker(select);
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
      item.disabled = !resolved;
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

      item.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        if (item.disabled) return;
        selectConsensusPreset(select, preset.id);
      });

      state.menu.appendChild(item);
    });

    // "Custom" oeffnet die volle Modell-Liste (bewusst ohne Beschreibungen —
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
    customHint.textContent = customActive
      ? window.App.getModelOptionLabel(selectedOption)
      : "Pick a specific model";
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

  function renderConsensusBackRow(select, state) {
    const back = document.createElement("button");
    back.type = "button";
    back.className = "model-picker-option model-picker-back-option";
    const chevron = document.createElement("span");
    chevron.className = "model-picker-option-chevron is-back";
    chevron.setAttribute("aria-hidden", "true");
    back.appendChild(chevron);
    const label = document.createElement("span");
    label.className = "model-picker-option-label";
    label.textContent = "Presets";
    back.appendChild(label);

    back.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      state.view = "presets";
      renderCustomModelPicker(select);
    });

    state.menu.appendChild(back);
  }

  function renderCustomModelPicker(select) {
    const state = getModelPickerState(select);
    if (!state) return;

    state.menu.innerHTML = "";

    // Preset-Ebene: Fast/Balanced/Thorough + Custom statt der Modell-Liste.
    if (state.presets && state.view !== "custom") {
      renderConsensusPresetMenu(select, state);
      syncCustomModelPicker(select);
      return;
    }

    if (state.presets) {
      renderConsensusBackRow(select, state);
    }

    Array.from(select.options).forEach(option => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "model-picker-option";
      item.dataset.value = option.value;
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
      if (!badges.length && option.classList.contains("early-option")) {
        badges.push("Early");
      }
      const hasProBadge = badges.some(badge => badge.toLowerCase() === "pro");
      const hasEarlyBadge = badges.some(badge => badge.toLowerCase() === "early");
      item.classList.toggle("is-premium", option.classList.contains("premium-option") || hasProBadge);
      item.classList.toggle("is-early", option.classList.contains("early-option") || hasEarlyBadge);

      const label = document.createElement("span");
      label.className = "model-picker-option-label";
      label.textContent = window.App.getModelOptionLabel(option);
      item.appendChild(label);

      badges.forEach(badgeText => {
        const badge = document.createElement("span");
        badge.className = badgeText.toLowerCase() === "pro"
          ? "pro-badge model-picker-pro-badge"
          : "model-picker-early-badge";
        badge.textContent = badgeText;
        item.appendChild(badge);
      });

      let pointerStart = null;
      let didCommitFromPointer = false;

      function commitSelection(event) {
        event.preventDefault();
        event.stopPropagation();
        if (option.disabled) return;

        select.selectedIndex = option.index;
        select.value = option.value;
        select.dispatchEvent(new Event("input", { bubbles: true }));
        select.dispatchEvent(new Event("change", { bubbles: true }));
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
    // landen ohne Umweg direkt in der vollen Modell-Liste.
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
  window.App.openModelPicker = openModelPicker;
  window.App.collapseExpandedModelPicker = collapseExpandedModelPicker;
  window.App.initCustomModelPicker = initCustomModelPicker;
})();
