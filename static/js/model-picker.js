// =====================================================================
// model-picker.js
// Modell-Auswahl + Custom-Picker-UI. In eigene IIFE gekapselt.
// Zwei Teile:
//   1) Model-Selection-State: Wiederherstellen/Persistieren der pro-Provider
//      Auswahl (Dropdown + Ein/Ausschluss-Checkbox), Tier-Defaults, Reorder-
//      Animation beim Ein-/Ausschliessen.
//   2) Custom-Model-Picker: das eigene, getunte Listbox-Dropdown, das die
//      nativen <select> ueberlagert (expandedModelPicker ist modul-privat).
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

    // --- NEU: Consensus Modell wiederherstellen ---
    const consensusSelect = document.getElementById("consensusModelDropdown");
    const savedConsensus = localStorage.getItem("pref_select_consensus"); // Eigener Key

    if (consensusSelect && savedConsensus) {
      // Prüfen, ob die Option existiert und für den User freigeschaltet ist
      const option = consensusSelect.querySelector(`option[value="${savedConsensus}"]`);
      if (option && !option.disabled) {
        consensusSelect.value = savedConsensus;
      }
    }
  };

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

    if (state.displayButton) {
      state.displayButton.querySelector(".model-picker-display-text").textContent = selectedLabel;
      state.displayButton.title = selectedLabel;
    }

    state.host.setAttribute("aria-label", `Choose model: ${selectedLabel}`);
    state.menu.querySelectorAll(".model-picker-option").forEach(item => {
      const isSelected = item.dataset.value === select.value;
      item.classList.toggle("is-selected", isSelected);
      item.setAttribute("aria-selected", String(isSelected));
    });
  }

  function renderCustomModelPicker(select) {
    const state = getModelPickerState(select);
    if (!state) return;

    state.menu.innerHTML = "";
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
          ? "model-picker-pro-badge"
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
    select._customModelPicker = { host, menu, displayButton };

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
