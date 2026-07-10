// =====================================================================
// user-tier.js
// Tier-/Pro-UI: Badge, Upgrade-Link, Deep-Search-Sperre, Premium-Modell-
// Optionen je nach Pro/Free/ausgeloggt. In eigene IIFE gekapselt.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.updateUserTierUI, window.updatePremiumModelsState.
// Abhaengigkeiten: window.setCurrentUsageLimits (optional waehrend Init), window.restoreModelSelections,
// window.syncCustomModelPickers, window.App.updateDeepThinkText,
// window.App.applyTierDefaultModels, window.isUserPro / window.isUserEarly (State).
// =====================================================================

(function () {
  function updateUserTierUI(isPro, isLoggedIn = false, isEarly) { // Standardmäßig false
    // 1. Globalen Status aktualisieren. Pro schliesst Early-Zugang ein.
    // Viele Aufrufer (z.B. nach jeder Query) uebergeben kein isEarly -> dann den
    // beim Login ermittelten Zustand (window.isUserEarly) beibehalten, statt
    // Early-Nutzern den Zugang faelschlich zu entziehen.
    if (isEarly === undefined) {
      isEarly = Boolean(window.isUserEarly);
    }
    window.isUserPro = isPro;
    const hasEarlyAccess = Boolean(isPro || isEarly);
    window.isUserEarly = hasEarlyAccess;

    // Follow-up-Affordance neu rendern: Pro-Badge/Teaser hängen am Tier.
    window.App?.followup?.render?.();

    // 2. Elemente referenzieren
    const badge = document.getElementById("proBadge");
    const upgradeLink = document.getElementById("upgradeLink");
    const deepSearchLabel = document.querySelector('.switch.deep-switch');

    // === CASE 1: NICHT EINGELOGGT ===
    if (!isLoggedIn) {
      // Alles verstecken
      if (badge) badge.style.display = "none";
      if (upgradeLink) upgradeLink.style.display = "none";

      // Optional: Standard-Limits (Free) oder ganz sperren
      window.setCurrentUsageLimits?.(false);

      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false, false);

      // Deep Search sperren (wie bei Free User)
      if (deepSearchLabel) {
        deepSearchLabel.classList.add("locked");
        deepSearchLabel.title = "Login required";
      }
      return; // Funktion hier beenden!
    }

    // === CASE 2: EINGELOGGT (Pro oder Free) ===
    if (isPro) {
      // --- PRO USER ---
      if (badge) badge.style.display = "inline-block";
      if (upgradeLink) upgradeLink.style.display = "none";

      // Limits
      window.setCurrentUsageLimits?.(true);

      // Dropdowns entsperren (Pro schliesst Early ein)
      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(true, true);

      // Deep Search entsperren
      if (deepSearchLabel) {
        deepSearchLabel.classList.remove("locked");
        deepSearchLabel.title = "Deep Think enabled";
        const input = deepSearchLabel.querySelector('input');
        if (input) input.style.pointerEvents = "auto";
      }

    } else {
      // --- FREE USER (EINGELOGGT) ---
      if (badge) badge.style.display = "none";
      if (upgradeLink) upgradeLink.style.display = "inline-block"; // Hier zeigen wir Upgrade

      // Limits
      window.setCurrentUsageLimits?.(false);

      // Dropdowns sperren (Early-Modelle nur mit Early-Tag)
      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false, hasEarlyAccess);

      // Deep Search ausschalten & sperren
      const deepToggle = document.getElementById("deepSearchToggle");
      if (deepToggle && deepToggle.checked) {
        deepToggle.checked = false;
        if (typeof window.App.updateDeepThinkText === 'function') window.App.updateDeepThinkText();
      }
      if (deepSearchLabel) {
        deepSearchLabel.classList.add("locked");
        deepSearchLabel.title = "Pro feature only";

        // WICHTIG: Klicks auf dem gesamten Label erlauben, damit der Listener feuert
        deepSearchLabel.style.pointerEvents = "auto";

        const input = deepSearchLabel.querySelector('input');
        if (input) input.style.pointerEvents = "auto";
      }
    }
  }

  function updatePremiumModelsState(isPro, isEarly = false) {
    // Dropdown-IDs definieren (Consensus und OpenAI)
    const dropdownIds = [
      "consensusModelDropdown",
      "openaiModelSelect",
      "mistralModelSelect",
      "claudeModelSelect",
      "geminiModelSelect",
      "deepseekModelSelect",
      "grokModelSelect"
    ];

    dropdownIds.forEach(id => {
      const dropdown = document.getElementById(id);
      if (!dropdown) return;

      // Pro-Optionen: nur fuer Pro entsperren.
      dropdown.querySelectorAll('option.premium-option').forEach(option => {
        option.textContent = option.textContent
          .replace(/^Pro:\s*/i, '')
          .replace(' (Pro only)', '')
          .trim();
        option.disabled = !isPro;
      });

      // Early-Optionen: mit Early-Tag (oder Pro) entsperren.
      dropdown.querySelectorAll('option.early-option').forEach(option => {
        option.disabled = !isEarly;
      });

      // Falls die aktuell gewaehlte Option jetzt gesperrt ist (z.B. der
      // Consensus-Default Gemini-Frontier-Low fuer Nicht-Early-Nutzer), auf die
      // erste freigeschaltete Option zuruecksetzen.
      const selected = dropdown.options[dropdown.selectedIndex];
      if (selected && selected.disabled) {
        const firstEnabled = Array.from(dropdown.options).find(opt => !opt.disabled);
        if (firstEnabled) dropdown.selectedIndex = firstEnabled.index;
      }
    });

    if (typeof window.App.applyTierDefaultModels === "function") {
      window.App.applyTierDefaultModels(isPro, isEarly);
    }

    if (window.restoreModelSelections) {
      window.restoreModelSelections();
    }

    if (typeof window.syncCustomModelPickers === "function") {
      window.syncCustomModelPickers();
    }

    // FIX: Nach dem Restore prüfen, ob Deep Think aktiv ist, 
    // und die Texte wieder auf die Reasoning-Namen setzen.
    if (typeof window.App.updateDeepThinkText === "function") {
      window.App.updateDeepThinkText();
    }
  }

  window.updateUserTierUI = updateUserTierUI;
  window.updatePremiumModelsState = updatePremiumModelsState;
})();
