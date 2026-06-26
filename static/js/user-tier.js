// =====================================================================
// user-tier.js
// Tier-/Pro-UI: Badge, Upgrade-Link, Deep-Search-Sperre, Premium-Modell-
// Optionen je nach Pro/Free/ausgeloggt. In eigene IIFE gekapselt.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.updateUserTierUI, window.updatePremiumModelsState.
// Abhaengigkeiten: window.setCurrentUsageLimits, window.restoreModelSelections,
// window.syncCustomModelPickers, window.App.updateDeepThinkText,
// window.App.applyTierDefaultModels, window.isUserPro (State).
// =====================================================================

(function () {
  function updateUserTierUI(isPro, isLoggedIn = false) { // Standardmäßig false
    // 1. Globalen Status aktualisieren
    window.isUserPro = isPro;

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
      window.setCurrentUsageLimits(false);

      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false);

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
      window.setCurrentUsageLimits(true);

      // Dropdowns entsperren
      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(true);

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
      window.setCurrentUsageLimits(false);

      // Dropdowns sperren
      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false);

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

  function updatePremiumModelsState(isPro) {
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

      const premiumOptions = dropdown.querySelectorAll('option.premium-option');

      premiumOptions.forEach(option => {
        option.textContent = option.textContent
          .replace(/^Pro:\s*/i, '')
          .replace(' (Pro only)', '')
          .trim();

        if (isPro) {
          // --- PRO USER ---
          option.disabled = false;

        } else {
          // --- FREE USER ---
          option.disabled = true;

          // Falls durch einen Fehler (Cache) ausgewählt -> Reset
          if (option.selected) {
            dropdown.selectedIndex = 0;
          }
        }
      });
    });

    if (typeof window.App.applyTierDefaultModels === "function") {
      window.App.applyTierDefaultModels(isPro);
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
