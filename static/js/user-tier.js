// =====================================================================
// user-tier.js
// Tier-/Pro-UI: Badge, "Why limits"-Link, Deep-Search-Sperre, Premium-Modell-
// Optionen je nach Pro/Plus/Free/ausgeloggt. In eigene IIFE gekapselt.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.updateUserTierUI, window.updatePremiumModelsState,
// window.App.accountTier ({set, render}) fuer die Marke am Konto-Kuerzel.
// window.App.normalizeTier kommt aus app-state.js (head-Bundle).
// Abhaengigkeiten: window.setCurrentUsageLimits (optional waehrend Init), window.restoreModelSelections,
// window.syncCustomModelPickers, window.App.updateDeepThinkText,
// window.App.applyTierDefaultModels, window.isUserPro (State).
//
// Drei Stufen, zwei Flags (Serverseite: app/core/entitlements.py):
//   Free  - Basis-Modelle, kleines Kontingent.
//   Plus  - Modellauswahl wie Free, KEIN Deep Think, aber Anhaenge, Resolve
//           und das groesste Kontingent.
//   Pro   - alles.
// isUserPro bleibt das Modell-/Deep-Think-Flag und ist fuer Plus false;
// isUserPlus ist "Plus oder Pro". Wer nur isUserPro liest, sperrt Plus wie
// Free -- das ist die sichere Richtung.
// =====================================================================

(function () {
  const TIER_FREE = "free";
  const TIER_PLUS = "plus";
  const TIER_PRO = "pro";

  const normalizeTier = window.App.normalizeTier;

  function updateUserTierUI(tierValue, isLoggedIn = false) {
    const tier = normalizeTier(tierValue);
    const isPro = tier === TIER_PRO;
    const isPlus = tier === TIER_PLUS || isPro;

    // 1. Globalen Status aktualisieren.
    window.App.state.set("userTier", tier, "userTier");
    window.App.state.set("isUserPro", isPro, "userTier");
    window.App.state.set("isUserPlus", isPlus, "userTier");

    // Follow-up-Affordance neu rendern (tierunabhängig, aber der Composer
    // wird bei jedem Tier-Wechsel ohnehin neu aufgebaut).
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
      window.setCurrentUsageLimits?.(TIER_FREE);

      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false);

      // Deep Search sperren (wie bei Free User)
      if (deepSearchLabel) {
        deepSearchLabel.classList.add("locked");
        deepSearchLabel.title = "Login required";
      }
      return; // Funktion hier beenden!
    }

    // === CASE 2: EINGELOGGT (Pro, Plus oder Free) ===
    // Das Badge traegt den Namen der Stufe; nur Free hat keines und sieht
    // dafuer den "Why limits"-Link.
    if (badge) {
      badge.style.display = tier === TIER_FREE ? "none" : "inline-block";
      if (tier !== TIER_FREE) badge.textContent = isPro ? "Pro" : "Plus";
      badge.classList.toggle("is-plus", tier === TIER_PLUS);
    }
    // Der Link ist ein Flex-Container (Glyph + Text), nicht inline-block.
    if (upgradeLink) upgradeLink.style.display = tier === TIER_FREE ? "inline-flex" : "none";

    // Limits: die Stufe selbst, nicht mehr nur "Pro ja/nein".
    window.setCurrentUsageLimits?.(tier);

    if (isPro) {
      const proModal = document.getElementById("proFeatureModal");
      if (proModal) proModal.style.display = "none";

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
      // --- FREE ODER PLUS (EINGELOGGT) ---
      // Beide fahren dieselbe Modellauswahl und haben kein Deep Think; genau
      // das macht Plus fuer Tester bezahlbar.
      if (typeof updatePremiumModelsState === "function") updatePremiumModelsState(false);

      // Deep Search ausschalten & sperren
      const deepToggle = document.getElementById("deepSearchToggle");
      if (deepToggle && deepToggle.checked) {
        deepToggle.checked = false;
        if (typeof window.App.updateDeepThinkText === 'function') window.App.updateDeepThinkText();
      }
      if (deepSearchLabel) {
        deepSearchLabel.classList.add("locked");
        deepSearchLabel.title = "Off by default: one Deep Think run costs a multiple of a normal one";

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
      ...(window.App?.modelPrefs || []).map(pref => pref.selectId)
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

      // Falls die aktuell gewaehlte Option jetzt gesperrt ist, auf die erste
      // freigeschaltete Option zuruecksetzen.
      const selected = dropdown.options[dropdown.selectedIndex];
      if (selected && selected.disabled) {
        const firstEnabled = Array.from(dropdown.options).find(opt => !opt.disabled);
        if (firstEnabled) dropdown.selectedIndex = firstEnabled.index;
      }
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

  // === Die Kontostufe am Konto-Kuerzel ======================================
  // Pro und Plus brauchen eine Kennzeichnung, die IMMER da ist -- das
  // #proBadge im Kontingent-Panel sieht nur, wer den Ring anklickt. Statt
  // einer weiteren Zeile in der ohnehin engen Fusszeile traegt sie das
  // Element, das den Account schon vertritt: der Kreis mit dem Kuerzel
  // wechselt die Farbe. Ausgeschrieben wird die Stufe erst im Konto-Popup,
  // eine Klick-Tiefe weiter -- dezent heisst hier "erkennbar, aber ohne
  // eigene Flaeche".
  //
  // Quelle ist accountTier, nicht userTier: userTier folgt dem sichtbaren
  // LAUF (run-view.js), und ein alter Free-Lauf haette die Marke am Konto
  // sonst ausgeknipst.
  function renderAccountTierMark() {
    const tier = normalizeTier(window.accountTier);

    // Das Kuerzel wird von firebase.js bei jedem Login neu geschrieben; wer
    // die Marke setzt, muss deshalb nach jedem Rendern noch einmal ran.
    const icon = document.getElementById("emailIcon");
    if (icon) {
      icon.classList.toggle("is-pro", tier === TIER_PRO);
      icon.classList.toggle("is-plus", tier === TIER_PLUS);
    }

    const label = document.getElementById("accountTierLabel");
    if (label) {
      label.hidden = tier === TIER_FREE;
      // is-subtle ist die Gold-Variante und gehoert deshalb nur Pro: bliebe sie
      // an Plus, faerbte .dark-mode .pro-badge.is-subtle den Text gold und
      // schlaege dabei .pro-badge.is-plus.
      label.classList.toggle("is-subtle", tier === TIER_PRO);
      label.classList.toggle("is-plus", tier === TIER_PLUS);
      if (tier !== TIER_FREE) label.textContent = tier === TIER_PRO ? "Pro" : "Plus";
    }
  }

  // Nur die beiden Konto-Endpunkte rufen das: /user_status beim Laden und
  // /usage als zweiter, autoritativer Schnappschuss. Ein Lauf darf hier nicht
  // hineinschreiben.
  function setAccountTier(tierValue) {
    window.App.state.set("accountTier", normalizeTier(tierValue), "userTier");
    renderAccountTierMark();
  }

  window.updateUserTierUI = updateUserTierUI;
  window.updatePremiumModelsState = updatePremiumModelsState;
  window.App.accountTier = { set: setAccountTier, render: renderAccountTierMark };
})();
