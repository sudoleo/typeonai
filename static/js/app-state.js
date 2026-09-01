// Central owner-enforced state for cross-module app data.  window properties
// remain read-only compatibility views while classic-script modules migrate.
(function () {
  window.App = window.App || {};

  // Deep-frozen: the owner table IS the enforcement. A shallow freeze left
  // each entry writable, so any script could reassign an owner and then write
  // whatever it liked through the front door (found by tests/js/app-state.test.mjs).
  const definitions = Object.freeze({
    lastQuestion: { owner: "run", initial: "" },
    currentEvidenceSources: { owner: "evidence", initial: [] },
    consensusCitationMeta: { owner: "consensus", initial: null },
    lastShareResultId: { owner: "share", initial: null },
    // Drei Stufen, zwei Flags: isUserPro heisst weiterhin "darf Frontier-
    // Modelle und Deep Think" (Plus -> false), isUserPlus heisst "darf
    // Anhaenge und Resolve" (Plus und Pro -> true). Wer nur eines der beiden
    // liest, sperrt Plus im Zweifel wie Free statt wie Pro.
    userTier: { owner: "userTier", initial: "free" },
    // accountTier ist NICHT dasselbe wie userTier. userTier ist die Stufe, die
    // gerade auf dem Schirm gilt -- run-view.js und query-send.js stellen sie
    // auf die Stufe des sichtbaren Laufs, damit Modellauswahl und Gates zu dem
    // passen, was da steht. accountTier ist die Stufe des KONTOS und hat genau
    // zwei Schreiber (/user_status und /usage in firebase.js). Die Marke am
    // Konto-Kuerzel haengt an dieser hier: sonst wechselt sie die Farbe, sobald
    // jemand einen alten Free-Lauf oeffnet.
    accountTier: { owner: "userTier", initial: "free" },
    isUserPro: { owner: "userTier", initial: false },
    isUserPlus: { owner: "userTier", initial: false },
    currentMaxLimit: { owner: "userTier", initial: null },
    currentDeepLimit: { owner: "userTier", initial: null },
    spinnerHTML: { owner: "runUi", initial: "" }
  });
  Object.values(definitions).forEach(Object.freeze);
  const values = Object.fromEntries(
    Object.entries(definitions).map(([key, value]) => [key, value.initial])
  );

  function assertOwner(key, owner) {
    const definition = definitions[key];
    if (!definition) throw new Error(`Unknown app state key: ${key}`);
    if (definition.owner !== owner) {
      throw new Error(`State ${key} belongs to ${definition.owner}, not ${owner}`);
    }
  }

  function set(key, value, owner) {
    assertOwner(key, owner);
    values[key] = value;
    document.dispatchEvent(new CustomEvent("app:state-change", {
      detail: { key, owner, value }
    }));
    return value;
  }

  function get(key) {
    if (!definitions[key]) throw new Error(`Unknown app state key: ${key}`);
    return values[key];
  }

  Object.keys(definitions).forEach((key) => {
    const existing = Object.prototype.hasOwnProperty.call(window, key)
      ? window[key]
      : undefined;
    if (existing !== undefined) values[key] = existing;
    Object.defineProperty(window, key, {
      configurable: false,
      enumerable: true,
      get: () => values[key],
      set: () => {
        throw new Error(`Direct write to window.${key} is forbidden; use App.state.set()`);
      }
    });
  });

  // Die eine Lesart der Kontostufe im Frontend. Sie steht hier statt in
  // user-tier.js, weil app-state.js im head-Bundle laedt: jeder spaetere
  // Konsument (run-view, watch, sidebar) findet sie dann schon vor.
  // Gegenstueck zu normalize_tier() in app/core/entitlements.py.
  function normalizeTier(value) {
    // Booleans bleiben erlaubt: aeltere Aufrufer reichen data.is_pro durch.
    if (value === true) return "pro";
    if (value === false || value === null || value === undefined) return "free";
    const text = String(value).trim().toLowerCase();
    if (text === "pro" || text === "premium") return "pro";
    if (text === "plus") return "plus";
    return "free";
  }

  window.App.normalizeTier = normalizeTier;
  window.App.state = Object.freeze({ get, set, definitions });
})();
