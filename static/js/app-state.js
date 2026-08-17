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
    isUserPro: { owner: "userTier", initial: false },
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

  window.App.state = Object.freeze({ get, set, definitions });
})();
