// Auth identity/generation owner. Firebase supplies provider callbacks; this
// module owns the race-prevention state consumed by bookmarks and app views.
(function () {
  window.App = window.App || {};
  let uid = null;
  let generation = 0;
  let known = false;

  function snapshot() {
    return Object.freeze({ known, uid, generation });
  }

  function syncCompatibilityState() {
    window.__consensioAuthState = snapshot();
  }

  function setIdentity(nextUid) {
    const normalized = nextUid || null;
    if (normalized !== uid) {
      uid = normalized;
      generation += 1;
    }
    syncCompatibilityState();
    return generation;
  }

  function publish(nextUid) {
    known = true;
    uid = nextUid || null;
    syncCompatibilityState();
    window.dispatchEvent(new CustomEvent("consensio:auth-state", {
      detail: window.__consensioAuthState
    }));
  }

  function isCurrent(expectedUid, expectedGeneration, providerUid) {
    return expectedGeneration === generation
      && !!expectedUid
      && uid === expectedUid
      && providerUid === expectedUid;
  }

  syncCompatibilityState();
  window.App.authState = Object.freeze({
    get uid() { return uid; },
    get generation() { return generation; },
    get known() { return known; },
    setIdentity,
    publish,
    isCurrent,
    snapshot
  });
})();

