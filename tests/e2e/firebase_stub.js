// E2E-Stub fuer /static/firebase.js - wird von Playwright per Route
// anstelle des echten Moduls ausgeliefert (siehe tests/e2e/conftest.py).
// Simuliert einen eingeloggten, E-Mail-verifizierten Free-User ohne echtes
// Firebase. Das Sentinel-Token akzeptiert das Backend nur mit MOCK_AUTH=1.
//
// Muss alle window.*-Vertraege bedienen, die andere Module OHNE Optional-
// Chaining aufrufen (saveBookmark, saveBookmarkConsensus, recordModelVote,
// sendFeedback, window.auth.currentUser) - siehe docs/codebase-map.md §8.

const E2E_TOKEN = "e2e-mock-token";

window.auth = {
  currentUser: {
    uid: "e2e-mock-user",
    email: "e2e@consens.io.invalid",
    emailVerified: true,
    getIdToken: async () => E2E_TOKEN,
  },
};

function configuredLimit(key, fallback) {
  const value = Number((window.APP_LIMITS || {})[key]);
  return Number.isFinite(value) ? value : fallback;
}

window.LIMITS = {
  FREE: {
    NORMAL: configuredLimit("free_usage_limit", 25),
    DEEP: configuredLimit("free_deep_search_limit", 12),
  },
  PRO: {
    NORMAL: configuredLimit("pro_usage_limit", 500),
    DEEP: configuredLimit("pro_deep_search_limit", 50),
  },
};
window.currentMaxLimit = window.LIMITS.FREE.NORMAL;
window.currentDeepLimit = window.LIMITS.FREE.DEEP;
window.isUserPro = false;
window.isUserEarly = false;
window.bookmarksData = [];

window.recordModelVote = () => {};
window.saveBookmark = () => {};
window.saveBookmarkConsensus = () => {};
window.loadBookmarks = async () => {};
window.deleteBookmark = async () => {};
window.sendFeedback = async () => ({ ok: true });

// Tier-UI als "eingeloggt, Free" initialisieren, sobald user-tier.js geladen
// ist. Module laufen vor den defer-Skripten; das echte firebase.js erledigt
// das im asynchronen onAuthStateChanged-Callback, daher hier ein Poll.
(function initTierUI(attempt) {
  if (typeof window.updateUserTierUI === "function") {
    window.updateUserTierUI(false, true, false);
    return;
  }
  if (attempt < 100) setTimeout(() => initTierUI(attempt + 1), 50);
})(0);
