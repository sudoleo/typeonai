(function () {
  window.App = window.App || {};
  const state = {
    telegram: null,
    limits: null,
    limitRequest: null,
    sessionEpoch: 0,
    authUid: window.__consensioAuthState?.uid || null,
    resetSession() {
      this.sessionEpoch += 1;
      this.telegram = null;
      this.limits = null;
      this.limitRequest = null;
    },
    updateAuthUid(uid) {
      const changed = uid !== this.authUid;
      this.authUid = uid;
      return changed;
    }
  };
  window.App.watchState = state;
})();

