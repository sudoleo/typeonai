(function () {
  const config = document.getElementById("adminBootstrapConfig");
  window.FIREBASE_CONFIG = {
    apiKey: config?.dataset.firebaseApiKey || "",
    authDomain: config?.dataset.firebaseAuthDomain || "",
    projectId: config?.dataset.firebaseProjectId || "",
    storageBucket: config?.dataset.firebaseStorageBucket || "",
    messagingSenderId: config?.dataset.firebaseMessagingSenderId || "",
    appId: config?.dataset.firebaseAppId || ""
  };
})();

