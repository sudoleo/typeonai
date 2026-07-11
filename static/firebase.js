import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, query, orderBy, onSnapshot, doc, setDoc, getDoc, increment, addDoc, deleteDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithCustomToken, signOut, onAuthStateChanged, sendPasswordResetEmail, sendEmailVerification, onIdTokenChanged } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
import {
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence,
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

const googleProvider = new GoogleAuthProvider();
// Optional: Kontoauswahl erzwingen
googleProvider.setCustomParameters({ prompt: "select_account" });

if (!window.FIREBASE_CONFIG || !window.FIREBASE_CONFIG.apiKey) {
  console.error("[FATAL] FIREBASE_CONFIG fehlt/leer. Abbruch (kein Auto-Init).", window.FIREBASE_CONFIG);
  // Optional: UI-Hinweis anzeigen
  const el = document.getElementById("loginError");
  if (el) el.textContent = "Init failed: Missing Firebase config.";
  // return;  // <- wenn du sicher abbrechen willst
}

// Initialisiere Firebase mit der globalen Konfiguration, die aus dem HTML kommt
const app = initializeApp(window.FIREBASE_CONFIG);
const db = getFirestore(app);

// Initialisiere Auth
const auth = getAuth(app);
window.auth = auth;

function trackAppEvent(eventName, eventData = {}) {
  if (typeof window.trackUmamiEvent === "function") {
    window.trackUmamiEvent(eventName, eventData);
  }
}

// Logout-Bestätigung als In-App-Modal (#logoutConfirmModal in index.html)
// statt Browser-window.confirm. Buttons werden einmalig beim Modul-Load
// verdrahtet; der Logout-Button im Account-Popup ruft nur openLogoutConfirm().
function closeLogoutConfirm() {
  const modal = document.getElementById("logoutConfirmModal");
  if (modal) modal.style.display = "none";
}

function performLogout() {
  trackAppEvent("auth_logout_click");
  closeLogoutConfirm();
  signOut(auth).catch(err => console.error("Logout-Fehler", err));
}

function openLogoutConfirm() {
  const modal = document.getElementById("logoutConfirmModal");
  if (!modal) {
    // Fallback (Seite ohne Modal-Markup): Browser-Dialog wie früher.
    if (window.confirm("Log out of consens.io?")) performLogout();
    return;
  }
  modal.style.display = "block";
  document.getElementById("logoutCancelBtn")?.focus();
}

(function initLogoutConfirmModal() {
  const modal = document.getElementById("logoutConfirmModal");
  if (!modal) return;
  document.getElementById("logoutConfirmBtn")?.addEventListener("click", performLogout);
  document.getElementById("logoutCancelBtn")?.addEventListener("click", closeLogoutConfirm);
  // Klick auf den Backdrop oder Escape schließt ohne Logout.
  modal.addEventListener("click", e => {
    if (e.target === modal) closeLogoutConfirm();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && modal.style.display === "block") closeLogoutConfirm();
  });
})();

// Standard-Persistenz global setzen (beim Laden, nicht im Click-Handler)
setPersistence(auth, browserLocalPersistence)
  .catch(() => setPersistence(auth, browserSessionPersistence))
  .catch(err => {
    console.error("[Auth] Failed to set persistence:", err);
  });

function renderMarkdownSafe(md) {
  const html = marked.parse(md || "");
  return DOMPurify.sanitize(html, {
    // nur sichere Protokolle für Links erlauben
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|tel:)/i
  });
}

// Optional: Nach dem Einfügen alle Links "sicher" machen
function enhanceLinks(rootEl) {
  if (!rootEl) return;
  rootEl.querySelectorAll("a[href]").forEach(a => {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  });
}

// Convenience: Sicher einfügen + Links härten
function injectHtmlSafe(containerEl, md) {
  containerEl.innerHTML = renderMarkdownSafe(md);
  enhanceLinks(containerEl);
}

function getConfiguredLimit(key, fallback) {
  const raw = (window.APP_LIMITS || {})[key];
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

// Globale Limits Definition
window.LIMITS = {
  FREE: {
    NORMAL: getConfiguredLimit("free_usage_limit", 25),
    DEEP: getConfiguredLimit("free_deep_search_limit", 12)
  },
  PRO: {
    NORMAL: getConfiguredLimit("pro_usage_limit", 500),
    DEEP: getConfiguredLimit("pro_deep_search_limit", 50)
  }
};

// Globale Variablen für den aktuellen Zustand (Startwert: Free)
window.currentMaxLimit = window.LIMITS.FREE.NORMAL;
window.currentDeepLimit = window.LIMITS.FREE.DEEP;

// merken, dass wir Bookmarks schon einmal geladen haben
let bookmarksLoaded = false;

async function checkUserStatusOnLoad(user, token) {
  if (!user || !token) return;

  try {
    const response = await fetch("/user_status", {
      method: "GET",
      headers: {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
      }
    });

    if (response.ok) {
      const data = await response.json();

      // 1. Globale Limits sofort aktualisieren. Pro schliesst Early ein.
      window.currentMaxLimit = data.limit;
      window.currentDeepLimit = data.deep_limit;
      window.isUserPro = data.is_pro;
      const hasEarlyAccess = Boolean(data.is_pro || data.is_early);
      window.isUserEarly = hasEarlyAccess;

      // 2. UI AKTUALISIEREN

      // A) Der saubere Weg (falls vorhanden):
      if (typeof window.updateUserTierUI === "function") {
          window.updateUserTierUI(data.is_pro, true, data.is_early);
      }
      if (typeof window.setCurrentUsageLimits === "function") {
          window.setCurrentUsageLimits(data.is_pro, data);
      } else {
          window.currentMaxLimit = data.limit;
          window.currentDeepLimit = data.deep_limit;
      }

      // B) FALLBACK (Hier war der Fehler):
      const badge = document.getElementById("proBadge");
      const upgradeLink = document.getElementById("upgradeLink");
      const premiumOptions = document.querySelectorAll('.premium-option');
      const earlyOptions = document.querySelectorAll('.early-option');

      // Early-Optionen: mit Early-Tag (oder Pro) entsperren, sonst sperren.
      earlyOptions.forEach(option => {
          option.disabled = !hasEarlyAccess;
      });

      if (data.is_pro) {
          // === IST PRO ===
          if (badge) badge.style.display = "inline-block";
          if (upgradeLink) upgradeLink.style.display = "none";

          premiumOptions.forEach(option => {
              option.disabled = false;
              option.textContent = option.textContent
                  .replace(/^Pro:\s*/i, '')
                  .replace(' (Pro only)', '')
                  .trim();
          });

      } else {
          // === IST FREE ===
          if (badge) badge.style.display = "none";
          if (upgradeLink) upgradeLink.style.display = "inline-block";

          premiumOptions.forEach(option => {
              option.disabled = true;
              option.textContent = option.textContent
                  .replace(/^Pro:\s*/i, '')
                  .replace(' (Pro only)', '')
                  .trim();

          });
      }
      
    }
  } catch (error) {
    console.error("Fehler beim User-Status Check:", error);
  }
}

onIdTokenChanged(auth, async (user) => {
  const loginContainer = document.getElementById("loginContainer");
  const usageOptions   = document.getElementById("usageOptions");

  if (user) {
    // **NEU: Harte Client-Gate—kein Token persistieren, keine Calls, wenn unverified**
    if (!user.emailVerified) {
      localStorage.removeItem("id_token");
      if (typeof window.updateQuestionInputAccess === "function") {
        window.updateQuestionInputAccess();
      }
      if (usageOptions) usageOptions.style.display = "none";
      if (loginContainer) loginContainer.innerText = "Verify your e-mail to continue";
      return; // <--- ganz wichtig
    }

    // ab hier nur noch verifizierte Nutzer
    const token = await user.getIdToken(/* forceRefresh= */ false);
    localStorage.setItem("id_token", token);
    if (typeof window.updateQuestionInputAccess === "function") {
      window.updateQuestionInputAccess();
    }

    try {
      // Dieser Call übernimmt nun die Arbeit für ALLE Login-Arten (Google, Email, Reload)
      const res = await fetch("/confirm-registration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: token })
      });
      
      if (!res.ok) {
        // Hilft dir beim Debuggen: Was genau sagt das Backend?
        const errData = await res.json();
        console.warn("Registration Check Failed:", errData);
      }
    } catch (err) {
      console.error("Error during confirm-registration:", err);
    }

    await checkUserStatusOnLoad(user, token);

    // 2) Usage laden
    fetchUsageData(token);

    // 3) Bookmarks einmal pro Login laden
    if (!bookmarksLoaded) {
      await loadBookmarks();
      bookmarksLoaded = true;
    }

    // 4) Usage-UI anzeigen
    if (usageOptions) {
      usageOptions.style.display = "block";
    }

    // Account-Bereich (Löschen) in den Settings einblenden
    const accountSection = document.getElementById("accountSettingsSection");
    if (accountSection) accountSection.style.display = "block";

    // 5) E‑Mail & Logout als Popup
    const emailInitial = user.email.charAt(0).toUpperCase();
    loginContainer.innerHTML = `
      <div class="email-container" style="position: relative; display: inline-block;">
        <span id="emailIcon" class="email-icon" style="display: inline-block; width: 32px; height: 32px; border-radius: 50%; background-color: var(--sidebar-border, #ddd); color: var(--text-color); text-align: center; line-height: 32px; font-weight: bold; font-size: 14px; cursor: pointer;">${emailInitial}</span>
        <div id="emailPopup" class="email-popup" style="display: none; position: absolute; top: 45px; right: 0; background-color: var(--container-bg, #fff); border: 1px solid var(--sidebar-border, #ddd); border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); z-index: 100; min-width: 150px;">
          <div class="popup-content" style="display: flex; flex-direction: column; align-items: flex-start;">
            <span class="user-email" style="margin-bottom: 12px; font-weight: 500; font-size: 0.95rem; color: var(--text-color);">${user.email}</span>
            <a id="sharedLinksButton" class="top-bar-about" style="cursor: pointer; text-decoration: none; align-self: flex-start; padding: 5px 0;">Shared links</a>
            <a id="watchedLinksButton" class="top-bar-about" style="cursor: pointer; text-decoration: none; align-self: flex-start; padding: 5px 0;">Watched</a>
            <a id="logoutButton" class="top-bar-about" style="cursor: pointer; text-decoration: none; align-self: flex-start; padding: 5px 0;">Logout</a>
          </div>
        </div>
      </div>
    `;
    const emailIcon = document.getElementById("emailIcon");
    const emailPopup = document.getElementById("emailPopup");
    const logoutButton = document.getElementById("logoutButton");
    const sharedLinksButton = document.getElementById("sharedLinksButton");
    const watchedLinksButton = document.getElementById("watchedLinksButton");

    emailIcon.addEventListener("click", e => {
      e.stopPropagation();
      emailPopup.style.display = emailPopup.style.display === "none" ? "block" : "none";
      trackAppEvent("app_account_menu_toggled", { open: emailPopup.style.display === "block" });
    });

    // Übersicht der geteilten Consensus-Links direkt aus dem User-Menü öffnen.
    // stopPropagation ist wichtig: der umschließende #loginContainer hat einen
    // Klick-Handler, der eingeloggte Nutzer sonst ausloggt.
    if (sharedLinksButton) {
      sharedLinksButton.addEventListener("click", e => {
        e.stopPropagation();
        emailPopup.style.display = "none";
        if (typeof window.openShareDialog === "function") {
          window.openShareDialog("list");
        }
      });
    }

    if (watchedLinksButton) {
      watchedLinksButton.addEventListener("click", e => {
        e.stopPropagation();
        emailPopup.style.display = "none";
        if (typeof window.openWatchDialog === "function") {
          window.openWatchDialog("list");
        }
      });
    }

    logoutButton.addEventListener("click", e => {
      // stopPropagation: Klick soll weder das Icon-Toggle noch den
      // loginContainer-Handler treffen.
      e.stopPropagation();
      emailPopup.style.display = "none";
      openLogoutConfirm();
    });

    document.addEventListener("click", e => {
      if (!loginContainer.contains(e.target)) {
        emailPopup.style.display = "none";
      }
    });

    } else {
        // Cleanup bei Logout
        localStorage.removeItem("id_token");
        if (typeof window.updateQuestionInputAccess === "function") {
          window.updateQuestionInputAccess();
        }
        loginContainer.innerText = "Log in";

        if (usageOptions) usageOptions.style.display = "none";

        const accountSection = document.getElementById("accountSettingsSection");
        if (accountSection) accountSection.style.display = "none";

        document.getElementById("bookmarksContainer").innerHTML = "";
        bookmarksLoaded = false;
        
        // A) Badge verstecken (Direkter Zugriff)
        const badge = document.getElementById("proBadge");
        if (badge) badge.style.display = "none";

        // B) Limits auf Free zurücksetzen
        window.currentMaxLimit = window.LIMITS.FREE.NORMAL;
        window.currentDeepLimit = window.LIMITS.FREE.DEEP;

        // C) Premium Modelle wieder sperren (HIER WAR DER FEHLER)
        window.isUserEarly = false;
        const premiumOptions = document.querySelectorAll('.premium-option');
        premiumOptions.forEach(option => {
            option.disabled = true;
            option.textContent = option.textContent
                .replace(/^Pro:\s*/i, '')
                .replace(' (Pro only)', '')
                .trim();

            // Falls ausgewählt (Cache-Problem), zurücksetzen auf Standard
            if (option.selected) {
                option.parentNode.selectedIndex = 0;
            }
        });

        // C2) Early-Modelle ebenfalls wieder sperren
        document.querySelectorAll('.early-option').forEach(option => {
            option.disabled = true;
            if (option.selected) {
                const parent = option.parentNode;
                const firstEnabled = Array.from(parent.options).find(opt => !opt.disabled);
                parent.selectedIndex = firstEnabled ? firstEnabled.index : 0;
            }
        });

        if (typeof window.updateUserTierUI === "function") {
            window.updateUserTierUI(false, false); // isPro=false, isLoggedIn=false
        }
      }
    });

function fetchUsageData(token) {
  // DOM-Elemente innerhalb der Funktion abrufen:
  const freeDisplay = document.getElementById("freeUsageDisplay");
  const deepDisplay = document.getElementById("deepUsageDisplay");
  
  // Sicherstellen, dass die Elemente vorhanden sind
  if (!freeDisplay || !deepDisplay) {
    console.error("Benötigte DOM-Elemente nicht gefunden.");
    return;
  }
  
  // API-Aufruf starten:
  fetch("/usage", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: token })
  })
    .then(response => response.json())
    .then(data => {
      if (typeof window.setCurrentUsageLimits === "function") {
        window.setCurrentUsageLimits(data.is_pro === true, data);
      } else {
        const totalLimit = Number(data.total_limit);
        const deepTotalLimit = Number(data.deep_total_limit);
        if (Number.isFinite(totalLimit)) window.currentMaxLimit = totalLimit;
        if (Number.isFinite(deepTotalLimit)) window.currentDeepLimit = deepTotalLimit;
      }
      freeDisplay.innerHTML = 'Requests: <strong>' + data.remaining + ' / ' + window.currentMaxLimit + '</strong>';
      deepDisplay.innerHTML = 'Deep Think: <strong>' + data.deep_remaining + ' / ' + window.currentDeepLimit + '</strong>';
    })
    .catch(err => console.error("Error when retrieving the quota:", err));
}

function mapFirebaseLoginError(error) {
  // Sicherheitsfreundliche, generische Messages
  switch (error.code) {
    case "auth/user-not-found":
    case "auth/wrong-password":
    case "auth/invalid-email":
      return "Login failed. Please check your e-mail and password.";

    case "auth/too-many-requests":
      return "Too many login attempts. Please try again later.";

    case "auth/network-request-failed":
      return "Network error. Please check your internet connection and try again.";

    default:
      return "An error occurred while logging in. Please try again.";
  }
}

function mapFirebaseRegisterError(error) {
  switch (error.code) {
    case "auth/email-already-in-use":
      return "This e-mail address is already in use.";
    case "auth/invalid-email":
      return "Please enter a valid e-mail address.";
    case "auth/weak-password":
      return "Password is too weak. Please choose a stronger password.";
    case "auth/network-request-failed":
      return "Network error. Please check your internet connection and try again.";
    default:
      return "Registration failed. Please try again.";
  }
}

function mapPasswordResetError(error) {
  switch (error.code) {
    case "auth/user-not-found":
      return "No account was found for this e-mail address.";
    case "auth/invalid-email":
      return "Please enter a valid e-mail address.";
    case "auth/network-request-failed":
      return "Network error. Please check your internet connection and try again.";
    default:
      return "An error occurred while resetting the password. Please try again.";
  }
}

// Login-Funktion
document.getElementById("loginButton").addEventListener("click", () => {
  const email = document.getElementById("loginEmail").value;
  const password = document.getElementById("loginPassword").value;
  trackAppEvent("auth_email_login_started");
  
  // Fehleranzeige erstmal leeren
  loginErr.textContent = "";

  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      if (user.emailVerified) {
        // Login erfolgreich, Token speichern und Seite neu laden
        user.getIdToken().then((token) => {
          localStorage.setItem("id_token", token);

          fetch("/confirm-registration", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id_token: token })
          })
            .catch(err => console.error("Confirm-Registration-Fehler:", err));

          window.location.href = "/app";
        });
      } else {
        try { localStorage.removeItem("id_token"); } catch {}
        alert("Please verify your e-mail address first. Check your inbox for the confirmation link.");
        signOut(auth);
      }
      trackAppEvent("auth_email_login_result", { status: user.emailVerified ? "success" : "unverified" });
    })
    .catch((error) => {
      // Statt error.message → gemappte, neutrale Meldung
      const msg = mapFirebaseLoginError(error);
      loginErr.textContent = msg;
      trackAppEvent("auth_email_login_result", { status: "error" });
    });
});

// --- Minimal-invasive Register-/Login-Umschaltung + Registrierung ---

const formEl = document.getElementById("loginForm");
const titleEl = document.getElementById("authTitle");
const singleMailNoteEl = document.getElementById("singleMailNote");

const emailEl = document.getElementById("loginEmail");
const emailConfirmEl = document.getElementById("loginEmailConfirm");
const passEl = document.getElementById("loginPassword");
const passConfirmEl = document.getElementById("loginPasswordConfirm");

const toggleRegisterBtn = document.getElementById("toggleRegister");
const confirmRegisterBtn = document.getElementById("confirmRegisterButton");

const loginBtn = document.getElementById("loginButton");
const registerErr = document.getElementById("registerError");
const loginErr = document.getElementById("loginError");

const forgotBtn = document.getElementById("forgotPasswordButton");

function setMode(mode) {
  formEl.dataset.mode = mode;

  const isRegister = mode === "register";
  // Titel & Hinweise
  titleEl.textContent = isRegister ? "Create account" : "Log in to consens.io";
  singleMailNoteEl.style.display = isRegister ? "block" : "none";

  // Felder ein-/ausblenden
  emailConfirmEl.style.display = isRegister ? "" : "none";
  passConfirmEl.style.display = isRegister ? "" : "none";

  // Primär-Buttons
  confirmRegisterBtn.style.display = isRegister ? "" : "none";
  loginBtn.style.display = isRegister ? "none" : "";

  // Forgot Password ausblenden im Register-Modus
  if (forgotBtn) forgotBtn.style.display = isRegister ? "none" : "";

  // Toggle-Text
  toggleRegisterBtn.textContent = isRegister
    ? "Back to login"
    : "New here? Create account";

  // Fehler leeren
  registerErr.textContent = "";
  loginErr.textContent = "";
}

toggleRegisterBtn.addEventListener("click", () => {
  const current = formEl.dataset.mode === "register" ? "register" : "login";
  setMode(current === "login" ? "register" : "login");
  trackAppEvent("auth_mode_changed", { mode: current === "login" ? "register" : "login" });
});

// --- Registrierung (läuft NICHT über loginButton, sondern über confirmRegisterButton) ---
confirmRegisterBtn.addEventListener("click", () => {
  registerErr.textContent = "";
  trackAppEvent("auth_register_started");

  const email = (emailEl.value || "").trim();
  const email2 = (emailConfirmEl.value || "").trim();
  const password = passEl.value || "";
  const password2 = passConfirmEl.value || "";

  // Client-Side-Validierung
  if (!email || !email2) {
    registerErr.textContent = "Please enter your e-mail twice.";
    return;
  }
  if (email !== email2) {
    registerErr.textContent = "E-mail addresses do not match.";
    return;
  }
  if (!password || !password2) {
    registerErr.textContent = "Please enter your password twice.";
    return;
  }
  if (password !== password2) {
    registerErr.textContent = "Passwords do not match.";
    return;
  }
  if (password.length < 8) {
    registerErr.textContent = "Password must be at least 8 characters.";
    return;
  }

  // Request an Backend
  fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email, password: password })
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.customToken) {
        // Nutzer mit dem Custom Token anmelden
        signInWithCustomToken(auth, data.customToken)
          .then(() => {
            sendEmailVerification(auth.currentUser)
              .then(() => {
                alert("Registration successful! Please confirm your e-mail address by clicking on the link in the e-mail.");
                signOut(auth);
                setMode("login");
                trackAppEvent("auth_register_result", { status: "success" });
              })
              .catch((error) => {
                // Keine rohen Firebase-Texte
                console.error("Error sending verification e-mail:", error);
                registerErr.textContent = "Error sending the verification e-mail. Please try again later.";
                trackAppEvent("auth_register_result", { status: "email_error" });
              });
          })
          .catch((error) => {
            const msg = mapFirebaseRegisterError(error);
            registerErr.textContent = msg;
            trackAppEvent("auth_register_result", { status: "error" });
          });
      } else if (data.detail) {
        registerErr.textContent = data.detail;
        trackAppEvent("auth_register_result", { status: "error" });
      } else {
        registerErr.textContent = "Unexpected response from server.";
        trackAppEvent("auth_register_result", { status: "error" });
      }
    })
    .catch((error) => {
      registerErr.textContent = error.message;
      trackAppEvent("auth_register_result", { status: "error" });
    });
});

// Standard: beim Öffnen im Login-Modus
setMode("login");

document.getElementById("forgotPasswordButton").addEventListener("click", () => {
  const email = document.getElementById("loginEmail").value;
  trackAppEvent("auth_password_reset_started");
  if (!email) {
    alert("Please enter your e-mail address to reset the password. Check your spam folder.");
    return;
  }
  sendPasswordResetEmail(auth, email)
    .then(() => {
      alert("An e-mail to reset your password has been sent to " + email);
      trackAppEvent("auth_password_reset_result", { status: "success" });
    })
    .catch((error) => {
      const msg = mapPasswordResetError(error);
      alert(msg);
      trackAppEvent("auth_password_reset_result", { status: "error" });
    });
});

// Klick auf den Login-Bereich: Öffne das Modal, wenn nicht angemeldet.
// Eingeloggt passiert hier bewusst NICHTS - das User-Icon öffnet sein eigenes
// Popup (stopPropagation), Logout läuft ausschließlich über den bestätigten
// Logout-Button im Popup. Vorher loggte ein Klick knapp neben das Icon aus.
document.getElementById("loginContainer").addEventListener("click", () => {
  if (!auth.currentUser) {
    document.getElementById("loginModal").style.display = "block";
    trackAppEvent("auth_modal_open");
  }
});

// Schließen des Modals
document.getElementById("closeLoginModal").addEventListener("click", () => {
  document.getElementById("loginModal").style.display = "none";
});

// --- Account löschen (DSGVO Art. 17) ---
document.getElementById("deleteAccountBtn")?.addEventListener("click", async () => {
  if (!auth.currentUser) {
    alert("Please log in first.");
    return;
  }
  const confirmed = window.confirm(
    "Delete your account permanently?\n\nThis removes your account, bookmarks, and all data stored about you. This cannot be undone."
  );
  if (!confirmed) return;

  trackAppEvent("auth_account_deletion_started");
  try {
    const token = await auth.currentUser.getIdToken(true);
    const res = await fetch("/delete_account", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: token })
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.status === "deleted") {
      trackAppEvent("auth_account_deletion_result", { status: "success" });
      try { localStorage.removeItem("id_token"); } catch {}
      alert("Your account and data have been deleted.");
      // Der Auth-Account existiert nicht mehr; lokale Session beenden und zur Startseite
      signOut(auth).catch(() => {});
      window.location.href = "/?landing=1";
    } else {
      trackAppEvent("auth_account_deletion_result", { status: "error" });
      alert(data.detail || data.error || "Account deletion failed. Please try again or contact us.");
    }
  } catch (err) {
    console.error("Account deletion failed:", err);
    trackAppEvent("auth_account_deletion_result", { status: "error" });
    alert("Account deletion failed. Please try again or contact us.");
  }
});

function isIOS() {
  return /iP(ad|hone|od)/i.test(navigator.userAgent);
}

function handleGoogleSignIn() {
  const loginErrorEl = document.getElementById("loginError");
  if (loginErrorEl) loginErrorEl.textContent = "";
  trackAppEvent("auth_google_login_started");

  // GANZ WICHTIG: signInWithPopup wird direkt im Click-Handler aufgerufen,
  // ohne vorherige await-/Promise-Ketten.
  signInWithPopup(auth, googleProvider)
    .then(result => {
      trackAppEvent("auth_google_login_result", { status: "success" });
      return afterGoogleLogin(result.user);
    })
    .catch(err => {
      console.error("Google sign-in failed:", err);
      trackAppEvent("auth_google_login_result", { status: "error" });

      if (!loginErrorEl) return;

      if (err.code === "auth/popup-blocked") {
        // Erster Klick auf Safari kann trotzdem noch geblockt werden,
        // aber wir geben einen klaren Hinweis.
        loginErrorEl.textContent =
          "Your browser blocked the Google login popup. Please allow pop-ups for consens.io and try again.";
        return;
      }

      if (err.code === "auth/popup-closed-by-user") {
        loginErrorEl.textContent =
          "The login window was closed before completing the sign-in.";
        return;
      }

      // statt err.message
      loginErrorEl.textContent = "Google sign-in failed. Please try again later.";
    });
}

document.getElementById("googleLoginButton")?.addEventListener("click", handleGoogleSignIn);

async function afterGoogleLogin(user) {
  // Jetzt *nach* erfolgreichem/versuchtem POST navigieren
  location.replace("/app");
}

// Restlicher Firebase-Code (z.B. Leaderboard, Funktionen, etc.)
const leaderboardRef = collection(db, "leaderboard");
const leaderboardQuery = query(
  leaderboardRef,
  orderBy("BestModel", "desc")
);
onSnapshot(leaderboardQuery, (snapshot) => {
  const leaderboardEl = document.getElementById("leaderboardContent");
  if (!leaderboardEl) return;

  const escapeHtml = (value) => {
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
  };

  let hasRows = false;
  let rank = 0;
  let html = '<div class="leaderboard-list" role="list"><div class="leaderboard-caption">Consensus leaders</div>';
  snapshot.forEach((doc) => {
    const data = doc.data();
    const bestModelVotes = data.BestModel || 0;
    if (!bestModelVotes) return;

    hasRows = true;
    rank += 1;
    const modelName = escapeHtml(doc.id);
    const voteLabel = bestModelVotes === 1 ? "pick" : "picks";

    html += `<div class="leaderboard-row" role="listitem">
      <span class="leaderboard-rank">${rank}</span>
      <span class="leaderboard-model">${modelName}</span>
      <span class="vote BestModel" title="BestModel">${bestModelVotes}<span>${voteLabel}</span></span>
    </div>`;
  });
  html += '</div>';
  leaderboardEl.innerHTML = hasRows
    ? html
    : '<div class="leaderboard-empty">No BestModel votes yet.</div>';
});

async function recordModelVote(model, type) {
  // Prüfe, ob der Nutzer eingeloggt ist.
  if (!auth.currentUser) {
    return;
  }
  
  const id_token = await auth.currentUser?.getIdToken(/* forceRefresh= */ false);
  if (!id_token) {
    console.error("No id_token available for voting.");
    return;
  }

  try {
    const response = await fetch("/vote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id_token: id_token,
        model: model,
        vote_type: type
      })
    });
    const data = await response.json();
    if (!response.ok) {
      console.error("Error recording vote:", data.detail);
    }
  } catch (error) {
    console.error("Error connecting to backend for vote recording:", error);
  }
}

window.recordModelVote = recordModelVote;

// Hilfsfunktion zum Kürzen des Textes auf maximal 5 Wörter
function truncateText(text, maxWords = 5) {
  const words = text.split(' ');
  if (words.length > maxWords) {
    return words.slice(0, maxWords).join(' ') + '...';
  }
  return text;
}

async function saveBookmark(question, response, modelName, mode) {
  if (!auth.currentUser) return;
  const id_token = await auth.currentUser.getIdToken(false);
  if (!id_token) return;

  // HIER: Quellen holen
  const sources = window.currentEvidenceSources || [];

  // Anhänge der zuletzt gesendeten Frage: nur Metadaten (Name/Typ/Größe),
  // die Dateidaten selbst werden bewusst NICHT in Firestore gespeichert.
  const attachmentsMeta = window.lastQuestionAttachmentsMeta || [];

  try {
    const res = await fetch("/bookmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // HIER: sources hinzufügen
      body: JSON.stringify({
        id_token,
        question,
        response,
        modelName,
        mode,
        sources: sources,
        attachments: attachmentsMeta
      })
    });
    const data = await res.json();

    if (!res.ok) {
      console.error("Error saving bookmark:", data.detail);
      return;
    }

    if (data.bookmark) {
        // Initialisiere Array falls leer
        if (!window.bookmarksData) window.bookmarksData = [];
        const existingIndex = window.bookmarksData.findIndex(b => b.id === data.bookmark.id);

        if (existingIndex > -1) {
            window.bookmarksData[existingIndex] = data.bookmark;

        } else {
            // Neu: Vorne ins Array
            window.bookmarksData.unshift(data.bookmark);
            // Und ins UI einfügen
            addBookmarkToDOM(data.bookmark);
        }
        trackAppEvent("app_bookmark_saved", { type: "model", mode });
    }

  } catch (error) {
    console.error("Error in saveBookmark:", error);
  }
}

window.saveBookmark = saveBookmark;

async function saveBookmarkConsensus(question, consensusText, differencesText, differencesData) {
  if (!auth.currentUser) return;
  const id_token = await auth.currentUser?.getIdToken(/* forceRefresh= */ false);
  if (!id_token) return;

  // HIER: Quellen holen
  const sources = window.currentEvidenceSources || [];

  try {
    const res = await fetch("/bookmark/consensus", {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({
         id_token: id_token,
         question: question,
         consensusText: consensusText,
         differencesText: differencesText,
         // Strukturierte Differences (Verdict, Karten, Badges) mitspeichern,
         // damit das Bookmark dieselbe Ansicht wie eine echte Query zeigt.
         differencesData: differencesData || null,
         sources: sources // HIER: Hinzufügen
       })
    });
    const data = await res.json();
    if (!res.ok) {
      console.error("Error saving consensus bookmark:", data.detail);
      return;
    }
    trackAppEvent("app_bookmark_saved", { type: "consensus" });
  } catch (error) {
    console.error("Error in saveBookmarkConsensus:", error);
  }
}
window.saveBookmarkConsensus = saveBookmarkConsensus;

// Diese Funktion füllt die UI mit den Daten eines Bookmarks
function loadSingleBookmarkUI(bookmark) {
    // Konsens-Button deaktivieren
    const conBtn = document.getElementById("consensusButton");
    if(conBtn) conBtn.disabled = true;

    // --- NEU: Quellen wiederherstellen ---
    // 1. Globale Variable setzen, damit injectMarkdown (in index.html) darauf zugreifen kann
    window.currentEvidenceSources = bookmark.sources || [];

    // 2. Die Quellen-Liste (unten im UI) visuell rendern (falls die Funktion existiert)
    if (window.renderEvidenceSources) {
        window.renderEvidenceSources(bookmark.sources || []);
    }

    if (bookmark && bookmark.responses) {
        
        // HELPER: Nutzt die globale Funktion injectMarkdown aus der index.html, 
        // um Markdown korrekt zu rendern, Copy-Buttons hinzuzufügen etc.
        const renderContent = (container, text) => {
            if (!container) return;
            
            const content = text || "";
            
            if (window.injectMarkdown) {
                // Nutzt die Logik aus index.html (Marked + DOMPurify + Copy Buttons)
                window.injectMarkdown(container, content);
            } else {
                // Fallback, falls injectMarkdown noch nicht geladen ist
                // (nutzt die lokale renderMarkdownSafe Funktion von oben in firebase.js)
                container.innerHTML = typeof renderMarkdownSafe === "function" 
                                      ? renderMarkdownSafe(content) 
                                      : content;
            }
        };

        // Funktion für die Modell-Boxen
        const setModelContent = (id, text) => {
            const el = document.getElementById(id);
            if (el) {
                const contentArea = el.querySelector(".collapsible-content");
                renderContent(contentArea, text);
            }
        };

        setModelContent("openaiResponse", bookmark.responses["OpenAI"]);
        setModelContent("mistralResponse", bookmark.responses["Mistral"]);
        setModelContent("claudeResponse", bookmark.responses["Anthropic"]);
        setModelContent("geminiResponse", bookmark.responses["Gemini"]);
        setModelContent("deepseekResponse", bookmark.responses["DeepSeek"]);
        setModelContent("grokResponse", bookmark.responses["Grok"]);

        // --- Konsens Boxen ---
        // Strukturierten Zustand (Verdict, Karten, Badges) eines früheren Laufs
        // zuerst zurücksetzen, damit das geladene Bookmark nicht dessen Reste zeigt.
        window.resetConsensusInsights?.();
        const consensusDiv = document.getElementById("consensusResponse");
        if (window.resetCredibilityFrame) {
            window.resetCredibilityFrame(consensusDiv?.querySelector(".consensus-differences"));
        }

        const consensusText = bookmark.responses["consensus"] || "";

        // WICHTIG: Die Konsens-Antwort zuerst rendern – die Claim-Badges
        // (Modell-Zustimmung) verankern sich am Text der Hauptantwort.
        const conMain = document.querySelector("#consensusResponse .consensus-main p");
        renderContent(conMain, consensusText);

        // --- Differences Box ---
        const conDiff = document.querySelector("#consensusResponse .consensus-differences p");

        // Strukturierte Ansicht (Verdict, Modellvergleich-Karten, Claim-Badges)
        // exakt wie nach einer echten Query rendern – sofern das Bookmark die
        // strukturierten differences_data enthält. Sonst Freitext-Fallback.
        const differencesData = bookmark.responses["differences_data"];
        const includedCount = ["OpenAI", "Mistral", "Anthropic", "Gemini", "DeepSeek", "Grok"]
            .filter(name => (bookmark.responses[name] || "").trim()).length;

        let structuredRendered = false;
        if (window.renderConsensusInsights && differencesData && typeof differencesData === "object") {
            structuredRendered = window.renderConsensusInsights(differencesData, includedCount);
        }

        // Resolve-Persistenz: Payload setzen, damit eine Resolve-Runde aus dem
        // geladenen Bookmark heraus ihr Ergebnis in dasselbe Bookmark schreibt.
        window.lastConsensusBookmarkPayload = (consensusText.trim() && bookmark.query) ? {
            question: bookmark.query,
            consensusText: consensusText,
            differencesText: bookmark.responses["differences"] || "",
            differencesData: (differencesData && typeof differencesData === "object") ? differencesData : null
        } : null;

        if (!structuredRendered) {
            // Freitext-Fallback (ältere Bookmarks ohne differences_data),
            // inkl. optionaler Credibility-Badges (Farben).
            let diffText = bookmark.responses["differences"] || "";
            if (window.applyCredibilityFrame) {
                window.applyCredibilityFrame(conDiff, diffText);
            }
            if (window.colorizeCredibility) {
                diffText = window.colorizeCredibility(diffText);
            }
            renderContent(conDiff, diffText);
        }

        // Konsens-Bereich genau wie nach einer echten Anfrage einblenden – aber nur,
        // wenn das Bookmark tatsächlich einen Konsens enthält. So erscheint der
        // (rahmenlose) Bereich sichtbar und funktional (Copy, Quellen-Links).
        if (consensusText.trim()) {
            window.revealConsensusOutput?.();
        } else {
            window.hideConsensusOutput?.();
        }

        // Toggles setzen (Deep Think) - wie gehabt
        if (bookmark.mode) {
            const deepToggle = document.getElementById("deepSearchToggle");

            // Erstmal resetten
            if (deepToggle && deepToggle.checked) deepToggle.click();

            // Dann korrekt setzen
            if (bookmark.mode === "Deep Think") {
                if (deepToggle && !deepToggle.checked) deepToggle.click();
            }
        }
        
        // Frage ins Eingabefeld setzen (optional, aber hilfreich)
        const questionInput = document.getElementById("questionInput");
        if (questionInput && bookmark.query) {
            questionInput.value = bookmark.query;
            questionInput.dispatchEvent(new Event("input", { bubbles: true }));
            window.syncDemoChipState?.();
            // Falls du eine globale Variable für die letzte Frage hast:
            if (typeof lastQuestion !== 'undefined') lastQuestion = bookmark.query;
        }

        // Anhänge des Bookmarks als Vorschau-Chips anzeigen (nur Metadaten,
        // die Dateien selbst sind nicht gespeichert und werden nicht mitgesendet)
        if (typeof window.showBookmarkAttachments === "function") {
            window.showBookmarkAttachments(bookmark.attachments || []);
        }
    }
    // === NEU: Citation-Meta immer nach dem Rendern setzen ===
    try {
        let includedModels = [];

        if (typeof window.getIncludedModelNamesForCitation === "function") {
            // liest aus dem DOM (nur Boxen mit Inhalt & nicht "excluded")
            includedModels = window.getIncludedModelNamesForCitation();
        }

        window.consensusCitationMeta = {
            question: bookmark.query || "",
            includedModels: includedModels,
            // falls du irgendwann das echte Consensus-Modell speicherst, hier ersetzen
            consensusModel: "GPT-5",
            url: window.location.href.split("#")[0],
            dateISO:
                bookmark.created_at ||
                bookmark.createdAt ||
                bookmark.created_at_iso ||
                new Date().toISOString()
        };
    } catch (err) {
        console.warn("Could not rebuild consensusCitationMeta from bookmark:", err);
        window.consensusCitationMeta = null;
    }
}

async function loadBookmarks() {
  if (!auth.currentUser) return;
  const id_token = await auth.currentUser.getIdToken(false);

  try {
    const res = await fetch("/bookmarks", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + id_token
      }
    });
    const data = await res.json();

    // Container leeren (entfernt auch die Skeleton-Platzhalter) – vor den
    // Early-Returns, damit der Skeleton auch bei einem Fehler verschwindet.
    const container = document.getElementById("bookmarksContainer");
    if (container) container.innerHTML = "";

    if (!res.ok) return;

    // Global speichern
    window.bookmarksData = data.bookmarks;

    [...data.bookmarks].reverse().forEach(bm => addBookmarkToDOM(bm));

  } catch (error) {
    console.error("Error in loadBookmarks:", error);
    // Skeleton auch im Fehlerfall (z. B. Netzwerkabbruch) entfernen.
    const container = document.getElementById("bookmarksContainer");
    if (container) container.innerHTML = "";
  }
}

window.loadBookmarks = loadBookmarks;

async function deleteBookmark(bookmarkId) {
  if (!auth.currentUser) return;
  const id_token = await auth.currentUser.getIdToken(false);
  if (!id_token) return;

  try {
    const res = await fetch("/bookmark", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token, bookmarkId })
    });
    const data = await res.json();

    if (!res.ok) {
      console.error("Error deleting bookmark:", data.detail);
      return;
    }

    // Lokales Array und DOM aktualisieren
    window.bookmarksData = window.bookmarksData.filter(b => b.id !== bookmarkId);
    const el = document.querySelector(`.bookmark[data-id="${bookmarkId}"]`);
    if (el) el.remove();
    trackAppEvent("app_bookmark_deleted");

  } catch (error) {
    console.error("Error in deleteBookmark:", error);
  }
}
window.deleteBookmark = deleteBookmark;

// Login per Enter-Taste auslösen: Bei Fokus im Email- oder Passwortfeld wird der Login-Button "geklickt".
document.getElementById("loginEmail").addEventListener("keydown", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    document.getElementById("loginButton").click();
  }
});

document.getElementById("loginPassword").addEventListener("keydown", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    document.getElementById("loginButton").click();
  }
});

function sendFeedback(message, email) {
  // Prüfe, ob der Nutzer eingeloggt ist
  if (!auth.currentUser) {
    console.error("sendFeedback: Kein aktueller Nutzer vorhanden.");
    return Promise.reject(new Error("Bitte logge dich ein, um Feedback zu senden."));
  }
  
  // Hole den aktuellen, gültigen ID-Token ohne forceRefresh
  return auth.currentUser.getIdToken()
    .then(idToken => {
      // Sende das Feedback an deinen Backend-Endpoint
      return fetch("/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message, email, id_token: idToken })
      });
    })
    .then(response => response.json())
    .catch(error => {
      console.error("sendFeedback: Fehler beim Senden des Feedbacks:", error);
      throw error;
    });
}

// Exponiere die Funktion, damit sie von index.html aus aufgerufen werden kann
window.sendFeedback = sendFeedback;

function addBookmarkToDOM(bookmark) {
  const container = document.getElementById("bookmarksContainer");
  
  // Prüfen, ob das Bookmark schon existiert (Update-Fall), um Duplikate zu vermeiden
  const existing = document.querySelector(`.bookmark[data-id="${bookmark.id}"]`);
  if (existing) {
      // Optional: Update Text falls er sich geändert hat
      return; 
  }

  const div = document.createElement("div");
  div.className = "bookmark";
  div.dataset.id = bookmark.id;
  div.style.position = "relative";
  div.innerHTML = `
    <p>${truncateText(bookmark.query)}</p>
    <span class="delete-bookmark" role="button" tabindex="0" aria-label="Delete bookmark" title="Delete bookmark">x</span>
  `;

  // Delete-Event
  const deleteControl = div.querySelector(".delete-bookmark");
  deleteControl.addEventListener("click", e => { 
       e.stopPropagation(); 
       deleteBookmark(bookmark.id); 
  });
  deleteControl.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      deleteBookmark(bookmark.id);
    }
  });

  // Click-Event -> Ruft jetzt die ausgelagerte Funktion auf
  div.addEventListener("click", () => {
    // Wir holen uns die aktuellsten Daten aus dem globalen Array (falls vorhanden), 
    // oder nehmen das übergebene Objekt.
    const currentData = window.bookmarksData.find(b => b.id === bookmark.id) || bookmark;
    loadSingleBookmarkUI(currentData);
    trackAppEvent("app_bookmark_opened");
  });

  // WICHTIG: prepend statt appendChild, damit es oben erscheint
  container.prepend(div);

  // Animation
  div.classList.add("fade-in");
  setTimeout(() => div.classList.remove("fade-in"), 500);
}
