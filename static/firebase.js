import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, doc, setDoc, getDoc, increment, addDoc, deleteDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
import { getAuth, signInWithEmailAndPassword, signOut, onAuthStateChanged, sendPasswordResetEmail, sendEmailVerification, onIdTokenChanged } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";
import {
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  setPersistence,
  browserLocalPersistence,
  browserSessionPersistence,
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

// Loaded as part of the content-addressed head group. Keeping this dependency
// in bundles.json means source mode cannot serve a stale nested module import.
const {
  showEmailVerificationGate,
  hideEmailVerificationGate,
} = window.App.emailVerification;

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

// Der Bestaetigungslink soll nicht in einer Firebase-Sackgasse enden, sondern
// zurueck in die App fuehren — dort steht die getippte Frage noch im Feld.
function verificationEmailSettings() {
  return {
    url: `${window.location.origin}/app?verified=1`,
    handleCodeInApp: false,
  };
}

function sendVerificationMail(user) {
  return sendEmailVerification(user, verificationEmailSettings());
}

// Firebase haelt emailVerified aus dem zuletzt geholten Token. Wer den Link
// gerade in einem anderen Tab geklickt hat, ist also serverseitig laengst
// bestaetigt, waehrend das lokale Objekt noch "false" sagt. reload() holt den
// echten Stand; das erzwungene Token loest onIdTokenChanged neu aus und die
// App faehrt ohne Neuladen in den verifizierten Zustand.
async function refreshVerificationState() {
  const current = auth.currentUser;
  if (!current) return false;
  await current.reload();
  if (!auth.currentUser?.emailVerified) return false;
  await auth.currentUser.getIdToken(true);
  return true;
}

// Logout-Bestätigung als In-App-Modal (#logoutConfirmModal in index.html)
// statt Browser-window.confirm. Buttons werden einmalig beim Modul-Load
// verdrahtet; der Logout-Button im Account-Popup ruft nur openLogoutConfirm().
function closeLogoutConfirm() {
  const modal = document.getElementById("logoutConfirmModal");
  if (modal) modal.style.display = "none";
}

async function performLogout() {
  if (performLogout.inProgress) return;
  performLogout.inProgress = true;
  trackAppEvent("auth_logout_click");
  closeLogoutConfirm();
  try {
    // Clear the HttpOnly server session before dropping the Firebase identity.
    // If this request fails, keeping Firebase signed in avoids presenting a
    // false "logged out" state while the cookie could still authorize calls.
    const response = await fetch("/auth/session", {
      method: "DELETE",
      credentials: "same-origin",
      cache: "no-store"
    });
    if (!response.ok) throw new Error(`Session cleanup failed (${response.status})`);

    // Abort while the Firebase identity is still available. Cancel hooks can
    // then release prepared-but-unconsumed usage reservations with the run's
    // bound token; clearAll after signOut removes the remaining projections.
    window.App?.runRegistry?.cancelAll?.("logout");
    await signOut(auth);
    setActiveAuthIdentity(null);
    resetLoadedRunAfterLogout();
    clearAuthenticatedUiState();
    clearLocalProviderKeys();
  } catch (error) {
    console.error("Logout failed:", error);
    window.App?.showPopup?.("Logout could not be completed. You are still signed in. Please try again.");
  } finally {
    performLogout.inProgress = false;
  }
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

function resetLoadedRunAfterLogout() {
  // A response belongs to the authenticated session that loaded it. Abort
  // active streams first so a late SSE event cannot render it again after the
  // DOM and share/bookmark context have been cleared.
  window.App?.runRegistry?.clearAll?.("logout");
  window.clearResponseBoxes?.({ silent: true });
  window.clearPreparedBookmarkShareResult?.();
  window.App.state.set("currentEvidenceSources", [], "evidence");
  window.App.state.set("consensusCitationMeta", null, "consensus");
  window.App?.chatSession?.reset?.();
  window.App?.bookmarkSession?.reset?.();
  window.App?.sharedModal?.close?.();
  window.App?.watch?.resetAfterLogout?.();
  document.body.classList.add("is-hero");
  window.syncHeroResponseAccess?.();
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
  const text = md || "";
  if (typeof window.marked?.parse !== "function" || typeof window.DOMPurify?.sanitize !== "function") {
    const node = document.createElement("div");
    node.textContent = text;
    window.App?.reportCriticalError?.({
      type: "dependency_unavailable",
      phase: "markdown_render",
      message: "Markdown renderer unavailable; displaying unformatted text.",
      details: "bookmark markdown"
    });
    return node.innerHTML;
  }
  const html = window.marked.parse(text);
  return window.DOMPurify.sanitize(html, {
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
    NORMAL: getConfiguredLimit("free_consensus_run_limit", 0),
    DEEP: getConfiguredLimit("free_deep_think_run_limit", 0)
  },
  PRO: {
    NORMAL: getConfiguredLimit("pro_consensus_run_limit", 0),
    DEEP: getConfiguredLimit("pro_deep_think_run_limit", 0)
  }
};

// Globale Variablen für den aktuellen Zustand (Startwert: Free)
window.App.state.set("currentMaxLimit", window.LIMITS.FREE.NORMAL, "userTier");
window.App.state.set("currentDeepLimit", window.LIMITS.FREE.DEEP, "userTier");

// merken, dass wir Bookmarks schon einmal geladen haben
let bookmarksLoaded = false;
let bookmarksNextCursor = null;
let bookmarksLoading = false;
let bookmarksLoadRequestId = 0;
let openedBookmarkId = null;
const bookmarkDetailCache = new Map();
const authState = window.App.authState;
let bookmarkViewEpoch = 0;
let accountMenuDocumentClickHandler = null;

// A pending bookmark fetch owns only the view epoch it started in. Selecting
// a live run or returning to the empty composer invalidates that fetch so its
// late response cannot replace the newer projection.
window.addEventListener("consensio:run-registry-change", event => {
  if (["created", "visible", "visible-cleared", "saved-view", "cleared"].includes(event.detail?.type)) {
    bookmarkViewEpoch += 1;
    // Share/Watch rehydration is view-owned just like bookmark detail loading.
    // A late result from saved view A must not replace the result id projected
    // by live run B.
    clearPreparedBookmarkShareResult();
  }
});

function publishAuthState(uid) {
  authState.publish(uid);
}

function clearAccountMenuDocumentListener() {
  if (!accountMenuDocumentClickHandler) return;
  document.removeEventListener("click", accountMenuDocumentClickHandler);
  accountMenuDocumentClickHandler = null;
}

function setActiveAuthIdentity(uid) {
  return authState.setIdentity(uid);
}

function isCurrentAuthenticatedUser(uid, generation) {
  return authState.isCurrent(uid, generation, auth.currentUser?.uid);
}

function setBookmarksAccess(isLoggedIn) {
  const section = document.querySelector(".bookmarks-section");
  const toggle = document.getElementById("bookmarksToggle");
  const searchHead = document.querySelector(".sidebar-bookmarks-head");
  const searchTrigger = document.getElementById("bookmarkSearchTrigger");
  const search = document.getElementById("chatSearch");
  section?.classList.toggle("is-locked", !isLoggedIn);
  if (toggle) {
    toggle.disabled = !isLoggedIn;
    toggle.setAttribute("aria-disabled", String(!isLoggedIn));
    toggle.title = isLoggedIn ? "Open or close bookmarks" : "Log in to use bookmarks";
  }
  if (searchTrigger) {
    searchTrigger.disabled = !isLoggedIn;
    searchTrigger.setAttribute("aria-disabled", String(!isLoggedIn));
  }
  if (search) {
    search.disabled = !isLoggedIn;
    if (!isLoggedIn) search.value = "";
  }
  if (!isLoggedIn) {
    searchHead?.classList.remove("is-searching");
    searchTrigger?.setAttribute("aria-expanded", "false");
  }
}

const LEGACY_PROVIDER_KEY_STORAGE = [
    "openaiKey",
    "mistralKey",
    "anthropicKey",
    "geminiKey",
    "deepseekKey",
    "grokKey",
];

function clearLegacyProviderKeys() {
  LEGACY_PROVIDER_KEY_STORAGE.forEach(key => localStorage.removeItem(key));
}

function clearLocalProviderKeys() {
  ["openrouterKey", ...LEGACY_PROVIDER_KEY_STORAGE]
    .forEach(key => {
      localStorage.removeItem(key);
      const input = document.getElementById(key);
      if (input) input.value = "";
    });
}

// Einmalige Migration für bereits eingeloggte Nutzer, die seit der
// OpenRouter-Umstellung noch keinen Logout ausgeführt haben.
clearLegacyProviderKeys();

function clearAuthenticatedUiState() {
  bookmarkViewEpoch += 1;
  window.App?.runRegistry?.clearAll?.("auth_reset");
  clearAccountMenuDocumentListener();
  localStorage.removeItem("id_token");
  window.App?.usageRun?.clear?.();
  window.App.state.set("isUserPro", false, "userTier");
  window.App.state.set("currentMaxLimit", window.LIMITS.FREE.NORMAL, "userTier");
  window.App.state.set("currentDeepLimit", window.LIMITS.FREE.DEEP, "userTier");

  ["freeUsageDisplay", "deepUsageDisplay", "watchUsageDisplay", "countdownDisplay"]
    .forEach(id => {
      const node = document.getElementById(id);
      if (node) node.textContent = "";
    });
  window.App?.sidebarQuota?.setOpen?.(false);
  window.App?.sidebarQuota?.sync?.();

  const container = document.getElementById("bookmarksContainer");
  if (container) container.innerHTML = "";
  bookmarksLoaded = false;
  bookmarksNextCursor = null;
  bookmarksLoading = false;
  bookmarksLoadRequestId += 1;
  openedBookmarkId = null;
  bookmarkDetailCache.clear();
  window.bookmarksData = [];
  setBookmarksAccess(false);
}

async function checkUserStatusOnLoad(user, token, generation) {
  if (!user || !token) return;

  try {
    const response = await fetch("/user_status", {
      method: "GET",
      headers: {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
      }
    });
    if (!isCurrentAuthenticatedUser(user.uid, generation)) return;

    if (response.ok) {
      const data = await response.json();
      if (!isCurrentAuthenticatedUser(user.uid, generation)) return;

      // 1. Globale Limits sofort aktualisieren.
      window.App.state.set("currentMaxLimit", data.limit, "userTier");
      window.App.state.set("currentDeepLimit", data.deep_limit, "userTier");
      window.App.state.set("isUserPro", data.is_pro, "userTier");

      // 2. UI AKTUALISIEREN

      // A) Der saubere Weg (falls vorhanden):
      if (typeof window.updateUserTierUI === "function") {
          window.updateUserTierUI(data.is_pro, true);
      }
      if (typeof window.setCurrentUsageLimits === "function") {
          window.setCurrentUsageLimits(data.is_pro, data);
      } else {
          window.App.state.set("currentMaxLimit", data.limit, "userTier");
          window.App.state.set("currentDeepLimit", data.deep_limit, "userTier");
      }

      // B) FALLBACK (Hier war der Fehler):
      const badge = document.getElementById("proBadge");
      const upgradeLink = document.getElementById("upgradeLink");
      const premiumOptions = document.querySelectorAll('.premium-option');
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
  const previousAuthUid = authState.uid;
  const generation = setActiveAuthIdentity(user?.emailVerified ? user.uid : null);

  if (user) {
    // Unbestaetigt: kein Token persistieren, keine authentifizierten Calls —
    // aber auch KEIN Rauswurf mehr. Der Nutzer bleibt angemeldet, sieht die
    // App und den Bestaetigungs-Streifen. Ausgeloggt zu werden, waehrend man
    // im Postfach sucht, war der teuerste Moment im ganzen Funnel.
    if (!user.emailVerified) {
      // Kann sein, dass der Link gerade geklickt wurde und nur das lokale
      // Objekt veraltet ist. Ergibt der frische Stand "bestaetigt", laeuft
      // dieser Handler gleich noch einmal — dann durch den verifizierten Ast.
      let becameVerified = false;
      try {
        becameVerified = await refreshVerificationState();
      } catch (_) {
        // Offline oder gedrosselt: dann eben der Streifen.
      }
      if (becameVerified) return;

      try { localStorage.removeItem("id_token"); } catch (_) {}
      resetLoadedRunAfterLogout();
      clearAuthenticatedUiState();
      if (previousAuthUid) clearLocalProviderKeys();
      fetch("/auth/session", {
        method: "DELETE",
        credentials: "same-origin",
        cache: "no-store"
      }).catch(() => {});
      showEmailVerificationGate({
        email: user.email || "",
        onResend: () => {
          trackAppEvent("auth_verification_resend");
          return sendVerificationMail(auth.currentUser || user);
        },
        onRecheck: async () => {
          const verified = await refreshVerificationState();
          trackAppEvent("auth_verification_recheck", { verified });
          return verified;
        },
        onSignOut: () => {
          trackAppEvent("auth_verification_gate_sign_out");
          signOut(auth).catch(error => console.error("Sign-out from verification gate failed:", error));
        },
      });
      trackAppEvent("auth_verification_gate_shown");
      if (typeof window.updateQuestionInputAccess === "function") {
        window.updateQuestionInputAccess();
      }
      // resetLoadedRunAfterLogout() hat den Composer eben geleert; die Frage,
      // auf die dieser Nutzer wartet, gehoert wieder hinein.
      window.App?.restoreQuestionDraft?.();
      if (usageOptions) usageOptions.style.display = "none";
      setBookmarksAccess(false);
      if (loginContainer) {
        loginContainer.innerHTML = "";
        loginContainer.hidden = true;
      }
      // Die Gast-Buttons oben rechts waeren hier tote Knoepfe (openAuthModal
      // steigt bei bestehender Session aus). Der Ausweg steht im Streifen.
      const unverifiedTopActions = document.getElementById("authTopActions");
      if (unverifiedTopActions) unverifiedTopActions.hidden = true;
      publishAuthState(null);
      return; // <--- ganz wichtig
    }

    hideEmailVerificationGate();

    if (previousAuthUid && previousAuthUid !== user.uid) {
      resetLoadedRunAfterLogout();
      clearAuthenticatedUiState();
      clearLocalProviderKeys();
    }

    // ab hier nur noch verifizierte Nutzer
    const token = await user.getIdToken(/* forceRefresh= */ false);
    if (!isCurrentAuthenticatedUser(user.uid, generation)) return;
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

    await checkUserStatusOnLoad(user, token, generation);
    if (!isCurrentAuthenticatedUser(user.uid, generation)) return;

    // 2) Usage laden
    fetchUsageData(token, user.uid, generation);

    // 3) Bookmarks einmal pro Login laden
    if (!bookmarksLoaded) {
      bookmarksLoaded = await loadBookmarks();
    }
    if (!isCurrentAuthenticatedUser(user.uid, generation)) return;
    setBookmarksAccess(true);
    window.App?.watch?.refreshQuota?.().catch(() => {});

    // 4) Usage-UI anzeigen
    if (usageOptions) {
      usageOptions.style.display = "block";
    }

    // Account-Reiter in den Settings einblenden. Seit die Einstellungen
    // Tabs sind, gehoert die Sichtbarkeit des PANELS allein dem Tab-Controller
    // (app-ui.js); hier wird nur noch der Reiter selbst freigegeben. Ein
    // inline gesetztes display wuerde den Controller sonst uebersteuern.
    window.App?.settingsTabs?.setTabAvailable?.("accountSettingsSection", true);

    // View-Switch-Pill ist immer sichtbar (auch ausgeloggt) — hier nichts tun.
    // Auth-Buttons oben rechts sind nur für Gäste.
    const authTopActions = document.getElementById("authTopActions");
    if (authTopActions) authTopActions.hidden = true;
    if (loginContainer) loginContainer.hidden = false;

    // Account-Label neben dem Avatar in der Sidebar-Fußzeile
    const accountLabel = document.getElementById("accountLabel");
    if (accountLabel) {
      const accountName = (user.displayName || user.email.split("@")[0]).trim();
      accountLabel.dataset.accountName = accountName;
      accountLabel.textContent = `${accountName} · ${window.isUserPro ? "Pro" : "Free"}`;
      accountLabel.title = user.email;
      accountLabel.hidden = false;
    }

    // 5) E‑Mail & Logout als Popup (öffnet aus der Sidebar-Fußzeile nach oben)
    const emailInitial = user.email.charAt(0).toUpperCase();
    clearAccountMenuDocumentListener();
    loginContainer.innerHTML = `
      <div class="email-container">
        <span id="emailIcon" class="email-icon" role="button" tabindex="0" aria-haspopup="menu" aria-expanded="false" aria-label="Open account menu">${emailInitial}</span>
        <div id="emailPopup" class="email-popup" role="menu" hidden>
          <div class="popup-content">
            <span class="user-email">${user.email}</span>
            <a id="sharedLinksButton" class="top-bar-about" role="menuitem">Shared links</a>
            <a id="watchedLinksButton" class="top-bar-about" role="menuitem">Watched</a>
            <a id="logoutButton" class="top-bar-about" role="menuitem">Logout</a>
          </div>
        </div>
      </div>
    `;
    const emailIcon = document.getElementById("emailIcon");
    const emailPopup = document.getElementById("emailPopup");
    const logoutButton = document.getElementById("logoutButton");
    const sharedLinksButton = document.getElementById("sharedLinksButton");
    const watchedLinksButton = document.getElementById("watchedLinksButton");

    function setAccountMenuOpen(isOpen) {
      if ((!emailPopup.hidden) === isOpen) return;
      emailPopup.hidden = !isOpen;
      emailIcon.setAttribute("aria-expanded", String(isOpen));
      trackAppEvent("app_account_menu_toggled", { open: isOpen });
    }

    emailIcon.addEventListener("click", e => {
      e.stopPropagation();
      setAccountMenuOpen(emailPopup.hidden);
    });
    emailIcon.addEventListener("keydown", e => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      e.stopPropagation();
      setAccountMenuOpen(emailPopup.hidden);
    });

    // Übersicht der geteilten Consensus-Links direkt aus dem User-Menü öffnen.
    // stopPropagation ist wichtig: der umschließende #loginContainer hat einen
    // Klick-Handler, der eingeloggte Nutzer sonst ausloggt.
    if (sharedLinksButton) {
      sharedLinksButton.addEventListener("click", e => {
        e.stopPropagation();
        setAccountMenuOpen(false);
        if (typeof window.openShareDialog === "function") {
          window.openShareDialog("list");
        }
      });
    }

    if (watchedLinksButton) {
      watchedLinksButton.addEventListener("click", e => {
        e.stopPropagation();
        setAccountMenuOpen(false);
        if (typeof window.openWatchDialog === "function") {
          window.openWatchDialog("list");
        }
      });
    }

    logoutButton.addEventListener("click", e => {
      // stopPropagation: Klick soll weder das Icon-Toggle noch den
      // loginContainer-Handler treffen.
      e.stopPropagation();
      setAccountMenuOpen(false);
      openLogoutConfirm();
    });

    accountMenuDocumentClickHandler = e => {
      if (!loginContainer.contains(e.target)) {
        setAccountMenuOpen(false);
      }
    };
    document.addEventListener("click", accountMenuDocumentClickHandler);
    publishAuthState(user.uid);

    } else {
        // Cleanup bei Logout
        hideEmailVerificationGate();
        resetLoadedRunAfterLogout();
        clearAuthenticatedUiState();
        if (previousAuthUid) clearLocalProviderKeys();
        fetch("/auth/session", {
          method: "DELETE",
          credentials: "same-origin",
          cache: "no-store"
        }).catch(() => {});
        if (typeof window.updateQuestionInputAccess === "function") {
          window.updateQuestionInputAccess();
        }
        loginContainer.innerHTML = "";
        loginContainer.hidden = true;

        // Gäste: Login/Sign-up-Buttons oben rechts einblenden (View-Switch
        // bleibt sichtbar; Watches bittet beim Klick um Login).
        const authTopActionsOff = document.getElementById("authTopActions");
        if (authTopActionsOff) authTopActionsOff.hidden = false;

        const accountLabelOff = document.getElementById("accountLabel");
        if (accountLabelOff) {
          accountLabelOff.textContent = "";
          delete accountLabelOff.dataset.accountName;
          accountLabelOff.removeAttribute("title");
          accountLabelOff.hidden = true;
        }

        const upgradeLinkOff = document.getElementById("upgradeLink");
        if (upgradeLinkOff) upgradeLinkOff.style.display = "none";

        if (usageOptions) usageOptions.style.display = "none";

        window.App?.settingsTabs?.setTabAvailable?.("accountSettingsSection", false);

        // A) Badge verstecken (Direkter Zugriff)
        const badge = document.getElementById("proBadge");
        if (badge) badge.style.display = "none";

        // B) Limits auf Free zurücksetzen
        window.App.state.set("currentMaxLimit", window.LIMITS.FREE.NORMAL, "userTier");
        window.App.state.set("currentDeepLimit", window.LIMITS.FREE.DEEP, "userTier");

        // C) Premium Modelle wieder sperren (HIER WAR DER FEHLER)
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

        if (typeof window.updateUserTierUI === "function") {
            window.updateUserTierUI(false, false); // isPro=false, isLoggedIn=false
        }
        publishAuthState(null);
      }
    });

async function fetchUsageData(token, uid, generation) {
  // DOM-Elemente innerhalb der Funktion abrufen:
  const freeDisplay = document.getElementById("freeUsageDisplay");
  const deepDisplay = document.getElementById("deepUsageDisplay");
  
  // Sicherstellen, dass die Elemente vorhanden sind
  if (!freeDisplay || !deepDisplay) {
    console.error("Benötigte DOM-Elemente nicht gefunden.");
    return;
  }
  
  try {
    const response = await fetch("/usage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: token })
    });
    if (!isCurrentAuthenticatedUser(uid, generation)) return false;
    const data = await response.json();
    if (!response.ok || !isCurrentAuthenticatedUser(uid, generation)) return false;

    // /usage is an authoritative tier snapshot too. This lets a Pro account
    // recover in the same session when the earlier /user_status request had a
    // transient failure.
    const isPro = data.is_pro === true;
    window.App.state.set("isUserPro", isPro, "userTier");
    if (typeof window.updateUserTierUI === "function") {
      window.updateUserTierUI(isPro, true);
    }
    if (typeof window.setCurrentUsageLimits === "function") {
      window.setCurrentUsageLimits(isPro, data);
    } else {
      const totalLimit = Number(data.total_limit);
      const deepTotalLimit = Number(data.deep_total_limit);
      if (Number.isFinite(totalLimit)) window.App.state.set("currentMaxLimit", totalLimit, "userTier");
      if (Number.isFinite(deepTotalLimit)) window.App.state.set("currentDeepLimit", deepTotalLimit, "userTier");
    }
    if (typeof window.App?.renderUsageDisplay === "function") {
      const usageView = window.App.runRegistry?.reconcileUsageSnapshot?.({
        uid,
        generation,
        user: auth.currentUser
      }, data, { authoritative: true }) || {
        remaining: data.remaining,
        deepRemaining: data.deep_remaining,
        totalLimit: window.currentMaxLimit,
        deepLimit: window.currentDeepLimit
      };
      window.App.renderUsageDisplay(usageView);
    } else {
      // Modul- und defer-Skripte koennen bei kaltem Cache unterschiedlich
      // schnell eintreffen. Der Fallback bewahrt denselben DOM-Vertrag.
      freeDisplay.innerHTML = 'Runs: <strong>' + data.remaining + ' / ' + window.currentMaxLimit + '</strong>';
      deepDisplay.innerHTML = 'Deep Think: <strong>' + data.deep_remaining + ' / ' + window.currentDeepLimit + '</strong>';
    }
    return true;
  } catch (err) {
    if (isCurrentAuthenticatedUser(uid, generation)) {
      console.error("Error when retrieving the quota:", err);
    }
    return false;
  }
}

window.refreshUsageData = async function () {
  const requestUser = auth.currentUser;
  if (!requestUser) return false;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  try {
    const token = await requestUser.getIdToken(false);
    if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return false;
    return await fetchUsageData(token, requestUid, requestGeneration);
  } catch (error) {
    if (isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
      console.error("Could not refresh usage after UTC reset:", error);
    }
    return false;
  }
};

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
        // Unbestaetigt heisst nicht mehr "raus": das Modal geht zu, die App
        // ist da, und der Streifen (onIdTokenChanged) erklaert den einen
        // fehlenden Schritt inklusive Resend.
        try { localStorage.removeItem("id_token"); } catch {}
        closeAuthModal();
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
const registrationSuccessEl = document.getElementById("registrationSuccess");
const registrationSuccessEmailEl = document.getElementById("registrationSuccessEmail");
const registrationLoginBtn = document.getElementById("registrationLoginButton");

function setRegisterPending(isPending) {
  confirmRegisterBtn.disabled = isPending;
  confirmRegisterBtn.setAttribute("aria-busy", String(isPending));
  confirmRegisterBtn.textContent = isPending ? "Sending setup link…" : "Send setup link";
}

function setMode(mode) {
  formEl.dataset.mode = mode;
  formEl.hidden = false;
  registrationSuccessEl.hidden = true;
  setRegisterPending(false);

  const isRegister = mode === "register";
  // Titel & Hinweise
  titleEl.textContent = isRegister ? "Create account" : "Log in to consens.io";
  singleMailNoteEl.classList.toggle("u-display-none", !isRegister);

  // Felder ein-/ausblenden
  emailConfirmEl.classList.toggle("u-display-none", !isRegister);
  passEl.classList.toggle("u-display-none", isRegister);
  passConfirmEl.classList.add("u-display-none");

  // Primär-Buttons
  confirmRegisterBtn.classList.toggle("u-display-none", !isRegister);
  loginBtn.classList.toggle("u-display-none", isRegister);

  // Forgot Password ausblenden im Register-Modus
  forgotBtn?.classList.toggle("u-display-none", isRegister);

  // Toggle-Text
  toggleRegisterBtn.textContent = isRegister
    ? "Back to login"
    : "New here? Create account";

  // Fehler leeren
  registerErr.textContent = "";
  loginErr.textContent = "";
}

function showRegistrationSuccess(email) {
  titleEl.textContent = "Check your inbox";
  registrationSuccessEmailEl.textContent = email;
  formEl.hidden = true;
  registrationSuccessEl.hidden = false;
  passEl.value = "";
  passConfirmEl.value = "";
  registrationSuccessEl.focus();
}

registrationLoginBtn.addEventListener("click", () => {
  setMode("login");
  emailEl.focus();
});

toggleRegisterBtn.addEventListener("click", () => {
  const current = formEl.dataset.mode === "register" ? "register" : "login";
  setMode(current === "login" ? "register" : "login");
  trackAppEvent("auth_mode_changed", { mode: current === "login" ? "register" : "login" });
});

// --- Registrierung (läuft NICHT über loginButton, sondern über confirmRegisterButton) ---
confirmRegisterBtn.addEventListener("click", () => {
  if (confirmRegisterBtn.disabled) return;
  registerErr.textContent = "";
  trackAppEvent("auth_register_started");

  const email = (emailEl.value || "").trim();
  const email2 = (emailConfirmEl.value || "").trim();
  // Client-Side-Validierung
  if (!email || !email2) {
    registerErr.textContent = "Please enter your e-mail twice.";
    return;
  }
  if (email !== email2) {
    registerErr.textContent = "E-mail addresses do not match.";
    return;
  }
  setRegisterPending(true);

  // Request an Backend
  fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email })
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.status === "check_inbox") {
        // Existing and new addresses follow the same mailbox-only setup path.
        // Never try caller-chosen credentials here: that would reveal whether
        // the backend had just created the account.
        showRegistrationSuccess(email);
        trackAppEvent("auth_register_result", { status: "success" });
      } else if (data.detail || data.error) {
        registerErr.textContent = data.detail || data.error;
        setRegisterPending(false);
        trackAppEvent("auth_register_result", { status: "error" });
      } else {
        registerErr.textContent = "Unexpected response from server.";
        setRegisterPending(false);
        trackAppEvent("auth_register_result", { status: "error" });
      }
    })
    .catch((error) => {
      console.error("Registration request failed:", error);
      registerErr.textContent = "We couldn't create your account. Check your connection and try again.";
      setRegisterPending(false);
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
let authModalReturnFocus = null;

function authModalFocusableElements() {
  const content = document.getElementById("loginModalContent");
  if (!content) return [];
  return Array.from(content.querySelectorAll(
    'button:not([disabled]):not([hidden]), a[href], input:not([disabled]):not([type="hidden"]), [tabindex]:not([tabindex="-1"])'
  )).filter(element => !element.hidden && element.getClientRects().length > 0);
}

function showAuthModal(mode, trigger) {
  const modal = document.getElementById("loginModal");
  if (!modal) return;
  if (mode) setMode(mode);
  if (modal.style.display !== "block") authModalReturnFocus = trigger || document.activeElement;
  modal.style.display = "block";
  const initialFocus = mode === "register"
    ? document.getElementById("loginEmail")
    : document.getElementById("loginEmail");
  (initialFocus || document.getElementById("closeLoginModal"))?.focus({ preventScroll: true });
}

function closeAuthModal() {
  const modal = document.getElementById("loginModal");
  if (!modal || modal.style.display !== "block") return;
  modal.style.display = "none";
  if (authModalReturnFocus?.isConnected && typeof authModalReturnFocus.focus === "function") {
    authModalReturnFocus.focus({ preventScroll: true });
  }
  authModalReturnFocus = null;
}

document.getElementById("loginContainer").addEventListener("click", event => {
  if (!auth.currentUser) {
    showAuthModal("login", event.currentTarget);
    trackAppEvent("auth_modal_open");
  }
});

// Auth-Buttons oben rechts (nur ausgeloggt sichtbar): öffnen das Modal direkt
// im passenden Modus — "Sign up" landet ohne Umweg im Registrierungsformular.
function openAuthModal(mode, trigger) {
  if (auth.currentUser) return;
  showAuthModal(mode, trigger);
  trackAppEvent("auth_modal_open", { mode });
}

document.getElementById("authTopLoginBtn")?.addEventListener("click", event => openAuthModal("login", event.currentTarget));
document.getElementById("authTopSignupBtn")?.addEventListener("click", event => openAuthModal("register", event.currentTarget));

// Schließen des Modals
document.getElementById("closeLoginModal").addEventListener("click", closeAuthModal);
document.getElementById("loginModal")?.addEventListener("click", event => {
  if (event.target === event.currentTarget) closeAuthModal();
});
document.addEventListener("keydown", event => {
  const modal = document.getElementById("loginModal");
  if (!modal || modal.style.display !== "block") return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeAuthModal();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = authModalFocusableElements();
  if (!focusable.length) {
    event.preventDefault();
    document.getElementById("loginModalContent")?.focus();
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
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
    if (res.status === 202 && data.cleanup_pending === true) {
      trackAppEvent("auth_account_deletion_result", { status: "cleanup_pending" });
      try { localStorage.removeItem("id_token"); } catch {}
      alert(data.message || "Your account is blocked. Remaining data cleanup is queued and will be retried automatically.");
      await signOut(auth).catch(() => {});
      window.location.href = "/?landing=1";
    } else if (res.ok && data.status === "deleted" && data.cleanup_pending !== true) {
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

async function recordModelVote(model, type, resultId = window.lastShareResultId) {
  // Prüfe, ob der Nutzer eingeloggt ist.
  if (!auth.currentUser) {
    return;
  }
  
  const id_token = await auth.currentUser?.getIdToken(/* forceRefresh= */ false);
  if (!id_token || !resultId) {
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
        vote_type: type,
        result_id: resultId
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

function bookmarkDisplayQuestion(bookmark) {
  return String(bookmark?.query || "").trim();
}

// Der Sidebar-Name ist die ERSTE Frage der Unterhaltung und bleibt danach
// stehen; `query` ist die zuletzt gestellte Frage und wandert mit jedem
// Follow-up weiter. Aeltere Bookmarks ohne gespeicherten Titel fallen auf
// die Frage zurueck.
function bookmarkDisplayTitle(bookmark) {
  return String(bookmark?.title || "").trim() || bookmarkDisplayQuestion(bookmark);
}

function bookmarkIdForQuestion(question) {
  const bytes = new TextEncoder().encode(String(question || ""));
  let binary = "";
  bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  return btoa(binary).replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50);
}

window.App = window.App || {};
window.App.bookmarkSession = {
  activeId: null,
  pending: null,

  begin(question, { followup = false } = {}) {
    if (!followup || !this.activeId) {
      this.activeId = bookmarkIdForQuestion(question) || null;
    }
    return this.activeId;
  },

  restore(bookmarkId) {
    const value = String(bookmarkId || "").trim();
    this.activeId = /^[A-Za-z0-9_]{1,100}$/.test(value) ? value : null;
    return this.activeId;
  },

  currentId() {
    return this.activeId;
  },

  // A run gets a local sidebar row immediately. It deliberately stays out of
  // bookmarksData until the server returns real metadata: the outline is a
  // promise that a bookmark is being created, not a usable saved result.
  setRunActive(isActive, question = "") {
    if (isActive) {
      if (!auth.currentUser) return;
      if (!this.activeId) this.begin(question);
      const bookmarkId = this.activeId;
      if (!bookmarkId) return;

      if (!this.pending || this.pending.id !== bookmarkId) {
        const initialMeta = window.bookmarksData?.find(item => item.id === bookmarkId) || null;
        if (this.pending?.finishTimer) clearTimeout(this.pending.finishTimer);
        this.pending = {
          id: bookmarkId,
          question: bookmarkDisplayTitle(initialMeta) || String(question || "").trim(),
          initialMeta,
          latestMeta: null,
          runActive: true,
          writes: 0,
          finishTimer: null
        };
      } else {
        this.pending.runActive = true;
        if (this.pending.finishTimer) clearTimeout(this.pending.finishTimer);
        this.pending.finishTimer = null;
      }
      ensurePendingBookmarkDOM(this.pending);
      return;
    }

    if (!this.pending) return;
    this.pending.runActive = false;
    this.schedulePendingFinish();
  },

  noteWriteStarted(bookmarkId) {
    if (this.pending?.id !== bookmarkId) return;
    this.pending.writes += 1;
    if (this.pending.finishTimer) clearTimeout(this.pending.finishTimer);
    this.pending.finishTimer = null;
  },

  noteWriteFinished(bookmarkId) {
    if (this.pending?.id !== bookmarkId) return;
    this.pending.writes = Math.max(0, this.pending.writes - 1);
    this.schedulePendingFinish();
  },

  noteSavedMeta(meta) {
    if (!meta?.id || this.pending?.id !== meta.id) return false;
    this.pending.latestMeta = meta;
    this.pending.question = bookmarkDisplayTitle(meta) || this.pending.question;
    ensurePendingBookmarkDOM(this.pending);
    return true;
  },

  // Model fan-out hands directly into auto-consensus. The zero-delay fence
  // lets that synchronous hand-off claim the same pending row before it can
  // briefly turn into a clickable bookmark between the two phases.
  schedulePendingFinish() {
    if (!this.pending || this.pending.runActive || this.pending.writes > 0) return;
    if (this.pending.finishTimer) clearTimeout(this.pending.finishTimer);
    const pendingId = this.pending.id;
    this.pending.finishTimer = setTimeout(() => {
      if (this.pending?.id === pendingId) this.finishPending();
    }, 0);
  },

  finishPending() {
    const pending = this.pending;
    if (!pending || pending.runActive || pending.writes > 0) return;
    if (pending.finishTimer) clearTimeout(pending.finishTimer);
    this.pending = null;
    const readyMeta = pending.latestMeta
      || window.bookmarksData?.find(item => item.id === pending.id)
      || pending.initialMeta;
    if (readyMeta) replacePendingBookmarkWithReady(readyMeta);
    else document.querySelector(`.bookmark.is-pending[data-id="${pending.id}"]`)?.remove();
  },

  reset() {
    const pending = this.pending;
    if (pending?.finishTimer) clearTimeout(pending.finishTimer);
    this.pending = null;
    if (pending?.initialMeta) replacePendingBookmarkWithReady(pending.initialMeta);
    else document.querySelector(`.bookmark.is-pending[data-id="${pending?.id || ""}"]`)?.remove();
    this.activeId = null;
  }
};

// Familien eines Bookmarks: dieselbe Quelle wie die App (window.App.modelPrefs,
// gespeist aus cfg.PROVIDERS). Bookmarks aelterer Laeufe kennen nur die
// Familien, die es damals gab -- fehlende Schluessel sind einfach leer.
function bookmarkModelKeys() {
  return (window.App?.modelPrefs || []).map(pref => pref.key);
}

function bookmarkModelAnswerCount(bookmark) {
  const responses = bookmark?.responses && typeof bookmark.responses === "object" ? bookmark.responses : {};
  return bookmarkModelKeys().filter(name => String(responses[name] || "").trim()).length;
}

// Ein Direktvergleich (Agent Mode aus) legt Modellantworten ab und nie einen
// Consensus — daran, nicht am heutigen Stand des Schalters, ist er beim Laden
// zu erkennen. Ohne diese Unterscheidung blieb ein solches Bookmark unter
// eingeschaltetem Agent Mode unsichtbar hinter "Compare answers" haengen und
// zeigte nur noch die Frage. Ein abgebrochener Consensus-Lauf faellt in
// dieselbe Kategorie: seine Antworten sind alles, was es zu zeigen gibt.
function isDirectComparisonBookmark(bookmark) {
  const consensus = String(bookmark?.responses?.consensus || "").trim();
  return !consensus && bookmarkModelAnswerCount(bookmark) > 0;
}

function bookmarkMeta(bookmark) {
  const responses = bookmark?.responses && typeof bookmark.responses === "object" ? bookmark.responses : {};
  const modelCount = Object.entries(responses).filter(([key, value]) =>
    !["consensus", "differences", "differences_data"].includes(key) && String(value || "").trim()
  ).length;
  return {
    id: bookmark?.id || "",
    query: bookmarkDisplayQuestion(bookmark),
    title: bookmarkDisplayTitle(bookmark),
    mode: bookmark?.mode || "",
    timestamp: bookmark?.timestamp || null,
    has_consensus: Boolean(String(responses.consensus || "").trim()),
    model_count: Number(bookmark?.model_count ?? modelCount) || 0,
    source_count: Number(bookmark?.source_count ?? bookmark?.sources?.length) || 0,
    attachment_count: Number(bookmark?.attachment_count ?? bookmark?.attachments?.length) || 0,
  };
}

function upsertBookmarkMeta(bookmark, {
  prepend = true,
  runId = null,
  writeType = "model",
  requestUid = auth.currentUser?.uid || null
} = {}) {
  const meta = bookmarkMeta(bookmark);
  if (!meta.id) return;
  if (!bookmarkWriteAllowed(requestUid, meta.id)) return;
  // Keep the local row disabled while the surrounding run (and its queued
  // persistence writes) is still in flight.
  if (runId && window.App.runRegistry?.get?.(runId)) {
    window.App.runRegistry.update(runId, context => {
      context.bookmark.latestMeta = meta;
      context.bookmark.id = meta.id;
      context.bookmark.title = bookmarkDisplayTitle(meta) || context.bookmark.title;
      if (writeType === "consensus") context.persistence.consensusWrite = true;
    }, { render: false, eventType: "persistence" });
  } else {
    window.App.bookmarkSession?.noteSavedMeta?.(meta);
  }
  if (!window.bookmarksData) window.bookmarksData = [];
  const existingIndex = window.bookmarksData.findIndex(item => item.id === meta.id);
  if (existingIndex >= 0) {
    window.bookmarksData[existingIndex] = meta;
    updateBookmarkDOM(meta);
  } else {
    if (prepend) window.bookmarksData.unshift(meta);
    else window.bookmarksData.push(meta);
    addBookmarkToDOM(meta, { prepend });
  }
  if (runId) window.App.bookmarkUi?.finalizeRun?.(window.App.runRegistry?.get?.(runId));
}

let lastBookmarkSaveNotice = { key: "", shownAt: 0 };
const bookmarkWriteChains = new Map();
const deletedBookmarkKeys = new Set();

function bookmarkMutationKey(uid, bookmarkId) {
  return `${String(uid || "")}:${String(bookmarkId || "")}`;
}

function bookmarkWriteAllowed(uid, bookmarkId) {
  return !deletedBookmarkKeys.has(bookmarkMutationKey(uid, bookmarkId));
}

function snapshotBookmarkValue(value) {
  if (value === undefined) return undefined;
  try { return structuredClone(value); } catch (_) {}
  try { return JSON.parse(JSON.stringify(value)); } catch (_) { return null; }
}

function enqueueBookmarkWrite(
  requestUid,
  requestGeneration,
  bookmarkId,
  operation,
  runId = null,
  { allowStaleAuth = false } = {}
) {
  if (runId && window.App.runRegistry?.get?.(runId)) {
    window.App.runRegistry.update(runId, context => {
      context.persistence.pendingWrites += 1;
      context.bookmark.writes += 1;
    }, { render: false, eventType: "persistence" });
  } else {
    window.App.bookmarkSession?.noteWriteStarted?.(bookmarkId);
  }
  // Generation fences decide whether an operation may start or update UI, but
  // the server resource is owned by uid+bookmarkId across login generations.
  // A rapid logout/re-login of the same account must still drain an older
  // in-flight fetch before a new write/delete to that document begins.
  const queueKey = `${requestUid}:${bookmarkId}`;
  const previous = bookmarkWriteChains.get(queueKey) || Promise.resolve();
  const current = previous
    // A failed model snapshot must not prevent the authoritative consensus
    // snapshot behind it from repairing/completing the same bookmark.
    .catch(() => undefined)
    .then(() => {
      if (!allowStaleAuth && !isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;
      return operation();
    });
  bookmarkWriteChains.set(queueKey, current);
  return current.finally(() => {
    if (bookmarkWriteChains.get(queueKey) === current) {
      bookmarkWriteChains.delete(queueKey);
    }
    if (runId && window.App.runRegistry?.get?.(runId)) {
      window.App.runRegistry.update(runId, context => {
        context.persistence.pendingWrites = Math.max(0, context.persistence.pendingWrites - 1);
        context.bookmark.writes = Math.max(0, context.bookmark.writes - 1);
      }, { render: false, eventType: "persistence" });
      window.App.bookmarkUi?.finalizeRun?.(window.App.runRegistry.get(runId));
    } else {
      window.App.bookmarkSession?.noteWriteFinished?.(bookmarkId);
    }
  });
}

function showBookmarkSaveError(status, detail, scope = "") {
  const normalizedDetail = String(detail || "").trim();
  let message = "This bookmark could not be saved.";
  if (normalizedDetail === "Bookmark limit reached.") {
    message = "Bookmark limit reached. Delete an older bookmark before saving a new one.";
  } else if (normalizedDetail === "Bookmark storage limit reached.") {
    message = "Bookmark storage is full. Delete older bookmarks before saving a new one.";
  } else if (normalizedDetail === "Bookmark is too large.") {
    message = "This result is too large to save as a bookmark.";
  } else if (status === 429) {
    message = "Bookmarks could not be saved right now. Please wait a minute and try again.";
  }

  // Ein Lauf speichert bis zu sechs Modellantworten parallel. Derselbe Fehler
  // darf deshalb genau einmal sichtbar werden, nicht als Popup-Kaskade.
  const key = `${scope}:${status}:${message}`;
  const now = Date.now();
  if (lastBookmarkSaveNotice.key === key && now - lastBookmarkSaveNotice.shownAt < 600_000) {
    return;
  }
  lastBookmarkSaveNotice = { key, shownAt: now };
  window.App?.showPopup?.(message);
}

async function saveBookmark(question, response, modelName, mode, previousQuestion = "", runOptions = null) {
  const boundRunId = String(runOptions?.runId || "").trim() || null;
  const requestUser = runOptions?.auth?.user || auth.currentUser;
  if (!requestUser) return;
  const requestUid = runOptions?.auth?.uid || requestUser.uid;
  const requestGeneration = runOptions?.auth?.generation ?? authState.generation;
  const bookmarkId = runOptions?.bookmarkId
    || window.App.bookmarkSession?.currentId?.()
    || bookmarkIdForQuestion(question);
  const waitForDeleteOutcome = !bookmarkWriteAllowed(requestUid, bookmarkId);
  // Snapshot before entering the queue. A later bookmark/view switch cannot
  // change the payload of this run while it waits behind an earlier write.
  const sources = Array.isArray(runOptions?.sources)
    ? runOptions.sources.map(source => ({ ...source }))
    : (window.currentEvidenceSources || []).map(source => ({ ...source }));
  const attachmentsMeta = Array.isArray(runOptions?.attachments)
    ? runOptions.attachments.map(item => ({ ...item }))
    : (window.lastQuestionAttachmentsMeta || []).map(item => ({ ...item }));
  return enqueueBookmarkWrite(requestUid, requestGeneration, bookmarkId, async () => {
    if (waitForDeleteOutcome && !bookmarkWriteAllowed(requestUid, bookmarkId)) return;
    const id_token = await requestUser.getIdToken(false);
    if (!id_token || !isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;

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
          bookmarkId: bookmarkId || null,
          previousQuestion,
          sources: sources,
          attachments: attachmentsMeta
        })
      });
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;
      const data = await res.json();
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;

      if (!res.ok) {
        console.error("Error saving bookmark:", data.detail);
        if (boundRunId) {
          window.App.runRegistry?.notePersistence?.(boundRunId, {
            status: "error",
            error: { type: "model", status: res.status, detail: data.detail || "Bookmark save failed" }
          });
        }
        if (!boundRunId || window.App.runRegistry?.isVisible?.(boundRunId)) {
          showBookmarkSaveError(res.status, data.detail, `${bookmarkId}:${question}`);
        }
        return;
      }

      if (data.bookmark) {
          if (!boundRunId) window.App.bookmarkSession?.restore?.(data.bookmark.id);
          upsertBookmarkMeta(data.bookmark, {
            runId: boundRunId,
            writeType: "model",
            requestUid
          });
          if (openedBookmarkId === data.bookmark.id) {
            bookmarkDetailCache.clear();
            bookmarkDetailCache.set(data.bookmark.id, data.bookmark);
          }
          trackAppEvent("app_bookmark_saved", { type: "model", mode });
      }

    } catch (error) {
      console.error("Error in saveBookmark:", error);
      if (boundRunId) {
        window.App.runRegistry?.notePersistence?.(boundRunId, {
          status: "error",
          error: { type: "model", detail: error?.message || "Bookmark save failed" }
        });
      }
    }
  }, boundRunId);
}

window.saveBookmark = saveBookmark;

async function saveBookmarkConsensus(question, consensusText, differencesText, differencesData,
                                     resultId, consensusModel, modelLabels,
                                     previousQuestion = "", previousTurn = null,
                                     conversation = null) {
  const requestUser = auth.currentUser;
  const boundRunId = String(conversation?.runId || "").trim() || null;
  const boundAuth = conversation?.auth || null;
  const boundRequestUser = boundAuth?.user || requestUser;
  if (!boundRequestUser) return;
  const requestUid = boundAuth?.uid || boundRequestUser.uid;
  const requestGeneration = boundAuth?.generation ?? authState.generation;
  const bookmarkId = conversation?.bookmarkId
    || window.App.bookmarkSession?.currentId?.()
    || bookmarkIdForQuestion(question);
  const waitForDeleteOutcome = !bookmarkWriteAllowed(requestUid, bookmarkId);
  // Only identifiers and the small legacy-required text copies travel over
  // this compatibility path. Sources, structured differences, labels and
  // provider answers are materialized from the owner-bound completed run.
  // Sending those ignored copies used to reject valid runs at source 25 even
  // though the authoritative snapshot supports 50.
  const payloadSnapshot = {
    question: String(question || ""),
    consensusText: String(consensusText || ""),
    differencesText: String(differencesText || ""),
    resultId: resultId || null,
    chatId: conversation?.chatId || null,
    turnId: conversation?.turnId || null,
    previousQuestion: String(previousQuestion || ""),
    previousTurn: snapshotBookmarkValue(previousTurn) || null
  };
  return enqueueBookmarkWrite(requestUid, requestGeneration, bookmarkId, async () => {
    if (waitForDeleteOutcome && !bookmarkWriteAllowed(requestUid, bookmarkId)) return;
    const id_token = await boundRequestUser.getIdToken(/* forceRefresh= */ false);
    if (!id_token || !isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;

    try {
      const requestBody = JSON.stringify({
        id_token: id_token,
        question: payloadSnapshot.question,
        consensusText: payloadSnapshot.consensusText,
        differencesText: payloadSnapshot.differencesText,
        resultId: payloadSnapshot.resultId,
        bookmarkId: bookmarkId || null,
        chatId: payloadSnapshot.chatId,
        turnId: payloadSnapshot.turnId,
        previousQuestion: payloadSnapshot.previousQuestion,
        previousTurn: payloadSnapshot.previousTurn
      });
      let res = null;
      let data = null;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          res = await fetch("/bookmark/consensus", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: requestBody,
            keepalive: true
          });
          data = await res.json().catch(() => ({}));
        } catch (networkError) {
          if (attempt >= 2) throw networkError;
          await new Promise(resolve => setTimeout(resolve, 250 * (4 ** attempt)));
          if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;
          continue;
        }
        const retryable = res.status === 408 || res.status === 425
          || res.status === 429 || res.status >= 500;
        if (res.ok || !retryable || attempt >= 2) break;
        const retryDelay = res.status === 429
          ? 1000 * (attempt + 1)
          : 250 * (4 ** attempt);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
        if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;
      }
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;
      if (!res.ok) {
        const errorDetail = data.detail || data.error || "Consensus bookmark save failed";
        data.detail = errorDetail;
        console.error("Error saving consensus bookmark:", errorDetail, data.details || "");
        if (boundRunId) {
          window.App.runRegistry?.notePersistence?.(boundRunId, {
            status: "error",
            error: { type: "consensus", status: res.status, detail: errorDetail }
          });
        }
        if (!boundRunId || window.App.runRegistry?.isVisible?.(boundRunId)) {
          showBookmarkSaveError(res.status, data.detail, `${bookmarkId}:${question}`);
        }
        return;
      }
      if (data.bookmark) {
        if (boundRunId && window.App.runRegistry?.get?.(boundRunId)) {
          const persistedBookmark = snapshotBookmarkValue(data.bookmark);
          window.App.runRegistry.update(boundRunId, context => {
            context.persistence.consensusBookmark = persistedBookmark;
            context.persistence.consensusVersionParts = bookmarkShareVersionParts(data.bookmark);
          }, { render: false, eventType: "persistence" });
        }
        if (!boundRunId) window.App.bookmarkSession?.restore?.(data.bookmark.id);
        upsertBookmarkMeta(data.bookmark, {
          runId: boundRunId,
          writeType: "consensus",
          requestUid
        });
        if (openedBookmarkId === data.bookmark.id) {
          bookmarkDetailCache.clear();
          bookmarkDetailCache.set(data.bookmark.id, data.bookmark);
        }
      }
      trackAppEvent("app_bookmark_saved", { type: "consensus" });
    } catch (error) {
      console.error("Error in saveBookmarkConsensus:", error);
      if (boundRunId) {
        window.App.runRegistry?.notePersistence?.(boundRunId, {
          status: "error",
          error: { type: "consensus", detail: error?.message || "Consensus bookmark save failed" }
        });
      }
    }
  }, boundRunId);
}
window.saveBookmarkConsensus = saveBookmarkConsensus;

window.acceptPersistedConsensusBookmark = function (bookmarkMeta, conversation = null) {
  const runId = String(conversation?.runId || "").trim() || null;
  const requestUid = conversation?.auth?.uid || auth.currentUser?.uid || null;
  if (!bookmarkMeta?.id || !requestUid) return false;
  upsertBookmarkMeta(bookmarkMeta, {
    runId,
    writeType: "consensus",
    requestUid
  });
  if (!runId) window.App.bookmarkSession?.restore?.(bookmarkMeta.id);
  trackAppEvent("app_bookmark_saved", {
    type: "consensus",
    source: "consensus_final"
  });
  return true;
};

let bookmarkShareResultVersion = 0;
window.currentBookmarkShareResultPromise = null;
window.currentBookmarkShareResultContext = null;

window.clearPreparedBookmarkShareResult = function () {
  bookmarkShareResultVersion += 1;
  window.currentBookmarkShareResultPromise = null;
  window.currentBookmarkShareResultContext = null;
  window.App.state.set("lastShareResultId", null, "share");
};

function prepareBookmarkShareResult(bookmark) {
  window.clearPreparedBookmarkShareResult();
  const bookmarkId = bookmark?.id;
  const consensusText = bookmark?.responses?.consensus;
  if (!auth.currentUser || !bookmarkId || !String(consensusText || "").trim()) {
    return null;
  }
  // Bind the exact snapshot on screen. The same bookmark document can advance
  // to a newer follow-up while this saved view remains visible; the server must
  // reject preparing that newer revision for an older view.
  window.currentBookmarkShareResultContext = {
    bookmarkId,
    versionParts: bookmarkShareVersionParts(bookmark)
  };
  return window.currentBookmarkShareResultContext;
}

function stableBookmarkJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableBookmarkJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${stableBookmarkJson(value[key])}`
    )).join(",")}}`;
  }
  const json = JSON.stringify(value);
  return json === undefined ? "null" : json;
}

function bookmarkShareVersionParts(bookmark) {
  const responses = bookmark?.responses && typeof bookmark.responses === "object"
    ? bookmark.responses : {};
  return [
    String(bookmark?.query || ""),
    String(responses.consensus || ""),
    String(bookmark?.chat_id || ""),
    String(bookmark?.turn_id || ""),
    String(bookmark?.share_result_id || ""),
    stableBookmarkJson(responses.differences_data ?? null)
  ];
}

async function bookmarkShareVersion(parts) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder !== "function") return "";
  const bytes = new TextEncoder().encode(JSON.stringify(Array.isArray(parts) ? parts : []));
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

async function requestBookmarkShareResult() {
  const context = window.currentBookmarkShareResultContext;
  if (!context || !auth.currentUser) return null;
  const version = bookmarkShareResultVersion;
  const requestUser = auth.currentUser;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  const bookmarkId = context.bookmarkId;
  const expectedVersion = await bookmarkShareVersion(context.versionParts);
  if (version !== bookmarkShareResultVersion) return null;
  const promise = enqueueBookmarkWrite(requestUid, requestGeneration, bookmarkId, async () => {
    try {
      if (version !== bookmarkShareResultVersion
          || !bookmarkWriteAllowed(requestUid, bookmarkId)) return null;
      const idToken = await requestUser.getIdToken(false);
      if (!idToken || !isCurrentAuthenticatedUser(requestUid, requestGeneration)) return null;
      const response = await fetch("/bookmark/consensus/share-result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: idToken, bookmarkId, expectedVersion })
      });
      const data = await response.json();
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return null;
      if (!response.ok) throw new Error(data.detail || "Could not prepare bookmark.");
      if (version === bookmarkShareResultVersion) {
        window.App.state.set("lastShareResultId", data.result_id || null, "share");
      }
      return version === bookmarkShareResultVersion ? window.lastShareResultId : null;
    } catch (error) {
      if (version === bookmarkShareResultVersion) {
        console.error("Error preparing bookmarked consensus for share/watch:", error);
      }
      return null;
    }
  });
  window.currentBookmarkShareResultPromise = promise;
  const resultId = await promise;
  if (!resultId && version === bookmarkShareResultVersion) {
    window.currentBookmarkShareResultPromise = null;
  }
  return resultId;
}

window.resolveCurrentShareResultId = async function () {
  if (window.lastShareResultId) return window.lastShareResultId;
  if (window.currentBookmarkShareResultPromise) {
    return await window.currentBookmarkShareResultPromise;
  }
  return await requestBookmarkShareResult();
};

function bookmarkModelPresentation() {
  return (window.App?.modelPrefs || []).map(pref => ({
    provider: pref.key,
    textId: pref.textId,
    citationLabel: pref.citationLabel || pref.key
  }));
}

function applyBookmarkModelPresentation(bookmark) {
  const responses = bookmark?.responses && typeof bookmark.responses === "object"
    ? bookmark.responses
    : {};
  const storedLabels = bookmark?.model_labels && typeof bookmark.model_labels === "object"
    ? bookmark.model_labels
    : {};
  const citationModels = [];

  bookmarkModelPresentation().forEach(({ provider, textId, citationLabel }) => {
    if (!String(responses[provider] || "").trim()) return;

    // model_labels is the immutable run provenance saved with the consensus.
    // Only change the response heading: selects/localStorage remain the source
    // of truth for the next run and must never be rewritten while browsing history.
    const storedLabel = typeof storedLabels[provider] === "string"
      ? storedLabels[provider].trim()
      : "";
    const visibleLabel = storedLabel || "Model not recorded";
    const labelEl = document.getElementById(textId);
    if (labelEl) {
      labelEl.textContent = visibleLabel;
      labelEl.title = storedLabel
        ? `Model used for this saved answer: ${storedLabel}`
        : "The exact model was not recorded for this older bookmark.";
    }
    citationModels.push(storedLabel ? `${citationLabel}: ${storedLabel}` : citationLabel);
  });

  return citationModels;
}

// Diese Funktion füllt die UI mit den Daten eines Bookmarks
function normalizeConversationTurn(turn) {
  if (!turn || turn.status !== "completed" || !turn.question || !turn.consensus) return null;
  return { ...turn, turn_id: turn.id || turn.turn_id || "" };
}

function bookmarkFallbackTurn(bookmark) {
  const question = bookmarkDisplayQuestion(bookmark);
  const responses = bookmark?.responses && typeof bookmark.responses === "object"
    ? bookmark.responses
    : {};
  const consensus = String(responses.consensus || "").trim();
  if (!question || !consensus) return null;

  const storedLabels = bookmark?.model_labels && typeof bookmark.model_labels === "object"
    ? bookmark.model_labels
    : {};
  const sources = Array.isArray(bookmark?.sources) ? bookmark.sources : [];
  const modelAnswers = {};
  bookmarkModelPresentation().forEach(({ provider }) => {
    const answer = String(responses[provider] || "").trim();
    if (!answer) return;
    modelAnswers[provider] = {
      provider,
      model_label: String(storedLabels[provider] || provider),
      answer,
      sources
    };
  });
  return {
    turn_id: String(bookmark?.turn_id || ""),
    status: "completed",
    question,
    attachments: Array.isArray(bookmark?.attachments) ? bookmark.attachments : [],
    mode: String(bookmark?.mode || ""),
    consensus_model: String(bookmark?.consensus_model || ""),
    consensus,
    differences: String(responses.differences || ""),
    differences_data: responses.differences_data || null,
    sources,
    model_answers: modelAnswers
  };
}

function materializeConversationBookmark(bookmark, conversationTurns) {
  const turns = (Array.isArray(conversationTurns) ? conversationTurns : [])
    .map(normalizeConversationTurn)
    .filter(Boolean)
    .sort((left, right) => Number(left.position || 0) - Number(right.position || 0));
  if (!turns.length) {
    return {
      bookmark,
      historyTurns: [],
      currentTurn: bookmarkFallbackTurn(bookmark)
    };
  }

  const currentTurn = turns[turns.length - 1];
  const responses = {
    consensus: currentTurn.consensus || "",
    differences: currentTurn.differences || "",
    differences_data: currentTurn.differences_data || null,
  };
  const modelLabels = {};
  Object.entries(currentTurn.model_answers || {}).forEach(([provider, item]) => {
    responses[provider] = typeof item === "string" ? item : String(item?.answer || "");
    const label = typeof item === "object" ? String(item?.model_label || "").trim() : "";
    if (label) modelLabels[provider] = label;
  });
  return {
    bookmark: {
      ...bookmark,
      query: currentTurn.question,
      mode: currentTurn.mode || bookmark.mode || "",
      // Die Anhaenge des Turns schlagen die des Dokuments: das Dokument kennt
      // nur die zuletzt gespeicherte Frage, der Turn kennt seine eigene. Ohne
      // diese Reihenfolge zeigte ein Chat, dessen letzte Frage ohne Datei
      // auskam, die Datei der vorigen Frage — oder gar keine. Chats von VOR
      // dieser Aenderung haben nichts am Turn stehen; fuer sie bleibt das
      // Dokument die einzige Quelle, und es beschreibt genau diese Frage.
      attachments: (Array.isArray(currentTurn.attachments) && currentTurn.attachments.length)
        ? currentTurn.attachments
        : (Array.isArray(bookmark.attachments) ? bookmark.attachments : []),
      responses,
      sources: Array.isArray(currentTurn.sources) ? currentTurn.sources : [],
      consensus_model: currentTurn.consensus_model || bookmark.consensus_model || "",
      model_labels: modelLabels,
      chat_id: bookmark.chat_id,
      turn_id: currentTurn.turn_id,
      previous_question: turns.length > 1 ? turns[turns.length - 2].question : "",
      previous_turn: turns.length > 1 ? turns[turns.length - 2] : null
    },
    historyTurns: turns.slice(0, -1),
    currentTurn
  };
}

async function loadBookmarkConversationOnce(bookmark) {
  if (!bookmark?.chat_id || !auth.currentUser) return [];
  const requestUser = auth.currentUser;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  const idToken = await requestUser.getIdToken(false);
  if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
    throw new Error("Authentication changed while loading bookmark");
  }
  const turns = [];
  const seenTurnIds = new Set();
  const seenCursors = new Set();
  let cursor = "";
  do {
    const path = "/bookmarks/" + encodeURIComponent(bookmark.id)
      + "/conversation?limit=50"
      + (cursor ? "&cursor=" + encodeURIComponent(cursor) : "");
    const response = await fetch(path, {
      headers: { "Authorization": "Bearer " + idToken }
    });
    if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
      throw new Error("Authentication changed while loading bookmark");
    }
    const data = await response.json();
    if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
      throw new Error("Authentication changed while loading bookmark");
    }
    if (!response.ok) {
      const error = new Error(data.detail || "Could not load bookmark conversation");
      error.retryable = response.status === 429 || response.status >= 500;
      throw error;
    }
    if (String(data.chat_id || "") !== String(bookmark.chat_id || "")) {
      throw new Error("Bookmark conversation did not match the saved chat");
    }
    (Array.isArray(data.turns) ? data.turns : []).forEach(turn => {
      const turnId = String(turn?.id || turn?.turn_id || "");
      if (turnId && seenTurnIds.has(turnId)) return;
      if (turnId) seenTurnIds.add(turnId);
      turns.push(turn);
    });
    const nextCursor = data.has_more ? String(data.next_cursor || "") : "";
    if (data.has_more && !nextCursor) {
      throw new Error("Bookmark conversation was incomplete");
    }
    if (nextCursor && seenCursors.has(nextCursor)) {
      throw new Error("Bookmark conversation pagination repeated a page");
    }
    if (nextCursor) seenCursors.add(nextCursor);
    cursor = nextCursor;
  } while (cursor);
  return turns;
}

async function loadBookmarkConversation(bookmark) {
  try {
    return await loadBookmarkConversationOnce(bookmark);
  } catch (error) {
    if (!error?.retryable) throw error;
    return await loadBookmarkConversationOnce(bookmark);
  }
}

function loadSingleBookmarkUI(sourceBookmark, conversationTurns = [], options = {}) {
    const materialized = materializeConversationBookmark(sourceBookmark, conversationTurns);
    const bookmark = materialized.bookmark;
    let bookmarkCitationModels = [];
    const displayQuestion = bookmarkDisplayQuestion(bookmark);
    // Ein Bookmark bringt seine eigene Ansicht mit: ein Direktvergleich laedt
    // als Direktvergleich zurueck (Composer oben, sechs Antworten darunter),
    // ein gefuehrter Lauf als Thread. Der aktuelle Agent-Mode-Schalter sagt
    // nur, was der NAECHSTE Lauf tun wird.
    const directComparison = isDirectComparisonBookmark(bookmark);
    // Ein geladenes Bookmark zeigt sofort Antworten und startet daher nie im
    // zentrierten Leerzustand.
    if (directComparison) {
        window.enterDirectComparisonView?.();
        window.App?.consensusPipeline?.dismiss?.();
    } else {
        window.exitHeroMode?.();
    }
    window.App?.followup?.reset?.();
    window.App?.chatSession?.reset?.();
    window.App?.followup?.clearHistory?.();
    window.App?.bookmarkSession?.restore?.(sourceBookmark?.id);
    if (materialized.historyTurns.length) {
        window.App?.followup?.renderStoredTurns?.(materialized.historyTurns);
    } else if (bookmark?.previous_turn) {
        window.App?.followup?.renderStoredTurn?.(bookmark.previous_turn);
    }
    const continuationTurn = materialized.currentTurn;
    const authoritativeChatRestored = continuationTurn && bookmark.chat_id
      ? window.App?.chatSession?.restoreCompletedChat?.(
          bookmark.chat_id,
          continuationTurn.turn_id
        ) === true
      : false;
    window.App?.runRegistry?.showSavedView?.(
      { type: "bookmark", bookmarkId: sourceBookmark?.id || "" },
      {
        bookmarkId: sourceBookmark?.id || "",
        chatId: authoritativeChatRestored ? bookmark.chat_id : "",
        turnId: authoritativeChatRestored ? continuationTurn?.turn_id : "",
        question: displayQuestion,
        consensus: String(bookmark?.responses?.consensus || ""),
        currentTurn: continuationTurn || null,
        historyTurns: materialized.historyTurns || [],
        continuationUnavailable: !continuationTurn,
        title: bookmarkDisplayTitle(bookmark),
        bookmarkMeta: bookmarkMeta(sourceBookmark)
      }
    );
    window.App?.setAppTitle?.(displayQuestion);
    // Never let Share/Watch target a run that was displayed previously.
    // A consensus bookmark gets a reusable server-side snapshot on Share/Watch.
    prepareBookmarkShareResult(bookmark);

    // --- NEU: Quellen wiederherstellen ---
    // 1. Globale Variable setzen, damit injectMarkdown (in index.html) darauf zugreifen kann
    window.App.state.set("currentEvidenceSources", bookmark.sources || [], "evidence");

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

        (window.App?.modelPrefs || []).forEach(pref => {
            setModelContent(pref.responseId, bookmark.responses[pref.key]);
        });

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
        const conMain = window.App.consensusBodyEl(consensusDiv);
        renderContent(conMain, consensusText);

        // --- Differences Box ---
        const conDiff = document.querySelector("#consensusResponse .consensus-differences p");

        // Strukturierte Ansicht (Verdict, Modellvergleich-Karten, Claim-Badges)
        // exakt wie nach einer echten Query rendern – sofern das Bookmark die
        // strukturierten differences_data enthält. Sonst Freitext-Fallback.
        const differencesData = bookmark.responses["differences_data"];
        const includedCount = bookmarkModelKeys()
            .filter(name => (bookmark.responses[name] || "").trim()).length;

        let structuredRendered = false;
        if (window.renderConsensusInsights && differencesData && typeof differencesData === "object") {
            structuredRendered = window.renderConsensusInsights(differencesData, includedCount);
        }

        // Resolve-Persistenz: Payload setzen, damit eine Resolve-Runde aus dem
        // geladenen Bookmark heraus ihr Ergebnis in dasselbe Bookmark schreibt.
        window.lastConsensusBookmarkPayload = (consensusText.trim() && bookmark.query) ? {
            question: bookmark.query,
            previousQuestion: bookmark.previous_question || "",
            previousTurn: bookmark.previous_turn || null,
            consensusText: consensusText,
            differencesText: bookmark.responses["differences"] || "",
            differencesData: (differencesData && typeof differencesData === "object") ? differencesData : null,
            conversation: materialized.currentTurn ? {
              bookmarkId: sourceBookmark.id,
              chatId: bookmark.chat_id,
              turnId: materialized.currentTurn.turn_id,
              modelResponses: Object.fromEntries(
                Object.entries(materialized.currentTurn.model_answers || {}).map(([provider, item]) => [
                  provider,
                  typeof item === "string" ? item : String(item?.answer || "")
                ])
              )
            } : { bookmarkId: sourceBookmark.id }
        } : null;

        if (!structuredRendered) {
            // Freitext-Fallback (ältere Bookmarks ohne differences_data),
            // inkl. optionaler Credibility-Badges (Farben). Das Panel bleibt
            // dann sichtbar aufgeklappt - der Freitext ist die einzige Analyse.
            window.App.differencesPanel?.expandForFallback?.();
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
            // Bookmark-Restore durchlaeuft keinen Query-/Consensus-Lifecycle.
            // Den fertigen Antwort-Footer daher explizit synchronisieren,
            // damit "Show model answers" und die vorhandenen Drawer immer
            // unter der wiederhergestellten Antwort stehen.
            window.updateAgentModeUI?.();
            window.App?.consensusPipeline?.renderProvenance?.();
            if (continuationTurn) {
                window.App?.followup?.offer?.(
                  displayQuestion,
                  consensusText,
                  continuationTurn
                );
            } else {
                window.App?.followup?.markContinuationUnavailable?.();
            }
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

        // Historische Modellnamen erst nach allen Mode-Synchronisierungen
        // anwenden, weil updateDeepThinkText sonst wieder die aktuelle Picker-
        // Auswahl in die Antworttitel schreiben würde.
        bookmarkCitationModels = applyBookmarkModelPresentation(bookmark);
        
        // Die Frage steht im Thread-Kopf über der Antwort; das Eingabefeld
        // unten bleibt frei für die nächste Frage. Der Direktvergleich kennt
        // keinen Thread-Kopf — dort sind die sechs Antworten das Ergebnis,
        // genau wie direkt nach dem Senden.
        if (bookmark.query) {
            window.App?.setThreadQuestion?.(directComparison ? "" : displayQuestion);
            const questionInput = document.getElementById("questionInput");
            if (questionInput) {
                questionInput.value = "";
                questionInput.dispatchEvent(new Event("input", { bubbles: true }));
                window.syncDemoChipState?.();
            }
            // Ein Zitat gehört zu der Antwort, aus der es stammt — auf dem
            // Schirm steht jetzt eine andere.
            window.App?.quote?.clear?.();
            // Falls du eine globale Variable für die letzte Frage hast:
            window.App.state.set("lastQuestion", displayQuestion, "run");
        }

        // Anhänge des Bookmarks als Vorschau-Chips anzeigen (nur Metadaten,
        // die Dateien selbst sind nicht gespeichert und werden nicht mitgesendet)
        if (typeof window.showBookmarkAttachments === "function") {
            window.showBookmarkAttachments(bookmark.attachments || []);
        }
    }
    if (!continuationTurn) {
        window.App?.followup?.markContinuationUnavailable?.();
    }
    // === NEU: Citation-Meta immer nach dem Rendern setzen ===
    try {
        let includedModels = bookmarkCitationModels;

        if (!includedModels.length && typeof window.getIncludedModelNamesForCitation === "function") {
            // liest aus dem DOM (nur Boxen mit Inhalt & nicht "excluded")
            includedModels = window.getIncludedModelNamesForCitation();
        }

        window.App.state.set("consensusCitationMeta", {
            question: displayQuestion,
            includedModels: includedModels,
            consensusModel: bookmark.consensus_model || "",
            url: window.location.href.split("#")[0],
            dateISO:
                bookmark.created_at ||
                bookmark.createdAt ||
                bookmark.created_at_iso ||
                new Date().toISOString()
        }, "consensus");
    } catch (err) {
        console.warn("Could not rebuild consensusCitationMeta from bookmark:", err);
        window.App.state.set("consensusCitationMeta", null, "consensus");
    }
    if (options.conversationLoadFailed) {
        const message = authoritativeChatRestored
          ? "Some earlier messages could not be displayed. Follow-ups still use the saved chat context."
          : "The full saved conversation could not be loaded. Only the visible answer is available as follow-up context.";
        window.App?.showPopup?.(message);
    }
}

function renderBookmarksLoadMore() {
  document.getElementById("bookmarksLoadMore")?.remove();
  if (!bookmarksNextCursor) return;
  const container = document.getElementById("bookmarksContainer");
  if (!container) return;
  const button = document.createElement("button");
  button.type = "button";
  button.id = "bookmarksLoadMore";
  button.className = "bookmarks-load-more";
  button.textContent = "Load more";
  button.addEventListener("click", () => loadBookmarks({ append: true }));
  container.appendChild(button);
}

function restoreRegistryRunRows() {
  const currentUid = auth.currentUser?.uid || null;
  (window.App.runRegistry?.list?.() || []).forEach(context => {
    if (!currentUid || context.auth?.uid !== currentUid) return;
    if (context.bookmark?.deleted) return;
    if (context.bookmark?.uiReady && context.bookmark?.latestMeta) {
      replacePendingBookmarkWithReady(context.bookmark.latestMeta, context.runId);
    } else {
      window.App.runView?.ensureRunRow?.(context);
    }
  });
}

function renderBookmarksLoadError(container) {
  if (!container) return;
  container.innerHTML = "";
  restoreRegistryRunRows();
  const state = document.createElement("div");
  state.className = "bookmarks-load-error";
  const message = document.createElement("p");
  message.textContent = "Bookmarks could not be loaded.";
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "bookmarks-load-more";
  retry.textContent = "Try again";
  retry.addEventListener("click", async () => {
    bookmarksLoaded = await loadBookmarks();
  });
  state.append(message, retry);
  container.appendChild(state);
}

async function loadBookmarks({ append = false, loadAll = false } = {}) {
  if (!auth.currentUser || bookmarksLoading) return false;
  const requestUser = auth.currentUser;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  const requestId = ++bookmarksLoadRequestId;
  bookmarksLoading = true;
  const container = document.getElementById("bookmarksContainer");
  try {
    const idToken = await requestUser.getIdToken(false);
    if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return false;
    let cursor = append ? bookmarksNextCursor : null;
    if (!append) {
      window.bookmarksData = [];
      bookmarksNextCursor = null;
      if (container) container.innerHTML = "";
      // A metadata refresh must not make an in-flight bookmark disappear.
      ensurePendingBookmarkDOM(window.App.bookmarkSession?.pending);
      restoreRegistryRunRows();
    }
    do {
      // 35 statt 30: auf einem grossen Monitor stand der "Load more"-Button
      // sonst dauerhaft unter einer Liste, die ohnehin komplett sichtbar war.
      const path = "/bookmarks?limit=35" + (cursor ? "&cursor=" + encodeURIComponent(cursor) : "");
      const response = await fetch(path, { headers: { "Authorization": "Bearer " + idToken } });
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return false;
      const data = await response.json();
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return false;
      if (!response.ok) throw new Error(data.detail || `Could not load bookmarks (${response.status})`);
      (data.bookmarks || []).forEach(item => upsertBookmarkMeta(item, { prepend: false }));
      bookmarksNextCursor = data.next_cursor || null;
      cursor = bookmarksNextCursor;
    } while (loadAll && cursor);
    renderBookmarksLoadMore();
    window.filterBookmarks?.(document.getElementById("chatSearch")?.value || "");
    return true;
  } catch (error) {
    if (isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
      console.error("Error in loadBookmarks:", error);
      if (!append || !window.bookmarksData?.length) renderBookmarksLoadError(container);
      else window.App?.showPopup?.("More bookmarks could not be loaded.");
    }
    return false;
  } finally {
    if (requestId === bookmarksLoadRequestId) bookmarksLoading = false;
  }
}

async function loadBookmarkDetail(bookmarkId) {
  if (!auth.currentUser) throw new Error("Authentication required");
  if (bookmarkDetailCache.has(bookmarkId)) return bookmarkDetailCache.get(bookmarkId);
  const requestUser = auth.currentUser;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  const idToken = await requestUser.getIdToken(false);
  if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
    throw new Error("Authentication changed while loading bookmark");
  }
  const response = await fetch("/bookmarks/" + encodeURIComponent(bookmarkId), {
    headers: { "Authorization": "Bearer " + idToken }
  });
  if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
    throw new Error("Authentication changed while loading bookmark");
  }
  const data = await response.json();
  if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) {
    throw new Error("Authentication changed while loading bookmark");
  }
  if (!response.ok) throw new Error(data.detail || "Could not load bookmark");
  bookmarkDetailCache.clear();
  bookmarkDetailCache.set(bookmarkId, data.bookmark);
  return data.bookmark;
}

window.openBookmark = async function (bookmarkId) {
  const liveContext = window.App.runRegistry?.findByBookmarkId?.(bookmarkId);
  if (liveContext) {
    window.App.runRegistry.show(liveContext.runId);
    trackAppEvent("app_bookmark_opened", { source: "run_registry" });
    return;
  }
  const requestEpoch = ++bookmarkViewEpoch;
  const requestUid = auth.currentUser?.uid || null;
  const requestGeneration = authState.generation;
  const viewIsCurrent = () => requestEpoch === bookmarkViewEpoch
    && isCurrentAuthenticatedUser(requestUid, requestGeneration);
  const row = document.querySelector(`.bookmark:not(.run-entry)[data-id="${bookmarkId}"]`);
  row?.classList.add("is-loading");
  try {
    const bookmark = await loadBookmarkDetail(bookmarkId);
    if (!viewIsCurrent()) return;
    let conversationTurns = [];
    let conversationLoadFailed = false;
    try {
      conversationTurns = await loadBookmarkConversation(bookmark);
      if (!viewIsCurrent()) return;
    } catch (conversationError) {
      if (!viewIsCurrent()) return;
      conversationLoadFailed = Boolean(bookmark?.chat_id);
      console.warn("Could not load full bookmark conversation; using saved continuation data.", conversationError);
    }
    if (!viewIsCurrent()) return;
    openedBookmarkId = bookmarkId;
    loadSingleBookmarkUI(bookmark, conversationTurns, { conversationLoadFailed });
    trackAppEvent("app_bookmark_opened");
  } catch (error) {
    if (!viewIsCurrent()) return;
    console.error("Error opening bookmark:", error);
    window.App?.showPopup?.("Could not load this bookmark.");
  } finally {
    row?.classList.remove("is-loading");
  }
};

window.loadAllBookmarkMetadata = async function () {
  if (!bookmarksNextCursor) return true;
  return loadBookmarks({ append: true, loadAll: true });
};

window.loadBookmarks = loadBookmarks;

async function deleteBookmark(bookmarkId) {
  const requestUser = auth.currentUser;
  if (!requestUser) return;
  const requestUid = requestUser.uid;
  const requestGeneration = authState.generation;
  const mutationKey = bookmarkMutationKey(requestUid, bookmarkId);
  // Fence immediately. Earlier model/consensus/share writes drain first in
  // the same queue; anything started after this click is rejected, so no late
  // callback can recreate the document after DELETE.
  deletedBookmarkKeys.add(mutationKey);
  window.App.runRegistry?.blockBookmarkMutation?.(bookmarkId);
  window.App.runRegistry?.cancelActionsForBookmark?.(bookmarkId, "bookmark_deleted");

  return enqueueBookmarkWrite(requestUid, requestGeneration, bookmarkId, async () => {
    let requestStarted = false;
    try {
      const id_token = await requestUser.getIdToken(false);
      if (!id_token) {
        deletedBookmarkKeys.delete(mutationKey);
        window.App.runRegistry?.unblockBookmarkMutation?.(bookmarkId);
        return;
      }
      requestStarted = true;
      const res = await fetch("/bookmark", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token, bookmarkId })
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        // A received 4xx proves that this request did not commit. A transport
        // failure or 5xx is ambiguous: the idempotent DELETE may already have
        // committed and only its response was lost. Keep the tombstone in that
        // case so queued/later saves cannot recreate the document.
        if (res.status >= 400 && res.status < 500) {
          deletedBookmarkKeys.delete(mutationKey);
          window.App.runRegistry?.unblockBookmarkMutation?.(bookmarkId);
        } else {
          window.App?.showPopup?.(
            "The deletion outcome is uncertain. Retry the deletion or reload before continuing this bookmark."
          );
        }
        console.error("Error deleting bookmark:", data.detail);
        return;
      }

      // The user intent and server mutation remain valid after logout, but UI
      // from an old auth generation must not repaint a newly signed-in view.
      if (!isCurrentAuthenticatedUser(requestUid, requestGeneration)) return;

      // Lokales Array und DOM aktualisieren.
      window.bookmarksData = window.bookmarksData.filter(b => b.id !== bookmarkId);
      bookmarkDetailCache.delete(bookmarkId);
      if (openedBookmarkId === bookmarkId) openedBookmarkId = null;
      (window.App.runRegistry?.list?.() || [])
        .filter(context => context.bookmark?.id === bookmarkId)
        .forEach(context => {
          if (window.App.runRegistry.isExecuting(context.runId)) {
            window.App.runRegistry.cancel(context.runId, "bookmark_deleted");
          }
          window.App.runRegistry.update(context.runId, current => {
            current.bookmark.deleted = true;
            current.bookmark.latestMeta = null;
            current.bookmark.uiReady = true;
          }, { render: false, eventType: "persistence" });
        });
      const selectedBasis = window.App.runRegistry?.getSelectedConversationBasis?.();
      if (selectedBasis?.bookmarkId === bookmarkId) {
        window.App.runRegistry?.clearVisible?.();
        window.clearResponseBoxes?.({ silent: true });
      }
      document.querySelectorAll(`.bookmark[data-id="${bookmarkId}"]`).forEach(el => el.remove());
      window.clearPreparedBookmarkShareResult?.();
      trackAppEvent("app_bookmark_deleted");
    } catch (error) {
      if (!requestStarted) {
        deletedBookmarkKeys.delete(mutationKey);
        window.App.runRegistry?.unblockBookmarkMutation?.(bookmarkId);
      } else {
        window.App?.showPopup?.(
          "The deletion outcome is uncertain. Retry the deletion or reload before continuing this bookmark."
        );
      }
      console.error("Error in deleteBookmark:", error);
    }
  }, null, { allowStaleAuth: true });
}
window.deleteBookmark = deleteBookmark;

// Passende Auth-Aktion per Enter auslösen – auch in den Bestätigungsfeldern.
[emailEl, emailConfirmEl, passEl, passConfirmEl].forEach((field) => field.addEventListener("keydown", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    if (formEl.dataset.mode === "register") {
      confirmRegisterBtn.click();
    } else {
      loginBtn.click();
    }
  }
}));

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

function updateBookmarkDOM(bookmark) {
  const row = document.querySelector(`.bookmark:not(.run-entry)[data-id="${bookmark.id}"]`);
  const label = row?.querySelector("p");
  if (label) label.textContent = truncateText(bookmarkDisplayTitle(bookmark));
}

function createReadyBookmarkRow(bookmark, runId = null) {
  const div = document.createElement("div");
  div.className = "bookmark";
  div.dataset.id = bookmark.id;
  if (runId) div.dataset.runId = runId;
  // Die Frage kommt als freier Nutzertext und darf nie als HTML interpretiert
  // werden - deshalb textContent statt Template-Interpolation in innerHTML.
  const label = document.createElement("p");
  label.textContent = truncateText(bookmarkDisplayTitle(bookmark));
  const deleteSpan = document.createElement("span");
  deleteSpan.className = "delete-bookmark";
  deleteSpan.setAttribute("role", "button");
  deleteSpan.setAttribute("tabindex", "0");
  deleteSpan.setAttribute("aria-label", "Delete bookmark");
  deleteSpan.setAttribute("title", "Delete bookmark");
  deleteSpan.textContent = "x";
  div.append(label, deleteSpan);

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
    const context = runId
      ? window.App.runRegistry?.get?.(runId)
      : window.App.runRegistry?.findByBookmarkId?.(bookmark.id);
    if (context) window.App.runRegistry.show(context.runId);
    else window.openBookmark(bookmark.id);
  });

  return div;
}

function ensurePendingBookmarkDOM(pending) {
  const container = document.getElementById("bookmarksContainer");
  if (!container || !pending?.id) return;
  let row = document.querySelector(`.bookmark[data-id="${pending.id}"]`);
  if (!row) {
    row = document.createElement("div");
    row.className = "bookmark";
    row.dataset.id = pending.id;
    container.prepend(row);
  }

  row.className = "bookmark is-pending";
  row.setAttribute("role", "status");
  row.setAttribute("aria-live", "polite");
  row.setAttribute("aria-disabled", "true");
  row.setAttribute(
    "aria-label",
    `Creating bookmark for ${pending.question || "this question"}. It will be available when the run is complete.`
  );
  row.title = "Bookmark is being created and will be available when the run is complete.";
  row.replaceChildren();

  const label = document.createElement("p");
  label.textContent = truncateText(pending.question || "New comparison");

  const spinner = document.createElement("span");
  spinner.className = "bookmark-pending-spinner";
  spinner.setAttribute("aria-hidden", "true");

  row.append(label, spinner);
}

function replacePendingBookmarkWithReady(bookmark, runId = null) {
  if (!bookmark?.id) return;
  const row = runId
    ? document.querySelector(`.bookmark.run-entry[data-run-id="${runId}"]`)
    : document.querySelector(`.bookmark[data-id="${bookmark.id}"]`);
  const readyRow = createReadyBookmarkRow(bookmark, runId);
  if (row) row.replaceWith(readyRow);
  else document.getElementById("bookmarksContainer")?.prepend(readyRow);
}

function addBookmarkToDOM(bookmark, { prepend = true } = {}) {
  const container = document.getElementById("bookmarksContainer");
  if (!container) return;

  // Prüfen, ob das Bookmark schon existiert (Update-Fall), um Duplikate zu vermeiden.
  // Ein Pending-Eintrag bleibt dabei absichtlich gesperrt.
  const existing = document.querySelector(`.bookmark[data-id="${bookmark.id}"]`);
  if (existing) return;

  const div = createReadyBookmarkRow(bookmark);

  if (prepend) container.prepend(div);
  else container.appendChild(div);

  // Animation
  div.classList.add("fade-in");
  setTimeout(() => div.classList.remove("fade-in"), 500);
}

function finalizeRegistryRunBookmark(context) {
  if (!context || context.status !== "succeeded") return false;
  if (context.bookmark?.uiReady) return true;
  if (context.persistence?.pendingWrites > 0) return false;
  const meta = context.bookmark?.latestMeta;
  if (!meta?.id) return false;
  // Agent-mode success is only a complete bookmark once the authoritative
  // consensus snapshot has returned. Direct comparisons intentionally have
  // no consensus and become ready after their model writes drain.
  if (context.config?.agentMode !== false && !context.persistence?.consensusWrite) return false;
  window.App.runRegistry?.update?.(context.runId, current => {
    current.bookmark.uiReady = true;
    current.bookmark.status = "ready";
    current.persistence.status = "saved";
  }, { render: false, eventType: "persistence" });
  replacePendingBookmarkWithReady(meta, context.runId);
  return true;
}

window.App.bookmarkUi = Object.freeze({
  finalizeRun: finalizeRegistryRunBookmark,
  replacePendingBookmarkWithReady,
  versionForParts: bookmarkShareVersion,
  versionParts: bookmarkShareVersionParts,
  invalidate(bookmarkId) {
    bookmarkDetailCache.delete(String(bookmarkId || ""));
  },
  canWrite(bookmarkId, uid = auth.currentUser?.uid) {
    return bookmarkWriteAllowed(uid, bookmarkId);
  }
});
