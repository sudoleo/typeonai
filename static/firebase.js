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

const FREE_USAGE_LIMIT = 25;

// merken, dass wir Bookmarks schon einmal geladen haben
let bookmarksLoaded = false;

onIdTokenChanged(auth, async (user) => {
  console.log("[onIdTokenChanged] user?", !!user);

  const loginContainer = document.getElementById("loginContainer");
  const usageOptions   = document.getElementById("usageOptions");

  if (user) {
    // **NEU: Harte Client-Gate—kein Token persistieren, keine Calls, wenn unverified**
    if (!user.emailVerified) {
      localStorage.removeItem("id_token");
      if (usageOptions) usageOptions.style.display = "none";
      if (loginContainer) loginContainer.innerText = "Verify your e-mail to continue";
      return; // <--- ganz wichtig
    }

    // ab hier nur noch verifizierte Nutzer
    const token = await user.getIdToken(/* forceRefresh= */ true);
    localStorage.setItem("id_token", token);

    try {
      await fetch("/confirm-registration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: token })
      });
    } catch (err) {
      console.error("Error during confirm-registration:", err);
    }

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

    // 5) E‑Mail‑Icon & Popup setzen
    const emailInitial = user.email.charAt(0).toUpperCase();
    loginContainer.innerHTML = `
      <div class="email-container">
        <span id="emailIcon" class="email-icon">${emailInitial}</span>
        <div id="emailPopup" class="email-popup">
          <div class="popup-content">
            <span class="user-email">${user.email}</span>
            <button id="logoutButton" class="logout-button">Logout</button>
          </div>
        </div>
      </div>
    `;
    const emailIcon    = document.getElementById("emailIcon");
    const emailPopup   = document.getElementById("emailPopup");
    const logoutButton = document.getElementById("logoutButton");

    emailIcon.addEventListener("click", e => {
      e.stopPropagation();
      emailPopup.classList.toggle("active");
    });
    logoutButton.addEventListener("click", () => {
      signOut(auth).then(() => emailPopup.classList.remove("active"))
                    .catch(err => console.error("Logout-Fehler", err));
    });
    document.addEventListener("click", e => {
      if (!loginContainer.contains(e.target)) {
        emailPopup.classList.remove("active");
      }
    });

  } else {
    // Cleanup bei Logout
    localStorage.removeItem("id_token");
    loginContainer.innerText = "Log in and use for free";
    if (usageOptions) usageOptions.style.display = "none";

    document.getElementById("bookmarksContainer").innerHTML = "";
    bookmarksLoaded = false;
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
      freeDisplay.innerHTML = 'Free requests: <strong>' + data.remaining + ' / ' + FREE_USAGE_LIMIT + '</strong>';
      deepDisplay.innerHTML = 'Deep Think: <strong>' + data.deep_remaining + ' / 12</strong>';
    })
    .catch(err => console.error("Error when retrieving the quota:", err));
}

// Login-Funktion
document.getElementById("loginButton").addEventListener("click", () => {
  const email = document.getElementById("loginEmail").value;
  const password = document.getElementById("loginPassword").value;
  
  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      if (user.emailVerified) {
        // Login erfolgreich, Token speichern und Seite neu laden
        user.getIdToken().then((token) => {
          localStorage.setItem("id_token", token);

          // ❶ Bestätige Registrierung beim Backend, damit die IP gesperrt wird
          fetch("/confirm-registration", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id_token: token })
          })
          .then(res => res.json())
          .then(info => console.log("Confirm-Registration:", info))
          .catch(err => console.error("Confirm-Registration-Fehler:", err));

          window.location.href = "/";
        });
      } else {
        try { localStorage.removeItem("id_token"); } catch {}
        alert("Please verify your e-mail address first. Check your inbox for the confirmation link.");
        signOut(auth);
      }
    })
    .catch((error) => {
      document.getElementById("loginError").innerText = error.message;
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
  titleEl.textContent = isRegister ? "Create account" : "Login";
  singleMailNoteEl.style.display = isRegister ? "block" : "block"; // ggf. "none" im Login

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
});

// --- Registrierung (läuft NICHT über loginButton, sondern über confirmRegisterButton) ---
confirmRegisterBtn.addEventListener("click", () => {
  registerErr.textContent = "";

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
            // Verifizierungs-Mail senden
            sendEmailVerification(auth.currentUser)
              .then(() => {
                alert("Registration successful! Please confirm your e-mail address by clicking on the link in the e-mail.");
                // Optional: Nach dem Versenden der Verifizierungs-Mail den Nutzer abmelden
                signOut(auth);
                // Zurück in Login-Modus gehen
                setMode("login");
              })
              .catch((error) => {
                registerErr.textContent =
                  "Error sending the verification e-mail: " + error.message;
              });
          })
          .catch((error) => {
            registerErr.textContent = error.message;
          });
      } else if (data.detail) {
        registerErr.textContent = data.detail;
      } else {
        registerErr.textContent = "Unexpected response from server.";
      }
    })
    .catch((error) => {
      registerErr.textContent = error.message;
    });
});

// Standard: beim Öffnen im Login-Modus
setMode("login");

// "Passwort vergessen?"-Funktion hinzufügen
document.getElementById("forgotPasswordButton").addEventListener("click", () => {
  const email = document.getElementById("loginEmail").value;
  if (!email) {
    alert("Please enter your e-mail address to reset the password. Check your Spam Folder.");
    return;
  }
  sendPasswordResetEmail(auth, email)
    .then(() => {
      alert("An e-mail to reset your password has been sent to " + email);
    })
    .catch((error) => {
      console.error("Error sending the password reset email:", error);
      alert("Fehler: " + error.message);
    });
});

// Klick auf den Login-Bereich: Öffne das Modal, wenn nicht angemeldet, oder melde ab, wenn schon eingeloggt
document.getElementById("loginContainer").addEventListener("click", () => {
  if (auth.currentUser) {
    signOut(auth).catch((error) => {
      console.error("Fehler beim Logout:", error);
    });
  } else {
    document.getElementById("loginModal").style.display = "block";
  }
});

// Schließen des Modals
document.getElementById("closeLoginModal").addEventListener("click", () => {
  document.getElementById("loginModal").style.display = "none";
});

function isIOS() {
  return /iP(ad|hone|od)/i.test(navigator.userAgent);
}

async function handleGoogleSignIn() {
  try {
    await setPersistence(auth, browserLocalPersistence).catch(() =>
      setPersistence(auth, browserSessionPersistence)
    );

    // 1) POPUP zuerst – auch auf iOS
    try {
      const result = await signInWithPopup(auth, googleProvider);
      await afterGoogleLogin(result.user);   // dein bestehender Code
      return;
    } catch (popupErr) {
      // 2) Fallback: Redirect (z.B. wenn Popup geblockt wurde)
      if (popupErr?.code === "auth/popup-blocked" || popupErr?.code === "auth/popup-closed-by-user") {
        await signInWithRedirect(auth, googleProvider);
        return;
      }
      throw popupErr; // echte Fehler weiterreichen
    }
  } catch (err) {
    console.error("Google sign-in failed:", err);
    document.getElementById("loginError")?.textContent =
      err.message || "Google sign-in failed.";
  }
}


document.getElementById("googleLoginButton")?.addEventListener("click", handleGoogleSignIn);

window.addEventListener("load", () => {
  getRedirectResult(auth)
    .then(async (result) => {
      if (result && result.user) {
        await afterGoogleLogin(result.user);
      }
    })
    .catch(err => console.error("getRedirectResult error:", err));
});

async function afterGoogleLogin(user) {
  console.log("[afterGoogleLogin] platform=iOS?", isIOS(), "emailVerified=", user.emailVerified);

  const token = await user.getIdToken(true);
  try { localStorage.setItem("id_token", token); } catch {}

  try {
    // Wichtig: keepalive verhindert, dass der POST beim Navigieren abbricht (<=64KB Body)
    await fetch("/confirm-registration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: token }),
      keepalive: true
    });
    // optional: minimale Wartezeit, damit iOS wirklich „durch“ ist
    await new Promise(r => setTimeout(r, 150));
  } catch (e) {
    console.error("confirm-registration (google) failed:", e);
    // Trotzdem weiter – onIdTokenChanged fängt den Rest auf
  }

  // Jetzt *nach* erfolgreichem/versuchtem POST navigieren
  location.replace("/");
}

// Restlicher Firebase-Code (z.B. Leaderboard, Funktionen, etc.)
const leaderboardRef = collection(db, "leaderboard");
const leaderboardQuery = query(
  leaderboardRef,
  orderBy("BestModel", "desc")
);
onSnapshot(leaderboardQuery, (snapshot) => {
  let html = '<table><tbody>';
  snapshot.forEach((doc) => {
    const data = doc.data();
    const bestVotes      = data.best      || 0;
    const excludeVotes   = data.exclude   || 0;
    const bestModelVotes = data.BestModel || 0;

    html += `<tr>
      <td>${doc.id}</td>
      <td>
        <span class="vote BestModel" title="BestModel">★ ${bestModelVotes}</span>
        <span class="vote best"      title="Best">&#10003; ${bestVotes}</span>
        <span class="vote exclude"   title="Exclude">&#10005; ${excludeVotes}</span>
      </td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById("leaderboardContent").innerHTML = html;
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
    } else {
      console.log("Vote recorded:", data.message);
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

  try {
    const res = await fetch("/bookmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token, question, response, modelName, mode })
    });
    const data = await res.json();

    if (!res.ok) {
      console.error("Error saving bookmark:", data.detail);
      return;
    }

    console.log("Bookmark gespeichert:", data.message);
    // Neu: nach dem Speichern die Sidebar komplett neu laden
    await loadBookmarks();
    bookmarksLoaded = false;

  } catch (error) {
    console.error("Error in saveBookmark:", error);
  }
}
window.saveBookmark = saveBookmark;



async function saveBookmarkConsensus(question, consensusText, differencesText) {
  if (!auth.currentUser) return;
  const id_token = await auth.currentUser?.getIdToken(/* forceRefresh= */ false);
  if (!id_token) return;
  try {
    const res = await fetch("/bookmark/consensus", {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({
         id_token: id_token,
         question: question,
         consensusText: consensusText,
         differencesText: differencesText
       })
    });
    const data = await res.json();
    if (!res.ok) {
      console.error("Error saving consensus bookmark:", data.detail);
    } else {
      console.log(data.message);
    }
  } catch (error) {
    console.error("Error in saveBookmarkConsensus:", error);
  }
}
window.saveBookmarkConsensus = saveBookmarkConsensus;


async function loadBookmarks() {
  if (!auth.currentUser) return;

  // Erzwinge hier ein frisches Token
  const id_token = await auth.currentUser.getIdToken(/* forceRefresh= */ false);

  try {
    const res = await fetch("/bookmarks", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + id_token
      }
    });
    const data = await res.json();

    if (!res.ok) {
      console.error("Error loading bookmarks:", data.detail);
      return;
    }
    
    // Speichere die abgerufenen Bookmarks global, um später darauf zugreifen zu können
    window.bookmarksData = data.bookmarks;
    
    let bookmarksHTML = "";
    data.bookmarks.forEach(bookmark => {
      bookmarksHTML += `
        <div class="bookmark" data-id="${bookmark.id}" style="position: relative;">
          <p>${truncateText(bookmark.query)}</p>
          <span class="delete-bookmark" style="position: absolute; right: 5px; top: 50%; transform: translateY(-50%); cursor: pointer;">x</span>
        </div>`;
    });
    
    const container = document.getElementById("bookmarksContainer");
    container.innerHTML = bookmarksHTML;
    
    // Füge die fade-in Klasse hinzu, um Animation zu triggern
    container.classList.add("fade-in");
    // Entferne die Klasse nach der Animation (0.5s entspricht der Animationsdauer)
    setTimeout(() => container.classList.remove("fade-in"), 500);
    
    // Löschen-Event hinzufügen
    document.querySelectorAll(".delete-bookmark").forEach(btn => {
      btn.addEventListener("click", (e) => {
        // Verhindert, dass das Klick-Event auch das Bookmark selbst auslöst
        e.stopPropagation();
        const bookmarkId = btn.parentElement.getAttribute("data-id");
        deleteBookmark(bookmarkId);
      });
    });
    
    // Klick-Event hinzufügen: Beim Klick auf ein Bookmark werden alle Antwortboxen aktualisiert
    document.querySelectorAll(".bookmark").forEach(item => {
      item.addEventListener("click", () => {
        // Konsens-Button deaktivieren, da hier ein Bookmark geladen wird
        document.getElementById("consensusButton").disabled = true;
        
        const bookmarkId = item.getAttribute("data-id");
        console.log("Bookmark clicked with ID:", bookmarkId);
        
        // Finde das Bookmark-Objekt in den global gespeicherten Daten
        const bookmark = window.bookmarksData.find(b => b.id === bookmarkId);
        if (bookmark && bookmark.responses) {
          // Update der Antwortboxen anhand der gespeicherten Antworten
          injectHtmlSafe(
            document.getElementById("openaiResponse").querySelector(".collapsible-content"),
            bookmark.responses["OpenAI"] || ""
          );
          injectHtmlSafe(
            document.getElementById("mistralResponse").querySelector(".collapsible-content"),
            bookmark.responses["Mistral"] || ""
          );
          injectHtmlSafe(
            document.getElementById("claudeResponse").querySelector(".collapsible-content"),
            bookmark.responses["Anthropic"] || ""
          );
          injectHtmlSafe(
            document.getElementById("geminiResponse").querySelector(".collapsible-content"),
            bookmark.responses["Gemini"] || ""
          );
          injectHtmlSafe(
            document.getElementById("deepseekResponse").querySelector(".collapsible-content"),
            bookmark.responses["DeepSeek"] || ""
          );
          injectHtmlSafe(
            document.getElementById("grokResponse").querySelector(".collapsible-content"),
            bookmark.responses["Grok"] || ""
          );
          injectHtmlSafe(
            document.getElementById("exaResponse").querySelector(".collapsible-content"),
            bookmark.responses["Exa"] || ""
          );
          injectHtmlSafe(
            document.getElementById("perplexityResponse").querySelector(".collapsible-content"),
            bookmark.responses["Perplexity"] || ""
          );
          
          // Aktualisiere auch die Konsens-Boxen, falls vorhanden
          injectHtmlSafe(
            document.getElementById("consensusResponse").querySelector(".consensus-main p"),
            bookmark.responses["consensus"] || ""
          );
          injectHtmlSafe(
            document.getElementById("consensusResponse").querySelector(".consensus-differences p"),
            bookmark.responses["differences"] || ""
          );

          // Nun: UI-Modus festlegen, indem wir den entsprechenden Toggle simulieren
          if (bookmark.mode) {
            // Für "Deep Think" – Wenn deepSearchToggle noch nicht aktiv ist, klicke darauf.
            if (bookmark.mode === "Deep Think") {
              const deepToggle = document.getElementById("deepSearchToggle");
              if (!deepToggle.checked) {
                deepToggle.click();  // Löst den Click-Handler aus, der die UI anpasst
              }
              // Gleichzeitig sicherstellen, dass der Web Search Toggle deaktiviert ist:
              const searchToggle = document.getElementById("searchModeToggle");
              if (searchToggle.checked) {
                searchToggle.click();
              }
            }
            // Für "Web Search"
            else if (bookmark.mode === "Web Search") {
              const searchToggle = document.getElementById("searchModeToggle");
              if (!searchToggle.checked) {
                searchToggle.click();
              }
              // Gleichfalls den Deep Think Toggle deaktivieren, falls aktiv:
              const deepToggle = document.getElementById("deepSearchToggle");
              if (deepToggle.checked) {
                deepToggle.click();
              }
            }
            // Wenn dein Bookmark einen anderen oder einen Standardmodus hat,
            // kannst du hier auch einen Default-Zustand setzen, z. B. beide Toggles deaktiviert:
            else {
              const deepToggle = document.getElementById("deepSearchToggle");
              if (deepToggle.checked) deepToggle.click();
              const searchToggle = document.getElementById("searchModeToggle");
              if (searchToggle.checked) searchToggle.click();
            }
          }
        } else {
          console.log("No responses found in this bookmark.");
        }
      });
    });
    
  } catch (error) {
    console.error("Error in loadBookmarks:", error);
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

    console.log("Bookmark gelöscht:", data.message);
    // Lokales Array und DOM aktualisieren
    window.bookmarksData = window.bookmarksData.filter(b => b.id !== bookmarkId);
    const el = document.querySelector(`.bookmark[data-id="${bookmarkId}"]`);
    if (el) el.remove();

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
      console.log("sendFeedback: Abgerufener ID-Token:", idToken);
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
  const div = document.createElement("div");
  div.className = "bookmark";
  div.dataset.id = bookmark.id;
  div.style.position = "relative";
  div.innerHTML = `
    <p>${truncateText(bookmark.query)}</p>
    <span class="delete-bookmark" 
          style="position:absolute; right:5px; top:50%; transform:translateY(-50%); cursor:pointer;">
      x
    </span>
  `;

  // Delete‑Button
  div.querySelector(".delete-bookmark")
     .addEventListener("click", e => { 
       e.stopPropagation(); 
       deleteBookmark(bookmark.id); 
     });

  // Click: Bookmark laden (wie in loadBookmarks)
  div.addEventListener("click", () => {
    // hier kannst du deine bestehende Logik kopieren,
    // die beim Klick aus window.bookmarksData die Details
    // in die Antwort‑Boxes schreibt.
    loadSingleBookmarkUI(bookmark);
  });

  container.appendChild(div);

  // Kurze Fade‑In‑Animation
  div.classList.add("fade-in");
  setTimeout(() => div.classList.remove("fade-in"), 500);
}