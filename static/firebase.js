import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, query, orderBy, onSnapshot, doc, setDoc, getDoc, increment, addDoc, deleteDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithCustomToken, signOut, onAuthStateChanged, sendPasswordResetEmail, sendEmailVerification } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

// Initialisiere Firebase mit der globalen Konfiguration, die aus dem HTML kommt
const app = initializeApp(window.FIREBASE_CONFIG);
const db = getFirestore(app);

// Initialisiere Auth
const auth = getAuth(app);
window.auth = auth;

let unsubscribeBookmarks = null;

onAuthStateChanged(auth, (user) => {
  const loginContainer = document.getElementById("loginContainer");
  const usageOptions = document.getElementById("usageOptions");
  if (user) {
    user.getIdToken().then((token) => {
      localStorage.setItem("id_token", token);
      if (usageOptions) {
        usageOptions.style.display = "block";
      }
    });
    loginContainer.innerText = `Logout (${user.email})`;

    // Starte die Bookmarks-Subscription, wenn sie noch nicht läuft
    if (!unsubscribeBookmarks) {
      unsubscribeBookmarks = loadBookmarks();
    }
  } else {
    localStorage.removeItem("id_token");
    loginContainer.innerText = "Log in and use for free";
    if (usageOptions) {
      usageOptions.style.display = "none";
    }
    // Bookmarks leeren
    document.getElementById("bookmarksContainer").innerHTML = "";
    // Falls onSnapshot abonniert war, abbestellen:
    if (unsubscribeBookmarks) {
      unsubscribeBookmarks();
      unsubscribeBookmarks = null;
    }
  }
});

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
          window.location.href = "/";
        });
      } else {
        alert("Please verify your e-mail address first. Check your inbox for the confirmation link.");
        signOut(auth);
      }
    })
    .catch((error) => {
      document.getElementById("loginError").innerText = error.message;
    });
});

document.getElementById("registerButton").addEventListener("click", () => {
  const email = document.getElementById("loginEmail").value;
  const password = document.getElementById("loginPassword").value;
  
  fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email, password: password })
  })
  .then(response => response.json())
  .then(data => {
    if (data.customToken) {
      // Nutzer mit dem Custom Token anmelden
      signInWithCustomToken(auth, data.customToken)
        .then((userCredential) => {
          // Sende nach erfolgreichem Login den Verifizierungs-Link
          sendEmailVerification(auth.currentUser)
            .then(() => {
              alert("Registration successful! Please confirm your e-mail address by clicking on the link in the e-mail.");
              // Optional: Nach dem Versenden der Verifizierungs-Mail den Nutzer abmelden
              signOut(auth);
            })
            .catch((error) => {
              document.getElementById("registerError").innerText = "Error sending the verification e-mail: " + error.message;
            });
        })
        .catch((error) => {
          document.getElementById("registerError").innerText = error.message;
        });
    } else if (data.detail) {
      document.getElementById("registerError").innerText = data.detail;
    }
  })
  .catch(error => {
    document.getElementById("registerError").innerText = error.message;
  });
});

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

// Restlicher Firebase-Code (z.B. Leaderboard, Funktionen, etc.)
const leaderboardRef = collection(db, "leaderboard");
const leaderboardQuery = query(leaderboardRef, orderBy("best", "desc"));
onSnapshot(leaderboardQuery, (snapshot) => {
  let html = '<table style="width:100%; table-layout: fixed; margin-left: -5px;">';
  html += '<thead><tr><th style="width:40%;">Model</th><th style="width:60%;">Votes</th></tr></thead>';
  html += '<tbody>';
  snapshot.forEach((doc) => {
    const data = doc.data();
    const bestVotes = data.best || 0;
    const excludeVotes = data.exclude || 0;
    const bestModelVotes = data.BestModel || 0;
    html += `<tr>
               <td>${doc.id}</td>
               <td>
                 <span title="Best">&#10003; ${bestVotes}</span>&nbsp;&nbsp;
                 <span title="Exclude">&#10005; ${excludeVotes}</span>&nbsp;&nbsp;
                 <span title="BestModel">best ${bestModelVotes}</span>
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
  
  const id_token = localStorage.getItem("id_token");
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

async function saveBookmark(question, response, modelName) {
  if (!auth.currentUser) return;
  const id_token = localStorage.getItem("id_token");
  if (!id_token) return;
  try {
    const res = await fetch("/bookmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
          id_token: id_token,
          question: question,
          response: response,
          modelName: modelName
      })
    });
    const data = await res.json();
    if (!res.ok) {
      console.error("Error saving bookmark:", data.detail);
    } else {
      console.log(data.message);
      // Nach erfolgreichem Speichern das Bookmark-Listing aktualisieren:
      loadBookmarks();
    }
  } catch (error) {
      console.error("Error in saveBookmark:", error);
  }
}
window.saveBookmark = saveBookmark;

async function saveBookmarkConsensus(question, consensusText, differencesText) {
  if (!auth.currentUser) return;
  const id_token = localStorage.getItem("id_token");
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
  
  const id_token = localStorage.getItem("id_token");
  if (!id_token) return;
  
  try {
    // Abruf der Bookmarks über den Backend-Endpoint
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
          document.getElementById("openaiResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["OpenAI"] || "");
          document.getElementById("mistralResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Mistral"] || "");
          document.getElementById("claudeResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Anthropic"] || "");
          document.getElementById("geminiResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Gemini"] || "");
          document.getElementById("deepseekResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["DeepSeek"] || "");
          document.getElementById("grokResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Grok"] || "");
          document.getElementById("exaResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Exa"] || "");
          document.getElementById("perplexityResponse").querySelector(".collapsible-content").innerHTML =
            marked.parse(bookmark.responses["Perplexity"] || "");
          
          // Aktualisiere auch die Konsens-Boxen, falls vorhanden
          document.getElementById("consensusResponse").querySelector(".consensus-main p").innerHTML =
            marked.parse(bookmark.responses["consensus"] || "");
          document.getElementById("consensusResponse").querySelector(".consensus-differences p").innerHTML =
            marked.parse(bookmark.responses["differences"] || "");
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
  const id_token = localStorage.getItem("id_token");
  if (!id_token) return;
  try {
    const res = await fetch("/bookmark", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
         id_token: id_token,
         bookmarkId: bookmarkId
      })
    });
    const data = await res.json();
    if (!res.ok) {
       console.error("Error deleting bookmark:", data.detail);
    } else {
       console.log(data.message);
       loadBookmarks();
    }
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
