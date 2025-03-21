import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, query, orderBy, onSnapshot, doc, setDoc, getDoc, increment, addDoc, deleteDoc } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithCustomToken, signOut, onAuthStateChanged, sendPasswordResetEmail } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

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
      return userCredential.user.getIdToken();
    })
    .then((token) => {
      localStorage.setItem("id_token", token);
      window.location.href = "/"; // Seite neu laden, damit der Token verfügbar ist
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
      // Mit dem Custom Token den Nutzer automatisch einloggen
      signInWithCustomToken(auth, data.customToken)
        .then((userCredential) => {
          document.getElementById("loginModal").style.display = "none";
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
    alert("Please enter your e-mail address to reset the password.");
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
  const modelRef = doc(db, "leaderboard", model);
  try {
    await setDoc(modelRef, { [type]: increment(1) }, { merge: true });
  } catch (error) {
    console.error("Error when updating the leaderboard entry:", error);
  }
}

// Hilfsfunktion zum Kürzen des Textes auf maximal 5 Wörter
function truncateText(text, maxWords = 5) {
  const words = text.split(' ');
  if (words.length > maxWords) {
    return words.slice(0, maxWords).join(' ') + '...';
  }
  return text;
}

window.recordModelVote = recordModelVote;

function saveBookmark(question, response, modelName) {
  if (!auth.currentUser) return;
  const userUid = auth.currentUser.uid;
  const docId = btoa(question).replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50);
  const docRef = doc(db, "users", userUid, "bookmarks", docId);

  // Statt "responses.OpenAI" als Feldname ein verschachteltes Objekt:
  const dataToMerge = {
    query: question,
    timestamp: new Date(),
    responses: {
      [modelName]: response
    }
  };

  setDoc(docRef, dataToMerge, { merge: true })
    .then(() => console.log(`Bookmark updated: ${modelName}`))
}

window.saveBookmark = saveBookmark;

function saveBookmarkConsensus(question, consensusText, differencesText) {
  if (!auth.currentUser) return;
  const userUid = auth.currentUser.uid;
  const docId = btoa(question).replace(/[^a-zA-Z0-9]/g, "_").slice(0, 50);
  const docRef = doc(db, "users", userUid, "bookmarks", docId);

  // Schreibe die Konsens-Daten in dasselbe "responses"-Objekt
  const dataToMerge = {
    responses: {
      consensus: consensusText,
      differences: differencesText
    }
  };

  // Merge: alte Felder bleiben erhalten
  setDoc(docRef, dataToMerge, { merge: true })
    .then(() => console.log("Consensus & differences saved."))
    .catch(error => console.error("Error when saving the consensus:", error));
}

window.saveBookmarkConsensus = saveBookmarkConsensus;

function loadBookmarks() {
  if (!auth.currentUser) return;
  const userUid = auth.currentUser.uid;
  const bookmarksCollection = collection(db, "users", userUid, "bookmarks");
  const q = query(bookmarksCollection, orderBy("timestamp", "desc"));
  return onSnapshot(q, (snapshot) => {
    let bookmarksHTML = "";
    snapshot.forEach((doc) => {
      const data = doc.data();
      bookmarksHTML += `<div class="bookmark" data-id="${doc.id}" style="position: relative;">
                        <p>${truncateText(data.query)}</p>
                        <span class="delete-bookmark" style="position: absolute; right: 5px; top: 50%; transform: translateY(-50%); cursor: pointer;">x</span>
                        </div>`;
    });
    document.getElementById("bookmarksContainer").innerHTML = bookmarksHTML;

    // Löschen-Event hinzufügen
    document.querySelectorAll(".delete-bookmark").forEach(btn => {
      btn.addEventListener("click", (e) => {
        // Verhindert, dass das Klick-Event auch den Klick auf das Bookmark auslöst
        e.stopPropagation();
        const bookmarkId = btn.parentElement.getAttribute("data-id");
        deleteBookmark(bookmarkId);
      });
    });

    // Entferne die automatische Deaktivierung hier:
    // document.getElementById("consensusButton").disabled = true;

    // Klick-Event hinzufügen: Beim Klick auf ein Bookmark wird der Konsens-Button deaktiviert und das Bookmark geladen
    document.querySelectorAll(".bookmark").forEach(item => {
      item.addEventListener("click", () => {
        // Konsens-Button deaktivieren, weil hier ein Bookmark geladen wird
        document.getElementById("consensusButton").disabled = true;

        const bookmarkId = item.getAttribute("data-id");
        console.log("Lade Bookmark-Dokument mit ID: ", bookmarkId);
        getDoc(doc(db, "users", userUid, "bookmarks", bookmarkId))
          .then(docSnapshot => {
            if (docSnapshot.exists()) {
              const bookmarkData = docSnapshot.data();
              console.log("Bookmark-Daten:", bookmarkData);
              if (bookmarkData.responses) {
                document.getElementById("openaiResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["OpenAI"] || "");
                document.getElementById("mistralResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["Mistral"] || "");
                document.getElementById("claudeResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["Anthropic"] || "");
                document.getElementById("geminiResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["Gemini"] || "");
                document.getElementById("deepseekResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["DeepSeek"] || "");
                document.getElementById("grokResponse")
                  .querySelector(".collapsible-content").innerHTML =
                  marked.parse(bookmarkData.responses["Grok"] || "");
                // Zusätzlich Konsens & Unterschiede laden:
                document.getElementById("consensusResponse")
                  .querySelector(".consensus-main p").innerHTML =
                  marked.parse(bookmarkData.responses["consensus"] || "");
                document.getElementById("consensusResponse")
                  .querySelector(".consensus-differences p").innerHTML =
                  marked.parse(bookmarkData.responses["differences"] || "");
              } else {
                console.log("No 'responses' found in the bookmark.");
              }
            } else {
              console.log("No bookmark document found.");
            }
          })
          .catch(error => console.error("Error loading the bookmark:", error));
      });
    });
  });
}

function deleteBookmark(bookmarkId) {
  if (!auth.currentUser) return;
  const userUid = auth.currentUser.uid;
  const bookmarkDocRef = doc(db, "users", userUid, "bookmarks", bookmarkId);
  deleteDoc(bookmarkDocRef)
    .then(() => {
      console.log("Bookmark deleted:", bookmarkId);
    })
    .catch((error) => {
      console.error("Error when deleting the bookmark:", error);
    });
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