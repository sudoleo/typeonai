import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, query, orderBy, onSnapshot, doc, setDoc, increment } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithCustomToken, signOut, onAuthStateChanged, sendPasswordResetEmail } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

// Initialisiere Firebase mit der globalen Konfiguration, die aus dem HTML kommt
const app = initializeApp(window.FIREBASE_CONFIG);
const db = getFirestore(app);

// Initialisiere Auth
const auth = getAuth(app);
window.auth = auth;

onAuthStateChanged(auth, (user) => {
  const loginContainer = document.getElementById("loginContainer");
  const usageOptions = document.getElementById("usageOptions");
  if (user) {
    user.getIdToken().then((token) => {
      localStorage.setItem("id_token", token);
      // Zeige den Bereich, sobald der Token gesetzt ist
      if (usageOptions) {
        usageOptions.style.display = "block";
      }
    });
    loginContainer.innerText = `Logout (${user.email})`;
  } else {
    localStorage.removeItem("id_token");
    loginContainer.innerText = "Einloggen und gratis nutzen";
    if (usageOptions) {
      usageOptions.style.display = "none";
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
          console.log("Benutzer erfolgreich angemeldet:", userCredential.user);
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
    alert("Bitte geben Sie Ihre E-Mail-Adresse ein, um das Passwort zurückzusetzen.");
    return;
  }
  sendPasswordResetEmail(auth, email)
    .then(() => {
      alert("Eine E-Mail zum Zurücksetzen Ihres Passworts wurde an " + email + " gesendet.");
    })
    .catch((error) => {
      console.error("Fehler beim Senden der Passwort-Zurücksetzen-E-Mail:", error);
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
  let html = '<table style="width:100%; table-layout: fixed;">';
  html += '<thead><tr><th style="width:40%;">Modell</th><th style="width:60%;">Votes</th></tr></thead>';
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
    console.log(`Vote für ${model} (${type}) erfolgreich registriert.`);
  } catch (error) {
    console.error("Fehler beim Aktualisieren des Leaderboard-Eintrags:", error);
  }
}

window.recordModelVote = recordModelVote;

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