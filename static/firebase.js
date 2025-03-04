import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getFirestore, collection, query, orderBy, onSnapshot, doc, setDoc, increment } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

// Initialisiere Firebase mit der globalen Konfiguration, die aus dem HTML kommt
const app = initializeApp(window.FIREBASE_CONFIG);
const db = getFirestore(app);

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
