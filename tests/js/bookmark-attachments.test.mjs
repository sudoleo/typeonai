/**
 * Ein Anhang gehoert zu der Frage, mit der er rausgegangen ist.
 *
 * Frueher stand er nur am Bookmark-DOKUMENT, das nur eine Fassung kennt: die
 * naechste Frage ohne Datei ueberschrieb sie, und der geladene Chat behauptete
 * anschliessend, es sei nie eine Datei dabei gewesen. Seit die Anhaenge am Turn
 * haengen, muessen zwei Dinge stimmen: der Turn schlaegt das Dokument, und ein
 * Chat von VOR dieser Aenderung faellt weiterhin auf das Dokument zurueck.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { loadScripts, ROOT } from "./helpers/appWindow.mjs";

// firebase.js ist ein ES-Modul mit Firebase-Imports und laesst sich nicht als
// Ganzes laden. Die drei reinen Funktionen, um die es hier geht, haengen an
// nichts als ihren Argumenten - also genau dieses Stueck ausschneiden und mit
// seinen Abhaengigkeiten aufrufen (wie in bookmark-pending-state.test.mjs).
function conversationHarness() {
  const source = readFileSync(path.join(ROOT, "static/firebase.js"), "utf8");
  const start = source.indexOf("function normalizeConversationTurn(turn) {");
  const end = source.indexOf("async function loadBookmarkConversationOnce", start);
  const create = Function(
    "bookmarkDisplayQuestion",
    "BOOKMARK_MODEL_PRESENTATION",
    `${source.slice(start, end)}\nreturn { materializeConversationBookmark };`
  );
  return create(
    bookmark => String(bookmark?.query || "").trim(),
    [{ provider: "OpenAI" }, { provider: "Gemini" }]
  );
}

function completedTurn(overrides = {}) {
  return {
    id: overrides.id || "turn-1",
    status: "completed",
    question: "What does this chart show?",
    consensus: "It shows a decline.",
    model_answers: {},
    ...overrides
  };
}

const CHART = { name: "chart.png", mime: "image/png", size: 2048 };

const BODY = `
<div id="threadAsk" class="thread-ask" hidden>
  <div class="thread-ask-label">Question</div>
  <div class="thread-ask-text" id="threadAskText"></div>
  <div class="attachment-bar message-attachments" id="threadAskAttachments" hidden></div>
  <button type="button" id="threadAskMore" class="thread-ask-more">Show full question</button>
</div>
<div class="chat-input-container">
  <div id="attachmentBar" class="attachment-bar" hidden></div>
  <textarea id="questionInput"></textarea>
  <button id="attachTrigger"></button>
  <div id="attachMenu" hidden><button id="attachUploadOption"></button></div>
  <input id="attachFileInput" type="file">
</div>
`;

function appWindow() {
  return loadScripts(["static/js/app-core.js", "static/js/attachments.js"], {
    body: BODY,
    // app-core.js fragt beim Laden das Farbschema ab; jsdom kennt matchMedia nicht.
    before: window => {
      window.matchMedia = () => ({
        matches: false,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {}
      });
    }
  });
}

describe("attachments of a saved question", () => {
  it("keeps the current turn's files instead of the document's", () => {
    const { materializeConversationBookmark } = conversationHarness();

    const materialized = materializeConversationBookmark(
      // Das Dokument haelt nur die zuletzt gespeicherte Frage fest - hier die
      // Folgefrage, die ohne Datei auskam.
      { id: "bm", query: "And in euros?", attachments: [] },
      [
        completedTurn({ id: "turn-1", position: 1, attachments: [CHART] }),
        completedTurn({ id: "turn-2", position: 2, question: "And in euros?" })
      ]
    );

    expect(materialized.bookmark.attachments).toEqual([]);
    expect(materialized.historyTurns).toHaveLength(1);
    expect(materialized.historyTurns[0].attachments).toEqual([CHART]);
  });

  it("shows the files of the last question when that is the one with files", () => {
    const { materializeConversationBookmark } = conversationHarness();

    const materialized = materializeConversationBookmark(
      { id: "bm", query: "What does this chart show?", attachments: [] },
      [completedTurn({ position: 1, attachments: [CHART] })]
    );

    expect(materialized.bookmark.attachments).toEqual([CHART]);
  });

  it("still reads the document for chats saved before turns carried files", () => {
    const { materializeConversationBookmark } = conversationHarness();

    const materialized = materializeConversationBookmark(
      { id: "bm", query: "What does this chart show?", attachments: [CHART] },
      [completedTurn({ position: 1 })]
    );

    expect(materialized.bookmark.attachments).toEqual([CHART]);
  });

  it("renders a read-only chip for the restored question", () => {
    const { window, document } = appWindow();

    window.App.setThreadQuestion("What does this chart show?");
    window.showBookmarkAttachments([CHART]);

    const row = document.getElementById("threadAskAttachments");
    expect(row.hidden).toBe(false);
    expect(row.querySelector(".attachment-chip-name").textContent).toBe("chart.png");
    // IMG statt Vorschaubild: die Datei selbst ist nicht gespeichert.
    expect(row.querySelector(".attachment-chip-icon").textContent).toBe("IMG");
    // Kein Viewer, kein Entfernen - der Chip ist Metadaten, kein Bedienelement.
    expect(row.querySelector(".attachment-chip-remove")).toBeNull();
    expect(row.querySelector(".attachment-chip").getAttribute("role")).toBeNull();
  });

  it("gives a .csv the type the server also knows", async () => {
    const { window, document } = appWindow();
    // Anhaenge haengen seit der Plus-Stufe an isUserPlus, nicht an isUserPro.
    window.isUserPlus = true;

    const input = document.getElementById("attachFileInput");
    // Was der Browser meldet, ist je nach System verschieden. Kaeme "text/csv"
    // so bis in den gespeicherten Chat, fiele die Datei dort still heraus.
    Object.defineProperty(input, "files", {
      configurable: true,
      value: [new window.File(["a,b\n1,2"], "rows.csv", { type: "text/csv" })]
    });
    input.dispatchEvent(new window.Event("change"));
    await new Promise(resolve => window.setTimeout(resolve, 20));

    expect(window.pendingAttachments).toHaveLength(1);
    expect(window.pendingAttachments[0].mime).toBe("text/plain");
  });

  it("does not carry the files into the next question", () => {
    const { window, document } = appWindow();

    window.App.setThreadQuestion("What does this chart show?");
    window.showBookmarkAttachments([CHART]);

    // Das Eingabefeld gehoert der naechsten Frage: dort darf nichts haengen,
    // was mit der geladenen rausgegangen ist.
    expect(window.pendingAttachments).toEqual([]);
    expect(document.getElementById("attachmentBar").hidden).toBe(true);
  });
});
