import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { loadScripts, ROOT } from "./helpers/appWindow.mjs";

function sessionHarness({ existing = [] } = {}) {
  const source = readFileSync(path.join(ROOT, "static/firebase.js"), "utf8");
  const start = source.indexOf("window.App.bookmarkSession = {");
  const end = source.indexOf("const BOOKMARK_MODEL_KEYS", start);
  const sessionSource = source.slice(start, end);
  const rendered = [];
  const ready = [];
  const removed = [];
  const fakeDocument = {
    querySelector(selector) {
      return { remove: () => removed.push(selector) };
    }
  };
  const window = { App: {}, bookmarksData: [...existing] };
  const create = Function(
    "window",
    "auth",
    "bookmarkIdForQuestion",
    "bookmarkDisplayTitle",
    "ensurePendingBookmarkDOM",
    "replacePendingBookmarkWithReady",
    "document",
    `${sessionSource}\nreturn window.App.bookmarkSession;`
  );
  const session = create(
    window,
    { currentUser: { uid: "user" } },
    () => "pending_id",
    bookmark => String(bookmark?.title || bookmark?.query || ""),
    pending => rendered.push({ ...pending }),
    meta => ready.push(meta),
    fakeDocument
  );
  return { session, window, rendered, ready, removed };
}

describe("pending bookmark session", () => {
  it("stays disabled until both the run and its persistence write finish", async () => {
    const { session, rendered, ready } = sessionHarness();
    session.begin("How does this work?");
    session.setRunActive(true, "How does this work?");
    session.noteWriteStarted("pending_id");
    session.noteSavedMeta({ id: "pending_id", title: "How does this work?" });
    session.setRunActive(false);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(session.pending).not.toBeNull();
    expect(ready).toEqual([]);

    session.noteWriteFinished("pending_id");
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(session.pending).toBeNull();
    expect(ready).toEqual([{ id: "pending_id", title: "How does this work?" }]);
    expect(rendered.length).toBeGreaterThanOrEqual(2);
  });

  it("does not flash ready between model fan-out and auto-consensus", async () => {
    const { session, ready, removed } = sessionHarness();
    session.begin("Question");
    session.setRunActive(true, "Question");
    session.setRunActive(false);
    session.setRunActive(true, "Question");

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(session.pending?.runActive).toBe(true);
    expect(ready).toEqual([]);
    expect(removed).toEqual([]);

    session.setRunActive(false);
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(session.pending).toBeNull();
    expect(removed).toHaveLength(1);
  });

  it("restores an existing conversation bookmark if a follow-up saves nothing", async () => {
    const previous = { id: "pending_id", title: "First question" };
    const { session, ready } = sessionHarness({ existing: [previous] });
    session.restore("pending_id");
    session.setRunActive(true, "Follow-up");
    session.setRunActive(false);

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(ready).toEqual([previous]);
  });
});

describe("pending bookmark markup", () => {
  it("renders an inaccessible loading row with a bookmark-frame spinner", () => {
    const source = readFileSync(path.join(ROOT, "static/firebase.js"), "utf8");
    const start = source.indexOf("function createReadyBookmarkRow(");
    const end = source.indexOf("function addBookmarkToDOM(", start);
    const helperSource = source.slice(start, end);
    const { window, document, dom } = loadScripts([], {
      body: '<div id="bookmarksContainer"></div>'
    });
    const create = Function(
      "document",
      "window",
      "truncateText",
      "bookmarkDisplayTitle",
      "deleteBookmark",
      `${helperSource}\nreturn { ensurePendingBookmarkDOM };`
    );
    const { ensurePendingBookmarkDOM } = create(
      document,
      window,
      value => value,
      bookmark => bookmark?.title || "",
      () => {}
    );

    ensurePendingBookmarkDOM({ id: "pending_id", question: "A careful question" });
    const row = document.querySelector('.bookmark[data-id="pending_id"]');
    expect(row.classList.contains("is-pending")).toBe(true);
    expect(row.getAttribute("aria-disabled")).toBe("true");
    expect(row.querySelector(".bookmark-pending-spinner")).not.toBeNull();
    expect(row.querySelector(".delete-bookmark")).toBeNull();
    expect(row.textContent.trim()).toBe("A careful question");
    expect(row.getAttribute("aria-label")).toContain("A careful question");
    dom.window.close();
  });
});
