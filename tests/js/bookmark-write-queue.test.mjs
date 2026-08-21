import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { ROOT } from "./helpers/appWindow.mjs";

function queueHarness() {
  const source = readFileSync(path.join(ROOT, "static/firebase.js"), "utf8");
  const start = source.indexOf("const bookmarkWriteChains = new Map();");
  const end = source.indexOf("function showBookmarkSaveError(", start);
  const queueSource = source.slice(start, end);
  return Function(
    "isCurrentAuthenticatedUser",
    "window",
    `${queueSource}\nreturn { enqueueBookmarkWrite };`
  )(() => true, { App: { bookmarkSession: {} } });
}

describe("bookmark write queue", () => {
  it("runs model and consensus writes for one bookmark in invocation order", async () => {
    const { enqueueBookmarkWrite } = queueHarness();
    const events = [];
    let releaseFirst;
    const firstGate = new Promise(resolve => { releaseFirst = resolve; });

    const first = enqueueBookmarkWrite("uid", 1, "bookmark", async () => {
      events.push("model:start");
      await firstGate;
      events.push("model:end");
    });
    const consensus = enqueueBookmarkWrite("uid", 1, "bookmark", () => {
      events.push("consensus");
    });

    await new Promise(resolve => setTimeout(resolve, 0));
    expect(events).toEqual(["model:start"]);
    releaseFirst();
    await Promise.all([first, consensus]);
    expect(events).toEqual(["model:start", "model:end", "consensus"]);
  });

  it("still runs the authoritative consensus write after a failed model write", async () => {
    const { enqueueBookmarkWrite } = queueHarness();
    const events = [];

    const failedModel = enqueueBookmarkWrite("uid", 1, "bookmark", () => {
      events.push("model");
      throw new Error("model snapshot failed");
    });
    const consensus = enqueueBookmarkWrite("uid", 1, "bookmark", () => {
      events.push("consensus");
    });

    await Promise.allSettled([failedModel, consensus]);
    expect(events).toEqual(["model", "consensus"]);
  });
});
