import { describe, expect, it, vi } from "vitest";

import { loadScripts } from "./helpers/appWindow.mjs";

function bootRegistry() {
  const user = {
    uid: "user-a",
    getIdToken: vi.fn(async () => "token-a")
  };
  const harness = loadScripts(["static/js/run-registry.js"], {
    before(window) {
      window.App = {
        authState: {
          uid: user.uid,
          generation: 7,
          snapshot: () => ({ uid: user.uid, generation: 7 })
        }
      };
      window.auth = { currentUser: user };
    }
  });
  return { ...harness, user, registry: harness.window.App.runRegistry };
}

function spec(question, extra = {}) {
  return {
    question,
    mode: "Standard",
    config: {
      agentMode: true,
      providers: [{ provider: "OpenAI", modelId: "gpt-test" }]
    },
    bookmarkId: `bookmark-${question.toLowerCase()}`,
    ...extra
  };
}

describe("App.runRegistry", () => {
  it("freezes each start snapshot and admits at most two executing runs", () => {
    const { registry, dom } = bootRegistry();
    const mutable = spec("A");
    const runA = registry.create(mutable);
    mutable.config.providers[0].modelId = "changed-after-start";
    const runB = registry.create(spec("B"));

    expect(runA.config.providers[0].modelId).toBe("gpt-test");
    expect(Object.isFrozen(runA.config)).toBe(true);
    expect(Object.isFrozen(runA.config.providers[0])).toBe(true);
    expect(registry.activeCount()).toBe(2);
    expect(() => registry.create(spec("C"))).toThrowError(
      expect.objectContaining({ code: "parallel_limit" })
    );

    registry.setStatus(runA.runId, "succeeded");
    const runC = registry.create(spec("C"));
    expect(runC.runId).not.toBe(runB.runId);
    expect(registry.activeCount()).toBe(2);
    dom.window.close();
  });

  it("projects only the selected run while background callbacks keep updating their owner", () => {
    const { registry, dom } = bootRegistry();
    const projections = [];
    registry.setProjector(context => projections.push(context?.runId || null));
    const runA = registry.create(spec("A"));
    const runB = registry.create(spec("B"));
    const countBeforeBackgroundUpdate = projections.length;

    registry.update(runA.runId, context => {
      context.consensus.text = "late A";
    });

    expect(runA.consensus.text).toBe("late A");
    expect(registry.visible().runId).toBe(runB.runId);
    expect(projections).toHaveLength(countBeforeBackgroundUpdate);

    registry.show(runA.runId);
    expect(projections.at(-1)).toBe(runA.runId);
    dom.window.close();
  });

  it("keeps execution, visible view, and selected follow-up basis independent", () => {
    const { registry, dom } = bootRegistry();
    const runA = registry.create(spec("A"));
    const runB = registry.create(spec("B"));
    const basis = {
      bookmarkId: "saved-bookmark",
      chatId: "0123456789abcdef0123456789abcdef",
      turnId: "abcdef0123456789abcdef0123456789",
      question: "Saved question",
      consensus: "Saved answer"
    };

    registry.showSavedView({ type: "bookmark", bookmarkId: "saved-bookmark" }, basis);

    expect(registry.visible()).toBeNull();
    expect(registry.activeCount()).toBe(2);
    expect(registry.isExecuting(runA.runId)).toBe(true);
    expect(registry.isExecuting(runB.runId)).toBe(true);
    expect(registry.getSelectedConversationBasis()).toMatchObject(basis);
    dom.window.close();
  });

  it("cancels exactly the addressed run and clearAll aborts the rest", () => {
    const { registry, window, dom } = bootRegistry();
    const runA = registry.create(spec("A"));
    const runB = registry.create(spec("B"));
    const queryA = new window.AbortController();
    const providerA = new window.AbortController();
    const queryB = new window.AbortController();
    const cancelA = vi.fn();
    const cancelB = vi.fn();
    runA.controllers.query = queryA;
    runA.controllers.providers.set("OpenAI", providerA);
    runA.cancelHook = cancelA;
    runB.controllers.query = queryB;
    runB.cancelHook = cancelB;

    expect(registry.cancel(runA.runId, "user")).toBe(true);
    expect(queryA.signal.aborted).toBe(true);
    expect(providerA.signal.aborted).toBe(true);
    expect(queryB.signal.aborted).toBe(false);
    expect(cancelA).toHaveBeenCalledOnce();
    expect(cancelB).not.toHaveBeenCalled();
    expect(runA.status).toBe("canceled");
    expect(runB.status).toBe("starting");

    registry.setStatus(runA.runId, "running");
    registry.setStatus(runA.runId, "succeeded");
    expect(runA.status).toBe("canceled");

    registry.clearAll("logout");
    expect(queryB.signal.aborted).toBe(true);
    expect(cancelB).toHaveBeenCalledOnce();
    expect(registry.list()).toEqual([]);
    expect(registry.activeCount()).toBe(0);
    dom.window.close();
  });

  it("serializes follow-ups per conversation and retains an uncertain-turn fence", () => {
    const { registry, dom } = bootRegistry();
    const basis = {
      key: "chat:0123456789abcdef0123456789abcdef",
      bookmarkId: "bookmark-chat",
      chatId: "0123456789abcdef0123456789abcdef",
      turnId: "abcdef0123456789abcdef0123456789",
      question: "Previous",
      consensus: "Previous answer"
    };
    const first = registry.create(spec("Follow-up 1", { basis, followup: true }));

    expect(() => registry.create(spec("Follow-up 2", { basis, followup: true })))
      .toThrowError(expect.objectContaining({ code: "conversation_busy" }));

    first.keepConversationLock = true;
    registry.setStatus(first.runId, "failed");
    expect(() => registry.create(spec("Follow-up retry", { basis, followup: true })))
      .toThrowError(expect.objectContaining({ code: "conversation_uncertain" }));

    registry.clearAll("test-reset");
    registry.blockBookmarkMutation(basis.bookmarkId);
    expect(() => registry.create(spec("Deleting follow-up", { basis, followup: true })))
      .toThrowError(expect.objectContaining({ code: "bookmark_deleting" }));
    registry.unblockBookmarkMutation(basis.bookmarkId);
    const completed = registry.create(spec("Follow-up completed", { basis, followup: true }));
    registry.setStatus(completed.runId, "succeeded");
    expect(() => registry.create(spec("Next follow-up", { basis, followup: true }))).not.toThrow();
    dom.window.close();
  });

  it("does not let a failed follow-up shadow the last successful bookmark snapshot", () => {
    const { registry, dom } = bootRegistry();
    const bookmarkId = "bookmark-shared";
    const successful = registry.create(spec("Successful", { bookmarkId }));
    registry.setStatus(successful.runId, "succeeded");
    const failed = registry.create(spec("Failed follow-up", { bookmarkId }));
    registry.setStatus(failed.runId, "failed");

    expect(registry.findByBookmarkId(bookmarkId)?.runId).toBe(successful.runId);
    dom.window.close();
  });

  it("fences out-of-order account usage snapshots but accepts an authoritative refresh", () => {
    const { registry, dom } = bootRegistry();
    const runA = registry.create(spec("Usage A"));
    const runB = registry.create(spec("Usage B"));

    expect(registry.reconcileUsageSnapshot(runA, {
      free_usage_remaining: 4,
      deep_remaining: 2
    })).toMatchObject({ remaining: 4, deepRemaining: 2 });
    expect(registry.reconcileUsageSnapshot(runB, {
      free_usage_remaining: 2,
      deep_remaining: 1
    })).toMatchObject({ remaining: 2, deepRemaining: 1 });
    expect(registry.reconcileUsageSnapshot(runA, {
      free_usage_remaining: 3,
      deep_remaining: 2
    })).toMatchObject({ remaining: 2, deepRemaining: 1 });
    expect(registry.reconcileUsageSnapshot(runA, {
      remaining: 5,
      deep_remaining: 3
    }, { authoritative: true })).toMatchObject({ remaining: 5, deepRemaining: 3 });
    dom.window.close();
  });

  it("owns costly post-actions under the same admission and logout boundary", () => {
    const { registry, window, dom } = bootRegistry();
    const runA = registry.create(spec("Action A"));
    const runB = registry.create(spec("Action B"));
    expect(() => registry.beginAction({
      key: "resolve:one",
      ownerRunId: runA.runId,
      bookmarkId: runA.bookmark.id
    })).toThrowError(expect.objectContaining({ code: "parallel_limit" }));

    registry.setStatus(runA.runId, "succeeded");
    const controller = new window.AbortController();
    const action = registry.beginAction({
      key: "resolve:one",
      ownerRunId: runA.runId,
      bookmarkId: runA.bookmark.id,
      controller
    });
    expect(registry.activeCount()).toBe(2);
    expect(() => registry.beginAction({
      key: "resolve:one",
      ownerRunId: runA.runId,
      bookmarkId: runA.bookmark.id
    })).toThrowError(expect.objectContaining({ code: "action_busy" }));

    registry.clearAll("logout");
    expect(controller.signal.aborted).toBe(true);
    expect(registry.snapshot().actions).toEqual([]);
    expect(action.actionId).toMatch(/^action_/);
    dom.window.close();
  });
});

describe("per-run ChatSession factory", () => {
  it("does not share completed or pending conversation identity", () => {
    const { window, dom } = loadScripts(["static/js/chat-session.js"]);
    const first = window.App.createChatSession();
    const second = window.App.createChatSession();
    const chatId = "0123456789abcdef0123456789abcdef";
    const turnId = "abcdef0123456789abcdef0123456789";

    expect(first.restoreCompletedChat(chatId, turnId)).toBe(true);
    first.beginRun({
      question: "Follow-up A",
      mode: "Standard",
      deepSearch: false,
      selectedModels: ["model-a", "model-b"],
      consensusModel: "judge-a",
      isFollowup: true,
      prepareSucceeded: true,
      useOwnKeys: false,
      usageRunKey: "usage-a",
      attachments: []
    });

    expect(first.activeChatId).toBe(chatId);
    expect(first.pendingChatId).toBe(chatId);
    expect(first.logicalRun.question).toBe("Follow-up A");
    expect(second.activeChatId).toBeNull();
    expect(second.pendingChatId).toBeNull();
    expect(second.logicalRun).toBeNull();
    expect(window.App.chatSession).not.toBe(first);
    expect(window.App.chatSession).not.toBe(second);
    dom.window.close();
  });
});
