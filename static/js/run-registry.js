// Browser-session owner for independent Consensus runs.
//
// A RunContext is the authority for every asynchronous callback.  The main
// DOM, the selected bookmark and the compatibility "current" globals are only
// projections of visibleRunId and are never consulted to finish a request.
(function () {
  "use strict";

  window.App = window.App || {};

  const MAX_ACTIVE_RUNS = 2;
  const EXECUTING = new Set(["starting", "running"]);
  const TERMINAL = new Set(["succeeded", "failed", "canceled"]);
  const runs = new Map();
  const conversationLocks = new Map();
  const usageSnapshotFences = new Map();
  const actions = new Map();
  const blockedBookmarkIds = new Set();
  let visibleRunId = null;
  let selectedConversationBasis = null;
  let visibleSavedView = null;
  let projector = null;
  let sequence = 0;
  let lastStartClaimAt = 0;

  class RunRegistryError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "RunRegistryError";
      this.code = code;
    }
  }

  function randomId() {
    const uuid = globalThis.crypto?.randomUUID?.();
    if (uuid) return `run_${uuid.replace(/-/g, "")}`;
    sequence += 1;
    return `run_${Date.now().toString(36)}_${sequence.toString(36)}_${Math.random().toString(16).slice(2)}`;
  }

  function cloneValue(value) {
    if (value === undefined) return undefined;
    if (typeof globalThis.structuredClone === "function") {
      try { return globalThis.structuredClone(value); } catch (_) {}
    }
    try { return JSON.parse(JSON.stringify(value)); } catch (_) { return value; }
  }

  function deepFreeze(value, seen = new WeakSet()) {
    if (!value || typeof value !== "object" || seen.has(value)) return value;
    seen.add(value);
    Object.values(value).forEach(item => deepFreeze(item, seen));
    return Object.freeze(value);
  }

  function cleanId(value) {
    return String(value || "").trim();
  }

  function normalizeBasis(value) {
    if (!value || typeof value !== "object") return null;
    const chatId = cleanId(value.chatId || value.chat_id);
    const turnId = cleanId(value.turnId || value.turn_id);
    const bookmarkId = cleanId(value.bookmarkId || value.bookmark_id);
    const basis = {
      key: cleanId(value.key),
      chatId,
      turnId,
      bookmarkId,
      question: String(value.question || ""),
      consensus: String(value.consensus || ""),
      currentTurn: cloneValue(value.currentTurn || value.current_turn || null),
      historyTurns: cloneValue(value.historyTurns || value.history_turns || []),
      continuationUnavailable: value.continuationUnavailable === true,
      title: String(value.title || value.question || ""),
      bookmarkMeta: cloneValue(value.bookmarkMeta || value.bookmark_meta || null)
    };
    if (!basis.key) {
      if (chatId) basis.key = `chat:${chatId}`;
      else if (bookmarkId) basis.key = `bookmark:${bookmarkId}`;
    }
    return basis;
  }

  function authSnapshot() {
    const state = window.App.authState?.snapshot?.() || window.__consensioAuthState || {};
    const user = window.auth?.currentUser || null;
    return Object.freeze({
      uid: user?.uid || state.uid || null,
      generation: state.generation,
      user
    });
  }

  function isAuthCurrent(context) {
    const expected = context?.auth;
    if (!expected?.uid || !expected.user) return false;
    return window.auth?.currentUser === expected.user
      && window.auth?.currentUser?.uid === expected.uid
      && window.App.authState?.generation === expected.generation;
  }

  function currentAuthMatches(auth) {
    return Boolean(auth?.uid && auth?.user)
      && window.auth?.currentUser === auth.user
      && window.auth?.currentUser?.uid === auth.uid
      && window.App.authState?.generation === auth.generation;
  }

  function usageNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  // Run endpoints return account-wide quota snapshots. Concurrent requests can
  // finish in a different order from the server-side reservations that created
  // those snapshots. Within one authenticated UTC day, never let an older,
  // larger remaining count overwrite a newer, smaller one. Explicit /usage or
  // release responses may seed an authoritative increase.
  function reconcileUsageSnapshot(owner, snapshot = {}, { authoritative = false } = {}) {
    const auth = owner?.auth || owner;
    if (!currentAuthMatches(auth)) return null;
    const utcDate = cleanId(snapshot.utc_date)
      || new Date(owner?.startedAt || Date.now()).toISOString().slice(0, 10);
    const key = `${auth.generation}:${auth.uid}:${utcDate}`;
    const prior = usageSnapshotFences.get(key) || { remaining: null, deepRemaining: null };
    const incomingRemaining = usageNumber(snapshot.remaining ?? snapshot.free_usage_remaining);
    const incomingDeep = usageNumber(snapshot.deepRemaining ?? snapshot.deep_remaining);
    const choose = (previous, incoming) => {
      if (incoming === null) return previous;
      if (authoritative || previous === null) return incoming;
      return Math.min(previous, incoming);
    };
    const next = {
      remaining: choose(prior.remaining, incomingRemaining),
      deepRemaining: choose(prior.deepRemaining, incomingDeep)
    };
    usageSnapshotFences.set(key, next);
    return {
      remaining: next.remaining ?? (snapshot.remaining ?? snapshot.free_usage_remaining),
      deepRemaining: next.deepRemaining ?? (snapshot.deepRemaining ?? snapshot.deep_remaining),
      totalLimit: snapshot.totalLimit ?? snapshot.total_limit ?? snapshot.limit ?? window.currentMaxLimit,
      deepLimit: snapshot.deepLimit ?? snapshot.deep_total_limit ?? snapshot.deep_limit ?? window.currentDeepLimit
    };
  }

  function activeCount() {
    let count = actions.size;
    runs.forEach(context => {
      if (EXECUTING.has(context.status)) count += 1;
    });
    return count;
  }

  function emit(type, context = null) {
    const detail = {
      type,
      runId: context?.runId || null,
      visibleRunId,
      activeCount: activeCount(),
      maxActiveRuns: MAX_ACTIVE_RUNS,
      context
    };
    window.dispatchEvent(new CustomEvent("consensio:run-registry-change", { detail }));
  }

  function renderVisible() {
    if (typeof projector !== "function") return;
    projector(visibleRunId ? runs.get(visibleRunId) || null : null, {
      visibleRunId,
      selectedConversationBasis,
      savedView: visibleSavedView
    });
  }

  function releaseConversationLock(context) {
    const key = context?.conversationLockKey;
    if (key && conversationLocks.get(key) === context.runId) {
      conversationLocks.delete(key);
    }
  }

  function create(spec = {}) {
    if (activeCount() >= MAX_ACTIVE_RUNS) {
      throw new RunRegistryError(
        "parallel_limit",
        `Two comparisons are already running. Open or cancel one before starting another.`
      );
    }

    const basis = normalizeBasis(spec.basis);
    if (basis?.bookmarkId && blockedBookmarkIds.has(basis.bookmarkId)) {
      throw new RunRegistryError(
        "bookmark_deleting",
        "This bookmark is being deleted. Wait for that action to finish before starting a follow-up."
      );
    }
    const conversationLockKey = spec.followup === true ? basis?.key || "" : "";
    if (conversationLockKey) {
      const ownerRunId = conversationLocks.get(conversationLockKey);
      const owner = ownerRunId ? runs.get(ownerRunId) : null;
      if (owner && EXECUTING.has(owner.status)) {
        throw new RunRegistryError(
          "conversation_busy",
          "A follow-up for this conversation is already running. Wait for it to finish or cancel it first."
        );
      }
      if (owner?.keepConversationLock) {
        throw new RunRegistryError(
          "conversation_uncertain",
          "The previous follow-up has not received a final server status yet. Reload this page and reopen the saved conversation before continuing."
        );
      }
      if (ownerRunId) conversationLocks.delete(conversationLockKey);
    }

    const runId = cleanId(spec.runId) || randomId();
    const startedAt = Date.now();
    const config = deepFreeze(cloneValue(spec.config || {}));
    const context = {
      runId,
      requestIdentity: cleanId(spec.requestIdentity) || runId,
      status: "starting",
      phase: "prepare",
      createdAt: startedAt,
      startedAt,
      updatedAt: startedAt,
      finishedAt: null,
      auth: spec.auth || authSnapshot(),
      question: String(spec.question || ""),
      mode: String(spec.mode || "Standard"),
      config,
      basis,
      conversationLockKey,
      bookmark: {
        id: cleanId(spec.bookmarkId),
        title: String(spec.bookmarkTitle || spec.question || ""),
        status: "pending",
        writes: 0,
        latestMeta: cloneValue(basis?.bookmarkMeta || null),
        uiReady: false,
        error: null
      },
      chatSession: spec.chatSession || null,
      usage: cloneValue(spec.usage || null),
      attachments: cloneValue(spec.attachments || []),
      attachmentMeta: cloneValue(spec.attachmentMeta || []),
      evidenceSources: [],
      modelResults: Object.create(null),
      consensus: {
        status: "idle",
        streamText: "",
        text: "",
        differences: "",
        differencesData: null,
        sources: [],
        resultId: null,
        citationMeta: null,
        modelLabels: Object.create(null),
        completedTurn: null,
        bookmarkPayload: null,
        error: null
      },
      progress: {
        totalModels: 0,
        completedModels: 0,
        successfulModels: 0,
        failedModels: 0
      },
      persistence: {
        pendingWrites: 0,
        modelWrites: 0,
        consensusWrite: false,
        consensusPromise: null,
        consensusBookmark: null,
        consensusVersionParts: null,
        status: "pending",
        errors: []
      },
      controllers: {
        query: null,
        providers: new Map(),
        consensus: null
      },
      historyTurns: cloneValue(basis?.historyTurns || []),
      previousExchange: basis && basis.question && basis.consensus
        ? {
            question: basis.question,
            consensus: basis.consensus,
            turn: cloneValue(basis.currentTurn)
          }
        : null,
      completedBasis: null,
      error: null,
      cancelHook: null,
      keepConversationLock: false,
      metadata: cloneValue(spec.metadata || {})
    };

    runs.set(runId, context);
    if (conversationLockKey) conversationLocks.set(conversationLockKey, runId);
    visibleRunId = runId;
    visibleSavedView = null;
    selectedConversationBasis = null;
    emit("created", context);
    renderVisible();
    return context;
  }

  function get(runId) {
    return runs.get(cleanId(runId)) || null;
  }

  function update(runId, mutator, { render = true, eventType = "updated" } = {}) {
    const context = get(runId);
    if (!context) return null;
    if (typeof mutator === "function") mutator(context);
    else if (mutator && typeof mutator === "object") Object.assign(context, mutator);
    context.updatedAt = Date.now();
    emit(eventType, context);
    if (render && visibleRunId === context.runId) renderVisible();
    return context;
  }

  function setStatus(runId, status, error = null) {
    if (![...EXECUTING, ...TERMINAL].includes(status)) return get(runId);
    const current = get(runId);
    // Terminal means terminal. A late callback must not resurrect a canceled
    // or failed context by setting it back to running/succeeded.
    if (!current || TERMINAL.has(current.status)) return current;
    return update(runId, context => {
      context.status = status;
      if (status === "running" && !context.startedAt) context.startedAt = Date.now();
      if (TERMINAL.has(status)) {
        context.finishedAt = Date.now();
        if (error) context.error = error;
        // A canceled/failed follow-up whose pending server turn has an
        // unknown disposition must keep the local chat fence. Starting a
        // sibling turn behind an uncertain predecessor would make ordering
        // ambiguous. Session teardown still clears every fence.
        if (!context.keepConversationLock) releaseConversationLock(context);
      }
    }, { eventType: TERMINAL.has(status) ? "finished" : "updated" });
  }

  function setPhase(runId, phase) {
    return update(runId, context => { context.phase = String(phase || context.phase); });
  }

  function isExecuting(runId) {
    return EXECUTING.has(get(runId)?.status);
  }

  function isVisible(runId) {
    return cleanId(runId) === visibleRunId;
  }

  function visible() {
    return visibleRunId ? get(visibleRunId) : null;
  }

  function basisFromContext(context) {
    if (!context || context.status !== "succeeded") return null;
    return normalizeBasis(context.completedBasis);
  }

  function show(runId, { selectConversation = true } = {}) {
    const context = get(runId);
    if (!context) return false;
    visibleRunId = context.runId;
    visibleSavedView = null;
    if (selectConversation) selectedConversationBasis = basisFromContext(context);
    emit("visible", context);
    renderVisible();
    return true;
  }

  function showSavedView(view, basis = null) {
    visibleRunId = null;
    visibleSavedView = cloneValue(view || null);
    selectedConversationBasis = normalizeBasis(basis);
    emit("saved-view", null);
    renderVisible();
  }

  function clearVisible({ keepConversationBasis = false } = {}) {
    visibleRunId = null;
    visibleSavedView = null;
    if (!keepConversationBasis) selectedConversationBasis = null;
    emit("visible-cleared", null);
    renderVisible();
  }

  function selectConversationBasis(basis) {
    selectedConversationBasis = normalizeBasis(basis);
    emit("conversation-selected", visible());
    return selectedConversationBasis;
  }

  function getSelectedConversationBasis() {
    return cloneValue(selectedConversationBasis);
  }

  function setCompletedBasis(runId, basis) {
    const normalized = normalizeBasis(basis);
    return update(runId, context => {
      context.completedBasis = normalized;
      if (visibleRunId === context.runId) selectedConversationBasis = normalized;
    });
  }

  function cancel(runId, reason = "user") {
    const context = get(runId);
    if (!context || !EXECUTING.has(context.status)) return false;
    context.controllers.query?.abort?.();
    context.controllers.providers?.forEach?.(controller => controller?.abort?.());
    context.controllers.consensus?.abort?.();
    try { context.cancelHook?.(reason, context); } catch (error) {
      console.error("Run cancellation cleanup failed:", error);
    }
    context.phase = "canceled";
    setStatus(context.runId, "canceled", { code: "canceled", reason });
    return true;
  }

  // Costly post-actions such as Resolve share the same two-slot admission and
  // browser-session cancellation boundary. They do not replace the immutable
  // result context they annotate, but the registry still owns their controller.
  function beginAction(spec = {}) {
    const key = cleanId(spec.key);
    const ownerRunId = cleanId(spec.ownerRunId);
    const bookmarkId = cleanId(spec.bookmarkId);
    // A bookmark is the mutation resource. Two visible run contexts may point
    // at the same bookmark, so post-actions must deduplicate on the bookmark
    // before falling back to their owning run.
    const ownerKey = bookmarkId ? `bookmark:${bookmarkId}` : (ownerRunId || "session");
    const identity = `${ownerKey}:${key}`;
    if (!key) throw new RunRegistryError("invalid_action", "This action could not be started.");
    if (bookmarkId && blockedBookmarkIds.has(bookmarkId)) {
      throw new RunRegistryError("bookmark_deleting", "This bookmark is being deleted.");
    }
    if (Array.from(actions.values()).some(action => action.identity === identity)) {
      throw new RunRegistryError("action_busy", "This action is already running.");
    }
    if (activeCount() >= MAX_ACTIVE_RUNS) {
      throw new RunRegistryError(
        "parallel_limit",
        "Two AI runs are already active. Wait for one to finish or cancel it first."
      );
    }
    const actionId = cleanId(spec.actionId) || `action_${randomId().slice(4)}`;
    const action = {
      actionId,
      identity,
      key,
      ownerRunId,
      bookmarkId,
      auth: spec.auth || authSnapshot(),
      controller: spec.controller || new AbortController(),
      startedAt: Date.now()
    };
    actions.set(actionId, action);
    emit("action-started", ownerRunId ? get(ownerRunId) : null);
    return action;
  }

  function finishAction(actionId) {
    const id = cleanId(actionId);
    const action = actions.get(id);
    if (!action) return false;
    actions.delete(id);
    emit("action-finished", action.ownerRunId ? get(action.ownerRunId) : null);
    return true;
  }

  function cancelActionsForBookmark(bookmarkId, reason = "bookmark_deleted") {
    const id = cleanId(bookmarkId);
    Array.from(actions.values()).forEach(action => {
      if (action.bookmarkId !== id) return;
      try { action.controller?.abort?.(reason); } catch (_) { action.controller?.abort?.(); }
      actions.delete(action.actionId);
    });
    emit("actions-canceled", null);
  }

  function blockBookmarkMutation(bookmarkId) {
    const id = cleanId(bookmarkId);
    if (!id) return false;
    blockedBookmarkIds.add(id);
    emit("bookmark-blocked", null);
    return true;
  }

  function unblockBookmarkMutation(bookmarkId) {
    const removed = blockedBookmarkIds.delete(cleanId(bookmarkId));
    if (removed) emit("bookmark-unblocked", null);
    return removed;
  }

  function cancelAll(reason = "session_end") {
    Array.from(actions.values()).forEach(action => {
      try { action.controller?.abort?.(reason); } catch (_) { action.controller?.abort?.(); }
    });
    actions.clear();
    Array.from(runs.values()).forEach(context => {
      if (EXECUTING.has(context.status)) cancel(context.runId, reason);
    });
  }

  function clearAll(reason = "session_end") {
    cancelAll(reason);
    runs.clear();
    conversationLocks.clear();
    usageSnapshotFences.clear();
    blockedBookmarkIds.clear();
    visibleRunId = null;
    selectedConversationBasis = null;
    visibleSavedView = null;
    emit("cleared", null);
    renderVisible();
  }

  function findByBookmarkId(bookmarkId) {
    const id = cleanId(bookmarkId);
    if (!id) return null;
    const matches = Array.from(runs.values()).filter(context => (
      context.bookmark.id === id
      && context.bookmark.deleted !== true
      // A failed/canceled follow-up is only a diagnostic run snapshot. It must
      // not shadow the last authoritative server bookmark when that bookmark
      // is opened again.
      && (EXECUTING.has(context.status) || context.status === "succeeded")
    ));
    matches.sort((left, right) => right.createdAt - left.createdAt);
    return matches[0] || null;
  }

  function list() {
    return Array.from(runs.values()).sort((left, right) => left.createdAt - right.createdAt);
  }

  function notePersistence(runId, patch = {}) {
    return update(runId, context => {
      Object.assign(context.persistence, patch);
      if (patch.error) context.persistence.errors.push(patch.error);
    }, { render: false, eventType: "persistence" });
  }

  // Prevent a second physical click in the same gesture window from turning
  // the freshly changed send button into an accidental cancel action.
  function claimStartAction() {
    const now = Date.now();
    if (now - lastStartClaimAt < 450) return false;
    lastStartClaimAt = now;
    return true;
  }

  function setProjector(nextProjector) {
    projector = typeof nextProjector === "function" ? nextProjector : null;
    renderVisible();
  }

  function snapshot() {
    return Object.freeze({
      visibleRunId,
      activeCount: activeCount(),
      maxActiveRuns: MAX_ACTIVE_RUNS,
      selectedConversationBasis: cloneValue(selectedConversationBasis),
      actions: Array.from(actions.values()).map(action => ({
        actionId: action.actionId,
        key: action.key,
        ownerRunId: action.ownerRunId,
        bookmarkId: action.bookmarkId,
        startedAt: action.startedAt
      })),
      runs: Array.from(runs.values()).map(context => ({
        runId: context.runId,
        status: context.status,
        phase: context.phase,
        question: context.question,
        bookmarkId: context.bookmark.id,
        createdAt: context.createdAt,
        finishedAt: context.finishedAt
      }))
    });
  }

  window.App.runRegistry = Object.freeze({
    MAX_ACTIVE_RUNS,
    RunRegistryError,
    create,
    get,
    update,
    setStatus,
    setPhase,
    isExecuting,
    isVisible,
    isAuthCurrent,
    reconcileUsageSnapshot,
    visible,
    show,
    showSavedView,
    clearVisible,
    selectConversationBasis,
    getSelectedConversationBasis,
    setCompletedBasis,
    cancel,
    beginAction,
    finishAction,
    cancelActionsForBookmark,
    blockBookmarkMutation,
    unblockBookmarkMutation,
    cancelAll,
    clearAll,
    activeCount,
    list,
    findByBookmarkId,
    notePersistence,
    claimStartAction,
    setProjector,
    renderVisible,
    snapshot
  });
})();
