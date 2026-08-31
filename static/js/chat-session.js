// Minimal bridge between the existing DOM-based run flow and /chats.
// No credentials or model answers are retained here.
(function () {
  "use strict";

  window.App = window.App || {};

  const ID_RE = /^[0-9a-f]{32}$/;

  function secureRequestId() {
    const cryptoApi = window.crypto;
    if (typeof cryptoApi?.randomUUID === "function") {
      return cryptoApi.randomUUID();
    }
    if (typeof cryptoApi?.getRandomValues !== "function") return null;
    const bytes = new Uint8Array(16);
    cryptoApi.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  function cleanString(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  // Anhaenge einer Frage als reine Metadaten. Sie reisen mit dem Turn, nicht
  // mit dem Bookmark-Dokument: eine Folgefrage ohne Datei raeumte sonst die
  // Datei der vorigen Frage mit weg. Der Server normalisiert noch einmal
  // (MIME-Allowlist, Obergrenze) — hier geht es nur darum, nichts Fremdes
  // mitzuschicken.
  function cleanAttachments(value) {
    if (!Array.isArray(value)) return [];
    return value.reduce((items, item) => {
      const name = cleanString(item?.name);
      const mime = cleanString(item?.mime);
      if (!name || !mime || items.length >= 2) return items;
      const size = Number(item?.size);
      items.push({
        name: name,
        mime: mime,
        size: Number.isFinite(size) && size > 0 ? Math.floor(size) : 0
      });
      return items;
    }, []);
  }

  function cleanModels(value) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    return value.reduce((models, item) => {
      const model = cleanString(item);
      if (model && !seen.has(model)) {
        seen.add(model);
        models.push(model);
      }
      return models;
    }, []);
  }

  async function responseJson(response) {
    try {
      return await response.json();
    } catch (_) {
      return {};
    }
  }

  function diagnostic(stage, status = 0) {
    console.warn("[chat-session] persistence unavailable", { stage, status });
  }

  // Fehlerursachen, die ein Wiederholen NICHT behebt: der Nutzer muss etwas
  // aendern (aufraeumen, neu anfangen, warten). Der Server liefert dafuer
  // error_code + Text; alles andere bleibt beim generischen Retry-Hinweis.
  const PERSISTENCE_ERROR_MESSAGES = {
    chat_limit_reached:
      "You have reached the maximum number of saved conversations. Delete one to start another.",
    turn_limit_reached:
      "This conversation has reached its maximum length. Start a new comparison to continue.",
    chat_rate_limited:
      "Too many new conversations in a short time. Please wait a moment.",
    pro_required: "This consensus engine is reserved for Pro users."
  };

  // Der globale HTTPException-Handler in main.py packt das FastAPI-"detail"
  // in "error" um ({"error": {...}}), waehrend FastAPI selbst {"detail": {...}}
  // liefert. Beide Formen muessen hier ankommen duerfen - sonst greift die
  // Zuordnung genau in Produktion nicht, wo der Handler aktiv ist.
  function persistenceErrorMessage(data) {
    for (const candidate of [data?.detail, data?.error, data]) {
      if (!candidate || typeof candidate !== "object") continue;
      const code = cleanString(candidate.error_code);
      if (code && PERSISTENCE_ERROR_MESSAGES[code]) {
        return PERSISTENCE_ERROR_MESSAGES[code];
      }
    }
    return "";
  }

  function abortError() {
    const error = new Error("Request aborted");
    error.name = "AbortError";
    return error;
  }

  function waitForRetry(milliseconds, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(abortError());
        return;
      }
      const timer = window.setTimeout(resolve, Math.max(0, milliseconds));
      signal?.addEventListener?.("abort", () => {
        window.clearTimeout(timer);
        reject(abortError());
      }, { once: true });
    });
  }

  function sameLogicalRun(left, right) {
    return Boolean(
      left
      && right
      && left.question === right.question
      && left.mode === right.mode
      && left.deepSearch === right.deepSearch
      && left.consensusModel === right.consensusModel
      && left.isFollowup === right.isFollowup
      && left.useOwnKeys === right.useOwnKeys
      && left.selectedModels.length === right.selectedModels.length
      && left.selectedModels.every((model, index) => model === right.selectedModels[index])
    );
  }

  // Every browser run owns one of these sessions.  The exported
  // window.App.chatSession remains the selected-view compatibility session;
  // run callbacks use a private instance created at start and therefore can
  // never be rebound by opening another bookmark.
  function createChatSession(initial = {}) {
    const chatSession = {
    activeChatId: null,
    activeTurnId: null,
    pendingChatId: null,
    pendingTurnId: null,
    pendingContextVersionId: null,
    pendingClientRequestId: null,
    pendingUsageRunKey: null,
    logicalRun: null,
    _turnPromise: null,
    _turnRequestUncertain: false,
    _contextPromise: null,
    _chatCreationAttempted: false,
    _persistenceDisabled: false,
    _needsReconcile: false,
    // Grund der letzten fehlgeschlagenen Chat-/Turn-Anlage, sofern der Server
    // einen genannt hat. Ein erreichtes Limit ist dauerhaft - "Please retry"
    // waere dort eine Sackgasse mit falscher Auskunft.
    lastPersistenceError: "",

    hasActiveChat() {
      return ID_RE.test(this.activeChatId || "") && ID_RE.test(this.activeTurnId || "");
    },

    canReuseUsageRun({
      question,
      mode,
      deepSearch,
      selectedModels,
      consensusModel,
      isFollowup,
      useOwnKeys
    }) {
      if (!this.pendingClientRequestId || !this.logicalRun) return false;
      return sameLogicalRun(this.logicalRun, {
        ...this.logicalRun,
        question: cleanString(question),
        mode: cleanString(mode),
        deepSearch: deepSearch === true,
        selectedModels: cleanModels(selectedModels),
        consensusModel: cleanString(consensusModel),
        isFollowup: isFollowup === true,
        useOwnKeys: useOwnKeys === true
      });
    },

    beginRun({
      question,
      mode,
      deepSearch,
      selectedModels,
      consensusModel,
      isFollowup,
      prepareSucceeded,
      useOwnKeys,
      usageRunKey,
      attachments
    }) {
      const followup = isFollowup === true;
      const nextRun = {
        question: cleanString(question),
        mode: cleanString(mode),
        deepSearch: deepSearch === true,
        selectedModels: cleanModels(selectedModels),
        consensusModel: cleanString(consensusModel),
        isFollowup: followup,
        useOwnKeys: useOwnKeys === true,
        prepareSucceeded: prepareSucceeded === true,
        // Bewusst NICHT Teil von sameLogicalRun: derselbe Lauf schickt beim
        // zweiten Versuch dieselben Dateien mit, und ein Wiederholungsversuch
        // darf daran nicht scheitern.
        attachments: cleanAttachments(attachments)
      };
      const retryingPendingTurn = Boolean(
        ID_RE.test(this.pendingChatId || "")
        && ID_RE.test(this.pendingTurnId || "")
        && this.pendingClientRequestId
        && sameLogicalRun(this.logicalRun, nextRun)
      );
      if (retryingPendingTurn) {
        this.logicalRun = nextRun;
        this._persistenceDisabled = prepareSucceeded !== true;
        if (!this.pendingUsageRunKey && cleanString(usageRunKey)) {
          this.pendingUsageRunKey = cleanString(usageRunKey);
        }
        return;
      }

      const abandoningPendingFollowup = Boolean(
        followup
        && ID_RE.test(this.pendingChatId || "")
        && ID_RE.test(this.pendingTurnId || "")
        && this.pendingChatId === this.activeChatId
      );
      if (!followup) {
        this.activeChatId = null;
        this.activeTurnId = null;
      } else if (abandoningPendingFollowup) {
        // Never append behind a predecessor whose completion was not confirmed.
        this.activeChatId = null;
        this.activeTurnId = null;
      }
      this.pendingChatId = followup && ID_RE.test(this.activeChatId || "")
        ? this.activeChatId
        : null;
      this.pendingTurnId = null;
      this.pendingContextVersionId = null;
      this.pendingClientRequestId = secureRequestId();
      this.pendingUsageRunKey = cleanString(usageRunKey) || null;
      this._turnPromise = null;
      this._turnRequestUncertain = false;
      this._contextPromise = null;
      this._chatCreationAttempted = followup;
      this._persistenceDisabled = prepareSucceeded !== true || !this.pendingClientRequestId;
      this._needsReconcile = false;
      this.lastPersistenceError = "";
      this.logicalRun = nextRun;
    },

    async ensurePendingTurn({ idToken, question, consensusModel, signal }) {
      const run = this.logicalRun;
      const normalizedQuestion = cleanString(question);
      if (
        !run
        || this._persistenceDisabled
        || !cleanString(idToken)
        || !normalizedQuestion
        || normalizedQuestion !== run.question
        || !run.mode
        || !run.selectedModels.length
      ) {
        return null;
      }

      if (run.isFollowup && !ID_RE.test(this.activeChatId || "")) {
        diagnostic("legacy_followup_without_active_chat");
        this._persistenceDisabled = true;
        return null;
      }

      if (ID_RE.test(this.pendingChatId || "") && ID_RE.test(this.pendingTurnId || "")) {
        const binding = {
          chatId: this.pendingChatId,
          turnId: this.pendingTurnId
        };
        if (ID_RE.test(this.pendingContextVersionId || "")) {
          binding.contextVersionId = this.pendingContextVersionId;
        }
        return binding;
      }
      if (this._turnPromise) return await this._turnPromise;

      // Never mutate logicalRun here: sameLogicalRun() compares a retry
      // against what the user actually chose when the run started, and
      // overwriting it would make a changed picker look like the same run.
      if (!run.consensusModel) {
        run.consensusModel = cleanString(consensusModel);
      }
      if (!run.consensusModel) return null;
      this._turnPromise = this._createPendingTurn(cleanString(idToken), signal);
      try {
        return await this._turnPromise;
      } finally {
        this._turnPromise = null;
      }
    },

    async _createPendingTurn(idToken, signal) {
      const run = this.logicalRun;
      const headers = {
        "Authorization": `Bearer ${idToken}`,
        "Content-Type": "application/json"
      };

      let chatId = this.pendingChatId;
      if (!chatId) {
        if (this._chatCreationAttempted) return null;
        this._chatCreationAttempted = true;
        try {
          const response = await fetch("/chats", {
            method: "POST",
            headers,
            body: "{}",
            signal
          });
          const data = await responseJson(response);
          chatId = data?.chat?.id;
          if (!response.ok || !ID_RE.test(chatId || "")) {
            diagnostic("create_chat", response.status);
            this.lastPersistenceError = persistenceErrorMessage(data);
            this._persistenceDisabled = true;
            return null;
          }
          this.pendingChatId = chatId;
        } catch (_) {
          diagnostic("create_chat");
          this._persistenceDisabled = true;
          return null;
        }
      }

      const turnPayload = {
        question: run.question,
        mode: run.mode,
        deep_search: run.deepSearch,
        selected_models: run.selectedModels,
        consensus_model: run.consensusModel,
        client_request_id: this.pendingClientRequestId,
        attachments: run.attachments || []
      };
      try {
        const response = await fetch(`/chats/${chatId}/turns`, {
          method: "POST",
          headers,
          body: JSON.stringify(turnPayload),
          signal
        });
        const data = await responseJson(response);
        const turnId = data?.turn?.id;
        if (!response.ok || !ID_RE.test(turnId || "")) {
          // A definitive client error happened before a valid create. A 5xx (or
          // a malformed success) can occur after the server transaction
          // committed but before its response read completed, so keep the
          // conversation fenced just like a transport loss.
          this._turnRequestUncertain = response.status >= 500 || response.ok;
          diagnostic("create_turn", response.status);
          this.lastPersistenceError = persistenceErrorMessage(data);
          return null;
        }
        this._turnRequestUncertain = false;
        this.pendingChatId = chatId;
        this.pendingTurnId = turnId;
        return { chatId, turnId };
      } catch (_) {
        this._turnRequestUncertain = true;
        diagnostic("create_turn");
        return null;
      }
    },

    requiresAuthoritativeContext() {
      return Boolean(
        this.logicalRun?.isFollowup
        && ID_RE.test(this.activeChatId || "")
        && this.pendingChatId === this.activeChatId
        && ID_RE.test(this.pendingTurnId || "")
      );
    },

    async ensureContext({ idToken, useOwnKeys, usageRunKey, memoryApiKey, signal }) {
      if (!this.requiresAuthoritativeContext()) return null;
      if (ID_RE.test(this.pendingContextVersionId || "")) {
        return {
          chatId: this.pendingChatId,
          turnId: this.pendingTurnId,
          contextVersionId: this.pendingContextVersionId
        };
      }
      if (this._contextPromise) return await this._contextPromise;
      this._contextPromise = this._buildContext({
        idToken: cleanString(idToken),
        useOwnKeys: useOwnKeys === true,
        usageRunKey: cleanString(usageRunKey) || this.pendingUsageRunKey,
        memoryApiKey: cleanString(memoryApiKey),
        signal
      });
      try {
        return await this._contextPromise;
      } finally {
        this._contextPromise = null;
      }
    },

    async _buildContext({ idToken, useOwnKeys, usageRunKey, memoryApiKey, signal }) {
      if (!idToken) throw new Error("Authentication required for chat context.");
      const body = { useOwnKeys };
      if (useOwnKeys) {
        if (memoryApiKey) body.openrouter_key = memoryApiKey;
      } else {
        if (!usageRunKey) throw new Error("The prepared usage run is unavailable.");
        body.usage_run_key = usageRunKey;
      }
      const url = `/chats/${this.pendingChatId}/turns/${this.pendingTurnId}/context`;
      const maxAttempts = 4;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${idToken}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify(body),
          signal
        });
        const data = await responseJson(response);
        if (response.status === 202) {
          if (attempt === maxAttempts) {
            throw new Error("Conversation context is still being prepared. Please retry.");
          }
          const retryHeader = Number(response.headers?.get?.("Retry-After"));
          const retryBody = Number(data?.retry_after_seconds);
          const seconds = Number.isFinite(retryHeader) ? retryHeader
            : (Number.isFinite(retryBody) ? retryBody : 1);
          await waitForRetry(Math.min(2000, Math.max(0, seconds * 1000)), signal);
          continue;
        }
        const context = data?.context;
        const versionId = context?.id;
        if (
          response.ok
          && ["ready", "degraded"].includes(context?.state)
          && ID_RE.test(versionId || "")
          && context?.target_turn_id === this.pendingTurnId
        ) {
          this.pendingContextVersionId = versionId;
          return {
            chatId: this.pendingChatId,
            turnId: this.pendingTurnId,
            contextVersionId: versionId,
            contextState: context.state
          };
        }
        const detail = typeof data?.detail === "string"
          ? data.detail
          : (data?.detail?.error || data?.error || "Conversation context could not be prepared.");
        throw new Error(detail);
      }
      throw new Error("Conversation context could not be prepared.");
    },

    contextBinding() {
      if (
        !ID_RE.test(this.pendingChatId || "")
        || !ID_RE.test(this.pendingTurnId || "")
        || !ID_RE.test(this.pendingContextVersionId || "")
      ) return null;
      return {
        chat_id: this.pendingChatId,
        turn_id: this.pendingTurnId,
        context_version_id: this.pendingContextVersionId
      };
    },

    markPendingUncertain() {
      if (ID_RE.test(this.pendingChatId || "") && ID_RE.test(this.pendingTurnId || "")) {
        this._needsReconcile = true;
      }
    },

    hasUncertainTurn() {
      return this._turnRequestUncertain === true || this._needsReconcile === true;
    },

    async inspectPendingTurn({ idToken, signal }) {
      if (!this._needsReconcile) return null;
      if (!ID_RE.test(this.pendingChatId || "") || !ID_RE.test(this.pendingTurnId || "")) {
        this._needsReconcile = false;
        return null;
      }
      const response = await fetch(`/chats/${this.pendingChatId}/turns/${this.pendingTurnId}`, {
        headers: { "Authorization": `Bearer ${cleanString(idToken)}` },
        signal
      });
      const data = await responseJson(response);
      if (!response.ok || !data?.turn) {
        throw new Error("The pending conversation turn could not be checked.");
      }
      this._needsReconcile = false;
      const state = data.turn.status;
      if (state === "failed") {
        this.handleConsensusResult({
          chatId: this.pendingChatId,
          turnId: this.pendingTurnId,
          chatPersisted: false,
          chatTurnState: "failed"
        });
      }
      return data.turn;
    },

    handleConsensusResult({ chatId, turnId, chatPersisted, chatTurnState }) {
      if (!ID_RE.test(chatId || "") || !ID_RE.test(turnId || "")) return;
      if (chatId !== this.pendingChatId || turnId !== this.pendingTurnId) return;
      if (chatTurnState === "pending") return;
      if (chatTurnState !== "completed" && chatTurnState !== "failed") return;
      if (chatTurnState === "completed" && chatPersisted === true) {
        this.activeChatId = chatId;
        this.activeTurnId = turnId;
      }
      // A failed target is discarded, but it must not damage the last
      // completed predecessor. A later retry becomes a new positioned turn;
      // the server-side context builder reads completed predecessors only.
      this._clearPending();
    },

    handleConsensusCancelled() {
      // A local transport abort is not an authoritative server disposition.
      this.markPendingUncertain();
    },

    restoreCompletedChat(chatId, turnId) {
      this.reset();
      if (!ID_RE.test(chatId || "") || !ID_RE.test(turnId || "")) return false;
      this.activeChatId = chatId;
      this.activeTurnId = turnId;
      return true;
    },

    _clearPending() {
      this.pendingChatId = null;
      this.pendingTurnId = null;
      this.pendingContextVersionId = null;
      this.pendingClientRequestId = null;
      this.pendingUsageRunKey = null;
      this._turnPromise = null;
      this._turnRequestUncertain = false;
      this._contextPromise = null;
      this._needsReconcile = false;
    },

    reset() {
      this.activeChatId = null;
      this.activeTurnId = null;
      this.pendingChatId = null;
      this.pendingTurnId = null;
      this.pendingContextVersionId = null;
      this.pendingClientRequestId = null;
      this.pendingUsageRunKey = null;
      this.logicalRun = null;
      this._turnPromise = null;
      this._turnRequestUncertain = false;
      this._contextPromise = null;
      this._chatCreationAttempted = false;
      this._persistenceDisabled = false;
      this._needsReconcile = false;
      this.lastPersistenceError = "";
    }
    };

    const initialChatId = cleanString(initial.activeChatId);
    const initialTurnId = cleanString(initial.activeTurnId);
    if (ID_RE.test(initialChatId) && ID_RE.test(initialTurnId)) {
      chatSession.activeChatId = initialChatId;
      chatSession.activeTurnId = initialTurnId;
    }
    return chatSession;
  }

  const chatSession = createChatSession();
  window.App.createChatSession = createChatSession;
  window.App.chatSession = chatSession;
})();
