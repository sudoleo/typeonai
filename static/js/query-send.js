// Multi-run query orchestration.
// Every await/callback is bound to one RunContext from App.runRegistry; the
// shared response DOM is written only by run-view.js when that run is visible.
(function () {
  "use strict";

  window.App = window.App || {};
  const registry = window.App.runRegistry;
  const trackAppEvent = window.App.trackAppEvent || (() => {});

  // Familien des Laufs: window.App.modelPrefs ist die eine Quelle (Server:
  // cfg.PROVIDERS). Der eine OpenRouter-Key gilt fuer alle Familien.
  function providerDefinitions() {
    return (window.App.modelPrefs || []).map(pref => ({
      provider: pref.key,
      endpoint: pref.endpoint,
      boxId: pref.responseId,
      checkboxId: pref.checkId,
      selectId: pref.selectId,
      keyName: "openrouterKey",
      pref
    }));
  }

  function isAbortError(error) {
    return error?.name === "AbortError";
  }

  function getActiveMode() {
    return document.getElementById("deepSearchToggle")?.checked ? "Deep Think" : "Standard";
  }

  function isDemoQuery(question) {
    return String(question || "").trim().toLowerCase() === "demo";
  }

  function secureKey() {
    return globalThis.crypto?.randomUUID?.()
      || `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`;
  }

  function bookmarkIdForRun(runId) {
    return `b_${String(runId || "").replace(/[^A-Za-z0-9_]/g, "_").slice(0, 94)}`;
  }

  function optionLabel(select) {
    const option = select?.options?.[select.selectedIndex];
    return String(option?.dataset?.modelLabel || option?.text || select?.value || "").trim();
  }

  function selectedProviders(attachmentCount, deepSearch) {
    return providerDefinitions().reduce((items, definition) => {
      if (!document.getElementById(definition.checkboxId)?.checked) return items;
      const select = document.getElementById(definition.selectId);
      const modelId = String(select?.value || "").trim();
      if (!modelId) return items;
      // Die Faehigkeit gehoert zum effektiven Modell, nicht zur Familie:
      // GLM 5.3 Flash ist multimodal, GLM 5.3 text-only.
      const acceptsAttachments = typeof window.App.modelAcceptsAttachments === "function"
        ? window.App.modelAcceptsAttachments(definition.pref, modelId, deepSearch)
        : definition.pref.handlesAttachments !== false;
      if (attachmentCount > 0 && !acceptsAttachments) return items;
      const deepLabel = window.App.deepThinkModelLabels?.[definition.provider];
      items.push({
        ...definition,
        modelId,
        modelLabel: deepSearch && deepLabel
          ? deepLabel
          : (optionLabel(select) || definition.provider)
      });
      return items;
    }, []);
  }

  function currentSystemPrompt() {
    const base = localStorage.getItem("systemPrompt")
      || "Please answer thoroughly and precisely, explaining your reasoning and covering the relevant details. Do not oversimplify. Do not ask any follow-up or clarifying questions; answer directly with the information available.";
    const now = new Date();
    const weekday = now.toLocaleDateString("en-US", { weekday: "long" });
    const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    return `Today is ${weekday}, ${date}. ${base}`;
  }

  function createUsage(deepThink, useOwnKeys) {
    return {
      key: useOwnKeys ? null : secureKey(),
      deepThink: deepThink === true,
      useOwnKeys: useOwnKeys === true,
      status: useOwnKeys ? "own_keys" : "new"
    };
  }

  function authIsCurrent(context) {
    return registry.isAuthCurrent(context);
  }

  function clearSensitiveRunData(context) {
    if (!context) return;
    context.credentials = null;
    context.attachments = [];
  }

  function runIsCurrent(context) {
    return registry.isExecuting(context.runId)
      && !context.controllers.query?.signal?.aborted
      && authIsCurrent(context);
  }

  function unwrapApiError(data) {
    const detail = data?.detail;
    if (detail && typeof detail === "object") {
      return { ...detail, error: detail.error || detail.message || "Request failed." };
    }
    return data || {};
  }

  function apiErrorMessage(data, fallback = "Request failed.") {
    const normalized = unwrapApiError(data);
    const candidate = normalized.error || normalized.detail || normalized.message || fallback;
    if (typeof candidate === "string") return candidate;
    if (Array.isArray(candidate)) {
      const messages = candidate.map(item => item?.msg || item?.message || item?.error).filter(Boolean);
      if (messages.length) return messages.join(" ");
    }
    if (candidate && typeof candidate === "object") {
      return String(candidate.error || candidate.message || candidate.detail || fallback);
    }
    return fallback;
  }

  function isUsageStorageBusy(data) {
    const normalized = unwrapApiError(data);
    return String(normalized.error_code || normalized.code || "").toLowerCase() === "usage_storage_busy";
  }

  function isUsageLimit(data, message = "") {
    if (window.App.usageLimit?.isLimitError) {
      return window.App.usageLimit.isLimitError(data, message);
    }
    const normalized = unwrapApiError(data);
    const code = String(normalized.error_code || normalized.code || "").toLowerCase();
    const text = String(message || normalized.error || normalized.detail || "").toLowerCase();
    return code.includes("limit") || /usage limit|quota|used up|exhausted/.test(text);
  }

  function updateUsage(context, data) {
    const normalized = unwrapApiError(data);
    if (normalized.usage_run_status) context.usage.status = normalized.usage_run_status;
    if (normalized.is_pro_user !== undefined) {
      context.usage.isProUser = normalized.is_pro_user === true;
    }
    // Die volle Stufe, damit ein Plus-Lauf die Ansicht nicht auf Free
    // zurueckstellt (is_pro_user ist fuer Plus false).
    if (normalized.tier !== undefined) {
      context.usage.tier = window.App.normalizeTier?.(normalized.tier) || "free";
    }
    // Usage is account-level rather than view-level. The auth fence prevents
    // a late response from a previous login from repainting the next account.
    if (authIsCurrent(context)) {
      const usageView = registry.reconcileUsageSnapshot?.(context, {
        remaining: normalized.free_usage_remaining,
        deepRemaining: normalized.deep_remaining,
        totalLimit: normalized.limit ?? window.currentMaxLimit,
        deepLimit: normalized.deep_limit ?? window.currentDeepLimit
      }) || {
        remaining: normalized.free_usage_remaining,
        deepRemaining: normalized.deep_remaining,
        totalLimit: normalized.limit ?? window.currentMaxLimit,
        deepLimit: normalized.deep_limit ?? window.currentDeepLimit
      };
      window.App.renderUsageDisplay?.(usageView);
      const noSavedViewSelected = !registry.visible()
        && !registry.getSelectedConversationBasis?.();
      // Wie in run-view.js: ein blosses "is_pro_user: false" ohne "tier" ist
      // seit der Plus-Stufe kein Free-Beleg mehr, sondern nur "nicht Pro".
      // Nur ein ausdrueckliches tier oder ein true darf die Anzeige stellen.
      const tierSignal = normalized.tier
        ?? (normalized.is_pro_user === true ? "pro" : null);
      if (tierSignal !== null
          && (registry.isVisible(context.runId) || noSavedViewSelected)) {
        window.updateUserTierUI?.(tierSignal, true);
      }
    }
  }

  async function prepareWithRetry(payload, signal) {
    let result = null;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const response = await fetch("/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal
      });
      let data = {};
      try { data = await response.json(); } catch (_) {
        data = { error: `Request failed with HTTP ${response.status}.` };
      }
      result = { response, data };
      if (response.ok || !isUsageStorageBusy(data) || attempt === 3) return result;
      await new Promise(resolve => window.setTimeout(resolve, attempt * 350));
    }
    return result;
  }

  function contextStreamRenderer(context, provider) {
    const RENDER_INTERVAL = 120;
    let timer = null;
    let lastRender = 0;

    function render() {
      timer = null;
      lastRender = Date.now();
      if (registry.isVisible(context.runId)) registry.renderVisible();
    }

    function scheduleRender() {
      if (!registry.isVisible(context.runId)) return;
      const elapsed = Date.now() - lastRender;
      if (elapsed >= RENDER_INTERVAL) render();
      else if (!timer) timer = window.setTimeout(render, RENDER_INTERVAL - elapsed);
    }

    return {
      append(chunk) {
        if (!runIsCurrent(context)) return;
        const result = context.modelResults[provider];
        result.status = "streaming";
        result.streamText += String(chunk || "");
        scheduleRender();
      },
      markReasoning() {
        if (!runIsCurrent(context)) return;
        const result = context.modelResults[provider];
        if (!result.streamText) result.status = "reasoning";
        scheduleRender();
      },
      stop() {
        if (timer) window.clearTimeout(timer);
        timer = null;
      }
    };
  }

  function conversationPayload(context, payload) {
    const binding = context.chatSession?.contextBinding?.();
    if (binding) Object.assign(payload, binding);
    else if (context.previousExchange && !context.basis?.chatId) {
      payload.context = {
        previous_question: context.previousExchange.question,
        previous_consensus: context.previousExchange.consensus
      };
    }
    return payload;
  }

  function persistenceOptions(context) {
    return {
      runId: context.runId,
      bookmarkId: context.bookmark.id,
      sources: context.evidenceSources.map(source => ({ ...source })),
      attachments: context.attachmentMeta.map(item => ({ ...item })),
      auth: context.auth
    };
  }

  async function runProvider(context, providerConfig, idToken, systemPrompt) {
    const result = context.modelResults[providerConfig.provider];
    const controller = new AbortController();
    context.controllers.providers.set(providerConfig.provider, controller);
    const abortProvider = () => controller.abort();
    context.controllers.query.signal.addEventListener("abort", abortProvider, { once: true });

    const payload = conversationPayload(context, {
      question: context.question,
      deep_search: context.config.deepSearch,
      system_prompt: systemPrompt,
      mode: context.mode,
      model: providerConfig.modelId,
      id_token: idToken,
      useOwnKeys: context.config.useOwnKeys,
      usage_run_key: context.usage.key
    });
    if (context.attachments.length) payload.attachments = context.attachments;
    if (context.config.useOwnKeys) payload.openrouter_key = context.credentials.openrouterKey || "";

    try {
      const response = await window.streamSSERequest(
        providerConfig.endpoint,
        payload,
        controller.signal,
        { delta: contextStreamRenderer(context, providerConfig.provider) }
      );
      if (!registry.isExecuting(context.runId) || !authIsCurrent(context) || result.status === "skipped") return;
      updateUsage(context, response.data || {});
      if (!response.ok) throw new Error(apiErrorMessage(response.data, `${providerConfig.provider} HTTP ${response.status}`));
      const data = response.data || {};
      if (!data.response) throw new Error(apiErrorMessage(data, `${providerConfig.provider} returned an empty response.`));

      const prepared = window.App.prepareResponseSourcesForEvidence
        ? window.App.prepareResponseSourcesForEvidence(data.response, data.sources || [], context.evidenceSources)
        : { markdown: data.response, sources: data.sources || [], evidenceSources: context.evidenceSources };
      context.evidenceSources = prepared.evidenceSources;
      result.status = "complete";
      result.text = prepared.markdown;
      result.streamText = prepared.markdown;
      result.sources = prepared.sources;
      result.rawSources = Array.isArray(data.sources) ? data.sources : [];
      result.error = null;
      context.progress.completedModels += 1;
      context.progress.successfulModels += 1;
      registry.update(context.runId, () => {});

      // Agent-mode provider answers are staged in the run context. Only the
      // completed consensus endpoint promotes its server-authoritative answer
      // set to the bookmark atomically. Writing each follow-up provider here
      // would temporarily mix a new failed turn with the previous consensus.
      if (authIsCurrent(context) && context.config.agentMode !== true) {
        const promise = window.saveBookmark?.(
          context.question,
          result.text,
          providerConfig.provider,
          context.mode,
          context.previousExchange?.question || "",
          persistenceOptions(context)
        );
        if (promise?.catch) promise.catch(() => undefined);
      }
    } catch (error) {
      if (result.status === "skipped") return;
      if (isAbortError(error)) {
        if (context.status === "canceled") result.status = "canceled";
        return;
      }
      result.status = "error";
      result.error = `${providerConfig.provider} error: ${error?.message || "Request failed."}`;
      context.progress.completedModels += 1;
      context.progress.failedModels += 1;
      registry.update(context.runId, () => {});
    } finally {
      context.controllers.query?.signal?.removeEventListener?.("abort", abortProvider);
      context.controllers.providers.delete(providerConfig.provider);
    }
  }

  async function releaseUnusedUsage(context, { allowAfterLogout = false } = {}) {
    if (!context?.usage?.key || !["new", "reserved"].includes(context.usage.status)) return;
    const key = context.usage.key;
    context.usage.status = "released_local";
    try {
      const token = await context.auth.user?.getIdToken?.(false);
      if (!token || (!allowAfterLogout && !authIsCurrent(context))) return;
      const response = await fetch("/usage/run/release", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: token, usage_run_key: key }),
        keepalive: true
      });
      const data = await response.json().catch(() => ({}));
      if (response.ok && authIsCurrent(context)) {
        const usageView = registry.reconcileUsageSnapshot?.(context, data, { authoritative: true });
        if (usageView) window.App.renderUsageDisplay?.(usageView);
      }
    } catch (_) {}
  }

  function restoreComposerAfterUnsentRun(context) {
    if (!registry.isVisible(context.runId)) return;
    const input = document.getElementById("questionInput");
    const composerAttachments = window.getAttachmentsPayload?.()
      || (Array.isArray(window.pendingAttachments) ? window.pendingAttachments : []);
    const composerOccupied = Boolean(input?.value?.trim())
      || Boolean(window.App.quote?.text?.())
      || composerAttachments.length > 0;
    // The user may have prepared a new draft while this request was in the
    // background and then reopened it only to cancel. Never replace that newer
    // composer ownership with the canceled run's frozen draft/attachments.
    if (composerOccupied) return;
    // Admission deliberately detaches the shared composer from its previous
    // conversation. If the run never reached provider fan-out, restore that
    // exact frozen basis so retrying the draft remains a follow-up.
    if (context.previousExchange && context.basis) {
      registry.selectConversationBasis(context.basis);
    }
    if (input) {
      input.value = context.metadata.draftQuestion || context.question;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      if (context.metadata.quotedContext) window.App.quote?.set?.(context.metadata.quotedContext);
    }
    if (Array.isArray(window.pendingAttachments) && context.attachments?.length) {
      window.pendingAttachments = context.attachments.map(item => ({ ...item }));
      window.renderAttachmentChips?.();
    }
  }

  function finishFailed(context, message, data = null) {
    if (!registry.isExecuting(context.runId)) return;
    const failedBeforeFanout = context.phase === "prepare";
    context.phase = "failed";
    context.error = { code: "run_failed", message: String(message || "The run failed.") };
    context.bookmark.status = "failed";
    if (isUsageLimit(data || {}, message)) {
      updateUsage(context, data || {});
      if (registry.isVisible(context.runId)) {
        window.App.usageLimit?.show?.({ data: data || {}, source: "run", phase: "prepare" });
      }
    }
    if (failedBeforeFanout) restoreComposerAfterUnsentRun(context);
    releaseUnusedUsage(context);
    clearSensitiveRunData(context);
    registry.setStatus(context.runId, "failed", context.error);
    trackAppEvent("app_query_completed", { status: "error", selected_models: context.progress.totalModels });
  }

  async function executeRun(context) {
    const signal = context.controllers.query.signal;
    try {
      let idToken = null;
      try { idToken = await context.auth.user?.getIdToken?.(); } catch (_) {}
      if (!runIsCurrent(context)) return;
      if (!idToken) {
        finishFailed(context, "Please log in before sending a question.");
        return;
      }

      const preparePayload = {
        question: context.question,
        system_prompt: context.config.systemPrompt,
        deep_search: context.config.deepSearch,
        mode: context.mode,
        useOwnKeys: context.config.useOwnKeys,
        id_token: idToken
      };
      if (context.usage.key) preparePayload.usage_run_key = context.usage.key;
      if (context.previousExchange && !context.basis?.chatId) {
        preparePayload.context = {
          previous_question: context.previousExchange.question,
          previous_consensus: context.previousExchange.consensus
        };
      }

      const prepared = await prepareWithRetry(preparePayload, signal);
      if (!runIsCurrent(context)) return;
      updateUsage(context, prepared.data || {});
      if (!prepared.response.ok && isUsageStorageBusy(prepared.data)) {
        finishFailed(context, apiErrorMessage(prepared.data, "The usage service is busy right now. Please try again."), prepared.data);
        if (registry.isVisible(context.runId)) window.App.usageLimit?.showTemporaryStorageBusy?.();
        return;
      }
      if (!prepared.response.ok && isUsageLimit(prepared.data)) {
        context.usage.status = prepared.data?.usage_run_status || context.usage.status;
        finishFailed(context, apiErrorMessage(prepared.data, "Usage limit reached."), prepared.data);
        return;
      }
      const effectivePrompt = prepared.response.ok && prepared.data?.system_prompt
        ? prepared.data.system_prompt
        : context.config.systemPrompt;

      context.chatSession.beginRun({
        question: context.question,
        mode: context.mode,
        deepSearch: context.config.deepSearch,
        selectedModels: context.config.providers.map(provider => provider.modelId),
        consensusModel: context.config.consensusModel,
        isFollowup: Boolean(context.previousExchange),
        prepareSucceeded: prepared.response.ok,
        useOwnKeys: context.config.useOwnKeys,
        usageRunKey: context.usage.key,
        attachments: context.attachmentMeta
      });

      if (context.previousExchange && context.basis?.chatId) {
        const turn = await context.chatSession.ensurePendingTurn({
          idToken,
          question: context.question,
          consensusModel: context.config.consensusModel,
          signal
        });
        if (!turn) throw new Error(
          context.chatSession.lastPersistenceError || "The conversation turn could not be prepared. Please retry."
        );
        const memoryKey = context.config.useOwnKeys
          ? context.credentials[context.config.memoryKeyName] || ""
          : "";
        await context.chatSession.ensureContext({
          idToken,
          useOwnKeys: context.config.useOwnKeys,
          usageRunKey: context.usage.key,
          memoryApiKey: memoryKey,
          signal
        });
        if (!context.chatSession.contextBinding()) throw new Error("The conversation context could not be bound.");
      }
      if (!runIsCurrent(context)) return;

      context.phase = "answers";
      registry.update(context.runId, () => {});
      await Promise.all(context.config.providers.map(provider => runProvider(context, provider, idToken, effectivePrompt)));
      if (!registry.isExecuting(context.runId)) return;
      context.controllers.query = null;

      const successes = Object.values(context.modelResults)
        .filter(result => result.status === "complete" && String(result.text || "").trim()).length;
      if (successes === 0) {
        window.App.reportCriticalError?.({
          type: "run_failed",
          phase: "model_fanout",
          message: "All selected model requests failed.",
          details: `${context.progress.totalModels} providers failed`
        });
        if (context.chatSession.pendingTurnId) {
          await window.App.executeConsensusRun?.(context, {
            trigger: "disposition",
            dispositionOnly: true
          });
          if (registry.isExecuting(context.runId)) {
            context.keepConversationLock = true;
            finishFailed(context, "All selected model requests failed.");
          }
          return;
        }
        finishFailed(context, "All selected model requests failed.");
        return;
      }

      if (context.config.agentMode && context.config.autoConsensus) {
        if (successes < 2 && context.chatSession.pendingTurnId) {
          await window.App.executeConsensusRun?.(context, { trigger: "disposition", dispositionOnly: true });
          if (registry.isExecuting(context.runId)) finishFailed(context, "At least two completed model answers are required.");
          return;
        }
        if (successes < 2) {
          finishFailed(context, "At least two model responses are required for a consensus.");
          return;
        }
        await window.App.executeConsensusRun?.(context, { trigger: "auto" });
        return;
      }

      if (context.config.agentMode) {
        // Models are complete, but this run still owns a potential manual
        // synthesis (and, for an authoritative follow-up, its pending turn).
        // Keep it active so the conversation lock and frozen credentials stay
        // attached to exactly this run until Consensus or Cancel resolves it.
        context.phase = "answers_ready";
        context.bookmark.status = "waiting";
        registry.update(context.runId, () => {});
        trackAppEvent("app_query_completed", { status: "answers_ready", selected_models: context.progress.totalModels });
        return;
      }

      context.phase = "done";
      context.bookmark.status = "succeeded";
      clearSensitiveRunData(context);
      registry.setStatus(context.runId, "succeeded");
      trackAppEvent("app_query_completed", { status: "success", selected_models: context.progress.totalModels });
    } catch (error) {
      if (isAbortError(error) || !registry.isExecuting(context.runId)) return;
      context.chatSession?.markPendingUncertain?.();
      if (context.chatSession?.pendingTurnId || context.chatSession?.hasUncertainTurn?.()) {
        context.keepConversationLock = true;
      }
      finishFailed(context, error?.message || "The comparison could not be completed.");
      if (registry.isVisible(context.runId)) window.App.showPopup?.(error?.message || "The comparison could not be completed.");
    }
  }

  function validateOwnKeys(providers, memoryKeyName) {
    const required = new Set(providers.map(provider => provider.keyName));
    if (memoryKeyName) required.add(memoryKeyName);
    return Array.from(required).filter(keyName => !String(localStorage.getItem(keyName) || "").trim());
  }

  function beginContext(question, draftQuestion, quotedContext) {
    const agentMode = window.isAgentModeEnabled?.() === true;
    const basis = registry.getSelectedConversationBasis();
    const followup = Boolean(agentMode && basis && basis.question && basis.consensus && !basis.continuationUnavailable);
    const deepSearch = document.getElementById("deepSearchToggle")?.checked === true;
    const useOwnKeys = document.getElementById("useOwnKeysSwitch")?.checked === true;
    const attachments = window.getAttachmentsPayload?.() || [];
    const attachmentMeta = attachments.map(item => ({ name: item.name, mime: item.mime, size: item.size || 0 }));
    if (attachments.length && !window.isUserPlus) {
      if (!window.App.showProFeatureModal?.("File uploads")) {
        window.App.showPopup?.("File uploads are off here. Remove the attachments to continue.");
      }
      return null;
    }
    const providers = selectedProviders(attachments.length, deepSearch);
    if (providers.length < 2) {
      window.App.showPopup?.("Choose at least two compatible models. Remove the attachment or select another model.");
      return null;
    }

    const consensusSelect = document.getElementById("consensusModelDropdown");
    const consensusModel = String(consensusSelect?.value || "").trim();
    const consensusModelLabel = optionLabel(consensusSelect);
    const memoryProvider = String(consensusSelect?.selectedOptions?.[0]?.dataset?.engineProvider || "").trim().toLowerCase();
    const memoryKeyName = "openrouterKey";
    if (useOwnKeys) {
      const missing = validateOwnKeys(providers, memoryKeyName);
      if (missing.length) {
        window.App.showPopup?.("Add your OpenRouter API key before sending.");
        return null;
      }
    }

    const authUser = window.auth?.currentUser || null;
    if (!authUser) {
      window.App.showPopup?.("Please log in before sending a question.");
      return null;
    }

    const chatSession = window.App.createChatSession?.({
      activeChatId: basis?.chatId,
      activeTurnId: basis?.turnId
    });
    const usage = createUsage(deepSearch, useOwnKeys);
    const config = {
      agentMode,
      autoConsensus: agentMode && document.getElementById("autoConsensusToggle")?.checked !== false,
      deepSearch,
      useOwnKeys,
      providers: providers.map(provider => ({ ...provider })),
      consensusModel,
      consensusModelLabel,
      memoryProvider,
      memoryKeyName,
      systemPrompt: currentSystemPrompt()
    };

    let context;
    try {
      context = registry.create({
        question,
        mode: getActiveMode(),
        config,
        basis,
        followup,
        bookmarkId: followup ? basis.bookmarkId : "",
        bookmarkTitle: followup ? basis.title : question,
        chatSession,
        usage,
        attachments,
        attachmentMeta,
        metadata: { draftQuestion, quotedContext }
      });
    } catch (error) {
      if (["parallel_limit", "conversation_busy", "conversation_uncertain", "bookmark_deleting"].includes(error?.code)) {
        window.App.showPopup?.(error.message);
        trackAppEvent("app_query_blocked", { reason: error.code });
        return null;
      }
      throw error;
    }

    context.bookmark.id = context.bookmark.id || bookmarkIdForRun(context.runId);
    context.credentials = Object.fromEntries(
      ["openrouterKey"].map(key => [key, String(localStorage.getItem(key) || "")])
    );
    config.providers.forEach(provider => {
      context.modelResults[provider.provider] = {
        provider: provider.provider,
        modelId: provider.modelId,
        modelLabel: provider.modelLabel,
        boxId: provider.boxId,
        status: "pending",
        streamText: "",
        text: "",
        sources: [],
        rawSources: [],
        error: null
      };
    });
    context.progress.totalModels = config.providers.length;
    context.controllers.query = new AbortController();
    if (followup && basis?.currentTurn) {
      const turnId = String(basis.currentTurn.turn_id || basis.currentTurn.id || "");
      const alreadyStored = context.historyTurns.some(turn => String(turn?.turn_id || turn?.id || "") === turnId && turnId);
      if (!alreadyStored) context.historyTurns.push(basis.currentTurn);
    }
    context.cancelHook = (reason) => {
      Object.values(context.modelResults).forEach(result => {
        if (["pending", "reasoning", "streaming"].includes(result.status)) {
          result.status = "canceled";
          result.error = "Request canceled.";
        }
      });
      if (["pending", "streaming", "differences"].includes(context.consensus.status)) {
        context.consensus.status = "canceled";
        context.consensus.error = { message: "Request canceled." };
      }
      context.chatSession?.handleConsensusCancelled?.();
      if (context.chatSession?.pendingTurnId || context.chatSession?.hasUncertainTurn?.()) {
        context.keepConversationLock = true;
      }
      context.bookmark.status = "canceled";
      if (registry.isVisible(context.runId) && context.phase === "prepare") {
        restoreComposerAfterUnsentRun(context);
      }
      clearSensitiveRunData(context);
      releaseUnusedUsage(context, { allowAfterLogout: reason === "logout" });
    };

    registry.update(context.runId, () => {});
    registry.setStatus(context.runId, "running");
    return context;
  }

  function handComposerToRun(context) {
    window.App.clearQuestionDraft?.();
    const input = document.getElementById("questionInput");
    if (input) {
      input.value = "";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      window.syncDemoChipState?.();
    }
    window.App.quote?.clear?.();
    const sentAttachments = window.App.attachments?.detachForMessage?.() || context.attachmentMeta;
    context.attachmentMeta = sentAttachments.map(item => ({ name: item.name, mime: item.mime, size: item.size || 0 }));
    window.App.composer?.collapse?.({ force: true });
    registry.renderVisible();
    // This is tied to the user's send gesture. Background projections never
    // invoke it, so a late run cannot steal the reader's scroll position.
    window.App.revealSentMessage?.();
  }

  window.sendQuestion = async function () {
    const visible = registry.visible();
    if (visible && registry.isExecuting(visible.runId)) {
      // The first rapid repeat is a duplicate gesture, not an intentional stop.
      if (Date.now() - visible.startedAt < 450) return;
      registry.cancel(visible.runId, "user");
      trackAppEvent("app_query_canceled");
      return;
    }
    if (!registry.claimStartAction()) return;

    const selectedCount = window.App.getSelectedModelCount?.() || 0;
    if (selectedCount < 2) {
      window.updateQuestionInputAccess?.();
      trackAppEvent("app_query_blocked", { reason: "minimum_models", selected_models: selectedCount });
      return;
    }
    if (window.updateQuestionInputAccess && !window.updateQuestionInputAccess()) return;
    if (window.validateInputText && !window.validateInputText()) return;

    const draftQuestion = document.getElementById("questionInput")?.value || "";
    const quotedContext = window.App.quote?.text?.() || "";
    const question = window.App.quote?.compose?.(draftQuestion) ?? draftQuestion;
    if (!String(question).trim()) return;

    if (isDemoQuery(question) && !registry.getSelectedConversationBasis()) {
      window.App.state.set("lastQuestion", question, "run");
      await window.runDemoFlow?.(question);
      return;
    }

    const useOwnKeys = document.getElementById("useOwnKeysSwitch")?.checked === true;
    const deepThink = document.getElementById("deepSearchToggle")?.checked === true;
    if (window.App.usageLimit?.blockIfExhausted?.({ useOwnKeys, deepThink, source: "send" })) {
      trackAppEvent("app_query_blocked", { reason: "usage_limit" });
      return;
    }
    window.App.usageLimit?.hide?.();

    const context = beginContext(String(question).trim(), draftQuestion, quotedContext);
    if (!context) return;
    handComposerToRun(context);
    trackAppEvent("app_query_started", {
      mode: context.mode,
      selected_models: context.progress.totalModels,
      custom_credentials: context.config.useOwnKeys,
      logged_in: true,
      agent_mode: context.config.agentMode,
      auto_consensus: context.config.autoConsensus
    });
    await executeRun(context);
  };

  window.cancelCurrentQuery = function (runId = null) {
    const target = runId || registry.visible()?.runId;
    if (!target) return false;
    const context = registry.get(target);
    if (!context || !registry.isExecuting(target)) return false;
    if (!["prepare", "answers"].includes(context.phase)) return false;
    const canceled = registry.cancel(target, "user");
    if (canceled) trackAppEvent("app_query_canceled");
    return canceled;
  };

  window.App.skipModel = function (boxId, runId = null) {
    const context = registry.get(runId || registry.visible()?.runId);
    if (!context || !registry.isExecuting(context.runId) || context.phase !== "answers") return false;
    const result = Object.values(context.modelResults).find(item => item.boxId === boxId);
    if (!result || !["pending", "reasoning", "streaming"].includes(result.status)) return false;
    result.status = "skipped";
    result.error = "Skipped — this model took too long, so the run went on without it.";
    context.controllers.providers.get(result.provider)?.abort?.();
    context.progress.completedModels += 1;
    context.progress.failedModels += 1;
    registry.update(context.runId, () => {});
    trackAppEvent("app_model_skipped", { model: result.provider });
    return true;
  };

  window.isQueryRequestRunning = function (runId = null) {
    const context = registry.get(runId || registry.visible()?.runId);
    return Boolean(context && registry.isExecuting(context.runId) && ["prepare", "answers"].includes(context.phase));
  };

  function isRunActive() {
    const context = registry.visible();
    return Boolean(context && registry.isExecuting(context.runId));
  }

  function setSendButtonRunning(running) {
    const button = document.getElementById("sendButton");
    if (!button) return;
    button.disabled = false;
    button.classList.toggle("is-cancel-action", running);
    button.title = running ? "Cancel this run" : "Send question";
    button.setAttribute("aria-label", button.title);
    button.innerHTML = running
      ? '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="8" y="8" width="8" height="8" rx="1.3"></rect></svg>'
      : '<svg viewBox="0 0 24 24" aria-hidden="true" style="width:13px;height:13px"><path d="M3.4 3.6 21 12 3.4 20.4 7 12z" fill="currentColor"></path></svg>';
    if (!running) window.updateQuestionInputAccess?.();
  }

  function syncSendButtonRunning() {
    setSendButtonRunning(isRunActive());
  }

  window.isRunActive = isRunActive;
  window.App.syncSendButtonRunning = syncSendButtonRunning;
})();
