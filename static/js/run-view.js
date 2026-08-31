// Projects the selected RunContext into the one main application DOM.
// Background callbacks only mutate their context; this adapter is the only
// place that mirrors run-owned data into compatibility globals and elements.
(function () {
  "use strict";

  window.App = window.App || {};
  const registry = window.App.runRegistry;
  if (!registry) return;

  // Familien aus der einen Frontend-Quelle (window.App.modelPrefs).
  function providers() {
    return (window.App.modelPrefs || []).map(pref => ({
      provider: pref.key,
      boxId: pref.responseId,
      textId: pref.textId
    }));
  }

  let projectedRunId = null;
  let projectedPhase = null;
  let projectedStatus = null;

  function selectedConfig(context, provider) {
    return (context.config?.providers || []).find(item => item.provider === provider) || null;
  }

  function renderModel(context, definition) {
    const box = document.getElementById(definition.boxId);
    const output = box?.querySelector(".collapsible-content");
    if (!box || !output) return;

    const config = selectedConfig(context, definition.provider);
    const result = context.modelResults[definition.provider] || null;
    const selected = Boolean(config);
    box.classList.toggle("excluded", !selected);
    delete box.dataset.responseError;
    delete box.dataset.responseSkipped;
    delete box.dataset.consensusAnswer;
    delete box.dataset.consensusSources;
    output.classList.remove("is-streaming");

    const modelText = document.getElementById(definition.textId);
    if (modelText && config?.modelLabel) modelText.textContent = config.modelLabel;

    if (!selected) {
      box.dataset.responseState = "idle";
      output.replaceChildren();
      return;
    }

    const status = result?.status || "pending";
    box.dataset.responseState = status === "streaming" ? "pending" : status;
    if (status === "error" || status === "skipped" || status === "canceled") {
      box.dataset.responseError = "true";
      if (status === "skipped") box.dataset.responseSkipped = "true";
      output.textContent = result?.error || (status === "canceled" ? "Request canceled." : "The model request failed.");
      return;
    }

    const markdown = String(result?.text || result?.streamText || "");
    if (markdown) {
      box.dataset.consensusAnswer = markdown;
      box.dataset.consensusSources = JSON.stringify(result?.sources || []);
      window.injectMarkdown?.(output, markdown, context.evidenceSources);
      if (status === "streaming") output.classList.add("is-streaming");
      return;
    }

    if (["pending", "reasoning", "streaming"].includes(status)) {
      output.innerHTML = window.spinnerHTML || "";
      if (status === "reasoning") {
        const label = output.querySelector(".thinking.typing-indicator");
        if (label) {
          label.dataset.text = "Reasoning";
          label.setAttribute("aria-label", "Reasoning");
        }
      }
      return;
    }
    output.replaceChildren();
  }

  function setConsensusVisible(visible) {
    const output = document.getElementById("consensusOutput");
    if (!output) return;
    output.classList.toggle("is-hidden", !visible);
    output.classList.toggle("is-visible", visible);
  }

  function renderConsensus(context) {
    const response = document.getElementById("consensusResponse");
    const body = window.App.consensusBodyEl?.(response);
    const differences = response?.querySelector(".consensus-differences p");
    if (!response || !body || !differences) return;

    window.resetConsensusInsights?.();
    window.resetCredibilityFrame?.(response.querySelector(".consensus-differences"));
    body.classList.remove("is-streaming");
    differences.classList.remove("is-streaming");

    const state = context.consensus;
    const text = String(state.text || state.streamText || "");
    const visible = Boolean(text) || ["pending", "streaming", "differences", "complete", "error"].includes(state.status);
    setConsensusVisible(visible);
    if (!visible) {
      body.replaceChildren();
      differences.replaceChildren();
      return;
    }

    if (text) {
      window.injectMarkdown?.(body, text, context.evidenceSources);
      if (state.status === "streaming") body.classList.add("is-streaming");
    } else if (["pending", "streaming"].includes(state.status)) {
      body.innerHTML = window.consensusSpinnerHTML || window.spinnerHTML || "";
    } else if (state.error) {
      body.textContent = String(state.error.message || state.error || "The consensus could not be completed.");
    } else {
      body.replaceChildren();
    }

    if (!state.text) {
      differences.replaceChildren();
      return;
    }

    const successful = Object.values(context.modelResults)
      .filter(result => result?.status === "complete" && String(result.text || "").trim()).length;
    let structured = false;
    if (state.differencesData && typeof state.differencesData === "object") {
      try {
        structured = window.renderConsensusInsights?.(state.differencesData, successful, {
          sources: context.evidenceSources || []
        }) === true;
      } catch (error) {
        console.error("Could not render run differences:", error);
        window.resetConsensusInsights?.();
      }
    }
    if (!structured) {
      const fallback = String(state.differences || "");
      if (fallback) {
        window.App.differencesPanel?.expandForFallback?.();
        const framed = window.colorizeCredibility ? window.colorizeCredibility(fallback) : fallback;
        window.injectMarkdown?.(differences, framed, context.evidenceSources);
      } else {
        differences.replaceChildren();
      }
    }
  }

  function syncCompatibilityState(context) {
    window.App.state.set("lastQuestion", context.question, "run");
    window.App.state.set("currentEvidenceSources", context.evidenceSources || [], "evidence");
    window.App.state.set("consensusCitationMeta", context.consensus.citationMeta || null, "consensus");
    window.App.state.set("lastShareResultId", context.consensus.resultId || null, "share");
    window.lastQuestionAttachmentsMeta = context.attachmentMeta || [];
    window.lastConsensusBookmarkPayload = context.consensus.bookmarkPayload || null;
    window.App.bookmarkSession?.restore?.(context.bookmark.id);

    window.App.chatSession?.reset?.();
    const completed = context.completedBasis;
    if (completed?.chatId && completed?.turnId) {
      window.App.chatSession?.restoreCompletedChat?.(completed.chatId, completed.turnId);
    }
  }

  function syncConversationProjection(context) {
    const followup = window.App.followup;
    followup?.renderStoredTurns?.(context.historyTurns || []);
    followup?.reset?.();
    if (context.status === "succeeded" && context.consensus.completedTurn && context.consensus.text) {
      followup?.offer?.(context.question, context.consensus.text, context.consensus.completedTurn);
    }
  }

  // What the finished run is made of, read from the run itself rather than
  // from the boxes on screen: the same run may be projected again long after
  // another view has written over them.
  function runFacts(context) {
    const models = Object.values(context.modelResults)
      .filter(result => result?.status === "complete" && String(result.text || "").trim()).length;
    if (!models) return null;
    const finishedAt = Number(context.finishedAt) || Date.now();
    const startedAt = Number(context.startedAt) || Number(context.createdAt) || finishedAt;
    return { models, durationMs: Math.max(0, finishedAt - startedAt) };
  }

  function syncPipeline(context, force) {
    const pipeline = window.App.consensusPipeline;
    if (!pipeline) return;
    // Ein Direktvergleich hat keinen gefuehrten Lauf: sechs Antworten, sonst
    // nichts. Der Fortschrittsblock kuendigt dort Schritte an, die nie kommen
    // (Consensus, Widerspruchspruefung), waehrend die Antworten daneben schon
    // sichtbar streamen.
    if (context.config?.agentMode === false) {
      pipeline.dismiss?.();
      return;
    }
    const phaseChanged = force || projectedPhase !== context.phase || projectedStatus !== context.status;
    if (!phaseChanged) {
      pipeline.renderProvenance?.();
      return;
    }

    pipeline.dismiss?.();
    if (context.status === "running" || context.status === "starting") {
      pipeline.onPrepare?.(context.startedAt);
      if (["answers", "consensus", "differences"].includes(context.phase)) {
        pipeline.onQueryStatus?.("running");
      }
      if (["consensus", "differences"].includes(context.phase)) {
        pipeline.onQueryStatus?.("complete");
        pipeline.onConsensusStart?.();
      }
      if (context.phase === "differences") pipeline.onDifferencesStart?.();
    } else if (context.status === "succeeded") {
      pipeline.setRunFacts?.(runFacts(context));
      pipeline.onConsensusEnd?.();
    } else {
      pipeline.setRunFacts?.(null);
    }
  }

  function syncAgentStatus(context, force) {
    if (context.config?.agentMode === false || typeof window.setAgentModeStatus !== "function") return;
    if (typeof window.projectAgentModeRun === "function") {
      window.projectAgentModeRun(context);
      return;
    }
    if (!force && projectedPhase === context.phase && projectedStatus === context.status) return;
    if (context.status === "failed") {
      window.setAgentModeStatus("error", context.error?.message || context.consensus?.error?.message || "The run failed.");
    } else if (context.status === "canceled") {
      window.setAgentModeStatus("canceled");
    } else if (context.status === "succeeded" || context.phase === "answers_ready") {
      window.setAgentModeStatus("complete");
    } else {
      window.setAgentModeStatus("running");
    }
  }

  // A projection is a one-way DOM write, and several of the surfaces it
  // writes into report back: the tier sync rebuilds the model picker, and the
  // picker asks the registry to render the visible run again. That is a
  // cycle -- project -> updateUserTierUI -> restoreModelSelections ->
  // setModelSelectionState -> renderVisible -> project -- and it ended in
  // "Maximum call stack size exceeded", which the run then reported as a
  // failure right after /prepare. A render requested while one is already
  // running is by definition redundant: it would read the same context and
  // write the same DOM, and everything the nested caller changed is
  // re-asserted from the run further down this same pass.
  let projecting = false;

  function project(context) {
    if (projecting) return;
    projecting = true;
    try {
      renderProjection(context);
    } finally {
      projecting = false;
    }
  }

  function renderProjection(context) {
    if (!context) {
      projectedRunId = null;
      projectedPhase = null;
      projectedStatus = null;
      window.projectAgentModeRun?.(null);
      // Nothing on screen belongs to a run any more: a saved bookmark, or a
      // cleared view. A run still going in the background reports itself in
      // its sidebar row, never through the guided-run block above the
      // composer -- that block would otherwise keep counting inside whatever
      // was opened instead.
      window.App.consensusPipeline?.detach?.();
      window.App.syncSendButtonRunning?.();
      return;
    }
    const forcePipeline = projectedRunId !== context.runId;
    projectedRunId = context.runId;

    // Only when the run actually disagrees with the tier on screen. Pushing
    // it on every pass rebuilt the whole model picker for each streamed
    // chunk, and the picker's own restore is what closed the cycle above.
    // "is_pro_user: false" heisst seit der Plus-Stufe nur noch "nicht Pro" --
    // es unterscheidet Free nicht mehr von Plus. Ein Lauf ohne ausdrueckliches
    // "tier" (aeltere Antwort, Tab von vor einem Deploy) darf die Stufe auf dem
    // Schirm deshalb nur noch ANHEBEN, nie senken: sonst schaltet ein
    // Plus-Konto beim ersten projizierten Lauf zurueck auf Free und verliert
    // Anhaenge und Resolve.
    const runTier = context.usage?.tier
      ?? (context.usage?.isProUser === true ? "pro" : null);
    if (runTier !== null
      && (window.App.normalizeTier?.(runTier) || "free") !== (window.userTier || "free")) {
      window.updateUserTierUI?.(runTier, true);
    }

    if (context.config?.agentMode === false) {
      window.enterDirectComparisonView?.();
    } else {
      window.exitHeroMode?.();
      document.body.classList.remove("direct-comparison-active");
    }

    window.App.setAppTitle?.(context.question);
    // Auch der Direktvergleich zeigt die Frage: er fuehrt keinen Thread, aber
    // ohne sie steht auf dem Schirm nur noch die Antwort auf etwas, das
    // nirgends mehr geschrieben steht.
    window.App.setThreadQuestion?.(context.question);
    window.App.setThreadQuestionAttachments?.(context.attachmentMeta || []);
    syncConversationProjection(context);
    providers().forEach(provider => renderModel(context, provider));
    window.renderEvidenceSources?.(context.evidenceSources || []);
    renderConsensus(context);
    syncCompatibilityState(context);
    if (context.config?.agentMode === false) window.projectAgentModeRun?.(context);
    else syncAgentStatus(context, forcePipeline);
    syncPipeline(context, forcePipeline);
    window.updateAgentModeUI?.();
    // The controls describe the next run; these body classes describe the
    // selected result. Re-assert the frozen run mode after the control UI has
    // synchronized itself from localStorage.
    document.body.classList.toggle("agent-mode-enabled", context.config?.agentMode !== false);
    document.body.classList.toggle(
      "agent-mode-running",
      context.config?.agentMode !== false && registry.isExecuting(context.runId)
    );
    window.syncHeroResponseAccess?.();
    window.App.syncSendButtonRunning?.();

    projectedPhase = context.phase;
    projectedStatus = context.status;
  }

  function statusLabel(context) {
    if (context.status === "succeeded") return "Completed";
    if (context.status === "failed") return "Failed";
    if (context.status === "canceled") return "Canceled";
    if (context.phase === "answers_ready") return "Answers ready";
    if (context.phase === "prepare") return "Preparing";
    if (context.phase === "answers") return "Models answering";
    if (context.phase === "consensus") return "Writing consensus";
    if (context.phase === "differences") return "Checking differences";
    return "Running";
  }

  function ensureRunRow(context) {
    if (!context?.bookmark?.id || !context.auth?.uid) return;
    if (context.bookmark.deleted) return;
    if (context.bookmark.uiReady) return;
    const container = document.getElementById("bookmarksContainer");
    if (!container) return;
    let row = document.querySelector(`.bookmark[data-run-id="${context.runId}"]`);
    if (!row) {
      row = document.createElement("div");
      row.className = "bookmark run-entry";
      row.dataset.runId = context.runId;
      row.dataset.id = context.bookmark.id;
      row.setAttribute("role", "button");
      row.setAttribute("tabindex", "0");
      row.addEventListener("click", () => registry.show(context.runId));
      row.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          registry.show(context.runId);
        }
      });
      const savedRow = document.querySelector(
        `.bookmark:not(.run-entry)[data-id="${context.bookmark.id}"]`
      );
      if (savedRow) savedRow.replaceWith(row);
      else container.prepend(row);
    }

    row.className = `bookmark run-entry run-status-${context.status}`;
    if (context.status === "starting" || context.status === "running") row.classList.add("is-pending");
    row.setAttribute("aria-label", `${context.bookmark.title || context.question}. ${statusLabel(context)}. Open run.`);
    row.title = `${statusLabel(context)} — open this run`;
    row.replaceChildren();
    const label = document.createElement("p");
    const words = String(context.bookmark.title || context.question || "New comparison").split(/\s+/);
    label.textContent = words.length > 5 ? `${words.slice(0, 5).join(" ")}...` : words.join(" ");
    const status = document.createElement("span");
    status.className = "run-entry-status";
    status.textContent = context.persistence?.errors?.length && context.status === "succeeded"
      ? `${statusLabel(context)} · not saved`
      : statusLabel(context);
    status.setAttribute("aria-hidden", "true");
    row.append(label, status);
  }

  window.addEventListener("consensio:run-registry-change", event => {
    const context = event.detail?.context;
    if (context) {
      ensureRunRow(context);
      if (context.status === "succeeded") window.App.bookmarkUi?.finalizeRun?.(context);
    }
  });

  window.App.runView = Object.freeze({ project, ensureRunRow });
  registry.setProjector(project);
})();
