// =====================================================================
// consensus-lifecycle.js
// Consensus lifecycle only: visibility, gate/availability, run state,
// abort/cancel, and Auto-Consensus toggle persistence.
// The consensus request/payload/rendering stays in templates/index.html.
//
// Exports keep the existing window contracts and add the narrow
// window.App.consensusLifecycle bridge used by the remaining run code.
// =====================================================================

(function () {
  window.App = window.App || {};

  let currentConsensusController = null;
  let currentConsensusRunId = 0;
  let consensusRequestRunning = false;
  let consensusRevealTimer = null;

  function trackAppEvent(eventName, eventData) {
    if (window.App && typeof window.App.trackAppEvent === "function") {
      window.App.trackAppEvent(eventName, eventData);
    } else if (typeof window.trackUmamiEvent === "function") {
      window.trackUmamiEvent(eventName, eventData || {});
    }
  }

  function revealConsensusOutput() {
    const consensusOutputEl = document.getElementById("consensusOutput");
    if (!consensusOutputEl) return;
    if (consensusRevealTimer) {
      clearTimeout(consensusRevealTimer);
      consensusRevealTimer = null;
    }
    consensusOutputEl.classList.remove("is-hidden");
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        consensusOutputEl.classList.add("is-visible");
        // Konsens lebt OBERHALB der Antwortboxen: Wer beim Reveal weiter
        // unten liest, wird sanft dorthin geholt (scroll-margin-top in CSS
        // hält Abstand zur Float-Nav). Nur scrollen, wenn nötig.
        const rect = consensusOutputEl.getBoundingClientRect();
        if (rect.top < 0 || rect.top > window.innerHeight * 0.65) {
          consensusOutputEl.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
    // Demo-Pfad ruft reveal direkt (ohne startRun): Pipeline auf Stufe 2.
    window.App?.consensusPipeline?.onConsensusStart?.();
  }

  function hideConsensusOutput() {
    const consensusOutputEl = document.getElementById("consensusOutput");
    if (!consensusOutputEl) return;
    consensusOutputEl.classList.remove("is-visible");
    if (consensusRevealTimer) clearTimeout(consensusRevealTimer);
    consensusRevealTimer = setTimeout(() => {
      consensusOutputEl.classList.add("is-hidden");
    }, 280);
  }

  function getCompletedIncludedAnswerCount() {
    const boxIds = [
      "openaiResponse",
      "mistralResponse",
      "claudeResponse",
      "geminiResponse",
      "deepseekResponse",
      "grokResponse"
    ];

    return boxIds.filter(id => {
      const box = document.getElementById(id);
      if (!box || box.classList.contains("excluded")) return false;
      if (box.dataset.responseError === "true") return false;
      const contentEl = box.querySelector(".collapsible-content");
      if (!contentEl || contentEl.querySelector(".thinking-wrap") || contentEl.classList.contains("is-streaming")) return false;
      // textContent, nicht innerText: die Antwortboxen liegen jetzt in beiden
      // Modi hinter dem "Compare answers"-Schalter, und innerText liefert
      // fuer ein display:none-Element den leeren String — die vorhandene
      // Antwort waere damit unsichtbar fuer die Zaehlung.
      const text = contentEl.textContent.trim();
      return text && text !== "Request canceled.";
    }).length;
  }

  function canGenerateConsensus() {
    if (consensusRequestRunning) return true;

    const agentModeWaitingForResponses =
      typeof window.isAgentModeEnabled === "function"
      && window.isAgentModeEnabled()
      && window.isAgentModeRunning();

    if (agentModeWaitingForResponses) return false;

    return getCompletedIncludedAnswerCount() >= 2;
  }

  function updateConsensusButtonAvailability() {
    return canGenerateConsensus();
  }

  function setSynthesizing(isSynthesizing) {
    document
      .getElementById("consensusResponse")
      ?.classList.toggle("is-synthesizing", !!isSynthesizing);
    // Der Differences-Bereich bleibt waehrend der Synthese ZU: sein Spinner
    // ist entfallen, ein offenes leeres Panel waere nur ein Loch unter der
    // Antwort. Der gefuehrte Lauf sagt an, dass geprueft wird; das Panel
    // meldet sich erst, wenn es Karten zu zeigen hat.
    if (isSynthesizing) window.App?.differencesPanel?.setSynthesizing?.();
  }

  function startRun() {
    currentConsensusRunId++;
    currentConsensusController = new AbortController();
    consensusRequestRunning = true;
    // Der Send-Button bleibt Cancel, bis auch Consensus/Differences stehen.
    window.App?.syncSendButtonRunning?.();
    window.App?.consensusPipeline?.onConsensusStart?.();
    return {
      runId: currentConsensusRunId,
      signal: currentConsensusController.signal
    };
  }

  function isActiveRun(runId) {
    return consensusRequestRunning
      && runId === currentConsensusRunId
      && currentConsensusController
      && !currentConsensusController.signal.aborted;
  }

  function finishRun(runId) {
    if (runId !== currentConsensusRunId) return;
    consensusRequestRunning = false;
    currentConsensusController = null;
    setSynthesizing(false);
    window.App?.syncSendButtonRunning?.();
    window.App?.consensusPipeline?.onConsensusEnd?.();
  }

  function isRunning() {
    return consensusRequestRunning;
  }

  function markPendingCanceled() {
    const consensusDiv = document.getElementById("consensusResponse");
    if (!consensusDiv) return;

    const mainEl = window.App.consensusBodyEl(consensusDiv);
    const diffEl = consensusDiv.querySelector(".consensus-differences p");
    if (mainEl && (mainEl.querySelector(".thinking-wrap") || mainEl.classList.contains("is-streaming"))) {
      mainEl.classList.remove("is-streaming");
      mainEl.innerText = "Request canceled.";
    }
    if (diffEl && (diffEl.querySelector(".thinking-wrap") || diffEl.classList.contains("is-streaming"))) {
      diffEl.classList.remove("is-streaming");
      diffEl.innerText = "";
    }
    if (window.resetCredibilityFrame) {
      window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
    }
  }

  function cancelCurrentConsensus() {
    if (!consensusRequestRunning || !currentConsensusController) return;
    const runId = currentConsensusRunId;
    currentConsensusController.abort();
    markPendingCanceled();
    // Abbruch: Pipeline sofort weg (vor finishRun, sonst blitzt "done" auf).
    window.App?.consensusPipeline?.dismiss?.();
    finishRun(runId);
    trackAppEvent("app_consensus_canceled");
  }

  function initAutoConsensusToggle() {
    const autoConsensusToggle = document.getElementById("autoConsensusToggle");
    if (!autoConsensusToggle) return;

    const storedAutoConsensus = localStorage.getItem("autoConsensus");
    if (storedAutoConsensus === null) {
      autoConsensusToggle.checked = true;
      localStorage.setItem("autoConsensus", "true");
    } else {
      autoConsensusToggle.checked = storedAutoConsensus === "true";
    }

    autoConsensusToggle.addEventListener("change", function () {
      localStorage.setItem("autoConsensus", this.checked);
      trackAppEvent("app_auto_consensus_changed", { enabled: this.checked });
      if (typeof window.showMobileInfoPopup === "function") {
        window.showMobileInfoPopup("Auto Consensus automatically generates a consensus after the model responses.");
      }
    });
  }

  window.revealConsensusOutput = revealConsensusOutput;
  window.hideConsensusOutput = hideConsensusOutput;
  window.canGenerateConsensus = canGenerateConsensus;
  window.updateConsensusButtonAvailability = updateConsensusButtonAvailability;
  window.cancelCurrentConsensus = cancelCurrentConsensus;

  window.App.consensusLifecycle = {
    initAutoConsensusToggle,
    startRun,
    isActiveRun,
    finishRun,
    setSynthesizing,
    markPendingCanceled,
    isRunning
  };
})();
