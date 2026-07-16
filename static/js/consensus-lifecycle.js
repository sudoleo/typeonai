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

  function setGate(disabled) {
    const btn = document.getElementById("consensusButton");
    if (btn) btn.disabled = disabled;
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
      const text = contentEl.innerText.trim();
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
    const canGenerate = canGenerateConsensus();
    const consensusButton = document.getElementById("consensusButton");
    if (consensusButton && !consensusRequestRunning) {
      consensusButton.disabled = !canGenerate;
      consensusButton.title = canGenerate
        ? "Generate Consensus"
        : "Consensus requires at least two completed model answers";
      consensusButton.setAttribute("aria-label", consensusButton.title);
    }
    return canGenerate;
  }

  function setButtonRunning(isRunning) {
    const consensusButton = document.getElementById("consensusButton");
    if (!consensusButton) return;

    consensusButton.disabled = false;
    consensusButton.classList.toggle("is-cancel-action", isRunning);
    consensusButton.title = isRunning ? "Cancel consensus" : "Generate Consensus";
    consensusButton.setAttribute("aria-label", isRunning ? "Cancel consensus" : "Generate Consensus");
    consensusButton.textContent = isRunning ? "Stop Consensus generation" : "Generate Consensus";
    if (!isRunning && window.updateConsensusButtonAvailability) {
      window.updateConsensusButtonAvailability();
    }
  }

  function setSynthesizing(isSynthesizing) {
    document
      .getElementById("consensusResponse")
      ?.classList.toggle("is-synthesizing", !!isSynthesizing);
  }

  function startRun() {
    currentConsensusRunId++;
    currentConsensusController = new AbortController();
    consensusRequestRunning = true;
    setButtonRunning(true);
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
    setButtonRunning(false);
    window.App?.consensusPipeline?.onConsensusEnd?.();
  }

  function isRunning() {
    return consensusRequestRunning;
  }

  function markPendingCanceled() {
    const consensusDiv = document.getElementById("consensusResponse");
    if (!consensusDiv) return;

    const mainEl = consensusDiv.querySelector(".consensus-main p");
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
    setGate,
    startRun,
    isActiveRun,
    finishRun,
    setSynthesizing,
    markPendingCanceled,
    isRunning
  };
})();
