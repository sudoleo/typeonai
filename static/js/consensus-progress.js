// =====================================================================
// consensus-progress.js
// Compact, observational two-stage pipeline below the question input.
// It mirrors the regular query lifecycle without owning request state:
// model answers are determinate; synthesis is intentionally indeterminate.
// The Agent Mode panel remains the sole progress UI while Agent Mode is on.
// =====================================================================

(function () {
  window.App = window.App || {};

  const ANSWERS_END = 66;
  let stage = "idle";
  let hideTimer = null;
  let handoffTimer = null;

  function getElement() {
    return document.getElementById("consensusPipeline");
  }

  function getIncludedBoxes() {
    return Array.from(
      document.querySelectorAll(".response-section > .response-box")
    ).filter(box => !box.classList.contains("excluded"));
  }

  function getAnswerCounts() {
    const boxes = getIncludedBoxes();
    const done = boxes.filter(box => {
      const responseState = box.dataset.responseState;
      if (responseState === "complete" || responseState === "error") return true;
      if (responseState === "pending") return false;

      // Demo responses do not use data-response-state. Their streaming class
      // and thinking element still provide a reliable completion signal.
      const content = box.querySelector(".collapsible-content");
      return Boolean(
        content
        && !content.querySelector(".thinking-wrap")
        && !content.classList.contains("is-streaming")
        && content.innerText.trim()
      );
    }).length;
    return { done, total: boxes.length };
  }

  function progressFor(currentStage, counts) {
    if (currentStage === "answers") {
      const ratio = counts.total ? counts.done / counts.total : 0;
      return 4 + Math.round(ratio * (ANSWERS_END - 4));
    }
    if (currentStage === "answers-done" || currentStage === "handoff" || currentStage === "consensus") {
      return ANSWERS_END;
    }
    if (currentStage === "done") return 100;
    return 0;
  }

  function accessibleStatus(currentStage, counts) {
    if (currentStage === "answers") {
      return `Model answers: ${counts.done} of ${counts.total} complete. Consensus and differences follow.`;
    }
    if (currentStage === "answers-done") {
      return "Model answers complete. Consensus and differences have not started.";
    }
    if (currentStage === "handoff" || currentStage === "consensus") {
      return "Model answers complete. Building consensus and differences.";
    }
    if (currentStage === "done") {
      return "Consensus and differences complete.";
    }
    return "";
  }

  function render() {
    const element = getElement();
    if (!element) return;

    const counts = getAnswerCounts();
    const visibleStage = stage === "handoff" ? "consensus" : stage;
    element.dataset.stage = visibleStage;
    element.style.setProperty("--pipeline-progress", `${progressFor(stage, counts)}%`);
    const status = document.getElementById("consensusPipelineStatus");
    if (status) status.textContent = accessibleStatus(stage, counts);

    const count = document.getElementById("consensusPipelineCount");
    if (count) {
      count.textContent = counts.total ? `${counts.done}/${counts.total}` : "";
    }
  }

  function clearTimers() {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
  }

  function show() {
    const element = getElement();
    if (!element) return;
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    element.hidden = false;
    render();
    requestAnimationFrame(() => {
      requestAnimationFrame(() => element.classList.add("is-visible"));
    });
  }

  function hide(delay = 0) {
    const element = getElement();
    if (!element) return;
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      element.classList.remove("is-visible");
      hideTimer = setTimeout(() => {
        element.hidden = true;
        hideTimer = null;
      }, 250);
    }, delay);
  }

  function dismiss() {
    clearTimers();
    stage = "idle";
    hide();
  }

  function settleWithoutConsensus() {
    if (stage === "idle") return;
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
    stage = "answers-done";
    render();
    hide(650);
  }

  function complete() {
    if (stage === "idle") return;
    clearTimers();
    stage = "done";
    render();
    hide(700);
  }

  function onQueryStatus(status) {
    if (status === "running") {
      if (typeof window.isAgentModeEnabled === "function" && window.isAgentModeEnabled()) {
        dismiss();
        return;
      }
      clearTimers();
      stage = "answers";
      show();
      return;
    }

    if (stage === "idle") return;

    if (status === "complete") {
      const autoConsensus = document.getElementById("autoConsensusToggle")?.checked !== false;
      const canGenerate = typeof window.canGenerateConsensus === "function"
        ? window.canGenerateConsensus()
        : true;

      if (autoConsensus && canGenerate) {
        stage = "handoff";
        render();
        // Defensive fallback: if synthesis never starts, do not leave a
        // perpetual activity indicator behind.
        handoffTimer = setTimeout(settleWithoutConsensus, 6000);
      } else {
        settleWithoutConsensus();
      }
      return;
    }

    if (status === "error" || status === "canceled" || status === "idle") {
      dismiss();
    }
  }

  function onConsensusStart() {
    // Manual runs and bookmark loads must not manufacture a query pipeline.
    if (stage !== "answers" && stage !== "handoff") return;
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
    stage = "consensus";
    render();
  }

  function onConsensusEnd() {
    if (stage !== "consensus" && stage !== "handoff") return;
    complete();
  }

  const responseSection = document.querySelector(".response-section");
  if (responseSection && typeof MutationObserver === "function") {
    const observer = new MutationObserver(() => {
      if (stage === "answers") render();
    });
    observer.observe(responseSection, {
      subtree: true,
      attributes: true,
      attributeFilter: ["data-response-state", "class"]
    });
  }

  window.App.consensusPipeline = {
    onQueryStatus,
    onConsensusStart,
    onConsensusEnd,
    dismiss
  };
})();
