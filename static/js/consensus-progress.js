// =====================================================================
// consensus-progress.js
// The guided run block below the question input.
//
// It stays observational: it owns no request state, it only watches the
// lifecycle events every run already emits and the response boxes those
// runs write into. What changed with the frameless shell is how much it
// shows at once — and that it is now the ONLY run UI. Agent Mode used to
// bring its own panel with its own chips, timer and status line; two
// progress displays for one request was exactly the surplus this rebuild
// is removing. Agent Mode keeps its behaviour (group the models, consensus
// automatically, individual answers hidden until asked for) and gives up
// its second opinion about how a run should look.
//
// One active step is visible: a label, a count, a timer, a single bar and
// a "Next" line. Finished steps shrink to a grey check line, the per-model
// rows exist only while the models are actually answering, and when the run
// ends the whole block folds away and hands over to the provenance line
// under the answer. The point is that a reader is never asked to hold four
// progress indicators in their head to know what the machine is doing.
// =====================================================================

(function () {
  window.App = window.App || {};

  // Share of the bar each phase owns. Answers dominate because they take
  // the longest; the two synthesis phases get their own bar instead of
  // sharing one, so "the answer is written" does not look like "done".
  const ANSWERS_END = 100;

  const STEPS = {
    prepare: {
      label: "Preparing the question",
      next: ["Ask the models", "Write the consensus", "Check for contradictions"]
    },
    answers: {
      label: "Models are answering",
      next: ["Write the consensus", "Check for contradictions"]
    },
    consensus: {
      label: "Writing the consensus",
      next: ["Check for contradictions"]
    },
    differences: {
      label: "Checking for contradictions",
      // Not a step list: at this point the useful thing to say is who is
      // doing the checking, because it is not one of the six.
      note: "An uninvolved model compares all answers. It does not get a vote."
    },
    done: { label: "Done" }
  };

  let stage = "idle";
  let past = [];
  let hideTimer = null;
  let handoffTimer = null;
  let ticker = null;
  let startedAt = 0;
  let answersFinishedAt = 0;
  let detailBuilt = false;
  let lastRunSummary = null;

  const $ = id => document.getElementById(id);
  const root = () => $("consensusRun");

  function seconds(ms) {
    return Math.max(0, Math.round(ms / 100) / 10);
  }

  function clockText(ms) {
    const total = Math.floor(Math.max(0, ms) / 1000);
    return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
  }

  function getIncludedBoxes() {
    return Array.from(
      document.querySelectorAll(".response-section > .response-box")
    ).filter(box => !box.classList.contains("excluded"));
  }

  function isBoxDone(box) {
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
  }

  function getAnswerCounts() {
    const boxes = getIncludedBoxes();
    return { done: boxes.filter(isBoxDone).length, total: boxes.length };
  }

  // ---- Per-model rows ------------------------------------------------
  // Only rendered while the models are answering. Each row grows on the
  // real stream (agent-mode.js already estimates this from the streamed
  // text) and freezes at the elapsed time when its model finishes, so the
  // number that stays on screen is measured, not invented.
  const rowTimes = new Map();

  function buildDetail() {
    const detail = $("runDetail");
    if (!detail) return;
    const boxes = getIncludedBoxes();
    detail.innerHTML = boxes.map(box => {
      const name = box.dataset.shortLabel || box.dataset.model || "Model";
      return '<span class="run-model" data-box="' + box.id + '">'
        + '<span class="run-model-name">' + name + "</span>"
        + '<span class="run-model-track"><i></i></span>'
        + '<span class="run-model-time">·</span>'
        + "</span>";
    }).join("");
    detail.hidden = boxes.length === 0;
    detailBuilt = true;
  }

  function renderDetail() {
    const detail = $("runDetail");
    if (!detail || detail.hidden) return;
    const progressByBox = window.App?.agentMode?.streamProgressByResponseId?.() || {};

    detail.querySelectorAll(".run-model").forEach(row => {
      const box = document.getElementById(row.dataset.box);
      if (!box) return;
      const done = isBoxDone(box);
      const share = done ? 1 : (progressByBox[box.id] || 0);

      row.dataset.state = done ? "done" : "running";
      const bar = row.querySelector(".run-model-track i");
      if (bar) bar.style.setProperty("--p", (share * 100).toFixed(1) + "%");

      const time = row.querySelector(".run-model-time");
      if (!time) return;
      if (done) {
        if (!rowTimes.has(box.id)) rowTimes.set(box.id, Date.now() - startedAt);
        time.textContent = seconds(rowTimes.get(box.id)).toFixed(1) + "s";
      } else {
        time.textContent = "·";
      }
    });
  }

  // ---- Rendering -----------------------------------------------------

  function progressFor(currentStage, counts) {
    if (currentStage === "prepare") return 0;
    if (currentStage === "answers") {
      const ratio = counts.total ? counts.done / counts.total : 0;
      return Math.round(ratio * ANSWERS_END);
    }
    // Both synthesis phases run their own bar. Neither reports a percentage
    // it cannot know, so they are indeterminate rather than fake-precise.
    if (currentStage === "consensus" || currentStage === "differences") return 0;
    if (currentStage === "done") return 100;
    return 0;
  }

  function accessibleStatus(currentStage, counts) {
    if (currentStage === "prepare") return "Preparing the question.";
    if (currentStage === "answers") {
      return `Models are answering: ${counts.done} of ${counts.total} done.`;
    }
    if (currentStage === "consensus") return "Writing the consensus.";
    if (currentStage === "differences") return "Checking the answers for contradictions.";
    if (currentStage === "done") return "Run complete.";
    return "";
  }

  function renderPast() {
    const el = $("runPast");
    if (!el) return;
    el.innerHTML = past.map(text => "<span>" + text + "</span>").join("");
    el.hidden = past.length === 0;
  }

  function render() {
    const el = root();
    if (!el) return;

    const counts = getAnswerCounts();
    const step = STEPS[stage];
    el.dataset.stage = stage;

    const label = $("runLabel");
    if (label && step) label.textContent = step.label;

    const count = $("runCount");
    if (count) {
      count.textContent = stage === "answers" && counts.total
        ? `${counts.done} of ${counts.total}`
        : "";
    }

    const track = $("runTrack");
    // Indeterminate while a phase cannot report a share of itself.
    if (track) {
      track.classList.toggle(
        "is-indeterminate",
        stage === "prepare" || stage === "consensus" || stage === "differences"
      );
    }

    const bar = $("runBar");
    if (bar) bar.style.setProperty("--p", progressFor(stage, counts) + "%");

    const next = $("runNext");
    if (next && step) {
      if (step.note) {
        next.textContent = step.note;
      } else if (step.next) {
        next.innerHTML = "Next: " + step.next.map(t => "<b>" + t + "</b>").join(" · ");
      } else {
        next.textContent = "";
      }
      next.hidden = !next.textContent.trim();
    }

    const status = $("runStatus");
    if (status) status.textContent = accessibleStatus(stage, counts);

    renderDetail();
  }

  function tick() {
    const time = $("runTime");
    if (time && startedAt) time.textContent = clockText(Date.now() - startedAt);
    if (stage === "answers") render();
  }

  function startTicker() {
    stopTicker();
    ticker = window.setInterval(tick, 200);
  }

  function stopTicker() {
    if (ticker) window.clearInterval(ticker);
    ticker = null;
  }

  // ---- Visibility ----------------------------------------------------

  function show() {
    const el = root();
    if (!el) return;
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    el.hidden = false;
    el.classList.remove("is-gone");
    el.style.maxHeight = "";
    render();
    requestAnimationFrame(() => {
      requestAnimationFrame(() => el.classList.add("is-visible"));
    });
  }

  // The run does not just disappear: it collapses, so the eye follows the
  // answer moving up rather than hunting for what vanished.
  function vanish(delay) {
    const el = root();
    if (!el) return;
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      el.style.maxHeight = el.scrollHeight + "px";
      requestAnimationFrame(() => el.classList.add("is-gone"));
      hideTimer = setTimeout(() => {
        el.hidden = true;
        el.classList.remove("is-visible", "is-gone");
        el.style.maxHeight = "";
        hideTimer = null;
      }, 520);
    }, delay);
  }

  function hideNow() {
    const el = root();
    if (!el) return;
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = null;
    el.hidden = true;
    el.classList.remove("is-visible", "is-gone");
    el.style.maxHeight = "";
  }

  function resetState() {
    stopTicker();
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
    past = [];
    rowTimes.clear();
    detailBuilt = false;
    startedAt = 0;
    answersFinishedAt = 0;
    const detail = $("runDetail");
    if (detail) {
      detail.hidden = true;
      detail.innerHTML = "";
    }
    renderPast();
  }

  function addPast(text) {
    past.push(text);
    renderPast();
  }

  function enter(nextStage) {
    stage = nextStage;
    render();
  }

  // ---- Provenance handover -------------------------------------------

  function renderProvenance() {
    const wrap = $("runProvenance");
    const facts = $("runProvenanceFacts");
    if (!wrap || !facts) return;

    const parts = [];
    if (lastRunSummary?.models) {
      parts.push("<b>" + lastRunSummary.models + " models</b>");
    }
    if (lastRunSummary?.durationMs) {
      parts.push(Math.round(lastRunSummary.durationMs / 1000) + " s");
    }
    // Counted from what is actually on screen — the inline markers the
    // insights pass wrote into the answer. No second bookkeeping that could
    // disagree with what the reader can see.
    const contested = document.querySelectorAll(
      "#consensusAnswerBody .cx-marker"
    ).length;
    if (contested > 0) {
      parts.push(contested === 1 ? "1 contested passage" : contested + " contested passages");
    }

    facts.innerHTML = parts.join(" · ");

    // "Replay run" only means something while there is a run to replay.
    const replay = $("runReplayButton");
    if (replay) replay.hidden = !lastRunSummary;

    // The line also hosts the follow-up offer, so it stays open when it has
    // no facts of its own but does carry an action.
    const actions = $("followupBar");
    const hasActions = Boolean(actions && !actions.hidden && actions.childElementCount);
    wrap.hidden = parts.length === 0 && !hasActions;
  }

  function clearProvenance() {
    const wrap = $("runProvenance");
    if (wrap) wrap.hidden = true;
    lastRunSummary = null;
  }

  // ---- Lifecycle hooks -----------------------------------------------

  function onPrepare() {
    resetState();
    clearProvenance();
    startedAt = Date.now();
    enter("prepare");
    show();
    startTicker();
  }

  function onQueryStatus(status) {
    if (status === "running") {
      // A run that never announced /prepare (demo, replay) still gets a clock.
      if (stage === "idle") {
        resetState();
        clearProvenance();
        startedAt = Date.now();
        show();
        startTicker();
      } else if (stage === "prepare") {
        const preset = window.App?.currentPresetLabel?.();
        const total = getIncludedBoxes().length;
        addPast("Question prepared" + (preset ? " · " + preset : "") + " · " + total + " models");
      }
      buildDetail();
      enter("answers");
      return;
    }

    if (stage === "idle") return;

    if (status === "complete") {
      const autoConsensus = document.getElementById("autoConsensusToggle")?.checked !== false;
      const canGenerate = typeof window.canGenerateConsensus === "function"
        ? window.canGenerateConsensus()
        : true;

      answersFinishedAt = Date.now();
      const counts = getAnswerCounts();
      const detail = $("runDetail");
      if (detail) detail.hidden = true;
      addPast(
        counts.done + " answers in "
        + seconds(answersFinishedAt - startedAt).toFixed(1) + " s"
      );

      if (autoConsensus && canGenerate) {
        enter("consensus");
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

  function settleWithoutConsensus() {
    if (stage === "idle") return;
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
    stopTicker();
    enter("done");
    vanish(600);
    stage = "idle";
  }

  function onConsensusStart() {
    // Manual runs and bookmark loads must not manufacture a query pipeline.
    if (stage !== "answers" && stage !== "consensus") return;
    if (handoffTimer) {
      clearTimeout(handoffTimer);
      handoffTimer = null;
    }
    const detail = $("runDetail");
    if (detail) detail.hidden = true;
    enter("consensus");
  }

  // The differences judge starts once the consensus text is complete.
  function onDifferencesStart() {
    if (stage !== "consensus") return;
    addPast("Consensus written");
    enter("differences");
  }

  function onConsensusEnd() {
    if (stage !== "consensus" && stage !== "differences") return;
    stopTicker();

    lastRunSummary = {
      models: getAnswerCounts().done,
      durationMs: Date.now() - startedAt
    };

    enter("done");
    renderProvenance();
    vanish(700);
    stage = "idle";
  }

  function dismiss() {
    resetState();
    stage = "idle";
    hideNow();
  }

  // "Replay run" puts the finished block back on screen without re-running
  // anything: it is a record, not a second request.
  function replay() {
    const el = root();
    if (!el || !lastRunSummary) return;
    stage = "done";
    render();
    show();
    vanish(3200);
    stage = "idle";
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

  document.addEventListener("click", event => {
    if (event.target.closest("#runReplayButton")) replay();
  });

  window.App.consensusPipeline = {
    onPrepare,
    onQueryStatus,
    onConsensusStart,
    onDifferencesStart,
    onConsensusEnd,
    renderProvenance,
    dismiss
  };
})();
