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
    // textContent, not innerText: the boxes sit behind "Compare answers"
    // and innerText reports nothing for a display:none element.
    const content = box.querySelector(".collapsible-content");
    return Boolean(
      content
      && !content.querySelector(".thinking-wrap")
      && !content.classList.contains("is-streaming")
      && content.textContent.trim()
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

  // A run is only as fast as its slowest model. Once enough models have
  // answered, a straggler is measurably late rather than merely slow — and
  // from that point the reader gets the choice to go on without it. The
  // offer stays quiet and only appears next to the row it belongs to.
  const SKIP_MIN_DONE = 2;
  const SKIP_LAG_MS = 8000;

  function skipOfferReady() {
    if (stage !== "answers" || !startedAt) return false;
    if (rowTimes.size < SKIP_MIN_DONE) return false;
    const lastFinished = Math.max(...rowTimes.values());
    return Date.now() - startedAt - lastFinished >= SKIP_LAG_MS;
  }

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
        + '<span class="run-model-skip"></span>'
        + "</span>";
    }).join("");
    detail.hidden = boxes.length === 0;
    detailBuilt = true;
  }

  function renderSkipOffer(row, box, done) {
    const slot = row.querySelector(".run-model-skip");
    if (!slot) return;
    const offer = !done && skipOfferReady();
    if (!offer) {
      if (slot.firstChild) slot.innerHTML = "";
      return;
    }
    if (slot.firstChild) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "run-model-skip-btn";
    const name = box.dataset.shortLabel || box.dataset.model || "this model";
    btn.textContent = "Taking longer — skip";
    btn.title = "Go on without " + name + ". Its answer is dropped from this run.";
    btn.setAttribute("aria-label", "Skip " + name + ", it is taking longer than expected");
    btn.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      btn.disabled = true;
      window.App?.skipModel?.(box.id);
    });
    slot.appendChild(btn);
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
      if (time) {
        if (done) {
          if (!rowTimes.has(box.id)) rowTimes.set(box.id, Date.now() - startedAt);
          time.textContent = seconds(rowTimes.get(box.id)).toFixed(1) + "s";
        } else {
          time.textContent = "·";
        }
      }

      renderSkipOffer(row, box, done);
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

  // ---- The three drawers ---------------------------------------------
  // Contradictions, model answers and sources are the everything-behind-the
  // answer. Each chip owns one disclosure and reports how much is in it, so
  // the reader can tell what is worth opening before opening it. The counts
  // are read off the rendered DOM for the same reason the contested count
  // is: a second tally could disagree with what is on screen.

  // count > 0 shows the number next to the label; count === 0 with
  // hasContent still offers the drawer but without a figure — the free-text
  // differences fallback has something to show and nothing to count.
  function syncTab(tabId, countId, panel, count, hasContent = count > 0) {
    const tab = $(tabId);
    if (!tab) return;
    const available = Boolean(panel) && hasContent;
    tab.hidden = !available;
    if (!available) return;
    const countEl = $(countId);
    if (countEl) countEl.textContent = count > 0 ? String(count) : "";
    tab.setAttribute("aria-expanded", String(isPanelOpen(panel)));
  }

  function isPanelOpen(panel) {
    if (!panel) return false;
    return panel.tagName === "DETAILS" ? panel.open : !panel.hidden;
  }

  function setPanelOpen(panel, open) {
    if (!panel) return;
    if (panel.tagName === "DETAILS") panel.open = open;
    else panel.hidden = !open;
  }

  function togglePanel(tab, panel) {
    if (!tab || !panel) return;
    const open = !isPanelOpen(panel);
    setPanelOpen(panel, open);
    tab.setAttribute("aria-expanded", String(open));
    if (open) {
      // Erst nach dem Layout scrollen: bei hidden -> sichtbar hatte der
      // Sources-Drawer sonst noch keine belastbare Geometrie. Bei langen
      // Quellenlisten fahren wir gezielt den ersten Eintrag an, damit nur ein
      // kleiner Inhaltsanfang ins Bild kommt statt der ganze Drawer zu springen.
      requestAnimationFrame(() => {
        const target = panel.id === "consensusSourcesPanel"
          ? panel.querySelector(".consensus-sources-list > li") || panel
          : panel;
        target.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    }
  }

  function countDifferences() {
    const cards = $("differencesCards");
    if (!cards || cards.hidden) return 0;
    return cards.querySelectorAll(".diff-card").length;
  }

  // Anything in the drawer at all: cards, the "no contradictions" empty
  // state, or the free-text fallback paragraph.
  function differencesHasContent() {
    const panel = $("consensusDifferencesPanel");
    if (!panel) return false;
    const cards = $("differencesCards");
    if (cards && !cards.hidden && cards.childElementCount) return true;
    const fallback = panel.querySelector(".consensus-differences-content > p");
    return Boolean(fallback && !fallback.hidden && fallback.textContent.trim());
  }

  function countSources() {
    return document.querySelectorAll("#consensusSourcesList > li").length;
  }

  // "Run again" ist kein Zurueckspulen, sondern ein kompletter zweiter Lauf:
  // alle Modelle antworten erneut, der Consensus wird neu geschrieben. Das
  // kostet dasselbe wie die erste Frage, und genau das muss am Knopf stehen,
  // bevor er geklickt wird — nicht erst im Zaehler danach.
  function quotaRuns() {
    try {
      return window.App?.sidebarQuota?.runs?.() || null;
    } catch (err) {
      return null;
    }
  }

  function labelRunAgain(button) {
    const runs = quotaRuns();
    const cost = $("runReplayCost");
    const unlimited = !!runs?.unlimited;

    if (cost) cost.textContent = unlimited ? "" : " · uses 1 run";

    const detail = ["Runs every model again and writes a new consensus."];
    if (unlimited) {
      detail.push("Your plan has unlimited runs.");
    } else {
      detail.push("It costs one run from your quota.");
      if (runs) {
        detail.push(runs.value > 0
          ? runs.value + " of " + runs.limit + " left today."
          : "No runs left today.");
      }
    }
    button.title = detail.join(" ");
  }

  function setComposerRunNotice(text) {
    const notice = $("composerRunNotice");
    if (!notice) return;
    notice.textContent = text || "";
    notice.hidden = !text;
  }

  function renderProvenance() {
    const wrap = $("runProvenance");
    const facts = $("runProvenanceFacts");
    if (!wrap || !facts) return;

    // The footer is what the run hands over to, so it never appears beside a
    // running one. Sources land while the models are still answering; without
    // this the "Sources" chip would show up under a half-written answer.
    if (stage !== "idle" && stage !== "done") {
      wrap.hidden = true;
      return;
    }

    // Die Fakten des Laufs: woraus und wie schnell. Die Zahl der strittigen
    // Passagen stand hier bis 2026-07-28 als dritte Angabe — dieselbe Aussage
    // wie "N critical" im Urteil und wie die Zahl an "Review differences",
    // dreimal formuliert in drei Zeilen. Sie steht jetzt nur noch dort, wo man
    // sie auch aufklappen kann.
    const parts = [];
    if (lastRunSummary?.models) {
      parts.push("<b>" + lastRunSummary.models + " models</b>");
    }
    if (lastRunSummary?.durationMs) {
      parts.push(Math.round(lastRunSummary.durationMs / 1000) + " s");
    }

    facts.innerHTML = parts.join(" · ");

    const runAgain = $("runReplayButton");
    if (runAgain) {
      runAgain.hidden = !lastRunSummary || !String(window.lastQuestion || "").trim();
      if (!runAgain.hidden) labelRunAgain(runAgain);
    }

    syncTab(
      "consensusDifferencesTab",
      "consensusDifferencesTabCount",
      $("consensusDifferencesPanel"),
      countDifferences(),
      differencesHasContent()
    );
    syncTab(
      "consensusSourcesTab",
      "consensusSourcesTabCount",
      $("consensusSourcesPanel"),
      countSources()
    );

    // "Compare answers" bekommt dieselbe Zahl wie die anderen beiden
    // Schubladen: wie viele Antworten dahinter liegen. Ohne sie war es die
    // einzige der drei Flaechen, bei der man erst aufklappen musste, um zu
    // wissen, was einen erwartet.
    const answersCount = $("consensusAnswersTabCount");
    if (answersCount) {
      const boxes = document.querySelectorAll(
        ".response-section > .response-box:not(.excluded)"
      ).length;
      answersCount.textContent = boxes > 0 ? String(boxes) : "";
    }

    // The footer also hosts the drawers, so it stays open when it has no
    // facts of its own but does carry one.
    const hasTabs = Boolean(
      document.querySelector("#consensusFooterTabs .consensus-tab:not([hidden])")
      || !$("agentModeAnswersRow")?.hidden
    );
    wrap.hidden = parts.length === 0 && !hasTabs;
  }

  function clearProvenance() {
    const wrap = $("runProvenance");
    if (wrap) wrap.hidden = true;
    lastRunSummary = null;
    ["consensusDifferencesTab", "consensusSourcesTab"].forEach(id => {
      const tab = $(id);
      if (tab) {
        tab.hidden = true;
        tab.setAttribute("aria-expanded", "false");
      }
    });
    setPanelOpen($("consensusSourcesPanel"), false);
  }

  // ---- Lifecycle hooks -----------------------------------------------

  // startedAtMs belongs to the run this block is about to describe. It is
  // passed when an already running run is re-opened, so the clock keeps
  // counting from the real start instead of from the moment it came back
  // on screen.
  function onPrepare(startedAtMs) {
    resetState();
    clearProvenance();
    setComposerRunNotice("");
    const owned = Number(startedAtMs);
    startedAt = Number.isFinite(owned) && owned > 0 ? owned : Date.now();
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
    // Ein manuell spaeter gestarteter Consensus und ein wiederhergestelltes
    // Ergebnis haben keine aktive Query-Pipeline mehr. Der Provenance-Fuss
    // (inklusive "Compare answers") gehoert trotzdem immer zur fertigen
    // Antwort.
    if (stage !== "consensus" && stage !== "differences") {
      renderProvenance();
      return;
    }
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
    // Render before hiding: data-stage is read by CSS and by anything that
    // asks what the block is showing, and a hidden block still has to answer
    // "nothing" rather than name the step it was on when it left.
    render();
    hideNow();
  }

  // The block and the facts under the answer describe ONE run. When what is
  // on screen is no longer that run -- a saved bookmark was opened while a
  // run keeps going in the background -- both leave with it. The run itself
  // is not touched: its progress stays readable in its sidebar row.
  function detach() {
    dismiss();
    clearProvenance();
  }

  // The provenance facts of a finished run, handed over when that run is
  // projected again after the view had moved elsewhere. null clears them,
  // so no answer ever wears another run's numbers.
  function setRunFacts(facts) {
    const models = Number(facts?.models);
    lastRunSummary = Number.isFinite(models) && models > 0
      ? { models, durationMs: Math.max(0, Number(facts?.durationMs) || 0) }
      : null;
    renderProvenance();
  }

  // A fresh comparison with the same question follows the normal composer
  // flow. It never sends automatically, so no run is started by surprise —
  // aber es soll auch niemand glauben, ein Wiederholen sei gratis. Deshalb
  // steht ab hier bis zum Absenden am Eingabefeld, was der Klick auf Senden
  // kostet.
  function prepareRunAgain() {
    const question = String(window.lastQuestion || "").trim();
    if (!question) return;
    $("newRunButton")?.click();
    const input = $("questionInput");
    if (!input) return;
    input.value = question;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.focus();

    const runs = quotaRuns();
    if (runs?.unlimited) {
      setComposerRunNotice("Same question, ready to send. Sending starts a complete new run — every model answers again.");
    } else if (runs && runs.value <= 0) {
      setComposerRunNotice("Same question, ready to send. A repeat is a complete new run, and you have no runs left today.");
    } else {
      setComposerRunNotice(
        "Same question, ready to send. A repeat is a complete new run and uses 1 run"
        + (runs ? " — " + runs.value + " left today." : " from your quota.")
      );
    }
  }

  // Das Kontingent kommt asynchron (Login, Antwort eines Laufs) und aendert
  // sich waehrend die Antwort schon dasteht. Der Preis am Knopf haengt daran,
  // also wird er nachgezogen statt einmal beim Rendern eingefroren.
  const usageSource = document.getElementById("usageDisplay");
  if (usageSource && typeof MutationObserver === "function") {
    new MutationObserver(() => {
      const button = $("runReplayButton");
      if (button && !button.hidden) labelRunAgain(button);
    }).observe(usageSource, { childList: true, subtree: true, characterData: true });
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
    if (event.target.closest("#runReplayButton")) {
      prepareRunAgain();
      return;
    }
    // Ein bewusst neuer Vergleich raeumt den Hinweis weg; der Hinweis gehoert
    // nur zu der einen vorbereiteten Wiederholung. (prepareRunAgain klickt
    // #newRunButton selbst — und setzt den Hinweis danach.)
    if (event.target.closest("#newRunButton")) setComposerRunNotice("");
    const diffTab = event.target.closest("#consensusDifferencesTab");
    if (diffTab) {
      togglePanel(diffTab, $("consensusDifferencesPanel"));
      return;
    }
    const sourcesTab = event.target.closest("#consensusSourcesTab");
    if (sourcesTab) togglePanel(sourcesTab, $("consensusSourcesPanel"));
  });

  // The differences drawer is a <details>; anything that opens or closes it
  // from elsewhere (the free-text fallback, a loaded bookmark) still has to
  // leave the chip telling the truth.
  document.getElementById("consensusDifferencesPanel")?.addEventListener("toggle", () => {
    const tab = $("consensusDifferencesTab");
    if (tab) tab.setAttribute("aria-expanded", String($("consensusDifferencesPanel").open));
  });

  window.App.consensusPipeline = {
    onPrepare,
    onQueryStatus,
    onConsensusStart,
    onDifferencesStart,
    onConsensusEnd,
    renderProvenance,
    setRunFacts,
    detach,
    dismiss
  };
})();
