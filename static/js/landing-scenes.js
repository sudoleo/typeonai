// =====================================================================
// landing-scenes.js
// Scroll-scrubbed product scenes on the landing page.
//
// The premise: a marketing mockup that loops on a timer is decoration.
// A mockup whose playhead IS the scroll position is a demonstration —
// the reader sets the pace, can hold a phase still, and can scrub back
// to re-read it. So scenes 01 and 02 pin their stage while the section
// scrolls past, and every frame is derived from one number: how far
// through the section you are.
//
// Both scenes render the real /app surfaces (the composer well, the
// guided run block from consensus-progress.js). The phase names, the
// "Next:" lines and the handover to the provenance footer are the same
// ones the product uses, so the page cannot quietly drift away from it.
//
// Reduced motion (or no IntersectionObserver): every scene is rendered
// at its end state once and never touched again.
// =====================================================================

(function () {
  "use strict";

  const reducedMotion = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // Progress of `value` through the window [from, to], clamped to 0..1.
  const span = (value, from, to) => clamp((value - from) / (to - from), 0, 1);

  // ---- Scene 01: Ask --------------------------------------------------

  // Dieselbe Frage, die /app?demo=1 tippt (static/demo.js): Wer aus dem Hero
  // in die Demo springt, sieht dort weiter, was hier anfaengt. Nur die erste
  // Zeile — der Nachrichtenentwurf wird auch in der App eingefuegt, nicht
  // getippt.
  const ASK_QUESTION = "I have to tell a client that our launch slips by two weeks. Can I send this as it is?";

  function buildAskScene(scene) {
    const text = scene.querySelector("[data-ask-text]");
    const composer = scene.querySelector("[data-ask-composer]");
    const chip = scene.querySelector("[data-ask-chip]");
    const send = scene.querySelector("[data-ask-send]");
    const notes = Array.from(scene.querySelectorAll("[data-ask-note]"));
    if (!text || !composer) return null;

    let lastChars = -1;
    let lastStep = -1;

    // The question wraps to more lines as it is typed, and on a narrow screen
    // that grows the field — which grows the scene, which is what the scroll
    // progress is measured against. So the field is given the height of the
    // FINISHED question up front and never changes size while typing.
    function remeasureAsk() {
      text.style.minHeight = "";
      const previous = text.textContent;
      text.textContent = ASK_QUESTION;
      const full = text.offsetHeight;
      text.textContent = previous;
      if (full) text.style.minHeight = full + "px";
    }

    remeasureAsk();
    renderAsk.remeasure = remeasureAsk;

    function renderAsk(p) {
      // 0.08 → 0.58 types the question. Because the character count is a
      // pure function of scroll, scrubbing back un-types it rather than
      // restarting a timer somewhere else.
      const typed = span(p, 0.08, 0.58);
      const chars = Math.round(typed * ASK_QUESTION.length);
      if (chars !== lastChars) {
        text.textContent = ASK_QUESTION.slice(0, chars);
        lastChars = chars;
      }
      // The caret shows while there is still something to type.
      text.classList.toggle("is-typing", p > 0.04 && typed < 1);

      // Once there is a question, the composer is live: field focused,
      // then the run switch, then send.
      composer.classList.toggle("is-active", typed > 0.02);
      if (chip) chip.classList.toggle("is-lit", p >= 0.62 && p < 0.78);
      if (send) {
        send.classList.toggle("is-ready", typed >= 1);
        send.classList.toggle("is-pressed", p >= 0.9);
      }

      // The three notes step through in time with the controls above them.
      const step = p < 0.62 ? 0 : (p < 0.78 ? 1 : 2);
      if (step !== lastStep) {
        notes.forEach((note, i) => {
          note.classList.toggle("is-active", i === step);
          note.classList.toggle("is-done", i < step);
        });
        lastStep = step;
      }
    }

    return renderAsk;
  }

  // ---- Scene 02: Run --------------------------------------------------

  // Phase boundaries in scroll progress. They mirror the four stages in
  // static/js/consensus-progress.js; answers own the longest stretch for
  // the same reason they do in a real run.
  const PREPARE_END = 0.10;
  const ANSWERS_END = 0.60;
  const CONSENSUS_END = 0.78;
  const DIFFERENCES_END = 0.92;

  // Where each model lands inside the answers window, and the second it
  // finishes on. A run is only as fast as its slowest model, so the
  // spread is deliberate rather than six bars moving in lockstep.
  const MODEL_FINISH = [0.28, 0.40, 0.53, 0.63, 0.79, 1.0];
  const MODEL_START = [0.0, 0.03, 0.01, 0.06, 0.04, 0.09];
  const ANSWERS_SECONDS = 17.4;
  const RUN_SECONDS = 24;

  const NEXT_LINES = {
    prepare: "Next: <b>Ask the models</b> · <b>Write the consensus</b> · <b>Check for contradictions</b>",
    answers: "Next: <b>Write the consensus</b> · <b>Check for contradictions</b>",
    consensus: "Next: <b>Check for contradictions</b>",
    differences: "An uninvolved model compares all answers. It does not get a vote.",
    done: ""
  };

  // Every step the run checks off, in order. render() takes a prefix of this
  // by phase; remeasure() uses the whole list to find the tallest state.
  const ALL_PAST = [
    "Question prepared · Balanced · 6 models",
    "6 answers in " + ANSWERS_SECONDS.toFixed(1) + " s",
    "Consensus written"
  ];

  const LABELS = {
    prepare: "Preparing the question",
    answers: "Models are answering",
    consensus: "Writing the consensus",
    differences: "Checking for contradictions",
    done: "Done"
  };

  function clockText(seconds) {
    const total = Math.floor(Math.max(0, seconds));
    return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
  }

  function buildRunScene(scene) {
    const run = scene.querySelector("[data-run]");
    if (!run) return null;

    const past = scene.querySelector("[data-run-past]");
    const label = scene.querySelector("[data-run-label]");
    const count = scene.querySelector("[data-run-count]");
    const time = scene.querySelector("[data-run-time]");
    const track = scene.querySelector("[data-run-track]");
    const bar = track && track.querySelector("i");
    const next = scene.querySelector("[data-run-next]");
    const detail = scene.querySelector("[data-run-detail]");
    const result = scene.querySelector("[data-run-result]");
    const rows = Array.from(scene.querySelectorAll("[data-model]"));
    const stack = scene.querySelector("[data-run-stack]");

    let lastStage = "";
    let lastPastCount = -1;

    // The panel must not change size while the run plays. The per-model rows
    // still collapse when the models are done — that is the app's behaviour —
    // but they now collapse INSIDE a stack locked to its tallest state, so
    // nothing below the mock moves and the scroll progress has a fixed
    // reference. Measured rather than hardcoded, because the tallest state
    // depends on how the labels wrap at the current width.
    function remeasureRun() {
      if (!stack) return;
      stack.style.minHeight = "";

      // The tallest state is not the one the run happens to be in right now:
      // it is every check line present, the longest phase label, the longest
      // "Next:" line and the model rows still on screen. Build that, measure
      // it, then let the next paint put the real state back — render() is a
      // pure function of progress, so nothing has to be saved and restored.
      const wasGone = run.classList.contains("is-gone");
      run.classList.add("is-measuring");
      run.classList.remove("is-gone");
      if (detail) detail.classList.remove("is-hidden");
      if (past) {
        past.classList.remove("is-hidden");
        past.innerHTML = ALL_PAST.map(t => "<span>" + t + "</span>").join("");
      }
      if (label) label.textContent = LABELS.differences;
      if (count) count.textContent = "6 of 6";
      if (next) {
        next.classList.remove("is-hidden");
        next.innerHTML = NEXT_LINES.prepare;
      }

      const tallest = Math.max(run.offsetHeight, result ? result.offsetHeight : 0);

      run.classList.remove("is-measuring");
      if (wasGone) run.classList.add("is-gone");
      if (tallest) stack.style.minHeight = tallest + "px";

      // Force the next paint to rewrite everything this just overwrote.
      lastStage = "";
      lastPastCount = -1;
    }

    remeasureRun();
    renderRun.remeasure = remeasureRun;

    function renderRun(p) {
      let stage;
      if (p < PREPARE_END) stage = "prepare";
      else if (p < ANSWERS_END) stage = "answers";
      else if (p < CONSENSUS_END) stage = "consensus";
      else if (p < DIFFERENCES_END) stage = "differences";
      else stage = "done";

      const answered = span(p, PREPARE_END, ANSWERS_END);

      // ---- per-model rows: only while the models actually answer ----
      const modelsVisible = stage === "answers";
      if (detail) detail.classList.toggle("is-hidden", !modelsVisible);

      let done = 0;
      rows.forEach((row, i) => {
        const from = MODEL_START[i];
        const to = MODEL_FINISH[i];
        const share = stage === "prepare" ? 0 : span(answered, from, to);
        const isDone = share >= 1;
        if (isDone) done += 1;

        row.dataset.state = isDone ? "done" : "running";
        const rowBar = row.querySelector(".lp-run-model-track i");
        if (rowBar) rowBar.style.setProperty("--p", (share * 100).toFixed(1) + "%");

        const rowTime = row.querySelector(".lp-run-model-time");
        if (rowTime) {
          rowTime.textContent = isDone
            ? (to * ANSWERS_SECONDS).toFixed(1) + "s"
            : "·";
        }
      });
      if (stage !== "prepare" && stage !== "answers") done = rows.length;

      // ---- the single active line ----
      run.dataset.stage = stage;
      if (label) label.textContent = LABELS[stage];
      if (count) count.textContent = stage === "answers" ? done + " of " + rows.length : "";
      if (time) {
        time.textContent = clockText(
          stage === "done" ? RUN_SECONDS : p * RUN_SECONDS
        );
      }

      // Phases that cannot honestly report a share of themselves sweep
      // instead of pretending to know a percentage — exactly as in /app.
      if (track) {
        track.classList.toggle(
          "is-indeterminate",
          stage === "prepare" || stage === "consensus" || stage === "differences"
        );
      }
      if (bar) {
        const pct = stage === "answers"
          ? (done / rows.length) * 100
          : (stage === "done" ? 100 : 0);
        bar.style.setProperty("--p", pct.toFixed(1) + "%");
      }

      if (next && stage !== lastStage) {
        next.innerHTML = NEXT_LINES[stage];
        next.classList.toggle("is-hidden", !NEXT_LINES[stage]);
      }

      // ---- finished steps shrink to a grey check line ----
      const doneSteps = stage === "prepare" ? 0
        : (stage === "answers" ? 1
        : (stage === "consensus" ? 2 : 3));
      const pastItems = ALL_PAST.slice(0, doneSteps);
      if (past && pastItems.length !== lastPastCount) {
        past.innerHTML = pastItems.map(t => "<span>" + t + "</span>").join("");
        past.classList.toggle("is-hidden", pastItems.length === 0);
        lastPastCount = pastItems.length;
      }

      // ---- handover: the run collapses, the answer takes over ----
      const finished = stage === "done";
      run.classList.toggle("is-gone", finished);
      if (result) result.classList.toggle("is-visible", finished);

      lastStage = stage;
    }

    return renderRun;
  }

  // ---- Driver ---------------------------------------------------------

  const BUILDERS = { ask: buildAskScene, run: buildRunScene };

  function init() {
    const scenes = Array.from(document.querySelectorAll(".lp-scroll-scene"));
    if (!scenes.length) return;

    const tracked = [];

    scenes.forEach(scene => {
      const build = BUILDERS[scene.dataset.scene];
      const render = build && build(scene);
      if (!render) return;

      const stage = scene.querySelector(".lp-scene-stage");
      const rail = scene.querySelector(".lp-scene-rail i");
      tracked.push({ scene, stage, rail, render, inView: !reducedMotion, last: -1 });

      if (reducedMotion) {
        // A still frame, chosen so each scene shows what it is about: the
        // finished question for 01, a run mid-flight (two steps checked off,
        // the judge working) for 02. Rendering 02 at the very end would
        // leave the page with no run on it at all.
        render(scene.dataset.scene === "run" ? 0.86 : 1);
        scene.classList.add("is-static");
      }
    });

    if (reducedMotion || !tracked.length) return;

    // Only scenes on screen are measured; everything else costs nothing.
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          const item = tracked.find(t => t.scene === entry.target);
          if (item) item.inView = entry.isIntersecting;
        });
        schedule();
      }, { rootMargin: "20% 0px 20% 0px" });
      tracked.forEach(item => io.observe(item.scene));
    }

    let frame = 0;
    let running = false;

    // Geometry is cached, NOT read per frame. Reading the scene's live height
    // every frame made the animation chase itself: a phase change resizes
    // something inside the mock, the changed height feeds straight back into
    // the progress that decides the phase, and the scene jumps between two
    // states. Cached geometry breaks that loop, and it also keeps the frame
    // loop free of forced layout. It is refreshed when the page really does
    // relayout — resize, rotate, late fonts — never mid-run.
    function remeasure() {
      const viewport = window.innerHeight || document.documentElement.clientHeight;
      const scrollY = window.scrollY || window.pageYOffset || 0;

      tracked.forEach(item => {
        if (typeof item.render.remeasure === "function") item.render.remeasure();
      });

      tracked.forEach(item => {
        const rect = item.scene.getBoundingClientRect();
        item.geo = {
          top: rect.top + scrollY,
          height: rect.height,
          stageHeight: item.stage ? item.stage.offsetHeight : 0,
          viewport
        };
      });
    }

    function measure(item) {
      const geo = item.geo;
      if (!geo) return 0;
      const scrollY = window.scrollY || window.pageYOffset || 0;
      // Travel is the distance the pinned stage stays put for. When the scene
      // is not tall enough to pin (short viewports, phones), the scene's own
      // pass through the viewport is the playhead instead.
      const travel = geo.height - geo.stageHeight;

      if (travel > 40) return clamp((scrollY - geo.top) / travel, 0, 1);
      return clamp(
        (scrollY + geo.viewport * 0.82 - geo.top) / (geo.height + geo.viewport * 0.5),
        0, 1
      );
    }

    function paintOnce() {
      let active = false;
      tracked.forEach(item => {
        if (!item.inView) return;
        active = true;
        const p = measure(item);
        // Repaint only on real movement. Scrubbing wants sub-pixel fidelity,
        // idling wants to cost nothing.
        if (Math.abs(p - item.last) < 0.0004) return;
        item.last = p;
        item.scene.style.setProperty("--sp", p.toFixed(4));
        if (item.rail) item.rail.style.setProperty("--p", (p * 100).toFixed(2) + "%");
        item.render(p);
      });
      return active;
    }

    // A frame loop rather than a scroll listener. Scroll events are throttled
    // during momentum scrolling on iOS and skipped entirely for some
    // programmatic scrolls, which shows up as a scene that jumps or freezes.
    // The loop only runs while a scene is actually on screen, so an idle page
    // schedules nothing at all.
    function loop() {
      frame = 0;
      const active = paintOnce();
      if (active) {
        frame = requestAnimationFrame(loop);
      } else {
        // Nothing on screen: drop to a quarter-second heartbeat instead of
        // stopping dead, so the scenes come back even where the scroll event
        // never arrives.
        running = false;
        window.setTimeout(schedule, 250);
      }
    }

    function schedule() {
      if (running || frame) return;
      running = true;
      frame = requestAnimationFrame(loop);
    }

    // A genuine relayout invalidates the cache; a phase change never does.
    let resizeTimer = 0;
    function refresh() {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        remeasure();
        tracked.forEach(item => { item.last = -1; });
        schedule();
      }, 120);
    }

    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", refresh, { passive: true });
    window.addEventListener("orientationchange", refresh, { passive: true });
    window.addEventListener("load", refresh);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(refresh).catch(() => {});
    }
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) schedule();
    });

    remeasure();
    paintOnce();
    schedule();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
