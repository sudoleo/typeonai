/* ==========================================================================
   Sidebar quota panel

   The footer ring + the panel above it are a *view* of the usage column that
   already exists (#usageDisplay). That column stays the single source of
   truth — app-core.js (renderUsageDisplay), firebase.js and watch.js keep
   writing into it exactly as before, and a MutationObserver mirrors whatever
   lands there into the ring and the bars.

   Doing it this way means the redesign adds a surface without adding a second
   place where "how many runs do I have left" can be computed — and therefore
   without a second place where it can be wrong.
   ========================================================================== */
(function () {
  "use strict";

  var RING_LENGTH = 87.9; // 2πr for r=14, matches the stroke-dasharray in the markup

  function el(id) {
    return document.getElementById(id);
  }

  /* "Runs: 2 / 3" → {value: 2, limit: 3}; "Runs: Unlimited" → {unlimited: true}.
     Returns null when the line is still a skeleton or was never filled. */
  function parseLine(node) {
    if (!node) return null;
    var strong = node.querySelector("strong");
    if (!strong) return null;
    var text = (strong.textContent || "").trim();
    if (!text || text === "...") return null;
    if (/unlimited/i.test(text)) return { unlimited: true };

    var parts = text.split("/");
    if (parts.length !== 2) return null;
    var value = Number(parts[0].trim());
    var limit = Number(parts[1].trim());
    if (!Number.isFinite(value) || !Number.isFinite(limit) || limit <= 0) return null;
    return { value: value, limit: limit };
  }

  function renderRow(rowId, valueId, data) {
    var row = el(rowId);
    var valueEl = el(valueId);
    if (!row || !valueEl) return;

    if (!data) {
      row.hidden = true;
      return;
    }

    row.hidden = false;
    var bar = row.querySelector(".quota-track i");
    var track = row.querySelector(".quota-track");

    if (data.unlimited) {
      valueEl.textContent = "Unlimited";
      if (bar) bar.style.setProperty("--p", "100%");
      if (track) track.classList.remove("is-low", "is-out");
      return;
    }

    valueEl.textContent = data.value + " / " + data.limit;
    // The bar shows what is LEFT, not what is spent: a full bar is good news.
    var share = Math.max(0, Math.min(1, data.value / data.limit));
    if (bar) bar.style.setProperty("--p", (share * 100).toFixed(1) + "%");
    if (track) {
      track.classList.toggle("is-out", data.value <= 0);
      track.classList.toggle("is-low", data.value > 0 && share <= 0.25);
    }
  }

  function renderRing(runs) {
    var trigger = el("quotaTrigger");
    var arc = el("quotaRingArc");
    var value = el("quotaTriggerValue");
    if (!trigger) return;

    // No usable number yet (guest, or still loading) → no ring at all.
    if (!runs) {
      trigger.hidden = true;
      return;
    }

    trigger.hidden = false;

    if (runs.unlimited) {
      if (arc) {
        arc.setAttribute("stroke-dashoffset", "0");
        arc.setAttribute("stroke", "var(--ink-2)");
      }
      if (value) value.textContent = "∞";
      trigger.title = "Unlimited runs";
      trigger.setAttribute("aria-label", "Unlimited runs");
      return;
    }

    var share = Math.max(0, Math.min(1, runs.value / runs.limit));
    if (arc) {
      arc.setAttribute("stroke-dashoffset", String(RING_LENGTH * (1 - share)));
      arc.setAttribute(
        "stroke",
        runs.value <= 0 ? "var(--dispute)" : share <= 0.25 ? "var(--partial)" : "var(--ink-2)"
      );
    }
    if (value) value.textContent = String(runs.value);

    var label = runs.value + " of " + runs.limit + " runs left";
    trigger.title = label;
    trigger.setAttribute("aria-label", label);
  }

  function sync() {
    var runs = parseLine(el("freeUsageDisplay"));
    var deep = parseLine(el("deepUsageDisplay"));
    var watches = parseLine(el("watchUsageDisplay"));

    renderRow("quotaRowRuns", "quotaRunsValue", runs);
    renderRow("quotaRowDeep", "quotaDeepValue", deep);
    renderRow("quotaRowWatch", "quotaWatchValue", watches);
    renderRing(runs);

    // Der Plan steht hier, nicht mehr neben "New comparison". Pro spricht
    // ueber das goldene Badge daneben (user-tier.js blendet #proBadge ein),
    // also weicht das Textlabel dann zurueck statt "Pro Pro" zu schreiben.
    var planLabel = el("quotaPlanLabel");
    if (planLabel) {
      planLabel.textContent = window.isUserPro ? "Pro" : "Free";
      planLabel.hidden = !!window.isUserPro;
    }

    // The countdown span carries the reset time ("Resets in 1 h 58 min").
    var countdown = el("countdownDisplay");
    var foot = el("quotaFoot");
    if (foot) {
      var text = countdown ? (countdown.textContent || "").trim() : "";
      foot.textContent = text;
      foot.hidden = !text;
    }
  }

  function setOpen(open) {
    var panel = el("sidebarQuota");
    var trigger = el("quotaTrigger");
    if (!panel) return;
    panel.classList.toggle("is-open", open);
    if (trigger) trigger.setAttribute("aria-expanded", String(open));
  }

  function init() {
    var source = el("usageDisplay");
    var trigger = el("quotaTrigger");
    var panel = el("sidebarQuota");
    if (!source || !trigger || !panel) return;

    sync();

    // characterData + subtree: renderUsageDisplay replaces children, other
    // writers only touch text nodes. Both have to reach us.
    new MutationObserver(sync).observe(source, {
      childList: true,
      subtree: true,
      characterData: true
    });

    trigger.addEventListener("click", function (event) {
      event.stopPropagation();
      setOpen(!panel.classList.contains("is-open"));
    });

    document.addEventListener("click", function (event) {
      if (!panel.classList.contains("is-open")) return;
      if (panel.contains(event.target) || trigger.contains(event.target)) return;
      setOpen(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && panel.classList.contains("is-open")) {
        setOpen(false);
        trigger.focus();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  // Andere Module (z. B. "Run again") muessen sagen koennen, was ein Klick
  // kostet. Sie lesen dafuer dieselbe Quelle wie der Ring, statt sich eine
  // zweite Rechnung zu bauen: null = noch unbekannt (Gast oder ladend).
  function runs() {
    return parseLine(el("freeUsageDisplay"));
  }

  // Dieselbe Lesart fuer den Deep-Think-Topf. usage-limit.js braucht ihn, um
  // vor dem Absenden sagen zu koennen, WELCHES Kontingent fehlt — und darf
  // sich dafuer keine zweite Parse-Regel bauen.
  function deep() {
    return parseLine(el("deepUsageDisplay"));
  }

  window.App = window.App || {};
  window.App.sidebarQuota = { sync: sync, setOpen: setOpen, runs: runs, deep: deep };
})();
