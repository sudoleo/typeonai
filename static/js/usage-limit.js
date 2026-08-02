/* ==========================================================================
   usage-limit.js — "dieser Lauf hat nicht stattgefunden"

   Ein aufgebrauchtes Kontingent war bis 2026-07-31 der einzige Fehlerfall,
   den die App gar nicht erzaehlt hat. Der Grund war kein fehlender Text,
   sondern ein fehlender ORT: der gefuehrte Lauf (consensus-progress.js)
   startet schon bei /prepare, und der Limit-Pfad meldete danach
   setAgentModeStatus("error") — was den Block per dismiss() wieder
   wegnimmt. Die eigentliche Meldung landete in den Antwortboxen, die im
   Agent Mode (Default) hinter "Compare answers" liegen. Ergebnis: eine
   Sekunde Fortschritt, dann eine leere Seite.

   Dieses Modul ist die eine Stelle, die diesen Zustand besitzt:

     - erkennt Limit-Antworten (ein Detektor, nicht drei),
     - prueft VOR dem Absenden gegen die Kontingent-Anzeige, damit der Lauf
       gar nicht erst scheinbar losgeht,
     - rendert eine bleibende Karte im Thread (#runBlocked), genau dort, wo
       sonst die Antwort stuende.

   Die Karte verkauft nichts. consens.io ist waehrend des Tests gratis, es
   gibt also keinen Kauf-Ausweg — sie sagt, wann das Kontingent
   zurueckkommt, und bietet an, was jetzt geht.
   ========================================================================== */
(function () {
  "use strict";

  window.App = window.App || {};

  function el(id) {
    return document.getElementById(id);
  }

  // --- Erkennung -------------------------------------------------------
  // FastAPI verpackt HTTPException-Details in {detail: {...}}; die Streams
  // reichen das Objekt teils schon ausgepackt weiter. Beides muss hier
  // ankommen duerfen.
  function unwrap(data) {
    var detail = data && data.detail;
    if (detail && typeof detail === "object") {
      var merged = {};
      Object.keys(detail).forEach(function (key) { merged[key] = detail[key]; });
      merged.error = detail.error || detail.message || "";
      return merged;
    }
    return data || {};
  }

  // Der Server kennt zwei Codes: "total_usage_limit_exceeded" und
  // "deep_think_usage_limit_exceeded" (chat.py, reserve_usage_run). Ein
  // Vergleich auf "usage_limit_exceeded" trifft deshalb nie — genau der
  // Fehler stand bis hierher in consensus-run.js.
  function isLimitError(data, message) {
    var normalized = unwrap(data);
    var code = String(normalized.error_code || normalized.code || "").toLowerCase();
    var text = String(message || normalized.error || normalized.detail || "").toLowerCase();
    return code.indexOf("limit") !== -1
      || text.indexOf("usage limit") !== -1
      || text.indexOf("quota") !== -1
      || text.indexOf("used up") !== -1
      || text.indexOf("exhausted") !== -1;
  }

  function bucketOf(data) {
    var normalized = unwrap(data);
    var code = String(normalized.error_code || "").toLowerCase();
    if (code.indexOf("deep_think") === 0) return "deep_think";
    if (code.indexOf("total") === 0) return "total";
    // Ohne eindeutigen Code entscheidet, welcher Topf tatsaechlich leer ist.
    if (normalized.deep_remaining === 0 && normalized.free_usage_remaining > 0) {
      return "deep_think";
    }
    return "total";
  }

  // --- Kontingent-Spiegel ----------------------------------------------
  // Dieselbe Quelle wie Ring und Panel (#usageDisplay ueber sidebar-quota),
  // damit die Karte nie eine andere Zahl behauptet als der Ring daneben.
  // null = noch unbekannt (Gast, oder noch nicht geladen).
  function quota(name) {
    try {
      var api = window.App.sidebarQuota;
      if (!api) return null;
      return name === "deep" ? (api.deep ? api.deep() : null) : api.runs();
    } catch (err) {
      return null;
    }
  }

  // Das Kontingent laeuft auf UTC-Tagen (usage_repository.py). Der
  // Countdown in der Sidebar rechnet historisch gegen lokale Mitternacht;
  // hier steht die belastbare Zahl, weil sie neben einer Absage steht.
  function resetInfo() {
    var now = new Date();
    var next = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1, 0, 0, 0, 0
    ));
    var ms = next - now;
    var hours = Math.floor(ms / 3600000);
    var minutes = Math.floor((ms % 3600000) / 60000);
    var relative = hours > 0
      ? hours + " h " + minutes + " min"
      : Math.max(1, minutes) + " min";
    var clock;
    try {
      clock = next.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch (err) {
      clock = "midnight UTC";
    }
    return { relative: relative, clock: clock, ms: ms };
  }

  /* "0 of 3 runs left today · resets at 02:00 (in 1 h 58 min)".
     Zahlen zuerst aus der Server-Absage (die frischeste Quelle), sonst aus
     demselben Spiegel, den der Ring liest. Ohne belastbare Zahl bleibt nur
     die Reset-Zeit stehen — eine erfundene Restzahl waere schlimmer als
     keine. */
  function metaLine(data, remainingKey, limitKey, mirrorName, noun) {
    var remaining = data ? data[remainingKey] : undefined;
    var limit = data ? data[limitKey] : undefined;
    if (remaining === undefined || remaining === null
      || limit === undefined || limit === null) {
      var mirror = quota(mirrorName);
      if (mirror && !mirror.unlimited) {
        remaining = mirror.value;
        limit = mirror.limit;
      } else {
        remaining = null;
      }
    }
    var reset = resetInfo();
    var tail = "resets at " + reset.clock + " (in " + reset.relative + ")";
    if (remaining === null || remaining === undefined) return tail;
    return remaining + " of " + limit + " " + noun + " left today · " + tail;
  }

  // --- Rendering -------------------------------------------------------

  function clearActions() {
    var actions = el("runBlockedActions");
    if (actions) actions.innerHTML = "";
    return actions;
  }

  function addAction(actions, label, title, handler, variant) {
    if (!actions) return null;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "run-blocked-btn" + (variant ? " is-" + variant : "");
    button.textContent = label;
    if (title) button.title = title;
    button.addEventListener("click", handler);
    actions.appendChild(button);
    return button;
  }

  function hide() {
    var card = el("runBlocked");
    if (!card) return;
    card.hidden = true;
    card.classList.remove("is-visible");
    clearActions();
  }

  function render(view) {
    var card = el("runBlocked");
    if (!card) {
      // Ohne die Karte darf der Zustand trotzdem nicht verschwinden.
      window.App.showPopup && window.App.showPopup(view.title + " " + view.body);
      return;
    }

    card.dataset.bucket = view.bucket || "total";

    var title = el("runBlockedTitle");
    if (title) title.textContent = view.title;

    var body = el("runBlockedBody");
    if (body) body.textContent = view.body;

    var meta = el("runBlockedMeta");
    if (meta) {
      meta.textContent = view.meta || "";
      meta.hidden = !view.meta;
    }

    var actions = clearActions();
    (view.actions || []).forEach(function (action) {
      addAction(actions, action.label, action.title, action.onClick, action.variant);
    });

    card.hidden = false;
    // Zwei Frames, damit der Einblend-Uebergang auch beim ersten Zeigen
    // greift (dieselbe Mechanik wie beim gefuehrten Lauf).
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { card.classList.add("is-visible"); });
    });
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // --- Die eigentliche Absage ------------------------------------------

  function deepThinkIsOn() {
    var toggle = el("deepSearchToggle");
    return !!(toggle && toggle.checked);
  }

  // Das Kontingent-Panel haengt im Sidebar-Fuss. Bei zugeklappter Sidebar
  // wuerde setOpen(true) etwas Unsichtbares oeffnen — also erst die Sidebar
  // ueber ihren eigenen Toggle aufmachen (eine Mechanik, nicht zwei) und
  // danach das Panel.
  function openQuotaPanel() {
    var sidebar = document.querySelector(".sidebar");
    var collapsed = sidebar
      && sidebar.classList.contains("collapsed")
      && !sidebar.classList.contains("active");
    if (collapsed) {
      var toggle = document.querySelector(".sidebar-toggle");
      if (toggle) toggle.click();
    }
    requestAnimationFrame(function () {
      if (window.App.sidebarQuota) window.App.sidebarQuota.setOpen(true);
      var trigger = el("quotaTrigger");
      if (trigger && !trigger.hidden) trigger.focus();
    });
  }

  function buildView(info) {
    var data = unwrap(info && info.data);
    var bucket = info && info.bucket ? info.bucket : bucketOf(data);
    var phase = info && info.phase;
    var view = { bucket: bucket, actions: [] };

    if (bucket === "deep_think") {
      var runsLeft = quota("runs");
      var hasNormalRuns = !!runsLeft && (runsLeft.unlimited || runsLeft.value > 0);

      view.title = "No Deep Think runs left today";
      view.body = hasNormalRuns
        ? "Deep Think puts the reasoning models on your question and has its own, smaller allowance — that one is used up. A normal run is still available, and your question is still in the box."
        : "Deep Think has its own, smaller allowance and it is used up for today.";
      view.meta = metaLine(data, "deep_remaining", "deep_limit", "deep", "Deep Think runs");

      if (hasNormalRuns) {
        view.actions.push({
          label: "Send without Deep Think",
          title: "Switch Deep Think off and send this question as a normal run.",
          variant: "primary",
          onClick: function () {
            var toggle = el("deepSearchToggle");
            if (toggle && toggle.checked) {
              toggle.checked = false;
              toggle.dispatchEvent(new Event("change", { bubbles: true }));
            }
            hide();
            track("deep_think_downgrade");
            if (typeof window.sendQuestion === "function") window.sendQuestion();
          }
        });
      }
    } else {
      view.title = "You are out of runs for today";
      view.body = phase === "consensus"
        ? "The models answered, but the consensus could not be written: your daily allowance ran out during this run. The individual answers are still there under “Compare answers”."
        : "A run asks every selected model and then writes the consensus, and your daily allowance for that is used up. Nothing was sent, and your question is still in the box.";
      view.meta = metaLine(data, "free_usage_remaining", "limit", "runs", "runs");
    }

    // Immer erreichbar: die Zahlen selbst. Der Ring im Sidebar-Fuss ist der
    // Ort, an dem das Kontingent lebt — die Karte schickt dorthin, statt
    // eine zweite Rechnung danebenzustellen.
    view.actions.push({
      label: "See your allowance",
      title: "Open the quota panel in the sidebar.",
      onClick: function () {
        track("open_quota");
        openQuotaPanel();
      }
    });

    return view;
  }

  function track(action) {
    try {
      window.App.trackAppEvent && window.App.trackAppEvent("app_usage_limit_blocked", {
        action: action
      });
    } catch (err) { /* Telemetrie darf die Meldung nie verhindern */ }
  }

  /* Zeigt die Absage. info: {data, bucket, phase, source}
     Gibt die verwendete Auspraegung zurueck, damit Aufrufer sie loggen
     koennen. */
  function show(info) {
    var view = buildView(info || {});
    render(view);
    try {
      window.App.trackAppEvent && window.App.trackAppEvent("app_usage_limit_shown", {
        bucket: view.bucket,
        source: (info && info.source) || "unknown",
        phase: (info && info.phase) || "prepare"
      });
    } catch (err) { /* s. o. */ }
    return view;
  }

  // Firestore can abort a hot transaction even after its own retry budget.
  // This is deliberately separate from quota exhaustion: nothing has been
  // denied permanently and the user should be able to retry the same question.
  function showTemporaryStorageBusy() {
    render({
      bucket: "temporary",
      title: "Temporarily unable to start this run",
      body: "The usage service is busy right now. No model was asked; please try this question again in a moment.",
      actions: [{
        label: "Try again",
        variant: "primary",
        title: "Retry this question with a fresh usage reservation.",
        onClick: function () {
          hide();
          if (typeof window.sendQuestion === "function") window.sendQuestion();
        }
      }]
    });
    try {
      window.App.trackAppEvent && window.App.trackAppEvent("app_usage_storage_busy");
    } catch (err) { /* Telemetrie darf die Meldung nie verhindern */ }
  }

  /* Vor dem Absenden: gibt einen Grund zurueck, wenn der Lauf sicher nicht
     durchgeht, sonst null. Bewusst konservativ — bei unbekanntem Stand
     (Gast, noch nicht geladen, eigene Keys) entscheidet weiterhin der
     Server. Lieber ein Server-Nein als ein falsches Client-Nein. */
  function preflight(options) {
    var opts = options || {};
    if (opts.useOwnKeys) return null;

    var runs = quota("runs");
    if (runs && !runs.unlimited && runs.value <= 0) return "total";

    if (opts.deepThink || deepThinkIsOn()) {
      var deep = quota("deep");
      if (deep && !deep.unlimited && deep.value <= 0) return "deep_think";
    }
    return null;
  }

  /* Blockiert den Versuch, falls kein Kontingent mehr da ist. true = der
     Aufrufer soll abbrechen. */
  function blockIfExhausted(options) {
    var bucket = preflight(options);
    if (!bucket) return false;
    show({
      bucket: bucket,
      source: (options && options.source) || "preflight",
      phase: "preflight"
    });
    return true;
  }

  var closeButton = el("runBlockedClose");
  if (closeButton) {
    closeButton.addEventListener("click", function () {
      hide();
      track("dismiss");
      var input = el("questionInput");
      if (input && !input.disabled) input.focus();
    });
  }

  // "New comparison" raeumt den Thread — die Absage gehoert zum geraeumten
  // Lauf und darf nicht ueber ihn hinaus stehenbleiben. Delegiert, weil der
  // Knopf auch aus dem Composer-Gate heraus geklickt wird.
  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest("#newRunButton")) hide();
  });

  window.App.usageLimit = {
    isLimitError: isLimitError,
    bucketOf: bucketOf,
    show: show,
    showTemporaryStorageBusy: showTemporaryStorageBusy,
    hide: hide,
    preflight: preflight,
    blockIfExhausted: blockIfExhausted
  };
})();
