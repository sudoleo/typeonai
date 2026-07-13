// Consensus Watch UI. Classic script/module contract: exports window.openWatchDialog
// (create dialog + dashboard) and window.openWatchDashboard (dashboard only).
(function () {
  const FEATURE_NUDGE_STORAGE_KEY = "consensio.watchFeatureNudge.dismissed.v1";
  let featureNudgeTimer = null;

  function featureNudgeWasDismissed() {
    try {
      return localStorage.getItem(FEATURE_NUDGE_STORAGE_KEY) === "true";
    } catch (_) {
      return false;
    }
  }

  function dismissWatchFeatureNudge(reason) {
    const nudge = document.getElementById("watchFeatureNudge");
    const wasActive = Boolean(featureNudgeTimer || nudge);
    if (featureNudgeTimer) {
      clearTimeout(featureNudgeTimer);
      featureNudgeTimer = null;
    }
    const anchor = document.querySelector(".watch-feature-anchor");
    if (nudge) nudge.remove();
    if (anchor) {
      anchor.classList.remove("has-feature-nudge");
      anchor.closest("h2")?.classList.remove("has-watch-feature-nudge");
    }
    try {
      localStorage.setItem(FEATURE_NUDGE_STORAGE_KEY, "true");
    } catch (_) {
      // Storage can be unavailable in hardened/private browser contexts.
    }
    if (wasActive) {
      window.App?.trackAppEvent?.("app_watch_feature_nudge_dismissed", {
        reason: reason || "dismissed"
      });
    }
  }

  function showWatchFeatureNudge() {
    if (featureNudgeWasDismissed() || featureNudgeTimer
        || document.getElementById("watchFeatureNudge")) return;
    // Only promote an immediately usable action. Guests and failed snapshot
    // persistence keep the normal Watch button without a marketing nudge.
    if (!window.auth?.currentUser || !window.lastShareResultId) return;

    featureNudgeTimer = setTimeout(() => {
      featureNudgeTimer = null;
      if (featureNudgeWasDismissed() || !window.auth?.currentUser
          || !window.lastShareResultId) return;
      const anchor = document.querySelector(".watch-feature-anchor");
      if (!anchor || document.getElementById("watchFeatureNudge")) return;

      const nudge = document.createElement("span");
      nudge.id = "watchFeatureNudge";
      nudge.className = "watch-feature-nudge";
      nudge.setAttribute("role", "status");
      nudge.setAttribute("aria-label", "New Consensus Watch feature");
      nudge.innerHTML = `
        <button type="button" class="watch-feature-nudge-close" aria-label="Dismiss new feature tip">&times;</button>
        <span class="watch-feature-nudge-label">New</span>
        <strong>Track this consensus</strong>
        <span class="watch-feature-nudge-copy">Rerun the question automatically and get notified when the consensus changes.</span>
      `;
      nudge.querySelector(".watch-feature-nudge-close").addEventListener("click", event => {
        event.stopPropagation();
        dismissWatchFeatureNudge("dismissed");
      });
      anchor.classList.add("has-feature-nudge");
      anchor.closest("h2")?.classList.add("has-watch-feature-nudge");
      anchor.appendChild(nudge);
      window.App?.trackAppEvent?.("app_watch_feature_nudge_shown");
    }, 650);
  }

  function els() {
    return {
      modal: document.getElementById("shareModal"),
      title: document.getElementById("shareModalTitle"),
      body: document.getElementById("shareModalBody")
    };
  }

  function closeDialog() {
    const { modal } = els();
    if (modal) modal.style.display = "none";
  }

  function popup(message) {
    window.App?.showPopup?.(message);
  }

  async function api(method, path, body) {
    const user = window.auth?.currentUser;
    if (!user) throw new Error("Please log in first.");
    const token = await user.getIdToken();
    const response = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + token },
      body: body ? JSON.stringify(body) : undefined
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (!response.ok) {
      const error = new Error(data.error || data.detail || ("HTTP " + response.status));
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function intervalOptions(selected) {
    const dailyDisabled = !window.isUserPro;
    return `
      <option value="weekly"${selected === "weekly" ? " selected" : ""}>Weekly</option>
      <option value="monthly"${selected === "monthly" ? " selected" : ""}>Monthly</option>
      <option value="daily"${selected === "daily" ? " selected" : ""}${dailyDisabled ? " disabled" : ""}>Daily${dailyDisabled ? " (Pro)" : ""}</option>
    `;
  }

  function browserTimezone() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    } catch (_) {
      return "UTC";
    }
  }

  function emailModeOptions(selected) {
    return `
      <option value="changes_only"${selected === "changes_only" || !selected ? " selected" : ""}>Material changes only</option>
      <option value="condition"${selected === "condition" ? " selected" : ""}>When my condition is met</option>
      <option value="every_run"${selected === "every_run" ? " selected" : ""}>Every new consensus (with content)</option>
    `;
  }

  function conditionField(value) {
    return `<div id="watchConditionWrap" hidden>
      <label class="watch-interval-label" for="watchCondition">Condition</label>
      <textarea id="watchCondition" class="watch-condition-input" maxlength="500" rows="3" placeholder="Example: An official launch date for Germany is announced.">${escapeHtml(value)}</textarea>
      <p class="watch-data-note">The condition is checked against each new consensus. You receive one e-mail when it changes from not met to met.</p>
    </div>`;
  }

  function bindConditionVisibility(select, wrapper) {
    if (!select || !wrapper) return;
    const sync = () => { wrapper.hidden = select.value !== "condition"; };
    select.addEventListener("change", sync);
    sync();
  }

  function openWatchDialog(view) {
    if (!window.auth?.currentUser) {
      popup("Please log in to use Consensus Watch.");
      return;
    }
    if (view === "list") {
      openWatchDashboard();
      return;
    }
    const { modal } = els();
    if (!modal) return;
    modal.style.display = "block";
    renderConfirm();
  }

  function renderConfirm() {
    const { title, body } = els();
    if (!body) return;
    title.textContent = "Watch this consensus";
    body.innerHTML = `
      <p>consens.io will rerun the <strong>original question</strong> at your chosen interval and apply your selected e-mail rule after each successful run.</p>
      <label class="watch-interval-label" for="watchVisibility">Watch page visibility</label>
      <select id="watchVisibility" class="watch-interval-select" required>
        <option value="" selected disabled>Choose who can open the page…</option>
        <option value="private">Private — only my account</option>
        <option value="public">Public — anyone with the link</option>
      </select>
      <p class="watch-data-note">Private pages require you to be signed in. Public pages are read-only and non-indexed unless reviewed separately.</p>
      <label class="watch-interval-label" for="watchInterval">Check interval ${window.isUserPro ? "" : '<span class="pro-badge is-subtle">Pro: daily</span>'}</label>
      <select id="watchInterval" class="watch-interval-select">${intervalOptions("weekly")}</select>
      <label class="watch-interval-label" for="watchRunTime">Run time</label>
      <input id="watchRunTime" class="watch-time-input" type="time" value="09:00" required>
      <p class="watch-data-note">Local time in <span id="watchTimezoneLabel"></span>. The first run is scheduled after the selected interval and normally starts within 30 minutes after this time.</p>
      <label class="watch-interval-label" for="watchEmailMode">E-mail notifications</label>
      <select id="watchEmailMode" class="watch-interval-select watch-email-select">${emailModeOptions("changes_only")}</select>
      ${conditionField("")}
      <p class="watch-data-note">“Every new consensus” includes the newly generated consensus text in each successful-run e-mail.</p>
      <p class="watch-data-note">Only the question is rerun. Attachments and follow-up context are never resent.</p>
      <div class="share-modal-actions">
        <button type="button" id="watchConfirmBtn" class="share-primary-btn">Start watching</button>
        <button type="button" id="watchCancelBtn" class="share-secondary-btn">Cancel</button>
        <button type="button" id="watchListLink" class="share-link-btn">Open dashboard</button>
      </div>`;
    document.getElementById("watchCancelBtn").addEventListener("click", closeDialog);
    document.getElementById("watchListLink").addEventListener("click", () => {
      closeDialog();
      openWatchDashboard();
    });
    document.getElementById("watchTimezoneLabel").textContent = browserTimezone();
    bindConditionVisibility(
      document.getElementById("watchEmailMode"),
      document.getElementById("watchConditionWrap")
    );
    const confirm = document.getElementById("watchConfirmBtn");
    if (!window.lastShareResultId) {
      confirm.disabled = true;
      confirm.textContent = "Run a consensus first";
    }
    confirm.addEventListener("click", async function () {
      if (!window.lastShareResultId) return;
      const visibility = document.getElementById("watchVisibility").value;
      const emailMode = document.getElementById("watchEmailMode").value;
      const condition = document.getElementById("watchCondition").value.trim();
      const runTime = document.getElementById("watchRunTime").value;
      if (!visibility) {
        popup("Choose whether the watch page is private or public.");
        return;
      }
      if (emailMode === "condition" && !condition) {
        popup("Enter the condition you want to monitor.");
        document.getElementById("watchCondition").focus();
        return;
      }
      if (!runTime) {
        popup("Choose a run time.");
        return;
      }
      this.disabled = true;
      this.textContent = "Starting…";
      try {
        const data = await api("POST", "/api/watch", {
          result_id: window.lastShareResultId,
          interval: document.getElementById("watchInterval").value,
          email_mode: emailMode,
          condition: condition,
          visibility: visibility,
          run_time: runTime,
          timezone: browserTimezone()
        });
        window.App?.trackAppEvent?.("app_watch_created", { interval: data.watch.interval });
        renderSuccess(data.watch);
      } catch (error) {
        this.disabled = false;
        this.textContent = "Start watching";
        if (error.status === 429) window.App?.showProFeatureModal?.("More Consensus Watches");
        popup("Watch could not be started: " + error.message);
      }
    });
  }

  function renderSuccess(watch) {
    const { title, body } = els();
    title.textContent = "Consensus Watch is active";
    const url = window.location.origin + (watch.share_path || "");
    body.innerHTML = `
      <p>We will check this question <strong></strong>. <span id="watchMailSummary"></span></p>
      <div class="share-modal-actions">
        <a id="watchOpenLink" class="share-secondary-btn" target="_blank" rel="noopener">Open history page</a>
        <button type="button" id="watchListLink" class="share-link-btn">Open dashboard</button>
      </div>`;
    body.querySelector("p strong").textContent = watch.run_time
      ? `${watch.interval} at ${watch.run_time} (${watch.timezone})`
      : watch.interval;
    document.getElementById("watchMailSummary").textContent = watch.email_mode === "every_run"
      ? "You will receive every new consensus including its content."
      : watch.email_mode === "condition"
        ? "You will be notified when your condition becomes true."
        : "You will be notified only after a material change.";
    body.querySelector("p").appendChild(document.createTextNode(
      watch.visibility === "private" ? " The history page is private." : " The history page is public."
    ));
    document.getElementById("watchOpenLink").href = url;
    document.getElementById("watchListLink").addEventListener("click", () => {
      closeDialog();
      openWatchDashboard();
    });
  }

  function makeButton(label, className, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.addEventListener("click", handler);
    return button;
  }

  // ------------------------------------------------------------------
  // Watch page (/app/watches): full view below the top bar with URL sync.
  // ------------------------------------------------------------------

  const WATCH_PAGE_PATH = "/app/watches";
  const APP_PATH = "/app";

  function dashEls() {
    return {
      page: document.getElementById("watchDashboard"),
      body: document.getElementById("watchDashBody"),
      close: document.getElementById("watchDashClose")
    };
  }

  function onWatchPagePath() {
    return window.location.pathname === WATCH_PAGE_PATH;
  }

  function wireWatchPage() {
    const { page, close } = dashEls();
    if (!page || page.dataset.wired) return;
    page.dataset.wired = "1";
    close?.addEventListener("click", closeWatchDashboard);
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && !page.hidden) closeWatchDashboard();
    });
    // Browser-Navigation (Back/Forward) hält Seite und URL synchron.
    window.addEventListener("popstate", () => {
      if (onWatchPagePath()) showWatchPage();
      else page.hidden = true;
    });
  }

  function closeWatchDashboard() {
    const { page } = dashEls();
    if (!page) return;
    page.hidden = true;
    if (!onWatchPagePath()) return;
    // Innerhalb der App geöffnet: echter History-Schritt zurück. Direkt auf
    // /app/watches geladen: URL ohne neuen History-Eintrag auf /app setzen.
    if (window.history.state && window.history.state.watchPage) {
      window.history.back();
    } else {
      window.history.replaceState(null, "", APP_PATH);
    }
  }

  function showWatchPage() {
    const { page } = dashEls();
    if (!page) return;
    wireWatchPage();
    page.hidden = false;
    renderDashboard();
  }

  function openWatchDashboard() {
    if (!window.auth?.currentUser) {
      popup("Please log in to use Consensus Watch.");
      return;
    }
    if (!onWatchPagePath()) {
      window.history.pushState({ watchPage: true }, "", WATCH_PAGE_PATH);
    }
    showWatchPage();
  }

  function initWatchPageRoute() {
    // Deep-Link /app/watches: Seite sofort zeigen, auf den asynchronen
    // Firebase-Auth-Status warten und erst dann laden.
    if (!onWatchPagePath()) return;
    const { page, body } = dashEls();
    if (!page || !body) return;
    wireWatchPage();
    page.hidden = false;
    body.innerHTML = '<p class="watch-dash-loading">Checking your session…</p>';
    const startedAt = Date.now();
    (function waitForAuth() {
      if (window.auth?.currentUser) {
        renderDashboard();
        return;
      }
      if (Date.now() - startedAt < 8000) {
        setTimeout(waitForAuth, 250);
        return;
      }
      body.innerHTML = "";
      const hint = document.createElement("div");
      hint.className = "watch-dash-empty";
      hint.textContent = "Please log in to see your Consensus Watch dashboard.";
      body.appendChild(hint);
    })();
  }

  function formatDateTime(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (isNaN(date.getTime())) return "";
    try {
      return new Intl.DateTimeFormat(undefined, {
        weekday: "short", month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit"
      }).format(date);
    } catch (_) {
      return date.toLocaleString();
    }
  }

  function relativeTime(iso) {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (isNaN(then)) return "";
    const diffMs = Date.now() - then;
    const minutes = Math.round(Math.abs(diffMs) / 60000);
    const suffix = diffMs >= 0 ? " ago" : " from now";
    if (minutes < 60) return Math.max(1, minutes) + " min" + suffix;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return hours + " h" + suffix;
    const days = Math.round(hours / 24);
    return days + " d" + suffix;
  }

  function buildSparkline(history) {
    const scores = history
      .map(point => point.agreement_score)
      .filter(score => typeof score === "number");
    if (scores.length < 2) return null;
    const width = 150, height = 42, pad = 4;
    const step = (width - 2 * pad) / (scores.length - 1);
    const y = score => pad + (height - 2 * pad) * (100 - score) / 100;
    const coords = scores.map((score, index) => ({ x: pad + step * index, y: y(score) }));
    const path = coords.map((c, i) => (i ? "L" : "M") + c.x.toFixed(1) + " " + c.y.toFixed(1)).join(" ");
    const area = path + ` L ${coords[coords.length - 1].x.toFixed(1)} ${height - pad} L ${coords[0].x.toFixed(1)} ${height - pad} Z`;
    const dots = history.map((point, index) => {
      if (typeof point.agreement_score !== "number") return "";
      const prev = index ? history[index - 1].agreement_score : null;
      const isEvent = point.changed || (typeof prev === "number" && Math.abs(point.agreement_score - prev) >= 15);
      const last = index === history.length - 1;
      if (!isEvent && !last) return "";
      const c = coords[index];
      if (!c) return "";
      return `<circle class="spark-dot${isEvent && !last ? " event" : ""}" cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="${last ? 3 : 2.4}"></circle>`;
    }).join("");
    const wrapper = document.createElement("div");
    wrapper.className = "watch-sparkline";
    wrapper.title = "Agreement score trend (" + scores.length + " checks)";
    wrapper.innerHTML =
      `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Agreement score trend">` +
      `<path class="spark-area" d="${area}"></path><path class="spark-path" d="${path}"></path>${dots}</svg>`;
    return wrapper;
  }

  function statusChip(watch) {
    const chip = document.createElement("span");
    chip.className = "watch-chip watch-chip-" + watch.status;
    chip.textContent = watch.status === "paused_error" ? "Paused after errors" : watch.status;
    return chip;
  }

  function renderDashboardStats(container, watches) {
    const active = watches.filter(watch => watch.status === "active");
    const nextRuns = active.map(watch => watch.next_run_at).filter(Boolean).sort();
    const weekAgo = Date.now() - 7 * 24 * 3600 * 1000;
    let recentChanges = 0;
    watches.forEach(watch => (watch.history || []).forEach(point => {
      if (point.changed && point.ts && new Date(point.ts).getTime() >= weekAgo) recentChanges += 1;
    }));
    const stats = document.createElement("div");
    stats.className = "watch-dash-stats";
    const chunks = [
      `<span class="watch-dash-stat"><strong>${active.length}</strong> active · ${watches.length - active.length} paused</span>`
    ];
    if (nextRuns.length) {
      chunks.push(`<span class="watch-dash-stat">Next check <strong>${escapeHtml(formatDateTime(nextRuns[0]))}</strong></span>`);
    }
    chunks.push(`<span class="watch-dash-stat"><strong>${recentChanges}</strong> change${recentChanges === 1 ? "" : "s"} in the last 7 days</span>`);
    stats.innerHTML = chunks.join("");
    container.appendChild(stats);
  }

  function renderBriefCard(container, brief) {
    const card = document.createElement("div");
    card.className = "watch-brief-card";
    const timezone = browserTimezone();
    card.innerHTML = `
      <div class="watch-brief-main">
        <label class="watch-brief-title">
          <span class="switch watch-brief-switch">
            <input type="checkbox" id="watchBriefToggle">
            <span class="slider"></span>
          </span>
          Morning Brief
        </label>
        <p class="watch-brief-note">One daily e-mail summarizing all your watches — current agreement,
        changes since the last brief, and upcoming checks. No extra model runs. Times use ${escapeHtml(timezone)}.</p>
      </div>
      <div class="watch-brief-controls" id="watchBriefControls" hidden>
        <input type="time" id="watchBriefTime" class="watch-time-input" aria-label="Brief delivery time">
        <select id="watchBriefMode" class="watch-interval-select" aria-label="Brief frequency">
          <option value="always">Every morning</option>
          <option value="changes_only">Only when something changed</option>
        </select>
      </div>`;
    container.appendChild(card);

    const toggle = card.querySelector("#watchBriefToggle");
    const controls = card.querySelector("#watchBriefControls");
    const timeInput = card.querySelector("#watchBriefTime");
    const modeSelect = card.querySelector("#watchBriefMode");
    toggle.checked = !!brief.enabled;
    controls.hidden = !brief.enabled;
    timeInput.value = brief.send_time || "07:00";
    modeSelect.value = brief.mode || "always";

    async function save(changes, revert) {
      toggle.disabled = timeInput.disabled = modeSelect.disabled = true;
      try {
        const data = await api("PATCH", "/api/my/watch-brief", changes);
        const saved = data.brief || {};
        toggle.checked = !!saved.enabled;
        controls.hidden = !saved.enabled;
        if (saved.send_time) timeInput.value = saved.send_time;
        if (saved.mode) modeSelect.value = saved.mode;
        popup(saved.enabled ? "Morning brief updated." : "Morning brief disabled.");
      } catch (error) {
        popup("Brief update failed: " + error.message);
        if (revert) revert();
      } finally {
        toggle.disabled = timeInput.disabled = modeSelect.disabled = false;
      }
    }

    toggle.addEventListener("change", () => {
      const enabled = toggle.checked;
      save(
        enabled
          ? { enabled: true, send_time: timeInput.value || "07:00", timezone: timezone, mode: modeSelect.value }
          : { enabled: false },
        () => { toggle.checked = !enabled; controls.hidden = enabled; }
      );
    });
    timeInput.addEventListener("change", () => {
      if (!timeInput.value || !toggle.checked) return;
      save({ send_time: timeInput.value, timezone: timezone });
    });
    modeSelect.addEventListener("change", () => {
      if (!toggle.checked) return;
      save({ mode: modeSelect.value });
    });
  }

  function renderWatchCard(watch, onListChanged) {
    const card = document.createElement("li");
    card.className = "watch-card";

    const top = document.createElement("div");
    top.className = "watch-card-top";
    const question = document.createElement("h3");
    question.className = "watch-card-question";
    const link = document.createElement("a");
    link.href = watch.share_path || "#";
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = watch.question || "(untitled)";
    question.appendChild(link);
    const chips = document.createElement("div");
    chips.className = "watch-card-chips";
    chips.appendChild(statusChip(watch));
    const visibility = document.createElement("span");
    visibility.className = "watch-chip";
    visibility.textContent = watch.visibility === "private" ? "Private" : "Public";
    chips.appendChild(visibility);
    top.append(question, chips);

    const bodyRow = document.createElement("div");
    bodyRow.className = "watch-card-body";
    const history = watch.history || [];
    const score = document.createElement("div");
    score.className = "watch-card-score";
    if (typeof watch.last_agreement_score === "number") {
      const previous = history.length >= 2 ? history[history.length - 2].agreement_score : null;
      let deltaHtml = "";
      if (typeof previous === "number" && previous !== watch.last_agreement_score) {
        const delta = watch.last_agreement_score - previous;
        deltaHtml = `<span class="watch-score-delta ${delta > 0 ? "up" : "down"}">${delta > 0 ? "▲" : "▼"} ${Math.abs(delta)}</span>`;
      }
      score.innerHTML = `<strong>${Math.round(watch.last_agreement_score)}</strong>` +
        `<span class="watch-score-max">/100 agreement</span>${deltaHtml}`;
    } else {
      score.innerHTML = '<span class="watch-score-empty">No check completed yet</span>';
    }
    bodyRow.appendChild(score);
    const sparkline = buildSparkline(history);
    if (sparkline) bodyRow.appendChild(sparkline);

    const meta = document.createElement("div");
    meta.className = "watch-card-meta";
    const metaLines = [];
    const lastEvent = [...history].reverse().find(point => point.changed && point.change_summary);
    if (lastEvent) {
      metaLines.push(
        `<span class="watch-meta-change${lastEvent.severity === "major" ? " major" : ""}">` +
        `${escapeHtml(lastEvent.change_summary)}</span> <span>(${escapeHtml(relativeTime(lastEvent.ts))}${lastEvent.severity ? ", " + escapeHtml(lastEvent.severity) : ""})</span>`
      );
    } else if (watch.last_run_at) {
      metaLines.push(`Last check ${escapeHtml(relativeTime(watch.last_run_at))} — no material change.`);
    }
    const scheduleBits = [watch.interval];
    if (watch.run_time) scheduleBits.push("at " + watch.run_time + (watch.timezone ? " (" + watch.timezone + ")" : ""));
    let scheduleLine = escapeHtml(scheduleBits.join(" "));
    if (watch.status === "active" && watch.next_run_at) {
      scheduleLine += " · next check " + escapeHtml(formatDateTime(watch.next_run_at));
    }
    metaLines.push(scheduleLine);
    if (watch.email_mode === "condition" && watch.condition) {
      const state = watch.last_condition_status === "met" ? "met"
        : watch.last_condition_status === "not_met" ? "not met" : "not evaluated yet";
      metaLines.push(`Condition (${escapeHtml(state)}): ${escapeHtml(watch.condition)}`);
    }
    meta.innerHTML = metaLines.join("<br>");
    bodyRow.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "watch-card-actions";
    const settingsBtn = makeButton("Settings", "share-link-btn", () => {
      settings.hidden = !settings.hidden;
      settingsBtn.textContent = settings.hidden ? "Settings" : "Hide settings";
    });
    const openBtn = document.createElement("a");
    openBtn.className = "share-secondary-btn";
    openBtn.href = watch.share_path || "#";
    openBtn.target = "_blank";
    openBtn.rel = "noopener";
    openBtn.textContent = "Open history page";
    const spacer = document.createElement("span");
    spacer.className = "watch-card-spacer";
    const active = watch.status === "active";
    const pause = makeButton(active ? "Pause" : "Resume", "share-secondary-btn", async () => {
      pause.disabled = true;
      try {
        await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), { status: active ? "paused" : "active" });
        onListChanged();
      } catch (error) {
        pause.disabled = false;
        if (error.status === 429) window.App?.showProFeatureModal?.("More Consensus Watches");
        popup("Update failed: " + error.message);
      }
    });
    const remove = makeButton("Delete", "share-danger-btn", async () => {
      if (!confirm("Delete this watch? Its existing history page will remain available with its current visibility.")) return;
      remove.disabled = true;
      try {
        await api("DELETE", "/api/watch/" + encodeURIComponent(watch.id));
        onListChanged();
      } catch (error) { remove.disabled = false; popup("Delete failed: " + error.message); }
    });
    actions.append(settingsBtn, openBtn, spacer, pause, remove);

    const settings = document.createElement("div");
    settings.className = "watch-card-settings";
    settings.hidden = true;
    const grid = document.createElement("div");
    grid.className = "watch-card-settings-grid";

    function field(labelText, control) {
      const wrap = document.createElement("label");
      wrap.className = "watch-field";
      const label = document.createElement("span");
      label.textContent = labelText;
      wrap.append(label, control);
      return wrap;
    }

    const select = document.createElement("select");
    select.className = "watch-interval-select";
    select.innerHTML = intervalOptions(watch.interval);
    select.addEventListener("change", async () => {
      select.disabled = true;
      try {
        await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), { interval: select.value });
        popup("Watch interval updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        select.value = watch.interval;
      } finally { select.disabled = false; }
    });
    const timeInput = document.createElement("input");
    timeInput.type = "time";
    timeInput.className = "watch-time-input";
    timeInput.value = watch.run_time || "";
    timeInput.title = watch.timezone ? `Run time (${watch.timezone})` : "Choose a local run time";
    timeInput.addEventListener("change", async () => {
      if (!timeInput.value) return;
      timeInput.disabled = true;
      const previous = watch.run_time || "";
      try {
        const data = await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), {
          run_time: timeInput.value,
          timezone: browserTimezone()
        });
        watch.run_time = data.watch.run_time;
        watch.timezone = data.watch.timezone;
        popup("Watch run time updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        timeInput.value = previous;
      } finally { timeInput.disabled = false; }
    });
    const emailSelect = document.createElement("select");
    emailSelect.className = "watch-interval-select watch-email-select";
    emailSelect.innerHTML = emailModeOptions(watch.email_mode);
    const conditionEditor = document.createElement("div");
    conditionEditor.className = "watch-condition-editor";
    conditionEditor.hidden = watch.email_mode !== "condition";
    const conditionInput = document.createElement("textarea");
    conditionInput.className = "watch-condition-input";
    conditionInput.maxLength = 500;
    conditionInput.rows = 2;
    conditionInput.placeholder = "Condition to monitor";
    conditionInput.value = watch.condition || "";
    const saveCondition = makeButton("Save condition", "share-secondary-btn", async () => {
      const condition = conditionInput.value.trim();
      if (!condition) {
        popup("Enter the condition you want to monitor.");
        return;
      }
      saveCondition.disabled = true;
      try {
        await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), {
          email_mode: "condition",
          condition: condition
        });
        watch.email_mode = "condition";
        watch.condition = condition;
        popup("Watch condition updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
      } finally { saveCondition.disabled = false; }
    });
    conditionEditor.append(conditionInput, saveCondition);
    emailSelect.addEventListener("change", async () => {
      conditionEditor.hidden = emailSelect.value !== "condition";
      if (emailSelect.value === "condition") {
        conditionInput.focus();
        return;
      }
      emailSelect.disabled = true;
      try {
        await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), { email_mode: emailSelect.value });
        watch.email_mode = emailSelect.value;
        popup("Watch e-mail preference updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        emailSelect.value = watch.email_mode || "changes_only";
      } finally { emailSelect.disabled = false; }
    });

    grid.append(
      field("Interval", select),
      field("Run time", timeInput),
      field("E-mail notifications", emailSelect),
      conditionEditor
    );
    settings.appendChild(grid);

    card.append(top, bodyRow, actions, settings);
    return card;
  }

  async function renderDashboard() {
    const { body } = dashEls();
    if (!body) return;
    body.innerHTML = '<p class="watch-dash-loading">Loading your watches…</p>';
    let watches = [];
    let brief = {};
    try {
      const [watchData, briefData] = await Promise.all([
        api("GET", "/api/my/watches"),
        api("GET", "/api/my/watch-brief").catch(() => ({ brief: {} }))
      ]);
      watches = watchData.watches || [];
      brief = briefData.brief || {};
    } catch (error) {
      body.innerHTML = "";
      const failed = document.createElement("p");
      failed.className = "watch-dash-loading";
      failed.textContent = "Could not load watches: " + error.message;
      body.appendChild(failed);
      return;
    }
    body.innerHTML = "";

    renderDashboardStats(body, watches);

    const briefTitle = document.createElement("h3");
    briefTitle.className = "watch-dash-section-title";
    briefTitle.textContent = "Daily digest";
    body.appendChild(briefTitle);
    renderBriefCard(body, brief);

    const listTitle = document.createElement("h3");
    listTitle.className = "watch-dash-section-title";
    listTitle.textContent = "Watched questions";
    body.appendChild(listTitle);

    if (!watches.length) {
      const empty = document.createElement("div");
      empty.className = "watch-dash-empty";
      empty.innerHTML = "You are not watching any consensus yet.<br>" +
        "Run a question, then use the <strong>Watch</strong> button next to the consensus to track how the model agreement evolves.";
      body.appendChild(empty);
      return;
    }
    const list = document.createElement("ul");
    list.className = "watch-dash-list";
    watches.forEach(watch => list.appendChild(renderWatchCard(watch, renderDashboard)));
    body.appendChild(list);
  }

  function initWatchButton() {
    const bar = document.querySelector("#consensusResponse .consensus-copy-inline");
    if (!bar || document.getElementById("consensusWatchButton")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.id = "consensusWatchButton";
    button.className = "consensus-share-pill watch-consensus-pill";
    button.title = "Watch this consensus for material changes";
    button.setAttribute("aria-label", "Watch this consensus for changes");
    button.innerHTML = '<svg class="share-pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6Z"></path><circle cx="12" cy="12" r="2.5"></circle></svg><span>Watch</span>';
    button.addEventListener("click", () => {
      dismissWatchFeatureNudge("opened");
      openWatchDialog("confirm");
    });
    const anchor = document.createElement("span");
    anchor.className = "watch-feature-anchor";
    anchor.appendChild(button);
    const actions = bar.querySelector(".consensus-actions-wrapper");
    bar.insertBefore(anchor, actions || null);
  }

  function initTopbarWatchesLink() {
    const link = document.getElementById("topbarWatchesLink");
    if (!link) return;
    link.addEventListener("click", event => {
      // SPA-Navigation ohne Reload; Cmd/Ctrl-Klick (neuer Tab) bleibt nativ.
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) return;
      event.preventDefault();
      openWatchDashboard();
    });
  }

  window.openWatchDialog = openWatchDialog;
  window.openWatchDashboard = openWatchDashboard;
  window.App.watch = Object.assign(window.App.watch || {}, {
    showFeatureNudge: showWatchFeatureNudge
  });
  initWatchButton();
  initTopbarWatchesLink();
  initWatchPageRoute();
})();
