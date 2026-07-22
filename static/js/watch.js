// Consensus Watch UI. Classic script/module contract: exports window.openWatchDialog
// (create dialog + dashboard) and window.openWatchDashboard (dashboard only).
(function () {
  const FEATURE_NUDGE_STORAGE_KEY = "consensio.watchFeatureNudge.dismissed.v1";
  const VIEW_SWITCH_HINT_STORAGE_KEY = "consensio.watchViewSwitchHint.seen.v1";
  const WATCH_WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
  ];
  let featureNudgeTimer = null;
  let telegramState = null;
  let watchLimitState = null;
  let watchLimitRequest = null;

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
        <strong>Keep this answer current</strong>
        <span class="watch-feature-nudge-copy">We will recheck this question and notify you only when the consensus materially changes.</span>
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
    if (window.App?.sharedModal?.close) {
      window.App.sharedModal.close();
      return;
    }
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

  async function loadTelegramState(force) {
    if (telegramState && !force) return telegramState;
    const data = await api("GET", "/api/my/telegram");
    telegramState = data.telegram || { configured: false, connected: false };
    return telegramState;
  }

  function normalizeWatchLimits(rawLimits, watches) {
    const list = Array.isArray(watches) ? watches : [];
    const activeFromList = list.filter(watch => watch.status === "active").length;
    const isPro = String(rawLimits?.plan || "").toLowerCase() === "pro"
      || (rawLimits?.plan == null && window.isUserPro === true);
    const configuredLimit = Number(rawLimits?.active_limit);
    const configFallback = Number((window.APP_LIMITS || {})[
      isPro ? "watch_pro_active_limit" : "watch_free_active_limit"
    ]);
    const activeLimit = Number.isFinite(configuredLimit) && configuredLimit >= 0
      ? configuredLimit : (Number.isFinite(configFallback) ? configFallback : 0);
    const configuredActive = Number(rawLimits?.active_count);
    const activeCount = Number.isFinite(configuredActive) && configuredActive >= 0
      ? configuredActive : activeFromList;
    const configuredPaused = Number(rawLimits?.paused_count);
    const pausedCount = Number.isFinite(configuredPaused) && configuredPaused >= 0
      ? configuredPaused : Math.max(0, list.length - activeFromList);
    return {
      plan: isPro ? "pro" : "free",
      activeCount: activeCount,
      activeLimit: activeLimit,
      remaining: Math.max(0, activeLimit - activeCount),
      pausedCount: pausedCount,
      dailyAvailable: rawLimits?.daily_available === true,
      atLimit: activeCount >= activeLimit
    };
  }

  async function loadWatchLimits(force) {
    if (watchLimitState && !force) return watchLimitState;
    if (watchLimitRequest && !force) return watchLimitRequest;
    watchLimitRequest = api("GET", "/api/my/watches")
      .then(data => {
        watchLimitState = normalizeWatchLimits(data.limits, data.watches);
        renderSidebarWatchQuota(watchLimitState);
        return watchLimitState;
      })
      .finally(() => { watchLimitRequest = null; });
    return watchLimitRequest;
  }

  function showWatchUpgrade() {
    window.App?.showProFeatureModal?.("More frequent Consensus Watch checks");
  }

  function renderSidebarWatchQuota(limits) {
    const target = document.getElementById("watchUsageDisplay");
    if (!target || !limits) return;
    target.innerHTML = `Watches: <strong>${limits.activeCount} / ${limits.activeLimit}</strong>`;
  }

  function renderWatchLimit(target, limits) {
    if (!target || !limits) return;
    const isPro = limits.plan === "pro";
    const planLabel = isPro ? "Pro plan" : "Free plan";
    const activeLabel = `${limits.activeCount} of ${limits.activeLimit} active`;
    const availabilityLabel = limits.atLimit
      ? "Limit reached"
      : `${limits.remaining} slot${limits.remaining === 1 ? "" : "s"} available`;
    const meterWidth = limits.activeLimit > 0
      ? Math.min(100, Math.round((limits.activeCount / limits.activeLimit) * 100)) : 100;
    target.hidden = false;
    target.classList.toggle("is-full", limits.atLimit);
    target.innerHTML = `
      <div class="watch-limit-main">
        <span class="watch-limit-plan">${planLabel}</span>
        <strong>${activeLabel}</strong>
        <span>${availabilityLabel}</span>
        <span class="watch-limit-meter" aria-hidden="true"><i style="width:${meterWidth}%"></i></span>
      </div>
      <div class="watch-limit-detail">
        <span>Paused Watches do not count.</span>
        ${isPro ? "" : `<span>Pro beta offers a larger Watch allowance and more frequent checks.</span><button type="button" class="watch-limit-upgrade">Request Pro access</button>`}
      </div>`;
    target.querySelector(".watch-limit-upgrade")?.addEventListener("click", showWatchUpgrade);
  }

  function applyDialogWatchLimit(limits) {
    const target = document.getElementById("watchDialogLimit");
    if (!target || !limits) return;
    renderWatchLimit(target, limits);
    const action = document.getElementById("watchQuestionNext")
      || document.getElementById("watchConfirmBtn");
    if (!action || !limits.atLimit) return;
    action.disabled = true;
    action.textContent = "Watch limit reached";
  }

  function refreshDialogWatchLimit() {
    const target = document.getElementById("watchDialogLimit");
    if (!target) return;
    if (watchLimitState) applyDialogWatchLimit(watchLimitState);
    loadWatchLimits().then(applyDialogWatchLimit).catch(() => {
      if (!target.isConnected) return;
      target.hidden = true;
    });
  }

  async function connectTelegram(onConnected) {
    let pendingWindow = null;
    try {
      pendingWindow = window.open("about:blank", "_blank");
      if (pendingWindow) pendingWindow.opener = null;
      const data = await api("POST", "/api/my/telegram/link", {});
      if (pendingWindow) pendingWindow.location = data.url;
      else window.location.href = data.url;
      popup("Start the bot in Telegram. This page will detect the connection automatically.");
      for (let attempt = 0; attempt < 20; attempt += 1) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        const state = await loadTelegramState(true);
        if (state.connected) {
          popup("Telegram connected.");
          onConnected?.(state);
          return state;
        }
      }
      popup("Connection not detected yet. Finish /start in Telegram, then refresh this page.");
    } catch (error) {
      try { pendingWindow?.close(); } catch (_) { /* ignored */ }
      popup("Telegram connection failed: " + error.message);
    }
    return null;
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

  function browserWeekday() {
    const sundayFirstIndex = new Date().getDay();
    return WATCH_WEEKDAYS[(sundayFirstIndex + 6) % 7];
  }

  function browserTomorrowWeekday() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    return WATCH_WEEKDAYS[(tomorrow.getDay() + 6) % 7];
  }

  function weekdayOptions(selected) {
    const current = WATCH_WEEKDAYS.includes(selected) ? selected : browserWeekday();
    return WATCH_WEEKDAYS.map(day =>
      `<option value="${day}"${day === current ? " selected" : ""}>${day[0].toUpperCase() + day.slice(1)}</option>`
    ).join("");
  }

  function formatWatchSchedule(watch) {
    let label = String(watch.interval || "weekly");
    if (watch.interval === "weekly" && WATCH_WEEKDAYS.includes(watch.run_weekday)) {
      label += " on " + watch.run_weekday[0].toUpperCase() + watch.run_weekday.slice(1);
    }
    if (watch.run_time) {
      label += " at " + watch.run_time + (watch.timezone ? " (" + watch.timezone + ")" : "");
    }
    return label;
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
      <textarea id="watchCondition" class="watch-condition-input" maxlength="500" rows="3" placeholder="Example: An official launch date for Germany is announced." aria-describedby="watchConditionNote watchConditionError">${escapeHtml(value)}</textarea>
      <p id="watchConditionNote" class="watch-data-note">The condition is checked against each new consensus. You receive one alert when it changes from not met to met.</p>
      <p id="watchConditionError" class="watch-field-error" role="alert" hidden></p>
    </div>`;
  }

  function bindConditionVisibility(select, wrapper) {
    if (!select || !wrapper) return;
    const sync = () => {
      wrapper.hidden = select.value !== "condition";
      if (wrapper.hidden) clearWatchFieldError(document.getElementById("watchCondition"));
    };
    select.addEventListener("change", sync);
    sync();
  }

  function bindWeekdayVisibility(intervalSelect, wrapper) {
    if (!intervalSelect || !wrapper) return;
    const sync = () => { wrapper.hidden = intervalSelect.value !== "weekly"; };
    intervalSelect.addEventListener("change", sync);
    sync();
  }

  function clearWatchFieldError(field) {
    if (!field) return;
    field.removeAttribute("aria-invalid");
    const error = document.getElementById(field.id + "Error");
    if (error) {
      error.textContent = "";
      error.hidden = true;
    }
  }

  function setWatchFieldError(field, message) {
    if (!field) return;
    field.setAttribute("aria-invalid", "true");
    const error = document.getElementById(field.id + "Error");
    if (error) {
      error.textContent = message;
      error.hidden = false;
    }
  }

  function focusWatchField(field) {
    if (!field) return;
    const collapsedSection = field.closest("details:not([open])");
    if (collapsedSection) collapsedSection.open = true;
    field.scrollIntoView({ behavior: "smooth", block: "center" });
    try {
      field.focus({ preventScroll: true });
    } catch (_) {
      field.focus();
    }
  }

  function bindWatchFieldErrorReset(field, eventName) {
    if (!field) return;
    field.addEventListener(eventName, () => clearWatchFieldError(field));
  }

  function openWatchDialog(view, options) {
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
    if (window.App?.sharedModal?.open) {
      window.App.sharedModal.open("watch");
    } else {
      modal.classList.add("is-watch-dialog");
      modal.style.display = "flex";
    }
    if (view === "create") renderQuestionStep(options?.question);
    else renderConfirm();
  }

  function normalizeWatchQuestion(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function renderQuestionStep(initialQuestion) {
    const { title, body } = els();
    if (!body) return;
    title.textContent = "Create a Consensus Watch";
    body.innerHTML = `
      <div class="watch-step-label">Step 1 of 2 · Question</div>
      <p class="watch-config-intro">Ask about something that may change over time. The first consensus will run on the schedule you choose next.</p>
      <div id="watchDialogLimit" class="watch-limit-summary is-dialog" aria-live="polite"><span>Checking Watch availability…</span></div>
      <div class="watch-config-field watch-question-field">
        <label class="watch-interval-label" for="watchQuestion">What do you want to monitor?</label>
        <textarea id="watchQuestion" class="watch-condition-input watch-question-input" maxlength="2000" rows="5" placeholder="Example: Has the EU guidance for general-purpose AI models changed?" aria-describedby="watchQuestionNote watchQuestionError">${escapeHtml(initialQuestion || "")}</textarea>
        <p id="watchQuestionNote" class="watch-data-note">Phrase it as a complete, neutral question. You can be specific about a market, policy, product, or time horizon.</p>
        <p id="watchQuestionError" class="watch-field-error" role="alert" hidden></p>
      </div>
      <details class="watch-question-guidance">
        <summary>What makes a useful watch question?</summary>
        <ul>
          <li>Focus on one decision, claim, or development.</li>
          <li>Add relevant scope, such as a country, audience, or timeframe.</li>
          <li>Avoid asking several unrelated questions at once.</li>
        </ul>
      </details>
      <p class="watch-config-assurance"><span aria-hidden="true">✓</span> No model run starts until the Watch reaches its scheduled check.</p>
      <div class="share-modal-actions">
        <button type="button" id="watchQuestionNext" class="share-primary-btn">Continue to schedule</button>
        <button type="button" id="watchCancelBtn" class="share-secondary-btn">Cancel</button>
        <button type="button" id="watchListLink" class="share-link-btn">Open dashboard</button>
      </div>`;
    const input = document.getElementById("watchQuestion");
    bindWatchFieldErrorReset(input, "input");
    document.getElementById("watchCancelBtn").addEventListener("click", closeDialog);
    document.getElementById("watchListLink").addEventListener("click", () => {
      closeDialog();
      openWatchDashboard();
    });
    document.getElementById("watchQuestionNext").addEventListener("click", () => {
      clearWatchFieldError(input);
      const question = normalizeWatchQuestion(input.value);
      if (question.length < 8) {
        setWatchFieldError(input, "Enter a complete question so the models know what to evaluate.");
        focusWatchField(input);
        return;
      }
      renderConfirm({ question: question });
    });
    refreshDialogWatchLimit();
    requestAnimationFrame(() => input.focus());
  }

  function renderConfirm(options) {
    const directQuestion = normalizeWatchQuestion(options?.question);
    const { title, body } = els();
    if (!body) return;
    title.textContent = directQuestion ? "Configure your Watch" : "Watch this consensus";
    body.innerHTML = `
      ${directQuestion ? '<div class="watch-step-label">Step 2 of 2 · Review and delivery</div>' : ""}
      <p class="watch-config-intro">${directQuestion
        ? "Weekly checks and material-change alerts are ready. Choose where we should notify you or customize the schedule."
        : "Weekly checks and material-change alerts are ready for the <strong>original question</strong>."}</p>
      <div id="watchDialogLimit" class="watch-limit-summary is-dialog" aria-live="polite"><span>Checking Watch availability…</span></div>
      ${directQuestion ? `<div class="watch-question-preview"><span>Question</span><strong>${escapeHtml(directQuestion)}</strong></div>` : ""}
      <div class="watch-setup-summary" aria-label="Watch defaults">
        <span class="watch-setup-summary-label">Ready with smart defaults</span>
        <div class="watch-setup-summary-chips">
          <span id="watchScheduleSummary"></span>
          <span id="watchAlertSummary"></span>
          <span id="watchVisibilitySummary"></span>
        </div>
      </div>
      <div class="watch-config-field watch-delivery-field">
        <span class="watch-interval-label">Delivery channels</span>
        <div class="watch-channel-options">
          <label class="watch-channel-option"><input type="checkbox" id="watchEmailEnabled" checked> E-mail</label>
          <label class="watch-channel-option"><input type="checkbox" id="watchTelegramEnabled" disabled> Telegram</label>
          <button type="button" id="watchTelegramConnect" class="share-link-btn">Connect Telegram</button>
        </div>
        <p id="watchTelegramNote" class="watch-data-note">Checking Telegram connection…</p>
        <p id="watchChannelsError" class="watch-field-error" role="alert" hidden></p>
      </div>
      <details id="watchAdvancedSettings" class="watch-advanced-settings">
        <summary><span>Customize schedule and alerts</span><small>Optional</small></summary>
        <div class="watch-advanced-settings-body">
          <div class="watch-config-field">
            <label class="watch-interval-label" for="watchVisibility">Page visibility</label>
            <select id="watchVisibility" class="watch-interval-select" required aria-describedby="watchVisibilityNote watchVisibilityError">
              <option value="private" selected>Private, only my account</option>
              <option value="public">Public, anyone with the link</option>
            </select>
            <p id="watchVisibilityNote" class="watch-data-note">Public pages are read-only and non-indexed by default.</p>
            <p id="watchVisibilityError" class="watch-field-error" role="alert" hidden></p>
          </div>
          <div class="watch-config-grid">
            <div class="watch-config-field">
              <label class="watch-interval-label" for="watchInterval">Interval ${window.isUserPro ? "" : '<span class="pro-badge is-subtle">Pro: daily</span>'}</label>
              <select id="watchInterval" class="watch-interval-select">${intervalOptions("weekly")}</select>
              <div id="watchWeekdayWrap" class="watch-weekday-wrap">
                <label class="watch-interval-label" for="watchWeekday">Run day</label>
                <select id="watchWeekday" class="watch-interval-select">${weekdayOptions(browserTomorrowWeekday())}</select>
              </div>
            </div>
            <div class="watch-config-field">
              <label class="watch-interval-label" for="watchRunTime">Run time</label>
              <input id="watchRunTime" class="watch-time-input" type="time" value="09:00" required aria-describedby="watchRunTimeNote watchRunTimeError">
              <p id="watchRunTimeNote" class="watch-data-note"><span id="watchTimezoneLabel"></span> · checks may begin up to 30 minutes later</p>
              <p id="watchRunTimeError" class="watch-field-error" role="alert" hidden></p>
            </div>
          </div>
          <div class="watch-config-field">
            <label class="watch-interval-label" for="watchEmailMode">Alert rule</label>
            <select id="watchEmailMode" class="watch-interval-select watch-email-select">${emailModeOptions("changes_only")}</select>
            <p class="watch-data-note">“Every new consensus” includes the full generated answer.</p>
            ${conditionField("")}
          </div>
        </div>
      </details>
      <p class="watch-config-assurance"><span aria-hidden="true">✓</span> Attachments and follow-up context are never resent.</p>
      <div class="share-modal-actions">
        <button type="button" id="watchConfirmBtn" class="share-primary-btn">Start watching</button>
        <button type="button" id="watchCancelBtn" class="share-secondary-btn">${directQuestion ? "Back" : "Cancel"}</button>
        <button type="button" id="watchListLink" class="share-link-btn">Open dashboard</button>
      </div>`;
    document.getElementById("watchCancelBtn").addEventListener("click", () => {
      if (directQuestion) renderQuestionStep(directQuestion);
      else closeDialog();
    });
    document.getElementById("watchListLink").addEventListener("click", () => {
      closeDialog();
      openWatchDashboard();
    });
    const visibilitySelect = document.getElementById("watchVisibility");
    const intervalSelect = document.getElementById("watchInterval");
    const weekdaySelect = document.getElementById("watchWeekday");
    const runTimeInput = document.getElementById("watchRunTime");
    const emailModeSelect = document.getElementById("watchEmailMode");
    const conditionInput = document.getElementById("watchCondition");
    const emailEnabledInput = document.getElementById("watchEmailEnabled");
    const telegramEnabledInput = document.getElementById("watchTelegramEnabled");
    const telegramConnect = document.getElementById("watchTelegramConnect");
    const telegramNote = document.getElementById("watchTelegramNote");
    const channelsError = document.getElementById("watchChannelsError");
    document.getElementById("watchTimezoneLabel").textContent = browserTimezone();
    bindConditionVisibility(emailModeSelect, document.getElementById("watchConditionWrap"));
    bindWeekdayVisibility(intervalSelect, document.getElementById("watchWeekdayWrap"));

    function updateSetupSummary() {
      const intervalLabel = intervalSelect.options[intervalSelect.selectedIndex]?.textContent.trim() || "Weekly";
      const weekdayLabel = weekdaySelect.options[weekdaySelect.selectedIndex]?.textContent.trim() || "";
      const scheduleParts = [intervalLabel];
      if (intervalSelect.value === "weekly" && weekdayLabel) scheduleParts.push(weekdayLabel);
      if (runTimeInput.value) scheduleParts.push(runTimeInput.value);
      document.getElementById("watchScheduleSummary").textContent = scheduleParts.join(" · ");
      document.getElementById("watchAlertSummary").textContent =
        emailModeSelect.options[emailModeSelect.selectedIndex]?.textContent.trim() || "Material changes only";
      document.getElementById("watchVisibilitySummary").textContent =
        visibilitySelect.value === "public" ? "Public page" : "Private page";
    }
    [visibilitySelect, intervalSelect, weekdaySelect, emailModeSelect].forEach(input => {
      input.addEventListener("change", updateSetupSummary);
    });
    runTimeInput.addEventListener("input", updateSetupSummary);
    updateSetupSummary();
    function syncTelegram(state) {
      telegramEnabledInput.disabled = !state.connected;
      telegramConnect.hidden = !!state.connected || !state.configured;
      telegramNote.textContent = !state.configured
        ? "Telegram notifications are not available yet."
        : state.connected
          ? `Connected${state.telegram_username ? " as @" + state.telegram_username : ""}.`
          : "Connect Telegram to enable this channel.";
    }
    loadTelegramState().then(syncTelegram).catch(() => {
      syncTelegram({ configured: false, connected: false });
    });
    telegramConnect.addEventListener("click", () => connectTelegram(syncTelegram));
    [emailEnabledInput, telegramEnabledInput].forEach(input => input.addEventListener("change", () => {
      channelsError.hidden = true;
      channelsError.textContent = "";
    }));
    bindWatchFieldErrorReset(visibilitySelect, "change");
    bindWatchFieldErrorReset(runTimeInput, "input");
    bindWatchFieldErrorReset(conditionInput, "input");
    const confirm = document.getElementById("watchConfirmBtn");
    if (!directQuestion && !window.lastShareResultId && window.currentBookmarkShareResultContext) {
      confirm.disabled = true;
      confirm.textContent = "Preparing saved consensus…";
      window.resolveCurrentShareResultId?.().then(resultId => {
        if (!confirm.isConnected) return;
        confirm.disabled = !resultId;
        confirm.textContent = resultId ? "Start watching" : "Saved consensus unavailable";
      });
    } else if (!directQuestion && !window.lastShareResultId) {
      confirm.disabled = true;
      confirm.textContent = "Run a consensus first";
    }
    refreshDialogWatchLimit();
    confirm.addEventListener("click", async function () {
      const resultId = directQuestion ? "" : await (window.resolveCurrentShareResultId?.()
        || Promise.resolve(window.lastShareResultId));
      if (!directQuestion && !resultId) return;
      const visibility = visibilitySelect.value;
      const emailMode = emailModeSelect.value;
      const condition = conditionInput.value.trim();
      const runTime = runTimeInput.value;
      [visibilitySelect, runTimeInput, conditionInput].forEach(clearWatchFieldError);
      const invalidFields = [];
      if (!visibility) {
        setWatchFieldError(visibilitySelect, "Choose whether this page should be private or public.");
        invalidFields.push(visibilitySelect);
      }
      if (!runTime) {
        setWatchFieldError(runTimeInput, "Choose a run time for the automatic check.");
        invalidFields.push(runTimeInput);
      }
      if (emailMode === "condition" && !condition) {
        setWatchFieldError(conditionInput, "Enter the condition you want to monitor.");
        invalidFields.push(conditionInput);
      }
      if (!emailEnabledInput.checked && !telegramEnabledInput.checked) {
        channelsError.textContent = "Keep at least one delivery channel enabled.";
        channelsError.hidden = false;
        invalidFields.push(emailEnabledInput);
      }
      if (invalidFields.length) {
        focusWatchField(invalidFields[0]);
        return;
      }
      this.disabled = true;
      this.textContent = "Starting…";
      try {
        const payload = {
          interval: document.getElementById("watchInterval").value,
          run_weekday: document.getElementById("watchInterval").value === "weekly"
            ? document.getElementById("watchWeekday").value : "",
          email_mode: emailMode,
          email_enabled: emailEnabledInput.checked,
          telegram_enabled: telegramEnabledInput.checked,
          condition: condition,
          visibility: visibility,
          run_time: runTime,
          timezone: browserTimezone()
        };
        if (directQuestion) payload.question = directQuestion;
        else payload.result_id = resultId;
        const data = await api("POST", "/api/watch", payload);
        watchLimitState = null;
        window.App?.trackAppEvent?.("app_watch_created", {
          interval: data.watch.interval,
          source: directQuestion ? "query_first" : "consensus"
        });
        renderSuccess(data.watch);
      } catch (error) {
        this.disabled = false;
        this.textContent = "Start watching";
        if (error.status === 429) {
          watchLimitState = null;
          loadWatchLimits(true).then(applyDialogWatchLimit).catch(() => {});
          if (!window.isUserPro) showWatchUpgrade();
        }
        popup("Watch could not be started: " + error.message);
      }
    });
  }

  function renderSuccess(watch) {
    const { title, body } = els();
    title.textContent = "Your Watch is active";
    const url = window.location.origin + (watch.share_path || "");
    body.innerHTML = `
      <div class="watch-success-card">
        <span class="watch-success-icon" aria-hidden="true">✓</span>
        <div>
          <strong id="watchStartSummary"></strong>
          <p id="watchMailSummary"></p>
          <div id="watchSuccessChips" class="watch-setup-summary-chips"></div>
        </div>
      </div>
      <div class="share-modal-actions">
        <a id="watchOpenLink" class="share-secondary-btn" target="_blank" rel="noopener">Open history page</a>
        <button type="button" id="watchListLink" class="share-link-btn">Open dashboard</button>
      </div>`;
    document.getElementById("watchStartSummary").textContent = watch.query_first
      ? `First check: ${formatWatchSchedule(watch)}`
      : `Next check: ${formatWatchSchedule(watch)}`;
    document.getElementById("watchMailSummary").textContent = watch.email_mode === "every_run"
      ? "You will receive every new consensus, including its content."
      : watch.email_mode === "condition"
        ? "You will be notified when your condition becomes true."
        : "You will be notified only after a material change.";
    const successChips = document.getElementById("watchSuccessChips");
    [
      watch.visibility === "private" ? "Private page" : "Public page",
      watch.email_enabled ? "E-mail" : "",
      watch.telegram_enabled ? "Telegram" : ""
    ].filter(Boolean).forEach(label => {
      const chip = document.createElement("span");
      chip.textContent = label;
      successChips.appendChild(chip);
    });
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
  // Watch page (/app/watches): full view with URL + segmented-switch sync.
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

  function setViewSwitchState(isWatchView) {
    const consensusButton = document.getElementById("viewSwitchConsensus");
    const watchesButton = document.getElementById("viewSwitchWatches");
    if (!consensusButton || !watchesButton) return;

    consensusButton.classList.toggle("is-active", !isWatchView);
    consensusButton.setAttribute("aria-pressed", String(!isWatchView));
    watchesButton.classList.toggle("is-active", isWatchView);
    watchesButton.setAttribute("aria-pressed", String(isWatchView));
  }

  function acknowledgeViewSwitchHint() {
    document.getElementById("viewSwitchWatches")?.classList.remove("has-watch-pulse");
    try { localStorage.setItem(VIEW_SWITCH_HINT_STORAGE_KEY, "true"); } catch (_) {}
  }

  function initViewSwitchHint() {
    const watchesButton = document.getElementById("viewSwitchWatches");
    if (!watchesButton || onWatchPagePath()) return;
    try {
      if (localStorage.getItem(VIEW_SWITCH_HINT_STORAGE_KEY) === "true") return;
    } catch (_) {}
    setTimeout(() => {
      if (!onWatchPagePath() && !document.getElementById("watchFeatureNudge")) {
        watchesButton.addEventListener("animationend", () => {
          watchesButton.classList.remove("has-watch-pulse");
        }, { once: true });
        watchesButton.classList.add("has-watch-pulse");
      }
    }, 1400);
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
      else {
        page.hidden = true;
        setViewSwitchState(false);
      }
    });
  }

  function closeWatchDashboard() {
    const { page } = dashEls();
    if (!page) return;
    page.hidden = true;
    setViewSwitchState(false);
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
    setViewSwitchState(true);
    renderDashboard();
  }

  function openWatchDashboard() {
    if (!window.auth?.currentUser) {
      popup("Please log in to use Consensus Watch.");
      return;
    }
    acknowledgeViewSwitchHint();
    if (!onWatchPagePath()) {
      window.history.pushState({ watchPage: true }, "", WATCH_PAGE_PATH);
    }
    showWatchPage();
  }

  function initWatchPageRoute() {
    // Deep-Link /app/watches: Seite sofort zeigen, auf den asynchronen
    // Firebase-Auth-Status warten und erst dann laden.
    if (!onWatchPagePath()) return;
    acknowledgeViewSwitchHint();
    const { page, body } = dashEls();
    if (!page || !body) return;
    wireWatchPage();
    page.hidden = false;
    setViewSwitchState(true);
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

  function trendHistory(watch) {
    const history = Array.isArray(watch.history) ? [...watch.history] : [];
    if (!watch.query_first && typeof watch.baseline_agreement_score === "number") {
      history.unshift({
        agreement_score: watch.baseline_agreement_score,
        changed: false,
        baseline: true
      });
    }
    return history;
  }

  function buildSparkline(history) {
    const points = history.filter(point => typeof point.agreement_score === "number");
    const width = 150, height = 42, pad = 4;
    const wrapper = document.createElement("div");
    wrapper.className = "watch-sparkline";
    if (!points.length) {
      wrapper.classList.add("is-empty");
      wrapper.title = "Agreement trend starts after the first check";
      wrapper.innerHTML =
        `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Agreement trend awaiting first check">` +
        `<path class="spark-placeholder" d="M ${pad} ${height / 2} L ${width - pad} ${height / 2}"></path></svg>`;
      return wrapper;
    }

    const y = score => pad + (height - 2 * pad) * (100 - score) / 100;
    if (points.length === 1) {
      const score = points[0].agreement_score;
      const scoreY = y(score);
      wrapper.classList.add("is-single");
      wrapper.title = "Agreement baseline; trend starts after the next check";
      wrapper.innerHTML =
        `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Agreement baseline ${Math.round(score)} out of 100">` +
        `<path class="spark-path" d="M ${pad} ${scoreY.toFixed(1)} L ${width - pad} ${scoreY.toFixed(1)}"></path>` +
        `<circle class="spark-dot" cx="${width - pad}" cy="${scoreY.toFixed(1)}" r="3"></circle></svg>`;
      return wrapper;
    }

    const step = (width - 2 * pad) / (points.length - 1);
    const coords = points.map((point, index) => ({
      x: pad + step * index,
      y: y(point.agreement_score)
    }));
    const path = coords.map((c, i) => (i ? "L" : "M") + c.x.toFixed(1) + " " + c.y.toFixed(1)).join(" ");
    const area = path + ` L ${coords[coords.length - 1].x.toFixed(1)} ${height - pad} L ${coords[0].x.toFixed(1)} ${height - pad} Z`;
    const dots = points.map((point, index) => {
      const prev = index ? points[index - 1].agreement_score : null;
      const isEvent = point.changed || (typeof prev === "number" && Math.abs(point.agreement_score - prev) >= 15);
      const last = index === points.length - 1;
      if (!isEvent && !last) return "";
      const c = coords[index];
      return `<circle class="spark-dot${isEvent && !last ? " event" : ""}" cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="${last ? 3 : 2.4}"></circle>`;
    }).join("");
    const includesBaseline = points.some(point => point.baseline);
    wrapper.title = "Agreement score trend (" + points.length + (includesBaseline ? " scores including baseline)" : " checks)");
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

  function latestHistoryPoint(watch) {
    const history = watch.history || [];
    return history.length ? history[history.length - 1] : null;
  }

  function driftState(watch) {
    const point = latestHistoryPoint(watch);
    if (!point) return { key: "baseline", label: "Awaiting first check", summary: "Baseline ready" };
    if (watch.query_first && (watch.history || []).length === 1) {
      return {
        key: "baseline",
        label: "Baseline established",
        summary: "The first scheduled consensus is ready.",
        point: point
      };
    }
    const changed = point.trigger === "changed" || point.changed;
    return {
      key: changed ? "changed" : "stable",
      label: changed ? "Changed" : "Stable",
      summary: point.change_summary || (changed
        ? "Material movement detected in the latest check."
        : "No material movement in the latest check."),
      point: point
    };
  }

  function renderDashboardStats(container, watches) {
    const active = watches.filter(watch => watch.status === "active");
    const nextRuns = active.map(watch => watch.next_run_at).filter(Boolean).sort();
    const weekAgo = Date.now() - 7 * 24 * 3600 * 1000;
    let recentChanges = 0;
    let recentChecks = 0;
    watches.forEach(watch => (watch.history || []).forEach(point => {
      if (!point.ts || new Date(point.ts).getTime() < weekAgo) return;
      recentChecks += 1;
      if (point.trigger === "changed" || point.changed) recentChanges += 1;
    }));
    const stats = document.createElement("div");
    stats.className = "watch-dash-stats";
    stats.innerHTML = `
      <article class="watch-dash-stat"><span>Active Watches</span><strong>${active.length}</strong><small>${watches.length - active.length} paused</small></article>
      <article class="watch-dash-stat is-drift"><span>Changes · 7 days</span><strong>${recentChanges}</strong><small>material movements</small></article>
      <article class="watch-dash-stat"><span>Checks · 7 days</span><strong>${recentChecks}</strong><small>successful runs</small></article>
      <article class="watch-dash-stat"><span>Next check</span><strong class="is-date">${nextRuns.length ? escapeHtml(relativeTime(nextRuns[0])) : "—"}</strong><small>${nextRuns.length ? escapeHtml(formatDateTime(nextRuns[0])) : "No active schedule"}</small></article>`;
    container.appendChild(stats);

    const changes = watches
      .map(watch => ({ watch: watch, state: driftState(watch) }))
      .filter(item => item.state.key === "changed")
      .sort((a, b) => new Date(b.state.point.ts || 0) - new Date(a.state.point.ts || 0))
      .slice(0, 3);
    if (changes.length) {
      const panel = document.createElement("section");
      panel.className = "watch-drift-feed";
      panel.innerHTML = `<div class="watch-drift-feed-head"><div><span class="watch-kicker">Latest changes</span><h2>Recent movement</h2></div><span>${changes.length} update${changes.length === 1 ? "" : "s"}</span></div>`;
      const list = document.createElement("div");
      list.className = "watch-drift-feed-list";
      changes.forEach(({ watch, state }) => {
        const item = document.createElement("a");
        item.href = watch.share_path || "#";
        item.target = "_blank";
        item.rel = "noopener";
        item.innerHTML = `<span class="watch-drift-feed-dot" aria-hidden="true"></span><span><strong>${escapeHtml(watch.question || "Untitled watch")}</strong><small>${escapeHtml(state.summary)}</small></span><time>${escapeHtml(relativeTime(state.point.ts))}</time>`;
        list.appendChild(item);
      });
      panel.appendChild(list);
      container.appendChild(panel);
    }
  }

  function renderBriefCard(container, brief, hasWatches) {
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
        <p class="watch-brief-note">${hasWatches
          ? `One daily e-mail summarizing all your watches, including current agreement, changes since the last brief, and upcoming checks. No extra model runs. Times use ${escapeHtml(timezone)}.`
          : "Create a watch first to activate your daily digest."}</p>
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
    card.classList.toggle("is-disabled", !hasWatches);
    toggle.checked = hasWatches && !!brief.enabled;
    toggle.disabled = !hasWatches;
    controls.hidden = !toggle.checked;
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
        toggle.disabled = !hasWatches;
        timeInput.disabled = modeSelect.disabled = false;
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

  function renderTelegramCard(container, state, onChanged) {
    const card = document.createElement("div");
    card.className = "watch-telegram-card";
    const identity = state.telegram_username
      ? "@" + state.telegram_username
      : (state.telegram_first_name || "your Telegram account");
    card.innerHTML = `
      <div class="watch-telegram-main">
        <strong class="watch-telegram-title">Telegram alerts</strong>
        <p class="watch-telegram-note"></p>
      </div>
      <div class="watch-telegram-actions"></div>`;
    const note = card.querySelector(".watch-telegram-note");
    const actions = card.querySelector(".watch-telegram-actions");
    if (!state.configured) {
      card.classList.add("is-disabled");
      note.textContent = state.linked
        ? "Telegram is linked, but notifications are temporarily unavailable on this consens.io deployment."
        : "Telegram notifications are not configured on this consens.io deployment.";
      if (state.linked) {
        actions.appendChild(makeButton("Disconnect", "share-link-btn", async function () {
          this.disabled = true;
          try {
            await api("DELETE", "/api/my/telegram", {});
            telegramState = null;
            onChanged();
          } catch (error) {
            this.disabled = false;
            popup("Disconnect failed: " + error.message);
          }
        }));
      }
    } else if (state.connected) {
      note.textContent = `Connected to ${identity}. Enable Telegram separately on each watch below.`;
      actions.appendChild(makeButton("Send test", "share-secondary-btn", async function () {
        this.disabled = true;
        try {
          await api("POST", "/api/my/telegram/test", {});
          popup("Test message sent to Telegram.");
        } catch (error) {
          popup("Test failed: " + error.message);
        } finally { this.disabled = false; }
      }));
      actions.appendChild(makeButton("Disconnect", "share-link-btn", async function () {
        if (!confirm("Disconnect Telegram? Watches keep their channel preference but cannot deliver there until you reconnect.")) return;
        this.disabled = true;
        try {
          await api("DELETE", "/api/my/telegram", {});
          telegramState = null;
          onChanged();
        } catch (error) {
          this.disabled = false;
          popup("Disconnect failed: " + error.message);
        }
      }));
    } else {
      note.textContent = "Connect once, then choose Telegram as a delivery channel for any Consensus Watch.";
      actions.appendChild(makeButton("Connect Telegram", "share-primary-btn", async function () {
        this.disabled = true;
        await connectTelegram(() => {
          telegramState = null;
          onChanged();
        });
        if (this.isConnected) this.disabled = false;
      }));
    }
    container.appendChild(card);
  }

  function renderNotificationsPanel(container, telegram, brief, hasWatches) {
    const storageKey = "consensus_watch_notifications_open";
    let isOpen = false;
    try { isOpen = window.localStorage.getItem(storageKey) === "true"; } catch (_) {}

    const panel = document.createElement("details");
    panel.className = "watch-notifications";
    panel.open = isOpen;
    const telegramStatus = telegram.configured
      ? (telegram.connected ? "Telegram connected" : "Telegram not connected")
      : "Telegram unavailable";
    const briefStatus = hasWatches && brief.enabled ? "Morning brief on" : "Morning brief off";
    panel.innerHTML = `
      <summary>
        <span class="watch-notifications-title">Notifications</span>
        <span class="watch-notifications-summary">${telegramStatus} · ${briefStatus}</span>
      </summary>
      <div class="watch-notifications-content"></div>`;
    panel.addEventListener("toggle", () => {
      try { window.localStorage.setItem(storageKey, String(panel.open)); } catch (_) {}
    });

    const content = panel.querySelector(".watch-notifications-content");
    renderTelegramCard(content, telegram, renderDashboard);
    renderBriefCard(content, brief, hasWatches);
    container.appendChild(panel);
  }

  function renderWatchCard(watch, onListChanged, telegram, collapseByDefault = false) {
    const card = document.createElement("li");
    card.className = "watch-card";
    const state = driftState(watch);
    card.dataset.drift = state.key;
    card.dataset.status = watch.status || "paused";

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
    if (watch.telegram_enabled) {
      const telegramChip = document.createElement("span");
      telegramChip.className = "watch-chip watch-chip-telegram";
      telegramChip.textContent = "Telegram";
      chips.appendChild(telegramChip);
    }
    if (watch.visibility !== "private" && (watch.indexed || watch.index_requested)) {
      const listing = document.createElement("span");
      listing.className = "watch-chip" + (watch.indexed ? " watch-chip-listed" : " watch-chip-review");
      listing.textContent = watch.indexed ? "On Google" : "Listing in review";
      chips.appendChild(listing);
    }
    const detailsToggle = makeButton("", "watch-card-toggle", () => {
      setCollapsed(!content.hidden);
    });
    detailsToggle.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m7 10 5 5 5-5"></path></svg>';

    const compactSummary = document.createElement("div");
    compactSummary.className = "watch-card-summary";
    const summaryItems = [
      typeof watch.last_agreement_score === "number"
        ? `${Math.round(watch.last_agreement_score)}/100 agreement`
        : "Awaiting first check",
      state.label,
      watch.last_run_at ? `Last checked ${relativeTime(watch.last_run_at)}` : "No completed check",
    ];
    if (watch.status === "active" && watch.next_run_at) {
      summaryItems.push(`Next ${formatDateTime(watch.next_run_at)}`);
    } else {
      summaryItems.push(formatWatchSchedule(watch));
    }
    compactSummary.innerHTML = summaryItems
      .filter(Boolean)
      .map(item => `<span>${escapeHtml(item)}</span>`)
      .join("");
    top.append(question, chips, detailsToggle, compactSummary);

    const content = document.createElement("div");
    content.className = "watch-card-content";
    function setCollapsed(collapsed) {
      content.hidden = collapsed;
      card.classList.toggle("is-collapsed", collapsed);
      detailsToggle.setAttribute("aria-expanded", String(!collapsed));
      detailsToggle.setAttribute("aria-label", collapsed ? "Show Watch details" : "Hide Watch details");
      detailsToggle.title = collapsed ? "Show details" : "Hide details";
    }
    setCollapsed(collapseByDefault);

    const bodyRow = document.createElement("div");
    bodyRow.className = "watch-card-body";
    const history = watch.history || [];
    const trend = trendHistory(watch);
    const drift = document.createElement("div");
    drift.className = "watch-card-drift is-" + state.key;
    const shiftScore = state.point?.opinion_map?.shift_score;
    drift.innerHTML = `
      <span class="watch-card-drift-dot" aria-hidden="true"></span>
      <span class="watch-card-drift-copy"><strong>${escapeHtml(state.label)}</strong><small>${escapeHtml(state.summary)}</small></span>
      ${typeof shiftScore === "number" ? `<span class="watch-card-shift"><strong>${Math.round(shiftScore)}</strong><small>/100 movement</small></span>` : ""}`;
    bodyRow.appendChild(drift);
    const score = document.createElement("div");
    score.className = "watch-card-score";
    if (typeof watch.last_agreement_score === "number") {
      const previous = trend.length >= 2 ? trend[trend.length - 2].agreement_score : null;
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
    bodyRow.appendChild(buildSparkline(trend));

    const meta = document.createElement("div");
    meta.className = "watch-card-meta";
    const metaLines = [];
    const lastEvent = [...history].reverse().find(point =>
      (point.trigger === "changed" || point.changed) && point.change_summary
    );
    if (lastEvent) {
      metaLines.push(
        `<span class="watch-meta-change${lastEvent.severity === "major" ? " major" : ""}">` +
        `${escapeHtml(lastEvent.change_summary)}</span> <span>(${escapeHtml(relativeTime(lastEvent.ts))}${lastEvent.severity ? ", " + escapeHtml(lastEvent.severity) : ""})</span>`
      );
    } else if (watch.last_run_at) {
      metaLines.push(`Last check ${escapeHtml(relativeTime(watch.last_run_at))}. No material change.`);
    }
    let scheduleLine = escapeHtml(formatWatchSchedule(watch));
    if (watch.status === "active" && watch.next_run_at) {
      scheduleLine += " · next check " + escapeHtml(formatDateTime(watch.next_run_at));
    }
    metaLines.push(scheduleLine);
    if (watch.email_mode === "condition" && watch.condition) {
      const state = watch.last_condition_status === "met" ? "met"
        : watch.last_condition_status === "not_met" ? "not met" : "not evaluated yet";
      metaLines.push(`Condition (${escapeHtml(state)}): ${escapeHtml(watch.condition)}`);
    }
    if (watch.telegram_muted_until && new Date(watch.telegram_muted_until).getTime() > Date.now()) {
      metaLines.push(`Telegram muted until ${escapeHtml(formatDateTime(watch.telegram_muted_until))}.`);
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
    openBtn.textContent = "Open monitor";
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
      const message = watch.awaiting_first_run
        ? "Delete this watch? Because no consensus has run yet, its empty monitor page will also be removed."
        : "Delete this watch? Its existing history page will remain available with its current visibility.";
      if (!confirm(message)) return;
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
    const weekdaySelect = document.createElement("select");
    weekdaySelect.className = "watch-interval-select";
    weekdaySelect.innerHTML = weekdayOptions(watch.run_weekday);
    const weekdayField = field("Run day", weekdaySelect);
    const syncWeekdayField = () => {
      weekdayField.hidden = select.value !== "weekly";
    };
    syncWeekdayField();
    select.addEventListener("change", async () => {
      const previousInterval = watch.interval;
      select.disabled = true;
      weekdaySelect.disabled = true;
      try {
        const data = await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), {
          interval: select.value,
          run_weekday: select.value === "weekly" ? weekdaySelect.value : ""
        });
        watch.interval = data.watch.interval;
        watch.run_weekday = data.watch.run_weekday;
        syncWeekdayField();
        popup("Watch interval updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        select.value = previousInterval;
        syncWeekdayField();
      } finally {
        select.disabled = false;
        weekdaySelect.disabled = false;
      }
    });
    weekdaySelect.addEventListener("change", async () => {
      const previousWeekday = watch.run_weekday || browserWeekday();
      weekdaySelect.disabled = true;
      try {
        const data = await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), {
          run_weekday: weekdaySelect.value
        });
        watch.run_weekday = data.watch.run_weekday;
        popup("Weekly run day updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        weekdaySelect.value = previousWeekday;
      } finally { weekdaySelect.disabled = false; }
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
        popup("Watch alert rule updated.");
      } catch (error) {
        popup("Update failed: " + error.message);
        emailSelect.value = watch.email_mode || "changes_only";
      } finally { emailSelect.disabled = false; }
    });

    const channelOptions = document.createElement("div");
    channelOptions.className = "watch-channel-options is-compact";
    const emailChannel = document.createElement("input");
    emailChannel.type = "checkbox";
    emailChannel.checked = watch.email_enabled !== false;
    const telegramChannel = document.createElement("input");
    telegramChannel.type = "checkbox";
    telegramChannel.checked = watch.telegram_enabled === true;
    telegramChannel.disabled = !telegram?.connected;
    const emailLabel = document.createElement("label");
    emailLabel.className = "watch-channel-option";
    emailLabel.append(emailChannel, document.createTextNode(" E-mail"));
    const telegramLabel = document.createElement("label");
    telegramLabel.className = "watch-channel-option";
    telegramLabel.title = telegram?.connected ? "" : "Connect Telegram in the notification card above.";
    telegramLabel.append(telegramChannel, document.createTextNode(" Telegram"));
    channelOptions.append(emailLabel, telegramLabel);

    async function saveChannel(input, fieldName) {
      const previous = !input.checked;
      if (!emailChannel.checked && !telegramChannel.checked) {
        input.checked = previous;
        popup("Keep at least one notification channel enabled.");
        return;
      }
      emailChannel.disabled = telegramChannel.disabled = true;
      try {
        const data = await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), {
          [fieldName]: input.checked
        });
        watch.email_enabled = data.watch.email_enabled;
        watch.telegram_enabled = data.watch.telegram_enabled;
        popup("Watch delivery channels updated.");
        onListChanged();
      } catch (error) {
        input.checked = previous;
        popup("Update failed: " + error.message);
        emailChannel.disabled = false;
        telegramChannel.disabled = !telegram?.connected;
      }
    }
    emailChannel.addEventListener("change", () => saveChannel(emailChannel, "email_enabled"));
    telegramChannel.addEventListener("change", () => saveChannel(telegramChannel, "telegram_enabled"));

    grid.append(
      field("Interval", select),
      weekdayField,
      field("Run time", timeInput),
      field("Alert rule", emailSelect),
      field("Delivery channels", channelOptions),
      conditionEditor
    );
    settings.appendChild(grid);
    if (watch.visibility !== "private" && watch.share_id) {
      settings.appendChild(buildListingBlock(watch, onListChanged));
    }

    content.append(bodyRow, actions, settings);
    card.append(top, content);
    return card;
  }

  // "Google listing": Owner nominiert die eigene öffentliche Watch-Seite für
  // den Suchindex. Setzt nur ein Anfrage-Flag – gelistet wird erst nach
  // menschlichem Review (Admin), nie automatisch.
  function buildListingBlock(watch, onListChanged) {
    const block = document.createElement("div");
    block.className = "watch-card-listing";
    const title = document.createElement("strong");
    title.className = "watch-listing-title";
    title.textContent = "Google listing";
    const note = document.createElement("p");
    note.className = "watch-listing-note";
    block.append(title, note);

    async function requestListing(button, want) {
      button.disabled = true;
      try {
        await api("POST", "/api/share/" + encodeURIComponent(watch.share_id) + "/indexing-request", { want: want });
        window.App?.trackAppEvent?.("app_watch_listing_request", { want: want });
        popup(want
          ? "Thanks! Your page is nominated — we review every page before it appears on Google."
          : "Listing request withdrawn.");
        onListChanged();
      } catch (error) {
        button.disabled = false;
        popup("Request failed: " + error.message);
      }
    }

    if (watch.indexed) {
      note.textContent = "This page is listed: it appears in Google's index, our sitemap, and “Related questions” on other pages.";
    } else if (watch.index_requested) {
      note.textContent = "Listing requested — a human reviews every page before it appears on Google. You can withdraw the request anytime.";
      block.appendChild(makeButton("Withdraw request", "share-secondary-btn", function () {
        requestListing(this, false);
      }));
    } else {
      note.textContent = watch.index_eligible
        ? "Public pages stay unlisted until you nominate them. This page meets the quality bar — a human still reviews it before it goes live on Google."
        : "Public pages stay unlisted until you nominate them. This page is below the quality bar (needs several models, sources, and a substantial answer), but you can still request a review.";
      block.appendChild(makeButton("Request Google listing", "share-secondary-btn", function () {
        requestListing(this, true);
      }));
    }
    return block;
  }

  async function renderDashboard() {
    const { body } = dashEls();
    if (!body) return;
    const limitTarget = document.getElementById("watchDashLimit");
    if (limitTarget) limitTarget.hidden = true;
    body.innerHTML = '<p class="watch-dash-loading">Loading your watches…</p>';
    let watches = [];
    let brief = {};
    let telegram = { configured: false, connected: false };
    try {
      const [watchData, briefData, telegramData] = await Promise.all([
        api("GET", "/api/my/watches"),
        api("GET", "/api/my/watch-brief").catch(() => ({ brief: {} })),
        api("GET", "/api/my/telegram").catch(() => ({ telegram: {} }))
      ]);
      watches = watchData.watches || [];
      watchLimitState = normalizeWatchLimits(watchData.limits, watches);
      renderWatchLimit(limitTarget, watchLimitState);
      brief = briefData.brief || {};
      telegram = telegramData.telegram || telegram;
      telegramState = telegram;
    } catch (error) {
      if (limitTarget) limitTarget.hidden = true;
      body.innerHTML = "";
      const failed = document.createElement("p");
      failed.className = "watch-dash-loading";
      failed.textContent = "Could not load watches: " + error.message;
      body.appendChild(failed);
      return;
    }
    body.innerHTML = "";

    if (!watches.length) {
      const empty = document.createElement("div");
      empty.className = "watch-dash-empty";
      empty.innerHTML = `
        <span class="watch-step-label">Consensus Watch</span>
        <strong>Keep changing answers current.</strong>
        <p>Choose a question once. consens.io checks it on your schedule and alerts you by e-mail or Telegram when the consensus materially moves.</p>
        <div class="watch-empty-flow" aria-label="How a Watch works">
          <span><b>1</b><small>Ask</small></span>
          <i aria-hidden="true"></i>
          <span><b>2</b><small>Check</small></span>
          <i aria-hidden="true"></i>
          <span><b>3</b><small>Alert</small></span>
        </div>
        <div class="watch-empty-examples">
          <span>Try an example</span>
          <div></div>
        </div>`;
      const actions = document.createElement("div");
      actions.className = "watch-empty-actions";
      const create = makeButton("Create your first Watch", "share-primary-btn", () => {
        openWatchDialog("create");
      });
      actions.appendChild(create);
      const examples = [
        "Has this regulation changed?",
        "Is this product available in Germany?",
        "Has the company changed its pricing?"
      ];
      const exampleList = empty.querySelector(".watch-empty-examples > div");
      examples.forEach(question => {
        exampleList.appendChild(makeButton(question, "watch-example-chip", () => {
          openWatchDialog("create", { question: question });
        }));
      });
      empty.appendChild(actions);
      body.appendChild(empty);
      return;
    }

    renderDashboardStats(body, watches);
    renderNotificationsPanel(body, telegram, brief, true);

    const listTitle = document.createElement("h3");
    listTitle.className = "watch-dash-section-title";
    listTitle.textContent = "Watches";
    body.appendChild(listTitle);
    const list = document.createElement("ul");
    list.className = "watch-dash-list";
    const collapseCards = watches.length > 2;
    watches.forEach(watch => list.appendChild(
      renderWatchCard(watch, renderDashboard, telegram, collapseCards)
    ));
    const filterBar = document.createElement("div");
    filterBar.className = "watch-filter-bar";
    const filterDefinitions = [
      ["all", "All", watches.length],
      ["changed", "Changed", watches.filter(watch => driftState(watch).key === "changed").length],
      ["stable", "Stable", watches.filter(watch => driftState(watch).key === "stable").length],
      ["paused", "Paused", watches.filter(watch => watch.status !== "active").length]
    ];
    filterDefinitions.forEach(([key, label, count], index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "watch-filter" + (index === 0 ? " is-active" : "");
      button.dataset.filter = key;
      button.setAttribute("aria-pressed", String(index === 0));
      button.innerHTML = `${escapeHtml(label)} <span>${count}</span>`;
      button.addEventListener("click", () => {
        filterBar.querySelectorAll(".watch-filter").forEach(item => {
          const active = item === button;
          item.classList.toggle("is-active", active);
          item.setAttribute("aria-pressed", String(active));
        });
        list.querySelectorAll(".watch-card").forEach(card => {
          card.hidden = key === "changed" ? card.dataset.drift !== "changed"
            : key === "stable" ? card.dataset.drift !== "stable"
              : key === "paused" ? card.dataset.status === "active"
                : false;
        });
      });
      filterBar.appendChild(button);
    });
    body.appendChild(filterBar);
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

  function initViewSwitch() {
    const consensusButton = document.getElementById("viewSwitchConsensus");
    const watchesButton = document.getElementById("viewSwitchWatches");
    if (!consensusButton || !watchesButton) return;

    consensusButton.addEventListener("click", closeWatchDashboard);
    watchesButton.addEventListener("click", () => {
      acknowledgeViewSwitchHint();
      openWatchDashboard();
    });
    setViewSwitchState(onWatchPagePath());
    initViewSwitchHint();
  }

  function initDashboardCreateButton() {
    document.getElementById("watchDashCreate")?.addEventListener("click", () => {
      openWatchDialog("create");
    });
  }

  function resetAfterLogout() {
    const { page, body } = dashEls();
    if (body) body.innerHTML = "";
    watchLimitState = null;
    watchLimitRequest = null;
    const quota = document.getElementById("watchUsageDisplay");
    if (quota) quota.innerHTML = "Watches: <strong>...</strong>";
    if (page) page.hidden = true;
    setViewSwitchState(false);
    if (onWatchPagePath()) window.history.replaceState(null, "", APP_PATH);
  }

  window.openWatchDialog = openWatchDialog;
  window.openWatchDashboard = openWatchDashboard;
  window.App.watch = Object.assign(window.App.watch || {}, {
    showFeatureNudge: showWatchFeatureNudge,
    refreshQuota: () => loadWatchLimits(true),
    resetAfterLogout: resetAfterLogout
  });
  initWatchButton();
  initViewSwitch();
  initDashboardCreateButton();
  initWatchPageRoute();
})();
