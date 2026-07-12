// Consensus Watch UI. Classic script/module contract: exports window.openWatchDialog.
(function () {
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
    const escaped = String(value || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
    return `<div id="watchConditionWrap" hidden>
      <label class="watch-interval-label" for="watchCondition">Condition</label>
      <textarea id="watchCondition" class="watch-condition-input" maxlength="500" rows="3" placeholder="Example: An official launch date for Germany is announced.">${escaped}</textarea>
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
    const { modal } = els();
    if (!modal) return;
    if (!window.auth?.currentUser) {
      popup("Please log in to use Consensus Watch.");
      return;
    }
    modal.style.display = "block";
    if (view === "list") renderWatchList();
    else renderConfirm();
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
        <button type="button" id="watchListLink" class="share-link-btn">Watched</button>
      </div>`;
    document.getElementById("watchCancelBtn").addEventListener("click", closeDialog);
    document.getElementById("watchListLink").addEventListener("click", renderWatchList);
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
        <button type="button" id="watchListLink" class="share-link-btn">Watched</button>
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
    document.getElementById("watchListLink").addEventListener("click", renderWatchList);
  }

  function makeButton(label, className, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.addEventListener("click", handler);
    return button;
  }

  async function renderWatchList() {
    const { title, body } = els();
    title.textContent = "Watched";
    body.innerHTML = '<p class="share-list-loading">Loading…</p>';
    let watches;
    try {
      watches = (await api("GET", "/api/my/watches")).watches || [];
    } catch (error) {
      body.textContent = "Could not load watches: " + error.message;
      return;
    }
    body.innerHTML = "";
    if (!watches.length) {
      const empty = document.createElement("p");
      empty.textContent = "You are not watching any consensus yet.";
      body.appendChild(empty);
      return;
    }
    const list = document.createElement("ul");
    list.className = "share-list watch-list";
    watches.forEach(watch => {
      const item = document.createElement("li");
      item.className = "watch-list-item";
      const main = document.createElement("div");
      main.className = "watch-list-main";
      const link = document.createElement("a");
      link.className = "share-list-question";
      link.href = watch.share_path || "#";
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = watch.question || "(untitled)";
      const status = document.createElement("span");
      status.className = "watch-status watch-status-" + watch.status;
      const statusLabel = watch.status === "paused_error" ? "Paused after errors" : watch.status;
      status.textContent = statusLabel + " · " + (watch.visibility === "private" ? "Private" : "Public");
      main.append(link, status);

      const controls = document.createElement("div");
      controls.className = "watch-list-controls";
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
      timeInput.setAttribute("aria-label", "Run time");
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
      emailSelect.setAttribute("aria-label", "E-mail notifications");
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
      const active = watch.status === "active";
      const pause = makeButton(active ? "Pause" : "Resume", "share-secondary-btn", async () => {
        pause.disabled = true;
        try {
          await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), { status: active ? "paused" : "active" });
          renderWatchList();
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
          item.remove();
          if (!list.children.length) renderWatchList();
        } catch (error) { remove.disabled = false; popup("Delete failed: " + error.message); }
      });
      controls.append(select, timeInput, emailSelect, conditionEditor, pause, remove);
      item.append(main, controls);
      list.appendChild(item);
    });
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
    button.addEventListener("click", () => openWatchDialog("confirm"));
    const actions = bar.querySelector(".consensus-actions-wrapper");
    bar.insertBefore(button, actions || null);
  }

  window.openWatchDialog = openWatchDialog;
  initWatchButton();
})();
