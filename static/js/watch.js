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

  function emailModeOptions(selected) {
    return `
      <option value="changes_only"${selected !== "every_run" ? " selected" : ""}>Material changes only</option>
      <option value="every_run"${selected === "every_run" ? " selected" : ""}>Every new consensus (with content)</option>
    `;
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
      <p>consens.io will rerun the <strong>original question</strong> at your chosen interval and e-mail you only when the result changes materially.</p>
      <div class="watch-public-note"><strong>A public page is required.</strong> Activating Watch creates a non-indexed, read-only share page if needed. Anyone with its link can view it.</div>
      <label class="watch-interval-label" for="watchInterval">Check interval ${window.isUserPro ? "" : '<span class="pro-badge is-subtle">Pro: daily</span>'}</label>
      <select id="watchInterval" class="watch-interval-select">${intervalOptions("weekly")}</select>
      <label class="watch-interval-label" for="watchEmailMode">E-mail notifications</label>
      <select id="watchEmailMode" class="watch-interval-select watch-email-select">${emailModeOptions("changes_only")}</select>
      <p class="watch-data-note">“Every new consensus” includes the newly generated consensus text in each successful-run e-mail.</p>
      <p class="watch-data-note">Only the question is rerun. Attachments and follow-up context are never resent.</p>
      <div class="share-modal-actions">
        <button type="button" id="watchConfirmBtn" class="share-primary-btn">Start watching</button>
        <button type="button" id="watchCancelBtn" class="share-secondary-btn">Cancel</button>
        <button type="button" id="watchListLink" class="share-link-btn">Watched</button>
      </div>`;
    document.getElementById("watchCancelBtn").addEventListener("click", closeDialog);
    document.getElementById("watchListLink").addEventListener("click", renderWatchList);
    const confirm = document.getElementById("watchConfirmBtn");
    if (!window.lastShareResultId) {
      confirm.disabled = true;
      confirm.textContent = "Run a consensus first";
    }
    confirm.addEventListener("click", async function () {
      if (!window.lastShareResultId) return;
      this.disabled = true;
      this.textContent = "Starting…";
      try {
        const data = await api("POST", "/api/watch", {
          result_id: window.lastShareResultId,
          interval: document.getElementById("watchInterval").value,
          email_mode: document.getElementById("watchEmailMode").value
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
    body.querySelector("p strong").textContent = watch.interval;
    document.getElementById("watchMailSummary").textContent = watch.email_mode === "every_run"
      ? "You will receive every new consensus including its content."
      : "You will be notified only after a material change.";
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
      status.textContent = watch.status === "paused_error" ? "Paused after errors" : watch.status;
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
      const emailSelect = document.createElement("select");
      emailSelect.className = "watch-interval-select watch-email-select";
      emailSelect.setAttribute("aria-label", "E-mail notifications");
      emailSelect.innerHTML = emailModeOptions(watch.email_mode);
      emailSelect.addEventListener("change", async () => {
        emailSelect.disabled = true;
        try {
          await api("PATCH", "/api/watch/" + encodeURIComponent(watch.id), { email_mode: emailSelect.value });
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
        if (!confirm("Delete this watch? Its existing public history will remain on the share page.")) return;
        remove.disabled = true;
        try {
          await api("DELETE", "/api/watch/" + encodeURIComponent(watch.id));
          item.remove();
          if (!list.children.length) renderWatchList();
        } catch (error) { remove.disabled = false; popup("Delete failed: " + error.message); }
      });
      controls.append(select, emailSelect, pause, remove);
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
