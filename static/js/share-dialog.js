// =====================================================================
// share-dialog.js
// Oeffentliche Share-Links: Bestaetigen, Erfolg, Liste/Revoke. In eigene
// IIFE gekapselt. Extrahiert aus templates/index.html (initApp-Closure).
// Export: window.openShareDialog (firebase.js User-Menue ruft es auf).
// Abhaengigkeiten: window.auth (firebase), window.lastShareResultId,
// window.App.showPopup, window.App.trackAppEvent.
// =====================================================================

(function () {
  function shareDialogEls() {
    return {
      modal: document.getElementById("shareModal"),
      body: document.getElementById("shareModalBody"),
      title: document.getElementById("shareModalTitle")
    };
  }

  function closeShareDialog() {
    const { modal } = shareDialogEls();
    if (modal) modal.style.display = "none";
  }

  function openShareDialog(view) {
    const { modal } = shareDialogEls();
    if (!modal) return;
    if (!window.auth?.currentUser) {
      window.App.showPopup("Please log in to share.");
      return;
    }
    modal.style.display = "block";
    if (view === "list") {
      renderShareListView();
    } else {
      renderShareConfirmView();
    }
  }

  async function shareApiRequest(method, path, body) {
    const user = window.auth?.currentUser;
    if (!user) throw new Error("Not logged in");
    const idToken = await user.getIdToken();
    const response = await fetch(path, {
      method,
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + idToken
      },
      body: body ? JSON.stringify(body) : undefined
    });
    let data = {};
    try { data = await response.json(); } catch (e) { /* leerer Body */ }
    if (!response.ok) {
      throw new Error(data.error || data.detail || ("HTTP " + response.status));
    }
    return data;
  }

  function copyShareUrl(url) {
    if (!navigator.clipboard) {
      const ta = document.createElement("textarea");
      ta.value = url;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      window.App.showPopup("Link copied to clipboard.");
      return;
    }
    navigator.clipboard.writeText(url).then(() => {
      window.App.showPopup("Link copied to clipboard.");
    }).catch(() => {
      alert("Copy to clipboard failed.");
    });
  }

  function renderShareConfirmView() {
    const { body, title } = shareDialogEls();
    if (!body) return;
    title.textContent = "Share this consensus publicly";
    body.innerHTML = `
      <p>This creates a <strong>public, read-only page</strong> with the question,
      the consensus answer, the differences and the sources of this run.</p>
      <ul class="share-modal-notes">
        <li>Anyone with the link can view it - no login required.</li>
        <li>The page contains no account data and is not linked to you publicly.</li>
        <li>It is a snapshot: later runs will not change it.</li>
        <li>You can revoke the link at any time under "My shared links".</li>
      </ul>
      <div class="share-modal-actions">
        <button type="button" id="shareConfirmBtn" class="share-primary-btn">Create public link</button>
        <button type="button" id="shareCancelBtn" class="share-secondary-btn">Cancel</button>
        <button type="button" id="shareListLink" class="share-link-btn">My shared links</button>
      </div>
    `;
    document.getElementById("shareCancelBtn").addEventListener("click", closeShareDialog);
    document.getElementById("shareListLink").addEventListener("click", renderShareListView);
    if (!window.lastShareResultId) {
      const confirmBtn = document.getElementById("shareConfirmBtn");
      confirmBtn.disabled = true;
      confirmBtn.textContent = "Run a consensus first";
    }
    document.getElementById("shareConfirmBtn").addEventListener("click", async function () {
      const resultId = window.lastShareResultId;
      if (!resultId) {
        window.App.showPopup("Please run a consensus first.");
        return;
      }
      this.disabled = true;
      this.textContent = "Creating link…";
      try {
        const data = await shareApiRequest("POST", "/api/share", { result_id: resultId });
        window.App.trackAppEvent("app_share_created", { reused: data.created === false });
        renderShareSuccessView(data.url);
      } catch (err) {
        this.disabled = false;
        this.textContent = "Create public link";
        window.App.showPopup("Sharing failed: " + err.message);
      }
    });
  }

  function renderShareSuccessView(url) {
    const { body, title } = shareDialogEls();
    if (!body) return;
    title.textContent = "Public link created";
    body.innerHTML = `
      <p>Your consensus is now available at:</p>
      <div class="share-url-row">
        <input type="text" id="shareUrlInput" readonly>
        <button type="button" id="shareCopyUrlBtn" class="share-primary-btn">Copy</button>
      </div>
      <div class="share-modal-actions">
        <a id="shareOpenLink" target="_blank" rel="noopener" class="share-secondary-btn">Open page</a>
        <button type="button" id="shareListLink" class="share-link-btn">My shared links</button>
      </div>
    `;
    const input = document.getElementById("shareUrlInput");
    input.value = url;
    input.addEventListener("focus", () => input.select());
    document.getElementById("shareOpenLink").href = url;
    document.getElementById("shareCopyUrlBtn").addEventListener("click", () => copyShareUrl(url));
    document.getElementById("shareListLink").addEventListener("click", renderShareListView);
  }

  async function renderShareListView() {
    const { body, title } = shareDialogEls();
    if (!body) return;
    title.textContent = "My shared links";
    body.innerHTML = `<p class="share-list-loading">Loading…</p>`;
    let data;
    try {
      data = await shareApiRequest("GET", "/api/my/shares");
    } catch (err) {
      body.innerHTML = "";
      const p = document.createElement("p");
      p.textContent = "Could not load your shared links: " + err.message;
      body.appendChild(p);
      return;
    }

    body.innerHTML = "";
    const shares = (data.shares || []).filter(s => s.status === "active");
    if (!shares.length) {
      const p = document.createElement("p");
      p.textContent = "You have no active shared links.";
      body.appendChild(p);
      return;
    }

    const list = document.createElement("ul");
    list.className = "share-list";
    shares.forEach(share => {
      const item = document.createElement("li");
      const url = (data.site_url || "") + share.path;

      const question = document.createElement("a");
      question.className = "share-list-question";
      question.textContent = share.question || "(untitled)";
      question.href = url;
      question.target = "_blank";
      question.rel = "noopener";

      const actions = document.createElement("div");
      actions.className = "share-list-actions";

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "share-secondary-btn";
      copyBtn.textContent = "Copy link";
      copyBtn.addEventListener("click", () => copyShareUrl(url));

      const revokeBtn = document.createElement("button");
      revokeBtn.type = "button";
      revokeBtn.className = "share-danger-btn";
      revokeBtn.textContent = "Revoke";
      revokeBtn.addEventListener("click", async () => {
        if (!confirm("Revoke this public link? Visitors will no longer be able to open it.")) return;
        revokeBtn.disabled = true;
        try {
          await shareApiRequest("DELETE", "/api/share/" + encodeURIComponent(share.share_id));
          window.App.trackAppEvent("app_share_revoked", {});
          window.App.showPopup("Share link revoked.");
          item.remove();
          if (!list.children.length) renderShareListView();
        } catch (err) {
          revokeBtn.disabled = false;
          window.App.showPopup("Revoking failed: " + err.message);
        }
      });

      actions.appendChild(copyBtn);
      actions.appendChild(revokeBtn);
      item.appendChild(question);
      item.appendChild(actions);
      list.appendChild(item);
    });
    body.appendChild(list);
  }

  // Global verfügbar machen, damit z.B. das User-Icon-Menü (firebase.js)
  // die Übersicht der geteilten Links direkt öffnen kann.
  window.openShareDialog = openShareDialog;

  (function initShareModal() {
    const modal = document.getElementById("shareModal");
    if (!modal) return;
    const closeBtn = document.getElementById("closeShareModal");
    if (closeBtn) closeBtn.addEventListener("click", closeShareDialog);
    window.addEventListener("click", (event) => {
      if (event.target === modal) closeShareDialog();
    });
  })();
})();
