/** Explicit add/correct actions for selected question and answer text. */
(function () {
  "use strict";

  const state = {
    selection: null,
    intent: "correct",
    busy: false,
    undoTimer: null,
    requestId: null,
    requestFeedback: null,
    returnFocus: null,
    directDraft: false
  };
  const user = () => window.auth?.currentUser?.uid ? window.auth.currentUser : null;
  const icons = {
    remember: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 3.75a6.25 6.25 0 0 0-3.87 11.16L7.5 19.75l4.5-2.5 4.5 2.5-.63-4.84A6.25 6.25 0 0 0 12 3.75Z"/><path d="M9.5 10.5h5M12 8v5"/></svg>',
    correct: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m14.75 5.25 4 4L9.5 18.5l-4.75.75.75-4.75 9.25-9.25Z"/><path d="m12.75 7.25 4 4"/></svg>',
    close: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m7 7 10 10M17 7 7 17"/></svg>',
    check: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m5.5 12.5 4 4 9-9"/></svg>'
  };

  function sourceForNode(node) {
    const el = node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement;
    if (el?.closest("#consensusAnswerBody, .thread-history-answer")) return "consensus";
    if (el?.closest(".response-box .collapsible-content")) return "model_answer";
    if (el?.closest(".thread-ask-text, .thread-history-question")) return "question";
    return null;
  }

  function sourceLabel(selection) {
    if (selection?.direct) return "Draft";
    if (selection?.kind === "consensus") return "Consensus";
    if (selection?.kind === "model_answer") return "Model answer";
    return "Your question";
  }

  function syncDraftAction() {
    const button = document.getElementById("rememberDraftButton");
    const input = document.getElementById("questionInput");
    if (!button || !input) return;
    button.hidden = !user() || !input.value.trim() || state.busy;
  }

  function openDraftDialog() {
    const input = document.getElementById("questionInput");
    const text = input?.value?.trim() || "";
    if (!user() || !text || text.length > 2000) return;
    state.selection = { text, kind: "question", direct: true };
    state.directDraft = true;
    openDialog("add", document.getElementById("rememberDraftButton"));
  }

  function ensureUi() {
    if (document.getElementById("memorySelectionMenu")) return;
    const menu = document.createElement("div");
    menu.id = "memorySelectionMenu";
    menu.className = "memory-selection-menu";
    menu.hidden = true;
    menu.setAttribute("role", "toolbar");
    menu.setAttribute("aria-label", "Memory actions for selected text");
    menu.innerHTML = `
      <button type="button" class="memory-selection-action" data-memory-intent="add">
        ${icons.remember}<span>Remember</span>
      </button>
      <span class="memory-selection-divider" aria-hidden="true"></span>
      <button type="button" class="memory-selection-action" data-memory-intent="correct">
        ${icons.correct}<span>Correct memory</span>
      </button>`;

    const backdrop = document.createElement("div");
    backdrop.id = "memoryEditBackdrop";
    backdrop.className = "memory-edit-backdrop";
    backdrop.hidden = true;
    backdrop.innerHTML = `
      <section class="memory-edit-dialog" role="dialog" aria-modal="true" aria-labelledby="memoryEditTitle" aria-describedby="memoryEditDescription">
        <header class="memory-edit-dialog-head">
          <span class="memory-edit-title-icon" id="memoryEditTitleIcon" aria-hidden="true"></span>
          <div>
            <h2 id="memoryEditTitle"></h2>
            <p id="memoryEditDescription"></p>
          </div>
          <button type="button" class="memory-edit-close" aria-label="Close">${icons.close}</button>
        </header>
        <div class="memory-edit-dialog-body">
          <div class="memory-edit-context">
            <span class="memory-edit-source" id="memoryEditSource"></span>
            <blockquote id="memoryEditSelection"></blockquote>
          </div>
          <div class="memory-edit-field-head">
            <label for="memoryEditCorrection" id="memoryEditLabel"></label>
            <span id="memoryEditCount" aria-hidden="true">0 / 500</span>
          </div>
          <textarea id="memoryEditCorrection" rows="4" maxlength="500" aria-describedby="memoryEditHelper memoryEditStatus"></textarea>
          <p class="memory-edit-helper" id="memoryEditHelper"></p>
          <p class="memory-edit-status" id="memoryEditStatus" role="status" aria-live="polite"></p>
        </div>
        <footer class="memory-edit-dialog-actions">
          <button type="button" class="memory-edit-cancel">Cancel</button>
          <button type="button" class="memory-edit-submit"><span class="memory-edit-submit-label"></span></button>
        </footer>
      </section>`;

    const toast = document.createElement("div");
    toast.id = "memoryEditToast";
    toast.className = "memory-edit-toast";
    toast.hidden = true;
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");

    document.body.append(menu, backdrop, toast);
    menu.querySelectorAll("[data-memory-intent]").forEach(button => {
      button.addEventListener("click", () => openDialog(button.dataset.memoryIntent, button));
    });
    backdrop.querySelector(".memory-edit-close").addEventListener("click", closeDialog);
    backdrop.querySelector(".memory-edit-cancel").addEventListener("click", closeDialog);
    backdrop.querySelector(".memory-edit-submit").addEventListener("click", submitEdit);
    backdrop.addEventListener("click", event => {
      if (event.target === backdrop) closeDialog();
    });
    const textarea = backdrop.querySelector("textarea");
    textarea.addEventListener("input", updateCounter);
    textarea.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") submitEdit();
    });
    backdrop.addEventListener("keydown", handleDialogKeys);
    document.getElementById("rememberDraftButton")?.addEventListener("click", openDraftDialog);
  }

  function makeRequestId() {
    return window.crypto?.randomUUID?.()
      || `edit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function hideMenu() {
    const menu = document.getElementById("memorySelectionMenu");
    if (menu) menu.hidden = true;
  }

  function showSelection(text, kind, rect) {
    if (!user() || !text || text.length > 2000 || !rect) return hideMenu();
    state.selection = { text, kind, direct: false };
    state.directDraft = false;
    const menu = document.getElementById("memorySelectionMenu");
    menu.hidden = false;
    menu.style.visibility = "hidden";
    requestAnimationFrame(() => {
      const menuWidth = menu.offsetWidth;
      const menuHeight = menu.offsetHeight;
      const center = rect.left + (rect.width / 2);
      const left = Math.max(10, Math.min(window.innerWidth - menuWidth - 10, center - (menuWidth / 2)));
      const below = rect.bottom + 10;
      const top = below + menuHeight <= window.innerHeight - 10
        ? below
        : Math.max(10, rect.top - menuHeight - 10);
      menu.style.left = `${left}px`;
      menu.style.top = `${top}px`;
      menu.style.visibility = "visible";
    });
  }

  function captureSelection(event) {
    if (state.busy || !document.getElementById("memoryEditBackdrop")?.hidden) return;
    if (event?.target?.closest?.(".memory-selection-menu, .memory-edit-toast")) return;
    const input = event?.target?.closest?.("#questionInput");
    if (input && input.selectionStart !== input.selectionEnd) {
      return showSelection(
        input.value.slice(input.selectionStart, input.selectionEnd).trim(),
        "question",
        input.getBoundingClientRect()
      );
    }
    const selection = window.getSelection();
    const text = selection?.toString().trim() || "";
    if (!text || selection.rangeCount !== 1) return hideMenu();
    const range = selection.getRangeAt(0);
    const kind = sourceForNode(range.commonAncestorContainer);
    if (!kind) return hideMenu();
    showSelection(text, kind, range.getBoundingClientRect());
  }

  function dialogCopy(intent) {
    if (intent === "add") {
      return {
        title: "Remember this",
        description: "Save a lasting fact or preference and keep related Memory consistent.",
        label: "What should consens.io remember?",
        helper: "Luna adds it or updates one clearly matching saved passage. Unrelated details stay unchanged.",
        placeholder: "For example: I live in Hanover.",
        submit: "Save to Memory",
        busy: "Saving…",
        progress: "Luna is checking for one related saved passage…"
      };
    }
    return {
      title: "Correct memory",
      description: "Update one outdated or incorrect passage in your Memory.",
      label: "What should consens.io remember instead?",
      helper: "Luna changes one matching passage. If none exists, your correction can be added as a new entry.",
      placeholder: "For example: I now work in Hanover.",
      submit: "Update Memory",
      busy: "Updating…",
      progress: "Luna is locating one matching Memory passage…"
    };
  }

  function openDialog(intent, trigger) {
    hideMenu();
    if (!state.selection || !user()) return;
    state.intent = intent === "add" ? "add" : "correct";
    state.returnFocus = trigger || null;
    const copy = dialogCopy(state.intent);
    const backdrop = document.getElementById("memoryEditBackdrop");
    const textarea = document.getElementById("memoryEditCorrection");
    document.getElementById("memoryEditTitle").textContent = copy.title;
    document.getElementById("memoryEditDescription").textContent = copy.description;
    document.getElementById("memoryEditTitleIcon").innerHTML = state.intent === "add" ? icons.remember : icons.correct;
    document.getElementById("memoryEditLabel").textContent = copy.label;
    document.getElementById("memoryEditHelper").textContent = copy.helper;
    document.getElementById("memoryEditSource").textContent = sourceLabel(state.selection);
    document.getElementById("memoryEditSelection").textContent = state.selection.text.slice(0, 360) + (state.selection.text.length > 360 ? "…" : "");
    document.querySelector(".memory-edit-submit-label").textContent = copy.submit;
    textarea.placeholder = copy.placeholder;
    textarea.value = state.intent === "add" ? state.selection.text.slice(0, 500) : "";
    textarea.removeAttribute("aria-invalid");
    document.getElementById("memoryEditStatus").textContent = "";
    state.requestId = makeRequestId();
    state.requestFeedback = textarea.value.trim();
    backdrop.hidden = false;
    document.documentElement.classList.add("memory-edit-open");
    updateCounter();
    requestAnimationFrame(() => {
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    });
  }

  function closeDialog() {
    if (state.busy) return;
    document.getElementById("memoryEditBackdrop").hidden = true;
    document.documentElement.classList.remove("memory-edit-open");
    state.returnFocus?.focus?.();
  }

  function handleDialogKeys(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeDialog();
      return;
    }
    if (event.key !== "Tab") return;
    const controls = [...event.currentTarget.querySelectorAll("button:not(:disabled), textarea:not(:disabled)")];
    if (!controls.length) return;
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function updateCounter() {
    const textarea = document.getElementById("memoryEditCorrection");
    const count = textarea?.value?.length || 0;
    const counter = document.getElementById("memoryEditCount");
    if (counter) {
      counter.textContent = `${count} / 500`;
      counter.dataset.near = count >= 400 ? "true" : "false";
    }
    textarea?.removeAttribute("aria-invalid");
    const status = document.getElementById("memoryEditStatus");
    if (status?.dataset.kind === "error") {
      status.textContent = "";
      delete status.dataset.kind;
    }
  }

  function setBusy(busy, message) {
    state.busy = busy;
    const backdrop = document.getElementById("memoryEditBackdrop");
    backdrop.querySelectorAll("button, textarea").forEach(control => { control.disabled = busy; });
    const submit = backdrop.querySelector(".memory-edit-submit");
    submit.classList.toggle("is-busy", busy);
    submit.querySelector("span").textContent = busy ? dialogCopy(state.intent).busy : dialogCopy(state.intent).submit;
    if (message !== undefined) {
      const status = document.getElementById("memoryEditStatus");
      status.textContent = message;
      status.dataset.kind = busy ? "progress" : "";
    }
    syncDraftAction();
  }

  function showFieldError(message) {
    const status = document.getElementById("memoryEditStatus");
    const textarea = document.getElementById("memoryEditCorrection");
    status.textContent = message;
    status.dataset.kind = "error";
    textarea.setAttribute("aria-invalid", "true");
    textarea.focus();
  }

  function errorMessage(data, fallback) {
    if (typeof data?.detail === "string") return data.detail;
    if (typeof data?.detail?.message === "string") return data.detail.message;
    return fallback;
  }

  async function post(path, body) {
    const authUser = user();
    if (!authUser) throw new Error("Please log in first.");
    const uid = authUser.uid;
    const token = await authUser.getIdToken();
    if (user()?.uid !== uid) throw new Error("Authentication changed.");
    const response = await fetch(path, {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty response */ }
    if (!response.ok) throw new Error(errorMessage(data, "Memory could not be updated."));
    return data;
  }

  async function submitEdit() {
    if (state.busy || !state.selection) return;
    const correction = document.getElementById("memoryEditCorrection").value.trim();
    if (!correction) return showFieldError("Enter what consens.io should remember.");
    if (state.requestFeedback !== correction) {
      state.requestId = makeRequestId();
      state.requestFeedback = correction;
    }
    setBusy(true, dialogCopy(state.intent).progress);
    try {
      const result = await post("/api/my/memory/edit", {
        client_request_id: state.requestId,
        source_kind: state.selection.kind,
        selected_text: state.selection.text,
        correction,
        intent: state.intent
      });
      if (result.status === "processing") throw new Error("This Memory action is already processing.");
      document.getElementById("memoryEditBackdrop").hidden = true;
      document.documentElement.classList.remove("memory-edit-open");
      await window.App?.userMemory?.load?.(true);
      if (state.directDraft && state.intent === "add") {
        const input = document.getElementById("questionInput");
        if (input?.value?.trim() === state.selection.text) {
          input.value = "";
          input.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }
      showUndo(result, state.intent);
      window.App?.trackAppEvent?.("app_memory_ai_edit", { intent: state.intent, status: "updated" });
    } catch (error) {
      showFieldError(error.message || "Memory could not be updated.");
    } finally {
      setBusy(false);
    }
  }

  function showUndo(result, intent) {
    const toast = document.getElementById("memoryEditToast");
    toast.replaceChildren();
    const check = document.createElement("span");
    check.className = "memory-edit-toast-icon";
    check.innerHTML = icons.check;
    const content = document.createElement("span");
    content.className = "memory-edit-toast-content";
    content.textContent = intent === "add"
      ? (result.operation === "append" ? "Added to Memory" : result.operation === "replace" ? "Memory updated" : "Saved to Memory")
      : "Memory updated";
    const undo = document.createElement("button");
    undo.type = "button";
    undo.className = "memory-edit-undo";
    undo.textContent = "Undo";
    undo.addEventListener("click", () => undoEdit(result.revision_id, undo));
    toast.append(check, content, undo);
    toast.hidden = false;
    clearTimeout(state.undoTimer);
    const expires = Date.parse(result.undo_expires_at || "");
    const delay = Number.isFinite(expires) ? Math.max(0, expires - Date.now()) : 60000;
    state.undoTimer = setTimeout(() => { toast.hidden = true; }, Math.min(delay, 60000));
  }

  async function undoEdit(revisionId, button) {
    button.disabled = true;
    const toast = document.getElementById("memoryEditToast");
    try {
      await post("/api/my/memory/undo", { revision_id: revisionId });
      await window.App?.userMemory?.load?.(true);
      toast.querySelector(".memory-edit-toast-content").textContent = "Previous Memory restored";
      button.remove();
      window.App?.trackAppEvent?.("app_memory_ai_edit", { intent: state.intent, status: "undone" });
      setTimeout(() => { toast.hidden = true; }, 2500);
    } catch (error) {
      toast.querySelector(".memory-edit-toast-content").textContent = error.message || "Memory could not be restored.";
      button.remove();
      setTimeout(() => { toast.hidden = true; }, 4000);
    }
  }

  function bind() {
    ensureUi();
    document.addEventListener("mouseup", event => setTimeout(() => captureSelection(event), 0));
    document.addEventListener("keyup", event => {
      if (event.key === "Shift" || event.shiftKey) captureSelection(event);
    });
    document.addEventListener("scroll", hideMenu, true);
    window.addEventListener("resize", hideMenu);
    document.getElementById("questionInput")?.addEventListener("input", syncDraftAction);
    window.addEventListener("consensio:auth-state", () => {
      hideMenu();
      syncDraftAction();
    });
    syncDraftAction();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
