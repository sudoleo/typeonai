// =====================================================================
// attachments.js
// Datei-Anhaenge (Pro-Feature): Attach-Menue, Upload-/Paste-/Drop-Validierung,
// Chips, Viewer-Vorschau, Bookmark-Vorschau-Chips. In eigene IIFE gekapselt.
// Extrahiert aus templates/index.html (initApp-Closure).
// Exporte: window.pendingAttachments, window.renderAttachmentChips,
// window.clearPendingAttachments, window.getAttachmentsPayload,
// window.showBookmarkAttachments.
// Call-time-Abhaengigkeiten: window.isUserPro, window.trackUmamiEvent,
// DOM (#attachTrigger, #attachMenu, #attachFileInput, #attachmentBar, ...).
// =====================================================================

(function () {
  // Telemetrie-Wrapper (entspricht trackAppEvent aus initApp).
  function trackAppEvent(eventName, eventData = {}) {
    if (typeof window.trackUmamiEvent === "function") {
      window.trackUmamiEvent(eventName, eventData);
    }
  }

  // --- ATTACHMENTS (Pro Feature) ---
  const ATTACH_MAX_FILES = 2;
  const ATTACH_MAX_BYTES = 5 * 1024 * 1024;
  const DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  const ATTACH_ALLOWED_MIMES = ["application/pdf", DOCX_MIME, "text/plain", "text/markdown", "text/csv", "image/png", "image/jpeg", "image/webp"];
  const ATTACH_TYPES_LABEL = "PDF, Word (.docx), TXT, MD, CSV, PNG, JPG, WebP";
  const DEEPSEEK_ATTACHMENT_MESSAGE = "DeepSeek is paused for this question because its API cannot read attachments. Remove the files to use DeepSeek again.";
  window.pendingAttachments = [];

  (function initAttachments() {
    const trigger = document.getElementById("attachTrigger");
    const menu = document.getElementById("attachMenu");
    const uploadOption = document.getElementById("attachUploadOption");
    const fileInput = document.getElementById("attachFileInput");
    const bar = document.getElementById("attachmentBar");
    const inputContainer = document.querySelector(".chat-input-container");
    const questionInput = document.getElementById("questionInput");
    if (!trigger || !menu || !uploadOption || !fileInput || !bar) return;

    let pendingFileReads = 0;
    let dragDepth = 0;
    let deepSeekSelectionBeforeAttachment = null;
    // Waehrend des Abgebens beim Senden ist der Composer zwar leer, die Dateien
    // sind aber gerade RAUSGEGANGEN: der Lauf-Block bleibt dann bestehen.
    let detachingForSend = false;

    function hasSendableAttachments() {
      return (window.pendingAttachments || []).some(function (att) {
        return !att.previewOnly && !!att.data;
      });
    }

    function syncDeepSeekAttachmentCompatibility() {
      const checkbox = document.getElementById("selectDeepSeek");
      if (!checkbox) return;

      const incompatible = hasSendableAttachments();
      const label = document.querySelector("label[for='selectDeepSeek']");
      const responseBox = document.getElementById("deepseekResponse");
      const excludeButton = responseBox?.querySelector(".exclude-btn");

      if (incompatible) {
        if (deepSeekSelectionBeforeAttachment === null) {
          deepSeekSelectionBeforeAttachment = checkbox.checked;
        }
        if (checkbox.checked) {
          window.App?.setModelSelectionState?.("deepseekResponse", false, {
            persist: false,
            syncCheckbox: true,
            animate: true
          });
        }
        checkbox.disabled = true;
        checkbox.setAttribute("aria-describedby", "attachmentProviderNotice");
        if (label) {
          label.classList.add("is-attachment-incompatible");
          label.title = DEEPSEEK_ATTACHMENT_MESSAGE;
        }
        if (excludeButton) {
          excludeButton.disabled = true;
          excludeButton.title = DEEPSEEK_ATTACHMENT_MESSAGE;
          excludeButton.setAttribute("aria-label", DEEPSEEK_ATTACHMENT_MESSAGE);
        }
        return;
      }

      checkbox.disabled = false;
      checkbox.removeAttribute("aria-describedby");
      if (label) label.classList.remove("is-attachment-incompatible");
      if (excludeButton) excludeButton.disabled = false;

      // Der Composer ist leer, WEIL der Nutzer die Dateien entfernt hat: der
      // naechste Lauf geht ohne Anhaenge raus, also faellt auch der Lauf-Block.
      // Beim Senden (detachingForSend) ist der Composer ebenfalls leer, die
      // Dateien sind aber gerade mit der Frage rausgegangen — dort bleibt der
      // Block bis zum naechsten Senden stehen.
      if (!detachingForSend) {
        window.App?.setRunModelBlock?.("deepseekResponse", false);
      }

      if (deepSeekSelectionBeforeAttachment !== null) {
        const shouldRestore = deepSeekSelectionBeforeAttachment;
        deepSeekSelectionBeforeAttachment = null;
        window.App?.setModelSelectionState?.("deepseekResponse", shouldRestore, {
          persist: false,
          syncCheckbox: true,
          animate: true
        });
      }
    }

    function setMenuOpen(open) {
      menu.hidden = !open;
      trigger.setAttribute("aria-expanded", String(open));
      trigger.classList.toggle("is-open", open);
      if (inputContainer) inputContainer.classList.toggle("attach-menu-open", open);
    }

    trigger.addEventListener("click", function (event) {
      event.stopPropagation();
      setMenuOpen(menu.hidden);
    });

    document.addEventListener("click", function (event) {
      if (!menu.hidden && !menu.contains(event.target) && event.target !== trigger) {
        setMenuOpen(false);
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      const viewer = document.getElementById("attachmentViewerModal");
      if (viewer && !viewer.hidden) {
        closeAttachmentViewer();
        return;
      }
      if (!menu.hidden) setMenuOpen(false);
    });

    // --- Viewer (einfache Vorschau beim Klick auf einen Chip) ---
    const viewerOverlay = document.getElementById("attachmentViewerModal");
    const viewerTitle = document.getElementById("attachmentViewerTitle");
    const viewerBody = document.getElementById("attachmentViewerBody");
    const viewerClose = document.getElementById("attachmentViewerClose");
    let viewerObjectUrl = null;

    function closeAttachmentViewer() {
      if (!viewerOverlay) return;
      viewerOverlay.hidden = true;
      if (viewerBody) viewerBody.innerHTML = "";
      if (viewerObjectUrl) {
        URL.revokeObjectURL(viewerObjectUrl);
        viewerObjectUrl = null;
      }
    }

    function base64ToBlob(base64Data, mime) {
      const binary = atob(base64Data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return new Blob([bytes], { type: mime });
    }

    function openAttachmentViewer(att) {
      if (!viewerOverlay || !viewerBody) return;
      closeAttachmentViewer();
      viewerTitle.textContent = att.name;
      viewerBody.innerHTML = "";

      if (att.previewOnly || !att.data) {
        const notice = document.createElement("div");
        notice.className = "attachment-viewer-notice";
        const icon = document.createElement("span");
        icon.className = "attachment-chip-icon";
        icon.textContent = chipIconLabel(att.mime);
        const text = document.createElement("p");
        text.textContent = "This file was attached to the saved chat. To keep storage light, only the file name is stored – not the file itself.";
        notice.appendChild(icon);
        notice.appendChild(text);
        viewerBody.appendChild(notice);
      } else if (att.mime === DOCX_MIME) {
        // Browser können DOCX nicht inline rendern – nur Hinweis zeigen.
        const notice = document.createElement("div");
        notice.className = "attachment-viewer-notice";
        const icon = document.createElement("span");
        icon.className = "attachment-chip-icon";
        icon.textContent = "DOC";
        const text = document.createElement("p");
        text.textContent = "Word documents cannot be previewed here. The extracted text is sent to the models with your question.";
        notice.appendChild(icon);
        notice.appendChild(text);
        viewerBody.appendChild(notice);
      } else if (att.mime.indexOf("image/") === 0) {
        const img = document.createElement("img");
        img.className = "attachment-viewer-image";
        img.alt = att.name;
        img.src = "data:" + att.mime + ";base64," + att.data;
        viewerBody.appendChild(img);
      } else {
        try {
          viewerObjectUrl = URL.createObjectURL(base64ToBlob(att.data, att.mime));
          const frame = document.createElement("iframe");
          frame.className = "attachment-viewer-frame";
          frame.title = att.name;
          frame.src = viewerObjectUrl;
          viewerBody.appendChild(frame);
        } catch (e) {
          const fallback = document.createElement("p");
          fallback.className = "attachment-viewer-notice";
          fallback.textContent = "Preview is not available in this browser.";
          viewerBody.appendChild(fallback);
        }
      }

      viewerOverlay.hidden = false;
      trackAppEvent("app_attachment_viewed", { mime: att.mime, preview_only: !!att.previewOnly });
    }

    if (viewerClose) viewerClose.addEventListener("click", closeAttachmentViewer);
    if (viewerOverlay) {
      viewerOverlay.addEventListener("click", function (event) {
        if (event.target === viewerOverlay) closeAttachmentViewer();
      });
    }

    function showAttachmentProGate(source) {
      setMenuOpen(false);
      trackAppEvent("app_attachment_locked_click", { source: source });
      const shown = window.App?.showProFeatureModal?.("File uploads");
      if (!shown) {
        window.App?.showPopup?.("File uploads are off here. Attached files make every one of the six calls a lot longer.");
      }
    }

    uploadOption.addEventListener("click", function () {
      if (!window.isUserPro) {
        showAttachmentProGate("picker");
        return;
      }
      setMenuOpen(false);
      fileInput.click();
    });

    function formatFileSize(bytes) {
      if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
      return Math.max(1, Math.round(bytes / 1024)) + " KB";
    }

    // Baut den Chip einer Datei. `readonly` macht ihn zum reinen
    // Anzeigeelement: kein Viewer, kein Entfernen — so haengt er an einer
    // bereits gesendeten Nachricht, deren Datei es nicht mehr gibt.
    function buildAttachmentChip(att, options) {
      const readonly = Boolean(options && options.readonly);
      const chip = document.createElement("div");
      chip.className = "attachment-chip";
      if (att.previewOnly) chip.classList.add("is-preview-only");

      if (!att.previewOnly && att.mime.indexOf("image/") === 0 && att.data) {
        const img = document.createElement("img");
        img.className = "attachment-chip-thumb";
        img.alt = "";
        img.src = "data:" + att.mime + ";base64," + att.data;
        chip.appendChild(img);
      } else {
        // Die Plakette traegt seit 2026-08-17 keine Typfarbe mehr (monochrome
        // App): der Dateityp steht als Text darin, nicht in einem Farbton.
        const icon = document.createElement("span");
        icon.className = "attachment-chip-icon";
        icon.textContent = chipIconLabel(att.mime);
        chip.appendChild(icon);
      }

      const meta = document.createElement("span");
      meta.className = "attachment-chip-meta";
      const nameEl = document.createElement("span");
      nameEl.className = "attachment-chip-name";
      nameEl.textContent = att.name;
      nameEl.title = att.name;
      const sizeEl = document.createElement("span");
      sizeEl.className = "attachment-chip-size";
      if (readonly) {
        sizeEl.textContent = att.size ? formatFileSize(att.size) : "";
      } else {
        sizeEl.textContent = att.previewOnly
          ? (att.size ? formatFileSize(att.size) + " · saved chat" : "saved chat")
          : formatFileSize(att.size);
      }
      meta.appendChild(nameEl);
      meta.appendChild(sizeEl);
      chip.appendChild(meta);

      if (readonly) return chip;

      chip.setAttribute("role", "button");
      chip.tabIndex = 0;
      chip.title = "Click to preview " + att.name;
      chip.addEventListener("click", function () {
        openAttachmentViewer(att);
      });
      chip.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openAttachmentViewer(att);
        }
      });
      return chip;
    }

    // Anhaenge einer bereits gesendeten Frage. Sie stehen im Thread an ihrer
    // Nachricht und sind reine Metadaten — die Dateien selbst sind mit dem
    // Lauf rausgegangen und werden nicht aufbewahrt.
    function renderMessageAttachments(container, attachmentsMeta) {
      if (!container) return 0;
      container.innerHTML = "";
      const items = (Array.isArray(attachmentsMeta) ? attachmentsMeta : [])
        .filter(function (item) { return item && item.name; });
      items.forEach(function (item) {
        container.appendChild(buildAttachmentChip({
          name: String(item.name),
          mime: String(item.mime || ""),
          size: Number(item.size) || 0,
          data: null
        }, { readonly: true }));
      });
      container.hidden = items.length === 0;
      return items.length;
    }

    // Beim Senden gibt der Composer seine Anhaenge ab: die Chips wandern an
    // die Nachricht, das Feld startet leer in die naechste Frage. Ein Anhang,
    // der nach dem Senden ueber dem leeren Feld haengen bleibt, behauptet
    // sonst, er gehoere zur naechsten Frage — mitgeschickt wurde er aber mit
    // der letzten.
    // Was von den Anhaengen an der Nachricht haengen bleibt: reine Metadaten.
    // Die Blase der gerade abgeschickten Frage braucht sie schon, BEVOR der
    // Composer die Dateien abgibt — der Lauf kann noch scheitern, dann muessen
    // sie unveraendert am Feld stehen.
    function messageMeta() {
      return (window.pendingAttachments || [])
        .filter(function (att) { return !att.previewOnly && att.data; })
        .map(function (att) {
          return { name: att.name, mime: att.mime, size: att.size || 0 };
        });
    }

    function detachForMessage() {
      const meta = messageMeta();
      if (window.pendingAttachments.length) {
        window.pendingAttachments = [];
        detachingForSend = true;
        try {
          renderAttachmentChips();
        } finally {
          detachingForSend = false;
        }
      }
      return meta;
    }

    function renderAttachmentChips() {
      bar.innerHTML = "";
      const items = window.pendingAttachments;
      bar.hidden = items.length === 0;

      items.forEach(function (att, index) {
        const chip = buildAttachmentChip(att);

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "attachment-chip-remove";
        removeBtn.title = "Remove attachment";
        removeBtn.setAttribute("aria-label", "Remove " + att.name);
        removeBtn.innerHTML = "&#10005;";
        removeBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          window.pendingAttachments.splice(index, 1);
          renderAttachmentChips();
        });
        chip.appendChild(removeBtn);

        bar.appendChild(chip);
      });

      if (hasSendableAttachments()) {
        const notice = document.createElement("p");
        notice.id = "attachmentProviderNotice";
        notice.className = "attachment-provider-notice";
        notice.setAttribute("role", "status");
        notice.setAttribute("aria-live", "polite");
        notice.textContent = DEEPSEEK_ATTACHMENT_MESSAGE;
        bar.appendChild(notice);
      }

      syncDeepSeekAttachmentCompatibility();
    }

    window.renderAttachmentChips = renderAttachmentChips;
    window.App = window.App || {};
    window.App.attachments = {
      detachForMessage: detachForMessage,
      messageMeta: messageMeta,
      renderMessageAttachments: renderMessageAttachments
    };

    window.clearPendingAttachments = function () {
      if (!window.pendingAttachments.length) return;
      window.pendingAttachments = [];
      renderAttachmentChips();
    };

    function inferMime(file) {
      if (ATTACH_ALLOWED_MIMES.indexOf(file.type) !== -1) return file.type;
      const name = (file.name || "").toLowerCase();
      if (name.endsWith(".pdf")) return "application/pdf";
      if (name.endsWith(".docx")) return DOCX_MIME;
      if (name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".markdown") || name.endsWith(".csv")) return "text/plain";
      if (name.endsWith(".png")) return "image/png";
      if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
      if (name.endsWith(".webp")) return "image/webp";
      return null;
    }

    function chipIconLabel(mime) {
      if (mime.indexOf("image/") === 0) return "IMG";
      if (mime === DOCX_MIME) return "DOC";
      if (mime.indexOf("text/") === 0) return "TXT";
      return "PDF";
    }

    function imageExtension(mime) {
      if (mime === "image/jpeg") return "jpg";
      if (mime === "image/webp") return "webp";
      return "png";
    }

    function attachmentName(file, mime, source, index) {
      const originalName = String(file.name || "").trim();
      if (originalName) return originalName;
      if (source === "paste") {
        return "pasted-image-" + Date.now() + (index ? "-" + (index + 1) : "") + "." + imageExtension(mime);
      }
      return "image-" + Date.now() + (index ? "-" + (index + 1) : "") + "." + imageExtension(mime);
    }

    function addFiles(files, options) {
      const source = options && options.source ? options.source : "picker";
      const imagesOnly = !!(options && options.imagesOnly);
      if (!files.length) return;

      if (!window.isUserPro) {
        showAttachmentProGate(source);
        return;
      }

      let unsupportedShown = false;
      let limitShown = false;

      files.forEach(function (file, index) {
        const mime = inferMime(file);
        if (!mime || (imagesOnly && mime.indexOf("image/") !== 0)) {
          if (!unsupportedShown) {
            alert(imagesOnly
              ? "Only PNG, JPG, and WebP images can be pasted here."
              : "'" + (file.name || "This file") + "' is not supported. Allowed: " + ATTACH_TYPES_LABEL + ".");
            unsupportedShown = true;
          }
          return;
        }
        if (file.size > ATTACH_MAX_BYTES) {
          alert("'" + attachmentName(file, mime, source, index) + "' is too large for the configured upload limit.");
          return;
        }
        if (window.pendingAttachments.length + pendingFileReads >= ATTACH_MAX_FILES) {
          if (!limitShown) {
            alert("You can attach up to " + ATTACH_MAX_FILES + " files per question.");
            limitShown = true;
          }
          return;
        }

        pendingFileReads += 1;
        const reader = new FileReader();
        reader.onload = function () {
          pendingFileReads = Math.max(0, pendingFileReads - 1);
          const result = String(reader.result || "");
          const base64Data = result.split(",", 2)[1] || "";
          if (!base64Data) return;
          if (window.pendingAttachments.length >= ATTACH_MAX_FILES) return;
          window.pendingAttachments.push({
            name: attachmentName(file, mime, source, index),
            mime: mime,
            size: file.size,
            data: base64Data
          });
          renderAttachmentChips();
          trackAppEvent("app_attachment_added", { mime: mime, source: source });
        };
        reader.onerror = function () {
          pendingFileReads = Math.max(0, pendingFileReads - 1);
          alert("The file could not be read. Please try again.");
        };
        reader.readAsDataURL(file);
      });
    }

    function transferFiles(dataTransfer) {
      if (!dataTransfer) return [];
      const directFiles = Array.from(dataTransfer.files || []);
      if (directFiles.length) return directFiles;
      return Array.from(dataTransfer.items || [])
        .filter(function (item) { return item.kind === "file"; })
        .map(function (item) { return item.getAsFile(); })
        .filter(Boolean);
    }

    function isImageLike(file) {
      if (String(file.type || "").toLowerCase().indexOf("image/") === 0) return true;
      const name = String(file.name || "").toLowerCase();
      return /\.(png|jpe?g|webp)$/.test(name);
    }

    function isFileDrag(event) {
      return Array.from((event.dataTransfer && event.dataTransfer.types) || []).indexOf("Files") !== -1;
    }

    function clearDragState() {
      dragDepth = 0;
      if (inputContainer) inputContainer.classList.remove("is-image-dragover");
    }

    fileInput.addEventListener("change", function () {
      const files = Array.from(fileInput.files || []);
      fileInput.value = "";
      addFiles(files, { source: "picker", imagesOnly: false });
    });

    if (questionInput) {
      questionInput.addEventListener("paste", function (event) {
        const files = transferFiles(event.clipboardData).filter(isImageLike);
        if (!files.length) return;
        event.preventDefault();
        addFiles(files, { source: "paste", imagesOnly: true });
      });
    }

    if (inputContainer) {
      inputContainer.addEventListener("dragenter", function (event) {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        dragDepth += 1;
        inputContainer.classList.add("is-image-dragover");
      });

      inputContainer.addEventListener("dragover", function (event) {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        inputContainer.classList.add("is-image-dragover");
      });

      inputContainer.addEventListener("dragleave", function (event) {
        if (!isFileDrag(event)) return;
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0) inputContainer.classList.remove("is-image-dragover");
      });

      inputContainer.addEventListener("drop", function (event) {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        const files = transferFiles(event.dataTransfer);
        clearDragState();
        // Drag-and-drop supports the same whitelist as the file picker.
        // Only clipboard paste stays image-only so normal text paste keeps
        // behaving like text input.
        addFiles(files, { source: "drop", imagesOnly: false });
      });

      window.addEventListener("dragend", clearDragState);
      window.addEventListener("drop", clearDragState);
    }

    window.addEventListener("pageshow", function () {
      window.setTimeout(syncDeepSeekAttachmentCompatibility, 0);
    });
  })();

  window.getAttachmentsPayload = function () {
    return (window.pendingAttachments || [])
      .filter(function (att) { return !att.previewOnly && att.data; })
      .map(function (att) {
        return { name: att.name, mime: att.mime, size: att.size, data: att.data };
      });
  };

  // Anhänge eines gespeicherten Bookmarks. Sie gehören zu der Frage, mit der
  // sie damals rausgegangen sind, und stehen deshalb an ihr im Thread — nicht
  // im Eingabefeld, das der naechsten Frage gehoert. Reine Metadaten: die
  // Dateien selbst sind nicht gespeichert.
  window.showBookmarkAttachments = function (attachmentsMeta) {
    window.clearPendingAttachments?.();
    window.App?.setThreadQuestionAttachments?.(attachmentsMeta);
  };
})();
