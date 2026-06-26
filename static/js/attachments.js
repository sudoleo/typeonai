// =====================================================================
// attachments.js
// Datei-Anhaenge (Pro-Feature): Attach-Menue, Upload-Validierung, Chips,
// Viewer-Vorschau, Bookmark-Vorschau-Chips. In eigene IIFE gekapselt.
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
  const ATTACH_ALLOWED_MIMES = ["application/pdf", "image/png", "image/jpeg", "image/webp"];
  window.pendingAttachments = [];

  (function initAttachments() {
    const trigger = document.getElementById("attachTrigger");
    const menu = document.getElementById("attachMenu");
    const uploadOption = document.getElementById("attachUploadOption");
    const fileInput = document.getElementById("attachFileInput");
    const bar = document.getElementById("attachmentBar");
    const inputContainer = document.querySelector(".chat-input-container");
    if (!trigger || !menu || !uploadOption || !fileInput || !bar) return;

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
        icon.textContent = att.mime === "application/pdf" ? "PDF" : "IMG";
        const text = document.createElement("p");
        text.textContent = "This file was attached to the saved chat. To keep storage light, only the file name is stored – not the file itself.";
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

    uploadOption.addEventListener("click", function () {
      if (!window.isUserPro) {
        setMenuOpen(false);
        trackAppEvent("app_attachment_locked_click");
        const modal = document.getElementById("proFeatureModal");
        if (modal) {
          modal.style.display = "block";
        } else {
          alert("File uploads are a Pro feature.");
        }
        return;
      }
      setMenuOpen(false);
      fileInput.click();
    });

    function formatFileSize(bytes) {
      if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
      return Math.max(1, Math.round(bytes / 1024)) + " KB";
    }

    function renderAttachmentChips() {
      bar.innerHTML = "";
      const items = window.pendingAttachments;
      bar.hidden = items.length === 0;

      items.forEach(function (att, index) {
        const chip = document.createElement("div");
        chip.className = "attachment-chip";
        if (att.previewOnly) chip.classList.add("is-preview-only");
        chip.setAttribute("role", "button");
        chip.tabIndex = 0;
        chip.title = "Click to preview " + att.name;

        if (!att.previewOnly && att.mime.indexOf("image/") === 0) {
          const img = document.createElement("img");
          img.className = "attachment-chip-thumb";
          img.alt = "";
          img.src = "data:" + att.mime + ";base64," + att.data;
          chip.appendChild(img);
        } else {
          const icon = document.createElement("span");
          icon.className = "attachment-chip-icon";
          if (att.mime.indexOf("image/") === 0) icon.classList.add("is-image");
          icon.textContent = att.mime.indexOf("image/") === 0 ? "IMG" : "PDF";
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
        sizeEl.textContent = att.previewOnly
          ? (att.size ? formatFileSize(att.size) + " · saved chat" : "saved chat")
          : formatFileSize(att.size);
        meta.appendChild(nameEl);
        meta.appendChild(sizeEl);
        chip.appendChild(meta);

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

        chip.addEventListener("click", function () {
          openAttachmentViewer(att);
        });
        chip.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openAttachmentViewer(att);
          }
        });

        bar.appendChild(chip);
      });
    }

    window.renderAttachmentChips = renderAttachmentChips;

    window.clearPendingAttachments = function () {
      if (!window.pendingAttachments.length) return;
      window.pendingAttachments = [];
      renderAttachmentChips();
    };

    function inferMime(file) {
      if (ATTACH_ALLOWED_MIMES.indexOf(file.type) !== -1) return file.type;
      const name = (file.name || "").toLowerCase();
      if (name.endsWith(".pdf")) return "application/pdf";
      if (name.endsWith(".png")) return "image/png";
      if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
      if (name.endsWith(".webp")) return "image/webp";
      return null;
    }

    fileInput.addEventListener("change", function () {
      const files = Array.from(fileInput.files || []);
      fileInput.value = "";
      if (!files.length) return;

      for (const file of files) {
        if (window.pendingAttachments.length >= ATTACH_MAX_FILES) {
          alert("You can attach up to " + ATTACH_MAX_FILES + " files per question.");
          break;
        }
        const mime = inferMime(file);
        if (!mime) {
          alert("'" + file.name + "' is not supported. Allowed: PDF, PNG, JPG, WebP.");
          continue;
        }
        if (file.size > ATTACH_MAX_BYTES) {
          alert("'" + file.name + "' is too large. The limit is 5 MB per file.");
          continue;
        }

        const reader = new FileReader();
        reader.onload = function () {
          const result = String(reader.result || "");
          const base64Data = result.split(",", 2)[1] || "";
          if (!base64Data) return;
          if (window.pendingAttachments.length >= ATTACH_MAX_FILES) return;
          window.pendingAttachments.push({
            name: file.name,
            mime: mime,
            size: file.size,
            data: base64Data
          });
          renderAttachmentChips();
          trackAppEvent("app_attachment_added", { mime: mime });
        };
        reader.readAsDataURL(file);
      }
    });
  })();

  window.getAttachmentsPayload = function () {
    return (window.pendingAttachments || [])
      .filter(function (att) { return !att.previewOnly && att.data; })
      .map(function (att) {
        return { name: att.name, mime: att.mime, size: att.size, data: att.data };
      });
  };

  // Zeigt Anhänge aus einem gespeicherten Bookmark als reine Vorschau-Chips
  // (nur Metadaten, ohne Dateidaten – wird beim nächsten Senden nicht mitgeschickt).
  window.showBookmarkAttachments = function (attachmentsMeta) {
    window.pendingAttachments = (Array.isArray(attachmentsMeta) ? attachmentsMeta : [])
      .map(function (meta) {
        return {
          name: String(meta.name || "attachment"),
          mime: String(meta.mime || ""),
          size: Number(meta.size) || 0,
          data: null,
          previewOnly: true
        };
      });
    if (typeof window.renderAttachmentChips === "function") {
      window.renderAttachmentChips();
    }
  };
})();
