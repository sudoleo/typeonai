// =====================================================================
// consensus-actions.js
// Consensus action controls: share button, copy consensus, copy citation.
// Extracted from templates/index.html without changing the consensus run.
// Dependencies: window.App.{showPopup,trackAppEvent}, window.auth,
// window.openShareDialog, window.consensusCitationMeta.
// =====================================================================

(function () {
  function showPopup(message) {
    if (window.App && typeof window.App.showPopup === "function") {
      window.App.showPopup(message);
    }
  }

  function trackAppEvent(eventName, eventData = {}) {
    if (window.App && typeof window.App.trackAppEvent === "function") {
      window.App.trackAppEvent(eventName, eventData);
    } else if (typeof window.trackUmamiEvent === "function") {
      window.trackUmamiEvent(eventName, eventData);
    }
  }

  function getIncludedModelNamesForCitation() {
    const map = [
      { boxId: "openaiResponse", selectId: "openaiModelSelect", label: "OpenAI" },
      { boxId: "mistralResponse", selectId: "mistralModelSelect", label: "Mistral" },
      { boxId: "claudeResponse", selectId: "claudeModelSelect", label: "Anthropic Claude" },
      { boxId: "geminiResponse", selectId: "geminiModelSelect", label: "Google Gemini" },
      { boxId: "deepseekResponse", selectId: "deepseekModelSelect", label: "DeepSeek" },
      { boxId: "grokResponse", selectId: "grokModelSelect", label: "Grok" }
    ];

    const names = [];
    map.forEach(({ boxId, selectId, label }) => {
      const box = document.getElementById(boxId);
      if (!box) return;
      if (box.classList.contains("excluded")) return;

      const txtEl = box.querySelector(".collapsible-content");
      const txt = txtEl ? txtEl.innerText.trim() : "";
      if (txt) {
        const select = document.getElementById(selectId);
        const modelName = select
          ? (select.options[select.selectedIndex]?.text || select.value || "")
          : "";
        names.push(modelName ? `${label}: ${modelName}` : label);
      }
    });

    return names;
  }

  function buildConsensusCitation() {
    const meta = window.consensusCitationMeta || {};

    let includedModels = Array.isArray(meta.includedModels) ? meta.includedModels : [];
    if (!includedModels.length) {
      includedModels = getIncludedModelNamesForCitation();
    }

    if (!includedModels.length) {
      return "";
    }

    const date = meta.dateISO ? new Date(meta.dateISO) : new Date();
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    const dateStr = `${yyyy}-${mm}-${dd}`;

    const question = (meta.question || document.getElementById("questionInput")?.value || "").trim();
    const modelsPart = includedModels.join(", ");
    const consensusSelect = document.getElementById("consensusModelDropdown");
    const consensusModel = meta.consensusModel
      || consensusSelect?.options[consensusSelect.selectedIndex]?.text
      || consensusSelect?.value
      || "";
    const cleanUrl = meta.url || window.location.href;

    const parts = [];
    parts.push(`consens.io. (${dateStr}).`);

    if (question) {
      parts.push(`Consensus answer to "${question}".`);
    } else {
      parts.push("Consensus answer.");
    }

    parts.push(`Models consulted: ${modelsPart}.`);

    if (consensusModel) {
      parts.push(`Consensus model: ${consensusModel}.`);
    }

    const consensusBox = document.getElementById("consensusResponse");
    const mainPara = consensusBox?.querySelector(".consensus-main p");
    const hrefSet = new Set();

    if (mainPara) {
      mainPara.querySelectorAll("a[href]").forEach(a => {
        const href = a.getAttribute("href");
        if (!href) return;
        if (href.startsWith("#")) return;
        hrefSet.add(href);
      });
    }

    const links = Array.from(hrefSet);
    if (links.length) {
      parts.push(`Sources: ${links.join(", ")}`);
    }

    parts.push(`Retrieved from ${cleanUrl}`);
    return parts.join(" ");
  }

  let consensusMenuEl = null;

  function closeConsensusActionsMenu() {
    if (!consensusMenuEl) return;
    consensusMenuEl.classList.remove("open");
    consensusMenuEl.style.display = "none";
    document.querySelectorAll(".consensus-actions-wrapper.open").forEach(el => {
      el.classList.remove("open");
    });
    document.querySelectorAll(".consensus-actions-toggle[aria-expanded='true']").forEach(btn => {
      btn.setAttribute("aria-expanded", "false");
    });
  }

  function copyText(text, successMessage, eventType) {
    if (!navigator.clipboard) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      closeConsensusActionsMenu();
      showPopup(successMessage);
      trackAppEvent("app_consensus_copied", { type: eventType });
      return;
    }

    navigator.clipboard.writeText(text).then(() => {
      closeConsensusActionsMenu();
      showPopup(successMessage);
      trackAppEvent("app_consensus_copied", { type: eventType });
    }).catch(() => {
      alert("Copy to clipboard failed.");
    });
  }

  function ensureConsensusActionsMenu() {
    if (consensusMenuEl) return consensusMenuEl;

    const menu = document.createElement("div");
    menu.id = "consensusActionsMenuGlobal";
    menu.className = "consensus-actions-menu";
    menu.setAttribute("role", "menu");

    const copyAnswerBtn = document.createElement("button");
    copyAnswerBtn.type = "button";
    copyAnswerBtn.className = "consensus-copy-icon-btn consensus-copy-btn";
    copyAnswerBtn.setAttribute("role", "menuitem");
    copyAnswerBtn.innerHTML = `
      <svg class="menu-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="9" width="11" height="11" rx="2.5"></rect>
        <path d="M5 15H4.5A2.5 2.5 0 0 1 2 12.5v-8A2.5 2.5 0 0 1 4.5 2h8A2.5 2.5 0 0 1 15 4.5V5"></path>
      </svg>
      <span class="copy-label">Copy consensus</span>
    `;
    copyAnswerBtn.title = "Copy consensus answer";

    copyAnswerBtn.addEventListener("click", () => {
      const consensusBox = document.getElementById("consensusResponse");
      const mainPara = consensusBox?.querySelector(".consensus-main p");
      // Agreement-Badges (z. B. "6/6") sind UI-Elemente und gehören nicht in
      // den kopierten Text. Klon ohne Badges kurz unsichtbar einhängen:
      // innerText braucht ein gerendertes Element, um Zeilenumbrüche zu
      // erhalten (und ignoriert display:none-Kinder nur dann zuverlässig).
      let mainText = "";
      if (mainPara) {
        const clone = mainPara.cloneNode(true);
        clone.querySelectorAll(".claim-badge, .copy-btn, .response-code-copy").forEach(el => el.remove());
        clone.style.position = "absolute";
        clone.style.left = "-99999px";
        clone.style.top = "0";
        document.body.appendChild(clone);
        mainText = clone.innerText.trim();
        clone.remove();
      }

      if (!mainText) {
        closeConsensusActionsMenu();
        alert("No consensus available yet.");
        return;
      }

      copyText(mainText, "Consensus copied to clipboard.", "answer");
    });

    const copyCitationBtn = document.createElement("button");
    copyCitationBtn.type = "button";
    copyCitationBtn.className = "consensus-copy-icon-btn consensus-citation-btn";
    copyCitationBtn.setAttribute("role", "menuitem");
    copyCitationBtn.innerHTML = `
      <svg class="menu-item-icon" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">
        <path d="M6 17h3l2-4V7H5v6h3zm8 0h3l2-4V7h-6v6h3z"></path>
      </svg>
      <span class="copy-label">Copy citation</span>
    `;
    copyCitationBtn.title = "Copy consensus citation";

    copyCitationBtn.addEventListener("click", () => {
      const citation = buildConsensusCitation();

      if (!citation) {
        closeConsensusActionsMenu();
        alert("No citation available yet.");
        return;
      }

      copyText(citation, "Citation copied to clipboard.", "citation");
    });

    menu.appendChild(copyAnswerBtn);
    menu.appendChild(copyCitationBtn);
    document.body.appendChild(menu);

    consensusMenuEl = menu;
    return menu;
  }

  function initConsensusCopyButtons() {
    const consensusBox = document.getElementById("consensusResponse");
    if (!consensusBox) return;

    const mainSection = consensusBox.querySelector(".consensus-main");
    if (!mainSection) return;

    const heading = mainSection.querySelector("h2");
    if (!heading) return;

    if (heading.querySelector(".consensus-copy-inline")) return;

    const inlineBar = document.createElement("span");
    inlineBar.className = "consensus-copy-inline";

    const shareTopBtn = document.createElement("button");
    shareTopBtn.type = "button";
    shareTopBtn.id = "consensusShareButton";
    shareTopBtn.className = "consensus-share-pill";
    shareTopBtn.title = "Share this consensus as a public page";
    shareTopBtn.setAttribute("aria-label", "Share this consensus publicly");
    shareTopBtn.innerHTML = `
      <svg class="share-pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 15V4"></path>
        <path d="M8.5 7.5 12 4l3.5 3.5"></path>
        <path d="M5 12v7a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 19v-7"></path>
      </svg>
    `;
    shareTopBtn.addEventListener("click", () => {
      if (!window.auth?.currentUser) {
        showPopup("Please log in to share a consensus publicly.");
        return;
      }
      window.openShareDialog("confirm");
    });

    const actionsWrapper = document.createElement("div");
    actionsWrapper.className = "consensus-actions-wrapper";

    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "consensus-actions-toggle";
    toggleBtn.setAttribute("aria-label", "Copy options");
    toggleBtn.setAttribute("aria-haspopup", "menu");
    toggleBtn.setAttribute("aria-expanded", "false");
    toggleBtn.innerHTML = `
      <svg class="quote-toggle-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M6 17h3l2-4V7H5v6h3zm8 0h3l2-4V7h-6v6h3z"></path>
      </svg>
    `;

    actionsWrapper.appendChild(toggleBtn);
    inlineBar.appendChild(shareTopBtn);
    inlineBar.appendChild(actionsWrapper);
    heading.appendChild(inlineBar);

    const menu = ensureConsensusActionsMenu();

    function closeMenu() {
      closeConsensusActionsMenu();
    }

    function openMenu() {
      closeMenu();

      menu.style.display = "block";
      menu.style.visibility = "hidden";
      menu.classList.add("open");
      actionsWrapper.classList.add("open");

      const rect = toggleBtn.getBoundingClientRect();
      const menuWidth = menu.offsetWidth || 180;

      const top = rect.bottom + 6;
      let left = rect.right - menuWidth;
      left = Math.max(8, left);

      menu.style.top = `${top}px`;
      menu.style.left = `${left}px`;
      menu.style.visibility = "visible";
      toggleBtn.setAttribute("aria-expanded", "true");
    }

    toggleBtn.addEventListener("click", (evt) => {
      evt.stopPropagation();
      const isOpen = menu.classList.contains("open");
      if (isOpen) {
        closeMenu();
      } else {
        openMenu();
      }
    });

    document.addEventListener("click", (evt) => {
      if (!menu.classList.contains("open")) return;
      if (menu.contains(evt.target) || actionsWrapper.contains(evt.target)) return;
      closeMenu();
    });

    window.addEventListener("scroll", () => {
      if (menu.classList.contains("open")) closeMenu();
    }, { passive: true });

    window.addEventListener("resize", () => {
      if (menu.classList.contains("open")) closeMenu();
    });
  }

  initConsensusCopyButtons();
})();
