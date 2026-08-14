// =====================================================================
// app-core.js
// Geteilte Basis (Uebergangsbus window.App) fuer die ausgelagerten
// Feature-Module und die verbleibende initApp-Closure.
// Haelt zentrale Config (modelPrefs, deepThinkModelLabels) und
// cross-cutting Helfer (getModelOptionLabel, getSelectedModelCount,
// trackAppEvent). MUSS vor den Feature-Modulen geladen werden.
//
// Hinweis: window.App ist bewusst ein TEMPORAERER Bus, um die Cluster
// schrittweise aus index.html zu loesen. Der echte State-Refactor folgt
// spaeter (DOM-als-State aufloesen).
// =====================================================================

(function () {
  window.App = window.App || {};

  // Telemetrie-Wrapper (Guard um window.trackUmamiEvent).
  function trackAppEvent(eventName, eventData = {}) {
    if (typeof window.trackUmamiEvent === "function") {
      window.trackUmamiEvent(eventName, eventData);
    }
  }

  function getSelectedModelCount() {
    return [
      "selectOpenAI",
      "selectMistral",
      "selectClaude",
      "selectGemini",
      "selectDeepSeek",
      "selectGrok"
    ].filter(id => document.getElementById(id)?.checked).length;
  }

  const DEFAULT_APP_TITLE = "Compare AI Answers | consens.io";

  function setAppTitle(question = "") {
    const normalized = String(question || "").replace(/\s+/g, " ").trim();
    if (!normalized) {
      document.title = DEFAULT_APP_TITLE;
      return;
    }

    const maxQuestionLength = 64;
    const shortened = normalized.length > maxQuestionLength
      ? `${normalized.slice(0, maxQuestionLength - 1).trimEnd()}…`
      : normalized;
    document.title = `${shortened} | consens.io`;
  }

  // Anhaenge der sichtbaren Frage. Sie gehoeren zu der Nachricht, mit der sie
  // rausgegangen sind — eine neue Frage erbt sie nicht. Gerendert werden sie
  // von attachments.js, das die Chip-Optik besitzt.
  let threadAttachments = [];

  function getThreadAttachments() {
    return threadAttachments.slice();
  }

  function setThreadQuestionAttachments(attachmentsMeta) {
    threadAttachments = (Array.isArray(attachmentsMeta) ? attachmentsMeta : [])
      .filter(item => item && item.name)
      .map(item => ({
        name: String(item.name),
        mime: String(item.mime || ""),
        size: Number(item.size) || 0
      }));
    const row = document.getElementById("threadAskAttachments");
    if (!row) return;
    if (typeof window.App.attachments?.renderMessageAttachments === "function") {
      window.App.attachments.renderMessageAttachments(row, threadAttachments);
      return;
    }
    row.innerHTML = "";
    row.hidden = true;
  }

  // Kopf des Threads (#threadAsk): zeigt die gestellte Frage über dem Lauf.
  // Leerer Text versteckt den Block wieder (New comparison, Clear). Lange
  // Fragen clampen per CSS auf drei Zeilen; is-long schaltet den Aufklapp-
  // Link frei, is-open hebt den Clamp auf.
  function setThreadQuestion(question = "") {
    const wrap = document.getElementById("threadAsk");
    const text = document.getElementById("threadAskText");
    if (!wrap || !text) return;

    const normalized = String(question || "").replace(/\s+/g, " ").trim();
    text.textContent = normalized;
    wrap.hidden = !normalized;
    wrap.classList.remove("is-open", "is-long");
    const more = document.getElementById("threadAskMore");
    if (more) more.textContent = "Show full question";
    // Eine neue Frage beginnt ohne Anhaenge; wer welche mitschickt, meldet sie
    // direkt nach dem Senden ueber setThreadQuestionAttachments an.
    setThreadQuestionAttachments([]);
    if (!normalized) return;

    requestAnimationFrame(() => syncThreadAskClamp(wrap, text));
    observeThreadAskWidth(wrap, text);
  }

  // Ob eine Frage laenger als drei Zeilen ist, haengt an der Breite des
  // Blocks - und die steht im ersten Frame noch nicht fest: Der Ausstieg aus
  // dem Hero animiert den Container, und die Sidebar aendert ihn spaeter noch
  // einmal. Wurde nur einmal gemessen, blieb "Show full question" bei einer
  // langen Frage aus und die vierte Zeile verschwand lautlos - beim
  // gefuehrten Lauf ausgerechnet das Ende der Frage. Deshalb misst ein
  // ResizeObserver nach jeder Groessenaenderung nach.
  let threadAskResizeObserver = null;

  function syncThreadAskClamp(wrap, text) {
    // Aufgeklappt gibt es nichts zu messen: dort ist scrollHeight gleich
    // clientHeight, und die Marke wuerde sich selbst zuruecknehmen.
    if (wrap.hidden || wrap.classList.contains("is-open")) return;
    const clamped = text.clientHeight;
    // scrollHeight allein reicht nicht: An einem geklammerten Block meldet er
    // je nach Zeitpunkt die geklammerte statt der vollen Hoehe - mal 4 Zeilen,
    // mal 3. Deshalb wird der Clamp fuer die Messung kurz aufgehoben. Das
    // passiert innerhalb eines Frames, es wird also nichts davon gezeichnet.
    const previous = text.style.webkitLineClamp;
    text.style.webkitLineClamp = "unset";
    const full = text.scrollHeight;
    text.style.webkitLineClamp = previous;
    wrap.classList.toggle("is-long", full > clamped + 2);
  }

  function observeThreadAskWidth(wrap, text) {
    if (threadAskResizeObserver || typeof ResizeObserver !== "function") return;
    threadAskResizeObserver = new ResizeObserver(() => syncThreadAskClamp(wrap, text));
    threadAskResizeObserver.observe(text);
  }

  // Dieselbe Geste fuer die aktive Frage (#threadAskMore) und fuer jede
  // archivierte im Verlauf: der Link gehoert immer zu der Frage, unter der er
  // steht, deshalb wird der Umschalter aus dem geklickten Knopf abgeleitet.
  document.addEventListener("click", (event) => {
    const more = event.target.closest(".thread-ask-more");
    if (!more) return;
    const wrap = more.closest(".thread-ask, .thread-history-question");
    if (!wrap) return;
    const open = wrap.classList.toggle("is-open");
    more.textContent = open ? "Collapse question" : "Show full question";
  });

  // Definition der Modelle und IDs (zentral, von mehreren Clustern genutzt).
  const modelPrefs = [
    { key: "OpenAI", provider: "openai", label: "OpenAI", checkId: "selectOpenAI", selectId: "openaiModelSelect", responseId: "openaiResponse", textId: "openaiModelText" },
    { key: "Mistral", provider: "mistral", label: "Mistral", checkId: "selectMistral", selectId: "mistralModelSelect", responseId: "mistralResponse", textId: "mistralModelText" },
    { key: "Anthropic", provider: "anthropic", label: "Claude", checkId: "selectClaude", selectId: "claudeModelSelect", responseId: "claudeResponse", textId: "claudeModelText" },
    { key: "Gemini", provider: "gemini", label: "Gemini", checkId: "selectGemini", selectId: "geminiModelSelect", responseId: "geminiResponse", textId: "geminiModelText" },
    { key: "DeepSeek", provider: "deepseek", label: "DeepSeek", checkId: "selectDeepSeek", selectId: "deepseekModelSelect", responseId: "deepseekResponse", textId: "deepseekModelText" },
    { key: "Grok", provider: "grok", label: "Grok", checkId: "selectGrok", selectId: "grokModelSelect", responseId: "grokResponse", textId: "grokModelText" }
  ];

  const deepThinkModelLabels = {
    OpenAI: "GPT-5.5",
    Mistral: "mistral-medium-3-5",
    Gemini: "gemini-3.1-pro-preview",
    Anthropic: "claude-opus-4-8",
    DeepSeek: "DeepSeek V4 Pro",
    Grok: "grok-4.3"
  };

  function getModelOptionLabel(option) {
    const explicitLabel = option?.dataset?.modelLabel;
    if (explicitLabel) return explicitLabel;
    return (option?.textContent || "").replace(/(?:\s*(?:Â·|·)\s*Pro)+$/i, "").trim();
  }

  // Einziges Renderziel des Konsenstextes. Frueher war das ein einzelnes <p>
  // unter .consensus-main, adressiert per ".consensus-main p" an einem guten
  // Dutzend Stellen. Das Inline-Marker-Rendering braucht einen stabilen
  // Blockcontainer, deshalb laeuft jeder Zugriff jetzt ueber diesen Helfer.
  // Der scope-Parameter erlaubt es, gezielt in einer bestimmten Konsens-Box zu
  // suchen (z. B. der von getConsensus gehaltenen Referenz).
  function consensusBodyEl(scope) {
    const root = scope || document;
    return (
      root.querySelector?.("#consensusAnswerBody")
      || root.querySelector?.(".consensus-main .consensus-answer-body")
      || null
    );
  }

  // Kurzlebiges Hinweis-Popup (cross-cutting UI-Helfer, von vielen Clustern genutzt).
  function showPopup(message) {
    const popup = document.createElement('div');
    popup.className = 'explanation-popup';
    popup.innerText = message;
    document.body.appendChild(popup);

    setTimeout(() => {
      popup.style.opacity = '1';
    }, 100);

    setTimeout(() => {
      popup.style.opacity = '0';
      setTimeout(() => {
        popup.remove();
      }, 300);
    }, 3000);
  }

  // Desktop-Schwelle des Hero-CSS (components-input.css): ab hier sind die
  // Response-Boxen ohne Agent Mode schon vor der ersten Frage sichtbar.
  const heroDesktopQuery = window.matchMedia("(min-width: 1100px)");

  // Haelt inert/aria-hidden der .response-section synchron zur CSS-Sichtbarkeit
  // im Hero: verborgen nur, wenn der Hero zentriert ist (Agent Mode aktiv oder
  // kein Desktop). Wird auch von agent-mode.js (updateAgentModeUI) gerufen.
  function syncHeroResponseAccess() {
    const responses = document.querySelector(".response-section");
    if (!responses) return;
    const directComparisonActive = document.body.classList.contains("direct-comparison-active");
    const hiddenInHero =
      document.body.classList.contains("is-hero") &&
      !directComparisonActive &&
      (document.body.classList.contains("agent-mode-enabled") || !heroDesktopQuery.matches);
    responses.inert = hiddenInHero;
    if (hiddenInHero) {
      responses.setAttribute("aria-hidden", "true");
    } else {
      responses.removeAttribute("aria-hidden");
    }
  }

  if (typeof heroDesktopQuery.addEventListener === "function") {
    heroDesktopQuery.addEventListener("change", syncHeroResponseAccess);
  }
  syncHeroResponseAccess();

  function exitHeroMode() {
    document.body.classList.remove("is-hero");
    syncHeroResponseAccess();
  }

  window.exitHeroMode = exitHeroMode;
  window.syncHeroResponseAccess = syncHeroResponseAccess;

  // Einziger Renderer fuer die Usage-Zeilen. API-Antworten ohne vollstaendige
  // Usage-Felder duerfen die zuletzt bekannte Anzeige nicht mit Fallback-Nullen
  // ueberschreiben. Das <strong>-Element ist zugleich Teil des Layout-Vertrags:
  // Label links, tabellarischer Wert rechts.
  function renderUsageDisplay({
    remaining,
    deepRemaining,
    totalLimit = window.currentMaxLimit,
    deepLimit = window.currentDeepLimit
  } = {}) {
    function renderLine(elementId, label, value, limit) {
      if (value === undefined || value === null) return;
      if (limit === undefined || limit === null) return;

      const element = document.getElementById(elementId);
      if (!element) return;

      const strong = document.createElement("strong");
      strong.textContent = value === "Unlimited" ? "Unlimited" : `${value} / ${limit}`;
      element.replaceChildren(document.createTextNode(`${label}: `), strong);
    }

    renderLine("freeUsageDisplay", "Runs", remaining, totalLimit);
    renderLine("deepUsageDisplay", "Deep Think", deepRemaining, deepLimit);
  }

  // Ein logischer UI-Lauf teilt genau einen serverseitigen Idempotency-Key
  // zwischen /prepare, allen parallelen /ask_* und /consensus. Kosten oder
  // Modellanzahl kommen bewusst nicht aus dem Client.
  const usageRun = {
    current: null,
    start(deepThink, useOwnKeys) {
      let key = null;
      if (!useOwnKeys) {
        key = globalThis.crypto?.randomUUID?.();
        if (!key) {
          key = `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`;
        }
      }
      this.current = {
        key,
        deepThink: deepThink === true,
        useOwnKeys: useOwnKeys === true,
        status: useOwnKeys ? "own_keys" : "new"
      };
      return this.current;
    },
    ensure(deepThink, useOwnKeys) {
      if (
        !this.current
        || this.current.deepThink !== (deepThink === true)
        || this.current.useOwnKeys !== (useOwnKeys === true)
      ) {
        return this.start(deepThink, useOwnKeys);
      }
      return this.current;
    },
    mark(status) {
      if (this.current && status) this.current.status = status;
    },
    clear() {
      this.current = null;
    }
  };

  Object.assign(window.App, {
    modelPrefs,
    deepThinkModelLabels,
    getModelOptionLabel,
    getSelectedModelCount,
    setAppTitle,
    setThreadQuestion,
    setThreadQuestionAttachments,
    getThreadAttachments,
    consensusBodyEl,
    trackAppEvent,
    showPopup,
    exitHeroMode,
    syncHeroResponseAccess,
    renderUsageDisplay,
    usageRun
  });
})();
