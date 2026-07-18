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
    return (option?.textContent || "").replace(/(?:\s*(?:Â·|·)\s*(?:Pro|Early))+$/i, "").trim();
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
    const hiddenInHero =
      document.body.classList.contains("is-hero") &&
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
    trackAppEvent,
    showPopup,
    exitHeroMode,
    syncHeroResponseAccess,
    usageRun
  });
})();
