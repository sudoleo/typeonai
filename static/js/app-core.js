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

  Object.assign(window.App, {
    modelPrefs,
    deepThinkModelLabels,
    getModelOptionLabel,
    getSelectedModelCount,
    trackAppEvent,
    showPopup
  });
})();
