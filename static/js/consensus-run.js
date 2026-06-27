// =====================================================================
// consensus-run.js
// Consensus request only: window.getConsensus builds the /consensus
// payload, drives the SSE stream, and renders the result (main answer,
// structured differences, citation/share metadata).
//
// Lifecycle (visibility, gate, run-state, abort) lives in
// consensus-lifecycle.js and is used here via window.App.consensusLifecycle.
// Query-send (the per-model /ask_* requests that produce the answers this
// consumes) stays in templates/index.html until its own extraction.
//
// Shared deps via existing window contracts:
//   - window.App.consensusLifecycle, window.App.trackAppEvent
//   - window.createStreamRenderer / streamSSERequest / injectMarkdown
//   - window.lastQuestion (also written by query-send in index.html)
//   - window.currentMaxLimit / window.currentDeepLimit (usage mirrors)
//   - window.saveBookmarkConsensus / window.recordModelVote (firebase.js)
//   - window.auth (firebase.js)
// =====================================================================

(function () {
  window.App = window.App || {};

  const consensusLifecycle = window.App.consensusLifecycle;
  const trackAppEvent = window.App.trackAppEvent;
  const createStreamRenderer = window.createStreamRenderer;
  const streamSSERequest = window.streamSSERequest;
  const injectMarkdown = window.injectMarkdown;

  function isActiveConsensusRun(runId) {
    return consensusLifecycle.isActiveRun(runId);
  }

  function finishConsensusRun(runId) {
    consensusLifecycle.finishRun(runId);
  }

  function setConsensusSynthesizing(isSynthesizing) {
    consensusLifecycle.setSynthesizing(isSynthesizing);
  }

  function isAbortError(error) {
    return error && error.name === "AbortError";
  }

  function parseBestModel(differencesText) {
    const regex = /BestModel:\s*(.*)/i;
    const match = differencesText.match(regex);
    return match ? match[1].trim() : null;
  }

  window.getConsensus = async function (trigger = "manual") {
    if (consensusLifecycle.isRunning()) {
      window.cancelCurrentConsensus();
      return;
    }

    if (
      typeof window.isAgentModeEnabled === "function"
      && window.isAgentModeEnabled()
      && window.isAgentModeRunning()
    ) {
      if (typeof window.updateConsensusButtonAvailability === "function") {
        window.updateConsensusButtonAvailability();
      }
      return;
    }

    const consensusRun = consensusLifecycle.startRun();
    const consensusRunId = consensusRun.runId;
    const consensusSignal = consensusRun.signal;

    const question =
      (document.getElementById("questionInput")?.value ?? window.lastQuestion ?? "")
        .trim();
    // Status, ob eigene API Keys genutzt werden sollen
    const useOwnKeys = document.getElementById("useOwnKeysSwitch").checked;

    let id_token = null;
    if (window.auth && window.auth.currentUser) {
      try {
        id_token = await window.auth.currentUser.getIdToken();
      } catch (e) {
        console.error("Token refresh error in consensus:", e);
      }
    }
    if (!id_token) {
      window.App.showPopup(useOwnKeys
        ? "Please log in before using your own API keys."
        : "Please log in before generating a consensus.");
      finishConsensusRun(consensusRunId);
      return;
    }

    // Wenn die Frage neu oder geändert ist, werden Firebase-Votes aktualisiert.
    if (!isActiveConsensusRun(consensusRunId)) {
      return;
    }

    if (question !== window.lastQuestion) {
      // Für jedes Modell prüfen, ob es als "best" markiert ist.

      // Ebenso für "excluded" (sofern du das separat erfassen möchtest).

      // Aktualisiere die letzte verarbeitete Frage.
      window.lastQuestion = question;
    }

    // Setze den Konsens-Bereich (Spinner etc.) und rufe anschließend deinen Konsens-Endpunkt auf.
    const consensusDiv = document.getElementById("consensusResponse");

    const consensus_model = document.getElementById("consensusModelDropdown").value;

    // Hole die Antwort-Boxen
    const openaiBox = document.getElementById("openaiResponse");
    const mistralBox = document.getElementById("mistralResponse");
    const claudeBox = document.getElementById("claudeResponse");
    const geminiBox = document.getElementById("geminiResponse");
    const deepseekBox = document.getElementById("deepseekResponse");
    const grokBox = document.getElementById("grokResponse");

    // Lies die Antworten (trim für überflüssige Leerzeichen)
    function isIncludedBox(box) {
      return box && !box.classList.contains("excluded");
    }

    function getIncludedAnswer(box) {
      if (!isIncludedBox(box)) return "";
      if (box.dataset.responseError === "true") return "";
      return (box.dataset.consensusAnswer || box.querySelector(".collapsible-content")?.innerText || "").trim();
    }

    function getIncludedSources(box) {
      if (!isIncludedBox(box) || !box.dataset.consensusSources) return [];
      try {
        const parsed = JSON.parse(box.dataset.consensusSources);
        return Array.isArray(parsed) ? parsed : [];
      } catch (e) {
        return [];
      }
    }

    // Abgewählte Modelle werden bewusst als leer gesendet.
    const answer_openai = getIncludedAnswer(openaiBox);
    const answer_mistral = getIncludedAnswer(mistralBox);
    const answer_claude = getIncludedAnswer(claudeBox);
    const answer_gemini = getIncludedAnswer(geminiBox);
    const answer_deepseek = getIncludedAnswer(deepseekBox);
    const answer_grok = getIncludedAnswer(grokBox);
    const model_sources = {
      OpenAI: getIncludedSources(openaiBox),
      Mistral: getIncludedSources(mistralBox),
      Anthropic: getIncludedSources(claudeBox),
      Gemini: getIncludedSources(geminiBox),
      DeepSeek: getIncludedSources(deepseekBox),
      Grok: getIncludedSources(grokBox)
    };

    // Überprüfe nur die Modelle, die nicht als "ausgeschlossen" markiert sind.
    const needOpenAI = isIncludedBox(openaiBox);
    const needGemini = isIncludedBox(geminiBox);
    const needMistral = isIncludedBox(mistralBox);
    const needClaude = isIncludedBox(claudeBox);
    const needDeepseek = isIncludedBox(deepseekBox);
    const needGrok = isIncludedBox(grokBox);
    const includedAnswerCount = [answer_openai, answer_mistral, answer_claude, answer_gemini, answer_deepseek, answer_grok]
      .filter(Boolean).length;
    const excludedModelCount = [openaiBox, mistralBox, claudeBox, geminiBox, deepseekBox, grokBox]
      .filter(box => box?.classList.contains("excluded")).length;

    if (
      !question ||
      !consensus_model ||
      includedAnswerCount < 2 ||
      (needOpenAI && !answer_openai) ||
      (needGemini && !answer_gemini) ||
      (needMistral && !answer_mistral) ||
      (needClaude && !answer_claude) ||
      (needDeepseek && !answer_deepseek) ||
      (needGrok && !answer_grok)
    ) {
      alert("Please provide at least two completed model answers before generating a consensus.");
      if (window.resetCredibilityFrame) {
        window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
      }
      consensusDiv.querySelector(".consensus-main p").innerText = "";
      consensusDiv.querySelector(".consensus-differences p").innerText = "";
      finishConsensusRun(consensusRunId);
      return;
    }

    trackAppEvent("app_consensus_started", {
      trigger,
      included_models: includedAnswerCount,
      excluded_models: excludedModelCount,
      custom_credentials: useOwnKeys,
      logged_in: !!window.auth?.currentUser
    });

    if (window.resetCredibilityFrame) {
      window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
    }
    // Rahmenlosen Konsens-Bereich sanft einblenden (Fade-In + Slide-Up).
    window.revealConsensusOutput?.();
    setConsensusSynthesizing(true);
    consensusDiv.querySelector(".consensus-main p").innerHTML = window.consensusSpinnerHTML || window.spinnerHTML;
    consensusDiv.querySelector(".consensus-differences p").innerHTML = window.consensusDifferencesSpinnerHTML || window.spinnerHTML;

    // Die übrigen Parameter wie "excluded_models" werden wie bisher ermittelt
    const excludedModels = [];
    if (openaiBox.classList.contains("excluded")) {
      excludedModels.push(openaiBox.getAttribute("data-model"));
    }
    if (mistralBox.classList.contains("excluded")) {
      excludedModels.push(mistralBox.getAttribute("data-model"));
    }
    if (claudeBox.classList.contains("excluded")) {
      excludedModels.push(claudeBox.getAttribute("data-model"));
    }
    if (geminiBox.classList.contains("excluded")) {
      excludedModels.push(geminiBox.getAttribute("data-model"));
    }
    if (deepseekBox.classList.contains("excluded")) {
      excludedModels.push(deepseekBox.getAttribute("data-model"));
    }
    if (grokBox.classList.contains("excluded")) {
      excludedModels.push(grokBox.getAttribute("data-model"));
    }

    // Hole API Keys aus localStorage
    const openaiKey = localStorage.getItem("openaiKey") || "";
    const mistralKey = localStorage.getItem("mistralKey") || "";
    const anthropicKey = localStorage.getItem("anthropicKey") || "";
    const geminiKey = localStorage.getItem("geminiKey") || "";
    const deepseekKey = localStorage.getItem("deepseekKey") || "";
    const grokKey = localStorage.getItem("grokKey") || "";

    // --------- NEU: genaue Konsensus-Metadaten für Zitation speichern ---------
    const includedModelsDetailed = [];

    function addModelForCitation(boxId, selectId, label) {
      const box = document.getElementById(boxId);
      if (!box) return;
      if (box.classList.contains("excluded")) return;

      const contentEl = box.querySelector(".collapsible-content");
      const text = contentEl ? contentEl.innerText.trim() : "";
      if (!text) return; // nur Modelle mit Antwort

      const select = document.getElementById(selectId);
      let modelName = "";
      if (select) {
        const opt = select.options[select.selectedIndex];
        // Reines Modell-Label ohne Badge-Suffix ("· New") für die Zitation.
        modelName = (opt?.dataset.modelLabel || opt?.text || select.value || "").trim();
      }

      includedModelsDetailed.push(modelName ? `${label}: ${modelName}` : label);
    }

    addModelForCitation("openaiResponse", "openaiModelSelect", "OpenAI");
    addModelForCitation("mistralResponse", "mistralModelSelect", "Mistral");
    addModelForCitation("claudeResponse", "claudeModelSelect", "Anthropic Claude");
    addModelForCitation("geminiResponse", "geminiModelSelect", "Google Gemini");
    addModelForCitation("deepseekResponse", "deepseekModelSelect", "DeepSeek");
    addModelForCitation("grokResponse", "grokModelSelect", "Grok");

    const consensusSelect = document.getElementById("consensusModelDropdown");
    const consensusModelValue = consensusSelect ? consensusSelect.value : "";
    const consensusModelLabel = consensusSelect
      ? (consensusSelect.options[consensusSelect.selectedIndex]?.text || consensusModelValue)
      : consensusModelValue;

    // URL für die Zitation „aufgeräumt“
    let cleanUrl = window.location.href;
    try {
      const urlObj = new URL(window.location.href);
      cleanUrl = urlObj.origin + urlObj.pathname;
    } catch (e) {
      // falls URL-Parsing scheitert, nimm einfach href
    }

    window.consensusCitationMeta = {
      question,
      includedModels: includedModelsDetailed,
      consensusModel: consensusModelLabel || consensusModelValue,
      dateISO: new Date().toISOString(),
      url: cleanUrl
    };

    // Share-Feature: result_id des letzten Laufs zurücksetzen; Modell-
    // Labels (Option-Text) für die serverseitige Snapshot-Zitation.
    window.lastShareResultId = null;
    const shareModelLabels = {};
    [["OpenAI", "openaiModelSelect"], ["Mistral", "mistralModelSelect"],
     ["Anthropic", "claudeModelSelect"], ["Gemini", "geminiModelSelect"],
     ["DeepSeek", "deepseekModelSelect"], ["Grok", "grokModelSelect"]
    ].forEach(([provider, selectId]) => {
      const select = document.getElementById(selectId);
      if (!select) return;
      const opt = select.options[select.selectedIndex];
      // Reines Modell-Label (data-model-label) ohne Badge-Suffix wie "· New".
      const label = (opt?.dataset.modelLabel || opt?.text || select.value || "").trim();
      if (label) shareModelLabels[provider] = label;
    });

    try {
      const consensusMainRenderer = createStreamRenderer(
        consensusDiv.querySelector(".consensus-main p"),
        () => isActiveConsensusRun(consensusRunId)
      );
      // Differences-Deltas werden nicht mehr live gerendert: die Engine
      // liefert JSON, das erst mit dem final-Event als strukturierte UI
      // (Verdict, Badges, Karten) dargestellt wird. Der Spinner bleibt
      // bis dahin stehen.
      const consensusRequestResult = await streamSSERequest("/consensus", {
          id_token: id_token,
          useOwnKeys: useOwnKeys,
          question: question,
          answer_openai: answer_openai,
          answer_mistral: answer_mistral,
          answer_claude: answer_claude,
          answer_gemini: answer_gemini,
          answer_deepseek: answer_deepseek,
          answer_grok: answer_grok,
          model_sources: model_sources,
          model_labels: shareModelLabels,
          consensus_model: consensus_model,
          excluded_models: excludedModels,
          openai_key: openaiKey,
          mistral_key: mistralKey,
          anthropic_key: anthropicKey,
          gemini_key: geminiKey,
          deepseek_key: deepseekKey,
          grok_key: grokKey,
          keepalive: true
        }, consensusSignal, {
          "consensus.delta": consensusMainRenderer
        });
      const data = consensusRequestResult.data;
      const consensusErrorDetail =
        data?.detail && typeof data.detail === "object"
          ? data.detail
          : null;
      const consensusErrorMessage =
        consensusErrorDetail?.error ||
        consensusErrorDetail?.message ||
        data.error ||
        data.detail ||
        `Consensus HTTP ${consensusRequestResult.status}`;

      if (!isActiveConsensusRun(consensusRunId)) {
        return;
      }

      // Aktualisiere Free Requests, falls vorhanden (Deep Think nicht benötigt)
      const freeUsageRemaining =
        data.free_usage_remaining !== undefined
          ? data.free_usage_remaining
          : consensusErrorDetail?.free_usage_remaining;
      const deepRemaining =
        data.deep_remaining !== undefined
          ? data.deep_remaining
          : consensusErrorDetail?.deep_remaining;

      if (freeUsageRemaining !== undefined) {
        document.getElementById("freeUsageDisplay").innerText =
          "Requests: " + freeUsageRemaining + " / " + window.currentMaxLimit;
      }
      if (deepRemaining !== undefined) {
        document.getElementById("deepUsageDisplay").innerText =
          "Deep Think: " + deepRemaining + " / " + window.currentDeepLimit;
      }

      if (consensusRequestResult.ok) {
        // Share-Feature: nur mit result_id aus dem Final-Event ist
        // Teilen möglich (serverseitiger Snapshot vorhanden).
        window.lastShareResultId = data.result_id || null;

        const mainEl = consensusDiv.querySelector(".consensus-main p");
        const diffEl = consensusDiv.querySelector(".consensus-differences p");

        if (mainEl) {
          // Konsens-Text inkl. [S1]-Links, Copy-Buttons usw.
          injectMarkdown(mainEl, data.consensus_response);
        }

        if (diffEl) {
          // Strukturierte Auswertung (Verdict-Header, Badges, Karten),
          // fällt bei fehlenden/ungültigen Daten auf den Freitext zurück.
          const structuredRendered = window.renderConsensusInsights
            ? window.renderConsensusInsights(data.differences_data, includedAnswerCount)
            : false;

          if (!structuredRendered) {
            const diffsMD = data.differences || "No differences found.";
            if (window.applyCredibilityFrame) {
              window.applyCredibilityFrame(diffEl, diffsMD);
            }
            const cleaned = window.colorizeCredibility
              ? window.colorizeCredibility(diffsMD)
              : diffsMD;

            // Differences auch über injectMarkdown → [S1]-Links inkl.
            injectMarkdown(diffEl, cleaned);
          }
        }

        if (window.auth?.currentUser) {
          window.saveBookmarkConsensus(question, data.consensus_response, data.differences, data.differences_data);
        }
        trackAppEvent("app_consensus_completed", {
          status: "success",
          trigger,
          included_models: includedAnswerCount
        });

        const bestModelFromConsensus =
          (data.differences_data && data.differences_data.best_model) ||
          parseBestModel(data.differences);
        if (bestModelFromConsensus) {
          window.recordModelVote(bestModelFromConsensus, "BestModel");
        }
      } else {
        if (window.resetCredibilityFrame) {
          window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
        }
        if (consensusErrorDetail?.error_code === "usage_limit_exceeded" && typeof window.setAgentModeStatus === "function" && window.isAgentModeEnabled?.()) {
          window.setAgentModeStatus("error", consensusErrorMessage);
        }
        consensusDiv.querySelector(".consensus-main p").innerText = "Error: " + consensusErrorMessage;
        consensusDiv.querySelector(".consensus-differences p").innerText = "";
        trackAppEvent("app_consensus_completed", {
          status: "error",
          trigger,
          included_models: includedAnswerCount
        });
      }

    } catch (error) {
      if (isAbortError(error) || !isActiveConsensusRun(consensusRunId)) {
        return;
      }
      console.error("Error fetching consensus:", error);
      if (window.resetCredibilityFrame) {
        window.resetCredibilityFrame(consensusDiv.querySelector(".consensus-differences"));
      }
      consensusDiv.querySelector(".consensus-main p").innerText = "Error in the consensus calculation.";
      consensusDiv.querySelector(".consensus-differences p").innerText = "";
      trackAppEvent("app_consensus_completed", {
        status: "error",
        trigger,
        included_models: includedAnswerCount
      });
    } finally {
      finishConsensusRun(consensusRunId);
    }
  };
})();
